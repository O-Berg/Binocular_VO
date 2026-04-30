import os
import cv2
import numpy as np

import config

os.environ.setdefault('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2')

try:
    import cupy as cp
    # Trigger a real kernel compile now to verify the toolkit is present.
    # cupy imports successfully even without the CUDA Toolkit, but crashes
    # on first kernel use if nvrtc*.dll is missing.
    _x = cp.array([1.0])
    cp.linalg.norm(_x)
    del _x
    _HAS_GPU = True
    print('[GPU] cupy OK — NCC will run on GPU')
except Exception:
    cp = None
    _HAS_GPU = False
    print('[GPU] cupy unavailable or CUDA Toolkit missing — falling back to numpy')

_NMS_KERNEL      = cv2.getStructuringElement(cv2.MORPH_RECT,    (config.NMS_KERNEL, config.NMS_KERNEL))
_SUPPRESS_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.SUPPRESS_RADIUS * 2 + 1,
                                                                   config.SUPPRESS_RADIUS * 2 + 1))

_HAS_CV_CUDA = False
try:
    _HAS_CV_CUDA = (
        bool(config.USE_GPU)
        and hasattr(cv2, 'cuda')
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
        and hasattr(cv2.cuda, 'createCornerHarrisDetector')
    )
except Exception:
    _HAS_CV_CUDA = False

_CUDA_HARRIS = None


def gpu_enabled():
    return bool(config.USE_GPU and _HAS_GPU)


def cupy_module():
    return cp if gpu_enabled() else None


class Feature:
    def __init__(self, x, y, weight, peak, std):
        self.pos    = np.array([x, y], dtype=np.float32)
        self.weight = weight
        self.peak   = peak
        self.std    = std


def harris_response(img):
    global _CUDA_HARRIS
    if _HAS_CV_CUDA:
        try:
            if _CUDA_HARRIS is None:
                _CUDA_HARRIS = cv2.cuda.createCornerHarrisDetector(
                    cv2.CV_8UC1,
                    blockSize=config.HARRIS_BLOCK,
                    ksize=config.HARRIS_BLOCK,
                    k=config.HARRIS_K,
                )
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            raw = _CUDA_HARRIS.compute(gpu_img).download()
            vis = cv2.normalize(np.maximum(0, raw), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return raw, vis
        except Exception:
            pass
    raw = cv2.cornerHarris(img, blockSize=config.HARRIS_BLOCK, ksize=config.HARRIS_BLOCK, k=config.HARRIS_K)
    vis = cv2.normalize(np.maximum(0, raw), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return raw, vis


def _peaks_to_features(clean, xs, ys):
    return [Feature(float(x), float(y), float(clean[y, x]), float(clean[y, x]), 0.0)
            for x, y in zip(xs, ys)]


def extract_features(harris_map):
    clean      = np.maximum(0, harris_map)
    global_max = clean.max()
    if global_max <= 0:
        return ([], []), (clean, clean, clean)

    dilated  = cv2.dilate(clean, _NMS_KERNEL)
    peak_map = (clean == dilated) & (clean > 0)

    strong_map = peak_map & (clean >= global_max * config.STRONG_THRESH_FRAC)
    ys1, xs1   = np.where(strong_map)
    pass1_vis  = strong_map.astype(np.float32) * 255

    if len(xs1) > config.NUM_STRONG:
        vals = clean[ys1, xs1]
        idx  = np.argpartition(vals, -config.NUM_STRONG)[-config.NUM_STRONG:]
        xs1, ys1 = xs1[idx], ys1[idx]

    strong_features = _peaks_to_features(clean, xs1, ys1)

    selected                    = np.zeros(clean.shape, dtype=np.uint8)
    selected[ys1, xs1]          = 1
    suppress_mask               = cv2.dilate(selected, _SUPPRESS_KERNEL).astype(bool)
    suppressed                  = clean * (~suppress_mask).astype(np.float32)
    suppressed_vis              = suppress_mask.astype(np.float32) * 255

    dilated2 = cv2.dilate(suppressed, _NMS_KERNEL)
    weak_map = (suppressed == dilated2) & (suppressed > 0)
    ys2, xs2 = np.where(weak_map)
    pass2_vis = suppressed * weak_map.astype(np.float32)

    if len(xs2) > config.NUM_WEAK:
        vals = suppressed[ys2, xs2]
        idx  = np.argpartition(vals, -config.NUM_WEAK)[-config.NUM_WEAK:]
        xs2, ys2 = xs2[idx], ys2[idx]

    weak_features = _peaks_to_features(suppressed, xs2, ys2)

    return (strong_features, weak_features), (pass1_vis, suppressed_vis, pass2_vis)


def extract_patches(features, img, half_p):
    if not features:
        return [], np.empty((0, (2 * half_p + 1) ** 2), dtype=np.float32)
    h, w  = img.shape
    side  = 2 * half_p + 1
    pos   = np.array([f.pos for f in features], dtype=np.int32)
    px, py = pos[:, 0], pos[:, 1]
    ok        = (py - half_p >= 0) & (py + half_p < h) & (px - half_p >= 0) & (px + half_p < w)
    valid_idx = np.where(ok)[0]
    if not len(valid_idx):
        return [], np.empty((0, side * side), dtype=np.float32)
    # Build a (H+2p, W+2p, side, side) sliding-window view of the padded image,
    # then index all feature locations at once — no Python loop.
    padded = np.pad(img, half_p, mode='edge').astype(np.float32)
    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(h, w, side, side),
        strides=(s0, s1, s0, s1),
    )
    vy, vx  = py[valid_idx], px[valid_idx]
    patches = windows[vy, vx].reshape(len(valid_idx), side * side)
    return valid_idx.tolist(), patches


def ncc_matrix(patches_a, patches_b, return_gpu=False):
    """Cosine similarity for all (a, b) pairs. Returns (N_a, N_b) float32."""
    if len(patches_a) == 0 or len(patches_b) == 0:
        if return_gpu and gpu_enabled():
            return cp.zeros((len(patches_a), len(patches_b)), dtype=cp.float32)
        return np.zeros((len(patches_a), len(patches_b)), dtype=np.float32)
    if gpu_enabled():
        a = cp.asarray(patches_a)
        b = cp.asarray(patches_b)
        a = a / cp.maximum(cp.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        b = b / cp.maximum(cp.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        scores = (a @ b.T).astype(cp.float32)
        return scores if return_gpu else cp.asnumpy(scores)
    a = patches_a / np.maximum(np.linalg.norm(patches_a, axis=1, keepdims=True), 1e-8)
    b = patches_b / np.maximum(np.linalg.norm(patches_b, axis=1, keepdims=True), 1e-8)
    return (a @ b.T).astype(np.float32)
