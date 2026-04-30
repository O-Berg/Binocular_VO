import glob
import os

import cv2
import numpy as np


def load_image_paths(sequence_id: int):
    base = f"dataset/sequences/{sequence_id:02d}"
    left  = sorted(glob.glob(os.path.join(base, "image_0", "*.png")))
    right = sorted(glob.glob(os.path.join(base, "image_1", "*.png")))
    return base, left, right


def detect_color(path: str) -> bool:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (img is not None and img.ndim == 3 and img.shape[2] == 3
            and not np.array_equal(img[:, :, 0], img[:, :, 1]))


def load_calibration(seq_path: str):
    """Parse calib.txt → (fx, fy, cx, cy, baseline) at half-resolution."""
    try:
        fx = fy = cx = cy = tx1 = None
        with open(os.path.join(seq_path, 'calib.txt')) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'P0:':
                    v = list(map(float, parts[1:]))
                    fx, fy, cx, cy = v[0], v[5], v[2], v[6]
                elif parts[0] == 'P1:':
                    v = list(map(float, parts[1:]))
                    tx1 = v[3]
        if fx and fx > 0 and tx1 is not None:
            baseline = abs(tx1) / fx
            return fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5, baseline
    except FileNotFoundError:
        pass
    return None


def load_gt_poses(sequence_id: int):
    """Load KITTI ground-truth poses as a list of (3,4) float64 arrays."""
    path = f"dataset/poses/{sequence_id:02d}.txt"
    poses = []
    try:
        with open(path) as f:
            for line in f:
                v = list(map(float, line.split()))
                if len(v) == 12:
                    poses.append(np.array(v, dtype=np.float64).reshape(3, 4))
    except FileNotFoundError:
        pass
    return poses
