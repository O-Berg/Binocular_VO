import cv2
import numpy as np

import config
from features import cupy_module, extract_patches, gpu_enabled, ncc_matrix


def _normalize_patches_np(patches):
    norms = np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), 1e-8)
    return patches / norms


def find_stereo_matches(features_l, features_r, img_l, img_r, patch_size):
    half_p = patch_size // 2
    valid_li, patches_l = extract_patches(features_l, img_l, half_p)
    valid_ri, patches_r = extract_patches(features_r, img_r, half_p)
    if not valid_li or not valid_ri:
        return []

    valid_l = [features_l[i] for i in valid_li]
    valid_r = [features_r[i] for i in valid_ri]
    pos_l = np.array([f.pos for f in valid_l], dtype=np.float32)
    pos_r = np.array([f.pos for f in valid_r], dtype=np.float32)

    # Rectified stereo only needs a narrow positive-disparity epipolar band.
    # Scoring just those sparse candidates is faster than a dense all-pairs GPU
    # matrix at the current feature counts.
    patches_l = _normalize_patches_np(patches_l)
    patches_r = _normalize_patches_np(patches_r)

    left_best_j = np.full(len(valid_l), -1, dtype=np.int32)
    left_best_s = np.full(len(valid_l), -2.0, dtype=np.float32)
    right_best_i = np.full(len(valid_r), -1, dtype=np.int32)
    right_best_s = np.full(len(valid_r), -2.0, dtype=np.float32)

    for i, (x_l, y_l) in enumerate(pos_l):
        disp = x_l - pos_r[:, 0]
        cand = np.where(
            (disp > 1.0)
            & (disp <= config.DISP_SEARCH)
            & (np.abs(y_l - pos_r[:, 1]) <= config.EPIPOLAR_MARGIN)
        )[0]
        if not len(cand):
            continue

        scores = patches_r[cand] @ patches_l[i]
        local_best = int(np.argmax(scores))
        j = int(cand[local_best])
        left_best_j[i] = j
        left_best_s[i] = float(scores[local_best])

        better = scores > right_best_s[cand]
        if np.any(better):
            bj = cand[better]
            right_best_s[bj] = scores[better]
            right_best_i[bj] = i

    keep = np.where(
        (left_best_j >= 0)
        & (left_best_s > config.NCC_THRESHOLD)
        & (right_best_i[left_best_j] == np.arange(len(valid_l)))
    )[0]

    return [
        (valid_l[int(i)], valid_r[int(left_best_j[i])],
         float(pos_l[i, 0] - pos_r[left_best_j[i], 0]), float(left_best_s[i]))
        for i in keep
    ]


def find_temporal_matches(prev_features, curr_features, prev_img, curr_img,
                          patch_size=11, return_score_matrix=True):
    half_p = patch_size // 2
    r_max  = config.TEMPORAL_SEARCH_RADIUS

    valid_pi, patches_p = extract_patches(prev_features, prev_img, half_p)
    valid_ci, patches_c = extract_patches(curr_features, curr_img, half_p)
    if not valid_pi or not valid_ci:
        return [], None, [], []

    valid_pf = [prev_features[i] for i in valid_pi]
    valid_cf = [curr_features[i] for i in valid_ci]
    pos_p = np.array([f.pos for f in valid_pf], dtype=np.float32)
    pos_c = np.array([f.pos for f in valid_cf], dtype=np.float32)

    if gpu_enabled():
        cp = cupy_module()
        score_matrix_gpu = ncc_matrix(patches_c, patches_p, return_gpu=True)
        pos_p_gpu = cp.asarray(pos_p)
        pos_c_gpu = cp.asarray(pos_c)

        dx = cp.abs(pos_c_gpu[:, 0:1] - pos_p_gpu[:, 0])
        dy = cp.abs(pos_c_gpu[:, 1:2] - pos_p_gpu[:, 1])
        win_mask = (dx <= r_max) & (dy <= r_max)
        masked = cp.where(win_mask, score_matrix_gpu, -1.0)

        best_rows = cp.argmax(masked, axis=0)
        col_idx = cp.arange(len(valid_pf))
        best_scores = masked[best_rows, col_idx]

        good_cols = cp.where(best_scores >= config.TEMPORAL_NCC_THRESHOLD)[0]
        good_rows = best_rows[good_cols]
        dists = cp.linalg.norm(pos_c_gpu[good_rows] - pos_p_gpu[good_cols], axis=1)
        keep = dists <= config.TEMPORAL_MAX_DIST

        keep_cols = cp.asnumpy(good_cols[keep])
        keep_rows = cp.asnumpy(good_rows[keep])
        reject_cols = cp.asnumpy(good_cols[~keep])
        reject_rows = cp.asnumpy(good_rows[~keep])
        scores_k = cp.asnumpy(best_scores[good_cols[keep]])

        matched_cells = list(zip(keep_cols.tolist(), keep_rows.tolist()))
        rejected_cells = list(zip(reject_cols.tolist(), reject_rows.tolist()))
        matches = [
            (prev_features[valid_pi[int(c)]], curr_features[valid_ci[int(r)]],
             pos_c[int(r)] - pos_p[int(c)], float(s))
            for c, r, s in zip(keep_cols.tolist(), keep_rows.tolist(), scores_k.tolist())
        ]
        score_matrix = cp.asnumpy(score_matrix_gpu) if return_score_matrix else None
        return matches, score_matrix, matched_cells, rejected_cells

    score_matrix = ncc_matrix(patches_c, patches_p)

    dx = np.abs(pos_c[:, 0:1] - pos_p[:, 0])
    dy = np.abs(pos_c[:, 1:2] - pos_p[:, 1])
    win_mask = (dx <= r_max) & (dy <= r_max)
    masked   = np.where(win_mask, score_matrix, -1.0)

    best_rows   = np.argmax(masked, axis=0)
    best_scores = masked[best_rows, np.arange(len(valid_pf))]

    good_cols = np.where(best_scores >= config.TEMPORAL_NCC_THRESHOLD)[0]
    good_rows = best_rows[good_cols]

    # Vectorised distance filter
    pp = pos_p[good_cols]
    cp_ = pos_c[good_rows]
    dists = np.linalg.norm(cp_ - pp, axis=1)
    keep = dists <= config.TEMPORAL_MAX_DIST

    matched_cells  = list(zip(good_cols[keep].tolist(),  good_rows[keep].tolist()))
    rejected_cells = list(zip(good_cols[~keep].tolist(), good_rows[~keep].tolist()))

    keep_cols = good_cols[keep]
    keep_rows = good_rows[keep]
    scores_k  = best_scores[keep_cols]
    matches = [
        (prev_features[valid_pi[c]], curr_features[valid_ci[r]],
         pos_c[r] - pos_p[c], float(s))
        for c, r, s in zip(keep_cols.tolist(), keep_rows.tolist(), scores_k.tolist())
    ]

    return matches, score_matrix if return_score_matrix else None, matched_cells, rejected_cells


def filter_temporal_outliers(matches, calib=None):
    """Remove geometrically inconsistent temporal matches.

    Uses RANSAC Essential Matrix when calibration is available; falls back to
    a MAD filter on displacement vectors otherwise.
    """
    if len(matches) < 5:
        return matches
    pts1 = np.float32([pf.pos for pf, _cf, _dv, _sc in matches])
    pts2 = np.float32([cf.pos for _pf, cf, _dv, _sc in matches])
    if calib is not None and len(matches) >= 8:
        fx, fy, cx, cy, _ = calib
        K = np.float32([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _, mask = cv2.findEssentialMat(pts1, pts2, K,
                                       method=cv2.RANSAC, prob=0.999,
                                       threshold=config.OUTLIER_RANSAC_THRESHOLD)
        if mask is not None and int(mask.sum()) >= 5:
            return [m for m, ok in zip(matches, mask.ravel()) if ok]
    vecs = pts2 - pts1
    med  = np.median(vecs, axis=0)
    res  = np.linalg.norm(vecs - med, axis=1)
    mad  = max(float(np.median(res)), 0.5)
    return [m for m, ok in zip(matches, (res <= config.OUTLIER_MAD_SCALE * mad).tolist()) if ok]
