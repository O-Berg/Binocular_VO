import cv2
import numpy as np

import config


def estimate_pose(matches, calib, prev_depth_of):
    """Estimate metric relative pose via PnP.

    Uses 3-D points from stereo depth at t-1 matched to 2-D observations at t.
    Returns (R 3×3, t 3×1, n_inliers), or (None, None, 0) on failure.
    """
    if len(matches) < config.PNP_MIN_INLIERS or calib is None:
        return None, None, 0
    fx, fy, cx, cy, baseline = calib
    K = np.float32([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    pts3d, pts2d = [], []
    for pf, cf, _dv, _sc in matches:
        d = prev_depth_of.get(id(pf))
        if d is None or d < 1.0:
            continue
        Z = fx * baseline / d
        if not (config.DEPTH_MIN_M < Z < config.DEPTH_MAX_M):
            continue
        pts3d.append([(pf.pos[0] - cx) * Z / fx,
                      (pf.pos[1] - cy) * Z / fy,
                      Z])
        pts2d.append(cf.pos)
    if len(pts3d) < config.PNP_MIN_INLIERS:
        return None, None, 0
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.float32(pts3d), np.float32(pts2d), K, None,
        iterationsCount=200, reprojectionError=config.PNP_REPROJ_ERROR,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < config.PNP_MIN_INLIERS:
        return None, None, 0
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, len(inliers)
