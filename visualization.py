import cv2
import numpy as np

import config

# Lazy vispy import — only needed when the 3D window is opened
_vispy_app    = None
_vispy_scene  = None


def _ensure_vispy():
    global _vispy_app, _vispy_scene
    if _vispy_app is None:
        from vispy import app as _a, scene as _s
        _a.use_app('pyqt5')
        _vispy_app   = _a
        _vispy_scene = _s


# ── 2-D helpers ───────────────────────────────────────────────────────────────

_CMAP_STOPS = np.array([           # BGR
    [100,  20,   0],   # far   — dark blue
    [200,  60,   0],   #       — blue
    [200, 200,   0],   #       — cyan
    [  0, 255,   0],   #       — green
    [  0, 200, 255],   #       — orange
    [  0,  60, 255],   #       — red-orange
    [  0,   0, 220],   # near  — red
], dtype=np.float32)


def disp_to_color(disp):
    t = float(np.clip((disp - config.DISP_MIN) / max(config.DISP_MAX - config.DISP_MIN, 1), 0, 1))
    n   = len(_CMAP_STOPS) - 1
    idx = min(int(t * n), n - 1)
    s   = t * n - idx
    b0, g0, r0 = _CMAP_STOPS[idx]
    b1, g1, r1 = _CMAP_STOPS[idx + 1]
    return (int(b0 + s * (b1 - b0)), int(g0 + s * (g1 - g0)), int(r0 + s * (r1 - r0)))


def _disp_to_color_array(disp_arr):
    """Vectorised version of disp_to_color. Returns (N, 3) float32 RGB in [0, 1]."""
    n = len(_CMAP_STOPS) - 1
    t = np.clip((disp_arr - config.DISP_MIN) / max(config.DISP_MAX - config.DISP_MIN, 1), 0.0, 1.0)
    pos = t * n
    idx = np.clip(pos.astype(np.int32), 0, n - 1)
    s   = (pos - idx)[:, np.newaxis]
    rgb = _CMAP_STOPS[idx] + s * (_CMAP_STOPS[idx + 1] - _CMAP_STOPS[idx])
    return rgb[:, ::-1] / 255.0  # BGR stops → RGB, normalise


def add_label(img, text):
    font, scale, thick, pad = cv2.FONT_HERSHEY_DUPLEX, 0.35, 1, 8
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + pad * 2, th + pad * 2), (20, 20, 20), -1)
    cv2.putText(img, text, (pad, th + pad), font, scale, (240, 240, 240), thick, cv2.LINE_AA)
    return img


# ── 3-D helpers (vispy) ───────────────────────────────────────────────────────

def _make_camera_frustum_segs(pose_4x4, size=1.5, aspect=4 / 3):
    """Blender-style camera gizmo: pyramid frustum + top indicator triangle."""
    h = size * 0.4
    w = h * aspect
    d = size
    local = np.array([
        [ 0,       0,    0],
        [-w,      -h,    d],
        [ w,      -h,    d],
        [ w,       h,    d],
        [-w,       h,    d],
        [ 0,  -h * 1.6,  d],
    ], dtype=np.float32)
    hom   = np.hstack([local, np.ones((6, 1), dtype=np.float32)])
    world = (np.float32(pose_4x4) @ hom.T).T[:, :3]
    world[:, 0] *= -1
    world[:, 1] *= -1
    o, tl, tr, br, bl, tip = (world[i] for i in range(6))
    return np.array([
        o,  tl,    o,  tr,    o,  br,    o,  bl,
        tl, tr,    tr, br,    br, bl,    bl, tl,
        tl, tip,   tip, tr,
    ], dtype=np.float32)


def stereo_to_world_pts(stereo_matches, calib, img_bgr, cam_pose):
    """Back-project stereo matches to world-space 3-D points.

    Returns (pts_world, depth_colors, feat_colors) — all float32 arrays.
    """
    if not stereo_matches or calib is None:
        e3, e4 = np.empty((0, 3), np.float32), np.empty((0, 4), np.float32)
        return e3, e4, e4
    fx, fy, cx, cy, baseline = calib
    h_img, w_img = img_bgr.shape[:2]

    # Unpack all matches into arrays at once
    pos_l = np.array([m[0].pos for m in stereo_matches], dtype=np.float32)
    disps = np.abs(np.array([m[2] for m in stereo_matches], dtype=np.float32))

    # Depth filter
    valid = disps >= 1.0
    pos_l, disps = pos_l[valid], disps[valid]
    if not len(disps):
        e3, e4 = np.empty((0, 3), np.float32), np.empty((0, 4), np.float32)
        return e3, e4, e4

    Z = fx * baseline / disps
    depth_ok = (Z > config.DEPTH_MIN_M) & (Z < config.DEPTH_MAX_M)
    pos_l, disps, Z = pos_l[depth_ok], disps[depth_ok], Z[depth_ok]
    if not len(Z):
        e3, e4 = np.empty((0, 3), np.float32), np.empty((0, 4), np.float32)
        return e3, e4, e4

    X = (pos_l[:, 0] - cx) * Z / fx
    Y = (pos_l[:, 1] - cy) * Z / fy
    pts_c = np.stack([X, Y, Z], axis=1)

    # World transform + axis flip
    hom   = np.hstack([pts_c, np.ones((len(pts_c), 1), np.float32)])
    pts_w = (np.float32(cam_pose) @ hom.T).T[:, :3]
    pts_w[:, 0] *= -1
    pts_w[:, 1] *= -1

    # Depth colours — vectorised
    rgb_d  = _disp_to_color_array(disps)
    d_cols = np.hstack([rgb_d, np.full((len(pts_w), 1), 0.85, np.float32)])

    # Feature colours — sample 5x5 patch mean for each point, fully vectorised
    ix = np.clip(pos_l[:, 0].astype(np.int32), 2, w_img - 3)
    iy = np.clip(pos_l[:, 1].astype(np.int32), 2, h_img - 3)
    # Build (N,5,5,3) patch block using stride_tricks on padded image
    pad    = 2
    padded = np.pad(img_bgr, ((pad, pad), (pad, pad), (0, 0)), mode='edge').astype(np.float32)
    s0, s1, s2 = padded.strides
    side = 2 * pad + 1
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(h_img, w_img, side, side, 3),
        strides=(s0, s1, s0, s1, s2),
    )
    patches = windows[iy, ix]                          # (N, 5, 5, 3)
    rgb_f   = patches.reshape(len(ix), -1, 3).mean(axis=1) / 255.0   # (N, 3)

    if config.PC_SATURATION != 1.0:
        gray  = rgb_f.mean(axis=1, keepdims=True)
        rgb_f = np.clip(gray + config.PC_SATURATION * (rgb_f - gray), 0.0, 1.0)

    f_cols = np.hstack([rgb_f[:, [0, 1, 2]], np.full((len(pts_w), 1), 0.9, np.float32)])

    return pts_w, d_cols, f_cols


def _pan_view(view, dx_screen, dy_screen):
    az   = np.radians(view.camera.azimuth)
    step = max(view.camera.distance * 0.04, 0.5)
    c = list(view.camera.center)
    c[0] += dx_screen * float(np.cos(az)) * step
    c[2] -= dx_screen * float(np.sin(az)) * step
    c[1] += dy_screen * step
    view.camera.center = tuple(c)


def open_vispy_canvas(calib, follow_cam_ref):
    """Create the vispy SceneCanvas.  follow_cam_ref is a mutable 1-element list
    so the key handler can toggle it without a nonlocal across modules.

    Keys (vispy window must have focus):
      F            snap orbit center to current estimated camera position
      L            toggle auto-follow
      Arrow keys   pan
    """
    _ensure_vispy()
    fx, _, _, _, baseline = calib
    canvas = _vispy_scene.SceneCanvas(
        title=(f'Point Cloud  fx={fx:.0f}px  B={baseline:.3f}m'
               f'  |  colour={"depth" if config.PC_DEPTH_COLOR else "feature"}'
               '  |  F=snap  L=follow  arrows=pan'),
        keys='interactive', size=(1200, 800), bgcolor='#0d0d12', show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera           = 'turntable'
    view.camera.up        = '+y'   # scene: X=left, Y=up, Z=forward (after OpenCV X/Y flip)
    view.camera.fov       = 60
    view.camera.distance  = 30
    view.camera.azimuth   = -90   # with up=+y, -90° puts camera on -Z axis looking toward +Z
    view.camera.elevation = 0     # 0° = horizon-level
    view.camera.center    = (0, 0, 15)
    _vispy_scene.visuals.XYZAxis(parent=view.scene)

    pc_markers = _vispy_scene.visuals.Markers(parent=view.scene)
    cam_lines  = _vispy_scene.visuals.Line(parent=view.scene, connect='segments',
                                           antialias=False, width=1.5)
    traj_line  = _vispy_scene.visuals.Line(parent=view.scene, connect='strip',
                                           width=2, antialias=True,
                                           color=(0.2, 0.9, 0.4, 0.9))

    # all_cam_poses_ref is set by the caller after construction
    all_cam_poses_ref = [None]

    def _on_key(event):
        k = event.key.name if hasattr(event.key, 'name') else str(event.key)
        if k == 'F':
            poses = all_cam_poses_ref[0]
            if poses:
                pos = poses[-1][:3, 3]
                view.camera.center = (float(-pos[0]), float(-pos[1]), float(pos[2]))
                print('[3D] snapped to current camera')
        elif k == 'L':
            follow_cam_ref[0] = not follow_cam_ref[0]
            print(f'[3D] follow camera → {"ON" if follow_cam_ref[0] else "OFF"}')
        elif k == 'Left':
            _pan_view(view, -1,  0)
        elif k == 'Right':
            _pan_view(view,  1,  0)
        elif k == 'Up':
            _pan_view(view,  0,  1)
        elif k == 'Down':
            _pan_view(view,  0, -1)

    canvas.events.key_press.connect(_on_key)
    _vispy_app.process_events()

    visuals = (pc_markers, cam_lines, traj_line)
    return canvas, view, visuals, all_cam_poses_ref


def update_vispy(view, visuals, all_stereo_pts, all_cam_poses, follow_cam):
    """Push point-cloud and camera-pose frustums to the GPU visuals."""
    pc_markers, cam_lines, traj_line = visuals

    pts_list = [e[0] for e in all_stereo_pts if len(e[0])]
    if pts_list:
        all_pts = np.vstack(pts_list)
        if config.PC_DEPTH_COLOR and all_cam_poses:
            # Recompute colour every update from current camera distance
            # so accumulated points shift colour as the car drives away.
            # cam_pos in vispy space has the same X/Y flip as the points.
            cam_world = all_cam_poses[-1][:3, 3]
            cam_vispy = np.float32([-cam_world[0], -cam_world[1], cam_world[2]])
            dists = np.linalg.norm(all_pts - cam_vispy, axis=1)
            # Sqrt curve stretches the gradient toward the near end so mid-range
            # points aren't all crammed into the warm colours.
            t = np.clip((dists - config.DEPTH_MIN_M) / max(config.DEPTH_MAX_M - config.DEPTH_MIN_M, 1.0), 0.0, 1.0)
            pseudo_disp = config.DISP_MAX * (1.0 - np.sqrt(t))
            rgb = _disp_to_color_array(pseudo_disp)
            face_color = np.column_stack([rgb, np.full(len(all_pts), 0.85, dtype=np.float32)])
        else:
            col_list = [e[2] for e in all_stereo_pts if len(e[0])]
            face_color = np.vstack(col_list)
        pc_markers.set_data(all_pts, face_color=face_color,
                            size=config.PC_POINT_SIZE, edge_width=0)

    if all_cam_poses:
        frm_segs, frm_cols = [], []
        n = len(all_cam_poses)
        for i, pose in enumerate(all_cam_poses):
            is_cur = (i == n - 1)
            alpha  = 0.2 + 0.8 * (i / max(n - 1, 1))
            segs   = _make_camera_frustum_segs(pose, size=1.5 if is_cur else 0.5)
            col    = ([0.15, 0.80, 1.0, alpha]
                      if is_cur else [0.55, 0.55, 0.55, alpha * 0.45])
            frm_segs.append(segs)
            frm_cols.extend([col] * len(segs))
        cam_lines.set_data(pos=np.vstack(frm_segs),
                           color=np.array(frm_cols, dtype=np.float32))

    if len(all_cam_poses) > 1:
        traj_pts = np.array([p[:3, 3] for p in all_cam_poses], dtype=np.float32)
        traj_pts[:, 0] *= -1
        traj_pts[:, 1] *= -1
        traj_line.set_data(pos=traj_pts)

    if follow_cam and all_cam_poses:
        pose = all_cam_poses[-1]
        pos  = pose[:3, 3]
        fwd  = pose[:3, 2]
        # Scene: X=left (-OpenCV_X), Y=up (-OpenCV_Y), Z=forward (OpenCV_Z).
        # up='+y' → azimuth rotates around +Y; reference forward is +X (azimuth=0).
        # Car's vispy forward: vx=-fwd[0], vz=fwd[2].
        # atan2(vx, vz) gives heading from +Z axis; subtract 90° to get azimuth
        # relative to vispy's +X reference so the camera sits behind the car.
        fwd_vx = float(-fwd[0])
        fwd_vz = float(fwd[2])
        heading = float(np.degrees(np.arctan2(fwd_vx, fwd_vz)))
        view.camera.center    = (float(-pos[0]), float(-pos[1]), float(pos[2]))
        view.camera.azimuth   = heading - 90.0
        view.camera.elevation = 15.0  # slightly above — past positions fly by below
        view.camera.distance  = 20.0


def process_vispy_events():
    _ensure_vispy()
    _vispy_app.process_events()


def render_vispy_frame(canvas):
    """Return the current 3D canvas as an RGB uint8 numpy array (H, W, 3)."""
    _vispy_app.process_events()
    return canvas.render(alpha=False)
