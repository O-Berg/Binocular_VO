import argparse
import cv2
import numpy as np
from collections import defaultdict, deque

import config
import dataset
import features as feat
import matching
import pose as posemod
import visualization as viz

class VOPipeline:
    def __init__(self, sequence_id: int):
        base, left_files, right_files = dataset.load_image_paths(sequence_id)
        if not left_files or not right_files:
            raise FileNotFoundError(f"No images found for sequence {sequence_id:02d}")

        self.left_files  = left_files
        self.right_files = right_files
        self.calib       = dataset.load_calibration(base)
        self.gt_poses    = dataset.load_gt_poses(sequence_id)
        self.is_color    = dataset.detect_color(left_files[0])
        self.num_frames  = min(len(left_files), len(right_files))

        # ── per-frame previous state ──────────────────────────────────────────
        self.prev_strong_l  = []
        self.prev_weak_l    = []
        self.prev_img_l_sm  = None
        self.prev_depth_of  = {}

        # queued values are committed at the top of the next iteration so
        # frame scrubbing (seek back) doesn't update prev with the current frame
        self._queued_strong   = None
        self._queued_weak     = None
        self._queued_img      = None
        self._queued_depth_of = {}
        self._last_frame_idx  = -1

        # ── accumulated trajectory / point cloud ──────────────────────────────
        self.cam_pose_world = np.eye(4, dtype=np.float64)
        self.all_cam_poses  = []
        self.all_stereo_pts = []
        self.all_displacements = []

        # ── 3-D window state ──────────────────────────────────────────────────
        self._vispy_canvas       = None
        self._vispy_view         = None
        self._vispy_visuals      = None
        self._vispy_poses_ref    = None
        self._3d_frame_counter   = 0
        self._follow_cam_ref     = [False]   # mutable so vispy key handler can toggle it

        # ── profiling ─────────────────────────────────────────────────────────
        self._times     = defaultdict(float)
        self._recent_times = deque(maxlen=30)
        self._profile_n = 0
        self._t         = cv2.getTickFrequency()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ms(self):
        return cv2.getTickCount() / self._t * 1000.0

    def _commit_queued(self, frame_idx):
        """Copy queued state into prev_* at the start of a new frame."""
        if frame_idx != self._last_frame_idx:
            if self._queued_strong is not None:
                self.prev_strong_l  = self._queued_strong
                self.prev_weak_l    = self._queued_weak
                self.prev_img_l_sm  = self._queued_img
                self.prev_depth_of  = self._queued_depth_of
            self._last_frame_idx = frame_idx

    def _read_frame(self, frame_idx):
        if self.is_color:
            img_l_bgr = cv2.imread(self.left_files[frame_idx],  cv2.IMREAD_COLOR)
            img_r_bgr = cv2.imread(self.right_files[frame_idx], cv2.IMREAD_COLOR)
            if img_l_bgr is None or img_r_bgr is None:
                return None
            img_l = cv2.cvtColor(img_l_bgr, cv2.COLOR_BGR2GRAY)
            img_r = cv2.cvtColor(img_r_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_l = cv2.imread(self.left_files[frame_idx],  cv2.IMREAD_GRAYSCALE)
            img_r = cv2.imread(self.right_files[frame_idx], cv2.IMREAD_GRAYSCALE)
            if img_l is None or img_r is None:
                return None
            img_l_bgr = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            img_r_bgr = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
        return img_l, img_r, img_l_bgr, img_r_bgr

    def _reset_sequence(self):
        self.all_stereo_pts.clear()
        self.all_cam_poses.clear()
        self.all_displacements.clear()
        self.cam_pose_world   = np.eye(4, dtype=np.float64)
        self.prev_strong_l    = []
        self.prev_weak_l      = []
        self.prev_img_l_sm    = None
        self.prev_depth_of    = {}
        self._queued_strong   = None
        self._queued_weak     = None
        self._queued_img      = None
        self._queued_depth_of = {}
        self._last_frame_idx  = -1

    # ── 3-D window ────────────────────────────────────────────────────────────

    def _open_3d(self):
        canvas, view, visuals, poses_ref = viz.open_vispy_canvas(
            self.calib, self._follow_cam_ref)
        poses_ref[0] = self.all_cam_poses
        self._vispy_canvas    = canvas
        self._vispy_view      = view
        self._vispy_visuals   = visuals
        self._vispy_poses_ref = poses_ref
        if self.all_stereo_pts:
            viz.update_vispy(view, visuals, self.all_stereo_pts,
                             self.all_cam_poses, self._follow_cam_ref[0])

    def _close_3d(self):
        self._vispy_canvas.close()
        viz.process_vispy_events()
        self._vispy_canvas = self._vispy_view = self._vispy_visuals = None

    # ── main pipeline step ────────────────────────────────────────────────────

    def process_frame(self, frame_idx, include_temporal_debug=False):
        """Run one frame through the full pipeline. Returns display data dict."""
        self._commit_queued(frame_idx)

        frame = self._read_frame(frame_idx)
        if frame is None:
            return None
        img_l, img_r, img_l_bgr, img_r_bgr = frame

        t0 = self._ms()
        img_l_sm     = cv2.resize(img_l,     (0, 0), fx=0.5, fy=0.5)
        img_r_sm     = cv2.resize(img_r,     (0, 0), fx=0.5, fy=0.5)
        img_l_sm_bgr = cv2.resize(img_l_bgr, (0, 0), fx=0.5, fy=0.5)
        img_r_sm_bgr = cv2.resize(img_r_bgr, (0, 0), fx=0.5, fy=0.5)

        t1 = self._ms()
        raw_l, vis_l = feat.harris_response(img_l_sm)
        raw_r, vis_r = feat.harris_response(img_r_sm)

        t2 = self._ms()
        (strong_l, weak_l), debug_l = feat.extract_features(raw_l)
        (strong_r, weak_r), debug_r = feat.extract_features(raw_r)
        features_l = strong_l + weak_l
        features_r = strong_r + weak_r

        t3 = self._ms()
        stereo = matching.find_stereo_matches(strong_l, strong_r, img_l_sm, img_r_sm, config.STEREO_PATCH_SIZE)
        stereo += matching.find_stereo_matches(weak_l,  weak_r,   img_l_sm, img_r_sm, config.STEREO_PATCH_SIZE)
        depth_of = {id(fl): abs(float(disp)) for fl, _, disp, _ in stereo if abs(disp) >= 1.0}

        t4 = self._ms()
        tm_s = tm_mx_s = tm_cells_s = tm_rej_s = None
        tm_w = tm_mx_w = tm_cells_w = tm_rej_w = None
        pose_str = "pose: ---"

        if self._vispy_canvas is not None:
            viz.process_vispy_events()

        if self.prev_img_l_sm is not None:
            if self.prev_strong_l:
                tm_s, tm_mx_s, tm_cells_s, tm_rej_s = matching.find_temporal_matches(
                    self.prev_strong_l, strong_l, self.prev_img_l_sm, img_l_sm,
                    config.TEMPORAL_PATCH_SIZE,
                    return_score_matrix=include_temporal_debug)
                tm_s = matching.filter_temporal_outliers(tm_s, self.calib)
            if self.prev_weak_l:
                tm_w, tm_mx_w, tm_cells_w, tm_rej_w = matching.find_temporal_matches(
                    self.prev_weak_l, weak_l, self.prev_img_l_sm, img_l_sm,
                    config.TEMPORAL_PATCH_SIZE,
                    return_score_matrix=include_temporal_debug)
                tm_w = matching.filter_temporal_outliers(tm_w, self.calib)

            tm_s = tm_s or []
            tm_w = tm_w or []
            R_rel, t_rel, n_pnp = posemod.estimate_pose(tm_s, self.calib, self.prev_depth_of)
            if R_rel is not None:
                rvec_rel, _ = cv2.Rodrigues(R_rel)
                speed_ms  = float(np.linalg.norm(t_rel)) * config.DATASET_FPS
                rot_deg_s = float(np.linalg.norm(rvec_rel)) * (180.0 / np.pi) * config.DATASET_FPS
                pose_str  = f"spd {speed_ms:.2f} m/s   rot {rot_deg_s:.1f} deg/s   pnp {n_pnp}"
                T_rel = np.eye(4, dtype=np.float64)
                T_rel[:3, :3] = R_rel.astype(np.float64)
                T_rel[:3, 3]  = t_rel.ravel().astype(np.float64)
                self.cam_pose_world = self.cam_pose_world @ np.linalg.inv(T_rel)

            temporal_matches = tm_s + tm_w
            self.all_displacements.append([
                (pf.pos.copy(), cf.pos.copy(), dv.copy(), sc,
                 depth_of.get(id(cf)), self.prev_depth_of.get(id(pf)))
                for pf, cf, dv, sc in temporal_matches
            ])
        else:
            temporal_matches = []
            tm_s = tm_w = []

        if self.calib is not None:
            pts_w, d_cols, f_cols = viz.stereo_to_world_pts(
                stereo, self.calib, img_l_sm_bgr, self.cam_pose_world)
            self.all_stereo_pts.append((pts_w, d_cols, f_cols))
            self.all_cam_poses.append(self.cam_pose_world.copy())
            if not config.PC_ACCUMULATE_ALL and len(self.all_stereo_pts) > config.PC_HISTORY:
                self.all_stereo_pts.pop(0)
                self.all_cam_poses.pop(0)

        self._queued_strong   = strong_l
        self._queued_weak     = weak_l
        self._queued_img      = img_l_sm.copy()
        self._queued_depth_of = depth_of

        t5 = self._ms()
        frame_times = {
            'resize': t1 - t0,
            'harris': t2 - t1,
            'extract': t3 - t2,
            'stereo': t4 - t3,
            'temporal': t5 - t4,
        }
        self._recent_times.append(frame_times)
        self._times['resize']   += frame_times['resize']
        self._times['harris']   += frame_times['harris']
        self._times['extract']  += frame_times['extract']
        self._times['stereo']   += frame_times['stereo']
        self._times['temporal'] += frame_times['temporal']
        self._times['match_pct'] += 100 * len(stereo) / len(features_l) if features_l else 0
        self._profile_n += 1
        n = self._profile_n
        total_ms = sum(v for k, v in self._times.items() if k != 'match_pct') / n
        recent_n = len(self._recent_times)
        recent = {
            k: sum(ft[k] for ft in self._recent_times) / recent_n
            for k in frame_times
        }
        recent_total = sum(recent.values())
        print(f"[profile @ frame {frame_idx}]  "
              f"resize={self._times['resize']/n:.1f}ms  "
              f"harris={self._times['harris']/n:.1f}ms  "
              f"extract={self._times['extract']/n:.1f}ms  "
              f"stereo={self._times['stereo']/n:.1f}ms  "
              f"temporal={self._times['temporal']/n:.1f}ms  "
              f"total={total_ms:.1f}ms  "
              f"fps={1000/total_ms:.1f}  "
              f"matched={self._times['match_pct']/n:.1f}%  "
              f"| rolling{recent_n}: stereo={recent['stereo']:.1f}ms  "
              f"temporal={recent['temporal']:.1f}ms  total={recent_total:.1f}ms  "
              f"fps={1000/recent_total:.1f}")

        return dict(
            frame_idx=frame_idx,
            img_l=img_l, img_r=img_r,
            img_l_bgr=img_l_bgr,
            img_l_sm=img_l_sm, img_r_sm=img_r_sm,
            img_l_sm_bgr=img_l_sm_bgr, img_r_sm_bgr=img_r_sm_bgr,
            vis_l=vis_l, vis_r=vis_r,
            debug_l=debug_l, debug_r=debug_r,
            strong_l=strong_l, weak_l=weak_l,
            strong_r=strong_r, weak_r=weak_r,
            features_l=features_l, features_r=features_r,
            stereo=stereo,
            temporal_matches=temporal_matches,
            tm_s=tm_s, tm_mx_s=tm_mx_s, tm_cells_s=tm_cells_s, tm_rej_s=tm_rej_s,
            tm_w=tm_w, tm_mx_w=tm_mx_w, tm_cells_w=tm_cells_w, tm_rej_w=tm_rej_w,
            depth_of=depth_of,
            pose_str=pose_str,
        )

    # ── display ───────────────────────────────────────────────────────────────

    def _build_main_display(self, d, fps):
        stereo_color_of  = {id(fl): viz.disp_to_color(disp) for fl, _, disp, _ in d['stereo']}
        temp_tracked_ids = {id(cf) for _, cf, _, _ in d['temporal_matches']}

        depth_full = np.zeros((d['img_l'].shape[0], d['img_l'].shape[1], 3), dtype=np.uint8)
        video_full = d['img_l_bgr'].copy()

        for fl, _, disp, score in d['stereo']:
            if id(fl) not in temp_tracked_ids:
                continue
            color  = viz.disp_to_color(disp)
            radius = max(2, int(((score - config.NCC_THRESHOLD) /
                                 (1.0 - config.NCC_THRESHOLD)) ** 0.5 * config.MATCH_DOT_RADIUS))
            cx_px, cy_px = int(fl.pos[0] * 2), int(fl.pos[1] * 2)
            cv2.circle(depth_full, (cx_px, cy_px), radius, color, -1)
            cv2.circle(video_full, (cx_px, cy_px), radius, color, -1)

        for pf, cf, _, _ in d['temporal_matches']:
            color = stereo_color_of.get(id(cf))
            if color is None:
                continue
            p1 = tuple((pf.pos * 2).astype(int))
            p2 = tuple((cf.pos * 2).astype(int))
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if dx * dx + dy * dy < 25:
                continue
            cv2.line(video_full, p1, p2, color, 2)
            cv2.line(depth_full, p1, p2, color, 2)

        n_tm = len(d['temporal_matches'])
        viz.add_label(depth_full, f"SPARSE DEPTH  |  {len(d['stereo'])} pts  |  {d['pose_str']}")
        viz.add_label(video_full, f"OVERLAY  |  L:{len(d['features_l'])}  R:{len(d['features_r'])}"
                                  f"  |  flow:{n_tm}  |  {fps:.1f} FPS")
        return np.vstack([depth_full, video_full])

    def _show_pipeline_window(self, d):
        def norm_vis(fmap):
            v = cv2.normalize(fmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

        p1 = np.hstack((d['img_l_sm_bgr'].copy(), d['img_r_sm_bgr'].copy()))
        viz.add_label(p1, "1. STEREO PAIR")
        p2 = np.hstack((cv2.cvtColor(d['vis_l'], cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(d['vis_r'], cv2.COLOR_GRAY2BGR)))
        viz.add_label(p2, "2. HARRIS RESPONSE")
        p3 = np.hstack((norm_vis(d['debug_l'][0]), norm_vis(d['debug_r'][0])))
        viz.add_label(p3, f"3. PASS 1 — STRONG PEAKS  (thresh={config.STRONG_THRESH_FRAC})")
        p4 = np.hstack((norm_vis(d['debug_l'][1]), norm_vis(d['debug_r'][1])))
        viz.add_label(p4, f"4. SUPPRESSION MASK  (r={config.SUPPRESS_RADIUS}px)")
        p5 = np.hstack((norm_vis(d['debug_l'][2]), norm_vis(d['debug_r'][2])))
        viz.add_label(p5, "5. PASS 2 — WEAK FILL-IN")

        def draw_two(img, strong, weak):
            for f in weak:
                cv2.circle(img, tuple(f.pos.astype(int)), 3, (255, 180, 0), 1)
            for f in strong:
                cv2.circle(img, tuple(f.pos.astype(int)), 3, (0, 200, 255), 1)
            return img

        sl, wl, sr, wr = d['strong_l'], d['weak_l'], d['strong_r'], d['weak_r']
        p6 = np.hstack((draw_two(d['img_l_sm_bgr'].copy(), sl, wl),
                        draw_two(d['img_r_sm_bgr'].copy(), sr, wr)))
        viz.add_label(p6, f"6. FINAL  L:{len(d['features_l'])} (s={len(sl)} w={len(wl)})  "
                          f"R:{len(d['features_r'])} (s={len(sr)} w={len(wr)})")
        cv2.imshow("VO Pipeline", np.vstack((p1, p2, p3, p4, p5, p6)))

    def _show_temporal_window(self, d):
        def _show_corr(name, score_mx, match_cells, reject_cells, n_matches):
            NC, NP = score_mx.shape
            cell    = max(1, min(2, 800 // max(NC, NP, 1)))
            vis_mat = cv2.resize(score_mx, (NP * cell, NC * cell),
                                 interpolation=cv2.INTER_NEAREST)
            heat = cv2.applyColorMap((vis_mat * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            r = max(2, cell)
            for col, row in reject_cells:
                cx2, cy2 = col * cell + cell // 2, row * cell + cell // 2
                cv2.line(heat, (cx2 - r, cy2 - r), (cx2 + r, cy2 + r), (0, 0, 220), 1)
                cv2.line(heat, (cx2 + r, cy2 - r), (cx2 - r, cy2 + r), (0, 0, 220), 1)
            for col, row in match_cells:
                cx2, cy2 = col * cell + cell // 2, row * cell + cell // 2
                cv2.circle(heat, (cx2, cy2), r, (255, 255, 255), -1)
            viz.add_label(heat, f"{name}  {NP}×{NC}  matched:{n_matches}  rejected:{len(reject_cells)}")
            cv2.imshow(name, heat)

        if d['tm_mx_s'] is not None:
            _show_corr("NCC STRONG", d['tm_mx_s'], d['tm_cells_s'] or [],
                       d['tm_rej_s'] or [], len(d['tm_s'] or []))
        if d['tm_mx_w'] is not None:
            _show_corr("NCC WEAK",   d['tm_mx_w'], d['tm_cells_w'] or [],
                       d['tm_rej_w'] or [], len(d['tm_w'] or []))

    # ── event loop ────────────────────────────────────────────────────────────

    def run(self, auto_pipeline: bool = False, pointcloud_recorder=None, stop_when=None):
        cv2.namedWindow("VO Depth Analysis", cv2.WINDOW_NORMAL)
        video_writer  = None
        frame_idx     = 0
        paused        = False
        show_pipeline = auto_pipeline
        show_temporal = False
        prev_tick     = cv2.getTickCount()

        if self.calib is not None:
            self._open_3d()
            self._follow_cam_ref[0] = True

        while frame_idx < self.num_frames:
            d = self.process_frame(frame_idx, include_temporal_debug=show_temporal)
            if d is None:
                break

            tick      = cv2.getTickCount()
            fps       = cv2.getTickFrequency() / (tick - prev_tick)
            prev_tick = tick

            display = self._build_main_display(d, fps)
            cv2.imshow("VO Depth Analysis", display)

            if config.SAVE_VIDEO:
                if video_writer is None:
                    h, w         = display.shape[:2]
                    video_writer = cv2.VideoWriter(
                        config.VIDEO_OUTPUT_PATH,
                        cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
                video_writer.write(display)

            if show_pipeline:
                self._show_pipeline_window(d)

            if show_temporal:
                self._show_temporal_window(d)

            if self._vispy_canvas is not None:
                self._3d_frame_counter += 1
                if self._3d_frame_counter % config.VIZ_UPDATE_HZ == 0 and self.all_stereo_pts:
                    viz.update_vispy(self._vispy_view, self._vispy_visuals,
                                     self.all_stereo_pts, self.all_cam_poses,
                                     self._follow_cam_ref[0])
                    if pointcloud_recorder is not None and not pointcloud_recorder.is_full:
                        rgb = viz.render_vispy_frame(self._vispy_canvas)
                        pointcloud_recorder.add_rgb(rgb)
                viz.process_vispy_events()

            key = cv2.waitKeyEx(30 if not paused else 0)

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('p'):
                show_pipeline = not show_pipeline
                if not show_pipeline:
                    cv2.destroyWindow("VO Pipeline")
            elif key == ord('t'):
                show_temporal = not show_temporal
                if not show_temporal:
                    cv2.destroyWindow("NCC STRONG")
                    cv2.destroyWindow("NCC WEAK")
            elif key == ord('v'):
                if self.calib is None:
                    print("[3D view] calib.txt not found — cannot project to 3D")
                elif self._vispy_canvas is None:
                    self._open_3d()
                else:
                    self._close_3d()
            elif key in (2424832, ord('a')):
                frame_idx = max(0, frame_idx - 1)
                continue
            elif key in (2555904, ord('d')):
                frame_idx = min(self.num_frames - 1, frame_idx + 1)
                continue

            if stop_when is not None and stop_when():
                break

            if not paused:
                frame_idx += 1
                if frame_idx >= self.num_frames:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    frame_idx = 0
                    self._reset_sequence()

        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


class GifRecorder:
    """Accumulate BGR frames from the OpenCV display and write a palette-optimised GIF on save."""

    def __init__(self, output_path: str, max_frames: int = 150,
                 fps: float = 15.0, scale: float = 0.5):
        self.output_path = output_path
        self.max_frames  = max_frames
        self.fps         = fps
        self.scale       = scale
        self._frames: list = []

    def add(self, bgr_frame):
        if len(self._frames) >= self.max_frames:
            return
        if self.scale != 1.0:
            h, w = bgr_frame.shape[:2]
            bgr_frame = cv2.resize(bgr_frame, (int(w * self.scale), int(h * self.scale)))
        self._frames.append(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))

    def add_rgb(self, rgb_frame):
        if len(self._frames) >= self.max_frames:
            return
        if self.scale != 1.0:
            h, w = rgb_frame.shape[:2]
            rgb_frame = cv2.resize(rgb_frame, (int(w * self.scale), int(h * self.scale)))
        self._frames.append(rgb_frame)

    def save(self):
        if not self._frames:
            print('[GIF] no frames captured - nothing written')
            return
        try:
            import imageio
        except ImportError:
            print('[GIF] imageio not installed - run: pip install imageio[ffmpeg]')
            return
        duration = 1.0 / self.fps
        print(f'[GIF] writing {len(self._frames)} frames -> {self.output_path}')
        imageio.mimsave(self.output_path, self._frames, duration=duration, loop=0)
        print(f'[GIF] saved {self.output_path}')

    @property
    def is_full(self):
        return len(self._frames) >= self.max_frames


def _build_pipeline_frame(d):
    """Render the 6-step feature pipeline into a single BGR image (for GIF capture)."""
    def norm_vis(fmap):
        v = cv2.normalize(fmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    def draw_two(img, strong, weak):
        for f in weak:
            cv2.circle(img, tuple(f.pos.astype(int)), 3, (255, 180, 0), 1)
        for f in strong:
            cv2.circle(img, tuple(f.pos.astype(int)), 3, (0, 200, 255), 1)
        return img

    p1 = np.hstack((d['img_l_sm_bgr'].copy(), d['img_r_sm_bgr'].copy()))
    viz.add_label(p1, "1. STEREO PAIR")
    p2 = np.hstack((cv2.cvtColor(d['vis_l'], cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(d['vis_r'], cv2.COLOR_GRAY2BGR)))
    viz.add_label(p2, "2. HARRIS RESPONSE")
    p3 = np.hstack((norm_vis(d['debug_l'][0]), norm_vis(d['debug_r'][0])))
    viz.add_label(p3, f"3. PASS 1 — STRONG PEAKS  (thresh={config.STRONG_THRESH_FRAC})")
    p4 = np.hstack((norm_vis(d['debug_l'][1]), norm_vis(d['debug_r'][1])))
    viz.add_label(p4, f"4. SUPPRESSION MASK  (r={config.SUPPRESS_RADIUS}px)")
    p5 = np.hstack((norm_vis(d['debug_l'][2]), norm_vis(d['debug_r'][2])))
    viz.add_label(p5, "5. PASS 2 — WEAK FILL-IN")
    sl, wl, sr, wr = d['strong_l'], d['weak_l'], d['strong_r'], d['weak_r']
    p6 = np.hstack((draw_two(d['img_l_sm_bgr'].copy(), sl, wl),
                    draw_two(d['img_r_sm_bgr'].copy(), sr, wr)))
    viz.add_label(p6, f"6. FINAL  L:{len(d['features_l'])} (s={len(sl)} w={len(wl)})  "
                      f"R:{len(d['features_r'])} (s={len(sr)} w={len(wr)})")
    return np.vstack((p1, p2, p3, p4, p5, p6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binocular Visual Odometry')
    parser.add_argument('--sequence', type=int, default=config.SEQUENCE_ID,
                        help=f'KITTI sequence ID (default: {config.SEQUENCE_ID})')
    parser.add_argument('--record-gif', choices=['depth', 'pipeline'], default=None,
                        help='Record a GIF: "depth" = main depth+overlay window, '
                             '"pipeline" = 6-step feature extraction window')
    parser.add_argument('--gif-out', default=None,
                        help='Output path for the GIF (default: depth.gif / pipeline.gif)')
    parser.add_argument('--gif-frames', type=int, default=config.GIF_FRAMES)
    parser.add_argument('--gif-fps', type=float, default=config.GIF_FPS)
    parser.add_argument('--gif-scale', type=float, default=config.GIF_SCALE)
    parser.add_argument('--pointcloud-color', choices=['depth', 'video'], default=None,
                        help='Point cloud GIF colour mode (default: config.PC_DEPTH_COLOR)')
    args = parser.parse_args()

    if args.pointcloud_color is not None:
        config.PC_DEPTH_COLOR = (args.pointcloud_color == 'depth')

    # CLI flag takes priority; fall back to config bools
    record_depth      = (args.record_gif == 'depth')    or (args.record_gif is None and config.RECORD_GIF_DEPTH)
    record_pipeline   = (args.record_gif == 'pipeline') or (args.record_gif is None and config.RECORD_GIF_PIPELINE)
    record_pointcloud = (args.record_gif is None and config.RECORD_GIF_POINTCLOUD)

    pipeline = VOPipeline(sequence_id=args.sequence)

    def _make_recorder(name):
        import os
        out_dir = os.path.join(config.GIF_OUTPUT_DIR, f'seq{args.sequence:02d}')
        os.makedirs(out_dir, exist_ok=True)
        gif_name = name
        if name == 'pointcloud':
            color_mode = 'depth-color' if config.PC_DEPTH_COLOR else 'video-color'
            gif_name = f'{name}_{color_mode}'
        out_path = args.gif_out or os.path.join(out_dir, f'{gif_name}.gif')
        r = GifRecorder(out_path, max_frames=args.gif_frames, fps=args.gif_fps, scale=args.gif_scale)
        print(f'[GIF] recording "{name}" -> {out_path}  '
              f'(max {args.gif_frames} frames @ {args.gif_fps} fps, scale={args.gif_scale})')
        return r

    depth_recorder      = _make_recorder('depth')      if record_depth      else None
    pipeline_recorder   = _make_recorder('pipeline')   if record_pipeline   else None
    pointcloud_recorder = _make_recorder('pointcloud') if record_pointcloud else None

    if pipeline_recorder is not None:
        _orig_show_pipeline = pipeline._show_pipeline_window

        def _recording_show_pipeline(d):
            _orig_show_pipeline(d)
            if not pipeline_recorder.is_full:
                pipeline_recorder.add(_build_pipeline_frame(d))
        pipeline._show_pipeline_window = _recording_show_pipeline

    if depth_recorder is not None:
        _orig_build = pipeline._build_main_display

        def _recording_build(d, fps):
            frame = _orig_build(d, fps)
            if not depth_recorder.is_full:
                depth_recorder.add(frame)
            return frame
        pipeline._build_main_display = _recording_build

    active_recorders = [r for r in (depth_recorder, pipeline_recorder, pointcloud_recorder)
                        if r is not None]

    pipeline.run(auto_pipeline=pipeline_recorder is not None,
                 pointcloud_recorder=pointcloud_recorder,
                 stop_when=lambda: active_recorders and all(r.is_full for r in active_recorders))

    if depth_recorder is not None:
        depth_recorder.save()
    if pipeline_recorder is not None:
        pipeline_recorder.save()
    if pointcloud_recorder is not None:
        pointcloud_recorder.save()
