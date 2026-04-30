# ── Dataset ───────────────────────────────────────────────────────────────────
SEQUENCE_ID = 3   # KITTI sequence to run (0–10 for odometry benchmark)

# ── Feature extraction ────────────────────────────────────────────────────────
HARRIS_K           = 0.04
HARRIS_BLOCK       = 3
NMS_KERNEL         = 7
NUM_STRONG         = 1000
NUM_WEAK           = 1000
STRONG_THRESH_FRAC = 0.025
SUPPRESS_RADIUS    = 5
USE_GPU            = True

# ── Stereo matching ───────────────────────────────────────────────────────────
NCC_THRESHOLD     = 0.95
DISP_SEARCH       = 40
EPIPOLAR_MARGIN   = 2
STEREO_PATCH_SIZE = 11
MATCH_DOT_RADIUS  = 5

# ── Temporal matching ─────────────────────────────────────────────────────────
TEMPORAL_SEARCH_RADIUS = 50
TEMPORAL_NCC_THRESHOLD = 0.95
TEMPORAL_MAX_DIST      = 40
TEMPORAL_PATCH_SIZE    = 11

# ── Depth colourmap ───────────────────────────────────────────────────────────
DISP_MIN = 0
DISP_MAX = 40

# ── Outlier filtering ─────────────────────────────────────────────────────────
OUTLIER_RANSAC_THRESHOLD = 1.5
OUTLIER_MAD_SCALE        = 3.0

# ── 3D display ────────────────────────────────────────────────────────────────
DEPTH_MIN_M       = 0
DEPTH_MAX_M       = 80.0
MAX_DISP_3D_M     = 5.0
VIZ_HISTORY       = 1
VIZ_UPDATE_HZ     = 1
PC_HISTORY        = 60
PC_ACCUMULATE_ALL = True
PC_DEPTH_COLOR    = True
PC_POINT_SIZE     = 4
PC_SATURATION     = 2.0

# ── Pose estimation ───────────────────────────────────────────────────────────
DATASET_FPS      = 10.0
PNP_REPROJ_ERROR = 2.0
PNP_MIN_INLIERS  = 6

# ── Output ────────────────────────────────────────────────────────────────────
SAVE_VIDEO        = False
VIDEO_OUTPUT_PATH = 'output.mp4'

RECORD_GIF_DEPTH      = False  # record main depth+overlay window to depth.gif on exit
RECORD_GIF_PIPELINE   = False  # record 6-step pipeline window to pipeline.gif on exit
RECORD_GIF_POINTCLOUD = False  # record 3D point cloud window to pointcloud.gif on exit
GIF_OUTPUT_DIR      = 'recordings'
GIF_FRAMES          = 150
GIF_FPS             = 10.0
GIF_SCALE           = 0.5
