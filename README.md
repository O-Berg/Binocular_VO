# Binocular Visual Odometry

Visual odometry, implemented with Claude Code, from raw KITTI stereo frames to a live 3D point cloud.

The pipeline takes a pair of camera images, extracts stable Harris features, matches them between the left and right views, estimates depth through disparity, tracks motion over time, and accumulates the result into a world-space point cloud.

<p align="center">
  <img src="recordings/seq04/depth.gif" width="48%" alt="Depth overlay from KITTI stereo frames">
  <img src="recordings/seq04/pointcloud_depth-color.gif" width="48%" alt="Accumulated point cloud with depth colouring">
</p>

---

## Pipeline

The project is built as a small end-to-end VO stack:

1. Load KITTI binocular image pairs.
2. Resize and preprocess the left/right frames.
3. Detect Harris responses and suppress nearby duplicate peaks.
4. Split features into strong anchors and weak fill-in points.
5. Match stereo features with NCC in a rectified epipolar band.
6. Convert disparity into 3D points using calibration.
7. Track features through time and reject geometric outliers.
8. Estimate camera motion and accumulate points in a 3D scene.

---

<p align="center">
  <img src="recordings/seq04/pipeline.gif" width="90%" alt="Feature extraction and stereo matching pipeline">
</p>

---

## Point Cloud Colouring

The point cloud can be rendered in two modes:

<p align="center">
  <img src="recordings/seq04/pointcloud_depth-color.gif" width="48%" alt="Point cloud coloured by depth">
  <img src="recordings/seq04/pointcloud_video-color.gif" width="48%" alt="Point cloud coloured by source video">
</p>

`depth` colour makes distance structure easier to read. `video` colour samples the source image patches so the cloud keeps some of the original scene appearance.

---

## Repository Structure

```text
Binocular_VO/
|-- main.py              # application entry point and VO loop
|-- config.py            # dataset, feature, matching, output, and GPU settings
|-- dataset.py           # KITTI sequence loading and calibration helpers
|-- features.py          # Harris features, patch extraction, and NCC backend
|-- matching.py          # stereo and temporal feature matching
|-- pose.py              # pose estimation and outlier filtering support
|-- visualization.py     # depth maps, overlays, and VisPy point-cloud rendering
|-- recordings/          # tracked GIF showcases for sequences 00-04
|-- requirements.txt     # Python dependencies
`-- dataset/sequences/   # local KITTI data, ignored by Git
```

---

## GPU Acceleration

The expensive temporal NCC matrix can run through CuPy when `USE_GPU = True` in `config.py`. On startup the app now reports the actual backend clearly:

```text
[GPU] cupy OK - temporal NCC acceleration enabled
```

or:

```text
[GPU] disabled by config.USE_GPU - using numpy
```

Stereo matching intentionally stays on the sparse CPU path, which is faster for the current candidate counts than a dense GPU all-pairs matrix. In short: GPU is used where it helps, and avoided where transfer and matrix overhead would slow the run down.

---

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default sequence:

```bash
python main.py
```

Run a specific KITTI sequence:

```bash
python main.py --sequence 4
```

Record point-cloud GIFs manually:

```bash
python main.py --sequence 4 --pointcloud-color depth
python main.py --sequence 4 --pointcloud-color video
```

GIF recording is disabled by default in `config.py` so normal runs do not overwrite the showcase recordings.

---

## Notes

The KITTI dataset is not committed to the repository. Place the sequences under:

```text
dataset/sequences/
```

Generated examples are stored under:

```text
recordings/seqXX/
```
