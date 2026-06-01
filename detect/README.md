# `detect/` — observation extraction

Turns images into the observations the solver consumes: `View{object_points, image_points}`.

## Two paths (v0.2)

- **Hand-rolled, always built (CPU, no deps).** `board.hpp` (geometry), `render_board.hpp`
  (synthetic anti-aliased checkerboard renderer), `corner_detect.hpp` (saddle-response
  detection + quadratic sub-pixel refinement), `checkerboard_detect.hpp` (match corners →
  `View`). Robust on **clean / rendered** boards (~0.12 px mean corner accuracy on tilted
  synthetic boards); this is what the dependency-free CI suite exercises. *Not* a production
  detector for noisy/blurred/occluded real images — that is the OpenCV path below or future
  hardening.

- **Optional OpenCV (`detect_opencv.hpp`, gated on `CALIBFORGE_HAS_OPENCV`).** Wraps
  OpenCV's robust `findChessboardCornersSB` / `cornerSubPix` (and, in future, `cv::aruco`
  for ChArUco / AprilGrid) for real-image detection. Built/tested only where OpenCV is
  installed (`calibforge_opencv_tests`); absent on the bare CI runner so the green gate is
  unaffected. OpenCV is Apache-2.0, optional — see [`../docs/DEPENDENCIES.md`](../docs/DEPENDENCIES.md).

- **Targetless (future):** feature tracks for online calibration; BEV photometric error in
  overlapping regions (surround-view); tie-points for UAV blocks. See
  [`../docs/RESEARCH.md`](../docs/RESEARCH.md) Theme 3.

Full from-scratch ChArUco/AprilGrid **marker decoding** (IDs under occlusion/rotation) is
delegated to the OpenCV path; a hand-rolled clean-render decoder is future work.
