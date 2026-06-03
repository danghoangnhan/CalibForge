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
  OpenCV's robust `findChessboardCornersSB` / `cornerSubPix` for checkerboards, and — when
  the aruco component is available (gated on `CALIBFORGE_HAS_OPENCV_ARUCO`, default in
  Ubuntu 24.04's `libopencv-dev`) — `cv::aruco::detectMarkers` +
  `interpolateCornersCharuco` for **ChArUco** boards and `cv::aruco::detectMarkers` for
  **AprilGrid** (Kalibr-style independent tag grid, DICT_APRILTAG_36h11). Built/tested
  only where OpenCV is installed (`calibforge_opencv_tests`); the `opencv` CI job
  installs `libopencv-dev` and runs them end-to-end. The bare CI runner remains the
  green-gate signal. OpenCV is Apache-2.0, optional — see
  [`../docs/DEPENDENCIES.md`](../docs/DEPENDENCIES.md).

- **Targetless (future):** feature tracks for online calibration; BEV photometric error in
  overlapping regions (surround-view); tie-points for UAV blocks. See
  [`../docs/RESEARCH.md`](../docs/RESEARCH.md) Theme 3.

ChArUco / AprilGrid support is therefore via the OpenCV-gated path (issue #8 closed under
that decision). A hand-rolled from-scratch ChArUco / AprilGrid marker decoder (full ID
recovery under occlusion / rotation, no OpenCV dependency) remains future work.
