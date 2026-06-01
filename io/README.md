# `io/` — interop

Read/write calibration in the formats the ecosystem expects:

- OpenCV YAML · ROS `CameraInfo` · Kalibr YAML · COLMAP.
- **Isaac Perceptor URDF** — emit a URDF/xacro at `/etc/nova/calibration/isaac_calibration.urdf` (REP-0103 frames, `base_link` root, fixed joints, canonical sensor names) for drop-in ingestion by the dominant edge stack. *(The exact camera-distortion model schema Perceptor accepts is unconfirmed — verify before relying; see docs/RESEARCH.md Theme 5.)*

**Status:** placeholder — no implementation yet.
