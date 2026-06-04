# CalibForge API

> **Stability:** the headers below are the entry points users link against. They are
> header-only and live under `core/`, `solve/`, `pipelines/`, `online/`, `detect/`, `io/`,
> `apply/`, `bindings/`. Detailed semantics live as comments in the headers themselves;
> this document is the index + the contract.

Everything in CalibForge lives inside `namespace calibforge` (sometimes with a sub-namespace
like `calibforge::online`, `calibforge::detect`, `calibforge::pipelines`).

---

## Camera models — `core/`

All camera models implement the model-agnostic `CameraModel` interface
([`core/include/calibforge/camera_model.hpp`](../core/include/calibforge/camera_model.hpp)):

```cpp
class CameraModel {
 public:
  virtual Vec2 project(const Vec3& point_cam) const = 0;
  virtual Vec3 unproject(const Vec2& pixel) const = 0;
  virtual Jacobian projectJacobianWrtParams(const Vec3& point_cam) const = 0;
  virtual Jacobian projectJacobianWrtPoint(const Vec3& point_cam) const = 0;
  virtual std::size_t numParams() const = 0;
  virtual std::string name() const = 0;
};
```

Built-in implementations:

| Class | Header | Params |
|---|---|---|
| `PinholeCamera` | `pinhole_camera.hpp` | `fx, fy, cx, cy` |
| `BrownConradyCamera` | `brown_conrady_camera.hpp` | `fx, fy, cx, cy, k1, k2, p1, p2, k3` |
| `KannalaBrandtCamera` (fisheye) | `kannala_brandt_camera.hpp` | `fx, fy, cx, cy, k1, k2, k3, k4` |
| `DoubleSphereCamera` | `double_sphere_camera.hpp` | `fx, fy, cx, cy, xi, alpha` |
| `EUCMCamera` | `eucm_camera.hpp` | `fx, fy, cx, cy, alpha, beta` |
| `GenericBSplineCamera` (Schöps CVPR 2020) | `generic_bspline_camera.hpp` | dense per-pixel ray field via cubic B-spline over `Nx × Ny` control points (`3 * Nx * Ny` direction values) |

The `GenericBSplineCamera` is the v1.0 dense-ray-field model. It exposes `fitFromParametricCamera(source)`
to initialize the control grid from any other `CameraModel`, which is the recommended starting
point for online generic-model recalibration.

---

## Solver — `solve/`

The solver is **agnostic to camera model and residual type**. You assemble a `DenseProblem`
out of parameter blocks + residual blocks, then call `solveLm(LmOptions)`.

- `ResidualBlock` ([`residual_block.hpp`](../solve/include/calibforge/residual_block.hpp))
  is the contract every residual implements (`evaluate(params, residual, jacobians)`).
- `DenseProblem` ([`dense_problem.hpp`](../solve/include/calibforge/dense_problem.hpp)) is
  the assembler / LM driver. It supports `EuclideanParam` and `SE3Param` / `SO3Param`
  manifolds, FastTriggs robust losses, and emits an information matrix at the solution.
- `solveLm(LmOptions, SolverBackend)` selects the per-iteration dense linear solver. The
  default `CpuCeres` keeps the host Eigen LDLT path; `GpuCuda`
  ([`cuda_linear_solver.hpp`](../solve/include/calibforge/cuda_linear_solver.hpp)) offloads the
  damped normal-equations solve `(JᵀJ + λ·diag(JᵀJ)) dx = −Jᵀr` to the GPU (cuBLAS SYRK/GEMV +
  cuSOLVER dense Cholesky) when built with CUDA, and transparently falls back to the host path
  otherwise (`cudaSolverAvailable()` mirrors the build). Analytic Jacobians stay host-assembled
  (rule 4). **Rule 1: the GPU is not automatically faster — it wins only past a measured
  crossover (~8 views / `n_tangent`≈52 on an RTX 5090; see [`BENCHMARKS.md`](./BENCHMARKS.md)),
  so CPU stays the default for small single calibrations.**

Built-in residuals:

| Residual | Header | Use |
|---|---|---|
| `ReprojectionResidual` | `reprojection_residual.hpp` | Point reprojection (cam + pose) |
| `StereoReprojectionResidual` | `stereo_reprojection_residual.hpp` | Reprojection through one extra extrinsic |
| `HandEyeResidual` | `hand_eye_residual.hpp` | AX = XB hand-eye constraint |
| `RollingShutterResidual` | `rolling_shutter_residual.hpp` | RS reprojection at row-specific pose (v0.4) |
| `CamImuGyroResidual` | `cam_imu_gyro_residual.hpp` | Cam-IMU rotation init (v0.4) |
| `ImuPreintegrationFactorResidual` | `imu_preintegration_residual.hpp` | Forster IMU factor (v0.4); pairs with `ImuPreintegrator` |

---

## Pipelines — `pipelines/` / `solve/calibrate_*`

Header | What it does
---|---
`calibrate_single.hpp` | Single-camera intrinsic + per-view-pose calibration. Returns intrinsics + poses + information matrix.
`calibrate_stereo.hpp` | Two cameras with a shared rigid baseline + per-view rig pose.
`calibrate_rig.hpp` | N-camera rig generalizing stereo: N intrinsics + (N-1) extrinsics + per-view rig poses.
`calibrate_hand_eye.hpp` | Robot AX = XB calibration over a vector of (A, B) sample pairs.
`calibrate_rolling_shutter.hpp` | RS readout-time `t_r` as a calibrated parameter (v0.4).
`calibrate_cam_imu.hpp` | Cam-IMU spatio-temporal extrinsic init via gyro (v0.4).

All pipeline functions take `LmOptions` and return a result struct including `LmSummary`,
`information` (`J^T J` at the solution, feed to `assessObservability`), and `num_residuals`
(feed to `parameterUncertainty`).

---

## Online recalibration (the v0.5 differentiator) — `online/`

The **observability gate** is mandatory on every online emission (CLAUDE.md rule 2 / `docs/RESEARCH.md` Theme 3).
Low cost does not mean trustworthy parameters — precision ≠ accuracy.

| Class | Header | Use |
|---|---|---|
| `OnlineIntrinsicTracker` | `online/online_calibration.hpp` | Drift tracking of single-camera intrinsics behind the gate |
| `OnlineExtrinsicTracker` | `online/online_extrinsic_tracker.hpp` | Drift tracking of rig extrinsics behind the gate + 6-axis motion gate |

Both expose `addFrame(view, pose) / tryEmit(min_confidence, ...)`. The intrinsic tracker
returns `Emission`; the extrinsic tracker returns `ExtrinsicEmission` (which adds the refined
`extrinsics`, `refused_for_motion`, and `unexcited_axes`). Both carry `emitted` (false ⇒
refused), the recovered parameters when emitted, `confidence` (observability rcond), `drift`
vs the reference, and `weak_parameters` (named directions the gate flagged).

---

## Targetless front-ends — `detect/`

| Header | Use |
|---|---|
| `feature_tracker.hpp` | Stateful KLT-lite over saddle corners. Produces `FeatureTrack` per landmark across N frames. |
| `triangulate.hpp` | Linear DLT triangulation + `FeatureTrack -> View` packing for `OnlineIntrinsicTracker`. |
| `bev_photometric.hpp` | BEV ground-plane photometric agreement + coarse-to-fine random search for surround-rig extrinsics (OpenCalib SurroundCameraCalib math, re-implemented). |
| `checkerboard_detect.hpp` / `corner_detect.hpp` / `board.hpp` | Target-based detection. |

---

## Orchestrators — `pipelines/online_*.hpp`

| Header | Use |
|---|---|
| `online_uav.hpp` | UAV in-flight self-cal: `FeatureTracker → triangulate → packMonocularViews → OnlineIntrinsicTracker`, returns the gated emission. |
| `online_surround_rig.hpp` | Surround-view targetless extrinsic refinement: BEV photometric coarse-to-fine random search behind the cost-reduction + 6-axis motion gates. |

---

## Apply (runtime undistort/rectify) — `apply/`

| Header | Use |
|---|---|
| `apply/warp_map.hpp` | Generate CPU warp map. Reference path; parity oracle for the GPU/Jetson paths. |
| `apply/remap_cuda.cu` | Server CUDA undistort (compiled when `nvcc` is found). |
| `apply/vpi_ldc.hpp` | Jetson VPI LDC export (PVA/VIC); compiled when the VPI SDK is found. |

---

## I/O interop — `io/`

| Header | Use |
|---|---|
| `io/opencv_yaml.hpp` | OpenCV `CameraInfo`-style YAML round-trip |
| `io/ros_camera_info.hpp` | ROS `sensor_msgs/CameraInfo` emit/parse |
| `io/kalibr_camchain.hpp` | Kalibr `camchain.yaml` parse |
| `io/isaac_urdf.hpp` | Isaac URDF + Perceptor URDF |

---

## Bindings — `bindings/`

| Subdir | Use |
|---|---|
| `bindings/python/` | pybind11 module (`calibforge`). Built when `-DCALIBFORGE_PYTHON=ON`. |
| `bindings/ros2/` | ROS 2 stubs (calibration-as-a-node). |

---

## Build matrix

| What | When built |
|---|---|
| `calibforge_core` (header-only INTERFACE) | always |
| `calibforge_tests` (~114 cases host-only; 122 with CUDA) | always (host-only, no CUDA needed) |
| `calibforge_opencv_tests` | when OpenCV is found |
| `calibforge_cuda` (CUDA undistort remap + native dense LM solver) + `calibforge_arch_probe` | when `nvcc` is found |
| `calibforge_python` | with `-DCALIBFORGE_PYTHON=ON` |
| `calibforge_bench` (CPU regime + CPU-vs-GPU solver rows when CUDA is present) | always |

Default arch matrix when CUDA is available: `sm_72;80;86;87;89;90` **plus `90-virtual`** (Jetson
+ server + desktop in one build — CLAUDE.md rule 6). The trailing `90-virtual` embeds `compute_90`
PTX so the fatbin JIT-forwards onto archs newer than the toolkit knows (e.g. `sm_120` / Blackwell
on a CUDA 12.0 host); validated firsthand on an RTX 5090 — see [`SPIKES.md` §E](./SPIKES.md).

---

## Cross-references

- `docs/DESIGN.md` — overall spec / intent.
- `docs/RESEARCH.md` — research validations behind each module decision.
- `docs/DEPENDENCIES.md` — borrow vs build, with licenses.
- `docs/SPIKES.md` — empirical de-risking results + what's deferred to CUDA / Jetson hosts.
- `docs/TUTORIAL.md` — end-to-end calibration walkthrough.
