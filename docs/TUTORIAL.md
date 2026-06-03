# CalibForge Tutorial

This walks through the common calibration recipes end-to-end. Every example is real CPU
code from the test suite — see `tests/test_*.cpp` for runnable references.

## 0. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CalibForge is header-only, but the build needs Eigen + Sophus + (optional) OpenCV. The
top-level `CMakeLists.txt` `FetchContent`s Eigen 3.4 and Sophus automatically; OpenCV is
gated and the rest of the suite stays green without it. CUDA is detected when `nvcc` is
present; the host CPU half (~35 tests) is the always-green signal.

## 1. Single-camera intrinsic calibration

```cpp
#include "calibforge/calibrate_single.hpp"
#include "calibforge/pinhole_camera.hpp"

using namespace calibforge;

// 1. Build views from observed (object_point, pixel) pairs.
std::vector<View> views = ...;          // one per checkerboard observation

// 2. Initial intrinsics + per-view-pose guesses.
Eigen::VectorXd intr0(4);
intr0 << 500, 500, 320, 240;            // fx, fy, cx, cy
std::vector<Sophus::SE3d> poses0 = ...;

// 3. Camera factory: build a PinholeCamera from the parameter vector.
CameraFactory mk = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
  return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
};

// 4. Solve. Returns refined intrinsics, refined poses, an LM summary, and the information
//    matrix J^T J at the solution.
SingleCameraResult res = calibrateSingleCamera(views, intr0, poses0, mk);
```

Swap `PinholeCamera` for `BrownConradyCamera`, `KannalaBrandtCamera`, `DoubleSphereCamera`,
or `EUCMCamera` — the pipeline is model-agnostic. See `tests/test_calibrate_single.cpp` for
every recovery test.

## 2. Observability gating — never trust intrinsics blindly

```cpp
#include "calibforge/observability.hpp"

ObservabilityReport rep = assessObservability(res.information);
if (!rep.observable) {
  // Geometry-only check failed — unobservable directions in your data.
  return;
}
ParameterUncertainty unc =
    parameterUncertainty(res.information, res.summary.final_cost, res.num_residuals,
                         {"fx","fy","cx","cy"});
for (const std::string& weak : unc.weak_parameters) {
  // Tell the operator which parameter the data didn't constrain.
}
```

CLAUDE.md rule 2: **never silently emit calibration parameters online.** Low cost ≠ accuracy.
The intrinsic / extrinsic trackers below wrap this gate for you.

## 3. Stereo, hand-eye, N-camera rig

```cpp
#include "calibforge/calibrate_stereo.hpp"
StereoResult sres = calibrateStereo(stereo_views, intr_l0, intr_r0, T_r_l_init,
                                    poses0, mk, mk);

#include "calibforge/calibrate_hand_eye.hpp"
HandEyeResult he = calibrateHandEye(hand_eye_samples, T_init);

#include "calibforge/calibrate_rig.hpp"
RigResult rig = calibrateRig(rig_views, intr_init_per_cam, extr_init_T_ck_c0,
                             poses0, factories_per_cam);
```

## 4. Rolling shutter (v0.4)

```cpp
#include "calibforge/calibrate_rolling_shutter.hpp"
RollingShutterResult rs = calibrateRollingShutter(views_with_row_index, t_r_init, ...);
// rs.t_r is the estimated whole-image readout time; t_r = 0 ⇒ global shutter.
```

The RS pipeline reuses the existing `DenseProblem` — RS is just another residual type,
not a separate solver. See `tests/test_rolling_shutter.cpp`.

## 5. Cam-IMU (v0.4)

Rotation init via gyro:
```cpp
#include "calibforge/calibrate_cam_imu.hpp"
CamImuRotResult ci = calibrateCamImuGyro(cam_omegas, imu_samples, init_R_ic, init_t_d);
```

Full Forster preintegration (bias / gravity / position):
```cpp
#include "calibforge/imu_preintegrator.hpp"
#include "calibforge/imu_preintegration_residual.hpp"

ImuPreintegrator pi(b_g_nom, b_a_nom);
for (const ImuSample& s : samples) pi.addSample(s.w, s.a, s.dt);

// The resulting factor block ties (R_i, p_i, v_i) to (R_j, p_j, v_j) given the integrated
// preintegration window + current (b_g, b_a, g_w). Plug into DenseProblem alongside
// ReprojectionResiduals for full visual-inertial bundle adjustment.
auto factor = std::make_unique<ImuPreintegrationFactorResidual>(&pi);
```

## 6. Online intrinsic recalibration (the v0.5 differentiator)

```cpp
#include "calibforge/online_calibration.hpp"

OnlineIntrinsicTracker tracker(mk, intr_reference, {"fx","fy","cx","cy"});
for (const auto& [view, pose] : new_data) tracker.addFrame(view, pose);

Emission e = tracker.tryEmit(/*min_confidence=*/1e-3);
if (e.emitted) {
  use_drifted(e.intrinsics);        // pass the gate -> safe to publish
  log("drift = " + std::to_string(e.drift));
} else {
  log_weak_directions(e.weak_parameters);  // the gate refused; ask the operator for the
                                            // data of the right kind
}
```

## 7. Online extrinsic recalibration — rig drift behind the gate

```cpp
#include "calibforge/online_extrinsic_tracker.hpp"

OnlineExtrinsicTracker rig_tracker(factories_per_cam, intr_reference_per_cam,
                                   extr_reference_T_ck_c0);
for (const auto& [view, rig_pose] : new_data) rig_tracker.addFrame(view, rig_pose);

ExtrinsicEmission e = rig_tracker.tryEmit(/*min_confidence=*/1e-3);
if (e.refused_for_motion) {
  // 6-axis motion gate refused; the rig hasn't moved on these axes:
  for (const std::string& a : e.unexcited_axes) /* ask for that motion */;
} else if (e.emitted) {
  use_drifted_extrinsics(e.extrinsics);
}
```

## 8. UAV in-flight self-cal orchestrator

```cpp
#include "calibforge/online_uav.hpp"

pipelines::OnlineUavOptions opts;
opts.triangulation_min_track_length = 4;
opts.emit_min_confidence = 1e-3;

pipelines::OnlineUav uav(mk, intr_reference, {"fx","fy","cx","cy"}, opts);
for (const auto& [image, T_world_cam] : vio_stream) uav.addFrame(image, T_world_cam);
online::Emission e = uav.tryEmit();
```

Internally: stateful `FeatureTracker` → `triangulateTracks` → `packMonocularViews` →
gated `OnlineIntrinsicTracker.tryEmit`. The orchestrator surfaces the gated emission
verbatim — autopilot decides whether to act on a refusal.

## 9. Surround-view targetless extrinsic refinement

```cpp
#include "calibforge/online_surround_rig.hpp"

pipelines::OnlineSurroundRigOptions opts;
opts.bev_grid.x_min = -3.0; opts.bev_grid.x_max = 3.0;
opts.bev_grid.y_min =  0.5; opts.bev_grid.y_max = 3.5;
opts.bev_grid.step  = 0.05;

pipelines::OnlineSurroundRig orch(factories_per_cam, intr_reference_per_cam,
                                  extr_reference_T_ck_c0, opts);
for (const auto& [images, rig_pose] : rig_stream) orch.addFrame(images, rig_pose);
auto e = orch.tryEmit();
```

Internally: coarse-to-fine BEV photometric random search (OpenCalib SurroundCameraCalib math,
re-implemented) behind the cost-reduction sanity gate + the same 6-axis motion gate as
`OnlineExtrinsicTracker`.

## 10. Apply the calibration

Runtime undistort/rectify path:
```cpp
#include "calibforge/warp_map.hpp"
WarpMap m = generateWarpMap(camera, target_size, ...);
Image8 out = remap(in, m);           // CPU reference
```
Server CUDA path (`apply/remap_cuda.cu`) and Jetson VPI-LDC export
(`apply/vpi_ldc.hpp`) follow the same `WarpMap` contract.

---

## CSV benchmark

```bash
cmake --build build --target calibforge_bench
./build/calibforge_bench > bench.csv
```

The CSV reports `cpu_ms_median`, `cpu_iters_median`, `final_cost_median` per problem size.
GPU rows (PyPose / Graphite / Ceres comparison) require a CUDA / Jetson host — see
`docs/SPIKES.md` §D.1 for the deferred plan.
