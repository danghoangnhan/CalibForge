# CalibForge

**A unified, NVIDIA-accelerated geometric camera-calibration library that both *estimates* and *applies* calibration across pinhole, fisheye, and generic models — built once, deployed on both edge (Jetson) and server.**

> **Status:** v0.5 in progress. The CPU calibration core, the observability-gated online tracker (the project differentiator), the IMU preintegration factor, the runtime undistort path, and the generic per-pixel B-spline model (CPU, header-only) are implemented and tested. The GPU solver back-ends and the edge↔server numerical-parity test are deferred to v1.0 pending a CUDA / Jetson host; the B-spline model's pipeline wiring + wide-FOV validation are likewise pending.

## Why it exists

Every building block of geometric calibration is public, but **no single library does all of it, GPU-accelerated, deployable on both edge and server.** Today you estimate parameters with one CPU tool (OpenCV, Kalibr), apply them with another (VPI), and re-implement when you move from a server to a Jetson. CalibForge is the missing integration: a camera-model-agnostic, differentiable calibration core with a CPU-default / GPU-when-it-pays solver, that estimates *and* emits runtime undistortion artifacts, from one codebase.

## Capability matrix

| Capability | v0.x | v1.0 |
|---|:---:|:---:|
| Pinhole · Brown–Conrady distortion | ✅ | ✅ |
| Kannala–Brandt fisheye | ✅ | ✅ |
| Double-sphere · EUCM (the nvTorchCam gap) | ✅ | ✅ |
| Generic / per-pixel B-spline model | — | ⏳ |
| Single-cam extrinsics (PnP) · Stereo + rectification | ✅ | ✅ |
| Multi-camera rig · Hand-eye | ✅ | ✅ |
| Camera–IMU rotation init + full Forster preintegration factor | ✅ | ✅ |
| Rolling-shutter calibration | ✅ | ✅ |
| Online intrinsic + extrinsic recalibration behind the observability gate | ✅ | ✅ |
| Targetless feature tracker | ✅ | ✅ |
| Runtime undistort: VPI-LDC export (Jetson) · CUDA / CV-CUDA (server) | ✅ | ✅ |
| Robust loss (FastTriggs Huber / Cauchy) | ✅ | ✅ |
| Per-parameter uncertainty + observability gate (the differentiator) | ✅ | ✅ |
| ROS `CameraInfo` · Isaac Perceptor URDF · OpenCV YAML · Kalibr camchain | ✅ | ✅ |
| Python bindings (pybind11) · ROS2 node | ✅ | ✅ |
| GPU solver back-end (PyPose primary, Graphite edge, Ceres CPU oracle) | — | ⏳ |
| Edge↔server numerical-parity test (FP32/bf16 vs FP64; Jetson vs server) | — | ⏳ |

Scope is strictly **geometric** (no photometric / color / structured-light).

## Repository layout

```
docs/
  DESIGN.md         Vision + engineering spec v0.1 (owner intent)
  RESEARCH.md       Cited, adversarially-verified mid-2026 research + revised stack
  DEPENDENCIES.md   Build-vs-borrow table + the permissive-only vendoring rule
  SPIKES.md         Empirical dependency de-risking results
core/               Camera models (pinhole, Brown-Conrady, KB fisheye, double-sphere, EUCM) + Lie groups
detect/             Checkerboard + ChArUco / AprilGrid (OpenCV-gated) + sparse feature tracker
solve/              DenseProblem / ResidualBlock interface, manifold LM, observability gate, IMU preint factor
pipelines/          Single / stereo / N-rig / hand-eye / cam-IMU rotation init / rolling-shutter
online/             OnlineIntrinsicTracker, OnlineExtrinsicTracker — the gated recalibration loop
apply/              Warp-map generation, VPI-LDC coefficient mapping (Jetson), CUDA remap (server)
io/                 OpenCV YAML, ROS CameraInfo, Isaac Perceptor URDF, Kalibr camchain
bindings/           Python (pybind11) + ROS2 node
CLAUDE.md           Architecture + the non-obvious rules that govern the code
```

## Building & testing

CUDA toolkit optional — the build degrades gracefully to a host-only configuration when `nvcc` is absent (CI runs this way). Multi-arch single source for the CUDA half:

```bash
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES="72;80;86;87;89;90"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CI matrix:

- `build-test` (gcc + clang, Ubuntu 24.04, no CUDA) — the green-gate signal.
- `strict` (gcc -Werror, required as of v0.2).
- `host-only config proof` — asserts the multi-arch CMake config compiles cleanly without `nvcc`.
- `opencv` — installs `libopencv-dev` and runs the OpenCV-gated suite (real-image detection, ChArUco / AprilGrid).
- `python` — pybind11 + pytest.

Optional flags:

- `-DCALIBFORGE_PYTHON=ON` — build the Python bindings.
- `-DCALIBFORGE_WERROR=ON` — strict warnings (matches the `strict` CI job).

## Design at a glance

- **Camera-model core** — adopts the nvTorchCam interface design (Apache-2.0); adds double-sphere & EUCM that nvTorchCam / Kornia / PyTorch3D do not ship. Analytic Jacobians on hot paths (FD-validated in tests).
- **Solver** — solver-agnostic `Problem` / `ResidualBlock` interface. CPU `DenseProblem` (manifold LM with FastTriggs robust loss) is the default; GPU back-ends (PyPose / Graphite) are tracked for v1.0 once a CUDA host benchmarks the regimes where they actually pay (CLAUDE.md rule 1).
- **Apply** — VPI LDC export on Jetson (PVA/VIC), CUDA / CV-CUDA on server (CLAUDE.md rule 5).
- **Online / targetless** — `OnlineIntrinsicTracker` and `OnlineExtrinsicTracker` both gate emission on `assessObservability` + `parameterUncertainty` + a 6-axis motion-excitation check. **Never silently emit calibration parameters online** (CLAUDE.md rule 2).
- **One CUDA source → server + Jetson** via the `-gencode` arch matrix.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full cited rationale and [`docs/DESIGN.md`](docs/DESIGN.md) for the vision.

## Documentation

The long-form design / research / dependency / spike docs live in [`docs/`](docs/) and are mirrored on the [GitHub Wiki](https://github.com/danghoangnhan/CalibForge/wiki) for browsability:

- [DESIGN](https://github.com/danghoangnhan/CalibForge/wiki/DESIGN) — vision + engineering spec v0.1 (owner intent).
- [RESEARCH](https://github.com/danghoangnhan/CalibForge/wiki/RESEARCH) — cited, adversarially-verified mid-2026 research + revised build-vs-borrow stack.
- [DEPENDENCIES](https://github.com/danghoangnhan/CalibForge/wiki/DEPENDENCIES) — enforceable dependency policy + in-tree status per dep.
- [SPIKES](https://github.com/danghoangnhan/CalibForge/wiki/SPIKES) — empirical dependency de-risking results + the deferred-to-CUDA-host work.

The repository copy under [`docs/`](docs/) is the canonical source — the wiki is updated from it.

## License

Open-source & free (permissive — Apache-2.0 intended). See the root `LICENSE`. Vendored dependencies must be permissive (Apache / BSD / MIT / MPL); GPL-family projects are reference-only — re-implement the math, never copy. See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
