# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What CalibForge is

A unified, NVIDIA-accelerated **geometric camera-calibration library** that both **estimates** and **applies** calibration across pinhole / fisheye / generic models, built **once** and deployed on both **edge** (Jetson) and **server** GPUs. The full vision is in [`docs/DESIGN.md`](docs/DESIGN.md); the research that validated/revised the design is in [`docs/RESEARCH.md`](docs/RESEARCH.md).

**Repo stage:** v0.5 in progress. The CPU calibration core (pinhole / Brown-Conrady / Kannala-Brandt / double-sphere / EUCM), stereo / N-rig / hand-eye / cam-IMU rotation init + Forster preintegration / rolling-shutter pipelines, observability-gated online intrinsic + extrinsic trackers (the project differentiator), runtime undistort (CPU + CUDA + VPI-LDC export), io interop (OpenCV YAML, ROS CameraInfo, Isaac URDF, Kalibr camchain), Python (pybind11) + ROS2 bindings, and the generic per-pixel B-spline model are all implemented. The **full cam-IMU pipeline** now closes v0.4: `calibrate_cam_imu_full` (`pipelines/`) wires the Forster preintegration factor + a new `CamImuPoseResidual` into a joint estimate of the **spatial translation extrinsic `p_ci`** + gyro/accel biases + gravity + nav-states, with a Schur-marginalized observability gate (RULE #2). The **B-spline model** is now wired into a calibration pipeline (`solve/calibrate_generic_bspline.hpp`), io (`io/generic_bspline_yaml.hpp`), and the Python bindings, and is **validated against wide-FOV double-sphere / Kannala-Brandt fisheye sources** to sub-pixel (closing "only a synthetic pinhole source"). ~127 tests host-only (137 with the CUDA half; 141 with the OpenCV-gated cases too) on a CUDA-less CI matrix (gcc + clang + Werror + OpenCV-gated + python). A **native CUDA dense LM solver back-end** (`SolverBackend::GpuCuda`, cuBLAS SYRK/GEMV + cuSOLVER Cholesky) plus an **FP32 variant** (`SolverBackend::GpuCudaF32`) are validated firsthand on **two** GPU hosts: an RTX 5090 (sm_120 / CUDA 12.0) and a **Grace-Blackwell GB10 (aarch64 + sm_121 / CUDA 13.0)** — the latter an edge-class ARM + Blackwell host where **FP32↔FP64 parity is measured** (the §D.3 deferral, now partly closed) and the single-source multi-arch build runs on sm_121 via `compute_90` PTX JIT. The CPU-vs-GPU crossover is **measured** (`docs/BENCHMARKS.md`, `docs/SPIKES.md` §E/§F). The PyPose / Graphite / MegBA sparse/batched borrows remain pending, as do a Jetson Orin (sm_87) crossover re-measurement + a real-dataset wide-FOV validation (`docs/SPIKES.md` §D). Read the docs below to understand intent + non-obvious rules before extending.

## Read these first (they encode decisions, not just description)

- [`docs/DESIGN.md`](docs/DESIGN.md) — the owner's vision/spec v0.1 (intent).
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — cited, adversarially-verified mid-2026 research; the **revised build-vs-borrow stack** lives here.
- [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) — the enforceable dependency table (what may be vendored).
- [`docs/SPIKES.md`](docs/SPIKES.md) — empirical dependency de-risking results (what actually built/ran on real hardware).

## Architecture (research-validated module layout)

```
core/       Camera models + Lie groups + Jacobians (differentiable, CUDA/C++).  Model-agnostic contract: project/unproject/jacobian.
detect/     Observation extraction: target-based (checkerboard/ChArUco/AprilGrid) + targetless (feature tracks, BEV photometric).
solve/      The numerical heart: a SOLVER-AGNOSTIC Problem / pluggable ResidualBlock interface, with CPU and GPU back-ends.
pipelines/  Workflows (single/stereo/rig/hand-eye/cam-IMU/online) — thin glue over solve/.
apply/      Runtime undistort/rectify: VPI-LDC export (Jetson PVA/VIC) + CUDA/CV-CUDA path (server).
io/         Interop: OpenCV YAML, ROS CameraInfo, Kalibr YAML, COLMAP, and Isaac Perceptor URDF.
bindings/   Python (pybind11) + ROS2 / Isaac.
```

## Non-obvious rules (these came from research — violating them wastes weeks)

1. **GPU is NOT automatically faster for calibration.** A single small calibration (few cameras) is typically faster on a CPU solver (Ceres) than on a batched GPU solver (~25× in Theseus's benchmarks). GPU BA earns its keep only on **batched fleet / large rigs / generic 10⁴-param models / online-continuous** work. Keep `solve/` solver-agnostic; **CPU is the default path for single small calibrations.** No published solver benchmarked the calibration regime — measure before assuming. *(Now measured for CalibForge's own native CUDA dense solver on TWO hosts: RTX 5090 — CPU wins ≤5 views, GPU crosses over at ~8; Grace-Blackwell GB10 — CPU still wins at 20 views (crossover beyond 20). Crossover is hardware-specific; re-measure per target — `docs/BENCHMARKS.md`, `docs/SPIKES.md` §E/§F.)*
2. **Never silently emit calibration parameters online.** Online/targetless recalibration must pass an **observability gate** (unobservable-direction check + degenerate-motion detection + covariance/Fisher-information) before emitting. *Low reported σ does not mean accurate* (precision ≠ accuracy). This gate is the project's differentiator and is genuinely unbuilt elsewhere.
3. **Permissive-only vendoring — and mind the *form*, not just the license.** CalibForge is open-source & free (permissive). Only vendor Apache/BSD/MIT/MPL code. **GPL/LGPL code (OpenVINS, MVIS, Ctrl-VIO, DeepLM) is reference-only — re-implement the math, never copy.** Re-implementation is *also* forced by **form**, even for permissively-licensed deps that can't link into the edge C++ binary: **Kalibr/iKalibr** (BSD/Apache **ROS apps**), **PyPose/Theseus/nvTorchCam/Kornia** (Apache/MIT **PyTorch**), **OpenCalib** (Apache **CLI tools**), **puzzlepaint** (BSD **Qt GUI**). (Note: Kalibr is BSD-3, *not* GPL — a common misconception; it's blocked by form + GPL-transitive SuiteSparse, not by its own license.) The reuse-**as-is** borrows are the C++ building blocks: Sophus/manif, Ceres/GTSAM, Graphite/MegBA (GPU), CV-CUDA, basalt-headers. See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) (the license × form table); add every new dep there with its license + form.
4. **Analytic Jacobians on hot residual paths; autodiff for prototyping.** On GPU, analytic Jacobians match autodiff accuracy at ~30% less time and ~40% less memory (MegBA).
5. **VPI's PVA/VIC engines are Jetson-only.** The server *apply* path must use VPI's CUDA backend or CV-CUDA — do not assume PVA/VIC offload on x86_64.
6. **One source, many arches.** Build with the `-gencode` matrix so a single codebase targets server + Jetson; an edge↔server determinism (parity) test is a first-class requirement.

## Build / target matrix

CUDA multi-arch, single source. CMake configures all targets at once:

```bash
# The default matrix is 72;80;86;87;89;90;90-virtual. On CUDA >= 13 (which removed Volta/Xavier)
# sm_72 is auto-dropped at configure time (RULE #6) — but ONLY when the matrix is left at its
# default; an explicit -DCMAKE_CUDA_ARCHITECTURES that lists 72 bypasses the drop and nvcc 13 fails.
cmake -S . -B build              # default multi-arch matrix (sm_72 auto-dropped on CUDA >= 13)
cmake --build build -j
# To pin specific arches on CUDA 13, pass a matrix WITHOUT 72, e.g. "80;86;87;89;90;90-virtual".
```

| Target | SM arch |
|---|---|
| Jetson Orin (primary edge) | `sm_87` |
| Jetson Xavier (legacy) | `sm_72` |
| Server (A100/H100/L4) | `sm_80/90/89` |
| Desktop dev (RTX) | `sm_86/89` |
| Blackwell (RTX 50xx / Grace-Blackwell GB10 / newer than the toolkit) | `compute_90` PTX via `90-virtual` → JIT (validated on RTX 5090 `sm_120`/CUDA 12.0 **and** GB10 `sm_121`/CUDA 13.0, RULE #6) |

> The CMake build produces the real `calibforge_tests` (~127 test cases host-only; 137 on a CUDA host — the `remap_cpu_gpu_parity` case + 7 native-solver cases + 2 FP32/FP64 parity cases; 141 with the OpenCV-gated cases); the CUDA half (the multi-arch matrix above + `calibforge_arch_probe` + `calibforge_cuda`, which bundles the native dense LM solver `solve/src/cuda_linear_solver.cu` with both FP64 + FP32 paths) is enabled only when `nvcc` is found and the host-only path degrades cleanly when it is not (see `.github/workflows/ci.yml`'s `config-only` job). The GPU dense solver, the FP32↔FP64 parity, and the CPU-vs-GPU benchmark have been run firsthand on an RTX 5090 (`docs/SPIKES.md` §E) **and** a Grace-Blackwell GB10 (aarch64 + sm_121 + CUDA 13.0, `docs/SPIKES.md` §F); the remaining deferred work (PyPose/Graphite/MegBA borrows, a true Jetson Orin sm_87 measurement, bf16 parity, real-dataset wide-FOV validation) is documented in `docs/SPIKES.md` §D.

## Borrow map (what implements each layer — see RESEARCH.md / DEPENDENCIES.md)

`core` ← **nvTorchCam** (+ add double-sphere & EUCM) · `solve` ← **native CUDA dense LM** (cuBLAS/cuSOLVER, built + validated) + **PyPose** (primary, tracked) / **Graphite** (edge, tracked) / **Ceres** (CPU + oracle) · Lie ← **Sophus/manif** + Theseus patterns · `apply` ← **VPI LDC** + CV-CUDA · `io` ← OpenCV/ROS/Kalibr/COLMAP + Isaac URDF · RS/cam-IMU ← **iKalibr** (re-implement) · online ← **OpenCalib SurroundCameraCalib** + the observability gate (build).
