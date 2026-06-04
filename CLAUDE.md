# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What CalibForge is

A unified, NVIDIA-accelerated **geometric camera-calibration library** that both **estimates** and **applies** calibration across pinhole / fisheye / generic models, built **once** and deployed on both **edge** (Jetson) and **server** GPUs. The full vision is in [`docs/DESIGN.md`](docs/DESIGN.md); the research that validated/revised the design is in [`docs/RESEARCH.md`](docs/RESEARCH.md).

**Repo stage:** v0.5 in progress. The CPU calibration core (pinhole / Brown-Conrady / Kannala-Brandt / double-sphere / EUCM), stereo / N-rig / hand-eye / cam-IMU rotation init + Forster preintegration / rolling-shutter pipelines, observability-gated online intrinsic + extrinsic trackers (the project differentiator), runtime undistort (CPU + CUDA + VPI-LDC export), io interop (OpenCV YAML, ROS CameraInfo, Isaac URDF, Kalibr camchain), Python (pybind11) + ROS2 bindings, the generic per-pixel B-spline model (CPU, header-only), and ~113 tests on a CUDA-less CI matrix (gcc + clang + Werror + OpenCV-gated + python) are all implemented. The GPU solver back-ends (PyPose / Graphite / MegBA) and the edge↔server numerical-parity test remain v1.0 deliverables pending a CUDA / Jetson host; the B-spline model's pipeline wiring + wide-FOV validation are likewise pending (see `docs/SPIKES.md` §D). Read the docs below to understand intent + non-obvious rules before extending.

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

1. **GPU is NOT automatically faster for calibration.** A single small calibration (few cameras) is typically faster on a CPU solver (Ceres) than on a batched GPU solver (~25× in Theseus's benchmarks). GPU BA earns its keep only on **batched fleet / large rigs / generic 10⁴-param models / online-continuous** work. Keep `solve/` solver-agnostic; **CPU is the default path for single small calibrations.** No published solver benchmarked the calibration regime — measure before assuming.
2. **Never silently emit calibration parameters online.** Online/targetless recalibration must pass an **observability gate** (unobservable-direction check + degenerate-motion detection + covariance/Fisher-information) before emitting. *Low reported σ does not mean accurate* (precision ≠ accuracy). This gate is the project's differentiator and is genuinely unbuilt elsewhere.
3. **Permissive-only vendoring.** CalibForge is open-source & free (permissive). Only vendor Apache/BSD/MIT/MPL code. **GPL/LGPL code (Kalibr, OpenVINS, MVIS, Ctrl-VIO, DeepLM) is reference-only — re-implement the math, never copy.** See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md); add every new dep there with its license + status.
4. **Analytic Jacobians on hot residual paths; autodiff for prototyping.** On GPU, analytic Jacobians match autodiff accuracy at ~30% less time and ~40% less memory (MegBA).
5. **VPI's PVA/VIC engines are Jetson-only.** The server *apply* path must use VPI's CUDA backend or CV-CUDA — do not assume PVA/VIC offload on x86_64.
6. **One source, many arches.** Build with the `-gencode` matrix so a single codebase targets server + Jetson; an edge↔server determinism (parity) test is a first-class requirement.

## Build / target matrix

CUDA multi-arch, single source. CMake configures all targets at once:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="72;80;86;87;89;90"
cmake --build build -j
```

| Target | SM arch |
|---|---|
| Jetson Orin (primary edge) | `sm_87` |
| Jetson Xavier (legacy) | `sm_72` |
| Server (A100/H100/L4) | `sm_80/90/89` |
| Desktop dev (RTX) | `sm_86/89` |

> The CMake build produces the real `calibforge_tests` (~113 test cases host-only; 114 with the CUDA `remap_cpu_gpu_parity` case on a CUDA host); the CUDA half (the multi-arch matrix above + `calibforge_arch_probe` + `calibforge_cuda`) is enabled only when `nvcc` is found and the host-only path degrades cleanly when it is not (see `.github/workflows/ci.yml`'s `config-only` job). The deferred work pending a CUDA / Jetson host is documented in `docs/SPIKES.md` §D.

## Borrow map (what implements each layer — see RESEARCH.md / DEPENDENCIES.md)

`core` ← **nvTorchCam** (+ add double-sphere & EUCM) · `solve` ← **PyPose** (primary) / **Graphite** (edge, tracked) / **Ceres** (CPU + oracle) · Lie ← **Sophus/manif** + Theseus patterns · `apply` ← **VPI LDC** + CV-CUDA · `io` ← OpenCV/ROS/Kalibr/COLMAP + Isaac URDF · RS/cam-IMU ← **iKalibr** (re-implement) · online ← **OpenCalib SurroundCameraCalib** + the observability gate (build).
