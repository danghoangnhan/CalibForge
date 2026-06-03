# CalibForge — A Unified GPU Geometric Camera Calibration Library

*(working codename — rename to avoid clash with the existing learned "GeoCalib" project)*

> **Provenance:** This is the original design spec v0.1 as authored by the project owner, preserved verbatim. Where mid-2026 deep research revised these choices (notably the GPU bundle-adjustment borrow), see [`RESEARCH.md`](./RESEARCH.md) and [`DEPENDENCIES.md`](./DEPENDENCIES.md). This document is the *intent*; `RESEARCH.md` is the *research-validated revision*.

**One line:** A single, NVIDIA-accelerated library that both **estimates** and **applies** geometric camera calibration across pinhole, fisheye and generic models — built once, deployed on the **edge** (Jetson on robots, UAVs, aerial platforms) and on **servers**.

**Status:** Design draft v0.1 · Target: research preview → v1.0 · Owner: Daniel Tu

> **Implementation status (v0.5 in progress):** §12 roadmap rows v0.1 through v0.5 are implemented and tested on a CUDA-less CI matrix (gcc + clang + Werror + OpenCV-gated + python). The v1.0 row — generic per-pixel model, GPU solver back-ends, hardened edge↔server packaging + benchmark suite — is partly stubbed (CPU-side determinism test + capability scaffolding) and otherwise deferred pending a CUDA / Jetson host. See the repo root `README.md` for the current capability matrix and `docs/SPIKES.md` §D for the deferred items.

---

## PART 1 — PITCH / VISION (read this first)

### The problem
Geometric camera calibration is the unglamorous foundation under every perception stack — SLAM, photogrammetry, ADAS, visual servoing, 3D reconstruction. Get it wrong and everything downstream drifts. Yet in 2026 the tooling is **fragmented**: you estimate parameters with one CPU tool, apply them with another, and re-implement everything again when you move from a server to an embedded board.

### The gap (this is the whole reason to build it)
Every building block exists in public, but **no single library does all of geometric calibration, GPU-accelerated, deployable on both edge and server.** Concretely:

| What you need | Best public option today | Its limitation |
|---|---|---|
| Parameter **estimation** | OpenCV `calib3d` | CPU-bound; CUDA only for remap |
| Accurate generic models | puzzlepaint/`camera_calibration` (Schöps et al.) | Research-grade, not productized for edge |
| Multi-cam + cam-IMU | Kalibr | CPU; research workflow |
| Differentiable camera math | nvTorchCam (NVIDIA) | Math layer, not a calibration toolkit |
| Fast **apply**/undistort on edge | NVIDIA VPI (LDC) | No estimation — you feed it coefficients |
| GPU image primitives | CV-CUDA (NVIDIA) | Primitives, not a solver |
| GPU bundle adjustment | MegBA, DeepLM | Standalone BA, not wired into a calibration pipeline |

The integration — estimate **and** apply, all GPU, one codebase for Jetson **and** server — **does not exist yet.** The idea is "partly public": the pieces are open, the unified product is not.

### The thesis
Build a **camera-model-agnostic, fully differentiable calibration core in CUDA** with a **GPU bundle-adjustment solver** at its center, that:
1. **Estimates** intrinsics, distortion, extrinsics, stereo/rig, hand-eye, and supports **online/targetless** recalibration.
2. **Emits** runtime artifacts (VPI-compatible remap tables, rectification maps) so the *apply* path is free on the edge.
3. **Compiles once** for x86 server GPUs and Jetson (`sm_*` arch flags) — no rewrite when you deploy.

### Why now
- **Online/targetless calibration matured in 2024–2025** for both surround-view (automotive) and in-flight UAV self-calibration — the research is ready to be productized.
- **GPU BA is solved** as a primitive (MegBA reports up to ~41× over Ceres) — the hard kernel exists to build on.
- **NVIDIA's edge stack converged** (VPI + CV-CUDA + Isaac ROS) but lacks a calibration-estimation layer to sit on top.

### Who it's for (v1 priority: edge)
- **Robotics** — Jetson-based robots: stereo + hand-eye, tight integration with the Isaac ecosystem.
- **UAV / aerial** — gimbal + downward mapping cameras: rolling-shutter handling, in-flight self-calibration, photogrammetry-grade accuracy, GPS/IMU fusion.
- *(Later)* **Automotive surround-view** — 4–6 fisheye 360° rigs, online targetless extrinsic recalibration under vibration.
- *(Later)* **Server/cloud** — batch calibration of large camera fleets.

### The bet in one sentence
*Whoever owns the unified, GPU, edge-to-server calibration layer owns a dependency that sits under every NVIDIA-based perception product.*

---

## PART 2 — ENGINEERING SPECIFICATION

### 1. Scope — capability matrix

| Capability | v0.x (MVP) | v1.0 | Notes |
|---|:---:|:---:|---|
| Pinhole intrinsics (focal, principal pt) | ✅ | ✅ | |
| Brown–Conrady (radial+tangential) distortion | ✅ | ✅ | matches VPI polynomial model |
| Kannala–Brandt fisheye / equidistant | ✅ | ✅ | core for UAV + surround-view |
| Division / generic per-pixel models | — | ✅ | borrow approach from Schöps generic model |
| Single-camera extrinsics (PnP) | ✅ | ✅ | |
| Stereo calibration + rectification | ✅ | ✅ | |
| Multi-camera rig (N cams, shared optimization) | — | ✅ | surround-view / aerial arrays |
| Hand-eye calibration | — | ✅ | robotics |
| Camera–IMU extrinsics (+ time offset) | — | ✅ | UAV / robotics, Kalibr-style |
| Rolling-shutter calibration (readout time) | — | ✅ | **critical for UAV CMOS sensors** |
| Online / targetless recalibration | prototype | ✅ | vibration drift on car/UAV |
| Bundle-adjustment back-end (GPU) | ✅ | ✅ | the make-or-break module |
| Runtime undistort/rectify (VPI export) | ✅ | ✅ | edge apply path |

Out of scope (v1): photometric/radiometric calibration, color calibration, structured-light depth calibration. Keep it strictly **geometric**.

### 2. Camera model layer (the foundation)
A single differentiable abstraction with `project()` / `unproject()` / `jacobian()` for every model. This is the contract everything else builds on. **Recommended base:** adopt the design (and ideally code) of **nvTorchCam** — it already provides camera-agnostic, differentiable, GPU-batched projection across model families, which is exactly this layer.

Models implement a common interface so the solver, the detector, and the apply-path are all model-independent. Adding a new lens model should mean writing one class, not touching the pipeline.

### 3. Architecture (layers)

```
            ┌─────────────────────────────────────────────┐
            │  Bindings: Python (pybind11) + ROS2 / Isaac   │
            ├─────────────────────────────────────────────┤
   ESTIMATE │  Pipelines: single · stereo · rig · hand-eye  │
            │             · cam-IMU · online/targetless     │
            ├──────────────────┬──────────────────────────┤
            │ Detection layer  │  GPU Bundle Adjustment    │  ← core IP
            │ (targets +       │  (Schur / PCG, LM/GN)     │
            │  targetless feat)│                           │
            ├──────────────────┴──────────────────────────┤
   SHARED   │  Camera-model core (differentiable, CUDA)     │  ← nvTorchCam-style
            ├─────────────────────────────────────────────┤
   APPLY    │  Runtime: undistort / rectify / remap export  │
            │  → VPI LDC tables · CV-CUDA ops               │
            ├─────────────────────────────────────────────┤
            │  CUDA core — one build → server + Jetson      │
            └─────────────────────────────────────────────┘
```

### 4. Module breakdown

**4.1 `core` — camera models & math (CUDA/C++).** Differentiable projection, distortion, SE(3)/SO(3) Lie-group ops, autodiff or analytic Jacobians. Header-only math where possible for portability across arch.

**4.2 `detect` — observation extraction.**
- *Target-based:* checkerboard, ChArUco/AprilGrid corner detection (GPU-accelerated), sub-pixel refinement.
- *Targetless:* feature tracks for online calibration; for surround-view, photometric error in overlapping bird's-eye regions; for UAV, tie-points from the image block.

**4.3 `solve` — GPU bundle adjustment (see §5).** The numerical heart.

**4.4 `pipelines` — calibration workflows.** Orchestrate detect → solve → validate for each scenario (single/stereo/rig/hand-eye/cam-IMU/online). Each pipeline is thin glue over `solve`.

**4.5 `apply` — runtime correction.** Generate rectification + undistortion maps; export to **VPI LDC** format so the edge runtime undistorts on PVA/VIC engines (keeps the GPU free); CV-CUDA fallback for warp/remap on platforms without those engines.

**4.6 `io` — interop.** Read/write OpenCV YAML, ROS `CameraInfo`, Kalibr YAML, COLMAP; export TensorRT-friendly artifacts for learned sub-components.

### 5. The GPU bundle-adjustment solver (make-or-break)
Calibration = nonlinear least squares minimizing reprojection error; solved iteratively with **Gauss-Newton / Levenberg-Marquardt**. On GPU the standard recipe is **Schur complement elimination** of 3D-point blocks followed by a **(preconditioned) conjugate-gradient** solve of the reduced camera system.

**Build vs. borrow:** Do **not** write this from scratch first. **MegBA** is open-source, implements exactly this (distributed Schur + PCG, GN/LM/Dog-Leg) and reports large speedups over Ceres/RootBA/DeepLM. Strategy:
1. **v0.x:** wrap MegBA (or DeepLM for a PyTorch-native path) as the back-end behind our solver interface.
2. **v1.0:** if MegBA's problem structure doesn't fit calibration residuals cleanly (it's tuned for SfM-scale BA), fork/specialize the kernels for calibration-sized problems and add our distortion/rolling-shutter residual blocks.

Solver interface must accept pluggable **residual blocks** (reprojection, IMU preintegration, rolling-shutter line-time, planar/ground constraints) so every pipeline reuses one optimizer.

> **⚠️ Research revision (mid-2026):** the build-vs-borrow conclusion here changed. DeepLM is GPLv3 (incompatible with a permissive library); MegBA is server-only (no Jetson). The research recommends **PyPose** (Apache-2.0, Jetson-capable) as the primary borrow, **Graphite** (MIT, Jetson-proven, bf16) as a tracked edge option, and **Ceres** as the CPU path — *because small single calibrations are faster on CPU than on batched GPU solvers.* See [`RESEARCH.md` Theme 1](./RESEARCH.md#theme-1).

### 6. Edge deployment & build targets
**Single CUDA codebase, multi-arch compile.** Same source compiled with multiple `-gencode arch=compute_XX` flags so one set of binaries covers server GPUs and Jetson.

| Platform | GPU arch (SM) | Notes |
|---|---|---|
| Jetson Orin (robot/UAV) | `sm_87` | primary edge target; has PVA + VIC for VPI LDC |
| Jetson Xavier | `sm_72` | legacy edge |
| Server (A100/H100/L4) | `sm_80 / 90 / 89` | batch + dev |
| Desktop dev (RTX) | `sm_86 / 89` | local iteration |

**Edge constraints to design around:** limited VRAM (problem partitioning matters — MegBA's partitioning helps), power budget, and offloading the *apply* path to PVA/VIC via VPI so the GPU stays available for perception. Use TensorRT for any learned detector/initializer.

> **⚠️ Research revision (mid-2026):** VPI's PVA/VIC engines are **Jetson-only** — on x86_64 servers VPI runs only its CUDA backend, so the *apply* path needs a CUDA/CV-CUDA fallback on server. See [`RESEARCH.md` Theme 5](./RESEARCH.md#theme-5).

### 7. Platform-specific calibration notes

**Robotics (Jetson).** Priorities: stereo + hand-eye, tight ROS2/Isaac integration (publish `CameraInfo`, consume from Argus). Closest to NVIDIA Isaac ROS / Perceptor sensor-calibration conventions — align formats with those.

**UAV / aerial (v1 priority).**
- **Rolling shutter is mandatory** — consumer/most pro UAV CMOS sensors expose line-by-line, adding distortion at flight speed. Model readout time as a calibrated parameter and add a rolling-shutter residual block.
- **In-flight self-calibration** — offline params drift due to platform vibration/thermal; add camera systematics as extra unknowns in block bundle adjustment. Determinability depends on flight geometry + enough tie points/GCPs — surface this as a confidence/observability report.
- **Sensor fusion** — GPS/IMU as priors; cam-IMU time-offset estimation.
- Target **photogrammetry-grade** reprojection accuracy, not just "looks undistorted."

**Automotive surround-view (later).** 4–6 fisheye cameras, 360°. Extrinsics drift from engine vibration, doors, bumps. Implement **online targetless** extrinsic calibration via photometric consistency in BEV overlap regions; coarse-to-fine search to tolerate large initial error. Note ISO 26262 / functional-safety implications if it ever feeds control.

### 8. Dependency matrix

| Dependency | Role | Edge-safe? | License |
|---|---|---|---|
| CUDA Toolkit | core compute | ✅ (JetPack) | NVIDIA |
| nvTorchCam | camera-model layer (adopt) | ✅ | open (verify) |
| MegBA / DeepLM | GPU BA back-end (wrap) | ✅ (VRAM permitting) | open (verify) |
| NVIDIA VPI | runtime undistort (LDC) | ✅ (Jetson PVA/VIC) | NVIDIA SDK |
| CV-CUDA | GPU image ops | ✅ (cloud+edge) | Apache-2.0 |
| cuSolver / cuSPARSE | linear algebra | ✅ | NVIDIA |
| OpenCV (calib3d) | reference/validation only | ✅ | Apache-2.0 |
| Ceres | CPU reference oracle | dev-only | BSD |
| Eigen | host-side math | ✅ | MPL2 |
| pybind11 | Python bindings | n/a | BSD |
| TensorRT | learned components | ✅ | NVIDIA |

*(Confirm each license before vendoring — see [`DEPENDENCIES.md`](./DEPENDENCIES.md) for the mid-2026 re-validated table.)*

### 9. Public API sketch

```python
import calibforge as cf

cam = cf.CameraModel.fisheye_kb()                 # Kannala–Brandt
obs = cf.detect.charuco(images, board)            # GPU corner detection
result = cf.calibrate.single(cam, obs,
                             solver="gpu_lm",
                             device="cuda")
print(result.rms_reproj_px, result.params)

# emit edge runtime artifact (consumed by VPI on Jetson)
result.export_vpi_ldc("front_cam.ldc")

# online / targetless recalibration loop (UAV / car)
tracker = cf.online.ExtrinsicTracker(rig, mode="targetless")
tracker.update(frame_batch)        # corrects drift continuously
```

C++ core mirrors this; ROS2 node wraps `online.ExtrinsicTracker`.

### 10. Accuracy & validation strategy
- **Reference oracle:** every estimator cross-checked against OpenCV/Ceres on the same data; GPU result must match CPU within tolerance.
- **Synthetic ground truth:** render scenes (Isaac Sim) with known intrinsics/extrinsics → exact error measurement, incl. rolling-shutter.
- **Public benchmarks:** surround-view calibration benchmark/dataset (arXiv 2312.16499); standard BA datasets for the solver.
- **Metrics:** RMS reprojection error (px), parameter error vs. GT, and an **observability/confidence** score (especially for online + UAV self-cal).
- **Edge parity test:** identical inputs on server vs. Jetson must produce identical params (determinism check).

### 11. Build-vs-borrow summary (don't reinvent)
- **Borrow:** nvTorchCam (model layer), MegBA/DeepLM (BA kernels), VPI (apply), CV-CUDA (image ops), OpenCV/Ceres (validation only).
- **Build (your IP):** the unified pipeline orchestration, the **calibration-specific residual blocks** (distortion, rolling-shutter, cam-IMU), **targetless/online** logic, the **VPI export bridge**, and the **single-build edge↔server packaging**.
- **License audit (do before coding):** confirm nvTorchCam and MegBA licenses permit your distribution model; VPI/TensorRT ship under NVIDIA SDK terms (fine on Jetson, check redistribution).

> **⚠️ Research revision (mid-2026):** the license audit is done — see [`DEPENDENCIES.md`](./DEPENDENCIES.md). DeepLM is dropped (GPLv3); the capable cam-IMU/RS tools (Kalibr, OpenVINS, MVIS, Ctrl-VIO) are GPL → reference-only; iKalibr (BSD-3) and OpenCalib SurroundCameraCalib (Apache-2.0) are the permissive borrows for those layers.

### 12. Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| **v0.1** | Camera-model core + GPU LM (wrapped MegBA) + pinhole/Brown-Conrady single-cam, validated vs OpenCV | ✅ shipped (CPU manifold LM; GPU back-end deferred to v1.0) |
| **v0.2** | Fisheye (KB) + stereo + VPI LDC export; runs on Orin | ✅ shipped (KB + DS + EUCM + stereo + warp-map; Orin-on-hardware deferred) |
| **v0.3** | Multi-cam rig + hand-eye; ROS2 node | ✅ shipped (N-rig + hand-eye + ROS2 node + pybind11) |
| **v0.4** | Rolling-shutter + cam-IMU (UAV focus); Isaac Sim synthetic validation | ✅ shipped (RS readout + cam-IMU rotation init + full Forster preintegration factor; Isaac Sim deferred to hardware) |
| **v0.5** | Online/targetless recalibration prototype (UAV + surround-view) | ✅ shipped (`OnlineExtrinsicTracker` + `OnlineIntrinsicTracker` behind the observability + 6-axis motion-excitation gate; `FeatureTracker` targetless source; BEV photometric + UAV/surround-rig orchestrators pending) |
| **v1.0** | Generic per-pixel model, hardened edge build, full benchmark suite, docs | ⏳ partial — CPU-runnable determinism test + accurate README/CLAUDE.md shipped; generic B-spline model + GPU solver back-ends + CPU-vs-GPU regime benchmark + Jetson↔server FP32/bf16↔FP64 parity test all deferred pending a CUDA / Jetson host |

### 13. Risks
- **BA back-end fit:** MegBA is tuned for SfM-scale problems; calibration problems are smaller/differently structured — may need kernel specialization (mitigate: keep solver interface abstract).
- **Edge VRAM:** large rigs + online tracking on Orin — lean on partitioning and PVA/VIC offload.
- **Online determinability:** UAV/car self-cal can be ill-conditioned — ship an observability score, never silently emit bad params.
- **License/redistribution** of vendored GPU libs — audit first (§11).
- **CUDA arch drift:** maintain the `-gencode` matrix as new Jetson/server GPUs ship.

---

## References
- puzzlepaint/camera_calibration (Schöps et al.): https://github.com/puzzlepaint/camera_calibration
- nvTorchCam (arXiv 2410.12074): https://arxiv.org/pdf/2410.12074
- MegBA — GPU distributed BA (arXiv 2112.01349): https://github.com/MegviiRobot/MegBA
- CV-CUDA: https://github.com/CVCUDA/CV-CUDA
- NVIDIA VPI — Lens Distortion Correction: https://docs.nvidia.com/vpi/1.1/algo_ldc.html
- CuSfM (arXiv 2510.15271): https://arxiv.org/pdf/2510.15271
- Two-step rolling-shutter correction in UAV photogrammetry: https://www.sciencedirect.com/science/article/abs/pii/S0924271619302849
- In-flight geometric calibration for mapping cameras (2025): https://www.sciencedirect.com/science/article/abs/pii/S0263224125004075
- Automatic surround-camera calibration for self-driving (arXiv 2305.16840): https://arxiv.org/pdf/2305.16840
- Surround-view calibration benchmark & dataset (arXiv 2312.16499): https://arxiv.org/abs/2312.16499
- Click-Calib (arXiv 2501.01557): https://arxiv.org/html/2501.01557v1
- OpenCV Camera Calibration & 3D Reconstruction: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- NVIDIA Isaac Sim sensor calibration: https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html
