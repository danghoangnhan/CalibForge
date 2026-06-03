# CalibForge — Dependency De-risking Spikes

Empirical checks of the [`RESEARCH.md`](./RESEARCH.md) borrow decisions, run on the dev host on **2026-06-01**. Goal: don't trust the papers alone — verify the load-bearing claims firsthand, and honestly record what could **not** be verified here.

> **Implementation status (v0.5 in progress).** Everything verified in §A–C below is reflected in the shipped code. §D items remain deferred — the **CPU subset of §D.3 (run-to-run bit-identical CPU determinism)** ships as `tests/test_determinism.cpp`, but the Jetson-vs-server + FP32/bf16 vs FP64 parity it scaffolds, the GPU-vs-CPU calibration-regime benchmark (§D.1), and the Graphite build + VPI LDC Jetson-hardware checks (§D.2/§D.4) all wait on a CUDA / Jetson host. See repo root `README.md` for the current capability matrix.

## Environment (this host)

| Tool | Status |
|---|---|
| OS | Linux WSL2 (x86_64) |
| cmake | 3.28.3 ✅ |
| git | 2.43.0 ✅ |
| python3 / pip | 3.12.3 / 24.0 ✅ |
| g++/gcc | 13.3.0 ✅ |
| **GPU / `nvidia-smi`** | ❌ **none** |
| **CUDA / `nvcc`** | ❌ **none** |
| PyTorch | not preinstalled (installed into a venv for the CPU smokes) |

**Consequence:** all **CUDA/GPU** spikes (Graphite build, any GPU timing, bf16) are **DEFERRED to a CUDA host** — they are not faked. CPU-feasible spikes (source inspection + CPU runtime smokes) are run here.

## A. Source-inspection spikes (✅ done — claims re-verified firsthand)

Shallow-cloned `NVlabs/nvTorchCam`, `pypose/pypose`, `sfu-rsl/graphite` and inspected:

| Claim (from RESEARCH.md) | Verified here? | Evidence |
|---|---|---|
| **nvTorchCam is Apache-2.0** | ✅ | `LICENSE` = "Apache License" |
| **nvTorchCam ships 8 models** | ✅ | `cameras.py` classes: Pinhole, Orthographic, Equirectangular, OpenCVFisheye (KB), OpenCV (Brown-Conrady), BackwardForwardPolynomialFisheye, Kitti360Fisheye, Cube |
| **nvTorchCam lacks DoubleSphere & EUCM** | ✅ | no `DoubleSphere`/`EUCM`/`Unified`/UCM class in `cameras.py` — **the gap is real** |
| **nvTorchCam uses Newton unprojection** | ✅ | `num_undistort_iters: int = 100` on the OpenCV/KB/KITTI fisheye models |
| **PyPose is Apache-2.0** | ✅ | `LICENSE` = "Apache License" |
| **PyPose has FastTriggs robust corrector** | ✅ | `pypose/optim/corrector.py` → `FastTriggs`; default in `optimizer.py` when no corrector given |
| **PyPose has LM (sparse)** | ✅ | `pypose.optim.LevenbergMarquardt` / `LM`; `sparse=True` path documented |
| **Graphite is MIT** | ✅ | `LICENSE.md` = "MIT License" |
| **Graphite requires CUDA** | ✅ | `CMakeLists.txt`: `project(Graphite LANGUAGES CUDA CXX VERSION 0.5.0)` → cannot build without `nvcc` (hence deferred here) |
| **Graphite v0.5.0, ICRA 2026** | ✅ | version 0.5.0 in CMake; README: "[May 2026] Graphite was accepted to IEEE ICRA 2026" |
| Graphite default arch matrix | ℹ️ | `CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 120` (note: includes Blackwell sm_120; **omits Xavier sm_72** — relevant if we target Xavier) |

## B. Scaffold verification (✅ done)

| Check | Result |
|---|---|
| `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="87;89"` | ✅ exit 0 — configures host-only, emits the intended "no CUDA toolkit → GPU targets skipped" warning (graceful degradation works) |
| Interface headers compile as C++17 | ✅ `g++ -std=c++17 -fsyntax-only` over all 5 headers (camera_model, lie, residual_block, problem, observability) → exit 0 |
| Multi-arch matrix logic | ✅ validated syntactically (full GPU compile requires a CUDA host) |

## C. CPU runtime smokes (✅ done — all pass)

Installed into `.spikes/.venv` (CPU wheels): **torch 2.12.0+cpu**, **torchvision 0.27.0+cpu**, **PyPose 0.9.5** (exactly the version RESEARCH.md cited), nvtorchcam (from the clone). Scripts: [`tools/spikes/spike_nvtorchcam.py`](../tools/spikes/spike_nvtorchcam.py), [`tools/spikes/spike_pypose_lm.py`](../tools/spikes/spike_pypose_lm.py).

| Smoke | Result |
|---|---|
| **nvTorchCam — pinhole** `project_to_pixel → pixel_to_ray` round-trip | ✅ worst \|1−cos\| = **6e-8** (exact) |
| **nvTorchCam — KB fisheye** round-trip (exercises the differentiable Newton inverse on CPU) | ✅ worst \|1−cos\| = **6e-8** (exact) |
| **nvTorchCam — DS/EUCM gap** (runtime class list) | ✅ confirmed absent; models = Pinhole, OpenCV(BrownConrady), OpenCVFisheye(KB), BackwardForwardPolynomialFisheye, Kitti360Fisheye, Cube, Equirectangular, Orthographic |
| **PyPose — LM convergence** (PoseInv) | ✅ error 6.1e-2 → **1.86e-9** in 2 iterations |
| **PyPose — FastTriggs corrector** | ✅ with a robust kernel (`Huber`) the default corrector resolves to **FastTriggs**; with no kernel it is `Trivial` |

**Correction surfaced by the spike (kept for the record):** FastTriggs is PyPose's default corrector **only when a robust kernel is supplied** (`optimizer.py`: `self.corrector = [FastTriggs(k) for k in kernel] if corrector is None`). With no kernel, the corrector is `Trivial` (plain least squares). The first version of the smoke wrongly asserted FastTriggs unconditionally — the firsthand run corrected that. The research claim ("PyPose offers FastTriggs as its robust-loss path") stands; the nuance is that it engages with a kernel. Available kernels: `Huber`, `PseudoHuber`, `Cauchy`, `SoftLOne`, `Arctan`, `Tolerant`, `Scale`.

**Takeaway:** both permissive borrows install and run correctly on a plain CPU box — nvTorchCam's model layer (incl. the Newton fisheye inverse) is numerically sound, and PyPose's LM + FastTriggs path is real and converges. Neither result speaks to GPU/edge performance (see §D).

## D. Deferred to a CUDA host (NOT run here — no GPU/`nvcc`)

These are the spikes that actually matter most for the architecture and must be run on a Jetson + a server GPU:

1. **PyPose vs Ceres on the calibration-sized regime** — the single most important number for the solver design (does "CPU wins small" hold on our hardware?). Needs Ceres + a representative few-camera/many-residual problem.
2. **Graphite build + BAL example at FP32 and bf16** — build friction + memory; it's a WIP prototype, so this is the highest-risk borrow to measure.
3. **Edge↔server determinism** — identical inputs on Jetson vs server GPU must yield identical params (and FP32/bf16 vs FP64 parity).
4. **VPI LDC** round-trip on a Jetson (PVA/VIC) vs CUDA backend on server.
