# CalibForge — Dependency De-risking Spikes

Empirical checks of the [`RESEARCH.md`](./RESEARCH.md) borrow decisions, run on the dev host on **2026-06-01**. Goal: don't trust the papers alone — verify the load-bearing claims firsthand, and honestly record what could **not** be verified here.

> **Implementation status (v0.5 in progress).** Everything verified in §A–C below is reflected in the shipped code. **§D.1 (GPU-vs-CPU on the calibration-sized regime) has now been run firsthand on a server CUDA host (RTX 5090 / CUDA 12.0) — see [§E](#e-cuda-host-spikes--run-firsthand-on-an-rtx-5090--cuda-120).** The remaining §D items are still deferred: the **CPU subset of §D.3 (run-to-run bit-identical CPU determinism)** ships as `tests/test_determinism.cpp`, but the *Jetson*-vs-server + FP32/bf16 vs FP64 parity it scaffolds (§D.3), the Graphite build (§D.2), and the VPI-LDC Jetson PVA/VIC check (§D.4) all still wait on a Jetson / second GPU. See repo root `README.md` for the current capability matrix.

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

1. **GPU vs CPU on the calibration-sized regime** — the single most important number for the solver design (does "CPU wins small" hold on our hardware?). ✅ **RUN FIRSTHAND on an RTX 5090 / CUDA 12.0 — see [§E](#e-cuda-host-spikes--run-firsthand-on-an-rtx-5090--cuda-120). RULE #1 holds: CPU wins the small single-calibration regime; the GPU dense back-end crosses over only as the problem grows.** (Measured against a native CalibForge cuBLAS/cuSOLVER dense LM back-end rather than PyPose — PyPose/Graphite/MegBA borrows remain pending, §D.2.)
2. **Graphite build + BAL example at FP32 and bf16** — build friction + memory; it's a WIP prototype, so this is the highest-risk borrow to measure.
3. **Edge↔server determinism** — identical inputs on Jetson vs server GPU must yield identical params (and FP32/bf16 vs FP64 parity).
4. **VPI LDC** round-trip on a Jetson (PVA/VIC) vs CUDA backend on server.

## E. CUDA-host spikes (✅ run firsthand on an RTX 5090 / CUDA 12.0)

Run on **2026-06-04** on a second host that *does* have a GPU + `nvcc` — closing the §D.1 question and exercising the single-source multi-arch build (RULE #6) on real silicon. This complements §A–C (which honestly recorded what the GPU-less dev host could not run); it does **not** rewrite that record.

### Environment (this host)

| Tool | Status |
|---|---|
| OS | Linux x86_64 |
| **GPU / `nvidia-smi`** | ✅ **NVIDIA GeForce RTX 5090** — compute capability **`sm_120` (Blackwell)**, driver 580.126.09 |
| **CUDA / `nvcc`** | ✅ **release 12.0, V12.0.140** |

> **The toolkit (CUDA 12.0) predates the GPU (sm_120).** CUDA 12.0's newest *native* arch is `sm_90`, so there is no `sm_120` SASS to emit. This is exactly the case the `90-virtual` entry in the arch matrix exists for — and the spike below confirms the forward-compat strategy works on real Blackwell hardware, not just in theory.

### E.1 Single-source multi-arch build + forward-compat JIT (RULE #6) — ✅

| Check | Result |
|---|---|
| `cmake -DCMAKE_CUDA_ARCHITECTURES="72;80;86;87;89;90;90-virtual"` + `cmake --build` | ✅ full matrix compiles end-to-end (~26 s, 6 SASS targets + PTX). The earlier "multi-arch validated *syntactically*" caveat (§B) is now validated by a real device compile. |
| Fatbin contents (`cuobjdump`) | ✅ carries `sm_72 / sm_80 / sm_86 / sm_87 / sm_89 / sm_90` SASS **plus `compute_90` PTX** |
| Runs on `sm_120` via PTX JIT | ✅ the GPU code executes correctly on the RTX 5090 even though the toolkit emits **no** `sm_120` SASS — the driver JIT-compiles the embedded `compute_90` PTX forward onto Blackwell. **This is the load-bearing proof that `90-virtual` delivers forward-compat without driver SASS-compat.** |
| Host-only degradation still clean | ✅ forcing `-DCMAKE_CUDA_COMPILER=NOTFOUND` configures `CUDA=OFF`, builds, and runs **114 tests** (the 8 CUDA-only cases compile out) — graceful degradation intact after the GPU work. |

### E.2 Native CUDA dense LM solver (#25) correctness — ✅

A from-scratch CalibForge GPU back-end (`solve/src/cuda_linear_solver.cu`) solves one damped-normal-equations LM step on the device — `J^T J` via cuBLAS DSYRK, `J^T r` via DGEMV, the SPD factor/solve via cuSOLVER dense Cholesky (`potrf`/`potrs`) — with analytic Jacobians still assembled on the host (RULE #4). All running on `sm_120` through the JIT'd `compute_90` PTX.

| Check (`tests/test_cuda_linear_solver.cpp`) | Result |
|---|---|
| Full suite | ✅ **122 / 122 pass** (114 host-only + 8 CUDA-only, incl. the `remap_cpu_gpu_parity` case) |
| One GPU LM step vs host Eigen LDLT | ✅ relative error **< 1e-8** (FP64 round-off; cross-implementation, not bit-exact) |
| Full `GpuCuda`-backend calibration vs `CpuCeres` | ✅ recovers identical intrinsics to **< 1e-5** |
| Non-SPD damped matrix reported (so the LM loop damps harder) | ✅ `potrf` non-PD ⇒ `cudaSolveLmStep` returns false; the next, more-damped step succeeds (deterministic exact-zero-pivot test) |
| Edge cases | ✅ `n=1` (1×1 normal matrix), ill-conditioned-but-SPD `J` (cond ≈ 1e6), and a held-constant parameter block (GPU tangent excludes its columns) all match the host path |

### E.3 §D.1 — GPU vs CPU on the calibration-sized regime (the single most important number) — ✅ RULE #1 holds

Identical `DenseProblem` (a single pinhole camera, a 9×9 board, `n_views` poses) solved with each backend; `cpu_ms`/`gpu_ms` are medians (15 reps ≤ 10 views, 5 reps above). Both backends converge to the same noise-free minimum on every row (the harness verifies this — the speedups are apples-to-apples). Key rows below; the **full sweep + reproduction command live in [`docs/BENCHMARKS.md`](./BENCHMARKS.md)** (`tools/benchmark/calibforge_bench.cpp`, RTX 5090):

| n_views | n_tangent | cpu_ms | gpu_ms | gpu_speedup | winner |
|---:|---:|---:|---:|---:|:--|
| 1 | 10 | 0.28 | 1.97 | **0.14×** | CPU (~7× faster) |
| 5 | 34 | 3.12 | 4.58 | 0.68× | CPU |
| **8** | 52 | 7.40 | 6.09 | **1.21×** | GPU (crossover) |
| 20 | 124 | 68.9 | 21.3 | 3.23× | GPU |
| 80 | 484 | 3514 | 529 | **6.64×** | GPU |

**Takeaway — RULE #1 validated firsthand.** For a single *small* calibration (1–5 cameras, the common case) the CPU wins decisively — at 1 view the host Eigen path is **~7× faster** than the GPU, because per-step host↔device transfer + `cudaMalloc`/launch overhead dwarf an O(n³) solve of a tens-of-rows matrix. The crossover sits at **~5–8 views (n_tangent ≈ 34–52)**; past it the GPU dense back-end's advantage widens to ~6–7× by 80 views. So **CPU stays the default for single small calibrations**, exactly as RESEARCH.md Theme 1 predicted, and the GPU back-end is an explicit opt-in (`SolverBackend::GpuCuda`) for the large/batched regime.

**Caveats (do not over-generalize this row):**
- **Server GPU, not the edge target.** This is an RTX 5090 (server/desktop class). The **primary edge target is Jetson Orin (`sm_87`)**, where the crossover will sit at a *different* n_views — §D.3's edge↔server measurement is still pending and must be re-run there. Crossover is hardware-specific; the benchmark prints "MEASURED here, never assumed."
- **Native back-end, not the borrows.** This measures CalibForge's own cuBLAS/cuSOLVER dense LM, *not* PyPose / Graphite / MegBA (§D.2 still pending). It is a *dense* solver; the sparse-Jacobian batched regime those borrows target is a separate measurement.
- **Per-step allocation overhead is real.** `cudaSolveLmStep` currently `cudaMalloc`/`cudaFree`s its device buffers and re-uploads `J` every LM step; caching those across steps would lower the crossover. Recorded as future work, not yet done.
