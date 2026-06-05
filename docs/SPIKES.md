# CalibForge — Dependency De-risking Spikes

Empirical checks of the [`RESEARCH.md`](./RESEARCH.md) borrow decisions, run on the dev host on **2026-06-01**. Goal: don't trust the papers alone — verify the load-bearing claims firsthand, and honestly record what could **not** be verified here.

> **Implementation status (v0.5 in progress).** Everything verified in §A–C below is reflected in the shipped code. **§D.1 (GPU-vs-CPU on the calibration-sized regime) has been run firsthand on TWO CUDA hosts — a server RTX 5090 / CUDA 12.0 ([§E](#e-cuda-host-spikes--run-firsthand-on-an-rtx-5090--cuda-120)) and an edge-class Grace-Blackwell GB10 / aarch64 / sm_121 / CUDA 13.0 ([§F](#f-second-cuda-host--grace-blackwell-gb10--run-firsthand-on-aarch64--sm_121--cuda-130)).** §F additionally closes **§D.3's FP32↔FP64 numerical parity** (measured via the new `GpuCudaF32` back-end) and fixes the **CUDA-13 arch-matrix breakage** (sm_72 removed in CUDA 13). Still deferred: a true **Jetson Orin (sm_87)** crossover measurement (the GB10 is a strong ARM+Blackwell proxy but not Orin), **bf16** parity, the **Graphite build (§D.2)** + PyPose/MegBA borrows, and the **VPI-LDC Jetson PVA/VIC check (§D.4)**. See repo root `README.md` for the current capability matrix.

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
| Host-only degradation still clean | ✅ forcing `-DCMAKE_CUDA_COMPILER=NOTFOUND` configures `CUDA=OFF`, builds, and runs **114 tests** (the 8 CUDA-only cases compile out) — graceful degradation intact after the GPU work. *(Counts as of this §E commit; the suite has since grown — currently 122 host-only / 132 with the CUDA half, see §F / CLAUDE.md.)* |

### E.2 Native CUDA dense LM solver (#25) correctness — ✅

A from-scratch CalibForge GPU back-end (`solve/src/cuda_linear_solver.cu`) solves one damped-normal-equations LM step on the device — `J^T J` via cuBLAS DSYRK, `J^T r` via DGEMV, the SPD factor/solve via cuSOLVER dense Cholesky (`potrf`/`potrs`) — with analytic Jacobians still assembled on the host (RULE #4). All running on `sm_120` through the JIT'd `compute_90` PTX.

| Check (`tests/test_cuda_linear_solver.cpp`) | Result |
|---|---|
| Full suite | ✅ **122 / 122 pass** (114 host-only + 8 CUDA-only, incl. the `remap_cpu_gpu_parity` case; suite size as of this §E commit — now 132 on a CUDA host, see §F) |
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

## F. Second CUDA host — Grace-Blackwell GB10 (✅ run firsthand on aarch64 + sm_121 / CUDA 13.0)

Run on **2026-06-05** on a **third host that is both edge-class and Blackwell-class**: an NVIDIA **GB10 "Grace-Blackwell"** (DGX Spark-class) — a **20-core ARM Cortex-X925 (aarch64) Grace CPU + a Blackwell GPU (`sm_121`, compute capability 12.1) sharing 121 GB of unified memory**, on **CUDA 13.0**. This is the closest proxy yet to the deferred Jetson target (ARM + integrated GPU + unified memory), and it closes two items §E could not: a **CUDA-13 toolkit** (which changed the supported arch set) and **FP32↔FP64 numerical parity** (§D.3) on real edge-class silicon.

### Environment (this host)

| Tool | Status |
|---|---|
| OS / arch | Linux **aarch64** (ARM Cortex-X925 "Grace", 20 cores) |
| **GPU / `nvidia-smi`** | ✅ **NVIDIA GB10** — compute capability **`sm_121` (Blackwell)**, 121 GB unified memory |
| **CUDA / `nvcc`** | ✅ **release 13.0, V13.0.88** |

### F.1 CUDA 13 dropped sm_72 — the hardcoded arch matrix broke configure (now fixed) — ✅

CUDA 13.0 **removes Volta/Xavier** (`sm_70`/`sm_72`): `nvcc -arch=compute_72` is now a **fatal** error, so the CUDA-12-era default matrix `72;80;86;87;89;90;90-virtual` failed `enable_language(CUDA)` outright on this host.

| Check | Result |
|---|---|
| Default matrix configures on CUDA 13 (before fix) | ❌ `nvcc fatal: Unsupported gpu architecture 'compute_72'` — configure aborts |
| Version-aware matrix (after fix) | ✅ `CMakeLists.txt` probes the nvcc version and **drops `sm_72` on CUDA ≥ 13** (keeps it on CUDA ≤ 12 for Xavier users); the default matrix configures cleanly → effective `80;86;87;89;90;90-virtual` |
| Fatbin contents (`cuobjdump`) | ✅ native SASS for `sm_80 / sm_86 / sm_87 / sm_89 / sm_90` **plus `compute_90` PTX** |

### F.2 Single-source multi-arch build runs on sm_121 via PTX JIT (RULE #6, again) — ✅

Same forward-compat story as §E (RTX 5090 / sm_120) but on a **different toolkit (13.0) and a different Blackwell arch (sm_121)** — the `90-virtual` PTX tail JIT-forwards onto sm_121 with no native sm_121 SASS in the fatbin.

| Check | Result |
|---|---|
| Full suite built with the DEFAULT matrix (tops out at `compute_90` PTX, **no** native sm_121) | ✅ **all 131 tests pass on the GB10 via PTX JIT** — the GPU dense solver + FP32 parity run on sm_121 through the JIT'd `compute_90` PTX |
| Native build (`-DCMAKE_CUDA_ARCHITECTURES=native`, sm_121 SASS) | ✅ also 131/131 — both paths agree |

### F.3 §D.3 — FP32 ↔ FP64 numerical parity (the deferred edge-precision question) — ✅ measured

A native **FP32 GPU back-end** (`SolverBackend::GpuCudaF32` / `cudaSolveLmStepF32`) was added: it casts the host FP64 Jacobian/residual down to single precision, runs the entire damped solve (SYRK/GEMV/Cholesky) in FP32, and casts `dx` back up — so the parity test compares a *genuine* single-precision GPU solve against the FP64 GPU + host oracles. Measured on the GB10 (`tests/test_cuda_fp32_fp64_parity.cpp`):

| Comparison (one well-conditioned LM step, m=64, n=10) | Relative error |
|---|---|
| GPU **FP64** vs host **FP64** (Eigen LDLT) | **3.1e-16** (machine precision) |
| GPU **FP32** vs host **FP64** | **1.7e-07** (the single-precision envelope) |
| GPU **FP32** vs host **FP32** (Eigen LDLT in float) | **1.2e-07** (the GPU FP32 path is a *correct* single-precision solve, not merely imprecise) |
| **Full pinhole calibration** with FP32 GPU steps vs FP64 CPU oracle | recovers identical intrinsics (the LM loop evaluates + accepts steps in FP64, so single-precision step *directions* converge to the same FP64 minimum) |

**Takeaway:** FP32 and FP64 AGREE to the single-precision round-off envelope (~1e-7) on a well-conditioned step, and an FP32-stepped *full calibration* lands on the same parameters as the FP64 oracle. The earlier "FP32/bf16 vs FP64 parity is unproven" caveat is now **measured** (bf16 remains future work). Caveat: FP32 degrades on *ill-conditioned* systems (`eps·cond` grows), so FP64 stays the default/accuracy oracle and FP32 is an explicit opt-in.

### F.4 CPU-vs-GPU crossover on the GB10 — RULE #1 holds, crossover is FURTHER OUT than on the RTX 5090 — ✅

Identical `DenseProblem` (single pinhole, 9×9 board, `n_views` poses), each backend, medians over 15 reps (≤10 views) / 5 reps (>10 views) — produced by the committed `tools/benchmark/calibforge_bench.cpp` on an optimized (`RelWithDebInfo`) build, which now times the FP32 (`SolverBackend::GpuCudaF32`) path too and emits the `gpu32_ms` column directly, so this table is reproducible from the tool:

| n_views | n_tangent | cpu_ms | gpu64_ms | gpu32_ms | gpu64_speedup | winner |
|---:|---:|---:|---:|---:|---:|:--|
| 1 | 10 | 0.17 | 9.30 | 1.45 | 0.02× | CPU (~55×) |
| 5 | 34 | 1.29 | 14.77 | 7.56 | 0.09× | CPU |
| 10 | 64 | 3.65 | 16.10 | 8.47 | 0.23× | CPU |
| 20 | 124 | 17.41 | 33.02 | 18.92 | 0.53× | CPU |
| 40 | 244 | 107.00 | 74.85 | 46.69 | 1.43× | **GPU** |
| 80 | 484 | 811.68 | 289.83 | 193.50 | 2.80× | **GPU** |

> `gpu64_speedup = cpu_ms / gpu64_ms` (>1 ⇒ GPU faster). Both backends converge in the same #iterations to the same noise-free minimum on every row (verified by the harness's apples-to-apples cost guard), so the speedups compare equal work. Build matters: an unoptimized (`-O0`) build inflates the Eigen CPU solver ~30× and spuriously moves the crossover — always benchmark an optimized build.

**Takeaway — RULE #1 validated on a *second, different* host, and the crossover MOVED.** On the RTX 5090 the GPU crossed over at ~8 views (§E.3); on the GB10 the **CPU wins through 20 views and the GPU only crosses over between 20 and 40** (gpu64 1.43× at 40, gpu32 already 0.92× at 20). The fast Grace ARM CPU + unified memory shifts the balance toward the CPU (and the per-step `cudaMalloc`/transfer overhead, §E caveat, is relatively larger here). This is exactly why the benchmark prints "crossover is hardware-specific, MEASURED here, never assumed" — two hosts, two crossovers. Note also **FP32 (`gpu32`) is consistently faster than FP64 (`gpu64`)** on the GPU (half the data + faster single-precision throughput), and is the first backend to overtake the CPU.

**Still pending after §F:** a true **Jetson Orin (sm_87)** measurement (the GB10 is a strong ARM+Blackwell proxy but not Orin); the **PyPose / Graphite / MegBA** sparse/batched borrows (§D.2); **bf16** parity; and **real-dataset** (vs synthetic) wide-FOV B-spline validation.
