# CalibForge Benchmarks

Calibration-regime timing for the solver back-ends (SPIKES.md §D.1). Generate the CPU rows
with the bundled harness:

```bash
cmake --build build --target calibforge_bench
./build/calibforge_bench > bench.csv
```

Each row is:

| column | meaning |
|---|---|
| `problem` | scenario label (e.g. `single_small`, `stereo`, `rig_8cam`) |
| `n_views` | number of calibration views |
| `n_points` | object points per view |
| `n_cams` | cameras in the rig |
| `cpu_ms_median` | median wall-clock per solve (steady_clock), ms |
| `cpu_iters_median` | median LM iterations to convergence |
| `final_cost_median` | median final 0.5·‖r‖² |

## CPU-vs-GPU solver crossover (native CUDA back-end)

The native CalibForge CUDA back-end (`SolverBackend::GpuCuda` — cuBLAS SYRK/GEMV for `J^T J` /
`J^T r` + cuSOLVER dense Cholesky, see `solve/cuda_linear_solver.hpp`) offloads the per-iteration
damped normal-equations solve; the analytic Jacobians stay on the host. The benchmark solves the
**same** `DenseProblem` (single camera, growing views → growing `n_tangent = 4 + 6·n_views`) with
each backend. The `dense_single` rows add columns `n_tangent,cpu_ms_median,gpu_ms_median,gpu_speedup,cpu_cost_median,gpu_cost_median`.

Measured on an **RTX 5090 (sm_120) + CUDA 12.0**, host CPU = single-threaded Eigen (CalibForge's
actual CPU path; not multi-threaded MKL). A second sweep on a **Grace-Blackwell GB10 (aarch64 +
sm_121 + CUDA 13.0)** follows below — the crossover is hardware-specific and lands at a different
`n_views` there:

| n_views | n_tangent | cpu_ms | gpu_ms | gpu_speedup | winner |
|--:|--:|--:|--:|--:|:--|
| 1  | 10  | 0.28   | 1.97   | **0.14** | CPU |
| 2  | 16  | 0.75   | 2.90   | **0.26** | CPU |
| 3  | 22  | 1.32   | 3.39   | **0.39** | CPU |
| 5  | 34  | 3.12   | 4.58   | **0.68** | CPU |
| 8  | 52  | 7.40   | 6.09   | 1.21 | GPU (crossover) |
| 10 | 64  | 12.07  | 7.65   | 1.58 | GPU |
| 20 | 124 | 68.9   | 21.3   | 3.23 | GPU |
| 40 | 244 | 462.1  | 82.4   | 5.61 | GPU |
| 80 | 484 | 3514.0 | 529.4  | **6.64** | GPU |

(Medians: 15 reps for ≤ 10 views, 5 reps above. Both backends converge to the same noise-free
minimum on every row — the harness verifies this and warns otherwise, so the speedups are
apples-to-apples; iteration counts agree backend-to-backend and are printed in the raw CSV.)

### Second host: Grace-Blackwell GB10 (aarch64 + sm_121 + CUDA 13.0)

The same `dense_single` sweep on the **GB10** (20-core ARM Cortex-X925 "Grace" CPU + Blackwell
`sm_121` GPU, 121 GB unified memory), produced by `tools/benchmark/calibforge_bench.cpp` on an
optimized build — the harness now times the FP32 path (`SolverBackend::GpuCudaF32`) and emits the
`gpu32_ms` column directly, so the FP32 column is reproducible from the committed tool. `gpu64_ms`
is the FP64 back-end:

| n_views | n_tangent | cpu_ms | gpu64_ms | gpu32_ms | gpu64_speedup | winner |
|--:|--:|--:|--:|--:|--:|:--|
| 1  | 10  | 0.17   | 9.30   | 1.45   | **0.02** | CPU (~55×) |
| 5  | 34  | 1.29   | 14.77  | 7.56   | **0.09** | CPU |
| 10 | 64  | 3.65   | 16.10  | 8.47   | **0.23** | CPU |
| 20 | 124 | 17.41  | 33.02  | 18.92  | **0.53** | CPU |
| 40 | 244 | 107.00 | 74.85  | 46.69  | **1.43** | **GPU** |
| 80 | 484 | 811.68 | 289.83 | 193.50 | **2.80** | **GPU** |

**Same rule, different crossover.** On the GB10 the **CPU wins through 20 views; the GPU only crosses
over between 20 and 40** (vs ~8 on the RTX 5090) — the sweep now extends to 40/80 so the crossover is
visible, not just asserted. The fast Grace ARM CPU + unified memory and the relatively larger
per-step `cudaMalloc`/transfer overhead push the balance toward the CPU. **FP32 (`gpu32`) is
consistently faster than FP64 (`gpu64`)** on the device (half the bytes + faster single-precision
throughput) and is the first backend to overtake the CPU. This is the empirical proof that "crossover
is hardware-specific, re-measure per target" — two hosts, two crossovers (`docs/SPIKES.md` §E, §F).
(Benchmark on an optimized build: an `-O0` build inflates the Eigen CPU solver ~30× and moves the crossover.)

**RULE #1 confirmed empirically on both hosts, not assumed:** the GPU is *not* automatically faster.
For a small single calibration the host wins — per-step host↔device transfer + malloc/launch
overhead dominates an O(n³) solve. The GPU crosses over at ~8 views on the RTX 5090 and between 20
and 40 on the GB10. **The crossover is hardware-specific** (a Jetson Orin sm_87 will differ again) and must
be re-measured per target — `SolverBackend::Auto` therefore stays on the CPU until a measured
threshold is wired. CPU↔GPU **and** FP32↔FP64 parity are verified
(`tests/test_cuda_linear_solver.cpp`, `tests/test_cuda_fp32_fp64_parity.cpp`): the GPU FP64 step
matches the host Eigen LDLT solve to < 1e-8 (3e-16 on the GB10), GPU FP32 agrees with the FP64
oracle to the single-precision envelope (~1.7e-7), and a full `GpuCuda`/`GpuCudaF32` solve recovers
the same calibration as the CPU backend.

## Status

- **CPU rows + native-CUDA GPU rows:** produced by `tools/benchmark/calibforge_bench.cpp` on a CUDA
  host (the `dense_single` block above). On a CUDA-less host only the CPU rows print.
- **PyPose / Graphite / MegBA back-ends:** still pending (#25) — the native CUDA back-end fills the
  GPU-dense-solver slot (FP64 + FP32); the sparse/batched/bf16 back-ends remain future work.
- **FP32↔FP64 numerical parity:** ✅ measured on the GB10 (`docs/SPIKES.md` §F.3 /
  `tests/test_cuda_fp32_fp64_parity.cpp`). **bf16** parity + a true **Jetson Orin (sm_87)** crossover
  re-measurement remain deferred (the GB10 is a strong ARM+Blackwell proxy but not Orin).
