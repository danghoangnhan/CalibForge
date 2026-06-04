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
actual CPU path; not multi-threaded MKL):

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

**RULE #1 confirmed empirically, not assumed:** the GPU is *not* automatically faster. For a small
single calibration (≤ ~5 views, `n_tangent` ≲ 34) the host wins — per-step host↔device transfer +
malloc/launch overhead dominates an O(n³) solve. The GPU crosses over at ~8 views (`n_tangent` ~52)
and the gap widens to ~6.6× by 80 views. **The crossover is hardware-specific** (a Jetson Orin
sm_87 will differ) and must be re-measured per target — `SolverBackend::Auto` therefore stays on
the CPU until a measured threshold is wired. FP64 CPU↔GPU parity is verified
(`tests/test_cuda_linear_solver.cpp`: the GPU step matches the host Eigen LDLT solve to < 1e-8, and
a full `GpuCuda` solve recovers the same calibration as the CPU backend; final costs agree above).

## Status

- **CPU rows + native-CUDA GPU rows:** produced by `tools/benchmark/calibforge_bench.cpp` on a CUDA
  host (the `dense_single` block above). On a CUDA-less host only the CPU rows print.
- **PyPose / Graphite / MegBA back-ends:** still pending (#25) — the native CUDA back-end fills the
  GPU-dense-solver slot; the sparse/batched/bf16 back-ends remain future work.
- **Jetson↔server FP32/bf16↔FP64 numerical-parity** rows: deferred — need a Jetson host
  (`docs/SPIKES.md` §D.3).
