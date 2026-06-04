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

## Status

- **CPU rows (Ceres-class CPU LM):** produced by `tools/benchmark/calibforge_bench.cpp`.
- **GPU rows (PyPose / Graphite / MegBA):** deferred to v1.0 — they require a CUDA / Jetson
  host (see `docs/SPIKES.md` §D and issue #25). Per CLAUDE.md rule 1, GPU is **not**
  automatically faster for small single calibrations; these rows must be **measured**, not
  assumed, and are expected to win only on batched fleet / large-rig / generic-model /
  online-continuous workloads.

> No numbers are committed yet: the comparison is only meaningful once both halves run on the
> same machine. Paste a generated `bench.csv` here when the GPU back-ends land.
