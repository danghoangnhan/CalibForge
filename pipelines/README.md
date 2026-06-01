# `pipelines/` — calibration workflows

Thin glue over [`solve/`](../solve): each pipeline orchestrates detect → solve → validate for one scenario.

`single` · `stereo` · `rig` (N-camera) · `hand-eye` · `cam-IMU` · `online`/`targetless`.

Each pipeline assembles the appropriate [`ResidualBlock`](../solve/include/calibforge/residual_block.hpp)s and routes to a [`SolverBackend`](../solve/include/calibforge/problem.hpp) (CPU for single small calibrations; GPU for batched/large/online). Online pipelines MUST pass results through the [`ObservabilityGate`](../solve/include/calibforge/observability.hpp) before emitting.

**Status:** placeholder — no implementation yet.
