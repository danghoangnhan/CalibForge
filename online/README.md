# `online/` — online / targetless recalibration (the differentiator)

The project's moat (CLAUDE.md RULE #2 / `docs/RESEARCH.md` Themes 2–3): online recalibration
**behind an observability/confidence gate** — *never silently emit* ill-conditioned parameters.
Low reported error does not mean a trustworthy estimate (precision ≠ accuracy).

- `online_calibration.hpp` — `OnlineIntrinsicTracker`: accumulates per-frame `View`s, re-estimates
  over the window, and **emits only when the estimate is observable AND its confidence clears a
  threshold** (`assessObservability` + `parameterUncertainty`); otherwise it refuses and names the
  weak directions, and reports drift vs the reference.

The observation source is front-end-agnostic — a `View` may come from a target (`detect/`) or,
later, from **targetless** feature tracks / BEV photometric constraints (the v0.5 data layer;
OpenCalib SurroundCameraCalib baseline, `docs/DEPENDENCIES.md`). The gated loop is the
unbuilt-elsewhere core. Extrinsic-drift tracking for rigs and the surround-view BEV front-end
follow.
