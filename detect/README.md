# `detect/` — observation extraction

Turns images into the observations the solver consumes.

- **Target-based:** checkerboard, ChArUco / AprilGrid corner detection (GPU-accelerated) + sub-pixel refinement.
- **Targetless:** feature tracks for online calibration; BEV photometric error in overlapping regions (surround-view); tie-points for UAV blocks.

**Status:** placeholder — no implementation yet. See [`../CLAUDE.md`](../CLAUDE.md) and [`../docs/RESEARCH.md`](../docs/RESEARCH.md) (Theme 3 for the targetless/BEV methods).
