# `apply/` — runtime correction (the free "apply" path)

Generates rectification + undistortion maps from estimated parameters and exports runtime artifacts.

- **Jetson:** export to **VPI LDC** warp-maps (`vpiWarpMapGenerateFromPolynomialLensDistortionModel` for Brown-Conrady k1..k6+p1,p2; `...FromFisheyeLensDistortionModel` for KB) → consumed by VPI Remap on **PVA/VIC** engines, keeping the GPU free for perception. Map OpenCV coefficients **by named field**, not raw array copy.
- **Server:** VPI's PVA/VIC engines are **Jetson-only** → use VPI's CUDA backend or **CV-CUDA** Remap on x86_64.

**Status:** placeholder — no implementation yet. See [`../docs/RESEARCH.md`](../docs/RESEARCH.md) Theme 5 for the VPI LDC API + the Jetson-only caveat.
