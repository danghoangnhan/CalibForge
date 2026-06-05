<!--
  Prepared issue status updates for GitHub issues #23 and #25.
  This host has no GitHub auth (no GH_TOKEN / gh CLI), so these could not be posted automatically.
  Post them with, e.g.:  gh issue comment 23 --body-file <(sed -n '/## Issue #23/,/## Issue #25/p' docs/ISSUE_UPDATES.md)
  (or copy/paste). Delete this file once posted.
-->

## Issue #23 — v0.4 cam-IMU follow-up: FULL pipeline landed

The headline remaining v0.4 item — **"the spatial translation extrinsic is not yet estimated (only R_ci), and the preintegration factor is not yet wired into a `calibrate_cam_imu_full` pipeline"** — is now **done** (CPU, host-validated on a Grace-Blackwell GB10 / aarch64).

**Delivered**
- `solve/include/calibforge/cam_imu_pose_residual.hpp` — a new `CamImuPoseResidual` (6-dim) that ties a measured camera-in-world pose `T_wc` to the IMU nav-state `(R_wi, p_wi)` through the **full spatial extrinsic `T_ic = (R_ci, p_ci)`**, with 4 analytic Jacobian blocks (FD-validated).
- `pipelines/include/calibforge/calibrate_cam_imu_full.hpp` — `calibrateCamImuFull(...)` wires the Forster preintegration factor + the pose residual into ONE `DenseProblem` and jointly estimates: the **translation lever arm `p_ci`**, `R_ci`, gyro/accel biases `b_g`/`b_a`, gravity `g_w`, and the per-keyframe nav-states `(R_wi, p_wi, v_wi)`.
- **Observability (RULE #2):** the returned `information` is the calibration block **Schur-marginalized over the nuisance nav-states** (`S = H_cc − H_cn H_nn⁻¹ H_nc`), so `assessObservability()` judges whether the extrinsic+biases(+gravity) are actually constrained by the motion — not the inflated full-state matrix.
- `tests/test_cam_imu_full.cpp` (3 cases): FD-validation of all 4 Jacobian blocks; full recovery from a synthetic forward-Euler-consistent 6-DOF trajectory (**`p_ci` recovered to ~2e-7**, `R_ci` ~6e-9, biases + gravity recovered, gate passes); and a degenerate **pure-translation** scene where the extrinsic is unobservable and the gate correctly **refuses** (confidence ~1e-18). *(The tight figures are values observed firsthand; the asserts gate more loosely — `< 1e-4` on the extrinsic — so the suite stays deterministic across x86/aarch64.)*

**Still open (the only thing keeping the epic open):**
- **Isaac Sim synthetic validation** — still deferred (no Isaac/GPU-render host here). The CPU synthetic generators are strengthened (a full 6-DOF trajectory with known biases/gravity, residual ≈ 0 at GT by construction), but the Isaac-specific render path is genuinely blocked on that environment.

---

## Issue #25 — v1.0: B-spline wiring, FP32 parity, CUDA-13 fix, second GPU host

Multiple v1.0 items advanced, validated firsthand on a **Grace-Blackwell GB10 (aarch64 + Blackwell `sm_121` + CUDA 13.0)** — an edge-class ARM + Blackwell host with unified memory (the closest proxy yet to the deferred Jetson target).

**Generic per-pixel B-spline model — now ✅ (was PARTIAL/⏳)**
- Wired into a calibration pipeline (`solve/include/calibforge/calibrate_generic_bspline.hpp`), io (`io/include/calibforge/generic_bspline_yaml.hpp`, lossless YAML round-trip), and the **Python bindings** (`GenericBSplineGrid`, `GenericBSplineCamera`, `fit_from_parametric`, `generic_bspline_to/from_yaml`, `calibrate_generic_bspline`).
- **Closes "the only accuracy test fits a synthetic pinhole source":** `tests/test_generic_bspline_calibrate.cpp` now calibrates the B-spline to a **wide-FOV double-sphere** *and* a **Kannala-Brandt fisheye** source — reprojection RMS drops to **~0.0002 px** and the fitted model reproduces the source across the calibrated FOV to **~0.0006 px**. (Real-*dataset* validation still pending — this is synthetic wide-FOV.) *(Tight figures are observed firsthand; the asserts gate at the looser sub-pixel `rms < 0.01` / functional `< 0.05` for cross-arch determinism headroom.)*

**Native CUDA solver — FP32 path + FP32↔FP64 parity (the §D.3 "unproven" item) — now measured**
- Added `SolverBackend::GpuCudaF32` + `cudaSolveLmStepF32` (the dense solver `.cu` is now templated over the device scalar). On the GB10: **GPU-FP64 vs host-FP64 = 3.1e-16**, **GPU-FP32 vs host-FP64 = 1.7e-7** (single-precision envelope), **GPU-FP32 vs host-FP32 = 1.2e-7** (a correct single-precision solve). A full FP32-stepped calibration recovers the same parameters as the FP64 oracle. `tests/test_cuda_fp32_fp64_parity.cpp`.

**RULE #6 — CUDA 13 broke the arch matrix; fixed + re-validated forward-compat**
- CUDA 13.0 **removed `sm_72`** (Volta/Xavier): the hardcoded default matrix made `enable_language(CUDA)` fail outright. `CMakeLists.txt` now probes the nvcc version and **drops `sm_72` on CUDA ≥ 13** (kept on ≤ 12). The single-source multi-arch build (topping out at `compute_90` PTX, **no** native sm_121) runs **all 131 tests on the GB10 via PTX JIT** — RULE #6 forward-compat proven on a second Blackwell arch.

**Benchmark — second host, crossover moved (RULE #1 re-validated)**
- On the GB10 the **CPU still wins at 20 views** (crossover beyond 20), vs ~8 on the RTX 5090 — the fast Grace ARM CPU + unified memory shifts the balance. "Crossover is hardware-specific, re-measure per target" is now empirical across two hosts (`docs/BENCHMARKS.md`, `docs/SPIKES.md` §F).

**Also:** fixed a cross-platform convergence-flag fragility (a double-sphere recovery flagged converged on x86 but not aarch64 on a bit-correct solution) — the LM `converged` flag is now scale-invariant (relative-gradient + trust-region-collapse-after-descent), hardening edge↔server determinism.

**Still open:** PyPose/Graphite/MegBA **sparse/batched** borrows (the native dense back-end fills the dense slot; those are PyTorch-form per RULE #3 and target the sparse regime); a **true Jetson Orin (sm_87)** crossover measurement; **bf16** parity; **real-dataset** wide-FOV validation; the single-build edge↔server packaging artifact.
