# CalibForge — Research Synthesis (mid-2026)

**What this is.** A citation-backed survey of the open problems and the concrete methods available to implement each component of CalibForge (see [`DESIGN.md`](./DESIGN.md) for the vision). It re-validates the design doc's build-vs-borrow choices as of **2026-06-01** and labels every method by **maturity** and **edge-readiness**.

**Implementation status (v0.5 in progress).** Theme 4's permissive-borrow conclusion (nvTorchCam-style interface, add DS + EUCM ourselves) and Theme 2's "residuals coexist behind one BA cost" architectural claim are both built and FD-validated in tree. Theme 3's differentiator — observability + 6-axis-excitation gating around online recalibration — is implemented for both intrinsics and extrinsics. Theme 1's calibration-regime GPU-vs-CPU benchmark (the single most important number listed in [Open questions](#open-questions--re-check-list)) remains deferred to a CUDA host. See repo root `README.md` for the current capability matrix and [`SPIKES.md`](./SPIKES.md) §D for the deferred work.

**How it was produced.** Five independent deep-research passes (one per theme), each: decompose into ~5–6 search angles → parallel web search → fetch ~20–26 primary sources → extract ~100–128 falsifiable claims → **3-vote adversarial verification** (a claim needs 2/3 refutes to be killed) → synthesize. Per-theme verification counts:

| Theme | Sources | Claims | Verified | Confirmed | Killed |
|---|---|---|---|---|---|
| 1 — GPU bundle adjustment | 20 | 99 | 25 | 22 | 3 |
| 2 — Rolling-shutter & cam-IMU | 24 | 119 | 25 | **25** | **0** |
| 3 — Online / targetless | 23 | 113 | 25 | 24 | 1 |
| 4 — Camera models & generic | 26 | 128 | 25 | 24 | 1 |
| 5 — Dependency re-validation | 22 | 106 | 25 | 22 | 3 |

**Reading the labels.** *Maturity:* production-proven / peer-reviewed / research-prototype. *Edge-readiness:* whether it actually runs on Jetson (most sources do **not** report on-device benchmarks — those labels are architectural inference and are flagged as such). **Time-sensitive facts** (versions, maintenance, licenses) are true as of 2026-06-01 and must be re-checked before relying on them; the re-check list is the [Open Questions](#open-questions--re-check-list) section.

---

<a name="theme-1"></a>
## Theme 1 — GPU bundle-adjustment solver core

**⚠️ Pivotal design insight.** The calibration-sized regime (few cameras, many residuals) does **not** automatically favor GPU BA. Theseus's own benchmarks show **Ceres (CPU C++) ~25× faster than batched GPU solvers at small batch sizes**; GPU BA wins on **batched / SfM-scale / large-rig / online-continuous** workloads. → The `solve` layer must be **solver-agnostic** with a CPU path (default for single small calibrations) and a GPU path (batched fleet, large rigs, generic 10⁴-param models, online). **No surveyed solver was benchmarked on the calibration regime — measure it ourselves.**

| Solver | Approach | License / maturity | Edge/Jetson | Source |
|---|---|---|---|---|
| **Graphite** ⭐ | Matrix-free PCG, block-Jacobi precond., implicit-Schur (Hessian-free), FP64/FP32/**bf16** | MIT · research-prototype (WIP), v0.5.0 (2026-05) | ✅ **only solver actually run on Jetson Orin Nano** (4× edge / 59× desktop vs g2o-PCG); bf16 ~4× less mem than MegBA | [arXiv 2509.26581](https://arxiv.org/abs/2509.26581) · [github.com/sfu-rsl/graphite](https://github.com/sfu-rsl/graphite) |
| **PyPose** ⭐ | LM + TrustRegion; pluggable solver/strategy/kernel/corrector; Cholesky/sparse/PCG | Apache-2.0 · active (v0.9.5, 2026-04) | ✅ any PyTorch GPU | [CVPR 2023 PDF](https://wonderren.github.io/files/wang23_pypose_cvpr.pdf) · [arXiv 2409.12190](https://arxiv.org/abs/2409.12190) · [github.com/pypose/pypose](https://github.com/pypose/pypose) |
| **MegBA** | Distributed Schur + distributed PCG (NCCL) | Apache-2.0 · production-adjacent research | ❌ none (multi-GPU server) | [arXiv 2112.01349](https://arxiv.org/abs/2112.01349) · [github.com/MegviiRobot/MegBA](https://github.com/MegviiRobot/MegBA) |
| **Theseus** | GN / LM(adaptive damping) / Dogleg, one differentiable layer; CHOLMOD/cudaLU/BaSpaCho solvers | MIT · mature (v0.2.2, 2024-09) | GPU (desktop) | [arXiv 2207.09442](https://arxiv.org/abs/2207.09442) · [github.com/facebookresearch/theseus](https://github.com/facebookresearch/theseus) |
| **STBA** | Stochastic clustering of the Reduced Camera System | research / CPU (2020, background) | — | [arXiv 2008.00446](https://arxiv.org/abs/2008.00446) |
| **Ceres** | CPU Schur+PCG/Cholesky, LM/Dogleg | BSD · production-proven | CPU (incl. Jetson CPU) | [ceres-solver.org](http://ceres-solver.org/automatic_derivatives.html) |

**Key facts (verified):**
- **PyPose FastTriggs** robust-loss corrector uses only the 1st-order kernel derivative → more numerically stable than Ceres' Triggs correction (which needs the unstable 2nd-order derivative). LieTensor GPU ops up to 42.9× faster than Theseus, ¼–½ memory. Converges to the same final error as Ceres/GTSAM.
- **Graphite** bf16: Venice-1778 950 MiB / MSE 0.85 vs MegBA 4528 MiB / MSE 0.84 (~4.8× less memory); completes Final-13682 where MegBA OOMs. bf16 needs stabilization (Hessian-diagonal clamping, Jacobian column rescaling, PCG residual normalization).
- **Theseus BaSpaCho** performs Schur-elimination math *inside* a supernodal Cholesky — no external Schur trick.

**Also surfaced (chase down):** [Matrix-Free Shared-Intrinsics BA (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Safari_Matrix-Free_Shared_Intrinsics_Bundle_Adjustment_CVPR_2025_paper.pdf) — *directly* calibration-relevant; RootBA / √BA ([arXiv 2103.01843](https://arxiv.org/abs/2103.01843)); GTSAM.

**Hard problems unaddressed by all surveyed tools (→ CalibForge builds these):** gauge-freedom / datum fixing on GPU; explicit Huber/Cauchy on GPU (only PyPose FastTriggs verified); RS-line-time / IMU-preintegration / planar residual blocks behind one interface; **CPU↔GPU and FP32/bf16↔FP64 numerical determinism** for reproducible intrinsics.

---

<a name="theme-2"></a>
## Theme 2 — Rolling-shutter & camera-IMU calibration

*(highest-confidence theme: 25/25 claims confirmed, 0 killed, all primary peer-reviewed sources)*

**Two design axes recur — choose deliberately:**
1. **RS readout:** whole-image readout time `t_r` (per-row `t = t_I + (m/M)·t_r`; `t_r=0` ⟹ global shutter) [MVIS, Kalibr, URS-NeRF] **OR** per-line delay `τ_ld` [Ctrl-VIO, Huai, iKalibr]. Equivalent (`t_r = M·τ_ld`); enters BA as a **reprojection residual at the row-specific pose**.
2. **Trajectory:** **discrete** per-frame + IMU preintegration + linear interp [MVIS, NW-RSBA, RSL-BA] **OR** **continuous-time B-spline** (SO(3)+ℝ³ split, or full SE(3)) [Kalibr, iKalibr, Ctrl-VIO]. Continuous-time scales to high-rate IMU + per-scanline RS; discrete is simpler for unordered images.

| Tool | Calibrates | License | Source |
|---|---|---|---|
| **iKalibr** ⭐ | **Targetless** continuous-time multi-sensor (IMU/radar/LiDAR/GS+RS cam/RGBD) + RS readout | **BSD-3** ✅ | [arXiv 2407.11420](https://arxiv.org/abs/2407.11420) · [github.com/Unsigned-Long/iKalibr](https://github.com/Unsigned-Long/iKalibr) |
| **Kalibr** | Continuous-time cam-IMU: extrinsics + time-offset + (opt) IMU intrinsics; *separately* RS intrinsics | Apache/BSD-ish | [github.com/ethz-asl/kalibr](https://github.com/ethz-asl/kalibr) |
| **MVIS** | Discrete batch NLS: IMU+cam intrinsics, multi-IMU, **joint GS+RS**, analytic IMU preintegration | **GPL** ⚠️ | [IJRR 2024 PDF](https://yangyulin.net/papers/2023_IJRR_mvis.pdf) |
| **OpenVINS online self-cal** | **Online** sliding-window EKF: IMU/cam intrinsics + extrinsics + time-offset + RS readout | **GPL** ⚠️ | [arXiv 2201.09170](https://arxiv.org/abs/2201.09170) |
| **Ctrl-VIO** | Continuous-time RS-VIO, online line-delay | GPL-3 ⚠️ | [github.com/APRIL-ZJU/Ctrl-VIO](https://github.com/APRIL-ZJU/Ctrl-VIO) |

**Residual composability (key architectural validation):** point reprojection + distortion + **RS readout** + **IMU preintegration** + (new) **line reprojection** (RSL-BA, ECCV 2024, Plücker lines, [arXiv 2408.05409](https://arxiv.org/abs/2408.05409)) all coexist in one NLS/MAP cost — exactly the pluggable-residual back-end CalibForge needs. UAV/SfM: **NW-RSBA** (CVPR 2023, [arXiv 2209.08503](https://arxiv.org/abs/2209.08503)) discrete RS-BA for unordered images, Normalization+covariance-Weighting *provably avoids planar degeneracy*.

**Observability (unanimous, two peer-reviewed proofs):** a fully-calibrated VINS has **exactly 4 unobservable directions** (global yaw + 3D translation); adding IMUs doesn't help; **all calibration params observable only under fully-excited 6-axis motion**. Degenerate motions (planar, pure-translation, constant-velocity) silently leave specific params unconstrained and **compound** → capture-guidance + confidence gate must enforce 6-axis excitation.

**Edge caveat:** no Jetson benchmarks in any source. Cross-ref: NVIDIA [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM), [pyCuSFM](https://github.com/nvidia-isaac/pyCuSFM) (GPU SfM/VSLAM on Jetson); Coco-LIC (continuous-time LiDAR-inertial-camera).

---

<a name="theme-3"></a>
## Theme 3 — Online / targetless recalibration

**Surround-view (multi-fisheye BEV):**
- **OpenCalib `SurroundCameraCalib`** ⭐ ([arXiv 2305.16840](https://arxiv.org/pdf/2305.16840), **Apache-2.0**, [github.com/OpenCalib/SurroundCameraCalib](https://github.com/OpenCalib/SurroundCameraCalib)) — minimizes **photometric error in BEV overlap** via **coarse-to-fine random search over 6-DOF**, tolerates ~3° init error (vs ~0.3° for gradient methods); pinhole **or** fisheye. **The open, online-capable baseline.**
- **Click-Calib** ([arXiv 2501.01557](https://arxiv.org/html/2501.01557v1), Jan 2025) — semi-automatic (click ground points), **offline**; *rejects* photometric error in favor of geometric reprojection-distance / Mean-Distance-Error.
- **2025 frontier:** OSCalib (end-to-end online, ground semantics + SLAM "Ground-Surround Alignment"; *no open code*), online monocular **camera-to-ground (C2G)** (ground features + wheel odometry + factor graph, explicit **stop-criteria gating**; *no open code*). Open code: **ROECS** ([github.com/z619850002/ROECS](https://github.com/z619850002/ROECS)).

**UAV in-flight self-calibration (mature determinability theory):** on-the-job self-cal (IO/systematics as unknowns in block BA) preferred but **geometry-dependent**. [Fraser ISPRS 2018](https://www.isprs.org/tc2-symposium2018/images/ISPRS-Invited-Fraser.pdf); [Roncella & Forlani, Sensors 2021](https://www.mdpi.com/1424-8220/21/18/6090): **oblique imagery necessary** for principal-distance in flat terrain (0.05–0.29 px w/ obliques vs 0.4–2.9 px without); cross-strips/orthogonal-roll reduce EO↔IO coupling; object-space depth variation breaks degeneracy; redundancy (>6 rays). [GNSS-constrained BA](https://www.mdpi.com/2072-4292/13/21/4222) cuts GCPs to **1** for a long corridor (0.04 m/0.05 m), handling the "bowl-effect" degeneracy.

**⚠️ The confidence/observability sub-question — crux for "never emit bad params":**
- **Rigorous machinery exists only in the optimization/VINS & photogrammetry tracks**, not in surround-view or learning methods. VINS ([OpenVINS T-RO 2023](https://arxiv.org/pdf/2201.09170)): analytic unobservable directions + enumerated **primitive degenerate motions that compound** → **blind online calibration is unsafe**; needs motion-aware gating + priors.
- **Fraser's caution (load-bearing):** **low reported σ does NOT guarantee a trustworthy solution** — covariance is realistic only after datum/config defects removed; object-space bias can far exceed covariance-indicated uncertainty. A single-level near-nadir block is **degenerate for IO self-cal even with ~480k points** (redundancy can't fix bad geometry). ⟹ **precision ≠ accuracy**; confidence must combine **observability/geometry checks WITH covariance/FIM**, never σ alone.
- **Learning-based** ([survey arXiv 2303.10559](https://arxiv.org/abs/2303.10559) + [Awesome-Deep-Camera-Calibration](https://github.com/KangLiao929/Awesome-Deep-Camera-Calibration)): can learn intrinsics+extrinsics from monocular video targetlessly, but **provide essentially no observability/covariance/degeneracy/drift validation**.

**Open gap = CalibForge's differentiation:** *no surveyed method combines (a) surround-view/online + (b) FIM/covariance confidence scoring + (c) measured drift-tracking under vibration/thermal.* An **observability-gated, covariance-aware online recalibrator that refuses to emit ill-conditioned params** is an open research target — and exactly the design doc's stated differentiator.

---

<a name="theme-4"></a>
## Theme 4 — Differentiable camera models & generic per-pixel

| Library | License / maturity | Models | Unprojection | Source |
|---|---|---|---|---|
| **nvTorchCam** ⭐ | Apache-2.0 · official NVIDIA Labs (Oct 2024) | 8: pinhole, ortho, Brown-Conrady, equirect, KB fisheye, polynomial fisheye, KITTI-360, cubemap. **No DS/EUCM** (confirmed by source inspection) | analytic where closed-form; **differentiable Newton (≤100 it) via implicit-function-theorem** otherwise | [github.com/NVlabs/nvTorchCam](https://github.com/NVlabs/nvTorchCam) · [arXiv 2410.12074](https://arxiv.org/abs/2410.12074) |
| **Kornia** | BSD · mature | pinhole, Brown-Conrady, KB-K3, ortho (composable distortion×projection) | analytic helpers | [kornia docs](https://kornia.readthedocs.io/en/latest/sensors.camera.html) |
| **PyTorch3D FishEyeCameras** | BSD · mature | KB-style radial+tangential+thin-prism | 50-iter Newton | [pytorch3d](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/fisheyecameras.html) |

**Generic / per-pixel (Schöps et al., CVPR 2020 — canonical):** dense per-pixel rays interpolated by **cubic B-spline** over a control-point grid (~5k–41k params vs 12). **Beats 12-param OpenCV on every tested camera** (D435-C: non-central generic 0.024/0.032 px vs OpenCV 0.092/0.091). **Non-central generic is consistently most accurate** even for near-pinhole cameras; **radial-only (258-param) is usually insufficient**; generic models **need more calibration images or they overfit**. → CalibForge v1.0 `noncentral_generic`. [arXiv 1912.02908](https://arxiv.org/abs/1912.02908) · [github.com/puzzlepaint/camera_calibration](https://github.com/puzzlepaint/camera_calibration) (own code BSD-3; replace its GPL/LGPL deps).

**Jacobians:** Ceres (CPU) — autodiff machine-exact, ~same coding cost as numeric, **~40% slower than hand-analytic**. **GPU (MegBA):** analytic = identical accuracy, **~30% faster, ~40% less memory** than autodiff (SoA "JetVector", one-thread-per-edge). → **analytic on hot paths, autodiff for prototyping.**

**Lie groups:** **Theseus** (MIT) closed-form exp/log/inv/compose + analytic tangent derivatives + autograd→tangent projection. **LieTorch** (BSD-3) tangent-space backprop but **limited maintenance + CUDA build issues**. **Sophus / manif** (C++, MIT) — well-established for the C++ core (not verified this run).

**Double-sphere / EUCM gap:** *no* surveyed differentiable GPU library natively ships DS/EUCM. Implement against **double-sphere** ([Usenko 3DV 2018, arXiv 1807.08957](https://arxiv.org/abs/1807.08957)) and **basalt-headers** ([github.com/VladyslavUsenko/basalt-headers](https://github.com/VladyslavUsenko/basalt-headers), DS/EUCM/KB C++). Also: [BabelCalib (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Lochman_BabelCalib_A_Universal_Approach_to_Calibrating_Central_Cameras_ICCV_2021_paper.pdf), [fisheye-calib-adapter](https://github.com/eowjd0512/fisheye-calib-adapter), [COLMAP camera models](https://colmap.github.io/cameras.html).

---

<a name="theme-5"></a>
## Theme 5 — Dependency re-validation (build-vs-borrow, mid-2026)

**The design doc's BA borrow choice is overturned: prefer PyPose over MegBA/DeepLM.**

| Dependency | License | Maint. (2026) | Edge/Jetson | Verdict |
|---|---|---|---|---|
| **DeepLM** | **GPLv3** ⛔ | unmaintained (8 commits, last 2023) | — | **DROP** — copyleft poisons redistribution; oracle-only |
| **MegBA** | Apache-2.0 | *unknown* ("stale" claim refuted 0-3) | ❌ none (NCCL2/NVLink) | **Server-half only** |
| **PyPose** ⭐ | Apache-2.0 | active (v0.9.5, 2026-04-12) | ✅ any PyTorch GPU incl. Jetson | **PRIMARY GPU-BA borrow** |
| **NVIDIA VPI** 4.0.7 | proprietary SDK (apps free to redistribute) | active (2026-03) | ✅ Jetson AGX Thor + x86_64 dGPU | **Borrow for apply** |
| **TensorRT** 10.16/11.0 | OSS parts Apache-2.0; **core SDK proprietary (EULA)** | active | ✅ JetPack | Use; runtime engine **not** freely redistributable |
| **CV-CUDA** | Apache-2.0 | active | ✅ | Borrow for GPU image ops |
| **puzzlepaint/camera_calibration** | own code **BSD-3** ✅; deps OpenGV/SuiteSparse **GPL/LGPL** ⚠️ | — | — | First-party code only; replace GPL/LGPL deps |

**Critical edge caveat:** VPI's **PVA/VIC engines are Jetson-only** — on x86_64 servers VPI runs only its **CUDA backend**. LDC API: `vpiWarpMapGenerateFromPolynomialLensDistortionModel()` (k1..k6 + p1,p2, OpenCV-rational-compatible via **named-field mapping**, not raw array copy) and `vpiWarpMapGenerateFromFisheyeLensDistortionModel()` (4 mappings, k1..k4 — differs from OpenCV fisheye) → feed a `VPIWarpMap` to Remap.

**Ecosystem alignment:** Isaac Perceptor ingests calibration as a **URDF/xacro at `/etc/nova/calibration/isaac_calibration.urdf`** (REP-0103 frames, `base_link` root) → `io` should emit this. *(Exact camera-model schema Perceptor accepts is UNCONFIRMED — a fisheye/ftheta3/rational/plumbob list was refuted 1-2.)*

**Competitors (proprietary — align, don't borrow):** **MSA Calibration Anywhere** (Main Street Autonomy; calibration-as-a-service → Isaac URDF; closest overlap), **RidgeRun CUDA Camera Undistort** ($3,999 GStreamer plugin; undistort-only), **Tangram Vision Metrical**.

---

## Consolidated recommendation — revised build-vs-borrow stack

> **License decision: CalibForge is Apache-2.0** (open-source & free, permissive, patent grant). GPL-family code (MVIS/OpenVINS, Ctrl-VIO, DeepLM) is **reference-only — read the math, re-implement; do not vendor.** Reuse is *also* gated by **form** (re-audited 2026-06-04): even permissively-licensed deps must be re-implemented if they can't link into the edge C++ binary — **Kalibr** (BSD-3, *not* GPL, but a ROS app + GPL-transitive SuiteSparse), **iKalibr** (BSD/Apache, ROS), **nvTorchCam/PyPose/Theseus/Kornia** (Apache/MIT, PyTorch), **OpenCalib** (Apache, CLI tools), **puzzlepaint** (BSD, Qt GUI). The reuse-**as-is** borrows are the C++ building blocks: Sophus/manif (MIT), Ceres/GTSAM (BSD), Graphite (MIT) / MegBA (Apache) GPU solvers, CV-CUDA (Apache), basalt-headers (BSD), Eigen (MPL-2.0). See [`DEPENDENCIES.md`](./DEPENDENCIES.md).

| Layer | Pick | Why |
|---|---|---|
| **Camera-model core** | **nvTorchCam** (Apache-2.0) **+ add DS & EUCM** (Usenko / basalt-headers) | Confirmed differentiable, GPU-batched, 8 models; DS/EUCM the only gap |
| **GPU BA solver** | **PyPose** (primary) · **Graphite** (tracked edge/bf16) · **Ceres** (CPU path + oracle); **DeepLM dropped**, **MegBA server-only** | License + edge fit; **small calibrations favor CPU** |
| **Lie-group / manifold** | **Sophus / manif** (C++) + **Theseus** analytic-Jacobian patterns | Permissive, mature; avoid LieTorch (CUDA build issues) |
| **Apply / runtime** | **VPI LDC** on Jetson (PVA/VIC) **+ CUDA/CV-CUDA fallback on server** | PVA/VIC are Jetson-only |
| **io / interop** | OpenCV / ROS / Kalibr / COLMAP **+ emit Isaac Perceptor URDF** | Drop-in ingestion by the edge stack |
| **RS / cam-IMU residuals** | **re-implement iKalibr** (BSD-3, ROS) continuous-time formulation; residuals compose into the unified BA cost | iKalibr permissive but ROS-form → re-implement; OpenVINS/MVIS GPL, Kalibr BSD-but-ROS+GPL-transitive = reference-only |
| **Online / targetless** | **OpenCalib SurroundCameraCalib** (Apache-2.0) baseline **+ observability-gated confidence engine (BUILD — the IP)** | The open gap = the differentiator |

### Top risks → mitigations
1. **"GPU = faster" is false for small calibrations** → abstract solver; CPU default for single small; **benchmark the calibration regime ourselves**.
2. **Graphite is WIP; MegBA server-only; capable cam-IMU tools are GPL** → prefer Apache/BSD borrows; keep solver interface swappable.
3. **Online can silently emit bad params** → observability/confidence gate is **mandatory and unbuilt** (also the differentiator).
4. **Edge VRAM + determinism** → bf16 needs stabilization; CPU↔GPU↔precision parity is unproven → explicit test.
5. **License poisoning** → DeepLM out; replace Schöps' GPL/LGPL transitive deps; TensorRT runtime proprietary.
6. **Competitors overlap** → differentiate on unified edge+server single-build + observability-gated online; align with Isaac URDF.

---

## Refuted claims (killed in verification — do not relitigate)

- **MegBA "stale/unmaintained since Feb 2022"** — refuted 0-3. Maintenance status is **UNKNOWN**, not stale.
- **Isaac Perceptor supports fisheye/ftheta3/rational/plumbob with OpenCV intrinsics** — refuted 1-2. The exact accepted camera-model schema is unconfirmed.
- **puzzlepaint standalone transitive-license disclaimer** — refuted 1-2 (retained only as a flag to verify OpenGV/SuiteSparse).
- **Kornia "only pinhole implemented / immature"** — refuted 1-2. Current Kornia ships Brown-Conrady, KB-K3, orthographic (stale doc note).
- **Theseus LM "does not support damping, all constraints soft"** — refuted 0-3. It does support adaptive damping.
- **Graphite numbers tied to "calibration-relevant memory budgets"** — refuted 1-2 (benchmarks are SfM-scale, not calibration-sized).
- **MegBA exact speedup multipliers (41.45×/64.576×/6.769×)** — split/over-claimed; treat as SfM-scale, context-bound.
- **OSCalib "uncertainty-aware covariance adjustment"** — refuted 0-3. Do not attribute formal uncertainty handling to OSCalib.

---

<a name="open-questions--re-check-list"></a>
## Open questions / re-check list (next research + spike backlog)

1. **Calibration-regime GPU benchmark** — how do Graphite/PyPose/MegBA perform on few-camera-many-residual problems (vs the SfM-scale benchmarks they all report)? *No source measured this.* → `SPIKES.md`.
2. **Isaac Perceptor's exact accepted camera/distortion models** and coefficient conventions.
3. **MegBA's true 2026 maintenance status** (the "stale" claim was refuted, leaving it unknown).
4. **DS/EUCM differentiable GPU implementations** — does any 2023-2026 lib provide them, or hand-implement on nvTorchCam primitives?
5. **Sophus/manif vs Theseus/LieTorch** for SE(3)/SO(3) with GPU/differentiability; is there a maintained PyTorch binding to prefer?
6. **VPI CUDA-backend LDC throughput vs CV-CUDA / hand-rolled Remap** on x86_64 server.
7. **Controlled drift-tracking under vibration/thermal** — no surveyed targetless method publishes one; it is an open contribution.
8. **Re-confirm time-sensitive facts** before relying: PyPose/VPI/TensorRT versions; nvTorchCam/iKalibr/OpenCalib licenses.
