# CalibForge — Dependency Policy (build-vs-borrow, mid-2026)

> Derived from [`RESEARCH.md`](./RESEARCH.md) (Theme 5, re-validated 2026-06-01). This is the enforceable table for code review. The root `LICENSE` / `NOTICE` are **owner-managed** — this file governs *what may be vendored* given that license.

## The vendoring rule

CalibForge is **open-source & free** (permissive — Apache-2.0 intended; see root `LICENSE`). Therefore:

- ✅ **Vendor / depend-on freely (with attribution):** Apache-2.0, BSD, MIT, MPL-2.0 dependencies.
- ⛔ **Reference-only — read the math, re-implement, do NOT copy code:** GPL / LGPL / AGPL dependencies, and any proprietary SDK whose redistribution terms we can't meet.
- ⚠️ **Permissive-but-watch-the-transitive-deps:** a BSD/MIT project that itself pulls GPL/LGPL (e.g. puzzlepaint via OpenGV/SuiteSparse) — vendor the first-party code only, replace the copyleft transitive deps.

Every new third-party dependency must be added to the table below **with its license and status** before it is introduced.

## Verdict table

| Dependency | Role | License | Status | Notes |
|---|---|---|---|---|
| **nvTorchCam** | Camera-model core | Apache-2.0 | ✅ **vendor/depend** | 8 models; add DS & EUCM ourselves |
| **PyPose** | GPU bundle adjustment (primary) | Apache-2.0 | ✅ **vendor/depend** | LM + FastTriggs; Jetson-capable |
| **Graphite** | GPU BA (tracked, edge/bf16) | MIT | ✅ **vendor/depend** (when matured) | WIP research prototype — pin a commit; not a hard dep yet |
| **Ceres** | CPU solver path + accuracy oracle | BSD-3 | ✅ **vendor/depend** | Fastest on small single calibrations |
| **MegBA** | GPU BA (server scale only) | Apache-2.0 | ✅ vendor (server build only) | NCCL2/NVLink; ❌ no Jetson |
| **CV-CUDA** | GPU image primitives | Apache-2.0 | ✅ **vendor/depend** | server *apply* path |
| **NVIDIA VPI** | Runtime undistort (LDC) | Proprietary SDK (apps free to redistribute) | ✅ depend (link) | PVA/VIC **Jetson-only**; CUDA backend on server |
| **TensorRT** | Learned components | OSS parts Apache-2.0; **core SDK proprietary (EULA)** | ⚠️ depend; **don't redistribute the runtime** | OSS plugins/parsers OK to vendor |
| **OpenCalib SurroundCameraCalib** | Online surround-view baseline | Apache-2.0 | ✅ **vendor/depend** | BEV photometric random-search |
| **iKalibr** | RS / cam-IMU continuous-time formulation | BSD-3 | ✅ **vendor/depend** | targetless multi-sensor |
| **Sophus** | SE(3)/SO(3) C++ core | MIT | ✅ **vendor/depend** | header-only |
| **manif** | SE(3)/SO(3) C++ core | MIT | ✅ **vendor/depend** | alternative to Sophus |
| **Theseus** | Differentiable optimizer / analytic-Jacobian patterns | MIT | ✅ **vendor/depend** | also a design reference |
| **LieTorch** | Lie-group autograd | BSD-3 | ⚠️ reference (CUDA build issues) | prefer Sophus/manif/Theseus |
| **OpenCV** (core/calib3d/imgproc) | Reference / validation; real-image detect + io interop | Apache-2.0 | ✅ **depend (optional)** | `find_package(OpenCV QUIET)` gate; absent on CI → header-only suite stays green; OpenCV-linked tests are gated. Oracle + io interop. |
| **Eigen** | Host-side math | MPL-2.0 | ✅ **vendor/depend** | |
| **pybind11** | Python bindings | BSD-3 | ✅ **vendor/depend** | |
| **basalt-headers** | DS/EUCM/KB C++ reference | BSD-3 (verify) | ✅ vendor (verify) | implement DS/EUCM against this |
| **puzzlepaint/camera_calibration** | Generic per-pixel B-spline model | own code BSD-3; deps OpenGV/SuiteSparse **GPL/LGPL** | ⚠️ **first-party code only** | replace copyleft transitive deps |
| **DeepLM** | GPU BA (design doc's option) | **GPLv3** | ⛔ **reference-only / oracle** | dropped — copyleft poisons a permissive library |
| **Kalibr** | cam-IMU calibration | GPL-family | ⛔ **reference-only** | read the math, re-implement |
| **OpenVINS** | Online VINS self-cal | GPL | ⛔ **reference-only** | most edge-suitable paradigm — re-implement |
| **MVIS** | Multi-visual-inertial cal | GPL | ⛔ **reference-only** | joint GS+RS — re-implement |
| **Ctrl-VIO** | RS-VIO online line-delay | GPL-3 | ⛔ **reference-only** | re-implement |

## Competitors (proprietary — align/watch, never borrow)
- **MSA Calibration Anywhere** (Main Street Autonomy) — calibration-as-a-service → Isaac Perceptor URDF. Closest scope overlap.
- **RidgeRun CUDA Camera Undistort** — $3,999 GStreamer plugin (undistort-only).
- **Tangram Vision Metrical** — commercial multi-sensor calibration.

## Re-check before relying (time-sensitive — see RESEARCH.md open questions)
Versions/licenses verified 2026-06-01: PyPose v0.9.5, VPI 4.0.7, TensorRT 10.16/11.0. Re-confirm nvTorchCam / iKalibr / OpenCalib licenses and MegBA's maintenance status at integration time.
