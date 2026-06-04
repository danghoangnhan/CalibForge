# CalibForge — Dependency Policy (build-vs-borrow, mid-2026)

> Derived from [`RESEARCH.md`](./RESEARCH.md) (Theme 5, re-validated 2026-06-01). This is the enforceable table for code review. The root `LICENSE` / `NOTICE` are **owner-managed** — this file governs *what may be vendored* given that license.

**Implementation status (v0.5 in progress).** The "Status" column in the table below distinguishes between **policy** ("✅ vendor / depend") and **actual code state** ("(in tree)" if the dependency is already wired through CMake / used by the implementation). Anything without "(in tree)" remains permitted by policy but is not yet a build-time dependency.

## The vendoring rule

CalibForge is **open-source & free** (permissive — Apache-2.0 intended; see root `LICENSE`). Therefore:

- ✅ **Vendor / depend-on freely (with attribution):** Apache-2.0, BSD, MIT, MPL-2.0 dependencies.
- ⛔ **Reference-only — read the math, re-implement, do NOT copy code:** GPL / LGPL / AGPL dependencies, and any proprietary SDK whose redistribution terms we can't meet.
- ⚠️ **Permissive-but-watch-the-transitive-deps:** a BSD/MIT project that itself pulls GPL/LGPL (e.g. puzzlepaint via OpenGV/SuiteSparse) — vendor the first-party code only, replace the copyleft transitive deps.

Every new third-party dependency must be added to the table below **with its license and status** before it is introduced.

## Verdict table

| Dependency | Role | License | Status | Notes |
|---|---|---|---|---|
| **nvTorchCam** | Camera-model core | Apache-2.0 | ✅ **interface adopted** | nvTorchCam's project/unproject contract is the C++ `CameraModel` interface; CalibForge implements pinhole / Brown-Conrady / KB / DS / EUCM in-tree (no nvTorchCam binary dep) |
| **PyPose** | GPU bundle adjustment (primary) | Apache-2.0 | ✅ **vendor/depend** (CUDA host) | LM + FastTriggs; CPU spike validated (`SPIKES.md` §C); GPU wiring deferred to v1.0 |
| **Graphite** | GPU BA (tracked, edge/bf16) | MIT | ✅ **vendor/depend** (when matured) | WIP research prototype — pin a commit; not a hard dep yet |
| **Ceres** | CPU solver path + accuracy oracle | BSD-3 | ✅ **vendor/depend** (oracle) | Fastest on small single calibrations; CalibForge ships its own CPU `DenseProblem` (manifold LM + FastTriggs) which the v1.0 benchmark will compare against |
| **MegBA** | GPU BA (server scale only) | Apache-2.0 | ✅ vendor (server build only) | NCCL2/NVLink; ❌ no Jetson; deferred to v1.0 |
| **CV-CUDA** | GPU image primitives | Apache-2.0 | ✅ **vendor/depend** | server *apply* path; not yet wired (no CUDA host) |
| **NVIDIA VPI** | Runtime undistort (LDC) | Proprietary SDK (apps free to redistribute) | ✅ depend (link) (in tree, gated) | PVA/VIC **Jetson-only**; CUDA backend on server. `apply/vpi_ldc.hpp` + `apply/vpi_coeffs.hpp` ship as gated bindings; CPU warp-map is the always-on parity reference |
| **TensorRT** | Learned components | OSS parts Apache-2.0; **core SDK proprietary (EULA)** | ⚠️ depend; **don't redistribute the runtime** | OSS plugins/parsers OK to vendor; not yet used |
| **OpenCalib SurroundCameraCalib** | Online surround-view baseline | Apache-2.0 | ✅ **re-implemented in-tree** | BEV photometric random-search math re-implemented (`detect/bev_photometric.hpp`, not vendored); shipped |
| **iKalibr** | RS / cam-IMU continuous-time formulation | BSD-3 | ✅ **vendor/depend** (math re-implemented) | Forster preintegration math re-implemented in `solve/imu_preintegrator.hpp` + `imu_preintegration_residual.hpp` — never copied verbatim |
| **Sophus** | SE(3)/SO(3) C++ core | MIT | ✅ **vendor/depend** (in tree) | Header-only via FetchContent; v1.22.10 pinned in `CMakeLists.txt` |
| **manif** | SE(3)/SO(3) C++ core | MIT | ✅ **vendor/depend** | alternative to Sophus; not currently used |
| **Theseus** | Differentiable optimizer / analytic-Jacobian patterns | MIT | ✅ **vendor/depend** | also a design reference |
| **LieTorch** | Lie-group autograd | BSD-3 | ⚠️ reference (CUDA build issues) | prefer Sophus/manif/Theseus |
| **OpenCV** (core/calib3d/imgproc/aruco) | Reference / validation; real-image detect + io interop | Apache-2.0 | ✅ **depend (optional, in tree)** | `find_package(OpenCV QUIET)` gate; absent on CI's `build-test` job → header-only suite stays green; new `opencv` CI job installs `libopencv-dev` and runs the gated suite (ChArUco / AprilGrid detection + OpenCV YAML cross-check) |
| **NVIDIA cuda-samples** | CUDA build/project template + helper_cuda idioms | BSD-3 | ✅ **reference (pattern, in tree)** | `apply/` CUDA build follows the cuda-samples template (`find_package(CUDAToolkit)`, separable compilation, `cuda_std_17`, `-lineinfo`, checkCudaErrors-style checks); no files vendored |
| **cuBLAS / cuSOLVER** | Dense GPU linear algebra for the native LM solver | Proprietary NVIDIA (CUDA Toolkit; freely redistributable runtime) | ✅ **depend (link), in tree (CUDA host)** | `solve/src/cuda_linear_solver.cu` uses cuBLAS DSYRK/DGEMV (JᵀJ, Jᵀr) + cuSOLVER dense Cholesky (`potrf`/`potrs`); linked via `CUDA::cublas` / `CUDA::cusolver` (hard requirement when `CALIBFORGE_HAS_CUDA`). System libraries — **linked, never vendored**; not present on the CUDA-less CI path |
| **Eigen** | Host-side math | MPL-2.0 | ✅ **vendor/depend** (in tree) | Header-only via FetchContent; 3.4.0 pinned |
| **pybind11** | Python bindings | BSD-3 | ✅ **vendor/depend** (in tree) | FetchContent; v2.13.6 pinned; gated by `-DCALIBFORGE_PYTHON=ON`; `python` CI job |
| **basalt-headers** | DS/EUCM/KB C++ reference | BSD-3 (verify) | ✅ vendor (math re-implemented) | DS and EUCM models implemented in-tree against this and the Usenko paper; not vendored as code |
| **puzzlepaint/camera_calibration** | Generic per-pixel B-spline model | own code BSD-3; deps OpenGV/SuiteSparse **GPL/LGPL** | ⚠️ **math re-implemented in-tree** | model math re-implemented (`core/generic_bspline_camera.hpp`, no code copied; transitive copyleft deps not used); shipped, pipeline wiring pending |
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
