# CalibForge — Dependency Policy (build-vs-borrow, mid-2026)

> Derived from [`RESEARCH.md`](./RESEARCH.md) (Theme 5, re-validated 2026-06-01; licenses + integration form re-audited firsthand 2026-06-04). This is the enforceable table for code review. The root `LICENSE` / `NOTICE` are **owner-managed** — this file governs *what may be vendored / linked* given that license.

**Implementation status (v0.5 in progress).** The **Use-as-is?** column distinguishes **policy** from **actual code state** — `(in tree)` means already wired through CMake / used by the implementation. Anything without `(in tree)` is permitted by policy but not yet a build-time dependency.

## The two-axis rule (license × form)

CalibForge is **open-source & free** (permissive — Apache-2.0 intended) and is a **header-oriented C++17 library** linked into a single CUDA/C++ binary deployed on **edge (Jetson, aarch64)** and **server (x86_64 + CUDA)** — there is **no Python runtime on the edge target and it is not a ROS application.** So "can we use a dependency *as-is*, without re-implementing?" has **two** gates, not one:

1. **License** — must be permissive enough to vendor/link into an Apache-2.0 library:
   - ✅ **OK to vendor/link:** Apache-2.0, BSD-2/3/4, MIT, MPL-2.0, Zlib, Boost.
   - ⚠️ **Watch the transitive deps:** a permissive project that *pulls* GPL/LGPL (e.g. SuiteSparse/CHOLMOD, which Ceres/g2o/Kalibr/puzzlepaint all touch) — vendor/link the first-party only and **disable the copyleft dep at build time**.
   - ⛔ **Reference-only (read the math, never copy/link):** GPL / LGPL / AGPL, and proprietary SDKs we can't redistribute.
2. **Form** — must be linkable into a C++ edge+server binary: a **header-only** or **compiled C++/CUDA** library with a real C++ API. A **Python/PyTorch** package, a **ROS (catkin) application**, a **standalone CLI tool**, or a **Qt GUI app** is *not* linkable into the edge runtime **even if its license is permissive** — those must be re-implemented (or used only as an offline oracle).

> **Key finding of the 2026-06-04 audit:** for CalibForge, **form blocks reuse far more often than license does.** Of the candidates that must be re-implemented, almost all are *permissively licensed* — they're just Python/PyTorch, ROS, or GUI/CLI applications. Only **GPL** code (OpenVINS, MVIS, Ctrl-VIO, DeepLM) is blocked by license.

Every new third-party dependency must be added to the table below **with its license, form, and use-as-is verdict** before it is introduced.

**Use-as-is legend:** ✅ link/vendor as-is · ⚠️ usable with a build-flag caveat · ✋ re-implement (license is fine, *form* blocks linking) · ⛔ reference-only (copyleft) · ℹ️ pattern/reference only (not a library)

## Directly usable as-is (the short answer)

These are permissive **and** in a linkable C++ form — adoptable without re-implementing:

- **Solvers** — **Ceres** (CPU, build `-DSUITESPARSE=OFF`), **GTSAM** (CPU), **Graphite** (GPU, MIT), **MegBA** (GPU, Apache; stale). *(CalibForge ships its own `DenseProblem` + native CUDA solver by choice — model-agnostic, header-only, the benchmark oracle — but any of these could back `solve/`.)*
- **Camera models** — **basalt-headers** (BSD-3, header-only; DS/EUCM/KB). *The one clear drop-in CalibForge chose to re-implement instead.*
- **Lie groups** — **Sophus** (in tree), **manif**.
- **Apply (GPU image)** — **CV-CUDA** (Jetson-supported).
- **Detect / IO** — **OpenCV** (in tree, optional; disable FFmpeg).
- **GPU linear algebra** — **cuBLAS / cuSOLVER** (in tree; linked, never vendored).

Everything else is **✋ re-implement (form)** or **⛔ reference-only (license)** — see the table.

## Verdict table

| Dependency | Role | License | Form | Use-as-is? | Notes |
|---|---|---|---|:--:|---|
| **nvTorchCam** | Camera-model core | Apache-2.0 | Python/PyTorch | ✋ reimplement (form) | Project/unproject **interface** adopted; pinhole/BC/KB/DS/EUCM re-implemented in C++ (a PyTorch pkg can't link into the edge binary). `nvdiffrast` optional dep is NVIDIA non-commercial |
| **basalt-headers** | DS/EUCM/KB camera models | BSD-3 | **header-only C++** | ✅ **directly usable** (currently re-implemented) | Header-only, permissive, **no GPL** — a genuine drop-in. CalibForge re-implemented DS/EUCM/KB against it + the Usenko paper for a uniform `CameraModel` interface; revisit if upkeep cost rises |
| **Kornia** | diff. CV / camera models | Apache-2.0 | Python/PyTorch | ✋ reimplement (form) | Python-only; reference/oracle for model math |
| **Ceres** | CPU NLS solver + oracle | BSD-3 + Apache-2.0 | C++ lib | ⚠️ link, `-DSUITESPARSE=OFF` | Directly usable; use Eigen-sparse to avoid GPL CHOLMOD/SPQR. CalibForge ships its own `DenseProblem` (model-agnostic, header-only) and benchmarks against Ceres |
| **GTSAM** | factor-graph solver (alt) | BSD-3 | C++ lib | ✅ **link/vendor** | Directly usable CPU backend; deps permissive (`EIGEN_MPL2_ONLY`); no CUDA |
| **g2o** | graph optimization (alt) | BSD-2 core; **GPL-3 parts** | C++ lib | ⚠️ core only | Link the BSD core (dense/Eigen); **skip** GPL `g2o_viewer`/`g2o_incremental` and SuiteSparse/CHOLMOD |
| **PyPose** | GPU BA / LM | Apache-2.0 | Python/PyTorch | ✋ reimplement (form) | LM + FastTriggs validated in CPU spike (§C); not linkable — re-implement in C++/CUDA, or use as offline oracle |
| **Graphite** | GPU BA (edge/bf16) | MIT | CUDA/C++ lib | ✅ **link/vendor** (when matured) | Directly usable GPU NLS; needs CUDA 12+ & NVIDIA cuDSS; WIP prototype (ICRA 2026) — pin a commit |
| **MegBA** | GPU BA (server) | Apache-2.0 | CUDA/C++ lib | ✅ **vendor/link** | Directly usable; NCCL optional; **stale (~2023)** — verify maintenance before adopting |
| **Theseus** | diff. optimizer / Jacobian patterns | MIT | Python/PyTorch | ✋ reimplement (form) | Design reference for analytic Jacobians; PyTorch — not linkable; optional CHOLMOD-GPL via scikit-sparse |
| **cuBLAS / cuSOLVER** | dense GPU linear algebra (native solver) | Proprietary NVIDIA (redistributable runtime) | NVIDIA SDK | ✅ **link (in tree)** | `solve/src/cuda_linear_solver.cu`: SYRK/GEMV + dense Cholesky via `CUDA::cublas`/`CUDA::cusolver` (hard-required under `CALIBFORGE_HAS_CUDA`). Linked, never vendored |
| **Sophus** | SE(3)/SO(3) Lie core | MIT | header-only C++ | ✅ **vendor (in tree)** | FetchContent, v1.22.10 pinned; Eigen-only (fmt disabled) |
| **manif** | SE(3)/SO(3) Lie (alt) | MIT | header-only C++ | ✅ **link/vendor** | Alternative to Sophus; not currently used |
| **LieTorch** | Lie autograd | BSD-3 | Python/PyTorch (CUDA ext) | ✋ reimplement / reference | Prefer Sophus/manif/Theseus; CUDA build issues |
| **iKalibr** | RS / cam-IMU continuous-time | BSD-3 + Apache-2.0 | **ROS (catkin) pkg** | ✋ reimplement (form) | Permissive but a **ROS application** (nodes/msgs) — not linkable. Forster preintegration math re-implemented in `solve/imu_preintegrator.hpp` + residual; never copied. *(was wrongly tagged "vendor/depend")* |
| **Kalibr** | cam-IMU / RS toolbox | **BSD-3** (corrected — *not* GPL) | **ROS1 pkg** + GPL-transitive (SuiteSparse) | ✋ reimplement (form + transitive) | **Corrected 2026-06-04:** first-party is BSD-3, not GPL. But it's a ROS1 app whose solver pulls GPL CHOLMOD ⇒ reference / re-implement, not vendor. Running it offline as an **oracle** is license-clean |
| **OpenVINS** | online VINS self-cal | **GPLv3** | C++ lib | ⛔ reference-only (copyleft) | Clean C++ lib + permissive deps, but GPLv3 poisons Apache-2.0 — re-implement the MSCKF math; offline oracle OK |
| **MVIS** | multi-visual-inertial cal | GPL | ROS / C++ | ⛔ reference-only | joint GS+RS — re-implement |
| **Ctrl-VIO** | RS-VIO online line-delay | GPL-3 | ROS / C++ | ⛔ reference-only | re-implement |
| **DeepLM** | GPU BA (design option) | **GPLv3** | Python / CUDA | ⛔ reference-only | dropped — copyleft poisons a permissive library |
| **OpenCalib** (SensorsCalibration) | online surround-view baseline | Apache-2.0 | **standalone CLI tools** | ✋ reimplement (form) | Bag of executables + bundled ceres/g2o (GPL-transitive). BEV photometric random-search math re-implemented (`detect/bev_photometric.hpp`); shipped |
| **puzzlepaint/camera_calibration** | generic per-pixel B-spline model | first-party BSD-3; deps SuiteSparse **GPL** | **Qt GUI app** | ✋ reimplement (form + transitive) | Qt GUI + SuiteSparse GPL; author says "don't reuse libvis". Model math re-implemented (`core/generic_bspline_camera.hpp`); shipped, pipeline wiring pending |
| **CV-CUDA** | GPU image primitives (apply path) | Apache-2.0 | CUDA/C++ lib | ✅ **link/vendor** | Directly usable; official Jetson/aarch64 support; Python bindings optional (off). Not yet wired |
| **NVIDIA VPI** | runtime undistort (LDC) | Proprietary SDK (apps free to redistribute) | NVIDIA SDK | ⚠️ link (gated, in tree) | PVA/VIC **Jetson-only**; CUDA backend on server. `apply/vpi_ldc.hpp` + `apply/vpi_coeffs.hpp` gated; CPU warp-map is the always-on parity reference |
| **TensorRT** | learned components | OSS parts Apache-2.0; core SDK proprietary (EULA) | NVIDIA SDK | ⚠️ link; don't redistribute runtime | OSS plugins/parsers OK to vendor; not yet used |
| **OpenCV** (core/calib3d/imgproc/aruco) | detect (ChArUco/AprilGrid) + IO interop + validation | Apache-2.0 (4.5.0+) | C++ lib | ✅ **depend (optional, in tree)** | `find_package(OpenCV QUIET)` gate; absent on CI's `build-test` job → header-only suite stays green; `opencv` CI job runs the gated suite. **Disable FFmpeg (GPL) at build** |
| **Eigen** | host-side math | MPL-2.0 | header-only C++ | ✅ **vendor (in tree)** | FetchContent; 3.4.0 pinned |
| **pybind11** | Python bindings | BSD-3 | header-only C++ | ✅ **vendor (in tree)** | FetchContent; v2.13.6 pinned; gated by `-DCALIBFORGE_PYTHON=ON`; `python` CI job |
| **NVIDIA cuda-samples** | CUDA build/project template | BSD-3 | reference pattern | ℹ️ pattern (in tree) | `apply/`/`solve/` CUDA builds follow the template (`find_package(CUDAToolkit)`, separable compilation, `cuda_std_17`, `-lineinfo`); no files vendored |

## Competitors (proprietary — align/watch, never borrow)
- **MSA Calibration Anywhere** (Main Street Autonomy) — calibration-as-a-service → Isaac Perceptor URDF. Closest scope overlap.
- **RidgeRun CUDA Camera Undistort** — $3,999 GStreamer plugin (undistort-only).
- **Tangram Vision Metrical** — commercial multi-sensor calibration.

## Re-check before relying (time-sensitive — see RESEARCH.md open questions)
Licenses + integration form re-audited firsthand **2026-06-04** (each repo's actual LICENSE file + form/transitive deps). Notable corrections from that audit: **Kalibr is BSD-3, not GPL** (blocked by ROS form + GPL-transitive SuiteSparse, not by its own license); **iKalibr** is re-implemented because it is a **ROS package**, not because of license. Versions previously verified 2026-06-01: PyPose v0.9.5, VPI 4.0.7, TensorRT 10.16/11.0. Re-confirm MegBA's maintenance status (stale ~2023) and Graphite's API stability at integration time.
