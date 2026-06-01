# CalibForge

**A unified, NVIDIA-accelerated geometric camera-calibration library that both *estimates* and *applies* calibration across pinhole, fisheye, and generic models — built once, deployed on both edge (Jetson) and server.**

> **Status:** 🚧 early — design + research complete, scaffold in place, implementation not yet started. This repository currently holds the design spec, a cited research synthesis, the dependency policy, and a module skeleton.

## Why it exists

Every building block of geometric calibration is public, but **no single library does all of it, GPU-accelerated, deployable on both edge and server.** Today you estimate parameters with one CPU tool (OpenCV, Kalibr), apply them with another (VPI), and re-implement when you move from a server to a Jetson. CalibForge is the missing integration: a camera-model-agnostic, differentiable calibration core with a GPU bundle-adjustment solver, that estimates *and* emits runtime undistortion artifacts, from one codebase.

## Capability matrix

| Capability | v0.x (MVP) | v1.0 |
|---|:---:|:---:|
| Pinhole intrinsics · Brown–Conrady distortion | ✅ | ✅ |
| Kannala–Brandt fisheye | ✅ | ✅ |
| Generic / per-pixel models | — | ✅ |
| Single-cam extrinsics (PnP) · Stereo + rectification | ✅ | ✅ |
| Multi-camera rig · Hand-eye | — | ✅ |
| Camera–IMU extrinsics (+ time offset) | — | ✅ |
| Rolling-shutter calibration | — | ✅ |
| Online / targetless recalibration | prototype | ✅ |
| GPU bundle-adjustment back-end · Runtime undistort (VPI export) | ✅ | ✅ |

Scope is strictly **geometric** (no photometric/color/structured-light in v1).

## Repository layout

```
docs/
  DESIGN.md         Vision + engineering spec v0.1 (owner intent)
  RESEARCH.md       Cited, adversarially-verified mid-2026 research + revised stack
  DEPENDENCIES.md   Build-vs-borrow table + the permissive-only vendoring rule
  SPIKES.md         Empirical dependency de-risking results
core/ detect/ solve/ pipelines/ apply/ io/ bindings/   Module skeleton (see CLAUDE.md)
CLAUDE.md           Architecture + the non-obvious rules that govern the code
```

## Design at a glance

- **Camera-model core** — adopt **nvTorchCam** (differentiable, GPU-batched), add double-sphere & EUCM.
- **Solver** — solver-agnostic interface: **Ceres** on CPU (fastest for single small calibrations + accuracy oracle), **PyPose** / **Graphite** on GPU (batched / large-rig / online). *GPU is not automatically faster — see CLAUDE.md rule #1.*
- **Apply** — **VPI LDC** on Jetson (PVA/VIC), CUDA/CV-CUDA on server.
- **Online/targetless** — **OpenCalib** BEV baseline + an **observability-gated confidence engine that refuses to emit ill-conditioned parameters** (the differentiator).
- **One CUDA source → server + Jetson** via the `-gencode` arch matrix.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full, cited rationale and [`docs/DESIGN.md`](docs/DESIGN.md) for the vision.

## Building

CUDA toolkit required. Multi-arch, single source:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="72;80;86;87;89;90"
cmake --build build -j
```

*(The current `CMakeLists.txt` is a skeleton that proves multi-arch configuration; library targets are not implemented yet.)*

## License

Open-source & free (permissive — Apache-2.0 intended). See the root `LICENSE` (owner-managed). Vendored dependencies must be permissive (Apache/BSD/MIT/MPL); GPL-family projects are reference-only. See [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
