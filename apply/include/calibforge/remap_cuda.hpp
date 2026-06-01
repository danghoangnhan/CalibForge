#pragma once
//
// CalibForge apply — server-side GPU undistort/rectify (CV-CUDA / CUDA bilinear remap).
//
// VPI's PVA/VIC engines are Jetson-only (CLAUDE.md rule 5), so the SERVER apply path uses a
// CUDA backend instead. remapBilinearCuda has the same signature as the CPU reference
// (remap.hpp) so the two are parity-testable. The kernel lives in apply/src/remap_cuda.cu,
// compiled only when a CUDA toolkit is present (CALIBFORGE_HAS_CUDA), exactly like
// src/arch_probe.cu. On a CUDA-less host the declaration is absent and the parity test is
// compiled out, so the green suite is unaffected.

#include "calibforge/image.hpp"
#include "calibforge/warp_map.hpp"

namespace calibforge {
namespace apply {

inline bool cudaRemapAvailable() {
#ifdef CALIBFORGE_HAS_CUDA
  return true;
#else
  return false;
#endif
}

#ifdef CALIBFORGE_HAS_CUDA
// GPU bilinear remap; bit-comparable (+/-1 LSB) to remapBilinear() on the same inputs.
Image8 remapBilinearCuda(const Image8& input, const WarpMap& map);
#endif

}  // namespace apply
}  // namespace calibforge
