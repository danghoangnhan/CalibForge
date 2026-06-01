#pragma once
//
// CalibForge apply — NVIDIA VPI LDC (Lens Distortion Correction) binding interface.
//
// On Jetson (CALIBFORGE_HAS_VPI defined by the build when the VPI SDK is found), the
// named-field VpiPolynomialCoeffs feed vpiWarpMapGenerateFromPolynomialLensDistortionModel
// and the PVA/VIC engines offload the remap, keeping the GPU free. PVA/VIC are JETSON-ONLY
// (CLAUDE.md rule 5); on x86_64 VPI runs only its CUDA backend. In every CPU/CI build the
// gate is OFF and generateWarpMap() (warp_map.hpp) is the parity reference the Jetson LDC
// table is later checked against. The real binding (including <vpi/...>) lives behind the
// gate and is intentionally absent here.

#include <array>

#include "calibforge/vpi_coeffs.hpp"
#include "calibforge/warp_map.hpp"

namespace calibforge {
namespace apply {

// True only when this build was configured against the Jetson VPI SDK.
inline bool vpiLdcAvailable() {
#ifdef CALIBFORGE_HAS_VPI
  return true;
#else
  return false;
#endif
}

#ifdef CALIBFORGE_HAS_VPI
// Declared only when the SDK is present; defined in a gated .cpp that includes <vpi/...>.
WarpMap generateWarpMapVpi(const VpiPolynomialCoeffs& coeffs,
                           const std::array<double, 4>& out_K, int w, int h);
#endif

}  // namespace apply
}  // namespace calibforge
