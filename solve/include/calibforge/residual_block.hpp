#pragma once
//
// CalibForge solve — the pluggable residual-block contract (INTERFACE STUB).
//
// The unified bundle-adjustment cost is a sum of heterogeneous residual blocks.
// Theme 2 of docs/RESEARCH.md verified (25/25 claims) that all of these coexist
// in ONE nonlinear-least-squares / MAP cost:
//   reprojection + distortion + rolling-shutter readout + IMU preintegration
//   + line reprojection (Plucker) + planar/ground constraints.
//
// One optimizer, many residual types — implementations live in solve/src and
// pipelines/ assemble them.

#include <cstddef>
#include <vector>

namespace calibforge {

enum class ResidualType {
  Reprojection,     // point reprojection (camera model + pose)
  Distortion,       // distortion-parameter residual
  RollingShutter,   // RS readout: reprojection evaluated at the row-specific pose (t_r or line-delay)
  CamImuGyroInit,   // cam/IMU rotation-only initialization (gyro <-> camera angular velocity)
  ImuPreintegration,// on-manifold Forster IMU factor: DeltaR / Deltav / Deltap between nav-states
  Line,             // RS line reprojection via Plucker parameterization (RSL-BA)
  Planar,           // planar / ground constraint
};

// Abstract residual block. evaluate() fills the residual and, when requested,
// the analytic Jacobians w.r.t. each connected parameter block.
class ResidualBlock {
 public:
  virtual ~ResidualBlock() = default;

  virtual ResidualType type() const = 0;
  virtual std::size_t residualDim() const = 0;

  // params: pointers to each connected parameter block (poses, intrinsics, ...).
  // residual: output, size residualDim().
  // jacobians: optional; jacobians[i] (if non-null) is d(residual)/d(params[i]),
  // row-major, residualDim() x block_size(i). Null entry => not requested.
  virtual void evaluate(double const* const* params,
                        double* residual,
                        double** jacobians) const = 0;

  // Robust loss: prefer the GPU-stable FastTriggs corrector (PyPose) over Ceres'
  // 2nd-order Triggs correction. See docs/RESEARCH.md Theme 1.
};

}  // namespace calibforge
