#pragma once
//
// CalibForge solve — cam-IMU spatial-extrinsic pose residual (header-only).
//
// Ties a MEASURED camera-in-world pose T_wc (from a prior target-based solve) to the IMU
// nav-state (R_wi, p_wi) through the full spatial extrinsic T_ic = (R_ci, p_ci):
//
//   T_wc = T_wi * T_ic   =>   R_wc = R_wi * R_ci,   p_wc = R_wi * p_ci + p_wi.
//
// R_ci is the camera->IMU rotation (same convention as cam_imu_gyro_residual.hpp:
// omega_imu = R_ci * omega_cam); p_ci is the camera origin expressed in the IMU frame —
// the spatial LEVER ARM that cam_imu_gyro_residual could not observe (rotation alone leaves
// p_ci in the null space). Stacked over keyframes with varied orientation it makes p_ci
// observable (hand-eye AX=XB translation), while the IMU preintegration factor supplies the
// metric scale / gravity / bias that pins p_wi. Together they are the calibrate_cam_imu_full
// pipeline (docs/RESEARCH.md Theme 2; the v0.4 follow-up that estimates the TRANSLATION
// extrinsic, not just R_ci).
//
// 6-dim residual: rows 0..2 rotation (SO(3) Log), rows 3..5 translation (world frame):
//   r_R = Log( R_wc_meas^T * R_wi * R_ci )
//   r_t = (R_wi * p_ci + p_wi) - p_wc_meas
//
// Parameter-block order (matches what calibrate_cam_imu_full assembles):
//   [0]: R_wi (SO3, 4 doubles via SO3Param)   — IMU orientation in world
//   [1]: p_wi (R^3)                            — IMU position in world
//   [2]: R_ci (SO3, 4 doubles via SO3Param)    — camera->IMU rotation (shared)
//   [3]: p_ci (R^3)                            — camera origin in IMU frame (shared lever arm)
//
// Analytic minimal Jacobians (right-perturbation tangent for the SO(3) blocks), FD-validated
// in tests/test_cam_imu_full.cpp.
//
// WEIGHTING: like every CalibForge ResidualBlock this emits RAW residuals (no sqrt-information) —
// rows 0..2 are radians, rows 3..5 are metres, and calibrate_cam_imu_full mixes this factor with
// the heterogeneous ImuPreintegrationFactorResidual at implicit unit relative weight. That matches
// the library-wide convention and is exact for the synthetic GT-zero validation, but on REAL noisy
// data the two factor families should be pre-whitened (scale rows by their sqrt-information) before
// the joint solve so neither dominates and the RULE #2 gate's covariance is properly metric.

#include <Eigen/Dense>

#include "calibforge/imu_preintegrator.hpp"  // so3RightJacobianInverse
#include "calibforge/manifold.hpp"            // SO3Param, skew3
#include "calibforge/residual_block.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace calibforge {

class CamImuPoseResidual : public ResidualBlock {
 public:
  // T_wc_meas: measured camera-in-world pose for this keyframe (camera->world). If your
  // measurement is world->camera (as SingleCameraResult::poses stores), pass its .inverse().
  explicit CamImuPoseResidual(const Sophus::SE3d& T_wc_meas)
      : R_wc_meas_(T_wc_meas.so3()), p_wc_meas_(T_wc_meas.translation()) {}

  ResidualType type() const override { return ResidualType::Reprojection; }
  std::size_t residualDim() const override { return 6; }

  void evaluate(double const* const* params, double* residual,
                double** jacobians) const override {
    const Sophus::SO3d R_wi = SO3Param::load(params[0]);
    const Eigen::Vector3d p_wi(params[1][0], params[1][1], params[1][2]);
    const Sophus::SO3d R_ci = SO3Param::load(params[2]);
    const Eigen::Vector3d p_ci(params[3][0], params[3][1], params[3][2]);

    const Eigen::Matrix3d Rwi = R_wi.matrix();
    const Eigen::Matrix3d Rci = R_ci.matrix();

    // r_R = Log(R_wc_meas^T * R_wi * R_ci);  r_t = R_wi p_ci + p_wi - p_wc_meas.
    const Sophus::SO3d M = R_wc_meas_.inverse() * R_wi * R_ci;
    const Eigen::Vector3d r_R = M.log();
    const Eigen::Vector3d r_t = Rwi * p_ci + p_wi - p_wc_meas_;

    for (int i = 0; i < 3; ++i) {
      residual[i] = r_R[i];
      residual[3 + i] = r_t[i];
    }
    if (!jacobians) return;

    const Eigen::Matrix3d JrInv = so3RightJacobianInverse(r_R);

    auto fill = [](double* J, int rows, int cols, const auto& Mx) {
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) J[r * cols + c] = Mx(r, c);
    };

    // [0] R_wi (3-tangent, right perturb R_wi <- R_wi exp(theta)):
    //   r_R:  R_wc^T R_wi exp(theta) R_ci = M exp(R_ci^T theta) => dr_R/dtheta = JrInv * R_ci^T.
    //   r_t:  R_wi exp(theta) p_ci => dr_t/dtheta = -R_wi [p_ci]_x.
    if (jacobians[0]) {
      Eigen::Matrix<double, 6, 3> J = Eigen::Matrix<double, 6, 3>::Zero();
      J.block<3, 3>(0, 0) = JrInv * Rci.transpose();
      J.block<3, 3>(3, 0) = -Rwi * skew3(p_ci);
      fill(jacobians[0], 6, 3, J);
    }
    // [1] p_wi (Euclidean): r_R independent; dr_t/dp_wi = I.
    if (jacobians[1]) {
      Eigen::Matrix<double, 6, 3> J = Eigen::Matrix<double, 6, 3>::Zero();
      J.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
      fill(jacobians[1], 6, 3, J);
    }
    // [2] R_ci (3-tangent, right perturb R_ci <- R_ci exp(eta)):
    //   r_R:  M exp(eta) => dr_R/deta = JrInv;  r_t independent of R_ci.
    if (jacobians[2]) {
      Eigen::Matrix<double, 6, 3> J = Eigen::Matrix<double, 6, 3>::Zero();
      J.block<3, 3>(0, 0) = JrInv;
      fill(jacobians[2], 6, 3, J);
    }
    // [3] p_ci (Euclidean): r_R independent; dr_t/dp_ci = R_wi.
    if (jacobians[3]) {
      Eigen::Matrix<double, 6, 3> J = Eigen::Matrix<double, 6, 3>::Zero();
      J.block<3, 3>(3, 0) = Rwi;
      fill(jacobians[3], 6, 3, J);
    }
  }

 private:
  Sophus::SO3d R_wc_meas_;
  Eigen::Vector3d p_wc_meas_;
};

}  // namespace calibforge
