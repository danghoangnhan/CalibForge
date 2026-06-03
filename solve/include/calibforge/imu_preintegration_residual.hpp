#pragma once
//
// CalibForge solve — Forster IMU preintegration factor residual (header-only).
//
// 9-dim residual that ties two consecutive nav-states (R_i, p_i, v_i) and (R_j, p_j, v_j)
// through a preintegrated IMU window (DeltaR, Deltav, Deltap accumulated at a nominal bias),
// the current (b_g, b_a) bias estimate, and the gravity vector g_w (world frame).
//
// Forster IJRR 2017 eqs. (37–39):
//   r_R = Log( ΔR_corr.transpose() * R_i.transpose() * R_j )
//   r_v = R_i.transpose() * (v_j - v_i - g_w * Δt) - (Δv + dv/dbg * δb_g + dv/dba * δb_a)
//   r_p = R_i.transpose() * (p_j - p_i - v_i Δt - 0.5 g_w Δt²) - (Δp + dp/dbg * δb_g + dp/dba * δb_a)
// where ΔR_corr = ΔR̃ * Exp(dR/dbg * δb_g),  δb_g = b_g - b_g_nom,  δb_a = b_a - b_a_nom.
//
// Parameter-block order (matches what calibrate_cam_imu_full assembles):
//   [0]: R_i  (SO3, 4 doubles via SO3Param)
//   [1]: p_i  (R^3)
//   [2]: v_i  (R^3)
//   [3]: R_j  (SO3, 4 doubles via SO3Param)
//   [4]: p_j  (R^3)
//   [5]: v_j  (R^3)
//   [6]: b_g  (R^3)
//   [7]: b_a  (R^3)
//   [8]: g_w  (R^3 — usually held constant via setParameterBlockConstant)
//
// Jacobians are MINIMAL (right-perturbation tangent for R_i, R_j; ambient for the
// Euclidean blocks), FD-validated in tests/test_imu_preintegration_residual.cpp.

#include <Eigen/Dense>

#include "calibforge/imu_preintegrator.hpp"
#include "calibforge/manifold.hpp"          // SO3Param, skew3
#include "calibforge/residual_block.hpp"
#include "sophus/so3.hpp"

namespace calibforge {

class ImuPreintegrationFactorResidual : public ResidualBlock {
 public:
  // Borrows a const pointer to a preintegrator that lives at least as long as this block.
  explicit ImuPreintegrationFactorResidual(const ImuPreintegrator* preint)
      : preint_(preint) {}

  ResidualType type() const override { return ResidualType::ImuPreintegration; }
  std::size_t residualDim() const override { return 9; }

  void evaluate(double const* const* params, double* residual,
                double** jacobians) const override {
    // Load parameter blocks.
    const Sophus::SO3d R_i = SO3Param::load(params[0]);
    const Eigen::Vector3d p_i(params[1][0], params[1][1], params[1][2]);
    const Eigen::Vector3d v_i(params[2][0], params[2][1], params[2][2]);
    const Sophus::SO3d R_j = SO3Param::load(params[3]);
    const Eigen::Vector3d p_j(params[4][0], params[4][1], params[4][2]);
    const Eigen::Vector3d v_j(params[5][0], params[5][1], params[5][2]);
    const Eigen::Vector3d b_g(params[6][0], params[6][1], params[6][2]);
    const Eigen::Vector3d b_a(params[7][0], params[7][1], params[7][2]);
    const Eigen::Vector3d g_w(params[8][0], params[8][1], params[8][2]);

    const Eigen::Vector3d dbg = b_g - preint_->bgNom();
    const Eigen::Vector3d dba = b_a - preint_->baNom();
    const double dT = preint_->deltaT();
    const double dT2 = dT * dT;

    // Bias-corrected preintegration (Forster eq. (44)).
    const Eigen::Vector3d phi_corr = preint_->dR_dbg() * dbg;
    const Sophus::SO3d dR_corr = preint_->deltaR() * Sophus::SO3d::exp(phi_corr);
    const Eigen::Vector3d dv_corr = preint_->deltav() + preint_->dv_dbg() * dbg
                                   + preint_->dv_dba() * dba;
    const Eigen::Vector3d dp_corr = preint_->deltap() + preint_->dp_dbg() * dbg
                                   + preint_->dp_dba() * dba;

    const Eigen::Matrix3d Rit = R_i.matrix().transpose();
    const Eigen::Vector3d term_v = v_j - v_i - g_w * dT;
    const Eigen::Vector3d term_p = p_j - p_i - v_i * dT - 0.5 * g_w * dT2;

    const Sophus::SO3d A = dR_corr.inverse() * R_i.inverse() * R_j;
    const Eigen::Vector3d r_R = A.log();
    const Eigen::Vector3d r_v = Rit * term_v - dv_corr;
    const Eigen::Vector3d r_p = Rit * term_p - dp_corr;

    for (int i = 0; i < 3; ++i) {
      residual[i]     = r_R[i];
      residual[3 + i] = r_v[i];
      residual[6 + i] = r_p[i];
    }
    if (!jacobians) return;

    // Cached pieces shared across Jacobian blocks.
    const Eigen::Matrix3d JrInv = so3RightJacobianInverse(r_R);
    const Eigen::Matrix3d A_mat = A.matrix();

    auto fill = [](double* J, int rows, int cols, const auto& M) {
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) J[r * cols + c] = M(r, c);
    };

    // [0] R_i (3-tangent). Rows 0..2: dr_R/dR_i = -JrInv * (R_j^T R_i).
    //                       Rows 3..5: dr_v/dR_i = skew(R_i^T (v_j-v_i-g dt)).
    //                       Rows 6..8: dr_p/dR_i = skew(R_i^T (p_j-p_i-v_i dt - 0.5 g dt^2)).
    if (jacobians[0]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(0, 0) = -JrInv * (R_j.matrix().transpose() * R_i.matrix());
      J.block<3, 3>(3, 0) = skew3(Rit * term_v);
      J.block<3, 3>(6, 0) = skew3(Rit * term_p);
      fill(jacobians[0], 9, 3, J);
    }

    // [1] p_i (Euclidean). dr_p/dp_i = -R_i^T; r_R, r_v independent.
    if (jacobians[1]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(6, 0) = -Rit;
      fill(jacobians[1], 9, 3, J);
    }

    // [2] v_i (Euclidean). dr_v/dv_i = -R_i^T; dr_p/dv_i = -R_i^T * dt; r_R independent.
    if (jacobians[2]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(3, 0) = -Rit;
      J.block<3, 3>(6, 0) = -Rit * dT;
      fill(jacobians[2], 9, 3, J);
    }

    // [3] R_j (3-tangent). dr_R/dR_j = JrInv; r_v / r_p independent.
    if (jacobians[3]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(0, 0) = JrInv;
      fill(jacobians[3], 9, 3, J);
    }

    // [4] p_j (Euclidean). dr_p/dp_j = R_i^T.
    if (jacobians[4]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(6, 0) = Rit;
      fill(jacobians[4], 9, 3, J);
    }

    // [5] v_j (Euclidean). dr_v/dv_j = R_i^T.
    if (jacobians[5]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(3, 0) = Rit;
      fill(jacobians[5], 9, 3, J);
    }

    // [6] b_g (Euclidean). dr_R/dbg = -JrInv * A^T * J_r(phi_corr) * dR_dbg.
    //                       dr_v/dbg = -dv_dbg.
    //                       dr_p/dbg = -dp_dbg.
    if (jacobians[6]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      const Eigen::Matrix3d Jr_phi = so3RightJacobian(phi_corr);
      J.block<3, 3>(0, 0) = -JrInv * A_mat.transpose() * Jr_phi * preint_->dR_dbg();
      J.block<3, 3>(3, 0) = -preint_->dv_dbg();
      J.block<3, 3>(6, 0) = -preint_->dp_dbg();
      fill(jacobians[6], 9, 3, J);
    }

    // [7] b_a (Euclidean). dr_v/dba = -dv_dba; dr_p/dba = -dp_dba; r_R independent.
    if (jacobians[7]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(3, 0) = -preint_->dv_dba();
      J.block<3, 3>(6, 0) = -preint_->dp_dba();
      fill(jacobians[7], 9, 3, J);
    }

    // [8] g_w (Euclidean). dr_v/dg = -R_i^T * dt; dr_p/dg = -0.5 * R_i^T * dt^2.
    if (jacobians[8]) {
      Eigen::Matrix<double, 9, 3> J = Eigen::Matrix<double, 9, 3>::Zero();
      J.block<3, 3>(3, 0) = -Rit * dT;
      J.block<3, 3>(6, 0) = -0.5 * Rit * dT2;
      fill(jacobians[8], 9, 3, J);
    }
  }

 private:
  const ImuPreintegrator* preint_;
};

}  // namespace calibforge
