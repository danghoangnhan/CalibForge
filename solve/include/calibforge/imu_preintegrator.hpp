#pragma once
//
// CalibForge solve — IMU preintegration (Forster et al. 2017) (header-only).
//
// Accumulates IMU samples between two keyframes into the on-manifold quantities
//   DeltaR_ij, Deltav_ij, Deltap_ij
// computed at a fixed nominal bias (b_g_nom, b_a_nom) provided at construction. When the bias
// estimate changes during optimization we DO NOT re-integrate the raw samples — we apply a
// first-order correction via the accumulated bias-Jacobians:
//   dDeltaR/d(b_g),   dDeltav/d(b_g),   dDeltav/d(b_a),   dDeltap/d(b_g),   dDeltap/d(b_a)
// (Forster IJRR 2017 eqs. (A.7-A.10).) These are what makes the preintegration "re-usable"
// across solver iterations — the differentiator vs naive re-integration.
//
// Reference math only — re-implements Forster's BSD-3-compatible formulation; never copies
// GPL Kalibr / Ctrl-VIO source (docs/RESEARCH.md Theme 2, CLAUDE.md rule 3).

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/manifold.hpp"  // skew3
#include "sophus/so3.hpp"

namespace calibforge {

// Right Jacobian of SO(3) at tangent phi. J_r(phi) = I when phi=0; for small phi the
// closed-form series-expansion below stays smooth across the singularity at |phi|=0.
inline Eigen::Matrix3d so3RightJacobian(const Eigen::Vector3d& phi) {
  const double th2 = phi.squaredNorm();
  const double th = std::sqrt(th2);
  if (th < 1e-7) {
    // Series: J_r(phi) ≈ I - 0.5 [phi]_x + (1/6) [phi]_x^2
    return Eigen::Matrix3d::Identity() - 0.5 * skew3(phi)
           + (1.0 / 6.0) * (skew3(phi) * skew3(phi));
  }
  const Eigen::Matrix3d K = skew3(phi);
  return Eigen::Matrix3d::Identity()
       - ((1.0 - std::cos(th)) / th2) * K
       + ((th - std::sin(th)) / (th2 * th)) * (K * K);
}

// Inverse of the right Jacobian.
inline Eigen::Matrix3d so3RightJacobianInverse(const Eigen::Vector3d& phi) {
  const double th2 = phi.squaredNorm();
  const double th = std::sqrt(th2);
  if (th < 1e-7) {
    return Eigen::Matrix3d::Identity() + 0.5 * skew3(phi)
           + (1.0 / 12.0) * (skew3(phi) * skew3(phi));
  }
  const Eigen::Matrix3d K = skew3(phi);
  const double half = 0.5;
  const double a = 1.0 / th2 - (1.0 + std::cos(th)) / (2.0 * th * std::sin(th));
  return Eigen::Matrix3d::Identity() + half * K + a * (K * K);
}

// Forster on-manifold IMU preintegrator. Construct with the bias at the time integration
// begins; addSample() accumulates one IMU measurement (gyro + accel, in body frame, dt > 0).
// At any time the accumulated DeltaR / Deltav / Deltap + bias Jacobians are available via
// the accessor methods. Designed for use inside ImuPreintegrationFactorResidual.
class ImuPreintegrator {
 public:
  ImuPreintegrator(const Eigen::Vector3d& b_g_nom, const Eigen::Vector3d& b_a_nom)
      : b_g_nom_(b_g_nom), b_a_nom_(b_a_nom),
        deltaR_(Sophus::SO3d()), deltav_(Eigen::Vector3d::Zero()),
        deltap_(Eigen::Vector3d::Zero()),
        dR_dbg_(Eigen::Matrix3d::Zero()),
        dv_dbg_(Eigen::Matrix3d::Zero()), dv_dba_(Eigen::Matrix3d::Zero()),
        dp_dbg_(Eigen::Matrix3d::Zero()), dp_dba_(Eigen::Matrix3d::Zero()),
        delta_t_(0.0) {}

  // Add one IMU sample. (w_meas, a_meas) are the raw gyro/accel reading at the start of the
  // interval dt. The Forster mid-point form trades a tiny bit of accuracy for code that's
  // bias-Jacobian-clean; here we use the simpler forward-Euler form (Forster eq. (35-36))
  // which matches the bias-Jacobian update eqs. (A.7-A.10) directly.
  void addSample(const Eigen::Vector3d& w_meas, const Eigen::Vector3d& a_meas, double dt) {
    const Eigen::Vector3d w_hat = w_meas - b_g_nom_;
    const Eigen::Vector3d a_hat = a_meas - b_a_nom_;
    const Eigen::Vector3d phi = w_hat * dt;
    const Sophus::SO3d dR = Sophus::SO3d::exp(phi);
    const Eigen::Matrix3d Rm = deltaR_.matrix();
    const Eigen::Matrix3d ak_skew = skew3(a_hat);
    const Eigen::Matrix3d Jr = so3RightJacobian(phi);

    // Bias Jacobians — propagate FIRST, before mutating DeltaR/Deltav/Deltap (Forster A.7-A.10).
    const Eigen::Matrix3d dR_dbg_next = dR.matrix().transpose() * dR_dbg_ - Jr * dt;
    const Eigen::Matrix3d dv_dbg_next = dv_dbg_ - Rm * ak_skew * dR_dbg_ * dt;
    const Eigen::Matrix3d dv_dba_next = dv_dba_ - Rm * dt;
    const Eigen::Matrix3d dp_dbg_next = dp_dbg_ + dv_dbg_ * dt
                                       - 0.5 * Rm * ak_skew * dR_dbg_ * dt * dt;
    const Eigen::Matrix3d dp_dba_next = dp_dba_ + dv_dba_ * dt - 0.5 * Rm * dt * dt;

    // State update (forward-Euler integration).
    deltap_ += deltav_ * dt + 0.5 * Rm * a_hat * dt * dt;
    deltav_ += Rm * a_hat * dt;
    deltaR_ = deltaR_ * dR;

    dR_dbg_ = dR_dbg_next;
    dv_dbg_ = dv_dbg_next;
    dv_dba_ = dv_dba_next;
    dp_dbg_ = dp_dbg_next;
    dp_dba_ = dp_dba_next;

    delta_t_ += dt;
  }

  const Eigen::Vector3d& bgNom() const { return b_g_nom_; }
  const Eigen::Vector3d& baNom() const { return b_a_nom_; }
  const Sophus::SO3d& deltaR() const { return deltaR_; }
  const Eigen::Vector3d& deltav() const { return deltav_; }
  const Eigen::Vector3d& deltap() const { return deltap_; }
  double deltaT() const { return delta_t_; }

  const Eigen::Matrix3d& dR_dbg() const { return dR_dbg_; }
  const Eigen::Matrix3d& dv_dbg() const { return dv_dbg_; }
  const Eigen::Matrix3d& dv_dba() const { return dv_dba_; }
  const Eigen::Matrix3d& dp_dbg() const { return dp_dbg_; }
  const Eigen::Matrix3d& dp_dba() const { return dp_dba_; }

 private:
  Eigen::Vector3d b_g_nom_;
  Eigen::Vector3d b_a_nom_;

  Sophus::SO3d deltaR_;
  Eigen::Vector3d deltav_;
  Eigen::Vector3d deltap_;

  // First-order bias-correction Jacobians (Forster A.7-A.10).
  Eigen::Matrix3d dR_dbg_;
  Eigen::Matrix3d dv_dbg_;
  Eigen::Matrix3d dv_dba_;
  Eigen::Matrix3d dp_dbg_;
  Eigen::Matrix3d dp_dba_;

  double delta_t_;
};

}  // namespace calibforge
