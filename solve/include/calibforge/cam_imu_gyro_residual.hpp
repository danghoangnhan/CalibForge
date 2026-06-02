#pragma once
//
// CalibForge solve — cam-IMU rotation extrinsic + time-offset residual (angular-velocity
// correspondence). The IMU gyro and the camera-derived angular velocity describe the same
// rigid body up to the spatial rotation R_ci and a temporal offset t_d:
//   omega_imu(t) = R_ci * omega_cam(t - t_d).
// This is the observable initialization for cam-IMU calibration; full IMU preintegration
// (bias / gravity / position) is a deeper follow-up (docs/RESEARCH.md Theme 2, re-implement
// iKalibr's BSD-3 formulation).
//
// Connects [R_ci (SO3), t_d (R^1)]. The camera angular velocity is a shared, time-sampled
// signal, linearly interpolated at tau = t_k - t_d. Residual r = R_ci * omega_cam(tau) - omega_imu.
// Analytic Jacobians: d/dR_ci = -R_ci [omega_cam]_x ;  d/dt_d = -R_ci * (d omega_cam/d tau).

#include <vector>

#include <Eigen/Dense>

#include "calibforge/manifold.hpp"
#include "calibforge/residual_block.hpp"
#include "sophus/so3.hpp"

namespace calibforge {

// Shared camera angular-velocity signal: strictly increasing times + samples.
struct AngularVelocitySignal {
  std::vector<double> times;
  std::vector<Eigen::Vector3d> omega;

  // Linear interpolation at tau (clamped to the sample range); also returns the segment slope.
  void sample(double tau, Eigen::Vector3d& w, Eigen::Vector3d& slope) const {
    const int n = static_cast<int>(times.size());
    if (n == 0) { w.setZero(); slope.setZero(); return; }
    if (tau <= times.front()) { w = omega.front(); slope.setZero(); return; }
    if (tau >= times.back()) { w = omega.back(); slope.setZero(); return; }
    int i = 0;
    while (i + 1 < n && times[i + 1] < tau) ++i;  // segment [i, i+1] contains tau
    const double dt = times[i + 1] - times[i];
    slope = (omega[i + 1] - omega[i]) / dt;
    w = omega[i] + slope * (tau - times[i]);
  }
};

class CamImuGyroResidual : public ResidualBlock {
 public:
  CamImuGyroResidual(const AngularVelocitySignal* cam, double imu_time,
                     const Eigen::Vector3d& imu_omega)
      : cam_(cam), tk_(imu_time), wimu_(imu_omega) {}

  ResidualType type() const override { return ResidualType::ImuPreintegration; }
  std::size_t residualDim() const override { return 3; }

  void evaluate(double const* const* params, double* residual,
                double** jacobians) const override {
    const Sophus::SO3d R = SO3Param::load(params[0]);  // R_ci
    const double td = params[1][0];
    Eigen::Vector3d wc, slope;
    cam_->sample(tk_ - td, wc, slope);
    const Eigen::Matrix3d Rm = R.matrix();
    const Eigen::Vector3d r = Rm * wc - wimu_;
    residual[0] = r[0];
    residual[1] = r[1];
    residual[2] = r[2];
    if (!jacobians) return;

    if (jacobians[0]) {  // 3 x 3, d/dR_ci = -R_ci [wc]_x
      const Eigen::Matrix3d J = -Rm * skew(wc);
      for (int row = 0; row < 3; ++row)
        for (int c = 0; c < 3; ++c) jacobians[0][row * 3 + c] = J(row, c);
    }
    if (jacobians[1]) {  // 3 x 1, d/dt_d = -R_ci * d(wc)/d tau
      const Eigen::Vector3d g = -Rm * slope;
      jacobians[1][0] = g[0];
      jacobians[1][1] = g[1];
      jacobians[1][2] = g[2];
    }
  }

 private:
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return S;
  }

  const AngularVelocitySignal* cam_;
  double tk_;
  Eigen::Vector3d wimu_;
};

}  // namespace calibforge
