#pragma once
//
// CalibForge pipeline — cam-IMU rotation extrinsic + time-offset (header-only, CPU).
//
// Estimates R_ci (camera->IMU rotation) and the temporal offset t_d by matching the IMU gyro
// to the camera-derived angular velocity: omega_imu(t) = R_ci * omega_cam(t - t_d). This is
// the observable spatio-temporal initialization; full IMU preintegration (bias/gravity/
// position) follows. The camera angular-velocity signal must outlive the solve.

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/cam_imu_gyro_residual.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/manifold.hpp"
#include "sophus/so3.hpp"

namespace calibforge {

struct CamImuResult {
  Sophus::SO3d R_ci;       // camera -> IMU rotation
  double time_offset = 0;  // t_d
  LmSummary summary;
  Eigen::MatrixXd information;  // tangent J^T J over [R_ci (3), t_d (1)]
  int num_residuals = 0;
};

inline CamImuResult calibrateCamImuRotation(const AngularVelocitySignal& cam_omega,
                                            const std::vector<double>& imu_times,
                                            const std::vector<Eigen::Vector3d>& imu_omega,
                                            const Sophus::SO3d& R_ci_init, double t_d_init,
                                            const LmOptions& opts = LmOptions{}) {
  std::array<double, 4> R{};
  std::array<double, 1> td{t_d_init};
  SO3Param::store(R_ci_init, R.data());

  DenseProblem problem;
  problem.addParameterBlock(R.data(), std::make_shared<SO3Param>());
  problem.addParameterBlock(td.data(), std::make_shared<EuclideanParam>(1));
  for (std::size_t k = 0; k < imu_times.size() && k < imu_omega.size(); ++k) {
    problem.addResidualBlock(
        std::make_unique<CamImuGyroResidual>(&cam_omega, imu_times[k], imu_omega[k]),
        {R.data(), td.data()});
  }

  const LmSummary s = problem.solveLm(opts);

  CamImuResult res;
  res.R_ci = SO3Param::load(R.data());
  res.time_offset = td[0];
  res.summary = s;
  res.information = problem.informationMatrix();
  res.num_residuals = problem.numResiduals();
  return res;
}

}  // namespace calibforge
