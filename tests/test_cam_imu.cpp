#include "calibforge/calibrate_cam_imu.hpp"

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/cam_imu_gyro_residual.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/observability.hpp"
#include "cf_test.hpp"
#include "sophus/so3.hpp"

using calibforge::AngularVelocitySignal;
using calibforge::assessObservability;
using calibforge::calibrateCamImuRotation;
using calibforge::CamImuGyroResidual;
using calibforge::CamImuResult;
using calibforge::LmOptions;
using calibforge::SO3Param;

static AngularVelocitySignal makeCamSignal() {
  AngularVelocitySignal sig;
  for (int i = 0; i <= 40; ++i) {
    const double t = i * 0.025;  // [0, 1]
    sig.times.push_back(t);
    sig.omega.push_back(Eigen::Vector3d(0.5 * std::sin(2 * t), 0.4 * std::cos(1.5 * t),
                                        0.3 * std::sin(t + 0.5)));
  }
  return sig;
}

// FD of the residual Jacobians wrt R_ci (SO3 tangent) and t_d.
CF_TEST(cam_imu_gyro_residual_jacobians_match_finite_difference) {
  AngularVelocitySignal cam = makeCamSignal();
  Sophus::SO3d Rci = Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.15));
  std::array<double, 4> R{};
  SO3Param::store(Rci, R.data());
  std::array<double, 1> td = {0.02};
  CamImuGyroResidual block(&cam, 0.5, Eigen::Vector3d(0.1, 0.2, 0.3));

  double JR[9], Jt[3], r0[3];
  double* jacs[2] = {JR, Jt};
  const double* params[2] = {R.data(), td.data()};
  block.evaluate(params, r0, jacs);

  auto resid = [&](const double* r, const double* t, double o[3]) {
    const double* ps[2] = {r, t};
    block.evaluate(ps, o, nullptr);
  };
  const double h = 1e-6;
  SO3Param so3;
  for (int k = 0; k < 3; ++k) {  // R_ci tangent
    double dp[3] = {0, 0, 0}, dm[3] = {0, 0, 0};
    dp[k] = h; dm[k] = -h;
    std::array<double, 4> Rp{}, Rm{};
    so3.retract(R.data(), dp, Rp.data());
    so3.retract(R.data(), dm, Rm.data());
    double rp[3], rm[3];
    resid(Rp.data(), td.data(), rp);
    resid(Rm.data(), td.data(), rm);
    for (int row = 0; row < 3; ++row)
      CF_EXPECT_NEAR(JR[row * 3 + k], (rp[row] - rm[row]) / (2 * h), 1e-4);
  }
  {  // t_d
    std::array<double, 1> p = {td[0] + h}, m = {td[0] - h};
    double rp[3], rm[3];
    resid(R.data(), p.data(), rp);
    resid(R.data(), m.data(), rm);
    for (int row = 0; row < 3; ++row)
      CF_EXPECT_NEAR(Jt[row], (rp[row] - rm[row]) / (2 * h), 1e-4);
  }
}

// Recover the rotation extrinsic R_ci and the time offset t_d from synthetic gyro data.
CF_TEST(cam_imu_recovers_rotation_and_time_offset) {
  AngularVelocitySignal cam = makeCamSignal();
  Sophus::SO3d Rci_gt = Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.20, 0.15));
  const double td_gt = 0.02;

  std::vector<double> imu_t;
  std::vector<Eigen::Vector3d> imu_w;
  for (int k = 4; k <= 36; ++k) {
    const double t = k * 0.025;  // t - td_gt stays within [0,1]
    Eigen::Vector3d wc, slope;
    cam.sample(t - td_gt, wc, slope);
    imu_t.push_back(t);
    imu_w.push_back(Rci_gt * wc);
  }

  Sophus::SO3d Rci0 = Rci_gt * Sophus::SO3d::exp(Eigen::Vector3d(0.05, -0.04, 0.06));
  LmOptions opts;
  opts.max_iterations = 200;
  CamImuResult res = calibrateCamImuRotation(cam, imu_t, imu_w, Rci0, /*t_d_init=*/0.0, opts);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-10);
  CF_EXPECT_TRUE((res.R_ci * Rci_gt.inverse()).log().norm() < 1e-3);
  CF_EXPECT_NEAR(res.time_offset, td_gt, 1e-4);
  CF_EXPECT_TRUE(assessObservability(res.information).observable);
}

// Constant-axis motion cannot observe the R_ci rotation about that axis: flagged unobservable.
CF_TEST(cam_imu_constant_axis_is_unobservable) {
  AngularVelocitySignal cam;
  for (int i = 0; i <= 40; ++i) {
    const double t = i * 0.025;
    cam.times.push_back(t);
    cam.omega.push_back(Eigen::Vector3d(0, 0, 0.5 + 0.3 * std::sin(2 * t)));  // fixed z-axis
  }
  Sophus::SO3d Rci_gt = Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.20, 0.15));
  const double td_gt = 0.02;
  std::vector<double> imu_t;
  std::vector<Eigen::Vector3d> imu_w;
  for (int k = 4; k <= 36; ++k) {
    const double t = k * 0.025;
    Eigen::Vector3d wc, slope;
    cam.sample(t - td_gt, wc, slope);
    imu_t.push_back(t);
    imu_w.push_back(Rci_gt * wc);
  }
  CamImuResult res = calibrateCamImuRotation(cam, imu_t, imu_w,
                                             Rci_gt * Sophus::SO3d::exp(Eigen::Vector3d(0.02, 0.01, 0.0)),
                                             0.0);
  CF_EXPECT_TRUE(!assessObservability(res.information).observable);
}
