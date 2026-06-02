#include "calibforge/calibrate_rolling_shutter.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/manifold.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/rolling_shutter_residual.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::calibrateRollingShutter;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::RollingShutterResidual;
using calibforge::RollingShutterResult;
using calibforge::RollingShutterView;
using calibforge::SE3Param;
using calibforge::Vec2;
using calibforge::Vec3;
using Eigen::VectorXd;
using Vec6 = Eigen::Matrix<double, 6, 1>;

static CameraFactory pinhole() {
  return [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
}

// Manifold-aware FD of all four RS residual Jacobians (intrinsics, pose, velocity, t_r).
CF_TEST(rolling_shutter_residual_jacobians_match_finite_difference) {
  std::array<double, 4> intr = {500, 510, 320, 240};
  Sophus::SE3d T(Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.05)), Eigen::Vector3d(-0.12, -0.08, 1.3));
  std::array<double, 7> pose{};
  SE3Param::store(T, pose.data());
  std::array<double, 6> vel = {0.1, -0.05, 0.02, 0.2, -0.1, 0.15};
  std::array<double, 1> tr = {0.025};
  const Vec3 Xw{0.15, -0.2, 0.05};
  RollingShutterResidual block(pinhole(), 4, Xw, Vec2{50.0, 60.0}, 0.6);

  double Ji[8], Jp[12], Jv[12], Jt[2], r0[2];
  double* jacs[4] = {Ji, Jp, Jv, Jt};
  const double* params[4] = {intr.data(), pose.data(), vel.data(), tr.data()};
  block.evaluate(params, r0, jacs);

  auto resid = [&](const double* a, const double* b, const double* c, const double* d, double o[2]) {
    const double* ps[4] = {a, b, c, d};
    block.evaluate(ps, o, nullptr);
  };
  const double h = 1e-6;
  SE3Param se3;
  for (int k = 0; k < 4; ++k) {  // intrinsics
    std::array<double, 4> p = intr, m = intr;
    p[k] += h; m[k] -= h;
    double rp[2], rm[2];
    resid(p.data(), pose.data(), vel.data(), tr.data(), rp);
    resid(m.data(), pose.data(), vel.data(), tr.data(), rm);
    CF_EXPECT_NEAR(Ji[k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Ji[4 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
  for (int k = 0; k < 6; ++k) {  // pose (SE3 tangent)
    double dp[6] = {0}, dm[6] = {0};
    dp[k] = h; dm[k] = -h;
    std::array<double, 7> pp{}, pm{};
    se3.retract(pose.data(), dp, pp.data());
    se3.retract(pose.data(), dm, pm.data());
    double rp[2], rm[2];
    resid(intr.data(), pp.data(), vel.data(), tr.data(), rp);
    resid(intr.data(), pm.data(), vel.data(), tr.data(), rm);
    CF_EXPECT_NEAR(Jp[k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jp[6 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
  for (int k = 0; k < 6; ++k) {  // velocity (Euclidean)
    std::array<double, 6> p = vel, m = vel;
    p[k] += h; m[k] -= h;
    double rp[2], rm[2];
    resid(intr.data(), pose.data(), p.data(), tr.data(), rp);
    resid(intr.data(), pose.data(), m.data(), tr.data(), rm);
    CF_EXPECT_NEAR(Jv[k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jv[6 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
  {  // t_r (Euclidean scalar)
    std::array<double, 1> p = {tr[0] + h}, m = {tr[0] - h};
    double rp[2], rm[2];
    resid(intr.data(), pose.data(), vel.data(), p.data(), rp);
    resid(intr.data(), pose.data(), vel.data(), m.data(), rm);
    CF_EXPECT_NEAR(Jt[0], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jt[1], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
}

// t_r = 0 reduces exactly to the global-shutter reprojection project(T*Xw) - obs.
CF_TEST(rolling_shutter_zero_readout_is_global_shutter) {
  std::array<double, 4> intr = {500, 500, 320, 240};
  Sophus::SE3d T(Sophus::SO3d::exp(Eigen::Vector3d(0.1, 0.2, -0.05)), Eigen::Vector3d(-0.1, -0.1, 1.4));
  std::array<double, 7> pose{};
  SE3Param::store(T, pose.data());
  std::array<double, 6> vel = {0.5, -0.3, 0.2, 0.4, -0.2, 0.3};  // any velocity
  std::array<double, 1> tr = {0.0};
  const Vec3 Xw{0.2, -0.1, 0.0};
  PinholeCamera cam(500, 500, 320, 240);
  Eigen::Vector3d Xc = T * Eigen::Vector3d(Xw[0], Xw[1], Xw[2]);
  Vec2 gs = cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()});

  RollingShutterResidual block(pinhole(), 4, Xw, Vec2{0.0, 0.0}, 0.8);
  double r[2];
  const double* params[4] = {intr.data(), pose.data(), vel.data(), tr.data()};
  block.evaluate(params, r, nullptr);
  CF_EXPECT_NEAR(r[0], gs[0], 1e-12);  // velocity irrelevant when t_r = 0
  CF_EXPECT_NEAR(r[1], gs[1], 1e-12);
}

// With IMU-known velocities, the shared readout time t_r is recovered from synthetic RS views.
CF_TEST(rolling_shutter_recovers_readout_time_with_known_velocity) {
  const double M = 480.0;
  const double tr_gt = 0.025;
  PinholeCamera cam(500, 500, 320, 240);
  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)), Eigen::Vector3d(-0.20, -0.20, 1.3)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.25, 0.18, -0.08)), Eigen::Vector3d(-0.25, -0.15, 1.1)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.08, 0.30, 0.12)), Eigen::Vector3d(-0.15, -0.25, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.18, 0.12, -0.20)), Eigen::Vector3d(-0.30, -0.10, 1.2)}};
  std::vector<Vec6> vels = {(Vec6() << 0.30, -0.20, 0.10, 0.5, -0.3, 0.4).finished(),
                            (Vec6() << -0.25, 0.30, -0.15, -0.4, 0.5, -0.2).finished(),
                            (Vec6() << 0.20, 0.25, 0.10, 0.3, 0.4, 0.5).finished(),
                            (Vec6() << 0.10, -0.30, 0.20, -0.5, 0.2, 0.3).finished()};

  std::vector<RollingShutterView> views;
  for (std::size_t i = 0; i < gt_poses.size(); ++i) {
    const Sophus::SE3d& T = gt_poses[i];
    Eigen::Vector3d rho = vels[i].head<3>(), omega = vels[i].tail<3>();
    RollingShutterView v;
    for (const auto& X : board) {
      Eigen::Vector3d Xw(X[0], X[1], X[2]);
      Eigen::Vector3d Xc_gs = T * Xw;
      Vec2 px_gs = cam.project(Vec3{Xc_gs.x(), Xc_gs.y(), Xc_gs.z()});
      double f = px_gs[1] / M;
      double s = f * tr_gt;
      Eigen::Vector3d P = Xw + s * (rho + omega.cross(Xw));
      Eigen::Vector3d Xc = T * P;
      v.object_points.push_back(X);
      v.image_points.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
      v.row_fracs.push_back(f);
    }
    views.push_back(v);
  }

  VectorXd intr0(4);
  intr0 << 500, 500, 320, 240;  // intrinsics known; perturb poses + t_r
  std::vector<Sophus::SE3d> poses0;
  Vec6 dp;
  dp << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  for (const auto& T : gt_poses) poses0.push_back(T * Sophus::SE3d::exp(dp));

  LmOptions opts;
  opts.max_iterations = 200;
  RollingShutterResult res = calibrateRollingShutter(views, intr0, poses0, vels, /*tr_init=*/0.0,
                                                     pinhole(), opts, /*optimize_velocities=*/false);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-8);
  CF_EXPECT_NEAR(res.readout_time, tr_gt, 1e-4);
}
