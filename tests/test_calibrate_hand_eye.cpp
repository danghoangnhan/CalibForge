#include "calibforge/calibrate_hand_eye.hpp"

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/hand_eye_residual.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/observability.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::assessObservability;
using calibforge::calibrateHandEye;
using calibforge::HandEyeResidual;
using calibforge::HandEyeResult;
using calibforge::HandEyeView;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::SE3Param;
using calibforge::Vec2;
using calibforge::Vec3;

static Sophus::SE3d perturb(const Sophus::SE3d& T) {
  Eigen::Matrix<double, 6, 1> d;
  d << 0.02, -0.015, 0.02, 0.03, -0.02, 0.015;
  return T * Sophus::SE3d::exp(d);
}

// Manifold-aware FD of the hand-eye residual's two SE(3) Jacobians (X and Z).
CF_TEST(hand_eye_residual_jacobians_match_finite_difference) {
  PinholeCamera cam(500, 500, 320, 240);
  Sophus::SE3d X(Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.2, 0.05)), Eigen::Vector3d(0.03, -0.02, 0.05));
  Sophus::SE3d Z(Sophus::SO3d::exp(Eigen::Vector3d(-0.1, 0.15, 0.2)), Eigen::Vector3d(0.1, 0.05, 1.2));
  Sophus::SE3d A(Sophus::SO3d::exp(Eigen::Vector3d(0.2, 0.1, -0.15)), Eigen::Vector3d(-0.2, 0.1, 0.3));
  std::array<double, 7> xa{}, za{};
  SE3Param::store(X, xa.data());
  SE3Param::store(Z, za.data());

  const Vec3 Xw{0.1, -0.15, 0.0};
  HandEyeResidual block(&cam, A, Xw, Vec2{40.0, 50.0});

  double Jx[12], Jz[12], r0[2];
  double* jacs[2] = {Jx, Jz};
  const double* params[2] = {xa.data(), za.data()};
  block.evaluate(params, r0, jacs);

  auto residual = [&](const double* xp, const double* zp, double out[2]) {
    const double* ps[2] = {xp, zp};
    block.evaluate(ps, out, nullptr);
  };
  const double h = 1e-6;
  SE3Param se3;
  for (int k = 0; k < 6; ++k) {
    double dp[6] = {0}, dm[6] = {0};
    dp[k] = h; dm[k] = -h;
    std::array<double, 7> xp{}, xm{}, zp{}, zm{};
    se3.retract(xa.data(), dp, xp.data());
    se3.retract(xa.data(), dm, xm.data());
    se3.retract(za.data(), dp, zp.data());
    se3.retract(za.data(), dm, zm.data());
    double rxp[2], rxm[2], rzp[2], rzm[2];
    residual(xp.data(), za.data(), rxp);
    residual(xm.data(), za.data(), rxm);
    residual(xa.data(), zp.data(), rzp);
    residual(xa.data(), zm.data(), rzm);
    CF_EXPECT_NEAR(Jx[k], (rxp[0] - rxm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jx[6 + k], (rxp[1] - rxm[1]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jz[k], (rzp[0] - rzm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jz[6 + k], (rzp[1] - rzm[1]) / (2 * h), 1e-4);
  }
}

// Build a hand-eye scene from good camera-vs-target views, derive the robot poses, and
// recover X = T_gripper_cam and Z = T_base_target from a perturbed initial guess.
static std::vector<HandEyeView> makeScene(const PinholeCamera& cam, const Sophus::SE3d& X,
                                          const Sophus::SE3d& Z,
                                          const std::vector<Sophus::SE3d>& cam_from_target,
                                          const std::vector<Vec3>& board) {
  std::vector<HandEyeView> views;
  for (const auto& T_cam_target : cam_from_target) {       // world(target)->cam
    HandEyeView v;
    const Sophus::SE3d T_target_cam = T_cam_target.inverse();
    const Sophus::SE3d T_base_cam = Z * T_target_cam;
    v.T_base_gripper = T_base_cam * X.inverse();
    v.object_points = board;
    for (const auto& Xw : board) {
      Eigen::Vector3d Xc = T_cam_target * Eigen::Vector3d(Xw[0], Xw[1], Xw[2]);
      v.image_points.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    }
    views.push_back(v);
  }
  return views;
}

CF_TEST(hand_eye_recovers_X_and_Z) {
  PinholeCamera cam(500, 500, 320, 240);
  Sophus::SE3d X(Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.20, 0.05)), Eigen::Vector3d(0.03, -0.02, 0.05));
  Sophus::SE3d Z(Sophus::SO3d::exp(Eigen::Vector3d(-0.10, 0.15, 0.20)), Eigen::Vector3d(0.40, 0.10, 0.20));

  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  std::vector<Sophus::SE3d> cam_from_target = {  // varied rotation axes -> observable
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)), Eigen::Vector3d(-0.20, -0.20, 1.3)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.25, 0.18, -0.08)), Eigen::Vector3d(-0.25, -0.15, 1.1)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.08, 0.30, 0.12)), Eigen::Vector3d(-0.15, -0.25, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.18, 0.12, -0.20)), Eigen::Vector3d(-0.30, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.12, -0.22, 0.18)), Eigen::Vector3d(-0.20, -0.28, 1.4)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.28, 0.05, 0.22)), Eigen::Vector3d(-0.18, -0.18, 1.0)}};

  std::vector<HandEyeView> views = makeScene(cam, X, Z, cam_from_target, board);

  LmOptions opts;
  opts.max_iterations = 300;
  HandEyeResult res = calibrateHandEye(views, perturb(X), perturb(Z), cam, opts);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-8);
  CF_EXPECT_TRUE((res.X * X.inverse()).log().norm() < 1e-3);
  CF_EXPECT_TRUE((res.Z * Z.inverse()).log().norm() < 1e-3);
  CF_EXPECT_TRUE(assessObservability(res.information).observable);
}

// A single view cannot separate X from Z (only the product X^{-1} A Z is constrained):
// the observability gate must flag it.
CF_TEST(hand_eye_single_view_is_unobservable) {
  PinholeCamera cam(500, 500, 320, 240);
  Sophus::SE3d X(Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.20, 0.05)), Eigen::Vector3d(0.03, -0.02, 0.05));
  Sophus::SE3d Z(Sophus::SO3d::exp(Eigen::Vector3d(-0.10, 0.15, 0.20)), Eigen::Vector3d(0.40, 0.10, 0.20));
  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  std::vector<Sophus::SE3d> one = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)), Eigen::Vector3d(-0.20, -0.20, 1.3)}};
  std::vector<HandEyeView> views = makeScene(cam, X, Z, one, board);

  HandEyeResult res = calibrateHandEye(views, perturb(X), perturb(Z), cam);
  CF_EXPECT_TRUE(!assessObservability(res.information).observable);
}
