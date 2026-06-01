#include "calibforge/dense_problem.hpp"

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/manifold.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/reprojection_residual.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraModel;
using calibforge::DenseProblem;
using calibforge::EuclideanParam;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::ReprojectionResidual;
using calibforge::SE3Param;
using calibforge::Vec2;
using calibforge::Vec3;
using Eigen::VectorXd;

static std::unique_ptr<CameraModel> makePinhole(const VectorXd& q) {
  return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
}

// The reprojection residual reproduces project(T*Xw) - observed.
CF_TEST(reprojection_residual_evaluate_matches_camera_project) {
  const double fx = 500, fy = 510, cx = 320, cy = 240;
  std::array<double, 4> intr = {fx, fy, cx, cy};
  Sophus::SE3d T(Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.05, 0.02)),
                 Eigen::Vector3d(-0.1, -0.05, 1.4));
  std::array<double, 7> pose{};
  SE3Param::store(T, pose.data());

  const Vec3 X{0.2, 0.1, 0.0};
  const Vec2 obs{123.0, 200.0};
  ReprojectionResidual block(makePinhole, 4, X, obs);

  double out[2];
  const double* params[2] = {intr.data(), pose.data()};
  block.evaluate(params, out, nullptr);

  PinholeCamera cam(fx, fy, cx, cy);
  Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
  Vec2 px = cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()});
  CF_EXPECT_NEAR(out[0], px[0] - obs[0], 1e-12);
  CF_EXPECT_NEAR(out[1], px[1] - obs[1], 1e-12);
}

// Manifold-aware finite-difference check of the minimal analytic Jacobians: the pose
// Jacobian is in SE(3) tangent space, so the FD perturbation must go through retract().
CF_TEST(reprojection_residual_jacobians_match_finite_difference) {
  std::array<double, 4> intr = {500, 510, 320, 240};
  Sophus::SE3d T(Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.05)),
                 Eigen::Vector3d(-0.12, -0.08, 1.3));
  std::array<double, 7> pose{};
  SE3Param::store(T, pose.data());

  const Vec3 X{0.15, -0.2, 0.05};
  const Vec2 obs{50.0, 60.0};
  ReprojectionResidual block(makePinhole, 4, X, obs);

  double Ji[8];  // 2 x 4
  double Jp[12];  // 2 x 6
  double* jacs[2] = {Ji, Jp};
  double r0[2];
  const double* params[2] = {intr.data(), pose.data()};
  block.evaluate(params, r0, jacs);

  auto residual = [&](const double* in, const double* pp, double out[2]) {
    const double* ps[2] = {in, pp};
    block.evaluate(ps, out, nullptr);
  };

  // d(residual)/d(intrinsics) via Euclidean perturbation.
  const double h = 1e-5;
  for (int k = 0; k < 4; ++k) {
    std::array<double, 4> ip = intr, im = intr;
    ip[k] += h; im[k] -= h;
    double rp[2], rm[2];
    residual(ip.data(), pose.data(), rp);
    residual(im.data(), pose.data(), rm);
    CF_EXPECT_NEAR(Ji[0 * 4 + k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Ji[1 * 4 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }

  // d(residual)/d(pose tangent) via SE3 retract perturbation.
  SE3Param se3;
  for (int k = 0; k < 6; ++k) {
    double dp[6] = {0, 0, 0, 0, 0, 0}, dm[6] = {0, 0, 0, 0, 0, 0};
    dp[k] = h; dm[k] = -h;
    std::array<double, 7> pp{}, pm{};
    se3.retract(pose.data(), dp, pp.data());
    se3.retract(pose.data(), dm, pm.data());
    double rp[2], rm[2];
    residual(intr.data(), pp.data(), rp);
    residual(intr.data(), pm.data(), rm);
    CF_EXPECT_NEAR(Jp[0 * 6 + k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jp[1 * 6 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
}

// Helper: build the standard synthetic single-camera scene and solve it via DenseProblem.
static void buildScene(std::vector<Vec3>& board, std::vector<Sophus::SE3d>& gt_poses) {
  for (int row = 0; row < 4; ++row)
    for (int c = 0; c < 4; ++c) board.push_back(Vec3{c * 0.1, row * 0.1, 0.0});
  gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.25, 0.10)), Eigen::Vector3d(-0.10, -0.20, 1.8)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.15, 0.10, -0.15)), Eigen::Vector3d(-0.25, -0.05, 1.4)}};
}

// The concrete interface recovers intrinsics + poses directly (no calibrate_single wrapper).
CF_TEST(dense_problem_recovers_pinhole_intrinsics) {
  const double fx = 500, fy = 500, cx = 320, cy = 240;
  PinholeCamera gt(fx, fy, cx, cy);
  std::vector<Vec3> board;
  std::vector<Sophus::SE3d> gt_poses;
  buildScene(board, gt_poses);

  std::vector<double> intr = {470, 530, 305, 255};  // wrong init
  std::vector<std::array<double, 7>> pose(gt_poses.size());
  Eigen::Matrix<double, 6, 1> perturb;
  perturb << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    SE3Param::store(gt_poses[i] * Sophus::SE3d::exp(perturb), pose[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr.data(), std::make_shared<EuclideanParam>(4));
  for (auto& p : pose) problem.addParameterBlock(p.data(), std::make_shared<SE3Param>());
  for (std::size_t i = 0; i < gt_poses.size(); ++i) {
    for (const auto& X : board) {
      Eigen::Vector3d Xc = gt_poses[i] * Eigen::Vector3d(X[0], X[1], X[2]);
      Vec2 px = gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()});
      problem.addResidualBlock(std::make_unique<ReprojectionResidual>(makePinhole, 4, X, px),
                               {intr.data(), pose[i].data()});
    }
  }

  auto s = problem.solveLm();
  CF_EXPECT_TRUE(s.converged);
  CF_EXPECT_TRUE(s.final_cost < 1e-8);
  CF_EXPECT_NEAR(intr[0], fx, 1e-3);
  CF_EXPECT_NEAR(intr[1], fy, 1e-3);
  CF_EXPECT_NEAR(intr[2], cx, 1e-3);
  CF_EXPECT_NEAR(intr[3], cy, 1e-3);
}

// setParameterBlockConstant freezes a block: with intrinsics held at ground truth and
// only poses perturbed, the solver recovers the poses while intrinsics stay put.
CF_TEST(dense_problem_set_parameter_block_constant_holds_block_fixed) {
  const double fx = 500, fy = 500, cx = 320, cy = 240;
  PinholeCamera gt(fx, fy, cx, cy);
  std::vector<Vec3> board;
  std::vector<Sophus::SE3d> gt_poses;
  buildScene(board, gt_poses);

  std::vector<double> intr = {fx, fy, cx, cy};  // exact, will be frozen
  std::vector<std::array<double, 7>> pose(gt_poses.size());
  Eigen::Matrix<double, 6, 1> perturb;
  perturb << 0.03, -0.03, 0.03, 0.03, -0.02, 0.02;
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    SE3Param::store(gt_poses[i] * Sophus::SE3d::exp(perturb), pose[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr.data(), std::make_shared<EuclideanParam>(4));
  for (auto& p : pose) problem.addParameterBlock(p.data(), std::make_shared<SE3Param>());
  problem.setParameterBlockConstant(intr.data());
  for (std::size_t i = 0; i < gt_poses.size(); ++i) {
    for (const auto& X : board) {
      Eigen::Vector3d Xc = gt_poses[i] * Eigen::Vector3d(X[0], X[1], X[2]);
      Vec2 px = gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()});
      problem.addResidualBlock(std::make_unique<ReprojectionResidual>(makePinhole, 4, X, px),
                               {intr.data(), pose[i].data()});
    }
  }

  auto s = problem.solveLm();
  CF_EXPECT_TRUE(s.converged);
  CF_EXPECT_TRUE(s.final_cost < 1e-10);
  CF_EXPECT_NEAR(intr[0], fx, 1e-12);  // frozen — must be bit-stable
  CF_EXPECT_NEAR(intr[1], fy, 1e-12);
  CF_EXPECT_NEAR(intr[2], cx, 1e-12);
  CF_EXPECT_NEAR(intr[3], cy, 1e-12);
}
