#include "calibforge/calibrate_single.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "calibforge/brown_conrady_camera.hpp"
#include "calibforge/camera_model.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::BrownConradyCamera;
using calibforge::calibrateSingleCamera;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::SingleCameraResult;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::Vec6;
using calibforge::View;
using Eigen::VectorXd;

// Small right-perturbation of a ground-truth pose, used to seed a wrong initial guess.
static Sophus::SE3d perturb(const Sophus::SE3d& T) {
  Vec6 d;
  d << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  return T * Sophus::SE3d::exp(d);
}

// End-to-end single-camera calibration (the v0.1 "validated vs ground truth"
// milestone), now with ANALYTIC Jacobians + SE(3) retraction: recover pinhole
// intrinsics + per-view poses from noise-free observations and a wrong initial guess.
CF_TEST(calibrate_single_recovers_pinhole_intrinsics_from_synthetic_views) {
  const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
  PinholeCamera gt_cam(fx, fy, cx, cy);

  std::vector<Vec3> board;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});

  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.25, 0.10)), Eigen::Vector3d(-0.10, -0.20, 1.8)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.15, 0.10, -0.15)), Eigen::Vector3d(-0.25, -0.05, 1.4)}};

  CameraFactory make_camera = [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };

  std::vector<View> views;
  for (const auto& T : gt_poses) {
    View v;
    for (const auto& X : board) {
      Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
      v.object_points.push_back(X);
      v.image_points.push_back(gt_cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    }
    views.push_back(v);
  }

  VectorXd intr0(4);
  intr0 << 470.0, 530.0, 305.0, 255.0;
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : gt_poses) poses0.push_back(perturb(T));

  SingleCameraResult res = calibrateSingleCamera(views, intr0, poses0, make_camera);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-8);
  CF_EXPECT_NEAR(res.intrinsics[0], fx, 1e-3);
  CF_EXPECT_NEAR(res.intrinsics[1], fy, 1e-3);
  CF_EXPECT_NEAR(res.intrinsics[2], cx, 1e-3);
  CF_EXPECT_NEAR(res.intrinsics[3], cy, 1e-3);
}

// Same pipeline recovering Brown-Conrady distortion (9 intrinsics) too, with analytic
// Jacobians. Richer scene (more points, strongly tilted views) so distortion is well
// observed; distortion initialized at zero as in real calibration.
CF_TEST(calibrate_single_recovers_brown_conrady_distortion_from_synthetic_views) {
  const double fx = 520.0, fy = 520.0, cx = 320.0, cy = 240.0;
  const double k1 = -0.15, k2 = 0.04, p1 = 0.001, p2 = -0.001, k3 = 0.0;
  BrownConradyCamera gt(fx, fy, cx, cy, k1, k2, p1, p2, k3);

  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});

  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.25, -0.10, 0.05)), Eigen::Vector3d(-0.20, -0.20, 1.3)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.30, 0.20, -0.10)), Eigen::Vector3d(-0.25, -0.15, 1.1)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, 0.35, 0.15)), Eigen::Vector3d(-0.15, -0.25, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, 0.15, -0.25)), Eigen::Vector3d(-0.30, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.15, -0.25, 0.20)), Eigen::Vector3d(-0.20, -0.30, 1.4)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.30, -0.20, -0.10)), Eigen::Vector3d(-0.10, -0.15, 1.0)}};

  CameraFactory make_camera = [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<BrownConradyCamera>(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]);
  };

  std::vector<View> views;
  for (const auto& T : gt_poses) {
    View v;
    for (const auto& X : board) {
      Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
      v.object_points.push_back(X);
      v.image_points.push_back(gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    }
    views.push_back(v);
  }

  VectorXd q0(9);
  q0 << 500.0, 540.0, 305.0, 250.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // distortion starts at zero
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : gt_poses) poses0.push_back(perturb(T));

  LmOptions opts;
  opts.max_iterations = 300;
  SingleCameraResult res = calibrateSingleCamera(views, q0, poses0, make_camera, opts);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-6);
  CF_EXPECT_NEAR(res.intrinsics[0], fx, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics[1], fy, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics[2], cx, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics[3], cy, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics[4], k1, 1e-3);
  CF_EXPECT_NEAR(res.intrinsics[5], k2, 1e-3);
  CF_EXPECT_NEAR(res.intrinsics[6], p1, 1e-3);
  CF_EXPECT_NEAR(res.intrinsics[7], p2, 1e-3);
  CF_EXPECT_NEAR(res.intrinsics[8], k3, 1e-3);
}
