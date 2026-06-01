#include "calibforge/least_squares.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::calibrateSingleCamera;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::RobustKernel;
using calibforge::RobustLoss;
using calibforge::robustWeightSqrt;
using calibforge::SingleCameraResult;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;
using Eigen::VectorXd;

// w = sqrt(rho'(s)) for s = ||r_block||^2.
CF_TEST(robust_weight_kernels_match_definition) {
  RobustLoss none{RobustKernel::None, 1.0};
  CF_EXPECT_NEAR(robustWeightSqrt(none, 0.0), 1.0, 1e-15);
  CF_EXPECT_NEAR(robustWeightSqrt(none, 999.0), 1.0, 1e-15);

  RobustLoss hub{RobustKernel::Huber, 2.0};
  CF_EXPECT_NEAR(robustWeightSqrt(hub, 1.0), 1.0, 1e-15);          // s <= delta^2 -> 1
  CF_EXPECT_NEAR(robustWeightSqrt(hub, 4.0), 1.0, 1e-15);          // s == delta^2 -> 1
  CF_EXPECT_NEAR(robustWeightSqrt(hub, 100.0), std::sqrt(0.2), 1e-12);  // sqrt(2/sqrt(100))

  RobustLoss cau{RobustKernel::Cauchy, 2.0};
  CF_EXPECT_NEAR(robustWeightSqrt(cau, 0.0), 1.0, 1e-15);
  CF_EXPECT_NEAR(robustWeightSqrt(cau, 4.0), 1.0 / std::sqrt(2.0), 1e-12);
  CF_EXPECT_NEAR(robustWeightSqrt(cau, 100.0), 1.0 / std::sqrt(26.0), 1e-12);
}

static Sophus::SE3d perturbPose(const Sophus::SE3d& T) {
  Eigen::Matrix<double, 6, 1> d;
  d << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  return T * Sophus::SE3d::exp(d);
}

// Build the standard pinhole scene, then corrupt a fraction of observations with a large
// offset. Robust Huber loss recovers near ground truth; plain least squares is skewed.
CF_TEST(robust_loss_recovers_intrinsics_despite_outliers) {
  const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
  PinholeCamera gt(fx, fy, cx, cy);

  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});

  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.25, 0.10)), Eigen::Vector3d(-0.10, -0.20, 1.8)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.15, 0.10, -0.15)), Eigen::Vector3d(-0.25, -0.05, 1.4)}};

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
  // Corrupt ~10% of points with a big asymmetric offset (drags cx/cy if unweighted).
  int corrupted = 0;
  for (std::size_t i = 0; i < views.size(); ++i)
    for (std::size_t j = 0; j < views[i].image_points.size(); ++j)
      if ((i * 25 + j) % 10 == 3) {
        views[i].image_points[j][0] += 40.0;
        views[i].image_points[j][1] += 40.0;
        ++corrupted;
      }
  CF_EXPECT_TRUE(corrupted >= 8);

  CameraFactory make_camera = [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
  VectorXd intr0(4);
  intr0 << 470.0, 530.0, 305.0, 255.0;
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : gt_poses) poses0.push_back(perturbPose(T));

  LmOptions plain;
  plain.max_iterations = 200;
  SingleCameraResult r_plain = calibrateSingleCamera(views, intr0, poses0, make_camera, plain);

  LmOptions robust;
  robust.max_iterations = 300;
  robust.robust = {RobustKernel::Cauchy, 1.0};  // redescending-style; rejects gross outliers
  SingleCameraResult r_robust = calibrateSingleCamera(views, intr0, poses0, make_camera, robust);

  const double err_robust_cx = std::fabs(r_robust.intrinsics[2] - cx);
  const double err_plain_cx = std::fabs(r_plain.intrinsics[2] - cx);
  const double err_robust_fx = std::fabs(r_robust.intrinsics[0] - fx);
  const double err_plain_fx = std::fabs(r_plain.intrinsics[0] - fx);

  // Robust recovers close to ground truth despite 10% gross outliers...
  CF_EXPECT_TRUE(err_robust_cx < 1.0);
  CF_EXPECT_TRUE(err_robust_fx < 2.0);
  // ...while the plain least-squares fit is badly skewed (errors of tens of pixels).
  CF_EXPECT_TRUE(err_plain_cx > 20.0);
  CF_EXPECT_TRUE(err_plain_fx > 20.0);
  CF_EXPECT_TRUE(err_plain_cx > 10.0 * err_robust_cx);
}

// Huber also improves on plain least squares (downweights, less aggressively than Cauchy).
CF_TEST(robust_huber_beats_plain_on_outliers) {
  const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
  PinholeCamera gt(fx, fy, cx, cy);
  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.25, 0.10)), Eigen::Vector3d(-0.10, -0.20, 1.8)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.15, 0.10, -0.15)), Eigen::Vector3d(-0.25, -0.05, 1.4)}};
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
  for (std::size_t i = 0; i < views.size(); ++i)
    for (std::size_t j = 0; j < views[i].image_points.size(); ++j)
      if ((i * 25 + j) % 10 == 3) {
        views[i].image_points[j][0] += 40.0;
        views[i].image_points[j][1] += 40.0;
      }
  CameraFactory make_camera = [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
  VectorXd intr0(4);
  intr0 << 470.0, 530.0, 305.0, 255.0;
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : gt_poses) poses0.push_back(perturbPose(T));

  LmOptions plain;
  plain.max_iterations = 200;
  SingleCameraResult r_plain = calibrateSingleCamera(views, intr0, poses0, make_camera, plain);
  LmOptions hub;
  hub.max_iterations = 300;
  hub.robust = {RobustKernel::Huber, 1.0};
  SingleCameraResult r_hub = calibrateSingleCamera(views, intr0, poses0, make_camera, hub);

  CF_EXPECT_TRUE(std::fabs(r_hub.intrinsics[2] - cx) < std::fabs(r_plain.intrinsics[2] - cx));
  CF_EXPECT_TRUE(std::fabs(r_hub.intrinsics[0] - fx) < std::fabs(r_plain.intrinsics[0] - fx));
}
