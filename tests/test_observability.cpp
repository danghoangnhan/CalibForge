#include "calibforge/observability.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <memory>
#include <vector>

#include "calibforge/calibrate_single.hpp"
#include "calibforge/camera_model.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::assessObservability;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::ObservabilityReport;
using calibforge::PinholeCamera;
using calibforge::SingleCameraResult;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::Vec6;
using calibforge::View;

// Calibrate a known pinhole camera (fx=fy=500, cx=320, cy=240) from the given poses.
static SingleCameraResult runPinholeCalib(const std::vector<Sophus::SE3d>& gt_poses) {
  PinholeCamera gt(500.0, 500.0, 320.0, 240.0);
  std::vector<Vec3> board;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});

  CameraFactory make = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
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
  Eigen::VectorXd q0(4);
  q0 << 470.0, 530.0, 305.0, 255.0;
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : gt_poses) {
    Vec6 d;
    d << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
    poses0.push_back(T * Sophus::SE3d::exp(d));
  }
  return calibrateSingleCamera(views, q0, poses0, make);
}

// A well-conditioned information matrix (independent, equally-scaled params) is observable.
CF_TEST(observability_identity_is_well_conditioned) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Identity(5, 5);
  ObservabilityReport rep = assessObservability(H);
  CF_EXPECT_TRUE(rep.observable);
  CF_EXPECT_NEAR(rep.confidence, 1.0, 1e-9);  // normalized reciprocal condition number
}

// Parameter SCALE differences alone must NOT read as ill-conditioned: two independent
// params, one scaled by 1e6, is still fully observable after diagonal normalization.
CF_TEST(observability_ignores_pure_scale_differences) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 2);
  H(0, 0) = 1e12;  // e.g. a focal-length-like param
  H(1, 1) = 1.0;   // e.g. an angle-like param
  ObservabilityReport rep = assessObservability(H);
  CF_EXPECT_TRUE(rep.observable);
  CF_EXPECT_NEAR(rep.confidence, 1.0, 1e-9);
}

// A rank-deficient information matrix (two perfectly coupled params) is NOT observable.
CF_TEST(observability_flags_rank_deficient_information) {
  Eigen::MatrixXd H(2, 2);
  H << 1.0, 1.0,
       1.0, 1.0;  // rank 1 -> a zero eigenvalue -> unobservable direction
  ObservabilityReport rep = assessObservability(H);
  CF_EXPECT_TRUE(!rep.observable);
  CF_EXPECT_TRUE(rep.confidence < 1e-9);
}

// A real well-conditioned calibration (several strongly-tilted views) is observable.
CF_TEST(observability_well_conditioned_calibration_is_observable) {
  std::vector<Sophus::SE3d> tilted = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.25, 0.10)), Eigen::Vector3d(-0.10, -0.20, 1.8)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.15, 0.10, -0.15)), Eigen::Vector3d(-0.25, -0.05, 1.4)}};
  SingleCameraResult res = runPinholeCalib(tilted);
  ObservabilityReport rep = assessObservability(res.information);
  std::printf("  [info] well-conditioned rcond = %.3e (cost=%.2e)\n",
              rep.confidence, res.summary.final_cost);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-8);
  CF_EXPECT_TRUE(rep.observable);
}

// A single frontoparallel (zero-tilt) view FITS the data (low cost) yet cannot separate
// focal from depth (and principal point from lateral position): the gate must flag it as
// NOT observable. This is the precision != accuracy guard in action.
CF_TEST(observability_flags_degenerate_single_frontoparallel_view) {
  std::vector<Sophus::SE3d> degenerate = {
      {Sophus::SO3d(), Eigen::Vector3d(-0.15, -0.15, 1.5)}};  // identity rotation = frontoparallel
  SingleCameraResult res = runPinholeCalib(degenerate);
  ObservabilityReport rep = assessObservability(res.information);
  std::printf("  [info] degenerate rcond = %.3e (cost=%.2e)\n",
              rep.confidence, res.summary.final_cost);
  CF_EXPECT_TRUE(!rep.observable);  // untrustworthy despite a good data fit
}
