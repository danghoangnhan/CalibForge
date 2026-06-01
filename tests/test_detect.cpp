#include "calibforge/checkerboard_detect.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/board.hpp"
#include "calibforge/calibrate_single.hpp"
#include "calibforge/camera_model.hpp"
#include "calibforge/corner_detect.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/render_board.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::PinholeCamera;
using calibforge::SingleCameraResult;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;
using calibforge::detect::CheckerboardSpec;
using calibforge::detect::Corner2;
using calibforge::detect::detectCheckerboardView;
using calibforge::detect::detectSaddleCorners;
using calibforge::detect::renderCheckerboard;
using Eigen::VectorXd;

// A near-frontal pose looking at the board centre from distance Z with a small tilt.
static Sophus::SE3d viewPose(const CheckerboardSpec& spec, double Z, const Eigen::Vector3d& tilt) {
  const Sophus::SO3d R = Sophus::SO3d::exp(tilt);
  const Eigen::Vector3d center(spec.extentX() / 2.0, spec.extentY() / 2.0, 0.0);
  // Place the board centre at cam-frame (0,0,Z): t = (0,0,Z) - R*center.
  return Sophus::SE3d(R, Eigen::Vector3d(0, 0, Z) - R * center);
}

static std::vector<Vec2> expectedCorners(const PinholeCamera& cam, const Sophus::SE3d& T,
                                         const CheckerboardSpec& spec) {
  std::vector<Vec2> px;
  for (const auto& X : spec.objectPoints()) {
    Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
    px.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
  }
  return px;
}

// Genuinely FIND the inner corners (saddle response + NMS + sub-pixel) on a rendered board:
// every ground-truth inner corner has a detection within sub-pixel tolerance.
CF_TEST(detect_saddle_corners_finds_board_corners_subpixel) {
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  CheckerboardSpec spec{7, 5, 0.025};
  Sophus::SE3d T = viewPose(spec, 0.55, Eigen::Vector3d(0.08, -0.06, 0.02));
  auto img = renderCheckerboard(cam, T, spec, 320, 240);

  std::vector<Corner2> found = detectSaddleCorners(img);
  CF_EXPECT_TRUE(found.size() >= spec.numCorners());

  std::vector<Vec2> gt = expectedCorners(cam, T, spec);
  int matched = 0;
  double worst = 0.0;
  for (const auto& g : gt) {
    double best = 1e300;
    for (const auto& c : found) {
      const double d = std::hypot(c.x - g[0], c.y - g[1]);
      if (d < best) best = d;
    }
    if (best < 0.25) ++matched;
    worst = std::max(worst, best);
  }
  std::printf("  [info] detect matched %d/%zu corners, worst-nearest = %.3f px\n", matched,
              spec.numCorners(), worst);
  CF_EXPECT_TRUE(matched == static_cast<int>(spec.numCorners()));
}

// Detected corners across several views feed single-camera calibration and recover intrinsics.
CF_TEST(detect_feeds_single_camera_calibration) {
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  CheckerboardSpec spec{7, 5, 0.025};
  std::vector<Eigen::Vector3d> tilts = {
      {0.10, -0.07, 0.03}, {-0.12, 0.09, -0.04}, {0.06, 0.13, 0.05},
      {0.14, 0.05, -0.10}, {-0.08, -0.11, 0.07}};
  std::vector<double> zs = {0.50, 0.58, 0.62, 0.54, 0.60};

  std::vector<View> views;
  for (std::size_t k = 0; k < tilts.size(); ++k) {
    Sophus::SE3d T = viewPose(spec, zs[k], tilts[k]);
    auto img = renderCheckerboard(cam, T, spec, 320, 240);
    std::vector<Vec2> expected = expectedCorners(cam, T, spec);
    View v = detectCheckerboardView(img, spec, expected);
    CF_EXPECT_TRUE(v.object_points.size() >= spec.numCorners() - 2);  // nearly all corners
    views.push_back(v);
  }

  CameraFactory make = [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
  // Initial poses: a rough guess (board centre in front), distortion-free pinhole.
  std::vector<Sophus::SE3d> poses0;
  for (std::size_t k = 0; k < views.size(); ++k)
    poses0.push_back(viewPose(spec, 0.55, Eigen::Vector3d(0.05, -0.05, 0.0)));
  VectorXd q0(4);
  q0 << 280.0, 320.0, 158.0, 124.0;

  SingleCameraResult res = calibforge::calibrateSingleCamera(views, q0, poses0, make);
  std::printf("  [info] detect->calib fx=%.3f fy=%.3f cx=%.3f cy=%.3f (cost=%.3e)\n",
              res.intrinsics[0], res.intrinsics[1], res.intrinsics[2], res.intrinsics[3],
              res.summary.final_cost);
  CF_EXPECT_TRUE(res.summary.converged);
  // The hand-rolled saddle detector has ~0.12 px correlated sub-pixel bias on tilted boards,
  // which concentrates into a few px of intrinsic error (the optional OpenCV path is tighter).
  CF_EXPECT_NEAR(res.intrinsics[0], 300.0, 3.0);
  CF_EXPECT_NEAR(res.intrinsics[1], 300.0, 3.0);
  CF_EXPECT_NEAR(res.intrinsics[2], 160.0, 3.0);
  CF_EXPECT_NEAR(res.intrinsics[3], 120.0, 3.0);
}
