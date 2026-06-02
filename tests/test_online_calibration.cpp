#include "calibforge/online_calibration.hpp"

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::PinholeCamera;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;
using calibforge::online::Emission;
using calibforge::online::OnlineIntrinsicTracker;
using Eigen::VectorXd;

static CameraFactory pinhole() {
  return [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
}

static std::vector<Vec3> board() {
  std::vector<Vec3> b;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) b.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  return b;
}

static View renderView(const PinholeCamera& cam, const Sophus::SE3d& T, const std::vector<Vec3>& b) {
  View v;
  for (const auto& X : b) {
    Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
    v.object_points.push_back(X);
    v.image_points.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
  }
  return v;
}

// Well-excited (tilted) frames: the gate passes -> the drifted intrinsics are emitted with a
// finite confidence and the measured drift from the reference.
CF_TEST(online_emits_and_tracks_drift_when_well_excited) {
  VectorXd reference(4);
  reference << 500, 500, 320, 240;
  PinholeCamera drifted(505, 503, 322, 238);  // calibration has drifted since the reference
  auto b = board();
  std::vector<Sophus::SE3d> poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.25, 0.10)), Eigen::Vector3d(-0.10, -0.20, 1.8)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.15, 0.10, -0.15)), Eigen::Vector3d(-0.25, -0.05, 1.4)}};

  OnlineIntrinsicTracker tracker(pinhole(), reference, {"fx", "fy", "cx", "cy"});
  Eigen::Matrix<double, 6, 1> dp;
  dp << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  for (const auto& T : poses) tracker.addFrame(renderView(drifted, T, b), T * Sophus::SE3d::exp(dp));
  CF_EXPECT_TRUE(tracker.windowSize() == 4);

  Emission e = tracker.tryEmit(/*min_confidence=*/1e-8);
  CF_EXPECT_TRUE(e.emitted);
  CF_EXPECT_TRUE(e.confidence > 1e-8);
  CF_EXPECT_NEAR(e.intrinsics[0], 505.0, 1e-2);  // recovered drifted focal
  CF_EXPECT_NEAR(e.intrinsics[2], 322.0, 1e-2);
  CF_EXPECT_NEAR(e.drift, std::sqrt(25.0 + 9.0 + 4.0 + 4.0), 1e-2);  // ||drifted - reference||
  CF_EXPECT_TRUE(e.weak_parameters.empty());
}

// A single frontoparallel frame FITS the data (low cost) yet cannot separate focal from depth:
// the gate REFUSES to emit and names the weak directions. The precision != accuracy guard.
CF_TEST(online_refuses_when_degenerate_despite_low_cost) {
  VectorXd reference(4);
  reference << 500, 500, 320, 240;
  PinholeCamera drifted(505, 503, 322, 238);
  auto b = board();
  Sophus::SE3d frontoparallel(Sophus::SO3d(), Eigen::Vector3d(-0.15, -0.15, 1.5));  // identity rot

  OnlineIntrinsicTracker tracker(pinhole(), reference, {"fx", "fy", "cx", "cy"});
  tracker.addFrame(renderView(drifted, frontoparallel, b), frontoparallel);

  Emission e = tracker.tryEmit(/*min_confidence=*/1e-8);
  CF_EXPECT_TRUE(!e.emitted);                  // refused — untrustworthy despite a perfect fit
  CF_EXPECT_TRUE(e.intrinsics.size() == 0);    // nothing emitted
  CF_EXPECT_TRUE(!e.weak_parameters.empty());  // the unobservable directions are named
}

// Stricter confidence threshold can also refuse a marginally-conditioned estimate.
CF_TEST(online_confidence_threshold_gates_emission) {
  VectorXd reference(4);
  reference << 500, 500, 320, 240;
  PinholeCamera cam(500, 500, 320, 240);
  auto b = board();
  std::vector<Sophus::SE3d> poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)}};
  OnlineIntrinsicTracker tracker(pinhole(), reference, {"fx", "fy", "cx", "cy"});
  for (const auto& T : poses) tracker.addFrame(renderView(cam, T, b), T);

  // A real conditioned calibration has confidence ~1e-6; an absurdly strict threshold refuses it.
  Emission strict = tracker.tryEmit(/*min_confidence=*/0.9);
  CF_EXPECT_TRUE(!strict.emitted);
  Emission ok = tracker.tryEmit(/*min_confidence=*/1e-8);
  CF_EXPECT_TRUE(ok.emitted);
}
