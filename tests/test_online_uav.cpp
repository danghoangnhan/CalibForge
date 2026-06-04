// Tests for pipelines/online_uav.hpp — the UAV in-flight self-cal orchestrator.
//
// The orchestrator wires FeatureTracker -> triangulate -> packMonocularViews -> gated
// emission. We verify the GLUE itself: feeding synthetic per-frame landmarks (skipping the
// image-based feature tracker by hand-building tracks) gates through to the OnlineIntrinsic
// emission. The deep tracker-level + intrinsic-tracker-level tests already live in their own
// files.

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"
#include "calibforge/feature_tracker.hpp"
#include "calibforge/image.hpp"
#include "calibforge/online_calibration.hpp"
#include "calibforge/online_uav.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/triangulate.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::Image8;
using calibforge::PinholeCamera;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;
using calibforge::detect::FeatureObservation;
using calibforge::detect::FeatureTrack;
using calibforge::detect::TriangulatedTrack;
using calibforge::online::Emission;
using calibforge::online::OnlineIntrinsicTracker;
using calibforge::pipelines::OnlineUav;
using calibforge::pipelines::OnlineUavOptions;

CF_TEST(online_uav_triangulate_pack_recovers_intrinsics_via_gated_emission) {
  // Synth: a single pinhole camera flying over a "world" of landmarks. The orchestrator's
  // job is to triangulate tracks + pack + hand to the gated intrinsic tracker; we exercise
  // it directly with hand-built tracks (skipping the image-based tracker — that's tested in
  // test_feature_tracker.cpp).
  const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
  PinholeCamera gt(fx, fy, cx, cy);

  CameraFactory make = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };

  // 25 landmarks (5x5 grid) spread over an area in front of the rig.
  std::vector<Vec3> Xs;
  for (int i = -2; i <= 2; ++i)
    for (int j = -2; j <= 2; ++j)
      Xs.push_back(Vec3{0.5 * i, 0.5 * j, 3.0 + 0.1 * (i + j)});

  // 8 well-excited poses (translation + rotation across all 3 axes).
  std::vector<Sophus::SE3d> poses = {
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.1, -0.05, 0.02)), Eigen::Vector3d(0, 0, 0)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.1, -0.03)), Eigen::Vector3d(0.2, 0, 0)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(-0.05, 0.15, 0.04)), Eigen::Vector3d(0.4, 0.1, 0)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.12, -0.08, 0.05)), Eigen::Vector3d(0.6, -0.05, 0)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.08, 0.04, -0.08)), Eigen::Vector3d(0.5, 0.2, 0.1)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(-0.12, -0.05, 0.1)), Eigen::Vector3d(0.3, 0.3, -0.05)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.06, 0.12, 0.05)), Eigen::Vector3d(0.1, 0.2, 0.05)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(-0.1, 0.08, -0.05)), Eigen::Vector3d(-0.1, 0.1, 0)),
  };

  // Build feature tracks for each landmark — observed across every frame.
  std::vector<FeatureTrack> tracks;
  for (std::size_t i = 0; i < Xs.size(); ++i) {
    FeatureTrack t;
    t.id = static_cast<int>(i);
    for (std::size_t f = 0; f < poses.size(); ++f) {
      const Eigen::Vector3d Xc =
          poses[f].inverse() * Eigen::Vector3d(Xs[i][0], Xs[i][1], Xs[i][2]);
      t.obs.push_back({static_cast<int>(f), gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()})});
    }
    tracks.push_back(std::move(t));
  }

  // Triangulate + pack + hand to the gated intrinsic tracker.
  const std::vector<TriangulatedTrack> tt =
      calibforge::detect::triangulateTracks(tracks, &gt, poses, 4, 1e-5);
  CF_EXPECT_TRUE(tt.size() > 15);
  const std::vector<View> views =
      calibforge::detect::packMonocularViews(tt, static_cast<int>(poses.size()));

  // Reference intrinsics slightly off ground truth.
  Eigen::VectorXd intr_ref(4);
  intr_ref << 480.0, 520.0, 310.0, 250.0;
  OnlineIntrinsicTracker tracker(make, intr_ref, {"fx", "fy", "cx", "cy"});
  for (std::size_t f = 0; f < views.size(); ++f) {
    if (views[f].object_points.empty()) continue;
    // ReprojectionResidual wants T_cam_world (Xc = T*Xw); poses[] are T_world_cam.
    tracker.addFrame(views[f], poses[f].inverse());
  }
  // Use the SHIPPED OnlineUav default threshold, not a hand-picked one — this is what the
  // orchestrator actually passes, and it catches a mis-tuned default that never emits.
  const Emission e = tracker.tryEmit(OnlineUavOptions{}.emit_min_confidence);
  CF_EXPECT_TRUE(e.emitted);
  // Recovered intrinsics close to GT.
  CF_EXPECT_NEAR(e.intrinsics[0], fx, 0.5);
  CF_EXPECT_NEAR(e.intrinsics[1], fy, 0.5);
  CF_EXPECT_NEAR(e.intrinsics[2], cx, 0.5);
  CF_EXPECT_NEAR(e.intrinsics[3], cy, 0.5);
  CF_EXPECT_TRUE(e.drift > 0.0);  // moved away from intr_ref
}

CF_TEST(online_uav_orchestrator_refuses_hovering_window) {
  // Drive the REAL OnlineUav orchestrator (addFrame(image, pose) -> FeatureTracker ->
  // triangulate -> gated emit) rather than the hand-wired glue above. A hovering UAV (all poses
  // identical) has ZERO triangulation parallax, so every track fails the parallax/cheirality
  // gates and NO landmark survives triangulation: tryEmit() hits its `tt.empty()` early-out and
  // REFUSES (emitted == false) before the observability gate is even reached — the upstream half
  // of the RULE #2 guarantee (never emit without constrained data). The image-based feature
  // tracker itself is covered by test_feature_tracker.cpp.
  CameraFactory make = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
  Eigen::VectorXd intr_ref(4);
  intr_ref << 500.0, 500.0, 320.0, 240.0;

  OnlineUav uav(make, intr_ref, {"fx", "fy", "cx", "cy"});
  CF_EXPECT_TRUE(!uav.tryEmit().emitted);  // empty window -> refuse

  const Sophus::SE3d hover(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0));
  for (int f = 0; f < 4; ++f) {
    Image8 blank(64, 48, 0);
    uav.addFrame(blank, hover);  // no motion => no parallax
  }
  CF_EXPECT_TRUE(uav.windowSize() == 4);
  CF_EXPECT_TRUE(!uav.tryEmit().emitted);  // zero parallax -> no landmarks -> gate refuses
}
