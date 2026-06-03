// Tests for OnlineExtrinsicTracker — the v0.5 extrinsic-drift differentiator.
//
//   1. Well-excited motion + observations consistent with the true (drifted-from-reference)
//      extrinsics: the tracker re-estimates, the observability + motion gates both pass, the
//      tracker emits, and the emitted extrinsics match ground truth while the reported drift
//      vs. the reference is the magnitude of the deliberate perturbation we baked in.
//   2. Degenerate motion (only translation along one axis, no rotation) — the 6-axis motion
//      gate refuses BEFORE even solving and reports which axes are unexcited.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_rig.hpp"  // RigView
#include "calibforge/calibrate_single.hpp"  // CameraFactory
#include "calibforge/online_extrinsic_tracker.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::PinholeCamera;
using calibforge::RigView;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::online::ExtrinsicEmission;
using calibforge::online::MotionExcitationOptions;
using calibforge::online::OnlineExtrinsicTracker;

namespace {

CameraFactory pinholeFactory() {
  return [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
}

// Build a (board_points, per_camera_views) frame given the rig pose T_c0_w and the per-camera
// stereo extrinsics T_ck_c0. Each camera projects every board point — clean noise-free.
RigView synthFrame(const std::vector<PinholeCamera>& cams,
                   const std::vector<Sophus::SE3d>& T_ck_c0_after_first,
                   const Sophus::SE3d& T_c0_w, const std::vector<Vec3>& board) {
  RigView v;
  v.object_points = board;
  v.image_points.assign(cams.size(), {});
  for (const Vec3& X : board) {
    const Eigen::Vector3d Xc0 = T_c0_w * Eigen::Vector3d(X[0], X[1], X[2]);
    v.image_points[0].push_back(cams[0].project(Vec3{Xc0.x(), Xc0.y(), Xc0.z()}));
    for (std::size_t k = 0; k < T_ck_c0_after_first.size(); ++k) {
      const Eigen::Vector3d Xck = T_ck_c0_after_first[k] * Xc0;
      v.image_points[k + 1].push_back(
          cams[k + 1].project(Vec3{Xck.x(), Xck.y(), Xck.z()}));
    }
  }
  return v;
}

}  // namespace

CF_TEST(online_extrinsic_tracker_emits_with_excitation_and_reports_drift) {
  std::vector<PinholeCamera> cams = {PinholeCamera(500, 500, 320, 240),
                                     PinholeCamera(510, 508, 322, 238),
                                     PinholeCamera(495, 497, 318, 242)};
  std::vector<Sophus::SE3d> T_ck_c0_gt = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)),
       Eigen::Vector3d(-0.10, 0.005, 0.002)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.015, 0.02, -0.01)),
       Eigen::Vector3d(0.10, -0.004, 0.003)}};

  // Reference extrinsics = small perturbation of GT (the "prior calibration").
  Eigen::Matrix<double, 6, 1> perturb;
  perturb << 0.015, -0.012, 0.010, 0.020, -0.015, 0.018;
  std::vector<Sophus::SE3d> T_ck_c0_ref = {T_ck_c0_gt[0] * Sophus::SE3d::exp(perturb),
                                           T_ck_c0_gt[1] * Sophus::SE3d::exp(-perturb)};
  const double expected_drift = perturb.norm() + perturb.norm();

  // Board.
  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});

  // 8 well-excited rig poses — both translation AND rotation vary across all 3 axes.
  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)),
       Eigen::Vector3d(-0.20, -0.20, 1.3)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.25, 0.18, -0.08)),
       Eigen::Vector3d(-0.25, -0.15, 1.1)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.08, 0.30, 0.12)),
       Eigen::Vector3d(-0.15, -0.25, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.18, 0.12, -0.20)),
       Eigen::Vector3d(-0.30, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.12, -0.22, 0.18)),
       Eigen::Vector3d(-0.20, -0.28, 1.4)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, -0.30, -0.10)),
       Eigen::Vector3d(-0.10, -0.18, 1.6)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.05, 0.25)),
       Eigen::Vector3d(-0.28, -0.05, 1.25)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.22, 0.20, -0.05)),
       Eigen::Vector3d(-0.18, -0.30, 1.35)}};

  // Reference intrinsics = ground-truth (held fixed during online re-estimation).
  std::vector<Eigen::VectorXd> intr_ref(3, Eigen::VectorXd(4));
  for (int c = 0; c < 3; ++c) {
    intr_ref[c] << cams[c].params()[0], cams[c].params()[1],
                   cams[c].params()[2], cams[c].params()[3];
  }
  std::vector<CameraFactory> facs(3, pinholeFactory());

  OnlineExtrinsicTracker tracker(facs, intr_ref, T_ck_c0_ref);
  for (const Sophus::SE3d& T_c0_w : gt_poses)
    tracker.addFrame(synthFrame(cams, T_ck_c0_gt, T_c0_w, board), T_c0_w);

  ExtrinsicEmission e = tracker.tryEmit(/*min_confidence=*/1e-6);
  CF_EXPECT_TRUE(e.emitted);
  CF_EXPECT_TRUE(!e.refused_for_motion);

  // Tracker recovered the GROUND-TRUTH extrinsics (consistent with the observations).
  CF_EXPECT_TRUE(e.extrinsics.size() == 2);
  for (int k = 0; k < 2; ++k) {
    Eigen::Matrix<double, 6, 1> err = (e.extrinsics[k] * T_ck_c0_gt[k].inverse()).log();
    CF_EXPECT_TRUE(err.norm() < 1e-3);
  }
  // Drift vs reference is the magnitude of the deliberate perturbation we baked in (within
  // a wide tolerance — drift sums two SE3 tangents).
  CF_EXPECT_TRUE(e.drift > 0.5 * expected_drift);
  CF_EXPECT_TRUE(e.drift < 1.5 * expected_drift);
  CF_EXPECT_TRUE(e.confidence > 0.0);
}

// Pure translation along one axis, zero rotation: the 6-axis motion gate refuses BEFORE
// solving and names the 5 unexcited axes (3 rotation + 2 translation).
CF_TEST(online_extrinsic_tracker_refuses_degenerate_motion) {
  std::vector<PinholeCamera> cams = {PinholeCamera(500, 500, 320, 240),
                                     PinholeCamera(510, 508, 322, 238)};
  std::vector<Sophus::SE3d> T_ck_c0_gt = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)),
       Eigen::Vector3d(-0.10, 0.005, 0.002)}};
  std::vector<Sophus::SE3d> T_ck_c0_ref = T_ck_c0_gt;

  std::vector<Vec3> board;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});

  // Translate along world x only; no rotation. (Rotation log = 0; only one translation
  // axis varies.)
  std::vector<Sophus::SE3d> gt_poses;
  for (int i = 0; i < 6; ++i)
    gt_poses.emplace_back(Sophus::SO3d(),
                          Eigen::Vector3d(-0.20 + 0.05 * i, -0.20, 1.3));

  std::vector<Eigen::VectorXd> intr_ref(2, Eigen::VectorXd(4));
  for (int c = 0; c < 2; ++c) {
    intr_ref[c] << cams[c].params()[0], cams[c].params()[1],
                   cams[c].params()[2], cams[c].params()[3];
  }
  std::vector<CameraFactory> facs(2, pinholeFactory());

  OnlineExtrinsicTracker tracker(facs, intr_ref, T_ck_c0_ref);
  for (const Sophus::SE3d& T_c0_w : gt_poses)
    tracker.addFrame(synthFrame(cams, T_ck_c0_gt, T_c0_w, board), T_c0_w);

  ExtrinsicEmission e = tracker.tryEmit(/*min_confidence=*/1e-6);
  CF_EXPECT_TRUE(!e.emitted);
  CF_EXPECT_TRUE(e.refused_for_motion);
  // Expect 5 unexcited axes: 3 rotation + 2 translation (only x varies).
  CF_EXPECT_TRUE(e.unexcited_axes.size() == 5);
  // Check the unexcited axes ARE rotation x/y/z + translation y/z (not trans_x).
  auto has = [&](const std::string& s) {
    return std::find(e.unexcited_axes.begin(), e.unexcited_axes.end(), s)
           != e.unexcited_axes.end();
  };
  CF_EXPECT_TRUE(has("rot_x"));
  CF_EXPECT_TRUE(has("rot_y"));
  CF_EXPECT_TRUE(has("rot_z"));
  CF_EXPECT_TRUE(has("trans_y"));
  CF_EXPECT_TRUE(has("trans_z"));
  CF_EXPECT_TRUE(!has("trans_x"));  // only excited axis
}
