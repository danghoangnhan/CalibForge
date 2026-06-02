// Issue #18 — confidence/datum handling for multi-block problems: per-parameter uncertainty
// for rig extrinsics, and naming the unobservable directions of a degenerate hand-eye.
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_hand_eye.hpp"
#include "calibforge/calibrate_rig.hpp"
#include "calibforge/observability.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::calibrateHandEye;
using calibforge::calibrateRig;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::HandEyeResult;
using calibforge::HandEyeView;
using calibforge::LmOptions;
using calibforge::parameterUncertainty;
using calibforge::PinholeCamera;
using calibforge::RigResult;
using calibforge::RigView;
using calibforge::Vec2;
using calibforge::Vec3;
using Eigen::VectorXd;

static Sophus::SE3d perturb(const Sophus::SE3d& T) {
  Eigen::Matrix<double, 6, 1> d;
  d << 0.02, -0.015, 0.02, 0.02, -0.01, 0.015;
  return T * Sophus::SE3d::exp(d);
}

// A well-conditioned 2-camera rig reports finite, small uncertainty on the shared extrinsic.
CF_TEST(rig_reports_finite_extrinsic_sigmas) {
  PinholeCamera cam0(500, 500, 320, 240), cam1(510, 508, 322, 238);
  Sophus::SE3d T_c1_c0(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)),
                       Eigen::Vector3d(-0.10, 0.005, 0.002));
  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  std::vector<Sophus::SE3d> poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)), Eigen::Vector3d(-0.20, -0.20, 1.3)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.25, 0.18, -0.08)), Eigen::Vector3d(-0.25, -0.15, 1.1)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.08, 0.30, 0.12)), Eigen::Vector3d(-0.15, -0.25, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.18, 0.12, -0.20)), Eigen::Vector3d(-0.30, -0.10, 1.2)}};

  std::vector<RigView> views;
  for (const auto& T : poses) {
    RigView v;
    v.object_points = board;
    v.image_points.resize(2);
    for (const auto& X : board) {
      Eigen::Vector3d Xc0 = T * Eigen::Vector3d(X[0], X[1], X[2]);
      Eigen::Vector3d Xc1 = T_c1_c0 * Xc0;
      v.image_points[0].push_back(cam0.project(Vec3{Xc0.x(), Xc0.y(), Xc0.z()}));
      v.image_points[1].push_back(cam1.project(Vec3{Xc1.x(), Xc1.y(), Xc1.z()}));
    }
    views.push_back(v);
  }

  CameraFactory f = [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
  std::vector<VectorXd> intr0(2, VectorXd(4));
  intr0[0] << 480, 520, 312, 250;
  intr0[1] << 525, 495, 330, 228;
  std::vector<Sophus::SE3d> extr0 = {perturb(T_c1_c0)};
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : poses) poses0.push_back(perturb(T));

  LmOptions opts;
  opts.max_iterations = 300;
  RigResult res = calibrateRig(views, intr0, extr0, poses0, {f, f}, opts);
  CF_EXPECT_TRUE(res.summary.converged);

  // Tangent layout: intr0(0-3), intr1(4-7), extrinsic T_c1_c0 (8-13), then per-view poses.
  std::vector<std::string> names = {"fx0", "fy0", "cx0", "cy0", "fx1", "fy1", "cx1", "cy1",
                                    "ex_tx", "ex_ty", "ex_tz", "ex_rx", "ex_ry", "ex_rz"};
  auto u = parameterUncertainty(res.information, res.summary.final_cost, res.num_residuals, names);
  for (int i = 8; i <= 13; ++i) {  // the extrinsic block
    CF_EXPECT_TRUE(std::isfinite(u.sigma[i]));
  }
  // No extrinsic direction is flagged weak in a well-conditioned rig.
  for (const auto& w : u.weak_parameters) CF_EXPECT_TRUE(w.rfind("ex_", 0) != 0);
}

// A single-view hand-eye is degenerate: parameterUncertainty NAMES the unobservable directions.
CF_TEST(hand_eye_degenerate_names_weak_directions) {
  PinholeCamera cam(500, 500, 320, 240);
  Sophus::SE3d X(Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.20, 0.05)), Eigen::Vector3d(0.03, -0.02, 0.05));
  Sophus::SE3d Z(Sophus::SO3d::exp(Eigen::Vector3d(-0.10, 0.15, 0.20)), Eigen::Vector3d(0.40, 0.10, 0.20));
  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  Sophus::SE3d cam_from_target(Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)),
                              Eigen::Vector3d(-0.20, -0.20, 1.3));
  HandEyeView v;
  const Sophus::SE3d T_target_cam = cam_from_target.inverse();
  v.T_base_gripper = (Z * T_target_cam) * X.inverse();
  v.object_points = board;
  for (const auto& Xw : board) {
    Eigen::Vector3d Xc = cam_from_target * Eigen::Vector3d(Xw[0], Xw[1], Xw[2]);
    v.image_points.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
  }

  HandEyeResult res = calibrateHandEye({v}, perturb(X), perturb(Z), cam);
  std::vector<std::string> names = {"X_tx", "X_ty", "X_tz", "X_rx", "X_ry", "X_rz",
                                    "Z_tx", "Z_ty", "Z_tz", "Z_rx", "Z_ry", "Z_rz"};
  auto u = parameterUncertainty(res.information, res.summary.final_cost, res.num_residuals, names);
  CF_EXPECT_TRUE(!u.weak_parameters.empty());  // the unobservable X/Z directions are named
}
