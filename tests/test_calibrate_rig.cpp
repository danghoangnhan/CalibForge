#include "calibforge/calibrate_rig.hpp"

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/observability.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::calibrateRig;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::RigResult;
using calibforge::RigView;
using calibforge::Vec2;
using calibforge::Vec3;
using Eigen::VectorXd;

static CameraFactory pinholeFactory() {
  return [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
}

static Sophus::SE3d perturbPose(const Sophus::SE3d& T) {
  Eigen::Matrix<double, 6, 1> d;
  d << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  return T * Sophus::SE3d::exp(d);
}

// A 3-camera rig: recover all intrinsics, both shared extrinsics T_ck_c0, and per-view poses
// from synthetic noise-free observations (each camera sees the full board).
CF_TEST(rig_recovers_three_camera_intrinsics_extrinsics_poses) {
  std::vector<PinholeCamera> cams = {
      PinholeCamera(500, 500, 320, 240), PinholeCamera(510, 508, 322, 238),
      PinholeCamera(495, 497, 318, 242)};
  std::vector<Sophus::SE3d> T_ck_c0 = {  // cam1<-cam0, cam2<-cam0
      {Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)), Eigen::Vector3d(-0.10, 0.005, 0.002)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.015, 0.02, -0.01)), Eigen::Vector3d(0.10, -0.004, 0.003)}};

  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)), Eigen::Vector3d(-0.20, -0.20, 1.3)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.25, 0.18, -0.08)), Eigen::Vector3d(-0.25, -0.15, 1.1)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.08, 0.30, 0.12)), Eigen::Vector3d(-0.15, -0.25, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.18, 0.12, -0.20)), Eigen::Vector3d(-0.30, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.12, -0.22, 0.18)), Eigen::Vector3d(-0.20, -0.28, 1.4)}};

  std::vector<RigView> views;
  for (const auto& T : gt_poses) {
    RigView v;
    v.object_points = board;
    v.image_points.resize(3);
    for (const auto& X : board) {
      Eigen::Vector3d Xc0 = T * Eigen::Vector3d(X[0], X[1], X[2]);
      v.image_points[0].push_back(cams[0].project(Vec3{Xc0.x(), Xc0.y(), Xc0.z()}));
      for (int k = 0; k < 2; ++k) {
        Eigen::Vector3d Xck = T_ck_c0[k] * Xc0;
        v.image_points[k + 1].push_back(cams[k + 1].project(Vec3{Xck.x(), Xck.y(), Xck.z()}));
      }
    }
    views.push_back(v);
  }

  std::vector<VectorXd> intr0(3, VectorXd(4));
  intr0[0] << 480, 520, 312, 250;
  intr0[1] << 525, 495, 330, 228;
  intr0[2] << 505, 488, 326, 250;
  std::vector<Sophus::SE3d> extr0;
  for (const auto& T : T_ck_c0) extr0.push_back(perturbPose(T));
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : gt_poses) poses0.push_back(perturbPose(T));
  std::vector<CameraFactory> facs(3, pinholeFactory());

  LmOptions opts;
  opts.max_iterations = 300;
  RigResult res = calibrateRig(views, intr0, extr0, poses0, facs, opts);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-8);
  for (int c = 0; c < 3; ++c) {
    CF_EXPECT_NEAR(res.intrinsics[c][0], cams[c].params()[0], 1e-2);  // fx
    CF_EXPECT_NEAR(res.intrinsics[c][2], cams[c].params()[2], 1e-2);  // cx
  }
  for (int k = 0; k < 2; ++k) {
    Eigen::Vector3d dt = res.extrinsics[k].translation() - T_ck_c0[k].translation();
    Eigen::Matrix<double, 6, 1> dl = (res.extrinsics[k] * T_ck_c0[k].inverse()).log();
    CF_EXPECT_TRUE(dt.norm() < 1e-3);
    CF_EXPECT_TRUE(dl.norm() < 1e-3);
  }
  CF_EXPECT_TRUE(calibforge::assessObservability(res.information).observable);
}
