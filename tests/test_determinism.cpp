// Determinism / parity test for the CPU calibration path. The same inputs must produce
// bit-identical outputs across consecutive runs — a precondition for the edge↔server parity
// test (SPIKES.md §D.3) that requires identical params across Jetson and server. This test
// catches non-determinism INSIDE the CPU half (thread races, hash-map iteration order,
// floating-point sensitivity in the LM loop) before the hardware parity test ever runs.
//
// Scope: CPU only. The bf16 / FP64 + Jetson-vs-server numerical parity (the actual goal of
// SPIKES.md §D.3) requires hardware not on the CI runner; that lands as a separate gated
// job (.github/workflows/ci-parity-edge.yml.disabled) when a Jetson runner is available.

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"
#include "calibforge/calibrate_stereo.hpp"
#include "calibforge/camera_model.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::calibrateSingleCamera;
using calibforge::calibrateStereo;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::SingleCameraResult;
using calibforge::StereoResult;
using calibforge::StereoView;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;
using Eigen::VectorXd;

namespace {

CameraFactory pinholeFactory() {
  return [](const VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
}

std::vector<View> synthSingleCamViews() {
  PinholeCamera cam(500.0, 510.0, 320.0, 240.0);
  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.05, r * 0.05, 0.0});
  std::vector<Sophus::SE3d> poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)),
       Eigen::Vector3d(-0.10, -0.10, 0.7)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.15, 0.20, -0.08)),
       Eigen::Vector3d(-0.15, -0.05, 0.6)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, 0.18, 0.15)),
       Eigen::Vector3d(-0.08, -0.15, 0.8)}};
  std::vector<View> views;
  for (const Sophus::SE3d& T : poses) {
    View v;
    v.object_points = board;
    for (const Vec3& X : board) {
      const Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
      v.image_points.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    }
    views.push_back(v);
  }
  return views;
}

}  // namespace

// Run the SAME calibrateSingleCamera call twice; expect bit-identical intrinsic + pose
// outputs. The LM loop, manifold retractions, Jacobian assembly, and DenseProblem ordering
// all need to be deterministic for this to hold.
CF_TEST(determinism_calibrate_single_runs_are_bit_identical) {
  const std::vector<View> views = synthSingleCamViews();
  VectorXd intr0(4);
  intr0 << 480, 520, 312, 250;
  std::vector<Sophus::SE3d> pose0;
  pose0.assign(views.size(),
               Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(-0.05, -0.05, 0.65)));

  LmOptions opts;
  opts.max_iterations = 200;

  const SingleCameraResult a =
      calibrateSingleCamera(views, intr0, pose0, pinholeFactory(), opts);
  const SingleCameraResult b =
      calibrateSingleCamera(views, intr0, pose0, pinholeFactory(), opts);

  CF_EXPECT_TRUE(a.summary.iterations == b.summary.iterations);
  CF_EXPECT_TRUE(a.summary.final_cost == b.summary.final_cost);  // bit-equal, not "close"
  CF_EXPECT_TRUE(a.intrinsics.size() == b.intrinsics.size());
  for (int i = 0; i < a.intrinsics.size(); ++i)
    CF_EXPECT_TRUE(a.intrinsics[i] == b.intrinsics[i]);
  CF_EXPECT_TRUE(a.poses.size() == b.poses.size());
  for (std::size_t k = 0; k < a.poses.size(); ++k) {
    const Eigen::Matrix4d Ma = a.poses[k].matrix();
    const Eigen::Matrix4d Mb = b.poses[k].matrix();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) CF_EXPECT_TRUE(Ma(r, c) == Mb(r, c));
  }
}

// Same idea for stereo — a richer problem (intrinsics + intrinsics + extrinsic + per-view
// poses + StereoReprojectionResidual composing through the extrinsic) — catches a wider
// class of non-determinism in the residual assembly.
CF_TEST(determinism_calibrate_stereo_runs_are_bit_identical) {
  PinholeCamera cam0(500.0, 510.0, 320.0, 240.0);
  PinholeCamera cam1(495.0, 505.0, 318.0, 242.0);
  const Sophus::SE3d T_c1_c0(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)),
                             Eigen::Vector3d(-0.10, 0.005, 0.002));

  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.05, r * 0.05, 0.0});
  std::vector<Sophus::SE3d> poses_gt = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)),
       Eigen::Vector3d(-0.10, -0.10, 0.7)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.15, 0.20, -0.08)),
       Eigen::Vector3d(-0.15, -0.05, 0.6)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, 0.18, 0.15)),
       Eigen::Vector3d(-0.08, -0.15, 0.8)}};

  std::vector<StereoView> views;
  for (const Sophus::SE3d& T : poses_gt) {
    StereoView v;
    v.object_points = board;
    for (const Vec3& X : board) {
      const Eigen::Vector3d Xc0 = T * Eigen::Vector3d(X[0], X[1], X[2]);
      v.image_points0.push_back(cam0.project(Vec3{Xc0.x(), Xc0.y(), Xc0.z()}));
      const Eigen::Vector3d Xc1 = T_c1_c0 * Xc0;
      v.image_points1.push_back(cam1.project(Vec3{Xc1.x(), Xc1.y(), Xc1.z()}));
    }
    views.push_back(v);
  }

  VectorXd intr0_a(4), intr1_a(4);
  intr0_a << 480, 520, 312, 250;
  intr1_a << 510, 495, 322, 232;
  const Sophus::SE3d extr0(Sophus::SO3d(), Eigen::Vector3d(-0.08, 0.0, 0.0));
  std::vector<Sophus::SE3d> pose0(views.size(),
                                  Sophus::SE3d(Sophus::SO3d(),
                                               Eigen::Vector3d(-0.05, -0.05, 0.65)));
  LmOptions opts;
  opts.max_iterations = 200;

  const StereoResult a = calibrateStereo(views, intr0_a, intr1_a, extr0, pose0,
                                         pinholeFactory(), pinholeFactory(), opts);
  const StereoResult b = calibrateStereo(views, intr0_a, intr1_a, extr0, pose0,
                                         pinholeFactory(), pinholeFactory(), opts);

  CF_EXPECT_TRUE(a.summary.iterations == b.summary.iterations);
  CF_EXPECT_TRUE(a.summary.final_cost == b.summary.final_cost);
  CF_EXPECT_TRUE(a.intrinsics0.size() == b.intrinsics0.size());
  for (int i = 0; i < a.intrinsics0.size(); ++i)
    CF_EXPECT_TRUE(a.intrinsics0[i] == b.intrinsics0[i]);
  for (int i = 0; i < a.intrinsics1.size(); ++i)
    CF_EXPECT_TRUE(a.intrinsics1[i] == b.intrinsics1[i]);
  const Eigen::Matrix4d Ma = a.T_cam1_cam0.matrix();
  const Eigen::Matrix4d Mb = b.T_cam1_cam0.matrix();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) CF_EXPECT_TRUE(Ma(r, c) == Mb(r, c));
}
