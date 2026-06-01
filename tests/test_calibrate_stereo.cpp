#include "calibforge/calibrate_stereo.hpp"

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/manifold.hpp"
#include "calibforge/observability.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/stereo_reprojection_residual.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::assessObservability;
using calibforge::calibrateStereo;
using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::SE3Param;
using calibforge::StereoReprojectionResidual;
using calibforge::StereoResult;
using calibforge::StereoView;
using calibforge::Vec2;
using calibforge::Vec3;
using Eigen::VectorXd;

static std::unique_ptr<CameraModel> mkPinhole(const VectorXd& q) {
  return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
}

// Manifold-aware FD of the stereo residual's three Jacobians (the composition through R01).
CF_TEST(stereo_residual_jacobians_match_finite_difference) {
  std::array<double, 4> intr = {510, 508, 322, 238};
  Sophus::SE3d T01(Sophus::SO3d::exp(Eigen::Vector3d(0.03, -0.02, 0.01)),
                   Eigen::Vector3d(-0.10, 0.01, 0.005));
  Sophus::SE3d Twc0(Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.05)),
                    Eigen::Vector3d(-0.15, -0.10, 1.3));
  std::array<double, 7> e01{}, ewc0{};
  SE3Param::store(T01, e01.data());
  SE3Param::store(Twc0, ewc0.data());

  const Vec3 X{0.15, -0.2, 0.0};
  StereoReprojectionResidual block(mkPinhole, 4, X, Vec2{50.0, 60.0});

  double Ji[8], Je01[12], Jewc0[12];
  double* jacs[3] = {Ji, Je01, Jewc0};
  double r0[2];
  const double* params[3] = {intr.data(), e01.data(), ewc0.data()};
  block.evaluate(params, r0, jacs);

  auto residual = [&](const double* in, const double* a, const double* b, double out[2]) {
    const double* ps[3] = {in, a, b};
    block.evaluate(ps, out, nullptr);
  };
  const double h = 1e-6;
  SE3Param se3;

  for (int k = 0; k < 4; ++k) {  // intrinsics1 (Euclidean)
    std::array<double, 4> ip = intr, im = intr;
    ip[k] += h; im[k] -= h;
    double rp[2], rm[2];
    residual(ip.data(), e01.data(), ewc0.data(), rp);
    residual(im.data(), e01.data(), ewc0.data(), rm);
    CF_EXPECT_NEAR(Ji[k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Ji[4 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
  for (int k = 0; k < 6; ++k) {  // T_c1c0 (SE3 tangent)
    double dp[6] = {0}, dm[6] = {0};
    dp[k] = h; dm[k] = -h;
    std::array<double, 7> ap{}, am{};
    se3.retract(e01.data(), dp, ap.data());
    se3.retract(e01.data(), dm, am.data());
    double rp[2], rm[2];
    residual(intr.data(), ap.data(), ewc0.data(), rp);
    residual(intr.data(), am.data(), ewc0.data(), rm);
    CF_EXPECT_NEAR(Je01[k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Je01[6 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
  for (int k = 0; k < 6; ++k) {  // T_wc0 (SE3 tangent), composed through R01
    double dp[6] = {0}, dm[6] = {0};
    dp[k] = h; dm[k] = -h;
    std::array<double, 7> bp{}, bm{};
    se3.retract(ewc0.data(), dp, bp.data());
    se3.retract(ewc0.data(), dm, bm.data());
    double rp[2], rm[2];
    residual(intr.data(), e01.data(), bp.data(), rp);
    residual(intr.data(), e01.data(), bm.data(), rm);
    CF_EXPECT_NEAR(Jewc0[k], (rp[0] - rm[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(Jewc0[6 + k], (rp[1] - rm[1]) / (2 * h), 1e-4);
  }
}

static Sophus::SE3d perturbPose(const Sophus::SE3d& T, double s = 1.0) {
  Eigen::Matrix<double, 6, 1> d;
  d << 0.02 * s, -0.02 * s, 0.02 * s, 0.02 * s, -0.01 * s, 0.01 * s;
  return T * Sophus::SE3d::exp(d);
}

// Recover both intrinsics, the shared cam0->cam1 extrinsic, and per-view poses from
// synthetic noise-free stereo observations.
CF_TEST(stereo_recovers_intrinsics_extrinsic_and_poses) {
  PinholeCamera cam0(500, 500, 320, 240);
  PinholeCamera cam1(510, 508, 322, 238);
  const Sophus::SE3d T01(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)),
                         Eigen::Vector3d(-0.10, 0.005, 0.002));  // Xc1 = T01 * Xc0

  std::vector<Vec3> board;
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) board.push_back(Vec3{c * 0.1, r * 0.1, 0.0});
  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.10, 0.05)), Eigen::Vector3d(-0.20, -0.20, 1.3)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.25, 0.18, -0.08)), Eigen::Vector3d(-0.25, -0.15, 1.1)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.08, 0.30, 0.12)), Eigen::Vector3d(-0.15, -0.25, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.18, 0.12, -0.20)), Eigen::Vector3d(-0.30, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.12, -0.22, 0.18)), Eigen::Vector3d(-0.20, -0.28, 1.4)}};

  std::vector<StereoView> views;
  for (const auto& T : gt_poses) {
    StereoView v;
    for (const auto& X : board) {
      Eigen::Vector3d Xc0 = T * Eigen::Vector3d(X[0], X[1], X[2]);
      Eigen::Vector3d Xc1 = T01 * Xc0;
      v.object_points.push_back(X);
      v.image_points0.push_back(cam0.project(Vec3{Xc0.x(), Xc0.y(), Xc0.z()}));
      v.image_points1.push_back(cam1.project(Vec3{Xc1.x(), Xc1.y(), Xc1.z()}));
    }
    views.push_back(v);
  }

  VectorXd i0(4), i1(4);
  i0 << 480, 520, 312, 250;
  i1 << 525, 495, 330, 228;
  Sophus::SE3d T01_init = perturbPose(T01);
  std::vector<Sophus::SE3d> poses0;
  for (const auto& T : gt_poses) poses0.push_back(perturbPose(T));

  LmOptions opts;
  opts.max_iterations = 300;
  StereoResult res = calibrateStereo(views, i0, i1, T01_init, poses0, mkPinhole, mkPinhole, opts);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(res.summary.final_cost < 1e-8);
  CF_EXPECT_NEAR(res.intrinsics0[0], 500, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics0[2], 320, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics1[0], 510, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics1[1], 508, 1e-2);
  CF_EXPECT_NEAR(res.intrinsics1[2], 322, 1e-2);
  // Relative extrinsic recovered (translation and rotation).
  Eigen::Vector3d dt = res.T_cam1_cam0.translation() - T01.translation();
  Eigen::Matrix<double, 6, 1> dlog = (res.T_cam1_cam0 * T01.inverse()).log();
  CF_EXPECT_TRUE(dt.norm() < 1e-3);
  CF_EXPECT_TRUE(dlog.norm() < 1e-3);

  // The stereo problem is well conditioned with these tilted views.
  CF_EXPECT_TRUE(assessObservability(res.information).observable);
}
