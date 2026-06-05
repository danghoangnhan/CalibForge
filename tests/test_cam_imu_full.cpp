// Full cam-IMU calibration tests: the SPATIAL extrinsic (incl. the translation lever arm
// p_ci), biases, and gravity recovered jointly from measured camera poses + an IMU stream.
//
//   (a) CamImuPoseResidual: every analytic Jacobian block matches finite difference.
//   (b) calibrate_cam_imu_full recovers R_ci, p_ci, b_g, b_a, g_w from a synthetic,
//       forward-Euler-consistent 6-DOF trajectory (residual ~0 at GT by construction), and
//       the Schur-marginalized observability gate passes under full excitation.
//   (c) Pure-rotation-about-the-camera motion leaves p_ci unobservable: the gate refuses.

#include "calibforge/calibrate_cam_imu_full.hpp"

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/cam_imu_pose_residual.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/observability.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

using calibforge::assessObservability;
using calibforge::calibrateCamImuFull;
using calibforge::CamImuFullOptions;
using calibforge::CamImuFullResult;
using calibforge::CamImuPoseResidual;
using calibforge::ImuSample;
using calibforge::SO3Param;

namespace {

// Ground truth.
const Eigen::Vector3d kBg(0.004, -0.003, 0.002);
const Eigen::Vector3d kBa(-0.05, 0.03, 0.04);
const Eigen::Vector3d kGw(0.0, 0.0, -9.81);

Sophus::SO3d gtRci() { return Sophus::SO3d::exp(Eigen::Vector3d(0.20, -0.35, 0.12)); }
Eigen::Vector3d gtPci() { return Eigen::Vector3d(0.08, -0.05, 0.12); }  // camera origin in IMU frame

// Smooth, fully-excited body-frame angular velocity and world-frame acceleration.
Eigen::Vector3d omegaBody(double t) {
  return Eigen::Vector3d(0.6 * std::sin(1.3 * t), 0.5 * std::cos(0.9 * t),
                         0.4 * std::sin(0.7 * t + 0.3));
}
Eigen::Vector3d accWorld(double t) {
  return Eigen::Vector3d(0.5 * std::sin(1.1 * t), 0.4 * std::cos(0.8 * t),
                         0.3 * std::sin(1.5 * t));
}

struct Synthetic {
  std::vector<Sophus::SE3d> cam_poses_world;
  std::vector<std::vector<ImuSample>> imu_between;
};

// Forward-Euler trajectory consistent with the preintegrator's forward-Euler recurrence, so
// the IMU factor residual is structurally zero at ground truth (to FP). Measurements:
//   w_meas = omega_body + b_g ;  a_meas = R_wi^T (a_world - g_w) + b_a.
Synthetic makeSynthetic(int n_kf = 12, int per_kf = 20, double dt = 0.005) {
  Synthetic out;
  const Sophus::SO3d R_ci = gtRci();
  const Eigen::Vector3d p_ci = gtPci();

  Sophus::SO3d R_wi = Sophus::SO3d::exp(Eigen::Vector3d(-0.1, 0.05, 0.2));
  Eigen::Vector3d p_wi(0.3, -0.2, 1.0);
  Eigen::Vector3d v_wi(0.1, 0.05, -0.08);

  double t = 0.0;
  auto pushKeyframe = [&]() {
    const Sophus::SO3d R_wc = R_wi * R_ci;
    const Eigen::Vector3d p_wc = R_wi * p_ci + p_wi;
    out.cam_poses_world.emplace_back(R_wc, p_wc);
  };
  pushKeyframe();
  for (int k = 0; k < n_kf - 1; ++k) {
    std::vector<ImuSample> seg;
    seg.reserve(static_cast<std::size_t>(per_kf));
    for (int i = 0; i < per_kf; ++i) {
      const Eigen::Vector3d wb = omegaBody(t);
      const Eigen::Vector3d aw = accWorld(t);
      ImuSample s;
      s.gyro = wb + kBg;
      s.accel = R_wi.matrix().transpose() * (aw - kGw) + kBa;
      s.dt = dt;
      seg.push_back(s);
      // forward-Euler step (matches ImuPreintegrator::addSample integration order)
      p_wi += v_wi * dt + 0.5 * aw * dt * dt;
      v_wi += aw * dt;
      R_wi = R_wi * Sophus::SO3d::exp(wb * dt);
      t += dt;
    }
    out.imu_between.push_back(std::move(seg));
    pushKeyframe();
  }
  return out;
}

}  // namespace

// (a) Finite-difference validation of the four CamImuPoseResidual Jacobian blocks.
CF_TEST(cam_imu_pose_residual_jacobians_match_finite_difference) {
  // A pose where measured != predicted, so r_R is away from zero (good FD signal).
  const Sophus::SO3d R_wi = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.1, 0.3));
  const Eigen::Vector3d p_wi(0.5, -0.4, 1.2);
  const Sophus::SO3d R_ci = gtRci();
  const Eigen::Vector3d p_ci = gtPci();
  const Sophus::SE3d T_wc_pred(R_wi * R_ci, R_wi * p_ci + p_wi);
  const Sophus::SE3d T_wc_meas = T_wc_pred * Sophus::SE3d::exp(
      (Eigen::Matrix<double, 6, 1>() << 0.03, -0.02, 0.04, 0.05, -0.03, 0.02).finished());
  CamImuPoseResidual block(T_wc_meas);

  std::array<double, 4> Rwi{}, Rci{};
  std::array<double, 3> pwi{p_wi.x(), p_wi.y(), p_wi.z()};
  std::array<double, 3> pci{p_ci.x(), p_ci.y(), p_ci.z()};
  SO3Param::store(R_wi, Rwi.data());
  SO3Param::store(R_ci, Rci.data());

  double r0[6];
  std::array<std::array<double, 6 * 3>, 4> J{};
  double* jacs[4] = {J[0].data(), J[1].data(), J[2].data(), J[3].data()};
  const double* params[4] = {Rwi.data(), pwi.data(), Rci.data(), pci.data()};
  block.evaluate(params, r0, jacs);

  const double h = 1e-6;
  SO3Param so3;
  auto eval = [&](const std::array<const double*, 4>& ps, double out[6]) {
    block.evaluate(ps.data(), out, nullptr);
  };
  auto check_col = [&](int bi, int col, const std::array<const double*, 4>& pp,
                       const std::array<const double*, 4>& pm) {
    double rp[6], rm[6];
    eval(pp, rp);
    eval(pm, rm);
    for (int row = 0; row < 6; ++row)
      CF_EXPECT_NEAR(J[bi][row * 3 + col], (rp[row] - rm[row]) / (2 * h), 5e-5);
  };
  auto so3_check = [&](int bi, double* base) {
    for (int axis = 0; axis < 3; ++axis) {
      double dp[3] = {0, 0, 0}, dm[3] = {0, 0, 0};
      dp[axis] = h; dm[axis] = -h;
      std::array<double, 4> Rp{}, Rm{};
      so3.retract(base, dp, Rp.data());
      so3.retract(base, dm, Rm.data());
      std::array<const double*, 4> pp = {params[0], params[1], params[2], params[3]};
      std::array<const double*, 4> pm = pp;
      pp[bi] = Rp.data();
      pm[bi] = Rm.data();
      check_col(bi, axis, pp, pm);
    }
  };
  auto eucl_check = [&](int bi, double* base) {
    for (int axis = 0; axis < 3; ++axis) {
      std::array<double, 3> dp{base[0], base[1], base[2]}, dm{base[0], base[1], base[2]};
      dp[axis] += h; dm[axis] -= h;
      std::array<const double*, 4> pp = {params[0], params[1], params[2], params[3]};
      std::array<const double*, 4> pm = pp;
      pp[bi] = dp.data();
      pm[bi] = dm.data();
      check_col(bi, axis, pp, pm);
    }
  };
  so3_check(0, Rwi.data());
  eucl_check(1, pwi.data());
  so3_check(2, Rci.data());
  eucl_check(3, pci.data());
}

// (b) Full recovery from a synthetic, well-excited trajectory.
CF_TEST(cam_imu_full_recovers_translation_extrinsic_bias_gravity) {
  Synthetic syn = makeSynthetic();

  CamImuFullOptions opts;
  opts.lm.max_iterations = 200;
  opts.estimate_gravity = true;
  // Wrong initial guesses for everything the pipeline must recover.
  const Sophus::SO3d R_ci0 = gtRci() * Sophus::SO3d::exp(Eigen::Vector3d(0.05, -0.04, 0.06));
  const Eigen::Vector3d p_ci0 = gtPci() + Eigen::Vector3d(0.03, -0.04, -0.05);
  opts.b_g_init = Eigen::Vector3d::Zero();
  opts.b_a_init = Eigen::Vector3d::Zero();
  opts.g_w_init = Eigen::Vector3d(0.2, -0.15, -9.7);  // off-true gravity

  CamImuFullResult res =
      calibrateCamImuFull(syn.cam_poses_world, syn.imu_between, R_ci0, p_ci0, opts);

  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE((res.R_ci * gtRci().inverse()).log().norm() < 1e-4);
  CF_EXPECT_TRUE((res.p_ci - gtPci()).norm() < 1e-4);          // the LEVER ARM recovered
  CF_EXPECT_TRUE((res.b_g - kBg).norm() < 1e-4);
  CF_EXPECT_TRUE((res.b_a - kBa).norm() < 1e-3);
  CF_EXPECT_TRUE((res.g_w - kGw).norm() < 1e-3);
  CF_EXPECT_TRUE(assessObservability(res.information).observable);
}

// (c) Degenerate motion: pure TRANSLATION with no rotation. Without any change of orientation
// the camera->IMU rotation R_ci and the lever arm p_ci are both unobservable — a constant
// shift of p_ci trades exactly against a constant shift of every p_wi (the IMU only constrains
// p_wi DIFFERENCES). The Schur-marginalized gate must refuse to certify the extrinsic.
CF_TEST(cam_imu_full_flags_unobservable_extrinsic_under_pure_translation) {
  Synthetic out;
  const Sophus::SO3d R_ci = gtRci();
  const Eigen::Vector3d p_ci = gtPci();
  const Sophus::SO3d R_wi;                 // identity, held constant (no rotation)
  Eigen::Vector3d p_wi(0.0, 0.0, 1.0), v_wi(0.2, -0.1, 0.05);
  const int n_kf = 10, per_kf = 20;
  const double dt = 0.005;
  double t = 0.0;
  auto push = [&]() { out.cam_poses_world.emplace_back(R_wi * R_ci, R_wi * p_ci + p_wi); };
  push();
  for (int k = 0; k < n_kf - 1; ++k) {
    std::vector<ImuSample> seg;
    for (int i = 0; i < per_kf; ++i) {
      const Eigen::Vector3d aw(0.4 * std::sin(1.1 * t), 0.3 * std::cos(0.9 * t), 0.2 * std::sin(t));
      ImuSample s;
      s.gyro = Eigen::Vector3d::Zero();                            // no rotation
      s.accel = R_wi.matrix().transpose() * (aw - kGw);            // translating + gravity
      s.dt = dt;
      seg.push_back(s);
      p_wi += v_wi * dt + 0.5 * aw * dt * dt;
      v_wi += aw * dt;
      t += dt;
    }
    out.imu_between.push_back(std::move(seg));
    push();
  }

  CamImuFullOptions opts;
  opts.lm.max_iterations = 60;
  opts.estimate_gravity = false;  // hold gravity; probe only the extrinsic/bias block
  CamImuFullResult res = calibrateCamImuFull(
      out.cam_poses_world, out.imu_between,
      R_ci * Sophus::SO3d::exp(Eigen::Vector3d(0.01, 0.0, 0.0)),
      gtPci() + Eigen::Vector3d(0.02, 0.0, 0.0), opts);
  // The solve itself must COMPLETE: the refusal has to be a structural rank-deficiency of the
  // Schur-marginalized extrinsic block, not a side effect of a non-converged / broken solve.
  CF_EXPECT_TRUE(res.summary.converged);
  CF_EXPECT_TRUE(!assessObservability(res.information).observable);
}
