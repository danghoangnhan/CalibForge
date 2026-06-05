#pragma once
//
// CalibForge pipeline — FULL cam-IMU calibration (header-only, CPU).
//
// The deeper follow-up to calibrate_cam_imu.hpp (which recovers only the rotation extrinsic
// R_ci + time offset t_d from the gyro/angular-velocity correspondence). This pipeline wires
// the Forster IMU preintegration FACTOR (solve/imu_preintegration_residual.hpp) into a joint
// estimate of:
//   * the SPATIAL extrinsic  T_ic = (R_ci, p_ci)   — including the translation LEVER ARM p_ci
//     that pure-rotation init cannot observe,
//   * the per-keyframe IMU nav-states (R_wi, p_wi, v_wi),
//   * the gyro / accel biases (b_g, b_a),
//   * the gravity vector g_w (world frame),
// from MEASURED camera-in-world poses (a prior target-based solve) + the raw IMU stream.
//
// Two residual families share one DenseProblem (docs/RESEARCH.md Theme 2: one optimizer, many
// residual types):
//   * CamImuPoseResidual  — ties each measured T_wc to (R_wi, p_wi) through (R_ci, p_ci);
//     stacked over varied orientation this makes p_ci observable (hand-eye AX=XB translation).
//   * ImuPreintegrationFactorResidual — chains consecutive nav-states through a preintegrated
//     IMU window, supplying the metric scale / gravity / bias that pin p_wi and v_wi.
//
// Observability (RULE #2): the returned `information` is the calibration block marginalized
// over the nuisance nav-states via the Schur complement S = H_cc - H_cn H_nn^{-1} H_nc, so
// assessObservability(result.information) judges whether the EXTRINSIC + biases (+ gravity)
// are actually constrained by the motion — never the inflated full-state matrix. Re-implements
// iKalibr's BSD-3 continuous-time idea over our discrete factor; never copies GPL Kalibr.

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/cam_imu_pose_residual.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/imu_preintegration_residual.hpp"
#include "calibforge/imu_preintegrator.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/manifold.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace calibforge {

// One raw IMU reading and the interval dt to the next sample (forward-Euler step).
struct ImuSample {
  Eigen::Vector3d gyro;   // raw body-frame angular velocity (rad/s), bias-corrupted
  Eigen::Vector3d accel;  // raw body-frame specific force (m/s^2), bias-corrupted
  double dt;              // > 0
};

struct CamImuFullOptions {
  LmOptions lm;
  bool estimate_gravity = true;  // false => g_w held at g_w_init (gravity-aligned world)
  Eigen::Vector3d g_w_init = Eigen::Vector3d(0.0, 0.0, -9.81);
  Eigen::Vector3d b_g_init = Eigen::Vector3d::Zero();
  Eigen::Vector3d b_a_init = Eigen::Vector3d::Zero();
};

struct CamImuFullResult {
  Sophus::SO3d R_ci;             // camera -> IMU rotation
  Eigen::Vector3d p_ci;          // camera origin in IMU frame (the lever arm)
  Eigen::Vector3d b_g, b_a, g_w;
  std::vector<Sophus::SO3d> R_wi;
  std::vector<Eigen::Vector3d> p_wi, v_wi;
  LmSummary summary;
  Eigen::MatrixXd information;            // calibration block, Schur-marginalized over nav-states
  std::vector<std::string> param_names;  // row labels for `information` (feed to parameterUncertainty)
  int num_residuals = 0;
};

// cam_poses_world[k] : measured camera-in-world pose T_wc at keyframe k (camera->world). If your
//   poses are world->camera (SingleCameraResult stores those), pass each .inverse().
// imu_between[k]     : IMU samples spanning keyframe k -> k+1 (size must be cam_poses_world-1).
inline CamImuFullResult calibrateCamImuFull(
    const std::vector<Sophus::SE3d>& cam_poses_world,
    const std::vector<std::vector<ImuSample>>& imu_between,
    const Sophus::SO3d& R_ci_init, const Eigen::Vector3d& p_ci_init,
    const CamImuFullOptions& opts = CamImuFullOptions{}) {
  const int nkf = static_cast<int>(cam_poses_world.size());
  CamImuFullResult res;
  if (nkf < 2 || static_cast<int>(imu_between.size()) != nkf - 1) return res;

  // --- Parameter-block storage (stable addresses for the problem's lifetime) ---------------
  std::vector<std::array<double, 4>> R_wi_s(static_cast<std::size_t>(nkf));
  std::vector<std::array<double, 3>> p_wi_s(static_cast<std::size_t>(nkf));
  std::vector<std::array<double, 3>> v_wi_s(static_cast<std::size_t>(nkf));
  std::array<double, 4> R_ci_s{};
  std::array<double, 3> p_ci_s{p_ci_init.x(), p_ci_init.y(), p_ci_init.z()};
  std::array<double, 3> bg_s{opts.b_g_init.x(), opts.b_g_init.y(), opts.b_g_init.z()};
  std::array<double, 3> ba_s{opts.b_a_init.x(), opts.b_a_init.y(), opts.b_a_init.z()};
  std::array<double, 3> gw_s{opts.g_w_init.x(), opts.g_w_init.y(), opts.g_w_init.z()};
  SO3Param::store(R_ci_init, R_ci_s.data());

  // Per-segment duration (sum of dt) for the velocity finite-difference initializer.
  std::vector<double> seg_T(static_cast<std::size_t>(nkf - 1), 0.0);
  for (int k = 0; k < nkf - 1; ++k)
    for (const ImuSample& s : imu_between[k]) seg_T[k] += s.dt;

  // Nav-state init: R_wi = R_wc R_ci^{-1}; p_wi = p_wc - R_wi p_ci; v from position differences.
  const Sophus::SO3d Rci0_inv = R_ci_init.inverse();
  for (int k = 0; k < nkf; ++k) {
    const Sophus::SO3d R_wi = cam_poses_world[k].so3() * Rci0_inv;
    const Eigen::Vector3d p_wi = cam_poses_world[k].translation() - R_wi * p_ci_init;
    SO3Param::store(R_wi, R_wi_s[k].data());
    for (int i = 0; i < 3; ++i) p_wi_s[k][i] = p_wi[i];
  }
  for (int k = 0; k < nkf; ++k) {
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    if (k < nkf - 1 && seg_T[k] > 0.0) {
      const Eigen::Vector3d pa(p_wi_s[k][0], p_wi_s[k][1], p_wi_s[k][2]);
      const Eigen::Vector3d pb(p_wi_s[k + 1][0], p_wi_s[k + 1][1], p_wi_s[k + 1][2]);
      v = (pb - pa) / seg_T[k];
    } else if (k > 0) {
      v = Eigen::Vector3d(v_wi_s[k - 1][0], v_wi_s[k - 1][1], v_wi_s[k - 1][2]);
    }
    for (int i = 0; i < 3; ++i) v_wi_s[k][i] = v[i];
  }

  // --- Preintegrators (must outlive the solve; reserve so addresses stay stable) -----------
  std::vector<ImuPreintegrator> preints;
  preints.reserve(static_cast<std::size_t>(nkf - 1));
  for (int k = 0; k < nkf - 1; ++k) {
    preints.emplace_back(opts.b_g_init, opts.b_a_init);
    for (const ImuSample& s : imu_between[k]) preints.back().addSample(s.gyro, s.accel, s.dt);
  }

  // --- Assemble the problem. Register NAV blocks first, CALIBRATION blocks last, so the
  //     trailing sub-matrix of J^T J is exactly the calibration block we Schur-marginalize. ---
  DenseProblem problem;
  auto so3 = std::make_shared<SO3Param>();
  auto e3 = std::make_shared<EuclideanParam>(3);
  for (int k = 0; k < nkf; ++k) {
    problem.addParameterBlock(R_wi_s[k].data(), so3);
    problem.addParameterBlock(p_wi_s[k].data(), e3);
    problem.addParameterBlock(v_wi_s[k].data(), e3);
  }
  problem.addParameterBlock(R_ci_s.data(), so3);
  problem.addParameterBlock(p_ci_s.data(), e3);
  problem.addParameterBlock(bg_s.data(), e3);
  problem.addParameterBlock(ba_s.data(), e3);
  problem.addParameterBlock(gw_s.data(), e3);
  if (!opts.estimate_gravity) problem.setParameterBlockConstant(gw_s.data());

  for (int k = 0; k < nkf; ++k)
    problem.addResidualBlock(std::make_unique<CamImuPoseResidual>(cam_poses_world[k]),
                             {R_wi_s[k].data(), p_wi_s[k].data(), R_ci_s.data(), p_ci_s.data()});
  for (int k = 0; k < nkf - 1; ++k)
    problem.addResidualBlock(
        std::make_unique<ImuPreintegrationFactorResidual>(&preints[static_cast<std::size_t>(k)]),
        {R_wi_s[k].data(), p_wi_s[k].data(), v_wi_s[k].data(), R_wi_s[k + 1].data(),
         p_wi_s[k + 1].data(), v_wi_s[k + 1].data(), bg_s.data(), ba_s.data(), gw_s.data()});

  res.summary = problem.solveLm(opts.lm);

  // --- Extract estimates ------------------------------------------------------------------
  res.R_ci = SO3Param::load(R_ci_s.data());
  res.p_ci = Eigen::Vector3d(p_ci_s[0], p_ci_s[1], p_ci_s[2]);
  res.b_g = Eigen::Vector3d(bg_s[0], bg_s[1], bg_s[2]);
  res.b_a = Eigen::Vector3d(ba_s[0], ba_s[1], ba_s[2]);
  res.g_w = Eigen::Vector3d(gw_s[0], gw_s[1], gw_s[2]);
  res.R_wi.resize(static_cast<std::size_t>(nkf));
  res.p_wi.resize(static_cast<std::size_t>(nkf));
  res.v_wi.resize(static_cast<std::size_t>(nkf));
  for (int k = 0; k < nkf; ++k) {
    res.R_wi[k] = SO3Param::load(R_wi_s[k].data());
    res.p_wi[k] = Eigen::Vector3d(p_wi_s[k][0], p_wi_s[k][1], p_wi_s[k][2]);
    res.v_wi[k] = Eigen::Vector3d(v_wi_s[k][0], v_wi_s[k][1], v_wi_s[k][2]);
  }
  res.num_residuals = problem.numResiduals();

  // --- Schur-marginalize the nav-states out of J^T J, leaving the calibration block --------
  // Nav tangent columns (registered first): 9 per keyframe. Calibration columns trail them.
  const Eigen::MatrixXd& H = problem.informationMatrix();
  const int total = problem.tangentSize();
  const int nnav = 9 * nkf;
  const int ncal = total - nnav;  // 15 (with gravity) or 12 (gravity held constant)
  if (ncal > 0 && nnav > 0 && total == H.rows()) {
    const Eigen::MatrixXd Hnn = H.topLeftCorner(nnav, nnav);
    const Eigen::MatrixXd Hnc = H.topRightCorner(nnav, ncal);
    const Eigen::MatrixXd Hcc = H.bottomRightCorner(ncal, ncal);
    // S = Hcc - Hnc^T Hnn^{-1} Hnc. Hnn is SPD (full nav excitation) => LDLT solve.
    res.information = Hcc - Hnc.transpose() * Hnn.ldlt().solve(Hnc);
  } else if (ncal > 0) {
    res.information = H.bottomRightCorner(ncal, ncal);
  }
  const char* base[] = {"R_ci_x", "R_ci_y", "R_ci_z", "p_ci_x", "p_ci_y", "p_ci_z",
                        "b_g_x",  "b_g_y",  "b_g_z",  "b_a_x",  "b_a_y",  "b_a_z"};
  for (const char* nm : base) res.param_names.emplace_back(nm);
  if (opts.estimate_gravity)
    for (const char* nm : {"g_w_x", "g_w_y", "g_w_z"}) res.param_names.emplace_back(nm);

  return res;
}

}  // namespace calibforge
