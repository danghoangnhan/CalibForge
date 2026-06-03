// Synthetic-trajectory tests for the Forster 9-dim IMU preintegration factor.
//
// Goal:
//   (a) Residual evaluates to ~0 at ground-truth states + biases + gravity.
//   (b) Every analytic Jacobian block matches a finite-difference re-evaluation in the
//       block's own tangent (R_i, R_j on SO(3) tangent; everything else on R³).
//
// Trajectory: constant body-frame angular velocity, zero linear acceleration in world frame,
// zero gravity. Forward-Euler IMU preintegration is exact for this signal — DeltaR matches
// Exp(omega * T), Deltav = 0, Deltap = 0 — so the residual is structurally zero at GT.

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/imu_preintegration_residual.hpp"
#include "calibforge/imu_preintegrator.hpp"
#include "calibforge/manifold.hpp"
#include "cf_test.hpp"
#include "sophus/so3.hpp"

using calibforge::ImuPreintegrationFactorResidual;
using calibforge::ImuPreintegrator;
using calibforge::SO3Param;

namespace {

struct Scenario {
  // Ground-truth states.
  Sophus::SO3d R_i, R_j;
  Eigen::Vector3d p_i, p_j, v_i, v_j;
  Eigen::Vector3d b_g, b_a, g_w;
  // Preintegrator integrated at b_g_nom = b_g, b_a_nom = b_a (so δb_g = δb_a = 0).
  ImuPreintegrator pre;

  Scenario() : pre(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()) {}
};

// Pure body-frame rotation at constant omega; zero linear motion; zero gravity. With these
// signals forward-Euler is exact; residual at GT is structurally zero.
Scenario makeRotationScenario() {
  Scenario s;
  s.R_i = Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.20, 0.15));
  s.p_i = Eigen::Vector3d(1.0, -0.5, 0.3);
  s.v_i = Eigen::Vector3d(0.0, 0.0, 0.0);
  s.b_g = Eigen::Vector3d(0.005, -0.003, 0.002);
  s.b_a = Eigen::Vector3d(-0.001, 0.0005, 0.0008);
  s.g_w = Eigen::Vector3d::Zero();

  s.pre = ImuPreintegrator(s.b_g, s.b_a);
  const Eigen::Vector3d omega_body(0.3, -0.2, 0.4);
  const double dt = 0.01;
  const int n = 50;
  for (int k = 0; k < n; ++k) {
    // w_meas = w_hat + b_g (true gyro + bias). True gyro is the body-frame omega.
    s.pre.addSample(omega_body + s.b_g, /*a_meas=*/s.b_a, dt);
  }
  const double T = n * dt;
  s.R_j = s.R_i * Sophus::SO3d::exp(omega_body * T);
  s.p_j = s.p_i + s.v_i * T;  // v_i = 0 so p_j = p_i
  s.v_j = s.v_i;
  return s;
}

// Pure linear motion at constant world-frame velocity (no accel, no rotation, no gravity).
// Forward-Euler integration is exact for these signals as well.
Scenario makeTranslationScenario() {
  Scenario s;
  s.R_i = Sophus::SO3d();  // identity
  s.p_i = Eigen::Vector3d(0.0, 0.0, 0.0);
  s.v_i = Eigen::Vector3d(0.4, -0.3, 0.1);
  s.b_g = Eigen::Vector3d(0.0, 0.0, 0.0);
  s.b_a = Eigen::Vector3d(0.0, 0.0, 0.0);
  s.g_w = Eigen::Vector3d::Zero();

  s.pre = ImuPreintegrator(s.b_g, s.b_a);
  const double dt = 0.01;
  const int n = 50;
  for (int k = 0; k < n; ++k) {
    s.pre.addSample(/*w_meas=*/Eigen::Vector3d::Zero(),
                    /*a_meas=*/Eigen::Vector3d::Zero(), dt);
  }
  const double T = n * dt;
  s.R_j = s.R_i;
  s.p_j = s.p_i + s.v_i * T;
  s.v_j = s.v_i;
  return s;
}

// Pack each parameter block into a flat array the residual can read.
struct Params {
  std::array<double, 4> R_i{}, R_j{};
  std::array<double, 3> p_i{}, p_j{}, v_i{}, v_j{}, b_g{}, b_a{}, g_w{};

  std::vector<const double*> ptrs() const {
    return {R_i.data(), p_i.data(), v_i.data(),
            R_j.data(), p_j.data(), v_j.data(),
            b_g.data(), b_a.data(), g_w.data()};
  }
};

Params toParams(const Scenario& s) {
  Params p;
  SO3Param::store(s.R_i, p.R_i.data());
  SO3Param::store(s.R_j, p.R_j.data());
  for (int i = 0; i < 3; ++i) {
    p.p_i[i] = s.p_i[i]; p.p_j[i] = s.p_j[i];
    p.v_i[i] = s.v_i[i]; p.v_j[i] = s.v_j[i];
    p.b_g[i] = s.b_g[i]; p.b_a[i] = s.b_a[i];
    p.g_w[i] = s.g_w[i];
  }
  return p;
}

}  // namespace

CF_TEST(imu_preintegration_residual_zero_at_ground_truth_rotation) {
  Scenario s = makeRotationScenario();
  ImuPreintegrationFactorResidual block(&s.pre);
  Params p = toParams(s);
  auto raw = p.ptrs();
  double r[9];
  block.evaluate(raw.data(), r, nullptr);
  double norm = 0.0;
  for (int i = 0; i < 9; ++i) norm += r[i] * r[i];
  CF_EXPECT_TRUE(std::sqrt(norm) < 1e-9);
}

CF_TEST(imu_preintegration_residual_zero_at_ground_truth_translation) {
  Scenario s = makeTranslationScenario();
  ImuPreintegrationFactorResidual block(&s.pre);
  Params p = toParams(s);
  auto raw = p.ptrs();
  double r[9];
  block.evaluate(raw.data(), r, nullptr);
  double norm = 0.0;
  for (int i = 0; i < 9; ++i) norm += r[i] * r[i];
  CF_EXPECT_TRUE(std::sqrt(norm) < 1e-9);
}

CF_TEST(imu_preintegration_residual_jacobians_match_finite_difference) {
  Scenario s = makeRotationScenario();
  // Perturb the GT slightly so the residual isn't at exactly zero — better FD signal.
  s.R_i = s.R_i * Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015));
  s.p_i += Eigen::Vector3d(0.05, -0.03, 0.04);
  s.v_i = Eigen::Vector3d(0.1, -0.05, 0.07);
  s.R_j = s.R_j * Sophus::SO3d::exp(Eigen::Vector3d(-0.01, 0.02, -0.005));
  s.p_j += Eigen::Vector3d(-0.04, 0.02, 0.03);
  s.v_j = Eigen::Vector3d(0.08, -0.03, 0.05);
  s.b_g += Eigen::Vector3d(0.001, -0.0005, 0.0008);
  s.b_a += Eigen::Vector3d(-0.0005, 0.001, -0.0008);
  s.g_w = Eigen::Vector3d(0.05, -0.03, 0.02);

  ImuPreintegrationFactorResidual block(&s.pre);

  Params p = toParams(s);
  auto raw = p.ptrs();
  double r0[9];
  std::array<std::array<double, 9 * 3>, 9> J{};
  double* jacs[9] = {J[0].data(), J[1].data(), J[2].data(), J[3].data(), J[4].data(),
                     J[5].data(), J[6].data(), J[7].data(), J[8].data()};
  block.evaluate(raw.data(), r0, jacs);

  const double h = 1e-6;
  SO3Param so3;

  auto eval = [&](const std::array<const double*, 9>& ps, double out[9]) {
    block.evaluate(ps.data(), out, nullptr);
  };

  auto check_column = [&](int block_index, int col, const std::array<const double*, 9>& pp,
                          const std::array<const double*, 9>& pm) {
    double rp[9], rm[9];
    eval(pp, rp);
    eval(pm, rm);
    for (int row = 0; row < 9; ++row) {
      const double fd = (rp[row] - rm[row]) / (2 * h);
      const double an = J[block_index][row * 3 + col];
      CF_EXPECT_NEAR(an, fd, 5e-4);
    }
  };

  // Helper for Euclidean blocks: build perturbed param pointer arrays.
  auto eucl_check = [&](int block_index, double* base) {
    for (int axis = 0; axis < 3; ++axis) {
      std::array<double, 3> backup{base[0], base[1], base[2]};
      std::array<double, 3> dp = backup, dm = backup;
      dp[axis] += h; dm[axis] -= h;

      std::array<const double*, 9> pp = {raw[0], raw[1], raw[2], raw[3], raw[4],
                                         raw[5], raw[6], raw[7], raw[8]};
      std::array<const double*, 9> pm = pp;
      pp[block_index] = dp.data();
      pm[block_index] = dm.data();

      check_column(block_index, axis, pp, pm);
    }
  };

  // R_i (SO3 tangent).
  for (int axis = 0; axis < 3; ++axis) {
    double dt_p[3] = {0, 0, 0}, dt_m[3] = {0, 0, 0};
    dt_p[axis] = h; dt_m[axis] = -h;
    std::array<double, 4> Rp{}, Rm{};
    so3.retract(p.R_i.data(), dt_p, Rp.data());
    so3.retract(p.R_i.data(), dt_m, Rm.data());
    std::array<const double*, 9> pp = {Rp.data(), raw[1], raw[2], raw[3], raw[4],
                                       raw[5], raw[6], raw[7], raw[8]};
    std::array<const double*, 9> pm = {Rm.data(), raw[1], raw[2], raw[3], raw[4],
                                       raw[5], raw[6], raw[7], raw[8]};
    check_column(0, axis, pp, pm);
  }
  eucl_check(1, p.p_i.data());
  eucl_check(2, p.v_i.data());
  // R_j (SO3 tangent).
  for (int axis = 0; axis < 3; ++axis) {
    double dt_p[3] = {0, 0, 0}, dt_m[3] = {0, 0, 0};
    dt_p[axis] = h; dt_m[axis] = -h;
    std::array<double, 4> Rp{}, Rm{};
    so3.retract(p.R_j.data(), dt_p, Rp.data());
    so3.retract(p.R_j.data(), dt_m, Rm.data());
    std::array<const double*, 9> pp = {raw[0], raw[1], raw[2], Rp.data(), raw[4],
                                       raw[5], raw[6], raw[7], raw[8]};
    std::array<const double*, 9> pm = {raw[0], raw[1], raw[2], Rm.data(), raw[4],
                                       raw[5], raw[6], raw[7], raw[8]};
    check_column(3, axis, pp, pm);
  }
  eucl_check(4, p.p_j.data());
  eucl_check(5, p.v_j.data());
  eucl_check(6, p.b_g.data());
  eucl_check(7, p.b_a.data());
  eucl_check(8, p.g_w.data());
}
