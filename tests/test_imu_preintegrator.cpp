// Unit tests for the Forster IMU preintegrator. Two layers:
//   1. addSample() correctness — integrating constant gyro/accel matches the closed-form
//      DeltaR / Deltav / Deltap exactly (forward-Euler is exact for those signals).
//   2. Bias-correction Jacobians — the first-order linearizations
//        ΔR(b_g_nom + δb) ≈ ΔR(b_g_nom) * Exp(dR_dbg * δb)
//        Δv(b_g_nom + δb_g, b_a_nom + δb_a) ≈ Δv₀ + dv_dbg * δb_g + dv_dba * δb_a
//        Δp(b_g_nom + δb_g, b_a_nom + δb_a) ≈ Δp₀ + dp_dbg * δb_g + dp_dba * δb_a
//      match a finite-difference re-integration at a perturbed bias.

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/imu_preintegrator.hpp"
#include "cf_test.hpp"
#include "sophus/so3.hpp"

using calibforge::ImuPreintegrator;

// Constant pure rotation: DeltaR = Exp(omega * T), Deltav = 0, Deltap = 0.
CF_TEST(imu_preintegrator_constant_gyro_matches_closed_form) {
  const Eigen::Vector3d omega(0.5, -0.3, 0.2);
  const double dt = 0.01;
  const int n = 50;
  ImuPreintegrator pre(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  for (int k = 0; k < n; ++k) pre.addSample(omega, Eigen::Vector3d::Zero(), dt);

  const double T = n * dt;
  const Sophus::SO3d expected = Sophus::SO3d::exp(omega * T);
  CF_EXPECT_TRUE((pre.deltaR() * expected.inverse()).log().norm() < 1e-9);
  CF_EXPECT_TRUE(pre.deltav().norm() < 1e-12);
  CF_EXPECT_TRUE(pre.deltap().norm() < 1e-12);
  CF_EXPECT_NEAR(pre.deltaT(), T, 1e-12);
}

// No rotation, constant body-frame accel a (gyro=0): Δv = a*T, Δp = 0.5*a*T².
CF_TEST(imu_preintegrator_constant_accel_matches_closed_form) {
  const Eigen::Vector3d a(0.4, -0.7, 0.3);
  const double dt = 0.01;
  const int n = 50;
  ImuPreintegrator pre(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  for (int k = 0; k < n; ++k) pre.addSample(Eigen::Vector3d::Zero(), a, dt);

  const double T = n * dt;
  CF_EXPECT_TRUE((pre.deltav() - a * T).norm() < 1e-12);
  CF_EXPECT_TRUE((pre.deltap() - 0.5 * a * T * T).norm() < 1e-12);
  CF_EXPECT_TRUE(pre.deltaR().log().norm() < 1e-12);
}

// Re-integrate from raw samples at a perturbed gyro bias; the first-order Jacobian dR_dbg
// must predict the change in DeltaR to within O(perturbation²) — finite-difference check.
CF_TEST(imu_preintegrator_bias_jacobian_dR_dbg_matches_fd) {
  std::vector<Eigen::Vector3d> ws, as;
  for (int k = 0; k < 40; ++k) {
    const double t = k * 0.01;
    ws.push_back(Eigen::Vector3d(0.3 + 0.1 * std::sin(2 * t), 0.2 * std::cos(1.7 * t),
                                 -0.1 + 0.05 * std::sin(3 * t)));
    as.push_back(Eigen::Vector3d(0.2 * std::sin(1.5 * t), 0.4 * std::cos(t),
                                 0.1 - 0.05 * std::sin(2 * t)));
  }
  const Eigen::Vector3d bg_nom(0.01, -0.02, 0.005);
  const Eigen::Vector3d ba_nom(0.001, 0.0, -0.002);

  ImuPreintegrator p0(bg_nom, ba_nom);
  for (std::size_t k = 0; k < ws.size(); ++k) p0.addSample(ws[k], as[k], 0.01);

  const double h = 1e-5;
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d db = Eigen::Vector3d::Zero();
    db[axis] = h;
    ImuPreintegrator pp(bg_nom, ba_nom);  // re-integrate at the nominal, then test prediction
    for (std::size_t k = 0; k < ws.size(); ++k) pp.addSample(ws[k] - db, as[k], 0.01);
    // pp.deltaR() corresponds to integrating with bg_nom replaced by (bg_nom + db) since
    // w_hat = w_meas - bg => increasing bg by db is equivalent to decreasing w_meas by db.
    const Sophus::SO3d dRp = pp.deltaR();
    const Sophus::SO3d dR0 = p0.deltaR();
    // Closed-form prediction: dRp ≈ dR0 * Exp(dR_dbg * db).
    const Eigen::Vector3d phi_predicted = p0.dR_dbg() * db;
    const Sophus::SO3d dR_predicted = dR0 * Sophus::SO3d::exp(phi_predicted);
    CF_EXPECT_TRUE((dRp * dR_predicted.inverse()).log().norm() < 1e-7);
  }
}

// First-order Jacobians dv_dbg / dv_dba / dp_dbg / dp_dba: re-integrate, the actual change
// should match J * δb up to O(δb²).
CF_TEST(imu_preintegrator_bias_jacobian_dv_dp_match_fd) {
  std::vector<Eigen::Vector3d> ws, as;
  for (int k = 0; k < 40; ++k) {
    const double t = k * 0.01;
    ws.push_back(Eigen::Vector3d(0.3 + 0.1 * std::sin(2 * t), 0.2 * std::cos(1.7 * t),
                                 -0.1 + 0.05 * std::sin(3 * t)));
    as.push_back(Eigen::Vector3d(0.5 + 0.2 * std::sin(1.5 * t), -0.3 + 0.4 * std::cos(t),
                                 0.1 - 0.05 * std::sin(2 * t)));
  }
  const Eigen::Vector3d bg_nom(0.01, -0.02, 0.005);
  const Eigen::Vector3d ba_nom(0.001, 0.0, -0.002);

  ImuPreintegrator p0(bg_nom, ba_nom);
  for (std::size_t k = 0; k < ws.size(); ++k) p0.addSample(ws[k], as[k], 0.01);

  const double h = 1e-6;
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d db = Eigen::Vector3d::Zero(); db[axis] = h;
    // Perturb bg via subtracting db from each gyro sample (== increasing bg).
    ImuPreintegrator pg(bg_nom, ba_nom);
    for (std::size_t k = 0; k < ws.size(); ++k) pg.addSample(ws[k] - db, as[k], 0.01);
    const Eigen::Vector3d fd_dv = (pg.deltav() - p0.deltav()) / h;
    const Eigen::Vector3d fd_dp = (pg.deltap() - p0.deltap()) / h;
    const Eigen::Vector3d an_dv = p0.dv_dbg().col(axis);
    const Eigen::Vector3d an_dp = p0.dp_dbg().col(axis);
    CF_EXPECT_TRUE((fd_dv - an_dv).norm() < 1e-3);
    CF_EXPECT_TRUE((fd_dp - an_dp).norm() < 1e-3);

    // Same for ba (subtract from each accel sample).
    ImuPreintegrator pa(bg_nom, ba_nom);
    for (std::size_t k = 0; k < ws.size(); ++k) pa.addSample(ws[k], as[k] - db, 0.01);
    const Eigen::Vector3d fd_dv_a = (pa.deltav() - p0.deltav()) / h;
    const Eigen::Vector3d fd_dp_a = (pa.deltap() - p0.deltap()) / h;
    const Eigen::Vector3d an_dv_a = p0.dv_dba().col(axis);
    const Eigen::Vector3d an_dp_a = p0.dp_dba().col(axis);
    CF_EXPECT_TRUE((fd_dv_a - an_dv_a).norm() < 1e-3);
    CF_EXPECT_TRUE((fd_dp_a - an_dp_a).norm() < 1e-3);
  }
}
