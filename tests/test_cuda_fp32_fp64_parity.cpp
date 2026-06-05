// FP32 <-> FP64 numerical-parity tests for the native CUDA dense LM back-end (#25 / RULE #6).
//
// docs/SPIKES.md §D.3 calls edge<->server FP32/bf16 vs FP64 parity "unproven". These tests MEASURE
// it firsthand on the build's actual GPU: (1) one damped LM step solved in single precision on the
// device agrees with the FP64 host LDLT to the single-precision envelope and with an FP32 host
// LDLT tightly (the GPU FP32 path is a CORRECT single-precision solve, not merely an imprecise
// one); (2) a FULL calibration whose per-step solves run in FP32 (SolverBackend::GpuCudaF32)
// recovers the same parameters as the FP64 CPU oracle — because the LM loop still evaluates and
// accepts steps in FP64, single-precision step directions converge to the same FP64 minimum.

#include <array>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/cuda_linear_solver.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/problem.hpp"
#include "calibforge/reprojection_residual.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraModel;
using calibforge::DenseProblem;
using calibforge::EuclideanParam;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::ReprojectionResidual;
using calibforge::SE3Param;
using calibforge::SolverBackend;
using calibforge::Vec2;
using calibforge::Vec3;

CF_TEST(cuda_fp32_path_availability_matches_build) {
#ifdef CALIBFORGE_HAS_CUDA
  CF_EXPECT_TRUE(calibforge::cudaSolverAvailable());
#else
  CF_EXPECT_TRUE(!calibforge::cudaSolverAvailable());
#endif
}

#ifdef CALIBFORGE_HAS_CUDA

CF_TEST(cuda_fp32_fp64_step_parity_well_conditioned) {
  // Well-conditioned full-column-rank J (m > n) so J^T J is SPD and benign. Compare a single
  // damped LM step across: host FP64 LDLT (oracle), host FP32 LDLT, GPU FP64, GPU FP32.
  const int m = 64, n = 10;
  std::mt19937_64 rng(0xBADC0FFEE);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd J(m, n);
  Eigen::VectorXd r(m);
  for (int i = 0; i < m; ++i) {
    r(i) = nd(rng);
    for (int j = 0; j < n; ++j) J(i, j) = nd(rng);
  }
  const double lambda = 1e-3;

  // Host FP64 oracle.
  const Eigen::MatrixXd JtJ = J.transpose() * J;
  const Eigen::VectorXd g = J.transpose() * r;
  Eigen::MatrixXd A = JtJ;
  A.diagonal() += lambda * JtJ.diagonal();
  const Eigen::VectorXd dx64 = A.ldlt().solve(-g);

  // Host FP32 reference (same equations in single precision).
  const Eigen::MatrixXf Jf = J.cast<float>();
  const Eigen::VectorXf rf = r.cast<float>();
  const Eigen::MatrixXf JtJf = Jf.transpose() * Jf;
  const Eigen::VectorXf gf = Jf.transpose() * rf;
  Eigen::MatrixXf Af = JtJf;
  Af.diagonal() += static_cast<float>(lambda) * JtJf.diagonal();
  const Eigen::VectorXf dxf_cpu = Af.ldlt().solve(-gf);

  Eigen::VectorXd dx_gpu64(n), dx_gpu32(n);
  CF_EXPECT_TRUE(calibforge::cudaSolveLmStep(J.data(), m, n, r.data(), lambda, dx_gpu64.data()));
  CF_EXPECT_TRUE(calibforge::cudaSolveLmStepF32(J.data(), m, n, r.data(), lambda, dx_gpu32.data()));

  const double ref = std::max(dx64.norm(), 1e-300);
  const double rel_gpu64 = (dx_gpu64 - dx64).norm() / ref;             // FP64 GPU vs FP64 host
  const double rel_gpu32 = (dx_gpu32 - dx64).norm() / ref;             // FP32 GPU vs FP64 host
  const double rel_gpu32_vs_cpu32 =
      (dx_gpu32 - dxf_cpu.cast<double>()).norm() / ref;                // FP32 GPU vs FP32 host

  CF_EXPECT_TRUE(rel_gpu64 < 1e-8);              // FP64 device matches host to round-off
  // Tolerances sit ~50-80x above the firsthand-measured GB10 envelope (rel_gpu32 ~1.7e-7 vs FP64,
  // ~1.2e-7 vs FP32; docs/SPIKES.md §F.3) — tight enough that an FP32-path regression losing even
  // ~2 of its ~7 significant digits fails the test, while leaving headroom for cross-arch ULP drift.
  CF_EXPECT_TRUE(rel_gpu32 < 1e-5);              // FP32 device within the single-precision envelope
  CF_EXPECT_TRUE(rel_gpu32_vs_cpu32 < 1e-5);     // and it IS a correct single-precision solve
}

namespace {
std::unique_ptr<CameraModel> makePinhole(const Eigen::VectorXd& q) {
  return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
}

struct SolveOut {
  std::array<double, 4> intr;
  calibforge::LmSummary summary;
};

// The standard noise-free single-camera pinhole scene solved with the requested backend.
SolveOut solveScene(SolverBackend backend) {
  const double fx = 500, fy = 500, cx = 320, cy = 240;
  PinholeCamera gt(fx, fy, cx, cy);
  std::vector<Vec3> board;
  for (int row = 0; row < 4; ++row)
    for (int c = 0; c < 4; ++c) board.push_back(Vec3{c * 0.1, row * 0.1, 0.0});
  std::vector<Sophus::SE3d> gt_poses = {
      {Sophus::SO3d::exp(Eigen::Vector3d(0.10, -0.05, 0.02)), Eigen::Vector3d(-0.15, -0.15, 1.5)},
      {Sophus::SO3d::exp(Eigen::Vector3d(-0.20, 0.15, -0.05)), Eigen::Vector3d(-0.20, -0.10, 1.2)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.05, 0.25, 0.10)), Eigen::Vector3d(-0.10, -0.20, 1.8)},
      {Sophus::SO3d::exp(Eigen::Vector3d(0.15, 0.10, -0.15)), Eigen::Vector3d(-0.25, -0.05, 1.4)}};

  std::vector<double> intr{470, 530, 305, 255};  // wrong init
  std::vector<std::array<double, 7>> pose(gt_poses.size());
  Eigen::Matrix<double, 6, 1> perturb;
  perturb << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    SE3Param::store(gt_poses[i] * Sophus::SE3d::exp(perturb), pose[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr.data(), std::make_shared<EuclideanParam>(4));
  for (auto& p : pose) problem.addParameterBlock(p.data(), std::make_shared<SE3Param>());
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    for (const auto& X : board) {
      Eigen::Vector3d Xc = gt_poses[i] * Eigen::Vector3d(X[0], X[1], X[2]);
      Vec2 px = gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()});
      problem.addResidualBlock(std::make_unique<ReprojectionResidual>(makePinhole, 4, X, px),
                               {intr.data(), pose[i].data()});
    }
  LmOptions opts;
  opts.max_iterations = 200;
  SolveOut out;
  out.summary = problem.solveLm(opts, backend);
  out.intr = {intr[0], intr[1], intr[2], intr[3]};
  return out;
}
}  // namespace

CF_TEST(cuda_fp32_full_calibration_matches_fp64_oracle) {
  // A FULL calibration whose per-iteration solves run in FP32 on the GPU vs the FP64 CPU oracle.
  // The LM loop still evaluates cost + accepts steps in FP64, so single-precision step directions
  // converge to the same FP64 minimum: edge-precision (FP32) and server-precision (FP64) AGREE.
  const SolveOut cpu = solveScene(SolverBackend::CpuCeres);
  const SolveOut f32 = solveScene(SolverBackend::GpuCudaF32);
  const double fx = 500, fy = 500, cx = 320, cy = 240;

  CF_EXPECT_TRUE(cpu.summary.converged && f32.summary.converged);
  CF_EXPECT_TRUE(f32.summary.final_cost < 1e-6);
  // FP32-stepped calibration recovers ground truth...
  CF_EXPECT_NEAR(f32.intr[0], fx, 5e-2);
  CF_EXPECT_NEAR(f32.intr[1], fy, 5e-2);
  CF_EXPECT_NEAR(f32.intr[2], cx, 5e-2);
  CF_EXPECT_NEAR(f32.intr[3], cy, 5e-2);
  // ...and agrees with the FP64 CPU oracle (edge<->server parity on this host).
  for (int k = 0; k < 4; ++k) CF_EXPECT_NEAR(f32.intr[k], cpu.intr[k], 5e-2);
}

#endif  // CALIBFORGE_HAS_CUDA
