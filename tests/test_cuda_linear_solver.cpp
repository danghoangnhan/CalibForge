// Tests for solve/cuda_linear_solver.* — the native CUDA dense LM back-end (#25 GPU solver).
//
// Availability matches the build config (mirrors test_remap's cuda_remap_availability check).
// On a CUDA host: (1) one damped-normal-equations step on the GPU matches the host Eigen LDLT
// solve to FP64 round-off; (2) a non-SPD damped matrix is reported (returns false) so the LM
// loop damps harder; (3) the full DenseProblem solve with the GpuCuda backend recovers the same
// calibration as the CPU backend.

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

CF_TEST(cuda_solver_availability_matches_build) {
#ifdef CALIBFORGE_HAS_CUDA
  CF_EXPECT_TRUE(calibforge::cudaSolverAvailable());
#else
  CF_EXPECT_TRUE(!calibforge::cudaSolverAvailable());
#endif
}

#ifdef CALIBFORGE_HAS_CUDA

CF_TEST(cuda_lm_step_matches_cpu_normal_equations) {
  // A deterministic, full-column-rank J (m > n) so J^T J is SPD. Solve the damped normal
  // equations on the host (Eigen LDLT) and on the GPU and require agreement to FP64 round-off.
  const int m = 64, n = 10;
  std::mt19937_64 rng(0xC0FFEE);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd J(m, n);  // column-major (Eigen default) == cuBLAS native
  Eigen::VectorXd r(m);
  for (int i = 0; i < m; ++i) {
    r(i) = nd(rng);
    for (int j = 0; j < n; ++j) J(i, j) = nd(rng);
  }
  const double lambda = 1e-3;

  // Host reference: (J^T J + lambda diag(J^T J)) dx = -J^T r.
  const Eigen::MatrixXd JtJ = J.transpose() * J;
  const Eigen::VectorXd g = J.transpose() * r;
  Eigen::MatrixXd A = JtJ;
  A.diagonal() += lambda * JtJ.diagonal();
  const Eigen::VectorXd dx_cpu = A.ldlt().solve(-g);

  Eigen::VectorXd dx_gpu(n);
  CF_EXPECT_TRUE(calibforge::cudaSolveLmStep(J.data(), m, n, r.data(), lambda, dx_gpu.data()));

  const double rel = (dx_gpu - dx_cpu).norm() / std::max(dx_cpu.norm(), 1e-300);
  CF_EXPECT_TRUE(rel < 1e-8);
}

CF_TEST(cuda_lm_step_reports_non_spd_so_caller_can_damp) {
  // A J with an all-zero column makes J^T J singular; with lambda=0 the damped matrix has a zero
  // pivot, so the GPU Cholesky must report failure (return false) — the signal the host LM uses
  // to damp harder rather than emit a garbage step.
  const int m = 8, n = 3;
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(m, n);
  for (int i = 0; i < m; ++i) { J(i, 0) = 1.0 + 0.1 * i; J(i, 1) = 0.5 * i; }  // col 2 stays zero
  Eigen::VectorXd r = Eigen::VectorXd::Ones(m);
  Eigen::VectorXd dx(n);
  CF_EXPECT_TRUE(!calibforge::cudaSolveLmStep(J.data(), m, n, r.data(), /*lambda=*/0.0, dx.data()));
}

CF_TEST(cuda_lm_step_handles_single_parameter) {
  // n==1: the damped normal matrix is 1x1. Exercises cuSOLVER potrf/potrs at the minimum size and
  // the scaleDiagonalKernel index formula (element (0,0) at i*n+i == 0). Must match the host scalar
  // solve dx = -(J^T r) / ((J^T J)(1 + lambda)).
  const int m = 8, n = 1;
  Eigen::MatrixXd J(m, n);
  Eigen::VectorXd r(m);
  for (int i = 0; i < m; ++i) { J(i, 0) = 0.5 + 0.25 * i; r(i) = 1.0 - 0.1 * i; }
  const double lambda = 1e-2;

  const Eigen::MatrixXd JtJ = J.transpose() * J;
  const Eigen::VectorXd g = J.transpose() * r;
  Eigen::MatrixXd A = JtJ;
  A.diagonal() += lambda * JtJ.diagonal();
  const Eigen::VectorXd dx_cpu = A.ldlt().solve(-g);

  Eigen::VectorXd dx_gpu(n);
  CF_EXPECT_TRUE(calibforge::cudaSolveLmStep(J.data(), m, n, r.data(), lambda, dx_gpu.data()));
  CF_EXPECT_NEAR(dx_gpu(0), dx_cpu(0), 1e-9);
}

CF_TEST(cuda_lm_step_matches_cpu_on_ill_conditioned_spd) {
  // Columns at very different scales make J^T J ill-conditioned (cond ~ 1e6) but still SPD. The
  // GPU SYRK + Cholesky must track the host LDLT on the SAME hard matrix — i.e. the device path is
  // numerically no worse than the host on bad conditioning (both lose the same low-order digits).
  const int m = 40, n = 4;
  std::mt19937_64 rng(0x1CECAFE);
  std::normal_distribution<double> nd(0.0, 1.0);
  const double scales[4] = {1e0, 1e-1, 1e1, 1e-2};  // cond(J^T J) ~ (1e1/1e-2)^2 = 1e6
  Eigen::MatrixXd J(m, n);
  Eigen::VectorXd r(m);
  for (int i = 0; i < m; ++i) {
    r(i) = nd(rng);
    for (int j = 0; j < n; ++j) J(i, j) = scales[j] * nd(rng);
  }
  const double lambda = 1e-6;

  const Eigen::MatrixXd JtJ = J.transpose() * J;
  const Eigen::VectorXd g = J.transpose() * r;
  Eigen::MatrixXd A = JtJ;
  A.diagonal() += lambda * JtJ.diagonal();
  const Eigen::VectorXd dx_cpu = A.ldlt().solve(-g);

  Eigen::VectorXd dx_gpu(n);
  CF_EXPECT_TRUE(calibforge::cudaSolveLmStep(J.data(), m, n, r.data(), lambda, dx_gpu.data()));
  const double rel = (dx_gpu - dx_cpu).norm() / std::max(dx_cpu.norm(), 1e-300);
  CF_EXPECT_TRUE(rel < 1e-6);  // ample margin over the eps*cond ~ 1e-10 the two solvers can differ
}

CF_TEST(cuda_lm_step_non_spd_becomes_solvable_after_damping) {
  // Two IDENTICAL columns make J^T J rank-deficient (singular) but with a strictly positive
  // diagonal: at lambda=0 the device Cholesky must fail (non-SPD), yet Marquardt damping (lambda>0
  // scales the diagonal up by 1+lambda, A = J^TJ + lambda*diag(J^TJ) >= 0 plus >0 => SPD) restores
  // definiteness. This is exactly the damp-harder-and-retry contract DenseProblem's LM loop relies
  // on — the first solve rejects, the next (damped) one succeeds.
  //
  // To make the lambda=0 failure DETERMINISTIC (not rounding-dependent), columns 0 and 1 are
  // all-ones with m=4, so (J^TJ)_00 = 4 exactly: Cholesky gives pivot1 = sqrt(4) = 2, then pivot2 =
  // 4 - (4/2)^2 = 0 EXACTLY (no cancellation slack), which LAPACK/cuSOLVER potrf flags as non-PD.
  // (A near-equal pair would leave pivot2 = a - a rounded to a tiny POSITIVE value and slip past.)
  const int m = 4, n = 3;
  Eigen::MatrixXd J(m, n);
  J.col(0) = Eigen::Vector4d(1.0, 1.0, 1.0, 1.0);
  J.col(1) = J.col(0);                              // exact duplicate -> exact rank deficiency
  J.col(2) = Eigen::Vector4d(0.5, -0.3, 1.0, -0.7);  // independent, keeps diag(J^T J) > 0
  Eigen::Vector4d r(0.2, 0.4, -0.1, 0.3);
  Eigen::VectorXd dx(n);

  // lambda = 0: singular normal matrix -> device Cholesky reports non-SPD (return false).
  CF_EXPECT_TRUE(!calibforge::cudaSolveLmStep(J.data(), m, n, r.data(), 0.0, dx.data()));

  // lambda > 0: damping lifts the diagonal, the damped matrix is SPD, the GPU solve now succeeds
  // and matches the host damped solve.
  const double lambda = 1e-3;
  const Eigen::MatrixXd JtJ = J.transpose() * J;
  const Eigen::VectorXd g = J.transpose() * r;
  Eigen::MatrixXd A = JtJ;
  A.diagonal() += lambda * JtJ.diagonal();
  const Eigen::VectorXd dx_cpu = A.ldlt().solve(-g);

  CF_EXPECT_TRUE(calibforge::cudaSolveLmStep(J.data(), m, n, r.data(), lambda, dx.data()));
  const double rel = (dx - dx_cpu).norm() / std::max(dx_cpu.norm(), 1e-300);
  CF_EXPECT_TRUE(rel < 1e-6);
}

namespace {
std::unique_ptr<CameraModel> makePinhole(const Eigen::VectorXd& q) {
  return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
}

// Build the standard noise-free single-camera scene into freshly-owned parameter storage and
// solve it with the requested backend. Returns the recovered intrinsics + the LM summary.
// When hold_intrinsics_const is set, the intrinsics block is fixed at GROUND TRUTH (a constant
// block cannot be recovered from a wrong init) and excluded from the tangent vector — so the GPU
// back-end receives n_tangent = 6*n_poses, exercising the constant-block masking on the GPU path.
struct SolveOut { std::array<double, 4> intr; calibforge::LmSummary summary; };
SolveOut solveScene(SolverBackend backend, bool hold_intrinsics_const = false) {
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

  SolveOut out;
  std::vector<double> intr =
      hold_intrinsics_const ? std::vector<double>{fx, fy, cx, cy}    // fixed block: start at truth
                            : std::vector<double>{470, 530, 305, 255};  // free block: wrong init
  std::vector<std::array<double, 7>> pose(gt_poses.size());
  Eigen::Matrix<double, 6, 1> perturb;
  perturb << 0.02, -0.02, 0.02, 0.02, -0.01, 0.01;
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    SE3Param::store(gt_poses[i] * Sophus::SE3d::exp(perturb), pose[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr.data(), std::make_shared<EuclideanParam>(4));
  for (auto& p : pose) problem.addParameterBlock(p.data(), std::make_shared<SE3Param>());
  if (hold_intrinsics_const) problem.setParameterBlockConstant(intr.data());
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    for (const auto& X : board) {
      Eigen::Vector3d Xc = gt_poses[i] * Eigen::Vector3d(X[0], X[1], X[2]);
      Vec2 px = gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()});
      problem.addResidualBlock(std::make_unique<ReprojectionResidual>(makePinhole, 4, X, px),
                               {intr.data(), pose[i].data()});
    }
  out.summary = problem.solveLm(LmOptions{}, backend);
  out.intr = {intr[0], intr[1], intr[2], intr[3]};
  return out;
}
}  // namespace

CF_TEST(dense_problem_gpu_backend_recovers_same_calibration_as_cpu) {
  const SolveOut cpu = solveScene(SolverBackend::CpuCeres);
  const SolveOut gpu = solveScene(SolverBackend::GpuCuda);
  const double fx = 500, fy = 500, cx = 320, cy = 240;

  CF_EXPECT_TRUE(cpu.summary.converged && gpu.summary.converged);
  CF_EXPECT_TRUE(cpu.summary.final_cost < 1e-8 && gpu.summary.final_cost < 1e-8);
  // GPU recovers ground truth...
  CF_EXPECT_NEAR(gpu.intr[0], fx, 1e-3);
  CF_EXPECT_NEAR(gpu.intr[1], fy, 1e-3);
  CF_EXPECT_NEAR(gpu.intr[2], cx, 1e-3);
  CF_EXPECT_NEAR(gpu.intr[3], cy, 1e-3);
  // ...and agrees with the CPU backend (both reach the same noise-free minimum).
  for (int k = 0; k < 4; ++k) CF_EXPECT_NEAR(gpu.intr[k], cpu.intr[k], 1e-5);
}

CF_TEST(dense_problem_gpu_backend_honors_constant_block) {
  // Hold the intrinsics constant: the GPU path must build/solve a tangent vector that EXCLUDES the
  // 4 intrinsic columns (n_tangent = 6*n_poses, not 4 + 6*n_poses) and leave the intrinsics
  // untouched, recovering the poses just like the CPU backend. Guards the constant-block masking
  // on the GPU code path, which the unconstrained test above never exercises.
  const SolveOut cpu = solveScene(SolverBackend::CpuCeres, /*hold_intrinsics_const=*/true);
  const SolveOut gpu = solveScene(SolverBackend::GpuCuda, /*hold_intrinsics_const=*/true);
  const double fx = 500, fy = 500, cx = 320, cy = 240;

  CF_EXPECT_TRUE(cpu.summary.converged && gpu.summary.converged);
  CF_EXPECT_TRUE(gpu.summary.final_cost < 1e-8);
  // The constant intrinsics did not move on the GPU path (held exactly at ground truth)...
  CF_EXPECT_NEAR(gpu.intr[0], fx, 1e-12);
  CF_EXPECT_NEAR(gpu.intr[1], fy, 1e-12);
  CF_EXPECT_NEAR(gpu.intr[2], cx, 1e-12);
  CF_EXPECT_NEAR(gpu.intr[3], cy, 1e-12);
  // ...and the GPU reached the same minimum as the CPU backend with the same constraint.
  CF_EXPECT_NEAR(gpu.summary.final_cost, cpu.summary.final_cost, 1e-9);
}

#endif  // CALIBFORGE_HAS_CUDA
