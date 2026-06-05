// CalibForge — CPU calibration-regime benchmark harness.
//
// Implements the CPU rows of SPIKES.md §D.1: time the CPU solver on representative
// calibration problem sizes (#views, #points, #cameras). GPU rows fill in when a CUDA host
// is available (the deferred half of v1.0 — see issue #25 / SPIKES.md §D).
//
// Outputs a CSV table to stdout to seed docs/BENCHMARKS.md (the GPU rows fill in on a CUDA
// host). Each row reports:
//   problem, n_views, n_points, n_cams, cpu_ms_median, cpu_iters_median, final_cost_median
//
// Build (via the top-level CMakeLists.txt):
//   cmake --build build --target calibforge_bench
//   ./build/calibforge_bench
// Or compile standalone (header-only library), as a single command (the SOPHUS define avoids
// Sophus' optional fmt dependency, matching the CMake build):
//   c++ -std=c++17 -O2 -DSOPHUS_USE_BASIC_LOGGING -I core/include -I solve/include -I pipelines/include -I build/_deps/eigen-src -I build/_deps/sophus-src tools/benchmark/calibforge_bench.cpp -o calibforge_bench

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_rig.hpp"
#include "calibforge/calibrate_single.hpp"
#include "calibforge/calibrate_stereo.hpp"
#include "calibforge/camera_model.hpp"
#include "calibforge/cuda_linear_solver.hpp"  // cudaSolverAvailable (GPU rows)
#include "calibforge/dense_problem.hpp"        // DenseProblem + SolverBackend (CPU-vs-GPU section)
#include "calibforge/manifold.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/problem.hpp"
#include "calibforge/reprojection_residual.hpp"
#include "sophus/se3.hpp"

namespace {

using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::DenseProblem;
using calibforge::EuclideanParam;
using calibforge::LmOptions;
using calibforge::PinholeCamera;
using calibforge::ReprojectionResidual;
using calibforge::RigView;
using calibforge::SE3Param;
using calibforge::SolverBackend;
using calibforge::StereoView;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;

CameraFactory pinholeFactory() {
  return [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
}

std::vector<Vec3> makeBoard(int rows, int cols, double pitch) {
  std::vector<Vec3> b;
  b.reserve(static_cast<std::size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) b.push_back(Vec3{c * pitch, r * pitch, 0.0});
  return b;
}

std::vector<Sophus::SE3d> makeRandomPoses(int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nrm(0.0, 0.2);
  std::uniform_real_distribution<double> zdist(1.0, 1.8);
  std::vector<Sophus::SE3d> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    out.emplace_back(Sophus::SO3d::exp(Eigen::Vector3d(nrm(rng), nrm(rng), nrm(rng))),
                     Eigen::Vector3d(nrm(rng), nrm(rng), zdist(rng)));
  }
  return out;
}

Sophus::SE3d perturbPose(const Sophus::SE3d& T, double scale, std::mt19937_64& rng) {
  std::normal_distribution<double> nrm(0.0, scale);
  Eigen::Matrix<double, 6, 1> dx;
  for (int i = 0; i < 6; ++i) dx[i] = nrm(rng);
  return T * Sophus::SE3d::exp(dx);
}

struct Timing {
  double median_ms = 0.0;
  int median_iters = 0;
  double median_final_cost = 0.0;
};

template <typename F>
Timing repeat(int trials, F fn) {
  std::vector<double> ms(trials);
  std::vector<int> its(trials);
  std::vector<double> costs(trials);
  for (int i = 0; i < trials; ++i) {
    // steady_clock is guaranteed monotonic; high_resolution_clock aliases the (non-steady)
    // system_clock on libstdc++ and can jump under NTP, corrupting an interval measurement.
    const auto t0 = std::chrono::steady_clock::now();
    auto r = fn();
    const auto t1 = std::chrono::steady_clock::now();
    ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    its[i] = r.iterations;
    costs[i] = r.final_cost;
  }
  std::sort(ms.begin(), ms.end());
  std::sort(its.begin(), its.end());
  std::sort(costs.begin(), costs.end());
  Timing t;
  t.median_ms = ms[ms.size() / 2];
  t.median_iters = its[its.size() / 2];
  t.median_final_cost = costs[costs.size() / 2];
  return t;
}

struct SolveOutcome {
  double final_cost = 0.0;
  int iterations = 0;
};

// Single-camera CPU calibration row.
SolveOutcome runSingle(int n_views, int board_rows, int board_cols, std::uint64_t seed) {
  PinholeCamera gt(500.0, 500.0, 320.0, 240.0);
  const std::vector<Vec3> board = makeBoard(board_rows, board_cols, 0.1);
  const std::vector<Sophus::SE3d> gt_poses = makeRandomPoses(n_views, seed);

  std::vector<View> views;
  views.reserve(static_cast<std::size_t>(n_views));
  for (const Sophus::SE3d& T : gt_poses) {
    View v;
    for (const Vec3& X : board) {
      const Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
      v.object_points.push_back(X);
      v.image_points.push_back(gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    }
    views.push_back(std::move(v));
  }
  std::mt19937_64 rng(seed ^ 0xa5a5);
  std::vector<Sophus::SE3d> poses_init;
  poses_init.reserve(gt_poses.size());
  for (const Sophus::SE3d& T : gt_poses) poses_init.push_back(perturbPose(T, 0.03, rng));

  Eigen::VectorXd intr0(4);
  intr0 << 480.0, 520.0, 310.0, 250.0;

  const auto res = calibforge::calibrateSingleCamera(views, intr0, poses_init,
                                                     pinholeFactory());
  SolveOutcome o;
  o.final_cost = res.summary.final_cost;
  o.iterations = res.summary.iterations;
  return o;
}

// Stereo CPU calibration row.
SolveOutcome runStereo(int n_views, int board_rows, int board_cols, std::uint64_t seed) {
  PinholeCamera gt_l(500.0, 500.0, 320.0, 240.0);
  PinholeCamera gt_r(510.0, 510.0, 318.0, 242.0);
  const Sophus::SE3d gt_extr(Sophus::SO3d::exp(Eigen::Vector3d(0.01, -0.01, 0.005)),
                             Eigen::Vector3d(-0.10, 0.0, 0.0));
  const std::vector<Vec3> board = makeBoard(board_rows, board_cols, 0.1);
  const std::vector<Sophus::SE3d> gt_poses = makeRandomPoses(n_views, seed);

  std::vector<StereoView> views;
  views.reserve(static_cast<std::size_t>(n_views));
  for (const Sophus::SE3d& T : gt_poses) {
    StereoView v;
    v.object_points = board;
    for (const Vec3& X : board) {
      const Eigen::Vector3d Xc0 = T * Eigen::Vector3d(X[0], X[1], X[2]);
      const Eigen::Vector3d Xc1 = gt_extr * Xc0;
      v.image_points0.push_back(gt_l.project(Vec3{Xc0.x(), Xc0.y(), Xc0.z()}));
      v.image_points1.push_back(gt_r.project(Vec3{Xc1.x(), Xc1.y(), Xc1.z()}));
    }
    views.push_back(std::move(v));
  }

  std::mt19937_64 rng(seed ^ 0x5a5a);
  Eigen::VectorXd intr_l0(4), intr_r0(4);
  intr_l0 << 480.0, 520.0, 310.0, 250.0;
  intr_r0 << 490.0, 530.0, 308.0, 252.0;
  const Sophus::SE3d extr0 = perturbPose(gt_extr, 0.02, rng);
  std::vector<Sophus::SE3d> poses_init;
  poses_init.reserve(gt_poses.size());
  for (const Sophus::SE3d& T : gt_poses) poses_init.push_back(perturbPose(T, 0.03, rng));

  const auto res = calibforge::calibrateStereo(views, intr_l0, intr_r0, extr0, poses_init,
                                               pinholeFactory(), pinholeFactory());
  SolveOutcome o;
  o.final_cost = res.summary.final_cost;
  o.iterations = res.summary.iterations;
  return o;
}

// N-camera rig CPU calibration row.
SolveOutcome runRig(int n_views, int n_cams, int board_rows, int board_cols,
                    std::uint64_t seed) {
  std::vector<PinholeCamera> gt_cams;
  gt_cams.reserve(static_cast<std::size_t>(n_cams));
  for (int c = 0; c < n_cams; ++c)
    gt_cams.emplace_back(500.0 + 2.0 * c, 500.0 + 2.0 * c, 320.0 - c, 240.0 + c);

  std::vector<Sophus::SE3d> gt_extr(static_cast<std::size_t>(n_cams - 1));
  for (int k = 0; k < n_cams - 1; ++k) {
    const double off = -0.10 - 0.10 * k;
    gt_extr[k] = Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.005 * k, -0.003 * k, 0.0)),
                              Eigen::Vector3d(off, 0.0, 0.0));
  }

  const std::vector<Vec3> board = makeBoard(board_rows, board_cols, 0.1);
  const std::vector<Sophus::SE3d> gt_poses = makeRandomPoses(n_views, seed);

  std::vector<RigView> views;
  views.reserve(static_cast<std::size_t>(n_views));
  for (const Sophus::SE3d& T : gt_poses) {
    RigView v;
    v.object_points = board;
    v.image_points.assign(static_cast<std::size_t>(n_cams), {});
    for (const Vec3& X : board) {
      const Eigen::Vector3d Xc0 = T * Eigen::Vector3d(X[0], X[1], X[2]);
      v.image_points[0].push_back(
          gt_cams[0].project(Vec3{Xc0.x(), Xc0.y(), Xc0.z()}));
      for (int c = 1; c < n_cams; ++c) {
        const Eigen::Vector3d Xck = gt_extr[c - 1] * Xc0;
        v.image_points[c].push_back(
            gt_cams[c].project(Vec3{Xck.x(), Xck.y(), Xck.z()}));
      }
    }
    views.push_back(std::move(v));
  }

  std::mt19937_64 rng(seed ^ 0x3c3c);
  std::vector<Eigen::VectorXd> intr_init(static_cast<std::size_t>(n_cams),
                                          Eigen::VectorXd(4));
  for (int c = 0; c < n_cams; ++c)
    intr_init[c] << 480.0, 520.0, 310.0, 250.0;
  std::vector<Sophus::SE3d> extr_init(static_cast<std::size_t>(n_cams - 1));
  for (int k = 0; k < n_cams - 1; ++k) extr_init[k] = perturbPose(gt_extr[k], 0.02, rng);
  std::vector<Sophus::SE3d> poses_init;
  poses_init.reserve(gt_poses.size());
  for (const Sophus::SE3d& T : gt_poses) poses_init.push_back(perturbPose(T, 0.03, rng));

  std::vector<CameraFactory> facs(static_cast<std::size_t>(n_cams), pinholeFactory());
  const auto res = calibforge::calibrateRig(views, intr_init, extr_init, poses_init, facs);
  SolveOutcome o;
  o.final_cost = res.summary.final_cost;
  o.iterations = res.summary.iterations;
  return o;
}

// Single-camera reprojection problem assembled directly on DenseProblem, solved with the chosen
// backend. Isolates the SOLVER (the linear algebra the GPU back-end changes) on identical input,
// so the CPU-vs-GPU comparison is apples-to-apples. n_tangent = 4 + 6*n_views grows with views.
// [[maybe_unused]]: only called from the CALIBFORGE_HAS_CUDA CPU-vs-GPU section below, so on a
// host-only build (no nvcc) it is unused — without this, -Werror=unused-function breaks the strict
// CI job. Kept compiled (not #ifdef'd out) so the host build still type-checks it.
[[maybe_unused]] SolveOutcome solveDenseSingle(int n_views, int board_rows, int board_cols,
                                               SolverBackend backend, std::uint64_t seed) {
  PinholeCamera gt(500.0, 500.0, 320.0, 240.0);
  const std::vector<Vec3> board = makeBoard(board_rows, board_cols, 0.1);
  const std::vector<Sophus::SE3d> gt_poses = makeRandomPoses(n_views, seed);

  std::mt19937_64 rng(seed ^ 0xa5a5);
  std::vector<double> intr = {480.0, 520.0, 310.0, 250.0};
  std::vector<std::array<double, 7>> pose(gt_poses.size());
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    SE3Param::store(perturbPose(gt_poses[i], 0.03, rng), pose[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr.data(), std::make_shared<EuclideanParam>(4));
  for (auto& p : pose) problem.addParameterBlock(p.data(), std::make_shared<SE3Param>());
  for (std::size_t i = 0; i < gt_poses.size(); ++i)
    for (const Vec3& X : board) {
      const Eigen::Vector3d Xc = gt_poses[i] * Eigen::Vector3d(X[0], X[1], X[2]);
      const Vec2 px = gt.project(Vec3{Xc.x(), Xc.y(), Xc.z()});
      problem.addResidualBlock(std::make_unique<ReprojectionResidual>(pinholeFactory(), 4, X, px),
                               {intr.data(), pose[i].data()});
    }
  const auto s = problem.solveLm(LmOptions{}, backend);
  SolveOutcome o;
  o.final_cost = s.final_cost;
  o.iterations = s.iterations;
  return o;
}

}  // namespace

int main() {
  std::printf("problem,n_views,n_points,n_cams,cpu_ms_median,cpu_iters_median,final_cost_median\n");

  // Single-camera sweep (problem grows in #views and #points). Calibration regime: small;
  // CPU should dominate. CLAUDE.md rule 1: measure before assuming.
  for (int v : {5, 10, 20, 40}) {
    const Timing t = repeat(5, [&]() { return runSingle(v, 8, 8, 0xC0FFEE); });
    std::printf("single,%d,%d,1,%.3f,%d,%.3e\n", v, 8 * 8, t.median_ms, t.median_iters,
                t.median_final_cost);
  }

  // Stereo sweep.
  for (int v : {5, 10, 20}) {
    const Timing t = repeat(5, [&]() { return runStereo(v, 6, 9, 0xBADC0DE); });
    std::printf("stereo,%d,%d,2,%.3f,%d,%.3e\n", v, 6 * 9, t.median_ms, t.median_iters,
                t.median_final_cost);
  }

  // N-camera rig sweep.
  for (int n_cams : {3, 4, 6}) {
    const Timing t = repeat(3, [&]() { return runRig(10, n_cams, 5, 7, 0xFEEDFACE); });
    std::printf("rig,%d,%d,%d,%.3f,%d,%.3e\n", 10, 5 * 7, n_cams, t.median_ms,
                t.median_iters, t.median_final_cost);
  }

  // --- CPU-vs-GPU solver comparison (native CUDA back-end, #25). Same DenseProblem solved with
  //     each backend. RULE #1: the GPU is NOT automatically faster — for small calibrations the
  //     host<->device transfer + per-step malloc/launch overhead dominates an O(n^3) solve, so
  //     the CPU wins; the GPU earns its keep only as the problem grows. The sweep DELIBERATELY
  //     starts at n_views=1 (n_tangent=10) so it actually samples the small-calibration regime
  //     RULE #1 is about and LOCATES the crossover, instead of starting past it (no silent caps).
  //     Measured on an RTX 5090 / CUDA 12.0: CPU wins through ~5 views, GPU crosses over at ~8
  //     (n_tangent~52); the gap then widens to ~6-7x by 80 views. Crossover is hardware-specific;
  //     re-measure per target (Jetson Orin sm_87 etc.) — never assume. ---
#ifdef CALIBFORGE_HAS_CUDA
  if (calibforge::cudaSolverAvailable()) {
    solveDenseSingle(20, 9, 9, SolverBackend::GpuCuda, 0xC0FFEE);     // warm up CUDA ctx / JIT / handles
    solveDenseSingle(20, 9, 9, SolverBackend::GpuCudaF32, 0xC0FFEE);  // warm up the FP32 device path too
    // Iteration counts are reported alongside the timings: a speedup is only meaningful if all
    // backends did the SAME amount of work (converged in ~the same #iterations to ~the same cost).
    // gpu64 = SolverBackend::GpuCuda (FP64), gpu32 = SolverBackend::GpuCudaF32 (single-precision
    // device step, FP64 cost/acceptance) — both vs the same CPU oracle, so the gpu32 column the
    // docs report (docs/BENCHMARKS.md / SPIKES.md §F.4) is reproducible from this committed tool.
    std::printf(
        "\nproblem,n_views,n_points,n_cams,n_tangent,cpu_ms_median,gpu64_ms_median,gpu32_ms_median,"
        "gpu64_speedup,gpu32_speedup,cpu_iters,gpu64_iters,gpu32_iters,cpu_cost_median,"
        "gpu64_cost_median,gpu32_cost_median\n");
    bool any_cost_divergence = false;
    for (int v : {1, 2, 3, 5, 8, 10, 20, 40, 80}) {
      // More reps where each solve is cheap (the small regime that pins the crossover, and where
      // timer noise matters most); fewer where a single solve already costs hundreds of ms.
      const int reps = (v <= 10) ? 15 : 5;
      const Timing tc =
          repeat(reps, [&]() { return solveDenseSingle(v, 9, 9, SolverBackend::CpuCeres, 0xC0FFEE); });
      const Timing tg =
          repeat(reps, [&]() { return solveDenseSingle(v, 9, 9, SolverBackend::GpuCuda, 0xC0FFEE); });
      const Timing tg32 = repeat(
          reps, [&]() { return solveDenseSingle(v, 9, 9, SolverBackend::GpuCudaF32, 0xC0FFEE); });
      const double speedup = tc.median_ms / std::max(tg.median_ms, 1e-9);
      const double speedup32 = tc.median_ms / std::max(tg32.median_ms, 1e-9);
      std::printf("dense_single,%d,%d,1,%d,%.3f,%.3f,%.3f,%.2f,%.2f,%d,%d,%d,%.3e,%.3e,%.3e\n", v,
                  9 * 9, 4 + 6 * v, tc.median_ms, tg.median_ms, tg32.median_ms, speedup, speedup32,
                  tc.median_iters, tg.median_iters, tg32.median_iters, tc.median_final_cost,
                  tg.median_final_cost, tg32.median_final_cost);
      // A speedup is apples-to-apples only if BOTH backends reach the same minimum. The scenes are
      // noise-free, so a correct solve drives final_cost to ~0 (machine precision, ~1e-23); the
      // meaningful check is "did both converge to the noise-free minimum?", NOT a relative cost
      // ratio (two ~1e-25 values have a meaningless ratio at the bottom of the basin). Flag a row
      // only if either backend's cost stays above a generous convergence floor — that would mean
      // one solver failed to converge while the other succeeded, invalidating the speedup (RULE #1:
      // precision != correctness; never report a speedup between non-equivalent solves).
      const double worst_cost = std::max(
          tc.median_final_cost, std::max(tg.median_final_cost, tg32.median_final_cost));
      if (worst_cost > 1e-6) {
        any_cost_divergence = true;
        std::fprintf(stderr,
                     "  WARNING dense_single n_views=%d: a backend did not reach the noise-free "
                     "minimum (cpu=%.3e gpu64=%.3e gpu32=%.3e) — the speedup for this row is NOT "
                     "apples-to-apples.\n",
                     v, tc.median_final_cost, tg.median_final_cost, tg32.median_final_cost);
      }
    }
    if (!any_cost_divergence)
      std::fprintf(stderr,
                   "  (both backends converged to the noise-free minimum on every row — speedups "
                   "are apples-to-apples.)\n");
    std::fprintf(stderr,
                 "\nGPU rows above: native CalibForge CUDA back-end (cuBLAS SYRK/GEMV + cuSOLVER "
                 "Cholesky) vs host Eigen, identical DenseProblem. gpu_speedup<1 => CPU faster "
                 "(small single calibration; RULE #1 — observed at the low-n_views rows, where the "
                 "CPU is the default backend). The crossover above which the GPU wins is "
                 "hardware-specific and MEASURED here, never assumed. PyPose/MegBA back-ends + "
                 "Jetson<->server parity remain pending (#25 / SPIKES.md §D).\n");
  } else
#endif
  {
    std::fprintf(stderr,
                 "\nGPU rows pending: this build has no CUDA back-end (CALIBFORGE_HAS_CUDA off). "
                 "Build on a CUDA host for the native-CUDA-vs-CPU solver rows. CPU rows above are "
                 "the apples-to-apples baseline.\n");
  }
  return 0;
}
