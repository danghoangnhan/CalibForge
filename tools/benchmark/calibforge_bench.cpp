// CalibForge — CPU calibration-regime benchmark harness.
//
// Implements the CPU rows of SPIKES.md §D.1: time the CPU solver on representative
// calibration problem sizes (#views, #points, #cameras). GPU rows fill in when a CUDA host
// is available (the deferred half of v1.0 — see issue #25 / SPIKES.md §D).
//
// Outputs a CSV table to stdout so it can be pasted into docs/BENCHMARKS.md when the GPU
// numbers exist. Each row reports:
//   problem, n_views, n_points, n_cams, cpu_ms_median, cpu_iters_median, final_cost_median
//
// Build (via the top-level CMakeLists.txt):
//   cmake --build build --target calibforge_bench
//   ./build/calibforge_bench
// Or compile standalone (header-only library):
//   c++ -std=c++17 -O2 -I core/include -I solve/include -I pipelines/include \
//       -I build/_deps/eigen-src -I build/_deps/sophus-src tools/benchmark/calibforge_bench.cpp \
//       -o calibforge_bench

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
#include "calibforge/pinhole_camera.hpp"
#include "sophus/se3.hpp"

namespace {

using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::PinholeCamera;
using calibforge::RigView;
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
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto r = fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
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

  std::fprintf(stderr,
               "\nGPU rows pending: SPIKES.md §D.1 (PyPose / Graphite / Ceres on the same "
               "regime) requires a CUDA / Jetson host. CPU rows above are the apples-to-apples "
               "baseline.\n");
  return 0;
}
