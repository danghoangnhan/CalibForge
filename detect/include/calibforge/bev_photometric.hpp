#pragma once
//
// CalibForge detect — BEV photometric residual + coarse-to-fine random search baseline
// (header-only, CPU). The surround-view targetless front-end for online extrinsic
// recalibration.
//
// Reference (math only, MIT-style re-implementation):
//   OpenCalib SurroundCameraCalib (Apache-2.0). The trick is that a surround-view rig with
//   overlapping FOVs on the ground plane has a photometric constraint: a point (X, Y, 0) on
//   the ground projects to PIXEL in each camera that sees it; the recovered intensities at
//   those pixels must AGREE up to lighting/exposure noise. Mis-calibrated extrinsics produce
//   a systematic BEV-overlap intensity gap. Random search on the extrinsics minimizes it.
//
// Output of this module is updated per-camera extrinsics; the online_surround_rig.hpp
// orchestrator wires the result behind the same observability/motion gates as the rest of
// the differentiator (RULE #2 — never silently emit ill-conditioned params).
//
// Coordinate convention:
//   - Ground plane = z=0 in the WORLD frame (the rig's local nav frame).
//   - The reference camera cam0 has pose T_c0_w (world -> cam0); we pass it in per frame.
//   - Each non-reference camera k has extrinsic T_ck_c0 (cam0 -> cam_k); cam_k pose for a
//     world point X is X_ck = T_ck_c0 * T_c0_w * X.
//
// Pixel sampling uses Image8::bilinear() (zero-border), consistent with the apply path.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"  // CameraModel, Vec2, Vec3
#include "calibforge/image.hpp"
#include "sophus/se3.hpp"

namespace calibforge {
namespace detect {

struct BevSample {
  double X = 0.0;  // world x, metres
  double Y = 0.0;  // world y, metres
};

struct BevGridOptions {
  double x_min = -2.0;
  double x_max = 2.0;
  double y_min = -2.0;
  double y_max = 2.0;
  double step = 0.05;
};

// Regular grid of world-frame ground-plane samples. The orchestrator sets the grid extent
// to cover the rig's expected overlap zones.
inline std::vector<BevSample> makeBevGrid(const BevGridOptions& opts) {
  std::vector<BevSample> samples;
  const int nx = static_cast<int>(std::floor((opts.x_max - opts.x_min) / opts.step)) + 1;
  const int ny = static_cast<int>(std::floor((opts.y_max - opts.y_min) / opts.step)) + 1;
  samples.reserve(static_cast<std::size_t>(nx) * ny);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      samples.push_back({opts.x_min + i * opts.step, opts.y_min + j * opts.step});
    }
  }
  return samples;
}

// Project a world-frame ground-plane sample (X, Y, 0) to camera k. Returns whether the
// projection is in front of the camera AND in-image; the resulting pixel is in (u, v).
inline bool projectGroundSample(const BevSample& s, const Sophus::SE3d& T_ck_w,
                                const CameraModel& cam, int img_w, int img_h, double& u,
                                double& v) {
  const Eigen::Vector3d X_w(s.X, s.Y, 0.0);
  const Eigen::Vector3d X_c = T_ck_w * X_w;
  if (X_c.z() <= 1e-3) return false;
  const Vec2 uv = cam.project(Vec3{X_c.x(), X_c.y(), X_c.z()});
  u = uv[0];
  v = uv[1];
  if (u < 0.0 || v < 0.0 || u > img_w - 1.0 || v > img_h - 1.0) return false;
  return true;
}

// Sum of squared intensity differences over all (sample, camera-pair) where BOTH cameras
// see the ground sample. Cameras unable to observe the sample are silently skipped — the
// signal comes from the OVERLAP zones. Returns 0 with overlap_count=0 when no overlaps
// occur, which the caller (random search / orchestrator) treats as "no information".
struct BevAgreementResult {
  double cost = 0.0;          // sum of squared differences (un-normalized)
  int overlap_count = 0;      // pairs counted in the cost; 0 => no signal
  double mean_sq_diff = 0.0;  // cost / overlap_count when overlap_count > 0
};

inline BevAgreementResult bevAgreementCost(
    const std::vector<BevSample>& samples,
    const std::vector<const CameraModel*>& cameras,                   // size N
    const std::vector<Sophus::SE3d>& T_ck_c0_with_identity_first,     // size N (0th is identity)
    const Sophus::SE3d& T_c0_w,
    const std::vector<const Image8*>& images,                          // size N
    int min_overlap = 1) {
  BevAgreementResult res;
  const int N = static_cast<int>(cameras.size());
  if (N < 2) return res;

  for (const BevSample& s : samples) {
    // Per-sample, gather intensity from each camera that sees it. A small fixed array keeps
    // the inner loop cache-friendly.
    std::array<double, 16> intens{};
    std::array<bool, 16> seen{};
    int count = 0;
    for (int k = 0; k < N && k < 16; ++k) {
      const Sophus::SE3d T_ck_w = T_ck_c0_with_identity_first[k] * T_c0_w;
      double u = 0.0, v = 0.0;
      if (!projectGroundSample(s, T_ck_w, *cameras[k], images[k]->width, images[k]->height,
                               u, v))
        continue;
      intens[k] = images[k]->bilinear(u, v);
      seen[k] = true;
      ++count;
    }
    if (count < std::max(2, min_overlap + 1)) continue;

    // Accumulate squared differences over every ordered camera pair that both saw the
    // sample. The pair count grows as O(count^2); for typical 4-6 surround rigs this is
    // bounded (<=15 pairs / sample).
    for (int a = 0; a < N; ++a) {
      if (!seen[a]) continue;
      for (int b = a + 1; b < N; ++b) {
        if (!seen[b]) continue;
        const double d = intens[a] - intens[b];
        res.cost += d * d;
        ++res.overlap_count;
      }
    }
  }
  if (res.overlap_count > 0) {
    res.mean_sq_diff = res.cost / static_cast<double>(res.overlap_count);
  }
  return res;
}

struct BevRandomSearchOptions {
  // Coarse-to-fine perturbation scales applied successively. Each level samples
  // attempts_per_level perturbations; the best is kept and the next level shrinks the scale.
  std::vector<double> rot_scales = {0.05, 0.02, 0.005};   // radians (SE3 tangent rot part)
  std::vector<double> trans_scales = {0.05, 0.02, 0.005}; // metres   (SE3 tangent trans part)
  int attempts_per_level = 64;
  std::uint64_t seed = 0xCA1BF09Eu;
};

struct BevRandomSearchResult {
  std::vector<Sophus::SE3d> extrinsics;   // size N-1, refined T_ck_c0
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int overlap_count = 0;
  double cost_reduction_ratio = 0.0;      // (initial - final) / max(initial, eps); in [0, 1]
};

// Coarse-to-fine random search refines extrinsics_init -> extrinsics_refined by minimizing
// the BEV agreement cost across the supplied (rig pose, image) sample set. Each call to the
// internal cost is averaged across all supplied rig frames + images so the result reflects
// the whole window, not a single snapshot.
//
// The "identity + extrinsics" layout matches the rest of the rig pipeline: cam0 is the
// anchor (its extrinsic is identity and is not optimized); only T_ck_c0 for k=1..N-1 move.
inline BevRandomSearchResult bevRandomSearchExtrinsics(
    const std::vector<BevSample>& samples,
    const std::vector<const CameraModel*>& cameras,           // size N
    const std::vector<Sophus::SE3d>& extrinsics_init,         // size N-1, T_ck_c0
    const std::vector<Sophus::SE3d>& rig_poses,               // size F, T_c0_w
    const std::vector<std::vector<const Image8*>>& images,    // size F, each size N
    const BevRandomSearchOptions& opts = {}) {
  BevRandomSearchResult out;
  const int N = static_cast<int>(cameras.size());
  if (N < 2) return out;
  if (extrinsics_init.size() + 1 != static_cast<std::size_t>(N)) return out;
  if (rig_poses.size() != images.size() || rig_poses.empty()) return out;

  auto evalAvg = [&](const std::vector<Sophus::SE3d>& extr) {
    std::vector<Sophus::SE3d> full(static_cast<std::size_t>(N));
    full[0] = Sophus::SE3d();  // identity
    for (int k = 1; k < N; ++k) full[k] = extr[k - 1];
    BevAgreementResult agg;
    for (std::size_t f = 0; f < rig_poses.size(); ++f) {
      const BevAgreementResult r = bevAgreementCost(samples, cameras, full, rig_poses[f],
                                                    images[f]);
      agg.cost += r.cost;
      agg.overlap_count += r.overlap_count;
    }
    if (agg.overlap_count > 0)
      agg.mean_sq_diff = agg.cost / static_cast<double>(agg.overlap_count);
    return agg;
  };

  std::vector<Sophus::SE3d> best = extrinsics_init;
  BevAgreementResult best_r = evalAvg(best);
  out.initial_cost = best_r.mean_sq_diff;
  out.overlap_count = best_r.overlap_count;

  std::mt19937_64 rng(opts.seed);
  std::normal_distribution<double> nrm(0.0, 1.0);
  const std::size_t levels = std::min(opts.rot_scales.size(), opts.trans_scales.size());
  for (std::size_t L = 0; L < levels; ++L) {
    const double rs = opts.rot_scales[L];
    const double ts = opts.trans_scales[L];
    for (int a = 0; a < opts.attempts_per_level; ++a) {
      std::vector<Sophus::SE3d> trial = best;
      for (Sophus::SE3d& T : trial) {
        Eigen::Matrix<double, 6, 1> dx;
        dx << ts * nrm(rng), ts * nrm(rng), ts * nrm(rng),
              rs * nrm(rng), rs * nrm(rng), rs * nrm(rng);
        T = T * Sophus::SE3d::exp(dx);
      }
      const BevAgreementResult r = evalAvg(trial);
      // Only accept when the perturbed extrinsics ALSO have valid overlap (random walks
      // can push a camera off the ground plane).
      if (r.overlap_count > 0 && r.mean_sq_diff < best_r.mean_sq_diff) {
        best = std::move(trial);
        best_r = r;
      }
    }
  }

  out.extrinsics = std::move(best);
  out.final_cost = best_r.mean_sq_diff;
  if (out.initial_cost > 1e-12)
    out.cost_reduction_ratio = (out.initial_cost - out.final_cost) / out.initial_cost;
  return out;
}

}  // namespace detect
}  // namespace calibforge
