#pragma once
//
// CalibForge detect — multi-view triangulation + FeatureTrack -> View glue (header-only).
//
// The targetless monocular path (UAV in-flight self-cal) emits FeatureTracks — sparse 2D
// observations of the same scene point across N frames of a single camera with known per-
// frame poses (from odometry / VIO). OnlineIntrinsicTracker consumes Views (3D-2D pairs +
// per-view pose). This module bridges them:
//
//   FeatureTrack{id, obs[(frame_id, uv)]} + camera model + per-frame poses (T_world_cam)
//   --triangulate-->  Vec3 landmark in world frame (inhomogeneous parallel-ray linear
//                     triangulation; cheirality + reciprocal-conditioning + parallax-angle gate)
//   --pack-->         std::vector<View> per frame for OnlineIntrinsicTracker.addFrame()
//
// POSE CONVENTION: poses are T_world_cam (camera -> world), so a camera-frame ray r_c lifts to
// world as r_w = R_wc * r_c and the camera centre is t_wc. Callers feeding these landmarks to
// OnlineIntrinsicTracker / ReprojectionResidual (which expect world -> camera, Xc = T * Xw)
// must invert the pose at that boundary — see pipelines/online_uav.hpp.
//
// The N-camera surround-rig targetless path uses BEV photometric residuals (bev_photometric.hpp)
// instead — cross-camera FOV overlap on the ground plane is the constraint that drives
// online extrinsic re-estimation there.
//
// CPU, dependency-free beyond Eigen + Sophus.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp" // View
#include "calibforge/camera_model.hpp"     // CameraModel, Vec2, Vec3
#include "calibforge/feature_tracker.hpp"  // FeatureTrack
#include "calibforge/manifold.hpp"         // skew3
#include "sophus/se3.hpp"

namespace calibforge {
namespace detect {

struct TriangulationResult {
  Vec3 point_world{0.0, 0.0, 0.0};
  bool ok = false;            // true when conditioning + parallax + cheirality all pass
  double condition = 0.0;     // smallest-to-largest singular-value ratio of A; flags ONLY
                              // exact/severe rank deficiency, NOT low-parallax weakness.
  double parallax_rad = 0.0;  // max angle between any pair of observation rays — the geometric
                              // triangulation strength (a depth point with near-parallel rays
                              // is poorly constrained even when `condition` looks fine).
};

namespace detail {
inline Eigen::Vector3d toEig(const Vec3& v) { return Eigen::Vector3d(v[0], v[1], v[2]); }
inline Vec3 fromEig(const Eigen::Vector3d& v) { return Vec3{v[0], v[1], v[2]}; }
}  // namespace detail

// Inhomogeneous "parallel-ray" linear triangulation of a single landmark. For each
// observation, the camera-frame ray r_c is lifted to world (r_w = R_wc * r_c) and we stack
// [r_w]_x (X - t_wc) = 0; rewritten as [r_w]_x X = [r_w]_x t_wc, 3 rows per obs. Linear least
// squares solves it; THREE gates then qualify the result:
//   * `condition` (reciprocal condition number of A) rejects exact/severe rank deficiency.
//   * `parallax_rad` (max inter-ray angle) rejects geometrically weak, near-parallel rays —
//     the conditioning number does NOT catch low parallax, which is the dominant failure mode
//     for forward-flying UAVs (small per-frame baseline vs scene depth).
//   * cheirality rejects points behind the camera, model-agnostically (works for >180deg FOV
//     lenses where a valid point can have camera-frame z<=0): X_cam must lie in the same
//     hemisphere as the model's own unproject ray (X_cam . r_c > 0).
inline TriangulationResult triangulateLinear(
    const std::vector<Vec2>& obs_uvs,
    const std::vector<const CameraModel*>& obs_cameras,
    const std::vector<Sophus::SE3d>& obs_T_world_cam,
    double condition_threshold = 1e-6,
    double min_parallax_rad = 0.017453292519943295 /* 1.0 deg */) {
  TriangulationResult out;
  const std::size_t N = obs_uvs.size();
  if (N < 2 || obs_cameras.size() != N || obs_T_world_cam.size() != N) return out;

  Eigen::MatrixXd A(3 * static_cast<int>(N), 3);
  Eigen::VectorXd b(3 * static_cast<int>(N));
  std::vector<Eigen::Vector3d> rays_cam(N);    // raw camera-frame rays, reused for cheirality
  std::vector<Eigen::Vector3d> rays_w_unit(N); // unit world rays, reused for parallax
  for (std::size_t i = 0; i < N; ++i) {
    const Eigen::Vector3d ray_cam = detail::toEig(obs_cameras[i]->unproject(obs_uvs[i]));
    rays_cam[i] = ray_cam;
    const Eigen::Vector3d ray_w = obs_T_world_cam[i].so3() * ray_cam;
    const double nw = ray_w.norm();
    rays_w_unit[i] = (nw > 1e-12) ? (ray_w / nw) : ray_w;
    const Eigen::Vector3d t_w = obs_T_world_cam[i].translation();
    const Eigen::Matrix3d K = skew3(ray_w);
    A.block<3, 3>(3 * static_cast<int>(i), 0) = K;
    b.segment<3>(3 * static_cast<int>(i)) = K * t_w;
  }

  // Parallax gate: the maximum angle between any two viewing rays. Below ~1deg the depth is
  // essentially unconstrained regardless of how well-conditioned A appears.
  double max_parallax = 0.0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j) {
      const double c = std::max(-1.0, std::min(1.0, rays_w_unit[i].dot(rays_w_unit[j])));
      max_parallax = std::max(max_parallax, std::acos(c));
    }
  out.parallax_rad = max_parallax;
  if (max_parallax < min_parallax_rad) return out;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& sv = svd.singularValues();
  if (sv(0) <= 0.0) return out;
  out.condition = sv(sv.size() - 1) / sv(0);
  if (out.condition < condition_threshold) return out;

  const Eigen::Vector3d X = svd.solve(b);

  // Cheirality, model-agnostic: X must be in front of every camera, i.e. on the same side as
  // that observation's viewing ray. (X_cam.z()>0 would wrongly reject valid points for
  // fisheye / double-sphere / EUCM lenses whose FOV exceeds 180deg.)
  for (std::size_t i = 0; i < N; ++i) {
    const Eigen::Vector3d X_cam = obs_T_world_cam[i].inverse() * X;
    if (X_cam.dot(rays_cam[i]) <= 0.0) return out;
  }
  out.point_world = detail::fromEig(X);
  out.ok = true;
  return out;
}

struct TriangulatedTrack {
  int track_id = 0;
  Vec3 point_world{0.0, 0.0, 0.0};
  std::vector<int> frame_ids;        // 1:1 with uvs
  std::vector<Vec2> uvs;
};

// Triangulate every track that crosses >= min_track_length frames. The single CameraModel*
// is shared across all observations (the UAV monocular case — intrinsic-tracking-only). For
// per-frame intrinsic / pose variation extend the API to a per-frame vector of cameras.
inline std::vector<TriangulatedTrack> triangulateTracks(
    const std::vector<FeatureTrack>& tracks,
    const CameraModel* camera,
    const std::vector<Sophus::SE3d>& T_world_cam_per_frame,
    int min_track_length = 3,
    double condition_threshold = 1e-6,
    double min_parallax_rad = 0.017453292519943295 /* 1.0 deg */) {
  std::vector<TriangulatedTrack> out;
  out.reserve(tracks.size());
  for (const FeatureTrack& t : tracks) {
    if (static_cast<int>(t.obs.size()) < min_track_length) continue;
    std::vector<Vec2> uvs;
    std::vector<const CameraModel*> cams;
    std::vector<Sophus::SE3d> poses;
    std::vector<int> fids;
    uvs.reserve(t.obs.size());
    cams.reserve(t.obs.size());
    poses.reserve(t.obs.size());
    fids.reserve(t.obs.size());
    for (const FeatureObservation& o : t.obs) {
      if (o.frame_id < 0 ||
          o.frame_id >= static_cast<int>(T_world_cam_per_frame.size())) continue;
      uvs.push_back(o.uv);
      cams.push_back(camera);
      poses.push_back(T_world_cam_per_frame[o.frame_id]);
      fids.push_back(o.frame_id);
    }
    if (static_cast<int>(uvs.size()) < min_track_length) continue;
    const TriangulationResult tr =
        triangulateLinear(uvs, cams, poses, condition_threshold, min_parallax_rad);
    if (!tr.ok) continue;
    TriangulatedTrack tt;
    tt.track_id = t.id;
    tt.point_world = tr.point_world;
    tt.frame_ids = std::move(fids);
    tt.uvs = std::move(uvs);
    out.push_back(std::move(tt));
  }
  return out;
}

// Pack triangulated tracks into per-frame Views. View[f] holds every (landmark, uv) pair
// the camera observed in frame f. Drops frames with zero observations from the output index
// by keeping them empty — OnlineIntrinsicTracker treats an empty View as a no-op residual
// contribution for that frame.
inline std::vector<View> packMonocularViews(
    const std::vector<TriangulatedTrack>& tracks, int num_frames) {
  if (num_frames <= 0) return {};  // guard the size_t cast below against a negative count
  std::vector<View> views(static_cast<std::size_t>(num_frames));
  for (const TriangulatedTrack& t : tracks) {
    for (std::size_t k = 0; k < t.frame_ids.size(); ++k) {
      const int f = t.frame_ids[k];
      if (f < 0 || f >= num_frames) continue;
      views[f].object_points.push_back(t.point_world);
      views[f].image_points.push_back(t.uvs[k]);
    }
  }
  return views;
}

}  // namespace detect
}  // namespace calibforge
