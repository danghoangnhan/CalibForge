#pragma once
//
// CalibForge detect — multi-view triangulation + FeatureTrack -> View glue (header-only).
//
// The targetless monocular path (UAV in-flight self-cal) emits FeatureTracks — sparse 2D
// observations of the same scene point across N frames of a single camera with known per-
// frame poses (from odometry / VIO). OnlineIntrinsicTracker consumes Views (3D-2D pairs +
// per-view pose). This module bridges them:
//
//   FeatureTrack{id, obs[(frame_id, uv)]} + camera model + per-frame poses
//   --triangulate-->  Vec3 landmark in world frame (linear DLT, cheirality + conditioning gate)
//   --pack-->         std::vector<View> per frame for OnlineIntrinsicTracker.addFrame()
//
// The N-camera surround-rig targetless path uses BEV photometric residuals (bev_photometric.hpp)
// instead — cross-camera FOV overlap on the ground plane is the constraint that drives
// online extrinsic re-estimation there.
//
// CPU, dependency-free beyond Eigen + Sophus.

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
  bool ok = false;            // true when conditioning + cheirality both pass
  double condition = 0.0;     // smallest-to-largest singular-value ratio (0 = rank-deficient)
};

namespace detail {
inline Eigen::Vector3d toEig(const Vec3& v) { return Eigen::Vector3d(v[0], v[1], v[2]); }
inline Vec3 fromEig(const Eigen::Vector3d& v) { return Vec3{v[0], v[1], v[2]}; }
}  // namespace detail

// Linear DLT triangulation of a single landmark via the "parallel-ray" cross-product form.
// For each observation, the camera-frame unit ray r_c is lifted to world (r_w = R_wc * r_c)
// and we stack [r_w]_x (X - t_wc) = 0; rewritten as [r_w]_x X = [r_w]_x t_wc, 3 rows per obs.
// Linear least squares solves it; cheirality + conditioning then gate the result.
inline TriangulationResult triangulateLinear(
    const std::vector<Vec2>& obs_uvs,
    const std::vector<const CameraModel*>& obs_cameras,
    const std::vector<Sophus::SE3d>& obs_T_world_cam,
    double condition_threshold = 1e-6) {
  TriangulationResult out;
  const std::size_t N = obs_uvs.size();
  if (N < 2 || obs_cameras.size() != N || obs_T_world_cam.size() != N) return out;

  Eigen::MatrixXd A(3 * static_cast<int>(N), 3);
  Eigen::VectorXd b(3 * static_cast<int>(N));
  for (std::size_t i = 0; i < N; ++i) {
    const Vec3 ray_cam = obs_cameras[i]->unproject(obs_uvs[i]);
    const Eigen::Vector3d ray_w = obs_T_world_cam[i].so3() * detail::toEig(ray_cam);
    const Eigen::Vector3d t_w = obs_T_world_cam[i].translation();
    const Eigen::Matrix3d K = skew3(ray_w);
    A.block<3, 3>(3 * static_cast<int>(i), 0) = K;
    b.segment<3>(3 * static_cast<int>(i)) = K * t_w;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& sv = svd.singularValues();
  if (sv(0) <= 0.0) return out;
  out.condition = sv(sv.size() - 1) / sv(0);
  if (out.condition < condition_threshold) return out;

  const Eigen::Vector3d X = svd.solve(b);

  // Cheirality: positive Z in every observing camera frame. Rejects mirror-image solutions
  // when the linear system has a sign ambiguity.
  for (std::size_t i = 0; i < N; ++i) {
    const Eigen::Vector3d X_cam = obs_T_world_cam[i].inverse() * X;
    if (X_cam.z() <= 0.0) return out;
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
    double condition_threshold = 1e-6) {
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
    const TriangulationResult tr = triangulateLinear(uvs, cams, poses, condition_threshold);
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
