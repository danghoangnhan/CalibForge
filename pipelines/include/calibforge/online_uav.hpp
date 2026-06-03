#pragma once
//
// CalibForge pipeline — UAV in-flight self-calibration orchestrator (header-only, CPU).
//
// Bundles the v0.5 targetless monocular path with sensible UAV defaults:
//
//   FeatureTracker (KLT-lite over saddle corners; detect/feature_tracker.hpp)
//     -> triangulateTracks via known per-frame VIO/odometry poses (detect/triangulate.hpp)
//     -> packMonocularViews into per-frame View{object_points, image_points}
//     -> OnlineIntrinsicTracker.addFrame + tryEmit (the gated emission core)
//
// The orchestrator does NOT trust the emitted intrinsics on its own — it returns the gated
// Emission verbatim so the UAV autopilot can refuse drift updates when the gate refuses
// (RULE #2 — never silently emit ill-conditioned params).
//
// UAV-tuned defaults (overridable):
//   - 6-axis motion-excitation is INHERENT for free-flying UAVs (yaw + bank + climb + lateral
//     all routinely excited within seconds), but we still enforce it at the tracker. For a
//     hovering UAV the gate will correctly refuse.
//   - Triangulation min track length = 4 frames (UAV translational baselines are short per
//     frame; longer tracks = better conditioning).

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"     // View, CameraFactory
#include "calibforge/camera_model.hpp"
#include "calibforge/feature_tracker.hpp"
#include "calibforge/image.hpp"
#include "calibforge/online_calibration.hpp"   // OnlineIntrinsicTracker, Emission
#include "calibforge/triangulate.hpp"
#include "sophus/se3.hpp"

namespace calibforge {
namespace pipelines {

struct OnlineUavOptions {
  detect::FeatureTrackerOptions tracker;
  int triangulation_min_track_length = 4;
  double triangulation_condition_threshold = 1e-5;
  double emit_min_confidence = 1e-4;
  ObservabilityOptions obs_opts{};
  LmOptions lm_opts{};
};

// Stateful UAV self-cal orchestrator. Push one (image, T_world_cam) per VIO frame; query
// tryEmit() periodically. Reset() resets the feature tracker AND the intrinsic-tracker
// window.
class OnlineUav {
 public:
  OnlineUav(CameraFactory make_camera, Eigen::VectorXd intrinsics_reference,
            std::vector<std::string> param_names = {}, OnlineUavOptions opts = {})
      : opts_(std::move(opts)),
        make_camera_(make_camera),
        intr_ref_(intrinsics_reference),
        tracker_(opts_.tracker),
        intrinsic_(std::move(make_camera), std::move(intrinsics_reference),
                   std::move(param_names)) {}

  // Add one VIO frame. The camera model is built from the current reference intrinsics for
  // triangulation (we DO NOT have the optimized intrinsics yet — that's what we're solving
  // for, downstream).
  void addFrame(const Image8& image, const Sophus::SE3d& T_world_cam) {
    tracker_.addFrame(image);
    poses_.push_back(T_world_cam);
  }

  std::size_t windowSize() const { return poses_.size(); }
  std::size_t numTracks() const { return tracker_.numTracks(); }

  void reset() {
    tracker_ = detect::FeatureTracker(opts_.tracker);
    intrinsic_.reset();
    poses_.clear();
  }

  // Triangulate + pack + gated-emit. Returns the Emission verbatim (incl. weak_parameters
  // when the gate refuses). Triangulation that fails conditioning / cheirality is silently
  // dropped — only valid landmarks reach the gate.
  online::Emission tryEmit() {
    online::Emission e;
    if (poses_.size() < 2) return e;

    const std::unique_ptr<CameraModel> cam = make_camera_(intr_ref_);
    const std::vector<detect::TriangulatedTrack> tt = detect::triangulateTracks(
        tracker_.tracks(), cam.get(), poses_, opts_.triangulation_min_track_length,
        opts_.triangulation_condition_threshold);
    if (tt.empty()) return e;

    const std::vector<View> views =
        detect::packMonocularViews(tt, static_cast<int>(poses_.size()));

    // Hand the (View, pose) pairs to the gated intrinsic tracker. Note that the
    // intrinsic-tracker's internal window starts empty for each tryEmit — we re-populate
    // every call so the orchestrator stays a cheap stateless filter on top of the tracker.
    intrinsic_.reset();
    for (std::size_t f = 0; f < views.size(); ++f) {
      if (views[f].object_points.empty()) continue;
      intrinsic_.addFrame(views[f], poses_[f]);
    }
    return intrinsic_.tryEmit(opts_.emit_min_confidence, opts_.obs_opts, opts_.lm_opts);
  }

 private:
  OnlineUavOptions opts_;
  CameraFactory make_camera_;
  Eigen::VectorXd intr_ref_;
  detect::FeatureTracker tracker_;
  online::OnlineIntrinsicTracker intrinsic_;
  std::vector<Sophus::SE3d> poses_;
};

}  // namespace pipelines
}  // namespace calibforge
