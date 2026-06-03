#pragma once
//
// CalibForge pipeline — surround-view rig online extrinsic recalibration (header-only, CPU).
//
// Bundles the v0.5 surround-view targetless path:
//
//   per-frame {rig_pose, per-camera images}
//     -> bevRandomSearchExtrinsics (coarse-to-fine BEV photometric random search)
//     -> apply two gates (RULE #2):
//          * cost-reduction sanity (BEV overlap cost must DROP meaningfully)
//          * 6-axis rig-pose motion-excitation (mirror of OnlineExtrinsicTracker's gate)
//     -> on pass: emit refined extrinsics + per-camera drift vs reference
//     -> on fail: refuse + name the failing gate (cost-stagnant or unexcited axes)
//
// The random-search baseline (OpenCalib SurroundCameraCalib, math RE-IMPLEMENTED) is the
// front-end here; LM refinement on photometric residuals would come later. Random search is
// robust to the large initial extrinsic error that surround rigs ship with from the factory.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/bev_photometric.hpp"
#include "calibforge/calibrate_single.hpp"   // CameraFactory
#include "calibforge/image.hpp"
#include "calibforge/online_extrinsic_tracker.hpp"  // MotionExcitationOptions, ExtrinsicEmission semantics
#include "sophus/se3.hpp"

namespace calibforge {
namespace pipelines {

struct OnlineSurroundRigOptions {
  detect::BevGridOptions bev_grid;
  detect::BevRandomSearchOptions search;
  online::MotionExcitationOptions motion;
  // Cost reduction must clear this fraction (e.g. 0.05 = 5% drop in mean squared intensity
  // diff over the BEV overlap zones) — guards against silent random-walk emission.
  double min_cost_reduction = 0.05;
  // Minimum number of overlap pairs sampled by the BEV grid across the window. Below this
  // the random search has no signal to work with; refuse the emit.
  int min_overlap_pairs = 200;
};

struct SurroundRigEmission {
  bool emitted = false;
  std::vector<Sophus::SE3d> extrinsics;   // size N-1, refined T_ck_c0
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double cost_reduction = 0.0;
  int overlap_count = 0;
  double drift = 0.0;                     // sum_k ||Log(extr_new[k] * extr_ref[k]^-1)||
  bool refused_for_motion = false;
  bool refused_for_no_signal = false;
  bool refused_for_no_reduction = false;
  std::vector<std::string> unexcited_axes;
};

class OnlineSurroundRig {
 public:
  OnlineSurroundRig(std::vector<CameraFactory> make_cameras,
                    std::vector<Eigen::VectorXd> intrinsics_reference,
                    std::vector<Sophus::SE3d> extrinsics_reference,
                    OnlineSurroundRigOptions opts = {})
      : opts_(std::move(opts)),
        make_(std::move(make_cameras)),
        intr_ref_(std::move(intrinsics_reference)),
        extr_ref_(std::move(extrinsics_reference)) {}

  void addFrame(const std::vector<const Image8*>& images,
                const Sophus::SE3d& rig_pose_T_c0_w) {
    frames_.push_back(images);
    poses_.push_back(rig_pose_T_c0_w);
  }
  std::size_t windowSize() const { return frames_.size(); }
  void reset() {
    frames_.clear();
    poses_.clear();
  }

  SurroundRigEmission tryEmit() const {
    SurroundRigEmission e;
    if (frames_.empty()) return e;

    // Motion gate FIRST (free, refuses cheaply on a degenerate window).
    const std::vector<std::string> unexcited = checkMotionExcitation(opts_.motion);
    if (!unexcited.empty()) {
      e.refused_for_motion = true;
      e.unexcited_axes = unexcited;
      return e;
    }

    // Build per-camera CameraModels from the reference intrinsics (held fixed during
    // photometric search; intrinsic recal is a separate path).
    std::vector<std::unique_ptr<CameraModel>> cam_storage;
    std::vector<const CameraModel*> cams;
    cam_storage.reserve(make_.size());
    cams.reserve(make_.size());
    for (std::size_t c = 0; c < make_.size(); ++c) {
      cam_storage.push_back(make_[c](intr_ref_[c]));
      cams.push_back(cam_storage.back().get());
    }

    const std::vector<detect::BevSample> samples = detect::makeBevGrid(opts_.bev_grid);

    const detect::BevRandomSearchResult r = detect::bevRandomSearchExtrinsics(
        samples, cams, extr_ref_, poses_, frames_, opts_.search);

    e.initial_cost = r.initial_cost;
    e.final_cost = r.final_cost;
    e.cost_reduction = r.cost_reduction_ratio;
    e.overlap_count = r.overlap_count;

    if (r.overlap_count < opts_.min_overlap_pairs) {
      e.refused_for_no_signal = true;
      return e;
    }
    if (r.cost_reduction_ratio < opts_.min_cost_reduction) {
      e.refused_for_no_reduction = true;
      return e;
    }

    e.emitted = true;
    e.extrinsics = r.extrinsics;
    double drift = 0.0;
    for (std::size_t k = 0; k < r.extrinsics.size() && k < extr_ref_.size(); ++k)
      drift += (r.extrinsics[k] * extr_ref_[k].inverse()).log().norm();
    e.drift = drift;
    return e;
  }

 private:
  // Mirror OnlineExtrinsicTracker's 6-axis motion gate so the two orchestrators apply the
  // same degeneracy semantics. Free-of-charge guard against random search emitting on a
  // hover / static window.
  std::vector<std::string> checkMotionExcitation(
      const online::MotionExcitationOptions& opts) const {
    std::vector<std::string> unexcited;
    if (poses_.size() < 2) {
      unexcited = {"trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"};
      return unexcited;
    }
    Eigen::Vector3d t_min = poses_.front().translation();
    Eigen::Vector3d t_max = t_min;
    Eigen::Vector3d r_min = poses_.front().so3().log();
    Eigen::Vector3d r_max = r_min;
    for (std::size_t i = 1; i < poses_.size(); ++i) {
      const Eigen::Vector3d t = poses_[i].translation();
      const Eigen::Vector3d r = poses_[i].so3().log();
      for (int a = 0; a < 3; ++a) {
        if (t[a] < t_min[a]) t_min[a] = t[a]; else if (t[a] > t_max[a]) t_max[a] = t[a];
        if (r[a] < r_min[a]) r_min[a] = r[a]; else if (r[a] > r_max[a]) r_max[a] = r[a];
      }
    }
    static const char* trans_names[3] = {"trans_x", "trans_y", "trans_z"};
    static const char* rot_names[3]   = {"rot_x",   "rot_y",   "rot_z"};
    for (int a = 0; a < 3; ++a) {
      if (t_max[a] - t_min[a] < opts.trans_range_threshold) unexcited.push_back(trans_names[a]);
      if (r_max[a] - r_min[a] < opts.rot_range_threshold)   unexcited.push_back(rot_names[a]);
    }
    return unexcited;
  }

  OnlineSurroundRigOptions opts_;
  std::vector<CameraFactory> make_;
  std::vector<Eigen::VectorXd> intr_ref_;
  std::vector<Sophus::SE3d> extr_ref_;
  std::vector<std::vector<const Image8*>> frames_;
  std::vector<Sophus::SE3d> poses_;
};

}  // namespace pipelines
}  // namespace calibforge
