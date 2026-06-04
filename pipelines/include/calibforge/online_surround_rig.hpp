#pragma once
//
// CalibForge pipeline — surround-view rig online extrinsic recalibration (header-only, CPU).
//
// Bundles the v0.5 surround-view targetless path:
//
//   per-frame {rig_pose, per-camera images}
//     -> bevRandomSearchExtrinsics (coarse-to-fine BEV photometric random search)
//     -> gate before emit (RULE #2):
//          * cheap PRE-FILTERS: 6-axis rig-pose motion-excitation + overlap coverage + the BEV
//            cost must drop meaningfully (guards against random-walk emission)
//          * the OBSERVABILITY gate (the real RULE #2 guard): build the photometric information
//            matrix H = J^T J of the BEV agreement residuals over the 6*(N-1) extrinsic tangent
//            (bevInformationMatrix) and run assessObservability()/parameterUncertainty() on it —
//            emit ONLY when H is observable and its reciprocal condition clears min_confidence.
//            A low photometric cost alone does NOT mean accurate extrinsics (precision != accuracy).
//     -> on pass: emit refined extrinsics + confidence + per-camera drift vs reference
//     -> on fail: refuse + name the failing gate (motion / no-signal / no-reduction /
//        ill-conditioned) + the weak/unobservable extrinsic directions
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
#include "calibforge/observability.hpp"      // assessObservability, parameterUncertainty (RULE #2)
#include "calibforge/online_extrinsic_tracker.hpp"  // MotionExcitationOptions, ExtrinsicEmission semantics
#include "sophus/se3.hpp"

namespace calibforge {
namespace pipelines {

struct OnlineSurroundRigOptions {
  detect::BevGridOptions bev_grid;
  detect::BevRandomSearchOptions search;
  online::MotionExcitationOptions motion;
  ObservabilityOptions obs_opts{};
  // The observability gate threshold: emit only when the photometric information matrix's
  // reciprocal condition number clears this. A healthy estimate scores roughly ~1e-3..1e-2 on
  // this BEV path (geometry-dependent; cf. solve/observability.hpp). This is the OBSERVABILITY
  // / CONDITIONING gate — NECESSARY but NOT sufficient for accuracy: a well-conditioned
  // photometric minimum can still be BIASED (the coarse BEV minimum is, in fact, biased), so a
  // high confidence here means "this direction is constrained by the data", NOT "this estimate
  // is accurate" (precision != accuracy, RULE #2). It guards against emitting unconstrained
  // directions; it does not certify the estimate is unbiased.
  double min_confidence = 1e-6;
  // Finite-difference step (extrinsic SE3 tangent) used to build the photometric information
  // matrix; small enough for accuracy, large enough to clear bilinear-sampling quantization.
  double fd_step = 1e-3;
  // --- The two below are cheap PRE-FILTERS demoted ahead of the observability gate above.
  //     They reject obvious random-walk / no-signal windows early (neither is an accuracy or
  //     observability guarantee — that is the gate above). ---
  // Cost reduction must clear this fraction (e.g. 0.05 = 5% drop in mean squared intensity
  // diff over the BEV overlap zones).
  double min_cost_reduction = 0.05;
  // Minimum number of overlap pairs sampled by the BEV grid across the window.
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
  double confidence = 0.0;               // observability reciprocal condition number [0,1]
  bool observable = false;               // photometric information matrix full-rank + conditioned
  std::vector<std::string> weak_parameters;     // weak/unobservable extrinsic directions
  bool refused_for_motion = false;
  bool refused_for_no_signal = false;
  bool refused_for_no_reduction = false;
  bool refused_for_observability = false;       // FIM gate refused (ill-conditioned extrinsics)
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

    // Input-size validation: refuse (rather than read out of bounds) on inconsistent
    // constructor / addFrame arguments.
    const std::size_t N = make_.size();
    if (N < 2 || intr_ref_.size() != N || extr_ref_.size() + 1 != N) {
      e.refused_for_no_signal = true;
      return e;
    }
    for (const std::vector<const Image8*>& imgs : frames_) {
      if (imgs.size() != N) {
        e.refused_for_no_signal = true;
        return e;
      }
    }

    // Motion-excitation PRE-FILTER first (free, refuses cheaply on a degenerate window).
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
    cam_storage.reserve(N);
    cams.reserve(N);
    for (std::size_t c = 0; c < N; ++c) {
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

    // Coverage + cost-reduction PRE-FILTERS (cheap; not the accuracy guarantee).
    if (r.overlap_count < opts_.min_overlap_pairs) {
      e.refused_for_no_signal = true;
      return e;
    }
    if (r.cost_reduction_ratio < opts_.min_cost_reduction) {
      e.refused_for_no_reduction = true;
      return e;
    }

    // OBSERVABILITY GATE (RULE #2): a low photometric cost does NOT mean the extrinsics are
    // accurate. Build the photometric information matrix at the refined extrinsics and require
    // it to be observable AND clear min_confidence before emitting. Surface the weak /
    // unobservable directions either way.
    const detect::BevInformation info = detect::bevInformationMatrix(
        samples, cams, r.extrinsics, poses_, frames_, opts_.fd_step);
    std::vector<std::string> extr_names;
    extr_names.reserve(6 * (N - 1));
    for (std::size_t k = 1; k < N; ++k) {
      const std::string base = "extr_" + std::to_string(k);
      for (const char* a : {"_tx", "_ty", "_tz", "_rx", "_ry", "_rz"})
        extr_names.push_back(base + a);
    }
    const ObservabilityReport rep = assessObservability(info.H, opts_.obs_opts);
    const ParameterUncertainty unc = parameterUncertainty(
        info.H, 0.5 * info.residual_ssd, info.num_residuals, extr_names, opts_.obs_opts);
    e.confidence = rep.confidence;
    e.observable = rep.observable;
    e.weak_parameters = unc.weak_parameters;
    if (!(rep.observable && rep.confidence >= opts_.min_confidence)) {
      e.refused_for_observability = true;
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
