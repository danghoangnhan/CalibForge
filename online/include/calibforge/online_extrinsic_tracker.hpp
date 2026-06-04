#pragma once
//
// CalibForge online — gated online rig-extrinsic drift tracker (the v0.5 differentiator).
//
// Mirrors OnlineIntrinsicTracker but for rig extrinsics: holds the calibrated intrinsics
// fixed, re-estimates the (N-1) inter-camera extrinsics across a sliding window of
// observations, and emits only when ALL the gates pass:
//
//   1. The observability gate (assessObservability + parameterUncertainty) on the
//      assembled information matrix — never emit ill-conditioned params (CLAUDE.md rule 2,
//      Fraser's "precision != accuracy").
//   2. A 6-axis motion-excitation check on the rig pose trajectory across the window —
//      VINS-style. A planar / pure-translation / single-axis-rotation window is silently
//      degenerate: the optimization may still converge to a low cost, but extrinsic
//      components along the unexcited axes are unobservable in principle. We reject before
//      even looking at the result.
//
// On a successful emit we report the (drifted) extrinsics + a confidence score + a drift
// magnitude vs the reference calibration. On rejection we surface the failing
// gate + the names of the weakly-/un-observed parameters so the front-end can ask for more
// data of the right kind.
//
// Observation source is front-end-agnostic — RigView{object_points, image_points} is what
// detect/ already produces from a calibration target; the same struct also accommodates
// triangulated feature-track landmarks (object_points = world-frame landmark positions) for
// the targetless path (epic #24 follow-up).

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_rig.hpp"
#include "calibforge/calibrate_single.hpp"   // CameraFactory
#include "calibforge/observability.hpp"
#include "sophus/se3.hpp"

namespace calibforge {
namespace online {

struct ExtrinsicEmission {
  bool emitted = false;                          // false => DO NOT use the estimated extrinsics
  std::vector<Sophus::SE3d> extrinsics;          // size N-1, T_ck_c0 (valid only when emitted)
  double confidence = 0.0;                       // observability rcond [0, 1]
  double drift = 0.0;                            // sum_k ||Log(extr_new[k] * extr_ref[k]^-1)||
  std::vector<std::string> weak_parameters;      // weak/unobservable param names (esp. on refusal)
  bool refused_for_motion = false;               // true when the 6-axis excitation gate rejected
  std::vector<std::string> unexcited_axes;       // e.g. "rot_x", "trans_z" — populated on refusal
};

struct MotionExcitationOptions {
  double rot_range_threshold = 0.10;     // radians; minimum (max - min) of axis-angle per axis
  double trans_range_threshold = 0.10;   // metres;  minimum (max - min) of translation per axis
};

// Sliding-window online rig-extrinsic tracker. Intrinsics are baked in via the camera
// factories + reference intrinsics (held constant during re-estimation). The reference
// extrinsics anchor drift accounting.
class OnlineExtrinsicTracker {
 public:
  OnlineExtrinsicTracker(std::vector<CameraFactory> make_cameras,
                         std::vector<Eigen::VectorXd> intrinsics_reference,
                         std::vector<Sophus::SE3d> extrinsics_reference)
      : make_(std::move(make_cameras)),
        intr_ref_(std::move(intrinsics_reference)),
        extr_ref_(std::move(extrinsics_reference)) {}

  void addFrame(const RigView& view, const Sophus::SE3d& rig_pose_init) {
    views_.push_back(view);
    poses_.push_back(rig_pose_init);
  }
  std::size_t windowSize() const { return views_.size(); }
  void reset() { views_.clear(); poses_.clear(); }

  // Re-estimate the (N-1) inter-camera extrinsics over the window and gate. Emits ONLY when
  // the assembled information matrix is observable, its rcond clears min_confidence, AND the
  // rig pose trajectory has excitation in every translation + rotation axis above the
  // thresholds in motion_opts. Otherwise refuses with the failing gate annotated.
  ExtrinsicEmission tryEmit(double min_confidence,
                            const ObservabilityOptions& obs_opts = {},
                            const MotionExcitationOptions& motion_opts = {},
                            const LmOptions& lm_opts = LmOptions{}) const {
    ExtrinsicEmission e;
    if (views_.empty() || extr_ref_.empty()) return e;

    // Motion-excitation gate FIRST — refuses cheaply on a degenerate window without solving.
    const std::vector<std::string> unexcited = checkMotionExcitation(motion_opts);
    if (!unexcited.empty()) {
      e.refused_for_motion = true;
      e.unexcited_axes = unexcited;
      return e;
    }

    // Solve with intrinsics held constant.
    const RigResult res = calibrateRig(views_, intr_ref_, extr_ref_, poses_, make_, lm_opts,
                                       /*intrinsics_fixed=*/true);

    // The 6 * (N-1) extrinsic-tangent columns of the information matrix are what we gate on.
    // calibrateRig's information matrix is ordered [intr blocks (constant, zero-width here),
    // extrinsic SE3 tangents (6 each), pose SE3 tangents (6 each)]. With intrinsics fixed,
    // their tangents are width 0 → the info matrix starts directly with the extrinsics.
    const int ncam_m1 = static_cast<int>(extr_ref_.size());
    std::vector<std::string> extr_names;
    extr_names.reserve(static_cast<std::size_t>(6 * ncam_m1));
    for (int k = 0; k < ncam_m1; ++k) {
      const std::string base = "extr_" + std::to_string(k + 1);
      for (const char* a : {"_tx", "_ty", "_tz", "_rx", "_ry", "_rz"})
        extr_names.push_back(base + a);
    }
    // Gate on the extrinsic information with the per-view POSES MARGINALIZED OUT. calibrateRig
    // adds every per-view pose as a FREE parameter block coupled to the extrinsics (see
    // calibrate_rig.hpp), so the raw diagonal sub-block H_ee OVERSTATES the extrinsic
    // information — it MASKS the directions in which extrinsic error aliases into the (also-free)
    // pose error. The information about the extrinsics ALONE is the Schur complement
    //     S = H_ee - H_ep H_pp^{-1} H_pe.
    // Marginalizing EXPOSES those alias/null directions: when an extrinsic direction is
    // observable only through a pose, S collapses to rank-deficient there even though H_ee is
    // not. assessObservability gates on the diagonally-normalized reciprocal condition number
    // and clamps lambda_min >= 0, so a collapsed direction drives the score -> 0 and the gate
    // REFUSES — closing the precision!=accuracy false-ACCEPT that gating on the raw H_ee allows
    // (emitting drifted extrinsics that are not actually constrained by the data). Empirically
    // the well-excited case still clears the gate (~1.5e-3) while a coupled-pose alias that the
    // raw block rates fully observable is correctly driven to refusal. (The BEV surround path's
    // block is already self-contained — its rig poses are fixed inputs, not free parameters —
    // so it needs no Schur step.)
    const int extr_width = 6 * ncam_m1;
    const int n = static_cast<int>(res.information.rows());
    Eigen::MatrixXd Hex;
    if (n <= extr_width) {
      // No free pose block present (or malformed): the block is already the marginal
      // information. (n < extr_width should not happen; fall back to the full matrix.)
      Hex = res.information;
    } else {
      const int pose_width = n - extr_width;
      const Eigen::MatrixXd Hee = res.information.topLeftCorner(extr_width, extr_width);
      const Eigen::MatrixXd Hep = res.information.topRightCorner(extr_width, pose_width);
      const Eigen::MatrixXd Hpp = res.information.bottomRightCorner(pose_width, pose_width);
      // S = H_ee - H_ep H_pp^{-1} H_pe (H_pe = H_ep^T by symmetry of the information matrix).
      const Eigen::MatrixXd S = Hee - Hep * Hpp.ldlt().solve(Hep.transpose());
      // allFinite() only guards a non-finite solve (NaN/Inf). A merely rank-deficient H_pp (an
      // unobservable pose) still yields a finite S whose marginalized null direction collapses
      // the normalized rcond -> 0 in assessObservability (which clamps lambda_min >= 0), so the
      // gate REFUSES rather than silently passing — no explicit rank test on H_pp is needed.
      Hex = S.allFinite() ? S : res.information;
    }
    const ObservabilityReport rep = assessObservability(Hex, obs_opts);
    const ParameterUncertainty unc = parameterUncertainty(
        Hex, res.summary.final_cost, res.num_residuals, extr_names, obs_opts);

    e.confidence = rep.confidence;
    e.weak_parameters = unc.weak_parameters;
    if (rep.observable && rep.confidence >= min_confidence) {
      e.emitted = true;
      e.extrinsics = res.extrinsics;
      double drift = 0.0;
      for (int k = 0; k < ncam_m1; ++k)
        drift += (res.extrinsics[k] * extr_ref_[k].inverse()).log().norm();
      e.drift = drift;
    }
    return e;
  }

 private:
  // Reject the window if ANY of the 6 rig-pose-trajectory axes is below its excitation
  // threshold (translation range OR rotation axis-angle range). The "axis-angle range" is a
  // per-axis range computed on the SO3 log of each per-view pose's rotation — cheap and good
  // enough as a degenerate-motion gate (VINS / cam-IMU rule: 6-DoF excitation required).
  std::vector<std::string> checkMotionExcitation(const MotionExcitationOptions& opts) const {
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

  std::vector<CameraFactory> make_;
  std::vector<Eigen::VectorXd> intr_ref_;
  std::vector<Sophus::SE3d> extr_ref_;
  std::vector<RigView> views_;
  std::vector<Sophus::SE3d> poses_;
};

}  // namespace online
}  // namespace calibforge
