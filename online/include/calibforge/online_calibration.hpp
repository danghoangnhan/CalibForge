#pragma once
//
// CalibForge online — gated online/targetless recalibration (the project's differentiator).
//
// RULE #2 (docs/RESEARCH.md Themes 2 & 3): NEVER silently emit calibration parameters online.
// This tracker accumulates per-frame observations, re-estimates over the window, and EMITS
// ONLY when the observability/confidence gate passes — combining the observability check
// (geometry / unobservable directions) WITH covariance/Fisher information, never reported
// sigma alone. Low cost does NOT mean a trustworthy estimate (precision != accuracy).
//
// The observation source is front-end-agnostic: the View{object_points, image_points} may
// come from a target (detect/) or, later, from targetless feature tracks / BEV photometric
// constraints. The gated loop here is the unbuilt-elsewhere core.

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"
#include "calibforge/observability.hpp"
#include "sophus/se3.hpp"

namespace calibforge {
namespace online {

struct Emission {
  bool emitted = false;                       // false => the gate refused (do NOT use the params)
  Eigen::VectorXd intrinsics;                 // valid only when emitted
  double confidence = 0.0;                    // observability reciprocal condition number [0,1]
  double drift = 0.0;                         // ||intrinsics - reference|| at the time of emission
  std::vector<std::string> weak_parameters;   // named weakly-/un-observed directions (esp. when refused)
};

// Online single-camera intrinsic recalibration (drift tracking) behind the gate.
class OnlineIntrinsicTracker {
 public:
  OnlineIntrinsicTracker(CameraFactory make_camera, Eigen::VectorXd intrinsics_reference,
                         std::vector<std::string> param_names = {})
      : make_(std::move(make_camera)),
        reference_(std::move(intrinsics_reference)),
        names_(std::move(param_names)) {}

  void addFrame(const View& view, const Sophus::SE3d& pose_init) {
    views_.push_back(view);
    poses_.push_back(pose_init);
  }
  std::size_t windowSize() const { return views_.size(); }
  void reset() {
    views_.clear();
    poses_.clear();
  }

  // Re-estimate over the window and gate. Emits the (drifted) intrinsics + confidence + drift
  // ONLY when the estimate is observable AND its confidence clears min_confidence; otherwise
  // refuses and reports the weak directions.
  Emission tryEmit(double min_confidence, const ObservabilityOptions& obs_opts = {},
                   const LmOptions& lm_opts = LmOptions{}) const {
    Emission e;
    if (views_.empty()) return e;

    const SingleCameraResult res =
        calibrateSingleCamera(views_, reference_, poses_, make_, lm_opts);
    const ObservabilityReport rep = assessObservability(res.information, obs_opts);
    const ParameterUncertainty unc = parameterUncertainty(
        res.information, res.summary.final_cost, res.num_residuals, names_, obs_opts);

    e.confidence = rep.confidence;
    e.weak_parameters = unc.weak_parameters;
    if (rep.observable && rep.confidence >= min_confidence) {
      e.emitted = true;
      e.intrinsics = res.intrinsics;
      e.drift = (res.intrinsics - reference_).norm();
    }
    return e;
  }

 private:
  CameraFactory make_;
  Eigen::VectorXd reference_;
  std::vector<std::string> names_;
  std::vector<View> views_;
  std::vector<Sophus::SE3d> poses_;
};

}  // namespace online
}  // namespace calibforge
