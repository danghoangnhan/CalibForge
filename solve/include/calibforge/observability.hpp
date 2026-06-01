#pragma once
//
// CalibForge solve — the observability / confidence gate (INTERFACE STUB).
//
// ⚠️ RULE #2 (docs/RESEARCH.md Themes 2 & 3): NEVER silently emit calibration
// parameters online. This gate is CalibForge's differentiator — no surveyed
// method combines online recalibration + covariance/Fisher-information confidence
// scoring + drift tracking.
//
// Two verified principles it must encode:
//   * VINS observability: a fully-calibrated system has exactly 4 unobservable
//     directions (global yaw + 3D translation); all calibration params become
//     observable only under fully-excited 6-axis motion; specific degenerate
//     motions (planar / pure-translation / constant-velocity) disable specific
//     params and COMPOUND.
//   * Fraser's caution: precision != accuracy. Low reported sigma does NOT mean
//     trustworthy — covariance is realistic only after datum/config defects are
//     removed, and object-space bias can exceed covariance-indicated uncertainty.
// => Combine observability/geometry checks WITH covariance/FIM. Never sigma alone.

#include <string>
#include <vector>

namespace calibforge {

struct ObservabilityReport {
  bool observable = false;                 // are all requested params observable given the data/motion?
  std::vector<std::string> unobservable;   // names of params/directions left unconstrained
  std::vector<std::string> degenerate_motions_detected;  // e.g. "planar", "pure-translation"
  double confidence = 0.0;                 // [0,1]: combined observability + covariance/FIM score
  double min_eigenvalue = 0.0;             // of the (gauge-reduced) information matrix
};

// Gate applied before emitting online/targetless calibration results.
class ObservabilityGate {
 public:
  virtual ~ObservabilityGate() = default;

  // Analyze the just-solved problem's information/covariance + motion profile.
  virtual ObservabilityReport analyze() const = 0;

  // Convenience predicates over analyze().
  virtual bool isObservable() const = 0;
  virtual double confidenceScore() const = 0;

  // Hard guard: returns false (and the caller MUST NOT emit params) when the
  // estimate is ill-conditioned below the configured confidence threshold.
  virtual bool emitAllowed(double min_confidence) const = 0;
};

}  // namespace calibforge
