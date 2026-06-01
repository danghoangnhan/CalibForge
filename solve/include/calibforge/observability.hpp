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

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace calibforge {

struct ObservabilityReport {
  bool observable = false;                 // are all requested params observable given the data/motion?
  std::vector<std::string> unobservable;   // names of params/directions left unconstrained
  std::vector<std::string> degenerate_motions_detected;  // e.g. "planar", "pure-translation"
  double confidence = 0.0;                 // [0,1]: combined observability + covariance/FIM score
  double min_eigenvalue = 0.0;             // smallest eigenvalue of the normalized information matrix
};

struct ObservabilityOptions {
  // Threshold on the diagonally-normalized reciprocal condition number (lambda_min/
  // lambda_max in [0,1]). Below this, a parameter direction is effectively
  // unconstrained by the data and results must NOT be emitted. Default leaves a wide
  // margin: observed values are ~1e-6 for a healthy calibration vs ~0 for a degenerate
  // one (an ~11-order-of-magnitude gap).
  double min_reciprocal_condition = 1e-9;
};

// Assess whether parameters are observable from the information matrix H = J^T J at
// the solution. H is DIAGONALLY NORMALIZED first (H -> D^{-1/2} H D^{-1/2}, D=diag(H))
// so pure parameter-scale differences (focal ~500 vs angles ~1) do NOT masquerade as
// ill-conditioning; only genuine coupling/unobservability drives lambda_min -> 0.
// (docs/RESEARCH.md Themes 2 & 3: precision != accuracy; gate on observability + FIM.)
inline ObservabilityReport assessObservability(const Eigen::MatrixXd& H,
                                               const ObservabilityOptions& opts = ObservabilityOptions{}) {
  ObservabilityReport rep;
  const int n = static_cast<int>(H.rows());
  if (n == 0) return rep;

  Eigen::VectorXd scale(n);
  for (int i = 0; i < n; ++i) {
    const double d = H(i, i);
    scale[i] = (d > 0.0) ? 1.0 / std::sqrt(d) : 0.0;  // 0 diagonal => totally unconstrained param
  }
  const Eigen::MatrixXd Hn = scale.asDiagonal() * H * scale.asDiagonal();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hn);
  const Eigen::VectorXd ev = es.eigenvalues();  // ascending
  const double lmin = std::max(0.0, ev(0));  // clamp tiny-negative noise on singular H
  const double lmax = ev(n - 1);
  const double rcond = (lmax > 0.0) ? lmin / lmax : 0.0;

  rep.min_eigenvalue = lmin;
  rep.confidence = rcond;  // in [0,1]; 1 = perfectly conditioned after normalization
  rep.observable = rcond > opts.min_reciprocal_condition;
  return rep;
}

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
