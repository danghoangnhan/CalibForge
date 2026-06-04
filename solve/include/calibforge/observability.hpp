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
#include <limits>
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
  // margin: a healthy calibration scores well above this floor — geometry-dependent,
  // roughly ~2e-4 (well-excited UAV intrinsics) up to ~1e-2 (overlapped surround rig) in
  // CalibForge's own scenarios — versus ~0 for a degenerate one (a many-order gap).
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

// Per-parameter uncertainty derived from the information matrix at the solution.
// Distinct from assessObservability() on purpose (Fraser's "precision != accuracy"): the
// observability gate asks *can these params be separated at all*; this asks *given the
// noise, how uncertain is each one*. Unobservable directions get sigma = +inf and are
// named as weak — never reported as a small, trustworthy sigma.
struct ParameterUncertainty {
  std::vector<double> sigma;                 // per-parameter std dev (sqrt of cov diagonal)
  std::vector<std::string> weak_parameters;  // names with non-finite or above-threshold sigma
  double sigma0_sq = 0.0;                     // reduced chi-square noise estimate (2*cost)/(m-n)
};

// cov = sigma0^2 * H^{-1}, sigma0^2 = (2*final_cost)/(m - n)   [final_cost = 0.5*||r||^2].
// H^{-1} is formed from the eigendecomposition; directions with eigenvalue below the
// observability cutoff (lambda < min_reciprocal_condition * lambda_max) are treated as
// unobservable => sigma = +inf for every parameter they support. `names` labels the
// flagged parameters; `weak_sigma_threshold` additionally flags finite sigmas above it
// (default off / +inf).
inline ParameterUncertainty parameterUncertainty(
    const Eigen::MatrixXd& H, double final_cost, int m,
    const std::vector<std::string>& names = {},
    const ObservabilityOptions& opts = ObservabilityOptions{},
    double weak_sigma_threshold = std::numeric_limits<double>::infinity()) {
  ParameterUncertainty u;
  const int n = static_cast<int>(H.rows());
  if (n == 0) return u;
  const int dof = m - n;
  u.sigma0_sq = (dof > 0) ? (2.0 * final_cost) / dof
                          : std::numeric_limits<double>::infinity();

  // Decide unobservability on the DIAGONALLY-NORMALIZED H (H -> D^{-1/2} H D^{-1/2}, as in
  // assessObservability) so pure parameter-scale differences don't masquerade as null
  // directions; recover the actual covariance via H^{-1} = S * Hn^{-1} * S, S = D^{-1/2}.
  Eigen::VectorXd scale(n);
  for (int i = 0; i < n; ++i) {
    const double d = H(i, i);
    scale[i] = (d > 0.0) ? 1.0 / std::sqrt(d) : 0.0;  // 0 => param totally unconstrained
  }
  const Eigen::MatrixXd Hn = scale.asDiagonal() * H * scale.asDiagonal();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hn);
  const Eigen::VectorXd ev = es.eigenvalues();   // ascending
  const Eigen::MatrixXd V = es.eigenvectors();   // columns are eigenvectors
  const double lmax = ev(n - 1);
  const double cutoff = opts.min_reciprocal_condition * lmax;

  u.sigma.assign(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double var_n = 0.0;  // (Hn^{-1})_ii
    bool unobservable = (scale[i] == 0.0);
    for (int k = 0; k < n && !unobservable; ++k) {
      const double vki = V(i, k);
      if (ev(k) <= cutoff || ev(k) <= 0.0) {
        if (std::fabs(vki) > 1e-9) unobservable = true;  // supports a null direction
      } else {
        var_n += (vki * vki) / ev(k);
      }
    }
    u.sigma[i] = (unobservable || !std::isfinite(u.sigma0_sq))
                     ? std::numeric_limits<double>::infinity()
                     : std::sqrt(u.sigma0_sq * scale[i] * scale[i] * var_n);
  }
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(u.sigma[i]) || u.sigma[i] > weak_sigma_threshold) {
      u.weak_parameters.push_back(i < static_cast<int>(names.size())
                                      ? names[i]
                                      : ("param" + std::to_string(i)));
    }
  }
  return u;
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
