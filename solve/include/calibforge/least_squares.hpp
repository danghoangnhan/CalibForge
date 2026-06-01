#pragma once
//
// CalibForge solve — dense Levenberg-Marquardt for small nonlinear least squares
// (header-only). This is the CPU path that, per docs/RESEARCH.md Theme 1, is the
// default for single small calibrations (where a CPU solver beats a batched GPU one).
//
// Minimizes 0.5 * ||r(params)||^2 with Marquardt diagonal damping; the reduced
// (camera) system / GPU back-ends plug in behind the same idea later.

#include <algorithm>
#include <cmath>
#include <functional>

#include <Eigen/Dense>

namespace calibforge {

struct LmOptions {
  int max_iterations = 100;
  double function_tolerance = 1e-12;   // stop when relative cost decrease < this
  double gradient_tolerance = 1e-12;   // stop when ||J^T r|| < this
  double parameter_tolerance = 1e-10;  // stop when ||dx|| < this*(||x||+this)
  double initial_lambda = 1e-3;
};

struct LmSummary {
  bool converged = false;
  int iterations = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
};

// Callback: given params (n), fill residual r (m) and Jacobian J (m x n).
using ResidualFn =
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&, Eigen::MatrixXd&)>;

inline LmSummary solveLevenbergMarquardt(const ResidualFn& fn,
                                         Eigen::VectorXd& params,
                                         const LmOptions& opts = LmOptions{}) {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  LmSummary s;
  VectorXd r;
  MatrixXd J;
  fn(params, r, J);
  double cost = 0.5 * r.squaredNorm();
  s.initial_cost = cost;

  double lambda = opts.initial_lambda;
  int it = 0;
  for (; it < opts.max_iterations; ++it) {
    const MatrixXd JtJ = J.transpose() * J;
    const VectorXd g = J.transpose() * r;  // gradient
    if (g.norm() < opts.gradient_tolerance) {
      s.converged = true;
      break;
    }

    bool step_accepted = false;
    for (int tries = 0; tries < 12; ++tries) {
      MatrixXd A = JtJ;
      A.diagonal() += lambda * JtJ.diagonal();  // Marquardt (scale-invariant) damping
      const VectorXd dx = A.ldlt().solve(-g);

      const VectorXd trial = params + dx;
      VectorXd r_t;
      MatrixXd J_t;
      fn(trial, r_t, J_t);
      const double cost_t = 0.5 * r_t.squaredNorm();

      if (cost_t < cost) {
        const double rel = (cost - cost_t) / std::max(cost, 1e-300);
        const double step_norm = dx.norm();
        params = trial;
        r = r_t;
        J = J_t;
        cost = cost_t;
        lambda = std::max(lambda * 0.3, 1e-12);
        step_accepted = true;
        if (rel < opts.function_tolerance) s.converged = true;
        if (step_norm < opts.parameter_tolerance * (params.norm() + opts.parameter_tolerance))
          s.converged = true;  // settled: step is negligible relative to scale
        break;
      }
      lambda *= 3.0;  // reject: damp harder and retry
    }

    if (s.converged) {
      ++it;
      break;
    }
    if (!step_accepted) break;  // could not decrease cost even with heavy damping
  }

  s.iterations = it;
  s.final_cost = cost;
  return s;
}

}  // namespace calibforge
