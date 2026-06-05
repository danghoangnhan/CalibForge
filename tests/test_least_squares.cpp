#include "calibforge/least_squares.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "cf_test.hpp"

using calibforge::LmSummary;
using calibforge::solveLevenbergMarquardt;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Classic nonlinear least squares: fit y = a*exp(b*x) to data generated from
// a_true=2.0, b_true=0.3. LM must recover the parameters from a poor initial guess.
CF_TEST(lm_recovers_exponential_model_params) {
  const double a_true = 2.0, b_true = 0.3;
  const std::vector<double> xs = {-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
  std::vector<double> ys;
  for (double x : xs) ys.push_back(a_true * std::exp(b_true * x));

  auto fn = [&](const VectorXd& p, VectorXd& r, MatrixXd& J) {
    const int m = static_cast<int>(xs.size());
    r.resize(m);
    J.resize(m, 2);
    const double a = p[0], b = p[1];
    for (int i = 0; i < m; ++i) {
      const double e = std::exp(b * xs[i]);
      r[i] = a * e - ys[i];
      J(i, 0) = e;               // dr/da
      J(i, 1) = a * xs[i] * e;   // dr/db
    }
  };

  VectorXd p(2);
  p << 1.0, 0.0;  // deliberately poor initial guess
  LmSummary s = solveLevenbergMarquardt(fn, p);

  CF_EXPECT_TRUE(s.converged);
  CF_EXPECT_NEAR(p[0], a_true, 1e-6);
  CF_EXPECT_NEAR(p[1], b_true, 1e-6);
  CF_EXPECT_TRUE(s.final_cost < 1e-12);
}

// Negative control for the scale-invariant `converged` predicate (the cross-arch determinism
// rework: relative-gradient stop + trust-region-collapse->converged). With the iteration budget
// capped far below what the poor initial guess needs, the solver must run out of budget and report
// converged==FALSE — it must NOT mislabel a mid-descent, non-stationary state as converged. Without
// this case the whole suite only ever asserts converged==TRUE, so a regression that declared
// success on a stalled/diverged solve would pass unnoticed.
CF_TEST(lm_reports_not_converged_when_iteration_budget_exhausted) {
  const double a_true = 2.0, b_true = 0.3;
  const std::vector<double> xs = {-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
  std::vector<double> ys;
  for (double x : xs) ys.push_back(a_true * std::exp(b_true * x));

  auto fn = [&](const VectorXd& p, VectorXd& r, MatrixXd& J) {
    const int m = static_cast<int>(xs.size());
    r.resize(m);
    J.resize(m, 2);
    const double a = p[0], b = p[1];
    for (int i = 0; i < m; ++i) {
      const double e = std::exp(b * xs[i]);
      r[i] = a * e - ys[i];
      J(i, 0) = e;
      J(i, 1) = a * xs[i] * e;
    }
  };

  VectorXd p(2);
  p << 1.0, 0.0;  // same poor guess as the recovery test
  calibforge::LmOptions opts;
  opts.max_iterations = 1;  // far too few to reach the minimum from this start
  const LmSummary s = solveLevenbergMarquardt(fn, p, opts);

  CF_EXPECT_TRUE(!s.converged);                   // ran out of budget mid-descent: NOT a minimum
  CF_EXPECT_TRUE(s.final_cost < s.initial_cost);  // but the step path did run and made progress
  CF_EXPECT_TRUE(s.iterations == 1);              // exactly the capped budget was spent
}
