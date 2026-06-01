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
