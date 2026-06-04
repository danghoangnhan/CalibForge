// Tests for core/generic_bspline_camera.hpp — the Schöps CVPR 2020 generic per-pixel ray
// field model. We verify:
//   1. unproject + project roundtrip (pixel -> ray -> pixel) is accurate when the model is
//      fitted from a parametric pinhole.
//   2. The fitted model agrees with the pinhole at the control points (which is how it was
//      constructed) and approximates it between them.
//   3. project's Jacobian wrt pixel space is consistent with finite differences via the
//      implicit-function-theorem chain.

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/generic_bspline_camera.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"

using calibforge::GenericBSplineCamera;
using calibforge::GenericBSplineGrid;
using calibforge::Jacobian;
using calibforge::PinholeCamera;
using calibforge::Vec2;
using calibforge::Vec3;

CF_TEST(generic_bspline_unproject_project_roundtrip_after_pinhole_fit) {
  PinholeCamera pin(500.0, 500.0, 320.0, 240.0);
  GenericBSplineGrid grid;
  grid.nx = 16; grid.ny = 12;
  grid.image_w = 640; grid.image_h = 480;
  grid.margin = -30.0;  // extend past the image so the cubic support window covers borders

  GenericBSplineCamera g(grid);
  g.fitFromParametricCamera(pin);

  // Test pixels well inside the image (not on the very corner) — the fitted model
  // closely matches pinhole there.
  std::vector<Vec2> pxs = {{320.0, 240.0}, {200.0, 150.0}, {420.0, 320.0},
                           {280.0, 200.0}, {340.0, 260.0}};
  for (const Vec2& px : pxs) {
    const Vec3 ray = g.unproject(px);
    // Re-project onto a sphere — scale ray by an arbitrary positive depth.
    const Vec3 X{ray[0] * 3.0, ray[1] * 3.0, ray[2] * 3.0};
    const Vec2 px2 = g.project(X);
    CF_EXPECT_NEAR(px[0], px2[0], 0.5);
    CF_EXPECT_NEAR(px[1], px2[1], 0.5);
  }
}

CF_TEST(generic_bspline_project_jacobian_wrt_point_matches_finite_difference) {
  PinholeCamera pin(500.0, 500.0, 320.0, 240.0);
  GenericBSplineGrid grid;
  grid.nx = 16; grid.ny = 12;
  grid.image_w = 640; grid.image_h = 480;
  grid.margin = -30.0;
  GenericBSplineCamera g(grid);
  g.fitFromParametricCamera(pin);

  const Vec3 X{0.4, -0.2, 3.0};
  const Jacobian J = g.projectJacobianWrtPoint(X);
  CF_EXPECT_TRUE(J.rows == 2 && J.cols == 3);

  const double h = 1e-6;
  for (int c = 0; c < 3; ++c) {
    Vec3 Xp = X, Xm = X;
    Xp[c] += h;
    Xm[c] -= h;
    const Vec2 pp = g.project(Xp);
    const Vec2 pm = g.project(Xm);
    const double dpu = (pp[0] - pm[0]) / (2.0 * h);
    const double dpv = (pp[1] - pm[1]) / (2.0 * h);
    // The generic model's Jacobian is derived analytically via the implicit function
    // theorem; FD agreement to a few percent is the practical bar (Gauss-Newton in
    // project() is only converged to ~1e-7 pixel tolerance).
    CF_EXPECT_NEAR(J.data[0 * 3 + c], dpu, std::max(0.1, std::fabs(dpu) * 0.05));
    CF_EXPECT_NEAR(J.data[1 * 3 + c], dpv, std::max(0.1, std::fabs(dpv) * 0.05));
  }
}

CF_TEST(generic_bspline_params_jacobian_has_sparse_support_pattern) {
  PinholeCamera pin(500.0, 500.0, 320.0, 240.0);
  GenericBSplineGrid grid;
  grid.nx = 16; grid.ny = 12;
  grid.image_w = 640; grid.image_h = 480;
  grid.margin = -30.0;
  GenericBSplineCamera g(grid);
  g.fitFromParametricCamera(pin);

  const Vec3 X{0.2, -0.1, 3.0};
  const Jacobian J = g.projectJacobianWrtParams(X);
  CF_EXPECT_TRUE(J.cols == static_cast<std::size_t>(3 * grid.nx * grid.ny));
  // At most 16 control points (4x4 cubic B-spline support) contribute, each with 3 nonzero
  // direction columns; the rest of the columns are exactly zero.
  int nonzero_cols = 0;
  for (std::size_t c = 0; c < J.cols; ++c) {
    const double m = std::max(std::fabs(J.data[c]), std::fabs(J.data[J.cols + c]));
    if (m > 1e-12) ++nonzero_cols;
  }
  CF_EXPECT_TRUE(nonzero_cols <= 16 * 3);
  CF_EXPECT_TRUE(nonzero_cols >= 4 * 3);  // at the boundary some support is clipped
}

CF_TEST(generic_bspline_params_jacobian_values_match_finite_difference) {
  // The sparsity test above is satisfied by a sign flip, a wrong scale, a dropped
  // normalization projector, or a transposed block — none of which it can detect. Verify the
  // actual VALUES of projectJacobianWrtParams against central differences over every param.
  PinholeCamera pin(500.0, 500.0, 320.0, 240.0);
  GenericBSplineGrid grid;
  grid.nx = 16; grid.ny = 12;
  grid.image_w = 640; grid.image_h = 480;
  grid.margin = -30.0;
  GenericBSplineCamera g(grid);
  g.fitFromParametricCamera(pin);

  const Vec3 X{0.1, -0.05, 3.0};  // projects comfortably inside the image
  const Jacobian J = g.projectJacobianWrtParams(X);
  const std::vector<double> p0 = g.params();
  const double h = 1e-4;
  for (std::size_t c = 0; c < p0.size(); ++c) {
    std::vector<double> pp = p0, pm = p0;
    pp[c] += h;
    pm[c] -= h;
    g.setParams(pp);
    const Vec2 uvp = g.project(X);
    g.setParams(pm);
    const Vec2 uvm = g.project(X);
    g.setParams(p0);
    const double fdu = (uvp[0] - uvm[0]) / (2.0 * h);
    const double fdv = (uvp[1] - uvm[1]) / (2.0 * h);
    // ~5e-3 absolute floor absorbs the ~1e-7 px Gauss-Newton noise amplified by 1/(2h);
    // 3% relative catches sign/scale/projector errors on the supporting columns.
    CF_EXPECT_NEAR(J.data[0 * J.cols + c], fdu, std::max(5e-3, std::fabs(fdu) * 0.03));
    CF_EXPECT_NEAR(J.data[1 * J.cols + c], fdv, std::max(5e-3, std::fabs(fdv) * 0.03));
  }
}

CF_TEST(generic_bspline_fit_approximates_source_pinhole_at_interior_pixels) {
  // Roundtrip (project∘unproject) only proves the model is self-consistent; it says nothing
  // about FIT FIDELITY. Check the fitted ray field actually tracks the source pinhole's rays
  // at interior pixels (a cubic B-spline approximates, so a small angular error is expected).
  PinholeCamera pin(500.0, 500.0, 320.0, 240.0);
  GenericBSplineGrid grid;
  grid.nx = 16; grid.ny = 12;
  grid.image_w = 640; grid.image_h = 480;
  grid.margin = -30.0;
  GenericBSplineCamera g(grid);
  g.fitFromParametricCamera(pin);

  std::vector<Vec2> pxs = {{320.0, 240.0}, {200.0, 150.0}, {420.0, 320.0},
                           {280.0, 200.0}, {500.0, 360.0}};
  for (const Vec2& px : pxs) {
    const Vec3 rg = g.unproject(px);  // already unit length
    const Vec3 rp_raw = pin.unproject(px);
    const double np = std::sqrt(rp_raw[0] * rp_raw[0] + rp_raw[1] * rp_raw[1]
                              + rp_raw[2] * rp_raw[2]);
    const double dot = (rg[0] * rp_raw[0] + rg[1] * rp_raw[1] + rg[2] * rp_raw[2]) / np;
    const double angle = std::acos(std::max(-1.0, std::min(1.0, dot)));
    CF_EXPECT_TRUE(angle < 0.01);  // < ~0.57 deg between fitted and source rays
  }
}
