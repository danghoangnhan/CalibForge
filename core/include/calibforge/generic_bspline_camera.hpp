#pragma once
//
// CalibForge core — Generic per-pixel B-spline camera model (Schöps et al., CVPR 2020).
// Header-only.
//
// "Why Having 10,000 Parameters in Your Camera Model is Better Than Twelve" — a dense ray
// field where each pixel maps to a 3D ray direction, smoothed by a CUBIC B-spline over a
// coarse control grid of Nx*Ny control points. Each control point stores an UNNORMALIZED
// 3-vector direction; unproject() bilinear-cubic-blends 16 control points (4x4 support of
// the B-spline at a pixel) and normalizes the result.
//
// Math reference only — re-implements Schöps' formulation; first-party puzzlepaint code is
// BSD-3 but we don't vendor it (CLAUDE.md rule 3, docs/DEPENDENCIES.md). Schöps's GPL-y
// transitive deps (OpenGV, SuiteSparse) are not vendored either.
//
// project() inverts unproject() via Gauss-Newton on a parametric (pinhole-style) initial
// guess. Jacobians are derived via the implicit function theorem on the unproject relation.
//
// Param layout (row-major, control[i*Ny+j] stores (dx, dy, dz)):
//   index 0..2     -> control[0,0]   direction
//   index 3..5     -> control[0,1]   direction
//   ...
//   total numParams() = 3 * Nx * Ny
//
// The control grid maps to the image domain by:
//   pixel (u, v) -> grid coord (a, b) = ((u - margin_u) / step_x, (v - margin_v) / step_y)
// with grid index (i, j) = (floor(a), floor(b)) and fractional (s, t) = (a - i, b - j).
// The B-spline support window is i-1..i+2 x j-1..j+2 (4x4).
//
// NUMERICAL CAVEAT: the dense Jacobian wrt params is 2 x (3*Nx*Ny) per residual. Practical
// grid sizes are ~12x8 to ~24x16 (288–1152 params); very large grids motivate the GPU
// sparse solver back-ends planned for v1.0 (#25).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"

namespace calibforge {

struct GenericBSplineGrid {
  int nx = 12;          // control points across u
  int ny = 8;           // control points across v
  int image_w = 640;
  int image_h = 480;
  // Margin (in pixels) of the FIRST/LAST control point relative to the image border.
  // Negative margin extends the grid OUTSIDE the image so the cubic B-spline 4x4 support
  // window covers the image corners cleanly. NOTE: the default 0.0 places the outer control
  // points ON the image border, so the corners are NOT fully inside the support window;
  // pixels there fall back to clamped boundary extrapolation (see rayAtPixel). Set
  // margin = -step/2 (Schöps' setup) for clean corner coverage.
  double margin = 0.0;
};

namespace bspline_detail {

// 1D cubic B-spline weights for fractional offset t in [0, 1). Returned weights are for
// control indices [base-1, base, base+1, base+2] where base is the integer part.
// b_{-1}(t) = (1-t)^3 / 6
// b_0  (t) = (3t^3 - 6t^2 + 4) / 6
// b_{+1}(t) = (-3t^3 + 3t^2 + 3t + 1) / 6
// b_{+2}(t) = t^3 / 6
inline std::array<double, 4> cubicBSplineWeights(double t) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  return {{(1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0,
           (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
           (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
           t3 / 6.0}};
}

inline std::array<double, 4> cubicBSplineWeightsDeriv(double t) {
  // d/dt of the four weights above.
  // b_{-1}': -3(1-t)^2/6 = -(1-t)^2/2
  // b_0'   : (9t^2 - 12t)/6 = (3t^2 - 4t)/2
  // b_{+1}': (-9t^2 + 6t + 3)/6 = (-3t^2 + 2t + 1)/2
  // b_{+2}': 3t^2/6 = t^2/2
  return {{-(1.0 - t) * (1.0 - t) / 2.0,
           (3.0 * t * t - 4.0 * t) / 2.0,
           (-3.0 * t * t + 2.0 * t + 1.0) / 2.0,
           t * t / 2.0}};
}

inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

}  // namespace bspline_detail

class GenericBSplineCamera : public CameraModel {
 public:
  // Construct with an empty (zero-direction) grid. Use fromParametricCamera or directly
  // setParam to populate the control points.
  explicit GenericBSplineCamera(const GenericBSplineGrid& grid)
      : grid_(grid),
        params_(static_cast<std::size_t>(3 * grid.nx * grid.ny), 0.0) {
    if (grid.nx < 4 || grid.ny < 4) {
      throw std::invalid_argument(
          "GenericBSplineCamera grid must be at least 4x4 (cubic B-spline support window)");
    }
    // Default to identity ray field: every control point = (0, 0, 1), so unproject() returns
    // (0, 0, 1) everywhere. With a constant field, project()'s Gauss-Newton Jacobian is
    // singular and cannot refine, so off-axis directions fail the convergence gate and project()
    // returns NaN (not the initial guess); populate the grid via fitFromParametricCamera()
    // before relying on project().
    for (int j = 0; j < grid.ny; ++j)
      for (int i = 0; i < grid.nx; ++i) params_[static_cast<std::size_t>(3 * (i * grid.ny + j)) + 2] = 1.0;
    step_x_ = (grid.image_w - 1.0 - 2.0 * grid.margin) / (grid.nx - 1);
    step_y_ = (grid.image_h - 1.0 - 2.0 * grid.margin) / (grid.ny - 1);
    if (step_x_ <= 0.0 || step_y_ <= 0.0)
      throw std::invalid_argument("GenericBSplineCamera grid: step must be positive");
  }

  GenericBSplineGrid grid() const { return grid_; }
  const std::vector<double>& params() const { return params_; }
  void setParams(const std::vector<double>& p) {
    if (p.size() != params_.size())
      throw std::invalid_argument("GenericBSplineCamera setParams: size mismatch");
    params_ = p;
  }

  // Fit a generic grid to a parametric camera by sampling its unproject at each control grid
  // position and storing that ray directly as the control point. NOTE: a uniform cubic
  // B-spline is APPROXIMATING, not interpolating — the evaluated field at control (i, j) is
  // the 1/6, 4/6, 1/6 blend of (i, j) with its neighbours, so the fitted model approximates
  // the source EVERYWHERE (it does not reproduce it exactly even at the control points). This
  // is fine as an initial guess for online generic-model recalibration; solve the de Boor
  // interpolation system instead if exact control-point reproduction is ever required.
  void fitFromParametricCamera(const CameraModel& source) {
    for (int j = 0; j < grid_.ny; ++j) {
      const double v = grid_.margin + j * step_y_;
      for (int i = 0; i < grid_.nx; ++i) {
        const double u = grid_.margin + i * step_x_;
        const Vec3 r = source.unproject(Vec2{u, v});
        const std::size_t off = static_cast<std::size_t>(3 * (i * grid_.ny + j));
        params_[off + 0] = r[0];
        params_[off + 1] = r[1];
        params_[off + 2] = r[2];
      }
    }
  }

  // CameraModel interface ----------------------------------------------------------------

  Vec3 unproject(const Vec2& px) const override {
    Vec3 r;
    Eigen::Matrix<double, 3, 2> dr_duv_unused;
    rayAtPixel(px[0], px[1], r, /*want_duv=*/false, dr_duv_unused);
    // Normalize.
    const double n = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (n < 1e-12) return Vec3{0.0, 0.0, 1.0};
    return Vec3{r[0] / n, r[1] / n, r[2] / n};
  }

  Vec2 project(const Vec3& point_cam) const override {
    const auto kNaN = std::numeric_limits<double>::quiet_NaN();
    // Target unit direction.
    const double n = std::sqrt(point_cam[0] * point_cam[0] + point_cam[1] * point_cam[1]
                              + point_cam[2] * point_cam[2]);
    if (n < 1e-12) return Vec2{kNaN, kNaN};  // a zero vector has no direction => not projectable
    const Vec3 d_target{point_cam[0] / n, point_cam[1] / n, point_cam[2] / n};
    // A forward-facing generic ray field cannot project a point that is not in front of it;
    // signal failure instead of clamping z and returning a diverged guess.
    if (d_target[2] <= 0.0) return Vec2{kNaN, kNaN};

    // Initial guess: pinhole-style projection using the image-center control point as the
    // approximate principal point and step_*/pi (radians-per-pixel-ish) as a coarse scale.
    // The Gauss-Newton iteration below corrects for the generic distortion.
    double u = grid_.image_w * 0.5 + (d_target[0] / d_target[2])
                                     * std::max(step_x_ * (grid_.nx - 1) * 0.5, 1.0);
    double v = grid_.image_h * 0.5 + (d_target[1] / d_target[2])
                                     * std::max(step_y_ * (grid_.ny - 1) * 0.5, 1.0);

    // Gauss-Newton on the unit-direction residual r = direction(u, v) - d_target.
    // The Jacobian J_uv = d(direction)/d(uv) (3x2). Step duv = -(J^T J)^-1 J^T r.
    double resid_norm = std::numeric_limits<double>::infinity();
    for (int it = 0; it < 30; ++it) {
      Vec3 r_raw;
      Eigen::Matrix<double, 3, 2> dr_duv;
      rayAtPixel(u, v, r_raw, /*want_duv=*/true, dr_duv);
      const double rn = std::sqrt(r_raw[0] * r_raw[0] + r_raw[1] * r_raw[1] + r_raw[2] * r_raw[2]);
      if (rn < 1e-12) break;
      const Eigen::Vector3d r_unit(r_raw[0] / rn, r_raw[1] / rn, r_raw[2] / rn);
      // Project the un-normalized Jacobian through d(r/||r||)/d(r): (I - r_unit r_unit^T) / ||r||.
      const Eigen::Matrix3d P =
          (Eigen::Matrix3d::Identity() - r_unit * r_unit.transpose()) / rn;
      const Eigen::Matrix<double, 3, 2> Junit = P * dr_duv;
      const Eigen::Vector3d resid =
          r_unit - Eigen::Vector3d(d_target[0], d_target[1], d_target[2]);
      resid_norm = resid.norm();
      const Eigen::Matrix2d JtJ = Junit.transpose() * Junit;
      const Eigen::Vector2d g = Junit.transpose() * resid;
      const Eigen::Vector2d step = JtJ.ldlt().solve(g);
      u -= step[0];
      v -= step[1];
      if (step.norm() < 1e-7) break;
    }

    // Convergence + sanity gate: a non-converged solve, or one that lands far outside the
    // image, is not a valid projection — return NaN rather than diverged finite garbage so
    // callers can detect the failure (the parametric models leave this to the caller, but the
    // generic model's GN can diverge to ~1e60, which is silently corrupting).
    const double pad_w = 2.0 * grid_.image_w, pad_h = 2.0 * grid_.image_h;
    if (!(resid_norm < 1e-3) || u < -pad_w || u > grid_.image_w + pad_w ||
        v < -pad_h || v > grid_.image_h + pad_h) {
      return Vec2{kNaN, kNaN};
    }
    return Vec2{u, v};
  }

  Jacobian projectJacobianWrtPoint(const Vec3& point_cam) const override {
    // Implicit function theorem: at the solution (u*, v*), direction(u*, v*) = d_target.
    // d direction/d uv * d uv/d xyz = d d_target/d xyz
    // -> d uv/d xyz = (J_dir_uv)^-1 * J_dir_xyz   (using pseudoinverse for 3->2).
    const Vec2 uv = project(point_cam);
    if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) {
      // project() rejected this point (behind camera / diverged / out of image): there is no
      // valid local linearization at a non-existent projection. Return a NaN Jacobian so a
      // BA/online solver DETECTS the failure (e.g. drops the residual) instead of folding
      // finite garbage into the normal equations — silently corrupting emitted params
      // (CLAUDE.md rule 2). NB: floor(NaN)->INT_MIN and min(1,NaN)->1 would otherwise launder
      // the NaN into a finite-but-meaningless block below.
      Jacobian Jn;
      Jn.rows = 2;
      Jn.cols = 3;
      Jn.data.assign(6, std::numeric_limits<double>::quiet_NaN());
      return Jn;
    }
    const double n = std::sqrt(point_cam[0] * point_cam[0] + point_cam[1] * point_cam[1]
                              + point_cam[2] * point_cam[2]);
    Vec3 r_raw;
    Eigen::Matrix<double, 3, 2> dr_duv;
    rayAtPixel(uv[0], uv[1], r_raw, /*want_duv=*/true, dr_duv);
    const double rn = std::sqrt(r_raw[0] * r_raw[0] + r_raw[1] * r_raw[1] + r_raw[2] * r_raw[2]);
    const Eigen::Vector3d r_unit(r_raw[0] / rn, r_raw[1] / rn, r_raw[2] / rn);
    const Eigen::Matrix3d P =
        (Eigen::Matrix3d::Identity() - r_unit * r_unit.transpose()) / rn;
    const Eigen::Matrix<double, 3, 2> Junit = P * dr_duv;

    // d d_target/d xyz = (I - d_target d_target^T) / ||point|| where d_target = point / ||point||.
    const Eigen::Vector3d dt(point_cam[0] / n, point_cam[1] / n, point_cam[2] / n);
    const Eigen::Matrix3d Jdir_xyz = (Eigen::Matrix3d::Identity() - dt * dt.transpose()) / n;

    const Eigen::Matrix2d JtJ = Junit.transpose() * Junit;
    const Eigen::Matrix<double, 2, 3> A = JtJ.ldlt().solve(Junit.transpose() * Jdir_xyz);

    Jacobian J;
    J.rows = 2;
    J.cols = 3;
    J.data.assign(6, 0.0);
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 3; ++c) J.data[static_cast<std::size_t>(r) * 3 + c] = A(r, c);
    return J;
  }

  Jacobian projectJacobianWrtParams(const Vec3& point_cam) const override {
    // d uv/d params via implicit function theorem on direction(u, v) - d_target = 0.
    // d direction/d uv * d uv/d params + d direction/d params = 0
    // d uv/d params = -(d direction/d uv)^pseudo-inv * d direction/d params.
    //
    // The B-spline support is 16 control points around (u, v); only those 16 contribute
    // 3 nonzero entries to d direction/d params each. The 2x(3*Nx*Ny) Jacobian is therefore
    // very sparse in absolute terms; we still emit the dense form (the dense solver path).
    Jacobian J;
    J.rows = 2;
    J.cols = static_cast<std::size_t>(3 * grid_.nx * grid_.ny);
    J.data.assign(static_cast<std::size_t>(2) * J.cols, 0.0);

    const Vec2 uv = project(point_cam);
    if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) {
      // See projectJacobianWrtPoint: a non-projectable point has no valid linearization;
      // emit a detectable NaN block rather than laundering NaN through floor()/min() into a
      // finite-garbage param Jacobian (CLAUDE.md rule 2).
      J.data.assign(J.data.size(), std::numeric_limits<double>::quiet_NaN());
      return J;
    }
    const double a = (uv[0] - grid_.margin) / step_x_;
    const double b = (uv[1] - grid_.margin) / step_y_;
    const int i0 = bspline_detail::clampi(static_cast<int>(std::floor(a)), 1, grid_.nx - 3);
    const int j0 = bspline_detail::clampi(static_cast<int>(std::floor(b)), 1, grid_.ny - 3);
    // Clamp the fractional coord into [0,1] so out-of-domain pixels give bounded constant
    // boundary extrapolation rather than evaluating the cubics far outside their valid range
    // (where the weights explode to ~1e5 and produce garbage). In-domain pixels are unaffected.
    const double s = std::max(0.0, std::min(1.0, a - i0));
    const double t = std::max(0.0, std::min(1.0, b - j0));
    const std::array<double, 4> wi = bspline_detail::cubicBSplineWeights(s);
    const std::array<double, 4> wj = bspline_detail::cubicBSplineWeights(t);

    // dr_raw/d uv at (u, v).
    Vec3 r_raw;
    Eigen::Matrix<double, 3, 2> dr_duv;
    rayAtPixel(uv[0], uv[1], r_raw, /*want_duv=*/true, dr_duv);
    const double rn = std::sqrt(r_raw[0] * r_raw[0] + r_raw[1] * r_raw[1] + r_raw[2] * r_raw[2]);
    const Eigen::Vector3d r_unit(r_raw[0] / rn, r_raw[1] / rn, r_raw[2] / rn);
    const Eigen::Matrix3d P =
        (Eigen::Matrix3d::Identity() - r_unit * r_unit.transpose()) / rn;
    const Eigen::Matrix<double, 3, 2> Junit = P * dr_duv;
    const Eigen::Matrix2d JtJ = Junit.transpose() * Junit;

    // For each of the 16 supporting control points, d(r_raw)/d(control[3]) = w_ij * I_3,
    // and d(r_unit)/d(control[3]) = P * (w_ij * I_3) = w_ij * P.
    // d uv/d control = -JtJ^-1 * Junit^T * (w_ij * P).
    for (int dj = -1; dj <= 2; ++dj) {
      for (int di = -1; di <= 2; ++di) {
        const int ii = i0 + di;
        const int jj = j0 + dj;
        if (ii < 0 || ii >= grid_.nx || jj < 0 || jj >= grid_.ny) continue;
        const double w = wi[di + 1] * wj[dj + 1];
        const Eigen::Matrix<double, 2, 3> M =
            -JtJ.ldlt().solve(Junit.transpose() * P) * w;
        const std::size_t col_base = static_cast<std::size_t>(3 * (ii * grid_.ny + jj));
        for (int rr = 0; rr < 2; ++rr) {
          for (int cc = 0; cc < 3; ++cc) {
            J.data[static_cast<std::size_t>(rr) * J.cols + col_base + cc] = M(rr, cc);
          }
        }
      }
    }
    return J;
  }

  std::size_t numParams() const override { return params_.size(); }
  std::string name() const override { return "generic_bspline"; }

 private:
  // Evaluate the un-normalized ray field at pixel (u, v). When want_duv is true the
  // 3x2 Jacobian d(ray_raw)/d(u, v) is also filled.
  void rayAtPixel(double u, double v, Vec3& r_raw, bool want_duv,
                  Eigen::Matrix<double, 3, 2>& dr_duv) const {
    const double a = (u - grid_.margin) / step_x_;
    const double b = (v - grid_.margin) / step_y_;
    const int i0 = bspline_detail::clampi(static_cast<int>(std::floor(a)), 1, grid_.nx - 3);
    const int j0 = bspline_detail::clampi(static_cast<int>(std::floor(b)), 1, grid_.ny - 3);
    // Clamp the fractional coord into [0,1] for out-of-domain pixels (constant boundary
    // extrapolation) — see projectJacobianWrtParams. In-domain pixels already have s,t in [0,1).
    const double s = std::max(0.0, std::min(1.0, a - i0));
    const double t = std::max(0.0, std::min(1.0, b - j0));
    const std::array<double, 4> wi = bspline_detail::cubicBSplineWeights(s);
    const std::array<double, 4> wj = bspline_detail::cubicBSplineWeights(t);
    const std::array<double, 4> dwi = want_duv ? bspline_detail::cubicBSplineWeightsDeriv(s)
                                               : std::array<double, 4>{};
    const std::array<double, 4> dwj = want_duv ? bspline_detail::cubicBSplineWeightsDeriv(t)
                                               : std::array<double, 4>{};

    double rx = 0.0, ry = 0.0, rz = 0.0;
    double dr_du_x = 0.0, dr_du_y = 0.0, dr_du_z = 0.0;
    double dr_dv_x = 0.0, dr_dv_y = 0.0, dr_dv_z = 0.0;
    for (int dj = -1; dj <= 2; ++dj) {
      for (int di = -1; di <= 2; ++di) {
        const int ii = bspline_detail::clampi(i0 + di, 0, grid_.nx - 1);
        const int jj = bspline_detail::clampi(j0 + dj, 0, grid_.ny - 1);
        const std::size_t off = static_cast<std::size_t>(3 * (ii * grid_.ny + jj));
        const double cx = params_[off + 0];
        const double cy = params_[off + 1];
        const double cz = params_[off + 2];
        const double w = wi[di + 1] * wj[dj + 1];
        rx += w * cx;
        ry += w * cy;
        rz += w * cz;
        if (want_duv) {
          // d w/d u = dwi * wj / step_x_ ; d w/d v = wi * dwj / step_y_.
          const double dw_du = dwi[di + 1] * wj[dj + 1] / step_x_;
          const double dw_dv = wi[di + 1] * dwj[dj + 1] / step_y_;
          dr_du_x += dw_du * cx;  dr_du_y += dw_du * cy;  dr_du_z += dw_du * cz;
          dr_dv_x += dw_dv * cx;  dr_dv_y += dw_dv * cy;  dr_dv_z += dw_dv * cz;
        }
      }
    }
    r_raw = Vec3{rx, ry, rz};
    if (want_duv) {
      dr_duv(0, 0) = dr_du_x;  dr_duv(0, 1) = dr_dv_x;
      dr_duv(1, 0) = dr_du_y;  dr_duv(1, 1) = dr_dv_y;
      dr_duv(2, 0) = dr_du_z;  dr_duv(2, 1) = dr_dv_z;
    }
  }

  GenericBSplineGrid grid_;
  std::vector<double> params_;
  double step_x_ = 0.0;
  double step_y_ = 0.0;
};

}  // namespace calibforge
