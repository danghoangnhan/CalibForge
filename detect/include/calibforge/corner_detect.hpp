#pragma once
//
// CalibForge detect — checkerboard saddle-corner detection + sub-pixel refinement
// (header-only, CPU).
//
// Checkerboard inner corners are intensity SADDLES. We score each pixel with
// -det(Hessian) = Ixy^2 - Ixx*Iyy (large where curvature has opposite signs), take
// non-max-suppressed peaks, then refine each to sub-pixel by fitting a local quadratic
// I ~ a x^2 + b xy + c y^2 + d x + e y + f and solving grad I = 0 (the saddle point).
// This is robust on clean (rendered) boards; real-image robustness is the optional OpenCV
// path (detect_opencv.hpp) or future hardening.

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/image.hpp"

namespace calibforge {
namespace detect {

struct Corner2 {
  double x = 0.0;
  double y = 0.0;
};

// Refine a corner near (x0,y0) to sub-pixel via a local quadratic saddle fit (window
// radius R). Returns the input if the window leaves the image or the fit degenerates.
inline Corner2 refineCornerSaddle(const Image8& im, double x0, double y0, int R = 5) {
  double cx = x0, cy = y0;
  for (int iter = 0; iter < 4; ++iter) {
    const int ix = static_cast<int>(std::lround(cx));
    const int iy = static_cast<int>(std::lround(cy));
    if (ix - R < 0 || iy - R < 0 || ix + R >= im.width || iy + R >= im.height) break;
    Eigen::Matrix<double, 6, 6> AtA = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> Atb = Eigen::Matrix<double, 6, 1>::Zero();
    for (int dy = -R; dy <= R; ++dy) {
      for (int dx = -R; dx <= R; ++dx) {
        const double xx = dx, yy = dy;
        Eigen::Matrix<double, 6, 1> phi;
        phi << xx * xx, xx * yy, yy * yy, xx, yy, 1.0;
        AtA += phi * phi.transpose();
        Atb += phi * static_cast<double>(im.at(ix + dx, iy + dy));
      }
    }
    const Eigen::Matrix<double, 6, 1> k = AtA.ldlt().solve(Atb);
    const double a = k[0], b = k[1], c = k[2], d = k[3], e = k[4];
    Eigen::Matrix2d Hm;
    Hm << 2.0 * a, b, b, 2.0 * c;
    if (std::fabs(Hm.determinant()) < 1e-12) break;
    const Eigen::Vector2d off = Hm.ldlt().solve(Eigen::Vector2d(-d, -e));
    if (!std::isfinite(off[0]) || !std::isfinite(off[1]) ||
        std::fabs(off[0]) > R || std::fabs(off[1]) > R)
      break;  // diverged: keep last estimate
    cx = ix + off[0];
    cy = iy + off[1];
  }
  return {cx, cy};
}

namespace detail {
inline std::vector<double> boxBlur(const Image8& im) {
  const int w = im.width, h = im.height;
  std::vector<double> out(static_cast<std::size_t>(w) * h, 0.0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double s = 0.0;
      int n = 0;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx, yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < w && yy < h) {
            s += im.at(xx, yy);
            ++n;
          }
        }
      out[static_cast<std::size_t>(y) * w + x] = s / n;
    }
  return out;
}
}  // namespace detail

// Find checkerboard corners: saddle response, non-max suppression, sub-pixel refinement.
inline std::vector<Corner2> detectSaddleCorners(const Image8& im, double rel_threshold = 0.05,
                                                int nms_radius = 4) {
  const int w = im.width, h = im.height;
  const std::vector<double> f = detail::boxBlur(im);
  std::vector<double> resp(static_cast<std::size_t>(w) * h, 0.0);
  double maxr = 0.0;
  for (int y = 1; y < h - 1; ++y)
    for (int x = 1; x < w - 1; ++x) {
      const double Ixx = f[y * w + x + 1] - 2 * f[y * w + x] + f[y * w + x - 1];
      const double Iyy = f[(y + 1) * w + x] - 2 * f[y * w + x] + f[(y - 1) * w + x];
      const double Ixy = 0.25 * (f[(y + 1) * w + x + 1] - f[(y + 1) * w + x - 1] -
                                 f[(y - 1) * w + x + 1] + f[(y - 1) * w + x - 1]);
      double r = Ixy * Ixy - Ixx * Iyy;  // -det(Hessian)
      if (r < 0.0) r = 0.0;
      resp[static_cast<std::size_t>(y) * w + x] = r;
      if (r > maxr) maxr = r;
    }
  const double thr = rel_threshold * maxr;
  std::vector<Corner2> out;
  for (int y = nms_radius; y < h - nms_radius; ++y)
    for (int x = nms_radius; x < w - nms_radius; ++x) {
      const double r = resp[static_cast<std::size_t>(y) * w + x];
      if (r < thr) continue;
      bool peak = true;
      for (int dy = -nms_radius; dy <= nms_radius && peak; ++dy)
        for (int dx = -nms_radius; dx <= nms_radius; ++dx)
          if (resp[static_cast<std::size_t>(y + dy) * w + (x + dx)] > r) {
            peak = false;
            break;
          }
      if (peak) out.push_back(refineCornerSaddle(im, x, y, 3));
    }
  return out;
}

}  // namespace detect
}  // namespace calibforge
