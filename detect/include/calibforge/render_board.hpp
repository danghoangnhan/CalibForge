#pragma once
//
// CalibForge detect — synthetic checkerboard renderer (header-only).
//
// Ground-truth generator for the detector tests: for each (super-sampled) output pixel we
// unproject a ray, transform it to world, intersect the board plane (Z=0), and look up the
// square colour. Supersampling gives anti-aliased edges so sub-pixel corner detection is
// well-posed. Pixels off the board get a neutral background.

#include <cmath>
#include <cstdint>

#include <Eigen/Dense>

#include "calibforge/board.hpp"
#include "calibforge/camera_model.hpp"
#include "calibforge/image.hpp"
#include "sophus/se3.hpp"

namespace calibforge {
namespace detect {

inline Image8 renderCheckerboard(const CameraModel& cam, const Sophus::SE3d& T_world_cam,
                                 const CheckerboardSpec& spec, int w, int h,
                                 std::uint8_t dark = 40, std::uint8_t light = 215,
                                 std::uint8_t background = 128, int supersample = 6) {
  Image8 img(w, h);
  const Sophus::SE3d Tcw = T_world_cam.inverse();  // cam -> world
  const Eigen::Vector3d C = Tcw.translation();      // camera centre in world
  const Eigen::Matrix3d Rcw = Tcw.rotationMatrix();
  const double s = spec.square_size;
  const double maxx = spec.extentX(), maxy = spec.extentY();
  const int ss = supersample;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double acc = 0.0;
      for (int sy = 0; sy < ss; ++sy) {
        for (int sx = 0; sx < ss; ++sx) {
          // Pixel (x,y) is centred at the integer coordinate (matching cam.project()'s
          // convention), so sub-samples span [x-0.5, x+0.5).
          const double u = x - 0.5 + (sx + 0.5) / ss;
          const double v = y - 0.5 + (sy + 0.5) / ss;
          const Vec3 ray = cam.unproject(Vec2{u, v});
          const Eigen::Vector3d d = Rcw * Eigen::Vector3d(ray[0], ray[1], ray[2]);
          double val = background;
          if (std::fabs(d.z()) > 1e-12) {
            const double lam = -C.z() / d.z();
            if (lam > 0.0) {
              const double X = C.x() + lam * d.x();
              const double Y = C.y() + lam * d.y();
              if (X >= 0.0 && Y >= 0.0 && X < maxx && Y < maxy) {
                const int ci = static_cast<int>(std::floor(X / s));
                const int cj = static_cast<int>(std::floor(Y / s));
                val = ((ci + cj) % 2 == 0) ? dark : light;
              }
            }
          }
          acc += val;
        }
      }
      img.at(x, y) = static_cast<std::uint8_t>(std::lround(acc / (ss * ss)));
    }
  }
  return img;
}

}  // namespace detect
}  // namespace calibforge
