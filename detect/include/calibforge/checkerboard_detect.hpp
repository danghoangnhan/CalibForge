#pragma once
//
// CalibForge detect — assemble a calibration View from detected checkerboard corners
// (header-only). Each board inner corner is matched to the nearest detected saddle corner
// within max_dist. `expected` gives approximate pixel locations of the inner corners (from
// a coarse homography/PnP; in the synthetic harness, the projected ground truth) to resolve
// the ordering — full from-scratch grid traversal is the optional OpenCV path / future work.

#include <cmath>
#include <cstddef>
#include <vector>

#include "calibforge/board.hpp"
#include "calibforge/calibrate_single.hpp"  // View
#include "calibforge/corner_detect.hpp"
#include "calibforge/image.hpp"

namespace calibforge {
namespace detect {

inline View detectCheckerboardView(const Image8& im, const CheckerboardSpec& spec,
                                   const std::vector<Vec2>& expected, double max_dist = 6.0,
                                   double rel_threshold = 0.05) {
  const std::vector<Corner2> corners = detectSaddleCorners(im, rel_threshold);
  const std::vector<Vec3> obj = spec.objectPoints();
  View v;
  for (std::size_t k = 0; k < obj.size() && k < expected.size(); ++k) {
    double best = 1e300;
    int bi = -1;
    for (std::size_t c = 0; c < corners.size(); ++c) {
      const double dx = corners[c].x - expected[k][0];
      const double dy = corners[c].y - expected[k][1];
      const double d = dx * dx + dy * dy;
      if (d < best) {
        best = d;
        bi = static_cast<int>(c);
      }
    }
    if (bi >= 0 && std::sqrt(best) <= max_dist) {
      v.object_points.push_back(obj[k]);
      v.image_points.push_back(Vec2{corners[bi].x, corners[bi].y});
    }
  }
  return v;
}

}  // namespace detect
}  // namespace calibforge
