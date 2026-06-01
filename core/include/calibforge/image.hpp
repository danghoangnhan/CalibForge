#pragma once
//
// CalibForge core — minimal 8-bit grayscale image (header-only), shared by detect/ (board
// rendering + corner detection) and apply/ (remap). Row-major; bilinear sampling with a
// zero (BORDER_CONSTANT) border, matching the apply-path remap reference.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace calibforge {

struct Image8 {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> data;  // size width*height, row-major

  Image8() = default;
  Image8(int w, int h, std::uint8_t fill = 0)
      : width(w), height(h), data(static_cast<std::size_t>(w) * h, fill) {}

  std::uint8_t& at(int x, int y) { return data[static_cast<std::size_t>(y) * width + x]; }
  std::uint8_t at(int x, int y) const { return data[static_cast<std::size_t>(y) * width + x]; }

  // Bilinear sample at sub-pixel (x,y); outside the image returns 0 (constant border).
  double bilinear(double x, double y) const {
    if (x < 0.0 || y < 0.0 || x > width - 1.0 || y > height - 1.0) return 0.0;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const double fx = x - x0;
    const double fy = y - y0;
    const double a = at(x0, y0), b = at(x1, y0), c = at(x0, y1), d = at(x1, y1);
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy;
  }
};

}  // namespace calibforge
