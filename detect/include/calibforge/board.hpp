#pragma once
//
// CalibForge detect — calibration-target geometry (header-only).
//
// A checkerboard with cols x rows INNER corners has (cols+1) x (rows+1) squares. Inner
// corner (i,j) sits at metric ((i+1)*square_size, (j+1)*square_size, 0); object points are
// emitted row-major (j outer, i inner) to match the detector's ordering.

#include <cstddef>
#include <vector>

#include "calibforge/camera_model.hpp"

namespace calibforge {
namespace detect {

struct CheckerboardSpec {
  int cols = 0;               // inner corners across
  int rows = 0;               // inner corners down
  double square_size = 0.0;   // metres

  std::size_t numCorners() const { return static_cast<std::size_t>(cols) * rows; }

  std::vector<Vec3> objectPoints() const {
    std::vector<Vec3> pts;
    pts.reserve(numCorners());
    for (int j = 0; j < rows; ++j)
      for (int i = 0; i < cols; ++i)
        pts.push_back(Vec3{(i + 1) * square_size, (j + 1) * square_size, 0.0});
    return pts;
  }

  // Extent of the full board (including the border squares), in metres.
  double extentX() const { return (cols + 1) * square_size; }
  double extentY() const { return (rows + 1) * square_size; }
};

}  // namespace detect
}  // namespace calibforge
