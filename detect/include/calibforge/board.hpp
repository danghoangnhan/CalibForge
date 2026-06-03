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

// ChArUco board: a checkerboard with embedded ArUco markers in the white squares.
// The geometry of the inner chessboard corners matches the (cols-1) x (rows-1) inner-corner
// pattern of a standard checkerboard whose square count is (cols, rows). square_size and
// marker_size are in metres; marker_size <= square_size. The aruco dictionary id is an OpenCV
// cv::aruco::PREDEFINED_DICTIONARY_NAME value (e.g. DICT_4X4_50 = 0); see detect_opencv.hpp.
struct CharucoBoardSpec {
  int squares_x = 0;          // chessboard squares across (cols)
  int squares_y = 0;          // chessboard squares down (rows)
  double square_size = 0.0;   // metres, chessboard square edge
  double marker_size = 0.0;   // metres, embedded ArUco marker edge (must be < square_size)
  int dict_id = 0;            // cv::aruco predefined dictionary id

  // Inner corners (chessboard corners on the board, where ChArUco places its sub-pixel hits).
  int innerCols() const { return squares_x - 1; }
  int innerRows() const { return squares_y - 1; }
  std::size_t numInnerCorners() const {
    return static_cast<std::size_t>(innerCols()) * innerRows();
  }

  // Inner-corner object points, row-major (j outer, i inner). Top-left inner corner sits at
  // (square_size, square_size, 0) — matches OpenCV's CharucoBoard convention.
  std::vector<Vec3> objectPoints() const {
    std::vector<Vec3> pts;
    pts.reserve(numInnerCorners());
    for (int j = 0; j < innerRows(); ++j)
      for (int i = 0; i < innerCols(); ++i)
        pts.push_back(Vec3{(i + 1) * square_size, (j + 1) * square_size, 0.0});
    return pts;
  }
};

// Kalibr-style AprilGrid: grid of independent AprilTag tags, each with its own ID 0..N-1
// (row-major). Per-tag object points are the four corners of the tag, top-left then clockwise
// (top-left, top-right, bottom-right, bottom-left), matching OpenCV's per-tag corner order.
// tag_spacing is the ratio of the gap between tags to the tag edge (Kalibr convention; gap =
// tag_size * tag_spacing). The aruco dictionary id is an OpenCV cv::aruco predefined dict id;
// the standard Kalibr AprilGrid uses cv::aruco::DICT_APRILTAG_36h11.
struct AprilGridSpec {
  int cols = 0;               // tags across
  int rows = 0;               // tags down
  double tag_size = 0.0;      // metres, tag edge length
  double tag_spacing = 0.3;   // ratio: gap = tag_size * tag_spacing
  int dict_id = 0;            // cv::aruco predefined dictionary id

  int numTags() const { return cols * rows; }
  std::size_t numCorners() const { return static_cast<std::size_t>(numTags()) * 4; }
  double pitch() const { return tag_size * (1.0 + tag_spacing); }

  // (tag_id, corner_in_tag) -> object point on the planar board (z=0). corner_in_tag ordered
  // as OpenCV returns them: 0=TL, 1=TR, 2=BR, 3=BL. The top-left tag (id=0) sits with its TL
  // corner at the origin.
  std::vector<Vec3> objectPoints() const {
    std::vector<Vec3> pts;
    pts.reserve(numCorners());
    const double s = tag_size;
    const double p = pitch();
    for (int j = 0; j < rows; ++j) {
      for (int i = 0; i < cols; ++i) {
        const double x0 = i * p;
        const double y0 = j * p;
        pts.push_back(Vec3{x0,     y0,     0.0});  // TL
        pts.push_back(Vec3{x0 + s, y0,     0.0});  // TR
        pts.push_back(Vec3{x0 + s, y0 + s, 0.0});  // BR
        pts.push_back(Vec3{x0,     y0 + s, 0.0});  // BL
      }
    }
    return pts;
  }
};

}  // namespace detect
}  // namespace calibforge
