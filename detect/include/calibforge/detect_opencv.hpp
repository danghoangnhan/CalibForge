#pragma once
//
// CalibForge detect — OPTIONAL OpenCV-backed real-image detection (header-only).
//
// Compiled only when the build finds OpenCV (CALIBFORGE_HAS_OPENCV). This is the
// production/real-image path the lean hand-rolled detector (corner_detect.hpp) is not: it
// wraps OpenCV's robust ChArUco/chessboard detectors + sub-pixel refinement. Exercised only
// in the gated calibforge_opencv_tests; the no-OpenCV CI suite uses the synthetic path.

#ifdef CALIBFORGE_HAS_OPENCV

#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "calibforge/board.hpp"
#include "calibforge/calibrate_single.hpp"  // View
#include "calibforge/image.hpp"

namespace calibforge {
namespace detect {

inline cv::Mat toMat(const Image8& im) {
  cv::Mat m(im.height, im.width, CV_8UC1);
  for (int y = 0; y < im.height; ++y)
    for (int x = 0; x < im.width; ++x) m.at<unsigned char>(y, x) = im.at(x, y);
  return m;
}

// Robust real-image checkerboard detection: OpenCV's SB detector with a sub-pixel fallback.
// Returns false if the full pattern is not found. Corners are paired with object points in
// the spec's row-major order (OpenCV returns the inner corners in a consistent raster order).
inline bool detectCheckerboardOpenCv(const Image8& im, const CheckerboardSpec& spec, View& out) {
  const cv::Mat gray = toMat(im);
  const cv::Size pattern(spec.cols, spec.rows);
  std::vector<cv::Point2f> corners;
  bool ok = cv::findChessboardCornersSB(gray, pattern, corners);
  if (!ok) {
    ok = cv::findChessboardCorners(gray, pattern, corners);
    if (ok)
      cv::cornerSubPix(
          gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
  }
  if (!ok || corners.size() != spec.numCorners()) return false;

  const std::vector<Vec3> obj = spec.objectPoints();
  out.object_points.clear();
  out.image_points.clear();
  for (std::size_t k = 0; k < obj.size(); ++k) {
    out.object_points.push_back(obj[k]);
    out.image_points.push_back(Vec2{corners[k].x, corners[k].y});
  }
  return true;
}

}  // namespace detect
}  // namespace calibforge

#endif  // CALIBFORGE_HAS_OPENCV
