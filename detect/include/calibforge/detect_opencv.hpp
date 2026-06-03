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

// The ChArUco / AprilGrid wrappers target OpenCV's pre-4.7 aruco API (Ubuntu 24.04 LTS ships
// OpenCV 4.6.0). Newer 4.7+ users can rebuild against the cv::ArucoDetector/CharucoDetector
// classes without touching the public CalibForge interface.
#if defined(__has_include)
#if __has_include(<opencv2/aruco.hpp>) && __has_include(<opencv2/aruco/charuco.hpp>)
#define CALIBFORGE_HAS_OPENCV_ARUCO 1
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#endif
#endif

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

#ifdef CALIBFORGE_HAS_OPENCV_ARUCO

// ChArUco detection via cv::aruco. Detects the embedded markers, then sub-pixel-interpolates
// the inner chessboard corners. ChArUco's strength is partial-occlusion tolerance — corners
// are returned for whichever subset of the inner grid the markers were able to resolve, NOT
// the full grid. We pair them with their object points by the per-corner id returned by
// interpolateCornersCharuco (row-major into the inner grid). Returns true if at least one
// corner was recovered.
inline bool detectCharucoOpenCv(const Image8& im, const CharucoBoardSpec& spec, View& out) {
  const cv::Mat gray = toMat(im);
  cv::Ptr<cv::aruco::Dictionary> dict =
      cv::aruco::getPredefinedDictionary(spec.dict_id);
  cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
      spec.squares_x, spec.squares_y, static_cast<float>(spec.square_size),
      static_cast<float>(spec.marker_size), dict);

  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  cv::aruco::detectMarkers(gray, dict, marker_corners, marker_ids);
  if (marker_ids.empty()) return false;

  std::vector<cv::Point2f> ch_corners;
  std::vector<int> ch_ids;
  const int n = cv::aruco::interpolateCornersCharuco(
      marker_corners, marker_ids, gray, board, ch_corners, ch_ids);
  if (n <= 0) return false;

  const std::vector<Vec3> obj = spec.objectPoints();
  out.object_points.clear();
  out.image_points.clear();
  for (std::size_t k = 0; k < ch_ids.size(); ++k) {
    const int id = ch_ids[k];
    if (id < 0 || static_cast<std::size_t>(id) >= obj.size()) continue;
    out.object_points.push_back(obj[id]);
    out.image_points.push_back(Vec2{ch_corners[k].x, ch_corners[k].y});
  }
  return !out.image_points.empty();
}

// AprilGrid detection via cv::aruco (Kalibr-style independent-tag grid). Detects each tag and
// emits its four corners paired with the tag's object points by tag_id (row-major over the
// grid). Like ChArUco, partial-grid recovery is fine — only the detected tags contribute. The
// dictionary should be an apriltag predefined dict (e.g. cv::aruco::DICT_APRILTAG_36h11).
inline bool detectAprilGridOpenCv(const Image8& im, const AprilGridSpec& spec, View& out) {
  const cv::Mat gray = toMat(im);
  cv::Ptr<cv::aruco::Dictionary> dict =
      cv::aruco::getPredefinedDictionary(spec.dict_id);

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  cv::aruco::detectMarkers(gray, dict, corners, ids);
  if (ids.empty()) return false;

  const std::vector<Vec3> obj = spec.objectPoints();  // 4 * numTags entries, ordered TL,TR,BR,BL
  const int n_tags = spec.numTags();
  out.object_points.clear();
  out.image_points.clear();
  for (std::size_t k = 0; k < ids.size(); ++k) {
    const int tag_id = ids[k];
    if (tag_id < 0 || tag_id >= n_tags) continue;
    if (corners[k].size() != 4) continue;
    const std::size_t base = static_cast<std::size_t>(tag_id) * 4;
    for (int c = 0; c < 4; ++c) {
      out.object_points.push_back(obj[base + c]);
      out.image_points.push_back(Vec2{corners[k][c].x, corners[k][c].y});
    }
  }
  return !out.image_points.empty();
}

#endif  // CALIBFORGE_HAS_OPENCV_ARUCO

}  // namespace detect
}  // namespace calibforge

#endif  // CALIBFORGE_HAS_OPENCV
