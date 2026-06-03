// Gated OpenCV tests for ChArUco + AprilGrid detection. End-to-end proof that issue #8 is
// covered through the OpenCV-gated path: render via cv::aruco's own helpers, detect via the
// detectCharucoOpenCv / detectAprilGridOpenCv wrappers, assert the recovered observations
// (a) populate the View structure correctly, (b) recover most/all of the target's corners,
// and (c) land within the rendered board's pixel extent in image coordinates.
//
// The no-OpenCV CI suite does not compile this file; the opencv CI job builds + runs it.
#ifdef CALIBFORGE_HAS_OPENCV
#ifdef CALIBFORGE_HAS_OPENCV_ARUCO

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core.hpp>

#include "calibforge/board.hpp"
#include "calibforge/detect_opencv.hpp"
#include "calibforge/image.hpp"
#include "cf_test.hpp"

using calibforge::Vec3;
using calibforge::View;
using calibforge::detect::AprilGridSpec;
using calibforge::detect::CharucoBoardSpec;
using calibforge::detect::detectAprilGridOpenCv;
using calibforge::detect::detectCharucoOpenCv;

// Convert an OpenCV CV_8UC1 image into the project's lightweight Image8 type, so the gated
// detector wrappers consume the same type the rest of the pipeline uses.
static calibforge::Image8 fromMat(const cv::Mat& m) {
  calibforge::Image8 im;
  im.width = m.cols;
  im.height = m.rows;
  im.data.assign(static_cast<std::size_t>(m.cols) * m.rows, 0);
  for (int y = 0; y < m.rows; ++y)
    for (int x = 0; x < m.cols; ++x) im.at(x, y) = m.at<unsigned char>(y, x);
  return im;
}

// Map each row-major-pair (object_point, image_point) the detector returned into a flat list
// for invariant checks. Returns (object-extent xy_max, image-extent xy_max) so we can verify
// that the spread of detections covers the board, plus the bounding box of image points.
struct DetectionStats {
  double obj_min_x = 1e30, obj_min_y = 1e30, obj_max_x = -1e30, obj_max_y = -1e30;
  double img_min_x = 1e30, img_min_y = 1e30, img_max_x = -1e30, img_max_y = -1e30;
};
static DetectionStats statsOf(const View& v) {
  DetectionStats s;
  for (std::size_t k = 0; k < v.object_points.size(); ++k) {
    s.obj_min_x = std::min(s.obj_min_x, v.object_points[k][0]);
    s.obj_max_x = std::max(s.obj_max_x, v.object_points[k][0]);
    s.obj_min_y = std::min(s.obj_min_y, v.object_points[k][1]);
    s.obj_max_y = std::max(s.obj_max_y, v.object_points[k][1]);
    s.img_min_x = std::min(s.img_min_x, v.image_points[k][0]);
    s.img_max_x = std::max(s.img_max_x, v.image_points[k][0]);
    s.img_min_y = std::min(s.img_min_y, v.image_points[k][1]);
    s.img_max_y = std::max(s.img_max_y, v.image_points[k][1]);
  }
  return s;
}

// Pearson correlation between object and image coordinates along each axis — checks that
// the detector returns object points and image points in matching order/correspondence (a
// near-perfect linear fit on a fronto-parallel synthetic render). >= 0.99 is the floor for a
// correct correspondence; bugs that shuffle ids drop this dramatically.
static double pearson(const std::vector<double>& x, const std::vector<double>& y) {
  const std::size_t n = x.size();
  if (n < 2) return 0.0;
  double mx = 0.0, my = 0.0;
  for (std::size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - mx, dy = y[i] - my;
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
  }
  if (sxx <= 0.0 || syy <= 0.0) return 0.0;
  return sxy / std::sqrt(sxx * syy);
}

// Render a ChArUco board into a planar image and detect. Asserts the wrapper returns a
// populated View with the right ordering (object<->image axes correlate strongly), recovers
// most of the inner corners on the clean synthetic render, and lands every point inside the
// drawn board area. Exact-pixel position prediction depends on rendering details we don't
// re-implement; the correlation + spread invariants are what catches detection breakage.
CF_TEST(opencv_detects_rendered_charuco) {
  CharucoBoardSpec spec;
  spec.squares_x = 6;
  spec.squares_y = 4;
  spec.square_size = 0.040;
  spec.marker_size = 0.030;
  spec.dict_id = cv::aruco::DICT_4X4_50;

  const int margin = 20;
  const int board_w = 600;
  const int board_h = 400;
  cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(spec.dict_id);
  cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(
      spec.squares_x, spec.squares_y, static_cast<float>(spec.square_size),
      static_cast<float>(spec.marker_size), dict);
  cv::Mat board_img;
  board->draw(cv::Size(board_w, board_h), board_img, margin, 1);

  View v;
  bool ok = detectCharucoOpenCv(fromMat(board_img), spec, v);
  CF_EXPECT_TRUE(ok);
  if (!ok) return;

  // Detector populated both arrays in lockstep and recovered the majority of the inner grid
  // (clean render -> we expect ~all inner corners; allow 2 misses).
  CF_EXPECT_TRUE(v.object_points.size() == v.image_points.size());
  CF_EXPECT_TRUE(v.image_points.size() + 2 >= spec.numInnerCorners());

  const DetectionStats s = statsOf(v);
  // Image points lie within the rendered board area (margin + a bit of slack for sub-pixel).
  CF_EXPECT_TRUE(s.img_min_x > margin - 2.0);
  CF_EXPECT_TRUE(s.img_min_y > margin - 2.0);
  CF_EXPECT_TRUE(s.img_max_x < board_w - margin + 2.0);
  CF_EXPECT_TRUE(s.img_max_y < board_h - margin + 2.0);
  // Object points populate the inner-corner grid extent (within square_size).
  CF_EXPECT_TRUE(s.obj_max_x - s.obj_min_x >= spec.square_size * (spec.innerCols() - 1) - 1e-9);
  CF_EXPECT_TRUE(s.obj_max_y - s.obj_min_y >= spec.square_size * (spec.innerRows() - 1) - 1e-9);

  // Per-axis object<->image correlation is near 1: detector returned each object point's
  // image projection (not a shuffled correspondence).
  std::vector<double> ox, oy, ix, iy;
  for (std::size_t k = 0; k < v.object_points.size(); ++k) {
    ox.push_back(v.object_points[k][0]); ix.push_back(v.image_points[k][0]);
    oy.push_back(v.object_points[k][1]); iy.push_back(v.image_points[k][1]);
  }
  CF_EXPECT_TRUE(pearson(ox, ix) > 0.99);
  CF_EXPECT_TRUE(pearson(oy, iy) > 0.99);
}

// Render an AprilGrid via cv::aruco's GridBoard (which emits the same tag-per-cell layout
// our AprilGridSpec encodes), detect, and apply the same invariants as the ChArUco test.
// markerSeparation > 0 is required by OpenCV (it rejects zero), so we use Kalibr's standard
// tag_spacing = 0.3 — the gap between tags is 0.3 * tag_size.
CF_TEST(opencv_detects_rendered_aprilgrid) {
  AprilGridSpec spec;
  spec.cols = 3;
  spec.rows = 2;
  spec.tag_size = 0.040;
  spec.tag_spacing = 0.3;
  spec.dict_id = cv::aruco::DICT_APRILTAG_36h11;

  const int margin = 30;
  const int board_w = 600;
  const int board_h = 400;
  cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(spec.dict_id);
  // GridBoard::create takes (markersX, markersY, markerLength, markerSeparation, dict).
  cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(
      spec.cols, spec.rows, static_cast<float>(spec.tag_size),
      static_cast<float>(spec.tag_size * spec.tag_spacing), dict);
  cv::Mat board_img;
  board->draw(cv::Size(board_w, board_h), board_img, margin, 1);

  View v;
  bool ok = detectAprilGridOpenCv(fromMat(board_img), spec, v);
  CF_EXPECT_TRUE(ok);
  if (!ok) return;

  // All tags should be detected on the clean synthetic render -> 4 corners per tag.
  CF_EXPECT_TRUE(v.object_points.size() == v.image_points.size());
  CF_EXPECT_TRUE(v.image_points.size() == static_cast<std::size_t>(4 * spec.numTags()));

  const DetectionStats s = statsOf(v);
  // Image points inside the rendered board area, with some sub-pixel slack.
  CF_EXPECT_TRUE(s.img_min_x > margin - 2.0);
  CF_EXPECT_TRUE(s.img_min_y > margin - 2.0);
  CF_EXPECT_TRUE(s.img_max_x < board_w - margin + 2.0);
  CF_EXPECT_TRUE(s.img_max_y < board_h - margin + 2.0);
  // Object points cover the grid extent (tag_size * cols, tag_size * rows scaled by pitch).
  CF_EXPECT_TRUE(s.obj_max_x - s.obj_min_x >= spec.tag_size * (spec.cols - 1) * (1.0 + spec.tag_spacing) - 1e-9);
  CF_EXPECT_TRUE(s.obj_max_y - s.obj_min_y >= spec.tag_size * (spec.rows - 1) * (1.0 + spec.tag_spacing) - 1e-9);

  // Per-axis object<->image correlation near 1 (correct id-to-corner correspondence).
  std::vector<double> ox, oy, ix, iy;
  for (std::size_t k = 0; k < v.object_points.size(); ++k) {
    ox.push_back(v.object_points[k][0]); ix.push_back(v.image_points[k][0]);
    oy.push_back(v.object_points[k][1]); iy.push_back(v.image_points[k][1]);
  }
  CF_EXPECT_TRUE(pearson(ox, ix) > 0.99);
  CF_EXPECT_TRUE(pearson(oy, iy) > 0.99);
}

#endif  // CALIBFORGE_HAS_OPENCV_ARUCO
#endif  // CALIBFORGE_HAS_OPENCV
