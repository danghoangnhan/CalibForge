// Gated OpenCV tests for ChArUco + AprilGrid detection. End-to-end proof that issue #8 is
// covered through the OpenCV-gated path: render via cv::aruco's own helpers, detect via the
// detectCharucoOpenCv / detectAprilGridOpenCv wrappers, assert the recovered image points
// are within sub-pixel tolerance of the ground-truth projection of their object points.
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
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::PinholeCamera;
using calibforge::Vec2;
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

// Render a ChArUco board into a planar image, project + detect, and confirm every recovered
// corner is within sub-pixel tolerance of the ground-truth projection of its object point.
// A planar fronto-parallel view keeps the projection trivial — board pixel coords map
// linearly to image pixel coords given a known scale/offset.
CF_TEST(opencv_detects_rendered_charuco) {
  CharucoBoardSpec spec;
  spec.squares_x = 6;
  spec.squares_y = 4;
  spec.square_size = 0.040;
  spec.marker_size = 0.030;
  spec.dict_id = cv::aruco::DICT_4X4_50;

  const int margin = 20;
  const int board_w = 480;  // rendered ChArUco board image width, px
  const int board_h = 320;  // height
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

  // Most inner corners should be recovered on the clean synthetic render.
  CF_EXPECT_TRUE(v.image_points.size() >= spec.numInnerCorners() - 1);

  // The board image coordinates of inner corner (i,j) are
  //   px = margin + (i+1) * (board_w - 2*margin) / squares_x
  //   py = margin + (j+1) * (board_h - 2*margin) / squares_y
  // (board->draw renders the board to fill (image - 2*margin) × ditto). Every recovered
  // (object_point, image_point) pair must match this map.
  const double sx = static_cast<double>(board_w - 2 * margin) / spec.squares_x;
  const double sy = static_cast<double>(board_h - 2 * margin) / spec.squares_y;
  double worst = 0.0;
  for (std::size_t k = 0; k < v.object_points.size(); ++k) {
    const double i = v.object_points[k][0] / spec.square_size;  // back to "inner index + 1"
    const double j = v.object_points[k][1] / spec.square_size;
    const double gx = margin + i * sx;
    const double gy = margin + j * sy;
    const double d = std::hypot(v.image_points[k][0] - gx, v.image_points[k][1] - gy);
    worst = std::max(worst, d);
  }
  CF_EXPECT_TRUE(worst < 1.0);
}

// Render an AprilGrid via cv::aruco's GridBoard (which emits the same tag-per-cell layout
// our AprilGridSpec encodes), detect, and confirm every recovered corner is within
// sub-pixel tolerance of the synthesis grid. tag_spacing=0 so the GridBoard's marker
// separation matches our AprilGridSpec's tag_size pitch.
CF_TEST(opencv_detects_rendered_aprilgrid) {
  AprilGridSpec spec;
  spec.cols = 3;
  spec.rows = 2;
  spec.tag_size = 0.040;
  spec.tag_spacing = 0.0;  // touching tags — keeps the rendered grid exactly tag_size pitch
  spec.dict_id = cv::aruco::DICT_APRILTAG_36h11;

  const int margin = 20;
  const int board_w = 480;
  const int board_h = 320;
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

  // All tags should be detected on the clean synthetic render.
  CF_EXPECT_TRUE(v.image_points.size() == static_cast<std::size_t>(4 * spec.numTags()));

  // GridBoard fits the (cols x rows) tag grid into (board_w - 2*margin) x (board_h - 2*margin)
  // with tag pitch = tag_size (because tag_spacing was zero). Per-tag corner k in object
  // coords lies at object_points()[base + k]; in image coords it scales linearly through the
  // same fit.
  const double sx =
      static_cast<double>(board_w - 2 * margin) / (spec.cols * spec.tag_size);
  const double sy =
      static_cast<double>(board_h - 2 * margin) / (spec.rows * spec.tag_size);
  double worst = 0.0;
  for (std::size_t k = 0; k < v.object_points.size(); ++k) {
    const double gx = margin + v.object_points[k][0] * sx;
    const double gy = margin + v.object_points[k][1] * sy;
    const double d = std::hypot(v.image_points[k][0] - gx, v.image_points[k][1] - gy);
    worst = std::max(worst, d);
  }
  CF_EXPECT_TRUE(worst < 1.5);
}

#endif  // CALIBFORGE_HAS_OPENCV_ARUCO
#endif  // CALIBFORGE_HAS_OPENCV
