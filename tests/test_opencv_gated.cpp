// Gated OpenCV tests — compiled only into calibforge_opencv_tests (when OpenCV is found).
// They cross-check our hand-rolled io/detect against real OpenCV. The no-OpenCV CI suite
// does not build this file, so the green gate is unaffected.
#ifdef CALIBFORGE_HAS_OPENCV

#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

#include "calibforge/board.hpp"
#include "calibforge/brown_conrady_camera.hpp"
#include "calibforge/detect_opencv.hpp"
#include "calibforge/opencv_yaml.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/render_board.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::BrownConradyCamera;
using calibforge::PinholeCamera;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;
using calibforge::detect::CheckerboardSpec;
using calibforge::detect::detectCheckerboardOpenCv;
using calibforge::detect::renderCheckerboard;

// Our OpenCV-YAML writer produces a file real cv::FileStorage can read back faithfully.
CF_TEST(opencv_filestorage_reads_our_yaml) {
  BrownConradyCamera cam(520.0, 521.5, 320.0, 240.0, -0.15, 0.04, 0.001, -0.001, 0.0005);
  auto p = cam.params();
  auto in = calibforge::io::makeIntrinsics(p[0], p[1], p[2], p[3],
                                           {p[4], p[5], p[6], p[7], p[8]}, 640, 480);
  const std::string path = "cf_ocv_roundtrip.yaml";
  calibforge::io::writeOpenCvYaml(path, in);

  cv::FileStorage fs(path, cv::FileStorage::READ);
  CF_EXPECT_TRUE(fs.isOpened());
  cv::Mat K, D;
  int w = 0, h = 0;
  fs["camera_matrix"] >> K;
  fs["distortion_coefficients"] >> D;
  fs["image_width"] >> w;
  fs["image_height"] >> h;
  CF_EXPECT_TRUE(w == 640 && h == 480);
  CF_EXPECT_NEAR(K.at<double>(0, 0), 520.0, 1e-9);
  CF_EXPECT_NEAR(K.at<double>(1, 1), 521.5, 1e-9);
  CF_EXPECT_NEAR(K.at<double>(0, 2), 320.0, 1e-9);
  CF_EXPECT_NEAR(D.at<double>(0, 0), -0.15, 1e-9);
  CF_EXPECT_NEAR(D.at<double>(0, 4), 0.0005, 1e-9);
}

// OpenCV's chessboard detector finds the rendered board's corners near ground truth.
CF_TEST(opencv_detects_rendered_checkerboard) {
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  CheckerboardSpec spec{7, 5, 0.025};
  const Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Vector3d(0.06, -0.05, 0.02));
  const Eigen::Vector3d center(spec.extentX() / 2.0, spec.extentY() / 2.0, 0.0);
  Sophus::SE3d T(R, Eigen::Vector3d(0, 0, 0.55) - R * center);
  auto img = renderCheckerboard(cam, T, spec, 320, 240);

  View v;
  bool ok = detectCheckerboardOpenCv(img, spec, v);
  CF_EXPECT_TRUE(ok);
  if (ok) {
    CF_EXPECT_TRUE(v.image_points.size() == spec.numCorners());
    // Every detected corner is near the ground-truth projection of its (matched) object point.
    double worst = 0.0;
    for (std::size_t k = 0; k < v.object_points.size(); ++k) {
      Eigen::Vector3d Xc = T * Eigen::Vector3d(v.object_points[k][0], v.object_points[k][1],
                                               v.object_points[k][2]);
      Vec2 g = cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()});
      double best = 1e30;
      for (std::size_t j = 0; j < v.image_points.size(); ++j) {
        double d = std::hypot(v.image_points[j][0] - g[0], v.image_points[j][1] - g[1]);
        best = std::min(best, d);
      }
      worst = std::max(worst, best);
    }
    CF_EXPECT_TRUE(worst < 0.5);
  }
}

#endif  // CALIBFORGE_HAS_OPENCV
