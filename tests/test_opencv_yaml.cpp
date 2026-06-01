#include "calibforge/opencv_yaml.hpp"

#include <string>

#include "calibforge/brown_conrady_camera.hpp"
#include "cf_test.hpp"

using calibforge::BrownConradyCamera;
using calibforge::io::Intrinsics;
using calibforge::io::makeIntrinsics;
using calibforge::io::parseOpenCvYaml;
using calibforge::io::toOpenCvYaml;

// Write a Brown-Conrady intrinsics through YAML and parse it back losslessly; re-serializing
// the parsed result yields a byte-identical string (round-trip is exact at double precision).
CF_TEST(opencv_yaml_roundtrips_brown_conrady) {
  BrownConradyCamera cam(520.0, 521.5, 320.0, 240.0, -0.15, 0.04, 0.001, -0.001, 0.0005);
  auto p = cam.params();  // fx,fy,cx,cy,k1,k2,p1,p2,k3
  Intrinsics in = makeIntrinsics(p[0], p[1], p[2], p[3],
                                 {p[4], p[5], p[6], p[7], p[8]}, 640, 480, "plumb_bob");

  std::string text = toOpenCvYaml(in);
  Intrinsics back = parseOpenCvYaml(text);

  CF_EXPECT_TRUE(back.image_width == 640 && back.image_height == 480);
  CF_EXPECT_TRUE(back.distortion_model == "plumb_bob");
  for (int i = 0; i < 9; ++i) CF_EXPECT_NEAR(back.camera_matrix[i], in.camera_matrix[i], 1e-15);
  CF_EXPECT_TRUE(back.distortion_coeffs.size() == 5);
  for (std::size_t i = 0; i < 5; ++i)
    CF_EXPECT_NEAR(back.distortion_coeffs[i], in.distortion_coeffs[i], 1e-15);
  // Re-serialize: must be byte-identical (lossless).
  CF_EXPECT_TRUE(toOpenCvYaml(back) == text);
}

// Parse a hard-coded OpenCV-produced sample (collapsed "0."/"1." tokens, multi-line data,
// scientific notation) — proves we can LOAD a real OpenCV calibration file.
CF_TEST(opencv_yaml_parses_opencv_produced_sample) {
  const std::string sample =
      "%YAML:1.0\n"
      "---\n"
      "image_width: 1280\n"
      "image_height: 720\n"
      "camera_matrix: !!opencv-matrix\n"
      "   rows: 3\n"
      "   cols: 3\n"
      "   dt: d\n"
      "   data: [ 5.2000000000000000e+02, 0., 6.4000000000000000e+02, 0.,\n"
      "       5.2000000000000000e+02, 3.6000000000000000e+02, 0., 0., 1. ]\n"
      "distortion_coefficients: !!opencv-matrix\n"
      "   rows: 1\n"
      "   cols: 5\n"
      "   dt: d\n"
      "   data: [ -1.5000000000000000e-01, 4.0000000000000001e-02, 1.0000000000000000e-03,\n"
      "       -1.0000000000000000e-03, 0. ]\n";
  Intrinsics in = parseOpenCvYaml(sample);
  CF_EXPECT_TRUE(in.image_width == 1280 && in.image_height == 720);
  CF_EXPECT_NEAR(in.camera_matrix[0], 520.0, 1e-12);  // fx
  CF_EXPECT_NEAR(in.camera_matrix[2], 640.0, 1e-12);  // cx
  CF_EXPECT_NEAR(in.camera_matrix[4], 520.0, 1e-12);  // fy
  CF_EXPECT_NEAR(in.camera_matrix[5], 360.0, 1e-12);  // cy
  CF_EXPECT_NEAR(in.camera_matrix[8], 1.0, 1e-12);
  CF_EXPECT_TRUE(in.distortion_coeffs.size() == 5);
  CF_EXPECT_NEAR(in.distortion_coeffs[0], -0.15, 1e-12);
  CF_EXPECT_NEAR(in.distortion_coeffs[1], 0.04, 1e-12);
  CF_EXPECT_NEAR(in.distortion_coeffs[4], 0.0, 1e-12);
}
