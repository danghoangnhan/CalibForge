// Stereo rectification (v0.5 capability-matrix completion): turns a calibrated stereo pair into a
// rectified pair whose epipolar lines are horizontal. The DEFINING property — a 3D point lands on
// the same rectified ROW in both views — is asserted to ~1e-6, plus the rotation/baseline/Q algebra
// and the warp-map R_rect plumbing. Mirrors the synthetic geometry of test_calibrate_stereo.cpp.

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/rectify_stereo.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

using calibforge::CameraModel;
using calibforge::computeStereoRectification;
using calibforge::PinholeCamera;
using calibforge::StereoRectification;
using calibforge::stereoRectificationToCameraInfo;
using calibforge::Vec2;
using calibforge::Vec3;

namespace {

calibforge::CameraFactory pinholeFactory() {
  return [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };
}

// cam0 = (500,500,320,240), cam1 = (510,508,322,238); small relative rotation + ~10cm x-baseline.
struct Scene {
  Eigen::VectorXd i0{(Eigen::VectorXd(4) << 500.0, 500.0, 320.0, 240.0).finished()};
  Eigen::VectorXd i1{(Eigen::VectorXd(4) << 510.0, 508.0, 322.0, 238.0).finished()};
  Sophus::SE3d T01{Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.015)),
                   Eigen::Vector3d(-0.10, 0.005, 0.002)};
  int w = 640, h = 480;
  // A grid of 3D points in cam0, comfortably inside both FOVs and in front of both cameras.
  std::vector<Eigen::Vector3d> pointsCam0() const {
    std::vector<Eigen::Vector3d> pts;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        pts.emplace_back(-0.2 + 0.1 * c, -0.2 + 0.1 * r, 1.0 + 0.05 * ((r + c) % 3));
    return pts;
  }
};

}  // namespace

CF_TEST(rectify_stereo_aligns_epipolar_rows) {
  const Scene s;
  const StereoRectification rect =
      computeStereoRectification(s.i0, s.i1, s.T01, s.w, s.h, pinholeFactory(), pinholeFactory());
  const double f = rect.K_rect[0], cy = rect.K_rect[3];

  // THE defining property: after rectification every 3D point shares one row across both views.
  for (const Eigen::Vector3d& Xc0 : s.pointsCam0()) {
    const Eigen::Vector3d Xc1 = s.T01 * Xc0;
    const Eigen::Vector3d r0 = rect.R0 * Xc0;  // rectified ray, cam0
    const Eigen::Vector3d r1 = rect.R1 * Xc1;  // rectified ray, cam1
    CF_EXPECT_TRUE(r0.z() > 0 && r1.z() > 0);  // in front of both rectified cameras
    const double v0 = f * r0.y() / r0.z() + cy;
    const double v1 = f * r1.y() / r1.z() + cy;
    CF_EXPECT_NEAR(v0, v1, 1e-6);  // same row => horizontal epipolar lines
  }
}

CF_TEST(rectify_stereo_warp_map_applies_rectifying_rotation) {
  const Scene s;
  const StereoRectification rect =
      computeStereoRectification(s.i0, s.i1, s.T01, s.w, s.h, pinholeFactory(), pinholeFactory());
  const double f = rect.K_rect[0], cx = rect.K_rect[2], cy = rect.K_rect[3];
  PinholeCamera cam0(s.i0[0], s.i0[1], s.i0[2], s.i0[3]);

  // The rectified warp map must equal: rotate the rectified ray back into the physical frame, then
  // project through the distorted source model (confirms generateWarpMap honored R_rect).
  const int px = 400, py = 260;
  const Vec2 src = rect.warp0.at(px, py);
  const double xn = (px - cx) / f, yn = (py - cy) / f;
  const Eigen::Vector3d d = rect.R0.transpose() * Eigen::Vector3d(xn, yn, 1.0);
  const Vec2 expect = cam0.project(Vec3{d.x(), d.y(), d.z()});
  CF_EXPECT_NEAR(src[0], expect[0], 1e-9);
  CF_EXPECT_NEAR(src[1], expect[1], 1e-9);
  CF_EXPECT_TRUE(rect.warp0.width == s.w && rect.warp0.height == s.h);
}

CF_TEST(rectify_stereo_rotations_baseline_and_Q_are_valid) {
  const Scene s;
  const StereoRectification rect =
      computeStereoRectification(s.i0, s.i1, s.T01, s.w, s.h, pinholeFactory(), pinholeFactory());
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R = s.T01.rotationMatrix();

  // R0, R1 are proper rotations.
  CF_EXPECT_NEAR((rect.R0 * rect.R0.transpose() - I).norm(), 0.0, 1e-12);
  CF_EXPECT_NEAR((rect.R1 * rect.R1.transpose() - I).norm(), 0.0, 1e-12);
  CF_EXPECT_NEAR(rect.R0.determinant(), 1.0, 1e-12);
  CF_EXPECT_NEAR(rect.R1.determinant(), 1.0, 1e-12);
  // Rectified frames parallel: R0 == R1*R.
  CF_EXPECT_NEAR((rect.R1 * R * rect.R0.transpose() - I).norm(), 0.0, 1e-9);

  // Baseline + ROS projection convention.
  const double b = s.T01.translation().norm();
  const double f = rect.K_rect[0];
  CF_EXPECT_NEAR(rect.baseline, b, 1e-12);
  CF_EXPECT_NEAR(rect.P1[3], -f * b, 1e-9);  // right camera Tx = -fx*baseline
  CF_EXPECT_NEAR(rect.P0[3], 0.0, 1e-12);

  // Q maps (u, v, disparity, 1) back to the correct metric depth Z = f*b/disparity.
  Eigen::Matrix4d Qm;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) Qm(r, c) = rect.Q[static_cast<std::size_t>(r) * 4 + c];
  const double Z = 2.5, disp = f * b / Z;
  const Eigen::Vector4d xyzw = Qm * Eigen::Vector4d(350.0, 250.0, disp, 1.0);
  CF_EXPECT_NEAR(xyzw[2] / xyzw[3], Z, 1e-9);
}

CF_TEST(rectify_stereo_emits_ros_camera_info_pair) {
  const Scene s;
  const StereoRectification rect =
      computeStereoRectification(s.i0, s.i1, s.T01, s.w, s.h, pinholeFactory(), pinholeFactory());
  calibforge::io::CameraInfo base0, base1;  // raw K/D would be filled by the caller in practice
  base0.camera_name = "left";
  base1.camera_name = "right";
  const auto pair = stereoRectificationToCameraInfo(rect, base0, base1);
  // R is the rectifying rotation (row-major), P is the rectified projection.
  CF_EXPECT_NEAR(pair.first.R[0], rect.R0(0, 0), 1e-12);
  CF_EXPECT_NEAR(pair.first.R[1], rect.R0(0, 1), 1e-12);  // row-major off-diagonal
  CF_EXPECT_NEAR(pair.second.R[3], rect.R1(1, 0), 1e-12);
  CF_EXPECT_NEAR(pair.second.P[3], rect.P1[3], 1e-12);  // baseline term survives
  CF_EXPECT_TRUE(pair.first.camera_name == "left" && pair.second.camera_name == "right");
}

CF_TEST(rectify_stereo_rejects_degenerate_baseline) {
  const Scene s;
  bool threw = false;
  try {
    computeStereoRectification(s.i0, s.i1, Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d::Zero()),
                               s.w, s.h, pinholeFactory(), pinholeFactory());
  } catch (const std::exception&) {
    threw = true;
  }
  CF_EXPECT_TRUE(threw);  // a pure-rotation rig has no baseline to rectify
}
