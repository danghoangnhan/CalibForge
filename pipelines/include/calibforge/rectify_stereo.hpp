#pragma once
//
// CalibForge pipeline — stereo rectification (header-only, CPU).
//
// Post-calibration geometry: turns a calibrated stereo pair (two intrinsics + the extrinsic
// T_cam1_cam0 from calibrateStereo) into a RECTIFIED pair whose image planes are coplanar and
// whose epipolar lines are horizontal — so disparity is a pure horizontal shift and depth is
// Z = f*b/disparity. This is the runtime artifact a stereo matcher consumes; it does NOT
// re-optimize anything.
//
// Bouguet construction (Xc1 = R*Xc0 + t):
//   * Split the relative rotation evenly so each camera meets in the middle:
//       w  = log(R);  r0 = exp(+w/2)  (cam0),  r1 = exp(-w/2)  (cam1).
//     Then r1*R == r0, so after the half-rotation both cameras share one "mid" orientation and
//     differ only by the translation.
//   * Build a common rectifying basis in that mid-frame with x along the baseline:
//       base = r0 * (cam1 origin in cam0 frame) = r0 * (-R^T t);
//       e1 = base/|base|,  e2 = (z x e1)/|.|,  e3 = e1 x e2;  R_common rows = [e1; e2; e3].
//   * Per-camera physical->rectified rotations:  R0 = R_common*r0,  R1 = R_common*r1.
//     These satisfy R0 == R1*R, which makes the two rectified frames parallel; combined with the
//     baseline-aligned x-axis it forces every 3D point onto the SAME rectified row in both views.
//
// The shared rectified intrinsics K_rect are pinhole; per-camera distortion is removed by the
// warp maps (which sample the distorted source via apply::generateWarpMap with R0 / R1), so this
// works unchanged for Brown-Conrady / Kannala-Brandt / double-sphere / EUCM via their factories.

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"  // CameraFactory
#include "calibforge/camera_model.hpp"
#include "calibforge/ros_camera_info.hpp"  // io::CameraInfo (rectified pair emitter)
#include "calibforge/warp_map.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace calibforge {

struct StereoRectification {
  Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();  // cam0 physical -> rectified (X_rect = R0*Xc0)
  Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();  // cam1 physical -> rectified
  std::array<double, 4> K_rect = {0, 0, 0, 0};       // shared rectified {fx, fy, cx, cy}
  std::array<double, 12> P0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // 3x4 row-major, [K|0]
  std::array<double, 12> P1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // 3x4, P1[3] = -fx*baseline
  std::array<double, 16> Q = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // 4x4 disp->depth
  double baseline = 0.0;       // b = |t|, metric distance between optical centers
  apply::WarpMap warp0, warp1;  // rectified output pixel -> distorted source pixel, per camera
};

// intrinsics{0,1}: first four entries are {fx, fy, cx, cy} (true for every CalibForge model).
// T_cam1_cam0: the calibrated extrinsic (Xc1 = T_cam1_cam0 * Xc0), e.g. StereoResult::T_cam1_cam0.
// make_cam{0,1}: the same factories passed to calibrateStereo (rebuild the distorted models).
inline StereoRectification computeStereoRectification(
    const Eigen::VectorXd& intrinsics0, const Eigen::VectorXd& intrinsics1,
    const Sophus::SE3d& T_cam1_cam0, int width, int height, const CameraFactory& make_cam0,
    const CameraFactory& make_cam1) {
  const Eigen::Matrix3d R = T_cam1_cam0.rotationMatrix();
  const Eigen::Vector3d t = T_cam1_cam0.translation();
  const double b = t.norm();
  if (b < 1e-9)
    throw std::runtime_error("computeStereoRectification: degenerate (near-zero) baseline");

  // Even half-rotation split: r0 = exp(+w/2) for cam0, r1 = exp(-w/2) for cam1 (r1*R == r0).
  const Eigen::Vector3d w = Sophus::SO3d(R).log();
  const Eigen::Matrix3d r0 = Sophus::SO3d::exp(0.5 * w).matrix();
  const Eigen::Matrix3d r1 = r0.transpose();  // = exp(-w/2)

  // Baseline direction in the shared mid-frame.
  const Eigen::Vector3d t_c0 = -R.transpose() * t;  // cam1 origin expressed in cam0 frame
  const Eigen::Vector3d base = r0 * t_c0;
  const Eigen::Vector3d zaxis(0.0, 0.0, 1.0);
  if (zaxis.cross(base.normalized()).norm() < 1e-9)
    throw std::runtime_error(
        "computeStereoRectification: baseline parallel to the optical axis (forward/aft stereo)");

  // Common rectifying basis: x = baseline, y perpendicular to baseline & old z, z = x cross y.
  const Eigen::Vector3d e1 = base.normalized();
  const Eigen::Vector3d e2 = zaxis.cross(e1).normalized();
  const Eigen::Vector3d e3 = e1.cross(e2);
  Eigen::Matrix3d R_common;
  R_common.row(0) = e1.transpose();
  R_common.row(1) = e2.transpose();
  R_common.row(2) = e3.transpose();

  StereoRectification out;
  out.R0 = R_common * r0;  // cam0 physical -> rectified
  out.R1 = R_common * r1;  // cam1 physical -> rectified  (R0 == R1*R)
  out.baseline = b;

  // Shared rectified intrinsics: one focal (square pixels, canonical Q), averaged principal point.
  const double f = 0.25 * (intrinsics0[0] + intrinsics0[1] + intrinsics1[0] + intrinsics1[1]);
  const double cx = 0.5 * (intrinsics0[2] + intrinsics1[2]);
  const double cy = 0.5 * (intrinsics0[3] + intrinsics1[3]);
  out.K_rect = {f, f, cx, cy};

  // ROS stereo convention: left P = K[I|0]; right P carries the baseline as Tx = -fx*b.
  out.P0 = {f, 0, cx, 0, 0, f, cy, 0, 0, 0, 1, 0};
  out.P1 = {f, 0, cx, -f * b, 0, f, cy, 0, 0, 0, 1, 0};
  // Disparity-to-depth: (u,v,d,1) -> (X,Y,Z,W) with Z = f*b/d.
  out.Q = {1, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 0, f, 0, 0, 1.0 / b, 0};

  const std::unique_ptr<CameraModel> cam0 = make_cam0(intrinsics0);
  const std::unique_ptr<CameraModel> cam1 = make_cam1(intrinsics1);
  out.warp0 = apply::generateWarpMap(*cam0, out.K_rect, width, height, out.R0);
  out.warp1 = apply::generateWarpMap(*cam1, out.K_rect, width, height, out.R1);
  return out;
}

// Fill the rectification (R) + projection (P) fields of a ROS CameraInfo pair from a computed
// rectification. base0/base1 carry the RAW per-camera K/D/width/height/name (ROS convention: K/D
// describe the unrectified camera; R/P describe the rectified output). Only R and P are overwritten.
inline std::pair<io::CameraInfo, io::CameraInfo> stereoRectificationToCameraInfo(
    const StereoRectification& rect, const io::CameraInfo& base0, const io::CameraInfo& base1) {
  auto rowMajor = [](const Eigen::Matrix3d& M) {
    std::array<double, 9> a{};
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) a[static_cast<std::size_t>(r) * 3 + c] = M(r, c);
    return a;
  };
  io::CameraInfo ci0 = base0, ci1 = base1;
  ci0.R = rowMajor(rect.R0);
  ci1.R = rowMajor(rect.R1);
  ci0.P = rect.P0;
  ci1.P = rect.P1;
  return {ci0, ci1};
}

}  // namespace calibforge
