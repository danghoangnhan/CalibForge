#pragma once
//
// CalibForge apply — undistortion warp-map generator (header-only, CPU; the VPI-LDC table).
//
// For each OUTPUT (undistorted/rectified) pixel we form the normalized ray of the target
// pinhole (out_K) and PROJECT it through the distorted camera model to get the SOURCE
// (distorted) pixel. That output->source table is exactly what VPI LDC / a remap consumes.
// Uses forward project() only (no Newton), so it is exact and model-agnostic. We build the
// ray from out_K analytically rather than CameraModel::unproject (which returns a unit ray,
// not a normalized image point) to avoid a normalization mismatch.
//
// STEREO RECTIFICATION: pass a per-camera rectifying rotation R_rect that maps a physical-camera
// point into the rectified frame (X_rect = R_rect * X_phys). The OUTPUT pixel is in the rectified
// frame, so its ray is rotated BACK into the physical frame (X_phys = R_rect^T * X_rect) before
// projecting through the distorted model. R_rect = identity (default) reproduces plain undistort.

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"

namespace calibforge {
namespace apply {

// Dense remap table: src[y*width+x] = the source pixel an output pixel (x,y) samples from.
struct WarpMap {
  int width = 0;
  int height = 0;
  std::vector<Vec2> src;

  Vec2 at(int x, int y) const { return src[static_cast<std::size_t>(y) * width + x]; }
};

// out_K = {fx, fy, cx, cy} of the desired undistorted/rectified image (often the input intrinsics).
// R_rect: per-camera rectifying rotation (physical -> rectified); identity for plain undistortion.
inline WarpMap generateWarpMap(const CameraModel& distorted_cam,
                               const std::array<double, 4>& out_K, int out_w, int out_h,
                               const Eigen::Matrix3d& R_rect = Eigen::Matrix3d::Identity()) {
  const double fx = out_K[0], fy = out_K[1], cx = out_K[2], cy = out_K[3];
  const Eigen::Matrix3d Rt = R_rect.transpose();  // rectified ray -> physical-camera ray
  WarpMap m;
  m.width = out_w;
  m.height = out_h;
  m.src.resize(static_cast<std::size_t>(out_w) * out_h);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const double xn = (x - cx) / fx;
      const double yn = (y - cy) / fy;
      const Eigen::Vector3d d = Rt * Eigen::Vector3d(xn, yn, 1.0);
      m.src[static_cast<std::size_t>(y) * out_w + x] =
          distorted_cam.project(Vec3{d.x(), d.y(), d.z()});
    }
  }
  return m;
}

}  // namespace apply
}  // namespace calibforge
