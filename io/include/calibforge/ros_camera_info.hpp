#pragma once
//
// CalibForge io — ROS sensor_msgs/CameraInfo emitter (header-only, dependency-free).
//
// Emits the camera_calibration_parsers YAML dialect. Distortion-model name follows the
// coefficient count: 5 -> plumb_bob, 8 -> rational_polynomial, 4 (KB) -> equidistant.
// For a monocular calibrated camera the projection matrix P is [K | 0].

#include <array>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "calibforge/brown_conrady_camera.hpp"

namespace calibforge {
namespace io {

struct CameraInfo {
  std::string camera_name = "camera";
  int width = 0;
  int height = 0;
  std::string distortion_model = "plumb_bob";
  std::vector<double> D;                                   // model-dependent length
  std::array<double, 9> K = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // 3x3 intrinsics, row-major
  std::array<double, 9> R = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // rectification (identity = monocular)
  std::array<double, 12> P = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // 3x4 projection
};

inline CameraInfo brownConradyToCameraInfo(const BrownConradyCamera& cam, int w, int h,
                                           const std::string& name = "camera") {
  const std::array<double, 9> q = cam.params();  // fx,fy,cx,cy,k1,k2,p1,p2,k3
  CameraInfo ci;
  ci.camera_name = name;
  ci.width = w;
  ci.height = h;
  ci.distortion_model = "plumb_bob";
  ci.D = {q[4], q[5], q[6], q[7], q[8]};  // ROS plumb_bob order: k1,k2,p1,p2,k3
  ci.K = {q[0], 0.0, q[2], 0.0, q[1], q[3], 0.0, 0.0, 1.0};
  ci.P = {q[0], 0.0, q[2], 0.0, 0.0, q[1], q[3], 0.0, 0.0, 0.0, 1.0, 0.0};
  return ci;
}

namespace detail {
inline std::string g(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  return buf;
}
template <typename It>
inline std::string list(It begin, It end) {
  std::ostringstream os;
  os << "[";
  for (It it = begin; it != end; ++it) {
    if (it != begin) os << ", ";
    os << g(*it);
  }
  os << "]";
  return os.str();
}
}  // namespace detail

inline std::string toCameraInfoYaml(const CameraInfo& ci) {
  std::ostringstream os;
  os << "image_width: " << ci.width << "\n";
  os << "image_height: " << ci.height << "\n";
  os << "camera_name: " << ci.camera_name << "\n";
  os << "camera_matrix:\n  rows: 3\n  cols: 3\n  data: "
     << detail::list(ci.K.begin(), ci.K.end()) << "\n";
  os << "distortion_model: " << ci.distortion_model << "\n";
  os << "distortion_coefficients:\n  rows: 1\n  cols: " << ci.D.size()
     << "\n  data: " << detail::list(ci.D.begin(), ci.D.end()) << "\n";
  os << "rectification_matrix:\n  rows: 3\n  cols: 3\n  data: "
     << detail::list(ci.R.begin(), ci.R.end()) << "\n";
  os << "projection_matrix:\n  rows: 3\n  cols: 4\n  data: "
     << detail::list(ci.P.begin(), ci.P.end()) << "\n";
  return os.str();
}

}  // namespace io
}  // namespace calibforge
