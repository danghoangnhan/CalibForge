#pragma once
//
// CalibForge io — Isaac Perceptor URDF emitter (header-only, dependency-free).
//
// Perceptor ingests a calibration URDF at /etc/nova/calibration/isaac_calibration.urdf with
// REP-103 frames rooted at base_link. We emit, per camera, a fixed joint base_link -> <name>
// carrying the mechanical extrinsic, then a second fixed joint <name> -> <name>_optical
// carrying the constant body->optical rotation (z-forward optical frame).
//
// Distortion is intentionally *not* embedded in the URDF: per Isaac Perceptor convention the
// URDF carries only the kinematic frame tree (base_link root, fixed joints, optical sub-frame
// per camera), and per-camera intrinsics + distortion travel alongside as a companion
// `CameraInfo` (see io/ros_camera_info.hpp). The frame_id stamped on both files lets the
// downstream pipeline join them. Linking the two is asserted by the
// `isaac_urdf_and_camera_info_share_frame_id` integration test.

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/frames.hpp"
#include "calibforge/ros_camera_info.hpp"
#include "sophus/se3.hpp"

namespace calibforge {
namespace io {

struct CameraExtrinsic {
  std::string name = "camera";
  Sophus::SE3d T_base_camera;  // base_link -> camera body frame (REP-103)
  CameraInfo info;             // intrinsics travel alongside
};

struct RigDescription {
  std::string base_frame = "base_link";
  std::vector<CameraExtrinsic> cameras;
};

namespace detail {
inline std::string g6(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return buf;
}
}  // namespace detail

inline std::string toIsaacUrdf(const RigDescription& rig) {
  std::ostringstream os;
  os << "<?xml version=\"1.0\"?>\n";
  os << "<robot name=\"calibforge_rig\">\n";
  os << "  <link name=\"" << rig.base_frame << "\"/>\n";

  const Eigen::Vector3d opt_rpy = rotationToRpy(bodyToOpticalRotation());

  for (const auto& cam : rig.cameras) {
    const Eigen::Vector3d t = cam.T_base_camera.translation();
    const Eigen::Vector3d rpy = rotationToRpy(cam.T_base_camera.rotationMatrix());

    os << "  <link name=\"" << cam.name << "\"/>\n";
    os << "  <joint name=\"" << rig.base_frame << "_to_" << cam.name << "\" type=\"fixed\">\n";
    os << "    <parent link=\"" << rig.base_frame << "\"/>\n";
    os << "    <child link=\"" << cam.name << "\"/>\n";
    os << "    <origin xyz=\"" << detail::g6(t.x()) << " " << detail::g6(t.y()) << " "
       << detail::g6(t.z()) << "\" rpy=\"" << detail::g6(rpy.x()) << " " << detail::g6(rpy.y())
       << " " << detail::g6(rpy.z()) << "\"/>\n";
    os << "  </joint>\n";

    os << "  <link name=\"" << cam.name << "_optical\"/>\n";
    os << "  <joint name=\"" << cam.name << "_to_optical\" type=\"fixed\">\n";
    os << "    <parent link=\"" << cam.name << "\"/>\n";
    os << "    <child link=\"" << cam.name << "_optical\"/>\n";
    os << "    <origin xyz=\"0 0 0\" rpy=\"" << detail::g6(opt_rpy.x()) << " "
       << detail::g6(opt_rpy.y()) << " " << detail::g6(opt_rpy.z()) << "\"/>\n";
    os << "  </joint>\n";
  }

  os << "</robot>\n";
  return os.str();
}

}  // namespace io
}  // namespace calibforge
