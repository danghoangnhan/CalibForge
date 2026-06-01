#pragma once
//
// CalibForge io — REP-103 frame conventions and URDF rpy helpers (header-only).
//
// ROS/Isaac body frames are REP-103: x-forward, y-left, z-up. A camera's OPTICAL frame is
// z-forward, x-right, y-down. The fixed body->optical rotation is therefore
//   R_body_optical = [ 0  0  1 ;  -1  0  0 ;  0 -1  0 ]  (x_opt=-y_body, y_opt=-z_body, z_opt=+x_body)
// which in URDF fixed-axis roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)) is (-pi/2, 0, -pi/2).

#include <cmath>

#include <Eigen/Dense>

namespace calibforge {
namespace io {

inline Eigen::Matrix3d bodyToOpticalRotation() {
  Eigen::Matrix3d R;
  R << 0, 0, 1,
       -1, 0, 0,
       0, -1, 0;
  return R;
}

// URDF rpy (fixed-axis XYZ: R = Rz(yaw) Ry(pitch) Rx(roll)) -> rotation matrix.
inline Eigen::Matrix3d rpyToRotation(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

// Inverse: extract (roll, pitch, yaw) from a rotation, matching the URDF convention above.
inline Eigen::Vector3d rotationToRpy(const Eigen::Matrix3d& R) {
  const double pitch = std::atan2(-R(2, 0), std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  const double roll = std::atan2(R(2, 1), R(2, 2));
  return Eigen::Vector3d(roll, pitch, yaw);
}

}  // namespace io
}  // namespace calibforge
