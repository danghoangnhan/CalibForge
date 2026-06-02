#pragma once
//
// CalibForge solve — hand-eye reprojection ResidualBlock (robot-world / hand-eye, AX=ZB).
//
// A camera (known intrinsics) on a moving robot observes a FIXED target. With the known
// gripper-in-base pose per view, the target point projects into the camera as
//   Xc = X^{-1} * A * Z * Xw,
// where X = T_gripper_cam (hand-eye), Z = T_base_target, and A = T_gripper_base (= the known
// T_base_gripper^{-1}). Connects the two unknown SE(3) blocks [X, Z]; the camera is fixed.
//
// Analytic minimal Jacobians (right perturbation X<-X exp(dx), Z<-Z exp(dz)):
//   d(Xc)/dX = [-I | [Xc]_x]                          (since X^{-1} -> exp(-dx) X^{-1})
//   d(Xc)/dZ = [R_M | -R_M [Xw]_x],  M = X^{-1} A Z
// then J = projectJacobianWrtPoint(Xc) * d(Xc)/d(.).

#include <array>
#include <memory>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/residual_block.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

class HandEyeResidual : public ResidualBlock {
 public:
  // `camera` and `T_gripper_base` (= known T_base_gripper^{-1}) are fixed per residual.
  HandEyeResidual(const CameraModel* camera, const Sophus::SE3d& T_gripper_base,
                  const Vec3& object_point, const Vec2& image_point)
      : cam_(camera),
        A_(T_gripper_base),
        Xw_(object_point[0], object_point[1], object_point[2]),
        obs_{image_point[0], image_point[1]} {}

  ResidualType type() const override { return ResidualType::Reprojection; }
  std::size_t residualDim() const override { return 2; }

  void evaluate(double const* const* params, double* residual,
                double** jacobians) const override {
    const Sophus::SE3d X = SE3Param::load(params[0]);
    const Sophus::SE3d Z = SE3Param::load(params[1]);
    const Sophus::SE3d M = X.inverse() * A_ * Z;   // Xc = M * exp(dz) * Xw at the linearization
    const Eigen::Vector3d Xc = M * Xw_;
    const Vec3 pc{Xc.x(), Xc.y(), Xc.z()};
    const Vec2 px = cam_->project(pc);
    residual[0] = px[0] - obs_[0];
    residual[1] = px[1] - obs_[1];
    if (!jacobians) return;

    Eigen::Matrix<double, 2, 3> dpix;
    {
      const Jacobian Jp = cam_->projectJacobianWrtPoint(pc);
      dpix << Jp.data[0], Jp.data[1], Jp.data[2],
              Jp.data[3], Jp.data[4], Jp.data[5];
    }
    if (jacobians[0]) {  // d(Xc)/dX = [-I | [Xc]_x]
      Eigen::Matrix<double, 3, 6> dX;
      dX.leftCols<3>() = -Eigen::Matrix3d::Identity();
      dX.rightCols<3>() = skew(Xc);
      writePose(jacobians[0], dpix * dX);
    }
    if (jacobians[1]) {  // d(Xc)/dZ = [R_M | -R_M [Xw]_x]
      const Eigen::Matrix3d RM = M.rotationMatrix();
      Eigen::Matrix<double, 3, 6> dZ;
      dZ.leftCols<3>() = RM;
      dZ.rightCols<3>() = -RM * skew(Xw_);
      writePose(jacobians[1], dpix * dZ);
    }
  }

 private:
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return S;
  }
  static void writePose(double* out, const Eigen::Matrix<double, 2, 6>& J) {
    for (int c = 0; c < 6; ++c) {
      out[c] = J(0, c);
      out[6 + c] = J(1, c);
    }
  }

  const CameraModel* cam_;
  Sophus::SE3d A_;
  Eigen::Vector3d Xw_;
  std::array<double, 2> obs_;
};

}  // namespace calibforge
