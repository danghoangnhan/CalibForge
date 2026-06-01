#pragma once
//
// CalibForge solve — stereo reprojection ResidualBlock for the second camera.
//
// cam1 observes a board point through TWO transforms: Xc1 = T_c1c0 * (T_wc0 * Xw), where
// T_wc0 is the per-view world->cam0 pose and T_c1c0 is the shared cam0->cam1 extrinsic
// (Xc1 = T_c1c0 * Xc0). Connects [intrinsics1, T_c1c0, T_wc0]. Analytic minimal Jacobians
// (the composition the issue calls out), all reusing the single-camera building blocks:
//   d/dintr1 = projectJacobianWrtParams(Xc1)
//   d/dT_c1c0 = dpix * [R01 | -R01 [Xc0]_x]
//   d/dT_wc0  = dpix * R01 * [Rwc0 | -Rwc0 [Xw]_x]
// where dpix = projectJacobianWrtPoint(Xc1), R01 = T_c1c0.R, Rwc0 = T_wc0.R.

#include <array>
#include <functional>
#include <memory>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/residual_block.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

class StereoReprojectionResidual : public ResidualBlock {
 public:
  using CameraFactory = std::function<std::unique_ptr<CameraModel>(const Eigen::VectorXd&)>;

  StereoReprojectionResidual(CameraFactory make_camera1, int num_intrinsics1,
                             const Vec3& object_point, const Vec2& image_point)
      : make_(std::move(make_camera1)),
        nin_(num_intrinsics1),
        Xw_(object_point[0], object_point[1], object_point[2]),
        obs_{image_point[0], image_point[1]} {}

  ResidualType type() const override { return ResidualType::Reprojection; }
  std::size_t residualDim() const override { return 2; }

  void evaluate(double const* const* params, double* residual,
                double** jacobians) const override {
    // params[0]=intrinsics1(nin), params[1]=T_c1c0(7), params[2]=T_wc0(7)
    const Eigen::VectorXd intr = Eigen::Map<const Eigen::VectorXd>(params[0], nin_);
    const Sophus::SE3d T01 = SE3Param::load(params[1]);
    const Sophus::SE3d Twc0 = SE3Param::load(params[2]);
    const Eigen::Matrix3d R01 = T01.rotationMatrix();
    const Eigen::Matrix3d Rwc0 = Twc0.rotationMatrix();
    const Eigen::Vector3d Xc0 = Twc0 * Xw_;
    const Eigen::Vector3d Xc1 = T01 * Xc0;
    const std::unique_ptr<CameraModel> cam = make_(intr);
    const Vec3 pc{Xc1.x(), Xc1.y(), Xc1.z()};
    const Vec2 px = cam->project(pc);
    residual[0] = px[0] - obs_[0];
    residual[1] = px[1] - obs_[1];
    if (!jacobians) return;

    Eigen::Matrix<double, 2, 3> dpix;
    if (jacobians[0] || jacobians[1] || jacobians[2]) {
      const Jacobian Jp = cam->projectJacobianWrtPoint(pc);
      dpix << Jp.data[0], Jp.data[1], Jp.data[2],
              Jp.data[3], Jp.data[4], Jp.data[5];
    }
    if (jacobians[0]) {  // 2 x nin (intrinsics1)
      const Jacobian Ji = cam->projectJacobianWrtParams(pc);
      for (int c = 0; c < nin_; ++c) {
        jacobians[0][c] = Ji.data[c];
        jacobians[0][nin_ + c] = Ji.data[nin_ + c];
      }
    }
    if (jacobians[1]) {  // 2 x 6 (T_c1c0): point is Xc0, rotation is R01
      Eigen::Matrix<double, 3, 6> dXc1;
      dXc1.leftCols<3>() = R01;
      dXc1.rightCols<3>() = -R01 * skew(Xc0);
      writePose(jacobians[1], dpix * dXc1);
    }
    if (jacobians[2]) {  // 2 x 6 (T_wc0): compose through R01
      Eigen::Matrix<double, 3, 6> dXc0;
      dXc0.leftCols<3>() = Rwc0;
      dXc0.rightCols<3>() = -Rwc0 * skew(Xw_);
      writePose(jacobians[2], dpix * (R01 * dXc0));
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

  CameraFactory make_;
  int nin_;
  Eigen::Vector3d Xw_;
  std::array<double, 2> obs_;
};

}  // namespace calibforge
