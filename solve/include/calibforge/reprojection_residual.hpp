#pragma once
//
// CalibForge solve — concrete reprojection ResidualBlock (camera model + SE(3) pose).
//
// Connects [intrinsics block (nin, Euclidean), pose block (7-double SE3)]; the residual
// is project(T * Xw) - observed (a 2-vector). Analytic MINIMAL Jacobians (the same
// hot-path Jacobians calibrate_single validated, docs/RESEARCH.md Theme 4):
//   jacobians[0] = d(pixel)/d(intrinsics)      2 x nin  = CameraModel::projectJacobianWrtParams
//   jacobians[1] = d(pixel)/d(pose tangent)    2 x 6    = projJacWrtPoint * [R | -R [Xw]_x]
// where [R | -R [Xw]_x] = d(T*Xw)/d(delta) for the right perturbation T <- T*exp(delta).

#include <array>
#include <functional>
#include <memory>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"   // Vec2, Vec3, CameraModel, Jacobian
#include "calibforge/manifold.hpp"       // SE3Param::load
#include "calibforge/residual_block.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

class ReprojectionResidual : public ResidualBlock {
 public:
  using CameraFactory = std::function<std::unique_ptr<CameraModel>(const Eigen::VectorXd&)>;

  ReprojectionResidual(CameraFactory make_camera, int num_intrinsics,
                       const Vec3& object_point, const Vec2& image_point)
      : make_camera_(std::move(make_camera)),
        nin_(num_intrinsics),
        Xw_(object_point[0], object_point[1], object_point[2]),
        obs_{image_point[0], image_point[1]} {}

  ResidualType type() const override { return ResidualType::Reprojection; }
  std::size_t residualDim() const override { return 2; }

  void evaluate(double const* const* params, double* residual,
                double** jacobians) const override {
    const Eigen::VectorXd intr = Eigen::Map<const Eigen::VectorXd>(params[0], nin_);
    const Sophus::SE3d T = SE3Param::load(params[1]);
    const Eigen::Matrix3d R = T.rotationMatrix();
    const Eigen::Vector3d Xc = T * Xw_;
    const std::unique_ptr<CameraModel> cam = make_camera_(intr);
    const Vec3 pc{Xc.x(), Xc.y(), Xc.z()};
    const Vec2 px = cam->project(pc);
    residual[0] = px[0] - obs_[0];
    residual[1] = px[1] - obs_[1];
    if (!jacobians) return;

    Eigen::Matrix<double, 2, 3> dpix;  // d(pixel)/d(point_cam), shared by both Jacobians
    if (jacobians[0] || jacobians[1]) {
      const Jacobian Jp = cam->projectJacobianWrtPoint(pc);
      dpix << Jp.data[0], Jp.data[1], Jp.data[2],
              Jp.data[3], Jp.data[4], Jp.data[5];
    }
    if (jacobians[0]) {  // 2 x nin
      const Jacobian Ji = cam->projectJacobianWrtParams(pc);
      for (int c = 0; c < nin_; ++c) {
        jacobians[0][c] = Ji.data[c];
        jacobians[0][nin_ + c] = Ji.data[nin_ + c];
      }
    }
    if (jacobians[1]) {  // 2 x 6
      Eigen::Matrix<double, 3, 6> dXc;
      dXc.leftCols<3>() = R;
      dXc.rightCols<3>() = -R * skew(Xw_);
      const Eigen::Matrix<double, 2, 6> Jpose = dpix * dXc;
      for (int c = 0; c < 6; ++c) {
        jacobians[1][c] = Jpose(0, c);
        jacobians[1][6 + c] = Jpose(1, c);
      }
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

  CameraFactory make_camera_;
  int nin_;
  Eigen::Vector3d Xw_;
  std::array<double, 2> obs_;
};

}  // namespace calibforge
