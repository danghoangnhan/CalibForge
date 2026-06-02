#pragma once
//
// CalibForge solve — rolling-shutter reprojection ResidualBlock (readout-time calibration).
//
// A CMOS sensor reads rows sequentially: a point captured at normalized row time f in [0,1]
// (f = observed_row / image_height) was imaged at offset s = f * t_r within the frame, where
// t_r is the calibrated whole-image readout time (t_r=0 => global shutter; MVIS/Kalibr model).
// Under a first-order constant-velocity model the world->cam point at offset s is
//   Xc = T_i * (Xw + s*(rho - [Xw]_x * omega)),   v_i = (rho, omega) the per-frame velocity twist.
//
// Connects [intrinsics (nin), pose T_i (SE3), velocity v_i (R^6), t_r (R^1, shared)]. Analytic
// minimal Jacobians (dpix = projectJacobianWrtPoint(Xc), R = T_i.R):
//   d/dintr = projectJacobianWrtParams(Xc)
//   d/dT_i  = dpix * R * [I | -[P]_x],      P = Xw + s*(rho - [Xw]_x omega)
//   d/dv_i  = dpix * R * [s*I | -s*[Xw]_x]
//   d/dt_r  = dpix * R * f*(rho - [Xw]_x omega)

#include <array>
#include <functional>
#include <memory>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/residual_block.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

class RollingShutterResidual : public ResidualBlock {
 public:
  using CameraFactory = std::function<std::unique_ptr<CameraModel>(const Eigen::VectorXd&)>;

  // row_frac = observed_row / image_height in [0,1] (the capture-time fraction within the frame).
  RollingShutterResidual(CameraFactory make_camera, int num_intrinsics, const Vec3& object_point,
                         const Vec2& image_point, double row_frac)
      : make_(std::move(make_camera)),
        nin_(num_intrinsics),
        Xw_(object_point[0], object_point[1], object_point[2]),
        obs_{image_point[0], image_point[1]},
        f_(row_frac) {}

  ResidualType type() const override { return ResidualType::RollingShutter; }
  std::size_t residualDim() const override { return 2; }

  void evaluate(double const* const* params, double* residual,
                double** jacobians) const override {
    // params[0]=intrinsics(nin), [1]=pose(7), [2]=velocity(6 rho,omega), [3]=t_r(1)
    const Eigen::VectorXd intr = Eigen::Map<const Eigen::VectorXd>(params[0], nin_);
    const Sophus::SE3d T = SE3Param::load(params[1]);
    const Eigen::Matrix3d R = T.rotationMatrix();
    Eigen::Vector3d rho(params[2][0], params[2][1], params[2][2]);
    Eigen::Vector3d omega(params[2][3], params[2][4], params[2][5]);
    const double tr = params[3][0];
    const double s = f_ * tr;

    const Eigen::Vector3d disp = rho - skew(Xw_) * omega;  // rho + omega x Xw  (first-order)
    const Eigen::Vector3d P = Xw_ + s * disp;              // displaced point (world frame)
    const Eigen::Vector3d Xc = T * P;
    const std::unique_ptr<CameraModel> cam = make_(intr);
    const Vec3 pc{Xc.x(), Xc.y(), Xc.z()};
    const Vec2 px = cam->project(pc);
    residual[0] = px[0] - obs_[0];
    residual[1] = px[1] - obs_[1];
    if (!jacobians) return;

    Eigen::Matrix<double, 2, 3> dpix;
    {
      const Jacobian Jp = cam->projectJacobianWrtPoint(pc);
      dpix << Jp.data[0], Jp.data[1], Jp.data[2],
              Jp.data[3], Jp.data[4], Jp.data[5];
    }
    const Eigen::Matrix<double, 2, 3> dpixR = dpix * R;  // d(pixel)/d(P) reused below
    if (jacobians[0]) {
      const Jacobian Ji = cam->projectJacobianWrtParams(pc);
      for (int c = 0; c < nin_; ++c) {
        jacobians[0][c] = Ji.data[c];
        jacobians[0][nin_ + c] = Ji.data[nin_ + c];
      }
    }
    if (jacobians[1]) {  // d/dT_i = dpix * R * [I | -[P]_x]
      Eigen::Matrix<double, 3, 6> dT;
      dT.leftCols<3>() = Eigen::Matrix3d::Identity();
      dT.rightCols<3>() = -skew(P);
      write(jacobians[1], dpixR * dT, 6);
    }
    if (jacobians[2]) {  // d/dv_i = dpix * R * [s*I | -s*[Xw]_x]
      Eigen::Matrix<double, 3, 6> dV;
      dV.leftCols<3>() = s * Eigen::Matrix3d::Identity();
      dV.rightCols<3>() = -s * skew(Xw_);
      write(jacobians[2], dpixR * dV, 6);
    }
    if (jacobians[3]) {  // d/dt_r = dpix * R * f*disp
      const Eigen::Vector2d g = dpixR * (f_ * disp);
      jacobians[3][0] = g[0];
      jacobians[3][1] = g[1];
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
  static void write(double* out, const Eigen::Matrix<double, 2, 6>& J, int cols) {
    for (int c = 0; c < cols; ++c) {
      out[c] = J(0, c);
      out[cols + c] = J(1, c);
    }
  }

  CameraFactory make_;
  int nin_;
  Eigen::Vector3d Xw_;
  std::array<double, 2> obs_;
  double f_;
};

}  // namespace calibforge
