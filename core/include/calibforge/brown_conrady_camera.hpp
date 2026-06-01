#pragma once
//
// CalibForge core — Brown-Conrady (OpenCV "plumb-bob") camera model (header-only).
//
// Params: fx, fy, cx, cy, k1, k2, p1, p2, k3  (4 intrinsic + 3 radial + 2 tangential).
// project: normalize p=(X,Y,Z) -> (x,y); apply radial+tangential distortion; affine.
//   r2 = x^2+y^2 ; radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
//   x_d = x*radial + 2 p1 x y + p2 (r2 + 2x^2)
//   y_d = y*radial + p1 (r2 + 2y^2) + 2 p2 x y
//   u = fx x_d + cx ; v = fy y_d + cy

#include <cmath>
#include <stdexcept>
#include <string>

#include "calibforge/camera_model.hpp"

namespace calibforge {

class BrownConradyCamera : public CameraModel {
 public:
  BrownConradyCamera(double fx, double fy, double cx, double cy,
                     double k1, double k2, double p1, double p2, double k3)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
        k1_(k1), k2_(k2), p1_(p1), p2_(p2), k3_(k3) {}

  Vec2 project(const Vec3& p) const override {
    const double x = p[0] / p[2];
    const double y = p[1] / p[2];
    double xd, yd;
    distort(x, y, xd, yd);
    return Vec2{fx_ * xd + cx_, fy_ * yd + cy_};
  }

  // Iterative undistortion (OpenCV-style fixed-point): recover normalized (x,y)
  // from distorted coords, then return the unit ray. Converges for moderate distortion.
  Vec3 unproject(const Vec2& px) const override {
    const double xd = (px[0] - cx_) / fx_;
    const double yd = (px[1] - cy_) / fy_;
    double x = xd, y = yd;  // initial guess = distorted coords
    for (int iter = 0; iter < kUndistortIters; ++iter) {
      const double r2 = x * x + y * y;
      const double radial = 1.0 + r2 * (k1_ + r2 * (k2_ + r2 * k3_));
      const double dx = 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
      const double dy = p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;
      x = (xd - dx) / radial;
      y = (yd - dy) / radial;
    }
    const double z = 1.0;
    const double n = std::sqrt(x * x + y * y + z * z);
    return Vec3{x / n, y / n, z / n};
  }

  // d(u,v)/d(fx,fy,cx,cy,k1,k2,p1,p2,k3), 2x9 row-major.
  Jacobian projectJacobianWrtParams(const Vec3& p) const override {
    const double x = p[0] / p[2];
    const double y = p[1] / p[2];
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    double xd, yd;
    distort(x, y, xd, yd);

    Jacobian J;
    J.rows = 2;
    J.cols = 9;
    J.data.assign(18, 0.0);
    // cols: 0=fx 1=fy 2=cx 3=cy 4=k1 5=k2 6=p1 7=p2 8=k3
    // row u = fx*x_d + cx
    J.data[0] = xd;                         // du/dfx
    J.data[2] = 1.0;                        // du/dcx
    J.data[4] = fx_ * x * r2;               // du/dk1
    J.data[5] = fx_ * x * r4;               // du/dk2
    J.data[6] = fx_ * 2.0 * x * y;          // du/dp1
    J.data[7] = fx_ * (r2 + 2.0 * x * x);   // du/dp2
    J.data[8] = fx_ * x * r6;               // du/dk3
    // row v = fy*y_d + cy
    J.data[9 + 1] = yd;                     // dv/dfy
    J.data[9 + 3] = 1.0;                    // dv/dcy
    J.data[9 + 4] = fy_ * y * r2;           // dv/dk1
    J.data[9 + 5] = fy_ * y * r4;           // dv/dk2
    J.data[9 + 6] = fy_ * (r2 + 2.0 * y * y); // dv/dp1
    J.data[9 + 7] = fy_ * 2.0 * x * y;      // dv/dp2
    J.data[9 + 8] = fy_ * y * r6;           // dv/dk3
    return J;
  }

  // d(u,v)/d(X,Y,Z), 2x3 row-major: chain rule pixel <- distort(x,y) <- (x,y)=(X/Z,Y/Z).
  Jacobian projectJacobianWrtPoint(const Vec3& p) const override {
    const double invZ = 1.0 / p[2];
    const double x = p[0] * invZ;
    const double y = p[1] * invZ;
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double radial = 1.0 + r2 * (k1_ + r2 * (k2_ + r2 * k3_));
    const double D = k1_ + 2.0 * k2_ * r2 + 3.0 * k3_ * r4;  // d(radial)/d(r2)

    // Distortion Jacobian d(x_d,y_d)/d(x,y) (2x2).
    const double dxd_dx = radial + 2.0 * x * x * D + 2.0 * p1_ * y + 6.0 * p2_ * x;
    const double dxd_dy = 2.0 * x * y * D + 2.0 * p1_ * x + 2.0 * p2_ * y;
    const double dyd_dx = dxd_dy;  // symmetric off-diagonal
    const double dyd_dy = radial + 2.0 * y * y * D + 6.0 * p1_ * y + 2.0 * p2_ * x;

    // d(x,y)/d(X,Y,Z): [1/Z, 0, -x/Z ; 0, 1/Z, -y/Z].
    const double dx_dX = invZ, dx_dZ = -x * invZ;
    const double dy_dY = invZ, dy_dZ = -y * invZ;

    Jacobian J;
    J.rows = 2;
    J.cols = 3;
    J.data = {fx_ * (dxd_dx * dx_dX),                 // du/dX
              fx_ * (dxd_dy * dy_dY),                 // du/dY
              fx_ * (dxd_dx * dx_dZ + dxd_dy * dy_dZ),  // du/dZ
              fy_ * (dyd_dx * dx_dX),                 // dv/dX
              fy_ * (dyd_dy * dy_dY),                 // dv/dY
              fy_ * (dyd_dx * dx_dZ + dyd_dy * dy_dZ)};  // dv/dZ
    return J;
  }

  std::size_t numParams() const override { return 9; }
  std::string name() const override { return "brown_conrady"; }

  // Intrinsics in constructor order (fx,fy,cx,cy,k1,k2,p1,p2,k3) — for io/apply interop.
  std::array<double, 9> params() const {
    return {fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_, k3_};
  }

 private:
  // Apply distortion to normalized image coords.
  void distort(double x, double y, double& xd, double& yd) const {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + r2 * (k1_ + r2 * (k2_ + r2 * k3_));
    xd = x * radial + 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    yd = y * radial + p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;
  }

  static constexpr int kUndistortIters = 20;

  double fx_, fy_, cx_, cy_;
  double k1_, k2_, p1_, p2_, k3_;
};

}  // namespace calibforge
