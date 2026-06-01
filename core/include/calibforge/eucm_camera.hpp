#pragma once
//
// CalibForge core — Extended Unified Camera Model (EUCM; Khomutenko et al. 2016).
// The second model nvTorchCam/Kornia/PyTorch3D do NOT ship (docs/RESEARCH.md Theme 4).
// Math RE-IMPLEMENTED; basalt-headers (BSD-3) is a cross-check reference only.
//
// Params: fx, fy, cx, cy, alpha, beta.
//   d = sqrt(beta*(X^2+Y^2) + Z^2)
//   denom = alpha*d + (1-alpha)*Z
//   u = fx*X/denom + cx ;  v = fy*Y/denom + cy
// Closed-form unprojection (no Newton).

#include <cmath>
#include <string>

#include "calibforge/camera_model.hpp"

namespace calibforge {

class EUCMCamera : public CameraModel {
 public:
  EUCMCamera(double fx, double fy, double cx, double cy, double alpha, double beta)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy), alpha_(alpha), beta_(beta) {}

  Vec2 project(const Vec3& p) const override {
    const double X = p[0], Y = p[1], Z = p[2];
    const double d = std::sqrt(beta_ * (X * X + Y * Y) + Z * Z);
    const double denom = alpha_ * d + (1.0 - alpha_) * Z;
    return Vec2{fx_ * X / denom + cx_, fy_ * Y / denom + cy_};
  }

  bool projectValid(const Vec3& p) const {
    const double X = p[0], Y = p[1], Z = p[2];
    const double d = std::sqrt(beta_ * (X * X + Y * Y) + Z * Z);
    const double w = (alpha_ <= 0.5) ? alpha_ / (1.0 - alpha_) : (1.0 - alpha_) / alpha_;
    return Z > -w * d;
  }

  Vec3 unproject(const Vec2& px) const override {
    const double mx = (px[0] - cx_) / fx_;
    const double my = (px[1] - cy_) / fy_;
    const double r2 = mx * mx + my * my;
    const double mz = (1.0 - beta_ * alpha_ * alpha_ * r2) /
                      (alpha_ * std::sqrt(1.0 - (2.0 * alpha_ - 1.0) * beta_ * r2) + 1.0 - alpha_);
    const double n = std::sqrt(mx * mx + my * my + mz * mz);
    return Vec3{mx / n, my / n, mz / n};
  }

  // d(u,v)/d(fx,fy,cx,cy,alpha,beta), 2x6 row-major.
  Jacobian projectJacobianWrtParams(const Vec3& p) const override {
    const double X = p[0], Y = p[1], Z = p[2];
    const double rho2 = X * X + Y * Y;
    const double d = std::sqrt(beta_ * rho2 + Z * Z);
    const double denom = alpha_ * d + (1.0 - alpha_) * Z;
    const double inv = 1.0 / denom, inv2 = inv * inv;
    const double dden_dalpha = d - Z;
    const double dden_dbeta = alpha_ * rho2 / (2.0 * d);
    Jacobian J;
    J.rows = 2;
    J.cols = 6;
    J.data.assign(12, 0.0);
    // cols: 0=fx 1=fy 2=cx 3=cy 4=alpha 5=beta
    J.data[0] = X * inv;
    J.data[2] = 1.0;
    J.data[4] = -fx_ * X * inv2 * dden_dalpha;
    J.data[5] = -fx_ * X * inv2 * dden_dbeta;
    J.data[6 + 1] = Y * inv;
    J.data[6 + 3] = 1.0;
    J.data[6 + 4] = -fy_ * Y * inv2 * dden_dalpha;
    J.data[6 + 5] = -fy_ * Y * inv2 * dden_dbeta;
    return J;
  }

  // d(u,v)/d(X,Y,Z), 2x3 row-major.
  Jacobian projectJacobianWrtPoint(const Vec3& p) const override {
    const double X = p[0], Y = p[1], Z = p[2];
    const double d = std::sqrt(beta_ * (X * X + Y * Y) + Z * Z);
    const double denom = alpha_ * d + (1.0 - alpha_) * Z;
    const double inv2 = 1.0 / (denom * denom);
    const double dd_dX = beta_ * X / d, dd_dY = beta_ * Y / d, dd_dZ = Z / d;
    const double dden_dX = alpha_ * dd_dX;
    const double dden_dY = alpha_ * dd_dY;
    const double dden_dZ = alpha_ * dd_dZ + (1.0 - alpha_);
    Jacobian J;
    J.rows = 2;
    J.cols = 3;
    J.data = {fx_ * (denom - X * dden_dX) * inv2, fx_ * (-X * dden_dY) * inv2, fx_ * (-X * dden_dZ) * inv2,
              fy_ * (-Y * dden_dX) * inv2, fy_ * (denom - Y * dden_dY) * inv2, fy_ * (-Y * dden_dZ) * inv2};
    return J;
  }

  std::size_t numParams() const override { return 6; }
  std::string name() const override { return "eucm"; }

 private:
  double fx_, fy_, cx_, cy_, alpha_, beta_;
};

}  // namespace calibforge
