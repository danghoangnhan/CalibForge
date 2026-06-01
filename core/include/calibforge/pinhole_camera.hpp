#pragma once
//
// CalibForge core — pinhole camera model (header-only).
//
// Intrinsics: fx, fy, cx, cy.  project: p=(X,Y,Z) -> (fx*X/Z + cx, fy*Y/Z + cy).
// Header-only so the same math compiles host-side and (later) __host__ __device__.

#include <cmath>
#include <stdexcept>
#include <string>

#include "calibforge/camera_model.hpp"

namespace calibforge {

class PinholeCamera : public CameraModel {
 public:
  PinholeCamera(double fx, double fy, double cx, double cy)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

  Vec2 project(const Vec3& p) const override {
    return Vec2{fx_ * p[0] / p[2] + cx_, fy_ * p[1] / p[2] + cy_};
  }

  Vec3 unproject(const Vec2& px) const override {
    const double x = (px[0] - cx_) / fx_;
    const double y = (px[1] - cy_) / fy_;
    const double z = 1.0;
    const double n = std::sqrt(x * x + y * y + z * z);
    return Vec3{x / n, y / n, z / n};
  }

  // d(u,v)/d(fx,fy,cx,cy), 2x4 row-major.
  //   u = fx*X/Z + cx  ->  [X/Z, 0,   1, 0]
  //   v = fy*Y/Z + cy  ->  [0,   Y/Z, 0, 1]
  Jacobian projectJacobianWrtParams(const Vec3& p) const override {
    const double xz = p[0] / p[2];
    const double yz = p[1] / p[2];
    Jacobian J;
    J.rows = 2;
    J.cols = 4;
    J.data = {xz, 0.0, 1.0, 0.0,
              0.0, yz, 0.0, 1.0};
    return J;
  }

  // d(u,v)/d(X,Y,Z), 2x3 row-major.
  //   u = fx*X/Z + cx -> [fx/Z, 0,     -fx*X/Z^2]
  //   v = fy*Y/Z + cy -> [0,    fy/Z,  -fy*Y/Z^2]
  Jacobian projectJacobianWrtPoint(const Vec3& p) const override {
    const double invZ = 1.0 / p[2];
    const double invZ2 = invZ * invZ;
    Jacobian J;
    J.rows = 2;
    J.cols = 3;
    J.data = {fx_ * invZ, 0.0, -fx_ * p[0] * invZ2,
              0.0, fy_ * invZ, -fy_ * p[1] * invZ2};
    return J;
  }

  std::size_t numParams() const override { return 4; }
  std::string name() const override { return "pinhole"; }

  // Intrinsic parameters in constructor order (fx, fy, cx, cy) — for io/apply interop.
  std::array<double, 4> params() const { return {fx_, fy_, cx_, cy_}; }

 private:
  double fx_, fy_, cx_, cy_;
};

}  // namespace calibforge
