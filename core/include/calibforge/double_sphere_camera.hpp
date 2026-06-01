#pragma once
//
// CalibForge core — Double-Sphere camera model (Usenko, Demmel & Cremers, 3DV 2018;
// arXiv 1807.08957). The model nvTorchCam/Kornia/PyTorch3D do NOT ship (docs/RESEARCH.md
// Theme 4). Math RE-IMPLEMENTED from the paper; basalt-headers (BSD-3) used as a
// cross-check reference only — no code copied (CLAUDE.md rule 3).
//
// Params: fx, fy, cx, cy, xi, alpha.
//   d1 = ||P|| ; z' = xi*d1 + Z ; d2 = sqrt(X^2+Y^2+z'^2)
//   denom = alpha*d2 + (1-alpha)*z'
//   u = fx*X/denom + cx ;  v = fy*Y/denom + cy
// Closed-form unprojection (no Newton).

#include <cmath>
#include <string>

#include "calibforge/camera_model.hpp"

namespace calibforge {

class DoubleSphereCamera : public CameraModel {
 public:
  DoubleSphereCamera(double fx, double fy, double cx, double cy, double xi, double alpha)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy), xi_(xi), alpha_(alpha) {}

  Vec2 project(const Vec3& p) const override {
    const double X = p[0], Y = p[1], Z = p[2];
    const double d1 = std::sqrt(X * X + Y * Y + Z * Z);
    const double zp = xi_ * d1 + Z;
    const double d2 = std::sqrt(X * X + Y * Y + zp * zp);
    const double denom = alpha_ * d2 + (1.0 - alpha_) * zp;
    return Vec2{fx_ * X / denom + cx_, fy_ * Y / denom + cy_};
  }

  // Projection-validity: the point must lie in front of the modeled surface
  // (Usenko eq.; basalt condition Z > -w2*d1).
  bool projectValid(const Vec3& p) const {
    const double X = p[0], Y = p[1], Z = p[2];
    const double d1 = std::sqrt(X * X + Y * Y + Z * Z);
    const double w1 = (alpha_ <= 0.5) ? alpha_ / (1.0 - alpha_) : (1.0 - alpha_) / alpha_;
    const double w2 = (w1 + xi_) / std::sqrt(2.0 * w1 * xi_ + xi_ * xi_ + 1.0);
    return Z > -w2 * d1;
  }

  Vec3 unproject(const Vec2& px) const override {
    const double mx = (px[0] - cx_) / fx_;
    const double my = (px[1] - cy_) / fy_;
    const double r2 = mx * mx + my * my;
    const double mz = (1.0 - alpha_ * alpha_ * r2) /
                      (alpha_ * std::sqrt(1.0 - (2.0 * alpha_ - 1.0) * r2) + 1.0 - alpha_);
    const double k = (mz * xi_ + std::sqrt(mz * mz + (1.0 - xi_ * xi_) * r2)) / (mz * mz + r2);
    const double dx = k * mx, dy = k * my, dz = k * mz - xi_;
    const double n = std::sqrt(dx * dx + dy * dy + dz * dz);
    return Vec3{dx / n, dy / n, dz / n};
  }

  // d(u,v)/d(fx,fy,cx,cy,xi,alpha), 2x6 row-major.
  Jacobian projectJacobianWrtParams(const Vec3& p) const override {
    const double X = p[0], Y = p[1], Z = p[2];
    const double d1 = std::sqrt(X * X + Y * Y + Z * Z);
    const double zp = xi_ * d1 + Z;
    const double d2 = std::sqrt(X * X + Y * Y + zp * zp);
    const double denom = alpha_ * d2 + (1.0 - alpha_) * zp;
    const double inv = 1.0 / denom, inv2 = inv * inv;
    const double dd2_dxi = zp * d1 / d2;
    const double dden_dxi = alpha_ * dd2_dxi + (1.0 - alpha_) * d1;
    const double dden_dalpha = d2 - zp;
    Jacobian J;
    J.rows = 2;
    J.cols = 6;
    J.data.assign(12, 0.0);
    // cols: 0=fx 1=fy 2=cx 3=cy 4=xi 5=alpha
    J.data[0] = X * inv;                          // du/dfx
    J.data[2] = 1.0;                              // du/dcx
    J.data[4] = -fx_ * X * inv2 * dden_dxi;       // du/dxi
    J.data[5] = -fx_ * X * inv2 * dden_dalpha;    // du/dalpha
    J.data[6 + 1] = Y * inv;                      // dv/dfy
    J.data[6 + 3] = 1.0;                          // dv/dcy
    J.data[6 + 4] = -fy_ * Y * inv2 * dden_dxi;
    J.data[6 + 5] = -fy_ * Y * inv2 * dden_dalpha;
    return J;
  }

  // d(u,v)/d(X,Y,Z), 2x3 row-major.
  Jacobian projectJacobianWrtPoint(const Vec3& p) const override {
    const double X = p[0], Y = p[1], Z = p[2];
    const double d1 = std::sqrt(X * X + Y * Y + Z * Z);
    const double zp = xi_ * d1 + Z;
    const double d2 = std::sqrt(X * X + Y * Y + zp * zp);
    const double denom = alpha_ * d2 + (1.0 - alpha_) * zp;
    const double inv2 = 1.0 / (denom * denom);
    const double dd1_dX = X / d1, dd1_dY = Y / d1, dd1_dZ = Z / d1;
    const double dzp_dX = xi_ * dd1_dX, dzp_dY = xi_ * dd1_dY, dzp_dZ = xi_ * dd1_dZ + 1.0;
    const double dd2_dX = (X + zp * dzp_dX) / d2;
    const double dd2_dY = (Y + zp * dzp_dY) / d2;
    const double dd2_dZ = (zp * dzp_dZ) / d2;
    const double dden_dX = alpha_ * dd2_dX + (1.0 - alpha_) * dzp_dX;
    const double dden_dY = alpha_ * dd2_dY + (1.0 - alpha_) * dzp_dY;
    const double dden_dZ = alpha_ * dd2_dZ + (1.0 - alpha_) * dzp_dZ;
    Jacobian J;
    J.rows = 2;
    J.cols = 3;
    J.data = {fx_ * (denom - X * dden_dX) * inv2, fx_ * (-X * dden_dY) * inv2, fx_ * (-X * dden_dZ) * inv2,
              fy_ * (-Y * dden_dX) * inv2, fy_ * (denom - Y * dden_dY) * inv2, fy_ * (-Y * dden_dZ) * inv2};
    return J;
  }

  std::size_t numParams() const override { return 6; }
  std::string name() const override { return "double_sphere"; }

 private:
  double fx_, fy_, cx_, cy_, xi_, alpha_;
};

}  // namespace calibforge
