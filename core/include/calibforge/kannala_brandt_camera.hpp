#pragma once
//
// CalibForge core — Kannala-Brandt (equidistant) fisheye camera model (header-only).
//
// Params: fx, fy, cx, cy, k1, k2, k3, k4  (4 intrinsic + 4 equidistant-polynomial).
//   r       = sqrt(X^2 + Y^2);  theta = atan2(r, Z)
//   theta_d = theta * (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
//   scale   = theta_d / r        (-> 1/Z at the optical axis: pinhole limit)
//   u = fx*scale*X + cx ;  v = fy*scale*Y + cy
//
// Reference math: Kannala & Brandt 2006 (docs/RESEARCH.md Theme 4); nvTorchCam ships KB
// but we re-implement with analytic Jacobians on the hot path.

#include <cmath>
#include <string>

#include "calibforge/camera_model.hpp"

namespace calibforge {

class KannalaBrandtCamera : public CameraModel {
 public:
  KannalaBrandtCamera(double fx, double fy, double cx, double cy,
                      double k1, double k2, double k3, double k4)
      : fx_(fx), fy_(fy), cx_(cx), cy_(cy), k1_(k1), k2_(k2), k3_(k3), k4_(k4) {}

  Vec2 project(const Vec3& p) const override {
    const double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    const double theta = std::atan2(r, p[2]);
    const double th2 = theta * theta;
    const double poly = 1.0 + th2 * (k1_ + th2 * (k2_ + th2 * (k3_ + th2 * k4_)));
    const double tr = (r > kEps) ? theta / r : 1.0 / p[2];  // theta/r -> 1/Z at the axis
    const double scale = tr * poly;                          // = theta_d / r
    return Vec2{fx_ * scale * p[0] + cx_, fy_ * scale * p[1] + cy_};
  }

  // Newton inverse on theta_d(theta) = r_u, then the unit ray.
  Vec3 unproject(const Vec2& px) const override {
    const double mx = (px[0] - cx_) / fx_;
    const double my = (px[1] - cy_) / fy_;
    const double ru = std::sqrt(mx * mx + my * my);  // = theta_d
    if (ru < kEps) return Vec3{0.0, 0.0, 1.0};
    double theta = ru;
    for (int i = 0; i < kUndistortIters; ++i) {
      const double th2 = theta * theta;
      const double f = theta * (1.0 + th2 * (k1_ + th2 * (k2_ + th2 * (k3_ + th2 * k4_)))) - ru;
      const double fp = 1.0 + th2 * (3.0 * k1_ + th2 * (5.0 * k2_ + th2 * (7.0 * k3_ + th2 * 9.0 * k4_)));
      theta -= f / fp;
    }
    const double s = std::sin(theta);
    return Vec3{s * mx / ru, s * my / ru, std::cos(theta)};
  }

  // d(u,v)/d(fx,fy,cx,cy,k1,k2,k3,k4), 2x8 row-major.
  Jacobian projectJacobianWrtParams(const Vec3& p) const override {
    const double r = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    const double theta = std::atan2(r, p[2]);
    const double th2 = theta * theta;
    const double poly = 1.0 + th2 * (k1_ + th2 * (k2_ + th2 * (k3_ + th2 * k4_)));
    const double tr = (r > kEps) ? theta / r : 1.0 / p[2];
    const double scale = tr * poly;

    Jacobian J;
    J.rows = 2;
    J.cols = 8;
    J.data.assign(16, 0.0);
    // cols: 0=fx 1=fy 2=cx 3=cy 4=k1 5=k2 6=k3 7=k4
    J.data[0] = scale * p[0];   // du/dfx
    J.data[2] = 1.0;            // du/dcx
    J.data[8 + 1] = scale * p[1];  // dv/dfy
    J.data[8 + 3] = 1.0;           // dv/dcy
    // d(scale)/dk_i = theta^(2i+1)/r = theta^(2i) * tr.
    double thp = th2;  // theta^2 (i=1)
    for (int i = 0; i < 4; ++i) {
      const double ds = thp * tr;            // d(scale)/dk_{i+1}
      J.data[4 + i] = fx_ * p[0] * ds;       // du/dk
      J.data[8 + 4 + i] = fy_ * p[1] * ds;   // dv/dk
      thp *= th2;
    }
    return J;
  }

  // d(u,v)/d(X,Y,Z), 2x3 row-major.
  Jacobian projectJacobianWrtPoint(const Vec3& p) const override {
    const double X = p[0], Y = p[1], Z = p[2];
    const double r = std::sqrt(X * X + Y * Y);
    Jacobian J;
    J.rows = 2;
    J.cols = 3;
    if (r < kEps) {  // pinhole limit at the optical axis
      const double invZ = 1.0 / Z;
      J.data = {fx_ * invZ, 0.0, -fx_ * X * invZ * invZ,
                0.0, fy_ * invZ, -fy_ * Y * invZ * invZ};
      return J;
    }
    const double theta = std::atan2(r, Z);
    const double th2 = theta * theta;
    const double poly = 1.0 + th2 * (k1_ + th2 * (k2_ + th2 * (k3_ + th2 * k4_)));
    const double theta_d = theta * poly;
    const double dtheta_d = 1.0 + th2 * (3.0 * k1_ + th2 * (5.0 * k2_ + th2 * (7.0 * k3_ + th2 * 9.0 * k4_)));
    const double s = theta_d / r;  // scale

    const double r2Z2 = r * r + Z * Z;
    const double dtheta_dX = (Z / r2Z2) * (X / r);
    const double dtheta_dY = (Z / r2Z2) * (Y / r);
    const double dtheta_dZ = -r / r2Z2;
    const double dr_dX = X / r, dr_dY = Y / r;  // dr/dZ = 0
    const double invr2 = 1.0 / (r * r);
    // ds/dq = (theta_d' * dtheta/dq * r - theta_d * dr/dq) / r^2
    const double ds_dX = (dtheta_d * dtheta_dX * r - theta_d * dr_dX) * invr2;
    const double ds_dY = (dtheta_d * dtheta_dY * r - theta_d * dr_dY) * invr2;
    const double ds_dZ = (dtheta_d * dtheta_dZ * r) * invr2;  // dr/dZ = 0

    J.data = {fx_ * (s + X * ds_dX), fx_ * X * ds_dY, fx_ * X * ds_dZ,
              fy_ * Y * ds_dX, fy_ * (s + Y * ds_dY), fy_ * Y * ds_dZ};
    return J;
  }

  std::size_t numParams() const override { return 8; }
  std::string name() const override { return "kannala_brandt"; }

 private:
  static constexpr int kUndistortIters = 20;
  static constexpr double kEps = 1e-9;

  double fx_, fy_, cx_, cy_;
  double k1_, k2_, k3_, k4_;
};

}  // namespace calibforge
