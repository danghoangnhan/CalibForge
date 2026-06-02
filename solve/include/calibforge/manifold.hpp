#pragma once
//
// CalibForge solve — minimal local-parameterization (manifold) abstraction for the
// Problem/ResidualBlock solver (header-only).
//
// Each parameter block has an AMBIENT representation (the doubles stored in memory)
// and a TANGENT (minimal DOF) the optimizer works in. retract() maps an ambient point
// + a tangent increment back onto the manifold. This is the Ceres "Manifold" /
// LocalParameterization role, pared down to the single op the LM loop needs.
//
//   EuclideanParam: ambient == tangent, x_out = x + delta.
//   SE3Param:       ambient = 7 [tx,ty,tz, qx,qy,qz,qw], tangent = 6 [translation; rotation],
//                   retract via the RIGHT perturbation T <- T * exp(delta) — matching the
//                   analytic pose Jacobian d(pixel)/d(point) * [R | -R [X]_x]
//                   (docs/RESEARCH.md Theme 4, "analytic on hot paths").

#include <Eigen/Dense>

#include "sophus/se3.hpp"

namespace calibforge {

class LocalParameterization {
 public:
  virtual ~LocalParameterization() = default;
  virtual int ambientSize() const = 0;   // doubles stored in memory
  virtual int tangentSize() const = 0;    // minimal DOF (optimizer columns)
  // x_out (ambient) = retraction of x (ambient) by delta (tangent).
  virtual void retract(const double* x, const double* delta, double* x_out) const = 0;
};

// R^n: ambient == tangent, plain vector addition.
class EuclideanParam : public LocalParameterization {
 public:
  explicit EuclideanParam(int size) : size_(size) {}
  int ambientSize() const override { return size_; }
  int tangentSize() const override { return size_; }
  void retract(const double* x, const double* delta, double* x_out) const override {
    for (int i = 0; i < size_; ++i) x_out[i] = x[i] + delta[i];
  }

 private:
  int size_;
};

// SO(3): stored as a unit quaternion [qx,qy,qz,qw], optimized in the 3-DoF tangent.
// Retraction is the right perturbation R <- R*exp(delta).
class SO3Param : public LocalParameterization {
 public:
  int ambientSize() const override { return 4; }
  int tangentSize() const override { return 3; }

  static Sophus::SO3d load(const double* x) {
    Eigen::Quaterniond q(x[3], x[0], x[1], x[2]);  // Eigen ctor order is (w, x, y, z)
    q.normalize();
    return Sophus::SO3d(q);
  }
  static void store(const Sophus::SO3d& R, double* x) {
    const Eigen::Quaterniond q = R.unit_quaternion();
    x[0] = q.x(); x[1] = q.y(); x[2] = q.z(); x[3] = q.w();
  }
  void retract(const double* x, const double* delta, double* x_out) const override {
    const Eigen::Vector3d d(delta[0], delta[1], delta[2]);
    store(load(x) * Sophus::SO3d::exp(d), x_out);
  }
};

// SE(3): stored as [t (3); unit quaternion (qx,qy,qz,qw)], optimized in the 6-DoF
// tangent [translation; rotation]. Retraction is the right perturbation T <- T*exp(delta).
class SE3Param : public LocalParameterization {
 public:
  int ambientSize() const override { return 7; }
  int tangentSize() const override { return 6; }

  static Sophus::SE3d load(const double* x) {
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);  // Eigen ctor order is (w, x, y, z)
    q.normalize();
    return Sophus::SE3d(q, Eigen::Vector3d(x[0], x[1], x[2]));
  }
  static void store(const Sophus::SE3d& T, double* x) {
    const Eigen::Vector3d t = T.translation();
    const Eigen::Quaterniond q = T.unit_quaternion();
    x[0] = t.x(); x[1] = t.y(); x[2] = t.z();
    x[3] = q.x(); x[4] = q.y(); x[5] = q.z(); x[6] = q.w();
  }
  void retract(const double* x, const double* delta, double* x_out) const override {
    Eigen::Matrix<double, 6, 1> d;
    d << delta[0], delta[1], delta[2], delta[3], delta[4], delta[5];
    store(load(x) * Sophus::SE3d::exp(d), x_out);
  }
};

}  // namespace calibforge
