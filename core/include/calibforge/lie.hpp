#pragma once
//
// CalibForge core — SE(3)/SO(3) on-manifold operations (INTERFACE STUB).
//
// Borrow map (docs/RESEARCH.md Theme 4): use Sophus / manif (C++, MIT) for the
// host core, and Theseus's analytic tangent-space derivative patterns (MIT) as
// reference. Avoid LieTorch (documented CUDA build issues). All optimization
// happens in the tangent space; gradients are projected onto the manifold.
//
// This header only declares the contract CalibForge relies on so the solver and
// camera-model layers can be written against it before the backing lib is wired.

#include <array>

namespace calibforge {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;   // row-major
using Mat4 = std::array<double, 16>;  // row-major homogeneous

// SO(3) rotation. exp/log map between so(3) (axis-angle, R^3) and the group.
struct SO3 {
  static SO3 exp(const Vec3& omega);  // tangent -> group
  Vec3 log() const;                   // group -> tangent
  SO3 inverse() const;
  SO3 operator*(const SO3& rhs) const;
  Vec3 act(const Vec3& v) const;      // rotate a point
  Mat3 matrix() const;
};

// SE(3) rigid transform. Tangent is twist (R^6): [translation; rotation].
struct SE3 {
  static SE3 exp(const std::array<double, 6>& xi);
  std::array<double, 6> log() const;
  SE3 inverse() const;
  SE3 operator*(const SE3& rhs) const;
  Vec3 act(const Vec3& point) const;  // transform a point
  Mat4 matrix() const;
};

}  // namespace calibforge
