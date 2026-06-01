#pragma once
//
// CalibForge apply — OpenCV-rational <-> VPI named-field distortion-coefficient mapping.
//
// VPI's polynomial lens-distortion model groups radial (k1..k6) and tangential (p1,p2)
// coefficients by NAMED FIELD. OpenCV/Brown-Conrady interleaves them differently
// ([k1 k2 p1 p2 k3 ...]). We map by field name, never by array index/memcpy — that
// ordering mismatch is the bug class issue #11 calls out.

#include <array>

#include "calibforge/brown_conrady_camera.hpp"

namespace calibforge {
namespace apply {

// Mirrors VPIPolynomialLensDistortionModel's named fields.
struct VpiPolynomialCoeffs {
  double k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0;  // radial (rational)
  double p1 = 0, p2 = 0;                                   // tangential
};

// Brown-Conrady (plumb-bob, 5 radial+tangential coeffs) -> VPI fields, by name.
inline VpiPolynomialCoeffs toVpiPolynomial(const BrownConradyCamera& cam) {
  const std::array<double, 9> q = cam.params();  // fx,fy,cx,cy,k1,k2,p1,p2,k3
  VpiPolynomialCoeffs c;
  c.k1 = q[4];  // k1
  c.k2 = q[5];  // k2
  c.k3 = q[8];  // k3  (note: NOT q[6]; plumb-bob orders k3 after the tangential terms)
  c.k4 = 0.0;   // plumb-bob has no higher radial (rational) terms
  c.k5 = 0.0;
  c.k6 = 0.0;
  c.p1 = q[6];  // p1
  c.p2 = q[7];  // p2
  return c;
}

}  // namespace apply
}  // namespace calibforge
