#include "calibforge/brown_conrady_camera.hpp"

#include <cmath>

#include "cf_test.hpp"

using calibforge::BrownConradyCamera;
using calibforge::Jacobian;
using calibforge::Vec2;
using calibforge::Vec3;

static double bc_norm3(const Vec3& v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Params order: fx, fy, cx, cy, k1, k2, p1, p2, k3.
// fx=fy=500, cx=320, cy=240, k1=0.1 (others 0); P=(0.1,0.2,2.0):
//   x=0.05, y=0.1, r2=0.0125, radial=1+0.1*0.0125=1.00125
//   x_d=0.0500625 -> u=500*0.0500625+320=345.03125
//   y_d=0.100125  -> v=500*0.100125 +240=290.0625
CF_TEST(brown_conrady_applies_radial_distortion) {
  BrownConradyCamera cam(500.0, 500.0, 320.0, 240.0, 0.1, 0.0, 0.0, 0.0, 0.0);
  Vec2 px = cam.project(Vec3{0.1, 0.2, 2.0});
  CF_EXPECT_NEAR(px[0], 345.03125, 1e-9);
  CF_EXPECT_NEAR(px[1], 290.0625, 1e-9);
}

static BrownConradyCamera bc_make(const double q[9]) {
  return BrownConradyCamera(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]);
}

// project (with full radial+tangential distortion) then unproject must recover
// the original ray direction. Exercises distortion in project AND the Newton inverse.
CF_TEST(brown_conrady_unproject_roundtrips_with_distortion) {
  BrownConradyCamera cam(500.0, 500.0, 320.0, 240.0, 0.1, 0.01, 0.001, 0.001, 0.001);
  Vec3 P{0.1, 0.2, 2.0};
  Vec2 px = cam.project(P);
  Vec3 dir = cam.unproject(px);
  double n = bc_norm3(P);
  CF_EXPECT_NEAR(dir[0], P[0] / n, 1e-7);
  CF_EXPECT_NEAR(dir[1], P[1] / n, 1e-7);
  CF_EXPECT_NEAR(dir[2], P[2] / n, 1e-7);
  CF_EXPECT_NEAR(bc_norm3(dir), 1.0, 1e-9);
}

// Analytic d(u,v)/d(fx,fy,cx,cy,k1,k2,p1,p2,k3) must match central finite difference.
CF_TEST(brown_conrady_param_jacobian_matches_finite_difference) {
  const double q[9] = {500.0, 520.0, 320.0, 240.0, 0.1, 0.01, 0.001, -0.002, 0.0005};
  BrownConradyCamera cam = bc_make(q);
  Vec3 P{0.12, -0.2, 2.0};

  Jacobian J = cam.projectJacobianWrtParams(P);
  CF_EXPECT_TRUE(J.rows == 2 && J.cols == 9 && J.data.size() == 18);

  const double h = 1e-5;
  for (int i = 0; i < 9; ++i) {
    double pp[9], pm[9];
    for (int j = 0; j < 9; ++j) { pp[j] = q[j]; pm[j] = q[j]; }
    pp[i] += h;
    pm[i] -= h;
    Vec2 up = bc_make(pp).project(P);
    Vec2 um = bc_make(pm).project(P);
    CF_EXPECT_NEAR(J.data[0 * 9 + i], (up[0] - um[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(J.data[1 * 9 + i], (up[1] - um[1]) / (2 * h), 1e-4);
  }
}

// Analytic d(pixel)/d(X,Y,Z) (through radial+tangential distortion) vs finite difference.
CF_TEST(brown_conrady_point_jacobian_matches_finite_difference) {
  BrownConradyCamera cam(500.0, 520.0, 320.0, 240.0, 0.1, 0.01, 0.001, -0.002, 0.0005);
  Vec3 P{0.12, -0.2, 2.0};

  Jacobian J = cam.projectJacobianWrtPoint(P);
  CF_EXPECT_TRUE(J.rows == 2 && J.cols == 3 && J.data.size() == 6);

  const double h = 1e-6;
  for (int i = 0; i < 3; ++i) {
    Vec3 pp = P, pm = P;
    pp[i] += h;
    pm[i] -= h;
    Vec2 up = cam.project(pp);
    Vec2 um = cam.project(pm);
    CF_EXPECT_NEAR(J.data[0 * 3 + i], (up[0] - um[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(J.data[1 * 3 + i], (up[1] - um[1]) / (2 * h), 1e-4);
  }
}
