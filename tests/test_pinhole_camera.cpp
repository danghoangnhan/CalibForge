#include "calibforge/pinhole_camera.hpp"

#include <cmath>

#include "cf_test.hpp"

using calibforge::Jacobian;
using calibforge::PinholeCamera;
using calibforge::Vec2;
using calibforge::Vec3;

static double norm3(const Vec3& v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// fx=fy=500, cx=320, cy=240; point (0.1, 0.2, 2.0):
//   u = 500*0.1/2.0 + 320 = 345 ; v = 500*0.2/2.0 + 240 = 290
CF_TEST(pinhole_projects_point_to_expected_pixel) {
  PinholeCamera cam(500.0, 500.0, 320.0, 240.0);
  auto px = cam.project(Vec3{0.1, 0.2, 2.0});
  CF_EXPECT_NEAR(px[0], 345.0, 1e-9);
  CF_EXPECT_NEAR(px[1], 290.0, 1e-9);
}

// unproject returns a UNIT ray; projecting then unprojecting must recover the
// original point's direction (P normalized), and the ray must be unit length.
CF_TEST(pinhole_unproject_roundtrips_to_original_ray) {
  PinholeCamera cam(500.0, 500.0, 320.0, 240.0);
  Vec3 P{0.1, 0.2, 2.0};
  Vec2 px = cam.project(P);
  Vec3 dir = cam.unproject(px);
  double n = norm3(P);
  CF_EXPECT_NEAR(dir[0], P[0] / n, 1e-9);
  CF_EXPECT_NEAR(dir[1], P[1] / n, 1e-9);
  CF_EXPECT_NEAR(dir[2], P[2] / n, 1e-9);
  CF_EXPECT_NEAR(norm3(dir), 1.0, 1e-9);
}

// The analytic d(pixel)/d(fx,fy,cx,cy) must match a central finite difference of
// project() (the research rule: analytic Jacobians validated against an oracle).
CF_TEST(pinhole_param_jacobian_matches_finite_difference) {
  const double fx = 500.0, fy = 520.0, cx = 320.0, cy = 240.0;
  PinholeCamera cam(fx, fy, cx, cy);
  Vec3 P{0.1, 0.2, 2.0};

  Jacobian J = cam.projectJacobianWrtParams(P);
  CF_EXPECT_TRUE(J.rows == 2 && J.cols == 4 && J.data.size() == 8);

  const double base[4] = {fx, fy, cx, cy};
  const double h = 1e-4;
  for (int i = 0; i < 4; ++i) {
    double pp[4] = {fx, fy, cx, cy};
    double pm[4] = {fx, fy, cx, cy};
    pp[i] += h;
    pm[i] -= h;
    Vec2 up = PinholeCamera(pp[0], pp[1], pp[2], pp[3]).project(P);
    Vec2 um = PinholeCamera(pm[0], pm[1], pm[2], pm[3]).project(P);
    CF_EXPECT_NEAR(J.data[0 * 4 + i], (up[0] - um[0]) / (2 * h), 1e-6);  // du/dparam_i
    CF_EXPECT_NEAR(J.data[1 * 4 + i], (up[1] - um[1]) / (2 * h), 1e-6);  // dv/dparam_i
  }
  (void)base;
}

// Analytic d(pixel)/d(X,Y,Z) (camera-frame point) must match central finite difference.
// Needed for the pose Jacobian in bundle adjustment / calibration.
CF_TEST(pinhole_point_jacobian_matches_finite_difference) {
  PinholeCamera cam(500.0, 520.0, 320.0, 240.0);
  Vec3 P{0.1, -0.2, 2.0};

  Jacobian J = cam.projectJacobianWrtPoint(P);
  CF_EXPECT_TRUE(J.rows == 2 && J.cols == 3 && J.data.size() == 6);

  const double h = 1e-6;
  for (int i = 0; i < 3; ++i) {
    Vec3 pp = P, pm = P;
    pp[i] += h;
    pm[i] -= h;
    Vec2 up = cam.project(pp);
    Vec2 um = cam.project(pm);
    CF_EXPECT_NEAR(J.data[0 * 3 + i], (up[0] - um[0]) / (2 * h), 1e-5);  // du/d{X,Y,Z}
    CF_EXPECT_NEAR(J.data[1 * 3 + i], (up[1] - um[1]) / (2 * h), 1e-5);  // dv/d{X,Y,Z}
  }
}
