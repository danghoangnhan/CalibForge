#include "calibforge/kannala_brandt_camera.hpp"

#include <cmath>

#include "cf_test.hpp"

using calibforge::Jacobian;
using calibforge::KannalaBrandtCamera;
using calibforge::Vec2;
using calibforge::Vec3;

static double kb_norm3(const Vec3& v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static KannalaBrandtCamera kb_make(const double q[8]) {
  return KannalaBrandtCamera(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
}

// Golden value at a 45-degree point P=(1,0,1) with k1=0.1 (others 0):
//   theta = atan2(1,1) = pi/4 ; theta_d = (pi/4)(1 + 0.1 (pi/4)^2) = 0.83384548
//   scale = theta_d/r = 0.83384548 ; u = 500*scale*1 + 320 = 736.92274 ; v = cy = 240 (Y=0)
CF_TEST(kannala_brandt_projects_to_golden_value) {
  KannalaBrandtCamera cam(500.0, 500.0, 320.0, 240.0, 0.1, 0.0, 0.0, 0.0);
  Vec2 px = cam.project(Vec3{1.0, 0.0, 1.0});
  CF_EXPECT_NEAR(px[0], 736.92274, 1e-4);
  CF_EXPECT_NEAR(px[1], 240.0, 1e-9);
}

// project (full KB polynomial) then Newton-unproject must recover the original ray.
CF_TEST(kannala_brandt_unproject_roundtrips) {
  KannalaBrandtCamera cam(420.0, 420.0, 320.0, 240.0, 0.05, -0.01, 0.004, -0.001);
  // Two points: moderate and wide-angle (theta ~ 0.9 rad).
  for (Vec3 P : {Vec3{0.3, -0.2, 1.0}, Vec3{0.9, 0.7, 1.0}}) {
    Vec2 px = cam.project(P);
    Vec3 dir = cam.unproject(px);
    double n = kb_norm3(P);
    CF_EXPECT_NEAR(dir[0], P[0] / n, 1e-7);
    CF_EXPECT_NEAR(dir[1], P[1] / n, 1e-7);
    CF_EXPECT_NEAR(dir[2], P[2] / n, 1e-7);
    CF_EXPECT_NEAR(kb_norm3(dir), 1.0, 1e-9);
  }
}

// Analytic d(u,v)/d(fx,fy,cx,cy,k1,k2,k3,k4) vs central finite difference.
CF_TEST(kannala_brandt_param_jacobian_matches_finite_difference) {
  const double q[8] = {500.0, 520.0, 320.0, 240.0, 0.05, 0.01, -0.005, 0.001};
  KannalaBrandtCamera cam = kb_make(q);
  Vec3 P{0.2, -0.15, 1.5};

  Jacobian J = cam.projectJacobianWrtParams(P);
  CF_EXPECT_TRUE(J.rows == 2 && J.cols == 8 && J.data.size() == 16);

  const double h = 1e-5;
  for (int i = 0; i < 8; ++i) {
    double pp[8], pm[8];
    for (int j = 0; j < 8; ++j) { pp[j] = q[j]; pm[j] = q[j]; }
    pp[i] += h;
    pm[i] -= h;
    Vec2 up = kb_make(pp).project(P);
    Vec2 um = kb_make(pm).project(P);
    CF_EXPECT_NEAR(J.data[0 * 8 + i], (up[0] - um[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(J.data[1 * 8 + i], (up[1] - um[1]) / (2 * h), 1e-4);
  }
}

// Analytic d(pixel)/d(X,Y,Z) through the KB polynomial vs finite difference.
CF_TEST(kannala_brandt_point_jacobian_matches_finite_difference) {
  KannalaBrandtCamera cam(500.0, 520.0, 320.0, 240.0, 0.05, 0.01, -0.005, 0.001);
  Vec3 P{0.2, -0.15, 1.5};

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
