#include "calibforge/double_sphere_camera.hpp"

#include <cmath>

#include "cf_test.hpp"

using calibforge::DoubleSphereCamera;
using calibforge::Jacobian;
using calibforge::Vec2;
using calibforge::Vec3;

static double ds_norm3(const Vec3& v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static DoubleSphereCamera ds_make(const double q[6]) {
  return DoubleSphereCamera(q[0], q[1], q[2], q[3], q[4], q[5]);
}

// Golden at P=(0.1,0.2,2.0), xi=0.3, alpha=0.6: d1=sqrt(4.05)=2.0124612,
// z'=2.6037384, d2=2.6133222, denom=2.6094886 ->
//   u = 50/denom + 320 = 339.16084 ; v = 100/denom + 240 = 278.32168.
CF_TEST(double_sphere_projects_to_golden_value) {
  DoubleSphereCamera cam(500.0, 500.0, 320.0, 240.0, 0.3, 0.6);
  Vec2 px = cam.project(Vec3{0.1, 0.2, 2.0});
  CF_EXPECT_NEAR(px[0], 339.16084, 1e-3);
  CF_EXPECT_NEAR(px[1], 278.32168, 1e-3);
}

CF_TEST(double_sphere_unproject_roundtrips) {
  DoubleSphereCamera cam(500.0, 500.0, 320.0, 240.0, 0.3, 0.6);
  for (Vec3 P : {Vec3{0.1, 0.2, 2.0}, Vec3{0.4, -0.3, 1.0}}) {
    Vec2 px = cam.project(P);
    Vec3 dir = cam.unproject(px);
    double n = ds_norm3(P);
    CF_EXPECT_NEAR(dir[0], P[0] / n, 1e-7);
    CF_EXPECT_NEAR(dir[1], P[1] / n, 1e-7);
    CF_EXPECT_NEAR(dir[2], P[2] / n, 1e-7);
    CF_EXPECT_NEAR(ds_norm3(dir), 1.0, 1e-9);
  }
}

CF_TEST(double_sphere_param_jacobian_matches_finite_difference) {
  const double q[6] = {500.0, 520.0, 320.0, 240.0, 0.3, 0.6};
  DoubleSphereCamera cam = ds_make(q);
  Vec3 P{0.2, -0.15, 1.5};

  Jacobian J = cam.projectJacobianWrtParams(P);
  CF_EXPECT_TRUE(J.rows == 2 && J.cols == 6 && J.data.size() == 12);

  const double h = 1e-6;
  for (int i = 0; i < 6; ++i) {
    double pp[6], pm[6];
    for (int j = 0; j < 6; ++j) { pp[j] = q[j]; pm[j] = q[j]; }
    pp[i] += h;
    pm[i] -= h;
    Vec2 up = ds_make(pp).project(P);
    Vec2 um = ds_make(pm).project(P);
    CF_EXPECT_NEAR(J.data[0 * 6 + i], (up[0] - um[0]) / (2 * h), 1e-4);
    CF_EXPECT_NEAR(J.data[1 * 6 + i], (up[1] - um[1]) / (2 * h), 1e-4);
  }
}

CF_TEST(double_sphere_point_jacobian_matches_finite_difference) {
  DoubleSphereCamera cam(500.0, 520.0, 320.0, 240.0, 0.3, 0.6);
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

// A point well in front projects validly; one well behind the modeled surface does not.
CF_TEST(double_sphere_project_validity_domain_flags_behind_surface) {
  DoubleSphereCamera cam(500.0, 500.0, 320.0, 240.0, 0.3, 0.6);
  CF_EXPECT_TRUE(cam.projectValid(Vec3{0.1, 0.2, 2.0}));
  CF_EXPECT_TRUE(!cam.projectValid(Vec3{0.1, 0.2, -3.0}));
}
