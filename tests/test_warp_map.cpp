#include "calibforge/warp_map.hpp"

#include <array>
#include <cmath>

#include "calibforge/brown_conrady_camera.hpp"
#include "calibforge/vpi_coeffs.hpp"
#include "calibforge/vpi_ldc.hpp"
#include "cf_test.hpp"

using calibforge::BrownConradyCamera;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::apply::generateWarpMap;
using calibforge::apply::toVpiPolynomial;
using calibforge::apply::vpiLdcAvailable;
using calibforge::apply::WarpMap;

// The generated map equals a direct per-pixel project of the target-pinhole ray through the
// distorted model (guards row-major indexing / off-by-one), and is semantically an
// undistortion table: unprojecting the source pixel recovers the output ray.
CF_TEST(warp_map_matches_per_pixel_undistort) {
  BrownConradyCamera cam(260.0, 260.0, 160.0, 120.0, -0.15, 0.04, 0.001, -0.001, 0.0005);
  const std::array<double, 4> out_K = {260.0, 260.0, 160.0, 120.0};
  WarpMap m = generateWarpMap(cam, out_K, 320, 240);
  CF_EXPECT_TRUE(m.width == 320 && m.height == 240);
  CF_EXPECT_TRUE(m.src.size() == 320u * 240u);

  for (int y : {10, 120, 230}) {
    for (int x : {10, 160, 300}) {
      const double xn = (x - out_K[2]) / out_K[0];
      const double yn = (y - out_K[3]) / out_K[1];
      Vec2 ref = cam.project(Vec3{xn, yn, 1.0});
      Vec2 got = m.at(x, y);
      CF_EXPECT_NEAR(got[0], ref[0], 1e-9);
      CF_EXPECT_NEAR(got[1], ref[1], 1e-9);
      // Semantic: undistorting the source pixel recovers the output pinhole ray direction.
      Vec3 ray = cam.unproject(got);
      const double n = std::sqrt(xn * xn + yn * yn + 1.0);
      CF_EXPECT_NEAR(ray[0], xn / n, 1e-6);
      CF_EXPECT_NEAR(ray[1], yn / n, 1e-6);
      CF_EXPECT_NEAR(ray[2], 1.0 / n, 1e-6);
    }
  }
}

// Coefficient mapping is by named field (k3 from the camera's k3 slot, not a raw index);
// the higher rational radial terms are zero for a plumb-bob model.
CF_TEST(vpi_polynomial_coeffs_map_by_named_field) {
  BrownConradyCamera cam(300.0, 300.0, 320.0, 240.0, -0.2, 0.05, 0.0011, -0.0009, 0.003);
  auto c = toVpiPolynomial(cam);
  CF_EXPECT_NEAR(c.k1, -0.2, 1e-15);
  CF_EXPECT_NEAR(c.k2, 0.05, 1e-15);
  CF_EXPECT_NEAR(c.k3, 0.003, 1e-15);
  CF_EXPECT_NEAR(c.p1, 0.0011, 1e-15);
  CF_EXPECT_NEAR(c.p2, -0.0009, 1e-15);
  CF_EXPECT_NEAR(c.k4, 0.0, 1e-15);
  CF_EXPECT_NEAR(c.k5, 0.0, 1e-15);
  CF_EXPECT_NEAR(c.k6, 0.0, 1e-15);
}

// vpiLdcAvailable() must MATCH the build configuration: true exactly when configured against
// the Jetson VPI SDK (CALIBFORGE_HAS_VPI), false on a CPU/CI host. (Asserting it is always
// false would fail on a real Jetson+VPI build — the rule-5/6 deliverable.)
CF_TEST(vpi_ldc_availability_matches_build) {
#ifdef CALIBFORGE_HAS_VPI
  CF_EXPECT_TRUE(vpiLdcAvailable());
#else
  CF_EXPECT_TRUE(!vpiLdcAvailable());
#endif
}
