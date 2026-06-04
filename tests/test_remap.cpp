#include "calibforge/remap.hpp"

#include <array>
#include <cmath>

#include <Eigen/Dense>

#include "calibforge/board.hpp"
#include "calibforge/brown_conrady_camera.hpp"
#include "calibforge/corner_detect.hpp"
#include "calibforge/image.hpp"
#include "calibforge/remap_cuda.hpp"
#include "calibforge/render_board.hpp"
#include "calibforge/warp_map.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::BrownConradyCamera;
using calibforge::Image8;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::apply::cudaRemapAvailable;
using calibforge::apply::generateWarpMap;
using calibforge::apply::remapBilinear;
using calibforge::apply::WarpMap;
using calibforge::detect::CheckerboardSpec;
using calibforge::detect::refineCornerSaddle;
using calibforge::detect::renderCheckerboard;

// An identity warp map (src == output coords) reproduces the input image.
CF_TEST(remap_identity_reproduces_input) {
  Image8 in(32, 24);
  for (int y = 0; y < 24; ++y)
    for (int x = 0; x < 32; ++x) in.at(x, y) = static_cast<std::uint8_t>((x * 7 + y * 5) % 256);
  WarpMap map;
  map.width = 32;
  map.height = 24;
  map.src.resize(32u * 24u);
  for (int y = 0; y < 24; ++y)
    for (int x = 0; x < 32; ++x) map.src[y * 32 + x] = Vec2{double(x), double(y)};

  Image8 out = remapBilinear(in, map);
  for (int y = 0; y < 24; ++y)
    for (int x = 0; x < 32; ++x) CF_EXPECT_TRUE(out.at(x, y) == in.at(x, y));
}

// Undistort chain: render a DISTORTED board, remap through the warp map, and confirm the
// corners in the undistorted image land at the (straight-line) pinhole-predicted positions.
CF_TEST(remap_undistorts_distorted_board) {
  BrownConradyCamera bc(300.0, 300.0, 160.0, 120.0, -0.18, 0.05, 0.0, 0.0, 0.0);
  CheckerboardSpec spec{7, 5, 0.025};
  const Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Vector3d(0.05, -0.04, 0.02));
  const Eigen::Vector3d center(spec.extentX() / 2.0, spec.extentY() / 2.0, 0.0);
  Sophus::SE3d T(R, Eigen::Vector3d(0, 0, 0.55) - R * center);

  Image8 distorted = renderCheckerboard(bc, T, spec, 320, 240);
  const std::array<double, 4> out_K = {300.0, 300.0, 160.0, 120.0};
  WarpMap map = generateWarpMap(bc, out_K, 320, 240);
  Image8 undist = remapBilinear(distorted, map);

  // In the undistorted image, a board corner should sit at its pinhole (undistorted) pixel.
  double worst = 0.0;
  for (const auto& X : spec.objectPoints()) {
    Eigen::Vector3d Xc = T * Eigen::Vector3d(X[0], X[1], X[2]);
    const double xn = Xc.x() / Xc.z(), yn = Xc.y() / Xc.z();
    Vec2 pinhole_px{out_K[0] * xn + out_K[2], out_K[1] * yn + out_K[3]};  // straight projection
    auto c = refineCornerSaddle(undist, pinhole_px[0], pinhole_px[1], 3);
    worst = std::max(worst, std::hypot(c.x - pinhole_px[0], c.y - pinhole_px[1]));
  }
  std::printf("  [info] undistort: worst corner vs pinhole = %.3f px\n", worst);
  CF_EXPECT_TRUE(worst < 1.0);  // detector + interpolation budget
}

// cudaRemapAvailable() must MATCH the build configuration, not assume a CPU-only host: it is
// true exactly when this build linked the CUDA backend (CALIBFORGE_HAS_CUDA) and false on a
// CPU/CI host. (Asserting it is always false fails on a real CUDA host — the v1.0 target.)
// The bit-exact CPU<->GPU parity test below is compiled in only on a CUDA host.
CF_TEST(cuda_remap_availability_matches_build) {
#ifdef CALIBFORGE_HAS_CUDA
  CF_EXPECT_TRUE(cudaRemapAvailable());
#else
  CF_EXPECT_TRUE(!cudaRemapAvailable());
#endif
}

#ifdef CALIBFORGE_HAS_CUDA
// On a CUDA host: the GPU remap reproduces the CPU reference within +/-1 LSB.
CF_TEST(remap_cpu_gpu_parity) {
  BrownConradyCamera bc(300.0, 300.0, 160.0, 120.0, -0.18, 0.05, 0.001, -0.001, 0.0);
  Sophus::SE3d T(Sophus::SO3d(), Eigen::Vector3d(-0.1, -0.075, 0.55));
  Image8 src = renderCheckerboard(bc, T, CheckerboardSpec{7, 5, 0.025}, 320, 240);
  WarpMap map = generateWarpMap(bc, {300.0, 300.0, 160.0, 120.0}, 320, 240);
  Image8 cpu = remapBilinear(src, map);
  Image8 gpu = calibforge::apply::remapBilinearCuda(src, map);
  int worst = 0;
  for (std::size_t i = 0; i < cpu.data.size(); ++i)
    worst = std::max(worst, std::abs(int(cpu.data[i]) - int(gpu.data[i])));
  CF_EXPECT_TRUE(worst <= 1);
}
#endif
