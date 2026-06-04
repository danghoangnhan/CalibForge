// Tests for detect/bev_photometric.hpp — BEV agreement cost + coarse-to-fine random search.

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/bev_photometric.hpp"
#include "calibforge/image.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraModel;
using calibforge::Image8;
using calibforge::PinholeCamera;
using calibforge::detect::BevAgreementResult;
using calibforge::detect::BevGridOptions;
using calibforge::detect::BevRandomSearchOptions;
using calibforge::detect::BevRandomSearchResult;
using calibforge::detect::BevSample;
using calibforge::detect::bevAgreementCost;
using calibforge::detect::bevRandomSearchExtrinsics;
using calibforge::detect::makeBevGrid;

namespace {

// Synthesize an image where intensity is a smooth function of the ground point (X, Y) that
// the pixel maps to. With true extrinsics, two cameras observing the same (X, Y) get the
// same intensity. With perturbed extrinsics, the recovered intensities mismatch.
Image8 makeGroundImage(const PinholeCamera& cam, const Sophus::SE3d& T_cam_w, int w, int h) {
  Image8 im(w, h, 0);
  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      // Unproject pixel, intersect with z=0 plane in world frame.
      const calibforge::Vec3 ray_cam = cam.unproject(calibforge::Vec2{static_cast<double>(u),
                                                                     static_cast<double>(v)});
      const Eigen::Vector3d ray_w = T_cam_w.so3().inverse() *
                                    Eigen::Vector3d(ray_cam[0], ray_cam[1], ray_cam[2]);
      const Eigen::Vector3d origin_w = T_cam_w.inverse().translation();
      if (std::abs(ray_w.z()) < 1e-6) continue;
      const double s = -origin_w.z() / ray_w.z();
      if (s <= 0.0) continue;
      const double X = origin_w.x() + s * ray_w.x();
      const double Y = origin_w.y() + s * ray_w.y();
      if (X < -5.0 || X > 5.0 || Y < -5.0 || Y > 5.0) continue;
      // Smooth intensity: large-period sinusoid in X+Y so cubic bilinear interpolation
      // doesn't get aliased. Range [0, 255].
      const double f = 0.5 + 0.4 * std::sin(1.5 * X) * std::cos(1.5 * Y);
      im.at(u, v) = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, 255.0 * f)));
    }
  }
  return im;
}

}  // namespace

CF_TEST(bev_grid_count_matches_extent_and_step) {
  BevGridOptions opts;
  opts.x_min = -1.0;
  opts.x_max = 1.0;
  opts.y_min = -1.0;
  opts.y_max = 1.0;
  opts.step = 0.25;
  const std::vector<BevSample> g = makeBevGrid(opts);
  // (1 - (-1)) / 0.25 + 1 = 9 in each axis => 81 samples.
  CF_EXPECT_TRUE(g.size() == 81);
}

CF_TEST(bev_agreement_cost_is_lower_with_true_extrinsics_than_with_perturbation) {
  // Two cameras looking at the ground plane in front of the rig.
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  // cam0 = the rig anchor. Place cam0 1.5m above the ground looking down + forward.
  const Sophus::SE3d T_c0_w(Sophus::SO3d::exp(Eigen::Vector3d(1.2, 0.0, 0.0)),
                            Eigen::Vector3d(0.0, 0.0, 1.5));
  // cam1 offset by 0.3m to the right of cam0, same orientation.
  const Sophus::SE3d T_c1_c0(Sophus::SO3d(), Eigen::Vector3d(-0.3, 0.0, 0.0));
  const Sophus::SE3d T_c1_w = T_c1_c0 * T_c0_w;

  Image8 im0 = makeGroundImage(cam, T_c0_w, 320, 240);
  Image8 im1 = makeGroundImage(cam, T_c1_w, 320, 240);

  std::vector<const CameraModel*> cams = {&cam, &cam};
  std::vector<const Image8*> imgs = {&im0, &im1};

  BevGridOptions g_opts;
  g_opts.x_min = -1.0;
  g_opts.x_max = 1.0;
  g_opts.y_min = 0.5;
  g_opts.y_max = 2.5;
  g_opts.step = 0.05;
  const std::vector<BevSample> samples = makeBevGrid(g_opts);

  // Cost with TRUE extrinsics.
  const BevAgreementResult r_true =
      bevAgreementCost(samples, cams, {Sophus::SE3d(), T_c1_c0}, T_c0_w, imgs);
  CF_EXPECT_TRUE(r_true.overlap_count > 100);

  // Cost with PERTURBED extrinsics.
  const Sophus::SE3d T_c1_c0_bad =
      T_c1_c0 * Sophus::SE3d::exp((Eigen::Matrix<double, 6, 1>() << 0.10, 0.05, 0.0,
                                                                   0.05, 0.0, 0.0)
                                       .finished());
  const BevAgreementResult r_bad =
      bevAgreementCost(samples, cams, {Sophus::SE3d(), T_c1_c0_bad}, T_c0_w, imgs);
  CF_EXPECT_TRUE(r_bad.overlap_count > 0);
  // Perturbed extrinsics must produce a strictly larger mean-squared diff.
  CF_EXPECT_TRUE(r_bad.mean_sq_diff > r_true.mean_sq_diff);
}

CF_TEST(bev_random_search_reduces_cost_from_perturbed_init) {
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  const Sophus::SE3d T_c0_w(Sophus::SO3d::exp(Eigen::Vector3d(1.2, 0.0, 0.0)),
                            Eigen::Vector3d(0.0, 0.0, 1.5));
  const Sophus::SE3d T_c1_c0(Sophus::SO3d(), Eigen::Vector3d(-0.3, 0.0, 0.0));
  const Sophus::SE3d T_c1_w = T_c1_c0 * T_c0_w;

  Image8 im0 = makeGroundImage(cam, T_c0_w, 320, 240);
  Image8 im1 = makeGroundImage(cam, T_c1_w, 320, 240);
  std::vector<const CameraModel*> cams = {&cam, &cam};
  std::vector<const Image8*> imgs = {&im0, &im1};

  BevGridOptions g_opts;
  g_opts.x_min = -1.0; g_opts.x_max = 1.0;
  g_opts.y_min = 0.5;  g_opts.y_max = 2.5;
  g_opts.step = 0.1;
  const std::vector<BevSample> samples = makeBevGrid(g_opts);

  // Perturbed initial extrinsics.
  const Sophus::SE3d T_c1_c0_init =
      T_c1_c0 * Sophus::SE3d::exp((Eigen::Matrix<double, 6, 1>() << 0.03, 0.02, 0.0,
                                                                   0.01, 0.0, 0.0)
                                       .finished());

  std::vector<std::vector<const Image8*>> per_frame = {imgs};
  std::vector<Sophus::SE3d> rig_poses = {T_c0_w};

  BevRandomSearchOptions opts;
  opts.rot_scales = {0.01, 0.003};
  opts.trans_scales = {0.02, 0.005};
  opts.attempts_per_level = 64;
  opts.seed = 0x12345;

  const BevRandomSearchResult r = bevRandomSearchExtrinsics(
      samples, cams, {T_c1_c0_init}, rig_poses, per_frame, opts);
  CF_EXPECT_TRUE(r.overlap_count > 50);
  // final_cost <= initial_cost and cost_reduction_ratio >= 0 are true BY CONSTRUCTION (best is
  // only replaced on a strict cost decrease) and prove nothing. The real test is ACCURACY:
  // the refined extrinsic must be strictly closer to the TRUE T_c1_c0 than the perturbed init.
  const double err_init = (T_c1_c0_init * T_c1_c0.inverse()).log().norm();
  const double err_refined = (r.extrinsics[0] * T_c1_c0.inverse()).log().norm();
  CF_EXPECT_TRUE(err_refined < err_init);
}
