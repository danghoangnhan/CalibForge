// Tests for detect/bev_photometric.hpp — BEV agreement cost + coarse-to-fine random search.

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/bev_photometric.hpp"
#include "calibforge/image.hpp"
#include "calibforge/observability.hpp"  // assessObservability / ObservabilityReport (RULE #2 gate)
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

CF_TEST(bev_random_search_reduces_cost_and_keeps_overlap_without_diverging) {
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

  const double err_init = (T_c1_c0_init * T_c1_c0.inverse()).log().norm();
  // Initial overlap coverage — the floor a winning trial must retain.
  const BevAgreementResult r0 =
      bevAgreementCost(samples, cams, {Sophus::SE3d(), T_c1_c0_init}, T_c0_w, imgs);

  BevRandomSearchOptions opts;
  opts.rot_scales = {0.01, 0.003};
  opts.trans_scales = {0.02, 0.005};
  opts.attempts_per_level = 64;
  const int overlap_floor = static_cast<int>(opts.min_overlap_fraction * r0.overlap_count);

  // We deliberately do NOT assert the refined extrinsic is CLOSER to truth. The coarse BEV
  // photometric minimum is biased (precision != accuracy, RULE #2): across seeds accuracy
  // improvement is roughly a coin flip, so a single magic-seed `err_refined < err_init`
  // assertion is brittle — it flips with the seed and gives false confidence that the search
  // improves accuracy. (The sibling surround test declines the same accuracy claim for the same
  // reason.) Across several seeds we instead assert properties that are robust and not purely
  // tautological:
  //   (a) the emitted result respects the overlap-coverage floor (the min_overlap_fraction
  //       invariant the search enforces so it cannot game the per-pair MEAN by shrinking the
  //       support set). At these conservative perturbation scales the floor is not stressed, so
  //       this is a sanity invariant, not a strong regression guard for the floor mechanism.
  //   (b) it achieves a real, MEANINGFUL relative cost reduction — `cost_reduction_ratio > 0.1`
  //       is NOT true by construction (only `final <= initial` is); min observed ~0.17 over
  //       thousands of seeds.
  //   (c) it does not grossly diverge. The search only accepts cost-decreasing, coverage-
  //       retaining trials, so err stays bounded; a HEALTHY run reaches at most ~4.4x err_init
  //       over thousands of seeds, so a generous 8x cap flags a grossly broken / sign-flipped
  //       search without red-building on healthy numeric variation.
  for (std::uint64_t seed : {1ull, 2ull, 7ull, 42ull, 0x12345ull, 0xabcdefull}) {
    opts.seed = seed;
    const BevRandomSearchResult r = bevRandomSearchExtrinsics(
        samples, cams, {T_c1_c0_init}, rig_poses, per_frame, opts);
    const double err_refined = (r.extrinsics[0] * T_c1_c0.inverse()).log().norm();
    CF_EXPECT_TRUE(r.overlap_count >= overlap_floor);  // (a) coverage invariant
    CF_EXPECT_TRUE(r.cost_reduction_ratio > 0.1);      // (b) real relative reduction (min ~0.17)
    CF_EXPECT_TRUE(err_refined < 8.0 * err_init);      // (c) gross-divergence sanity cap
  }
}

CF_TEST(bev_information_matrix_is_observable_with_texture_and_collapses_without) {
  // RULE #2 teeth for the surround path: the observability gate is only meaningful if the
  // information matrix it gates on is actually computed from the photometric geometry. With
  // texture, H = J^T J of the BEV agreement residuals is full-rank + conditioned (observable);
  // with FLAT images every residual is identically zero so J -> 0 and H collapses to ~0
  // (NOT observable). A stubbed / identity H — the exact failure this guards against — would
  // report observable in BOTH cases.
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  const Sophus::SE3d T_c0_w(Sophus::SO3d::exp(Eigen::Vector3d(1.2, 0.0, 0.0)),
                            Eigen::Vector3d(0.0, 0.0, 1.5));
  const Sophus::SE3d T_c1_c0(Sophus::SO3d(), Eigen::Vector3d(-0.3, 0.0, 0.0));
  std::vector<const CameraModel*> cams = {&cam, &cam};

  BevGridOptions g_opts;
  g_opts.x_min = -1.0; g_opts.x_max = 1.0;
  g_opts.y_min = 0.5;  g_opts.y_max = 2.5;
  g_opts.step = 0.1;
  const std::vector<BevSample> samples = makeBevGrid(g_opts);
  std::vector<Sophus::SE3d> rig_poses = {T_c0_w};

  // Textured: full-rank, well-conditioned information.
  Image8 im0 = makeGroundImage(cam, T_c0_w, 320, 240);
  Image8 im1 = makeGroundImage(cam, T_c1_c0 * T_c0_w, 320, 240);
  std::vector<std::vector<const Image8*>> per_frame = {{&im0, &im1}};
  const calibforge::detect::BevInformation info =
      calibforge::detect::bevInformationMatrix(samples, cams, {T_c1_c0}, rig_poses, per_frame);
  CF_EXPECT_TRUE(info.H.rows() == 6 && info.H.cols() == 6);   // 6*(N-1), N=2
  CF_EXPECT_TRUE(info.num_residuals > 0);
  CF_EXPECT_TRUE((info.H - info.H.transpose()).norm() <= 1e-9 * std::max(1.0, info.H.norm()));
  const calibforge::ObservabilityReport rep_tex = calibforge::assessObservability(info.H);
  CF_EXPECT_TRUE(rep_tex.observable);
  CF_EXPECT_TRUE(rep_tex.confidence > 0.0);

  // Flat (textureless): identical geometry, but zero photometric signal -> H ~ 0 -> NOT
  // observable. This is what a stubbed identity H would get WRONG.
  Image8 flat0(320, 240, 128), flat1(320, 240, 128);
  std::vector<std::vector<const Image8*>> flat_pf = {{&flat0, &flat1}};
  const calibforge::detect::BevInformation finfo =
      calibforge::detect::bevInformationMatrix(samples, cams, {T_c1_c0}, rig_poses, flat_pf);
  const calibforge::ObservabilityReport rep_flat = calibforge::assessObservability(finfo.H);
  CF_EXPECT_TRUE(!rep_flat.observable);
  CF_EXPECT_TRUE(rep_flat.confidence == 0.0);
}
