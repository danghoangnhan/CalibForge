// Tests for pipelines/online_surround_rig.hpp — surround-view BEV photometric extrinsic
// recalibration orchestrator. Mirrors the structure of OnlineExtrinsicTracker's tests: a
// well-excited scenario where the random search reduces the BEV cost vs initial, and a
// degenerate-motion scenario that the 6-axis motion gate refuses BEFORE searching.

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/bev_photometric.hpp"
#include "calibforge/calibrate_single.hpp"
#include "calibforge/image.hpp"
#include "calibforge/online_surround_rig.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraFactory;
using calibforge::CameraModel;
using calibforge::Image8;
using calibforge::PinholeCamera;
using calibforge::pipelines::OnlineSurroundRig;
using calibforge::pipelines::OnlineSurroundRigOptions;
using calibforge::pipelines::SurroundRigEmission;

namespace {

Image8 makeGroundImage(const PinholeCamera& cam, const Sophus::SE3d& T_cam_w, int w, int h) {
  Image8 im(w, h, 0);
  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
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
      const double f = 0.5 + 0.4 * std::sin(1.5 * X) * std::cos(1.5 * Y);
      im.at(u, v) = static_cast<std::uint8_t>(std::min(255.0, std::max(0.0, 255.0 * f)));
    }
  }
  return im;
}

}  // namespace

CF_TEST(online_surround_rig_refuses_degenerate_static_motion) {
  // Static rig: no translation, no rotation across the window. Motion gate refuses BEFORE
  // any BEV search.
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  Eigen::VectorXd intr(4);
  intr << 300.0, 300.0, 160.0, 120.0;
  CameraFactory mk = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };

  const Sophus::SE3d T_c0_w(Sophus::SO3d::exp(Eigen::Vector3d(1.2, 0.0, 0.0)),
                            Eigen::Vector3d(0.0, 0.0, 1.5));
  const Sophus::SE3d T_c1_c0(Sophus::SO3d(), Eigen::Vector3d(-0.3, 0.0, 0.0));

  Image8 im0 = makeGroundImage(cam, T_c0_w, 320, 240);
  Image8 im1 = makeGroundImage(cam, T_c1_c0 * T_c0_w, 320, 240);
  std::vector<const Image8*> frame_imgs = {&im0, &im1};

  OnlineSurroundRigOptions opts;
  opts.bev_grid.x_min = -1.0; opts.bev_grid.x_max = 1.0;
  opts.bev_grid.y_min = 0.5;  opts.bev_grid.y_max = 2.5;
  opts.bev_grid.step = 0.1;

  OnlineSurroundRig orch({mk, mk}, {intr, intr}, {T_c1_c0}, opts);
  for (int i = 0; i < 4; ++i) orch.addFrame(frame_imgs, T_c0_w);  // identical poses

  const SurroundRigEmission e = orch.tryEmit();
  CF_EXPECT_TRUE(!e.emitted);
  CF_EXPECT_TRUE(e.refused_for_motion);
  CF_EXPECT_TRUE(e.unexcited_axes.size() == 6);
}

namespace {

// Build a well-overlapped 2-camera surround scenario (near-nadir, heavy ground co-visibility)
// with the reference extrinsic slightly off the truth. Returns the orchestrator inputs.
struct SurroundScene {
  std::vector<Image8> store;
  std::vector<std::vector<const Image8*>> per_frame;
  std::vector<Sophus::SE3d> poses;
  Sophus::SE3d T_c1_c0_true;
  Sophus::SE3d T_c1_c0_ref;
};

SurroundScene makeOverlappingScene(const PinholeCamera& cam) {
  SurroundScene s;
  s.T_c1_c0_true = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(-0.15, 0.0, 0.0));
  s.T_c1_c0_ref = s.T_c1_c0_true * Sophus::SE3d::exp(
      (Eigen::Matrix<double, 6, 1>() << 0.02, 0.01, 0.0, 0.005, 0.005, 0.0).finished());
  s.poses = {
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(1.5, -0.1, -0.1)),
                   Eigen::Vector3d(0.0, 0.0, 2.0)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(1.7, 0.1, 0.1)),
                   Eigen::Vector3d(0.3, 0.2, 2.2)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(1.4, -0.12, 0.12)),
                   Eigen::Vector3d(-0.2, 0.3, 1.9)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(1.65, 0.13, -0.13)),
                   Eigen::Vector3d(0.25, -0.2, 2.1))};
  s.store.reserve(s.poses.size() * 2);
  for (const Sophus::SE3d& T_c0_w : s.poses) {
    s.store.push_back(makeGroundImage(cam, T_c0_w, 320, 240));
    s.store.push_back(makeGroundImage(cam, s.T_c1_c0_true * T_c0_w, 320, 240));
    s.per_frame.push_back({&s.store[s.store.size() - 2], &s.store[s.store.size() - 1]});
  }
  return s;
}

OnlineSurroundRigOptions overlappingOpts() {
  OnlineSurroundRigOptions opts;
  opts.bev_grid.x_min = -2.0; opts.bev_grid.x_max = 2.0;
  opts.bev_grid.y_min = -2.0; opts.bev_grid.y_max = 2.0;
  opts.bev_grid.step = 0.1;
  opts.search.rot_scales = {0.02, 0.008, 0.003};
  opts.search.trans_scales = {0.03, 0.01, 0.004};
  opts.search.attempts_per_level = 96;
  opts.min_cost_reduction = 0.001;
  opts.min_overlap_pairs = 50;
  return opts;
}

}  // namespace

CF_TEST(online_surround_rig_emits_behind_observability_gate) {
  // The estimate must clear the OBSERVABILITY gate before emission (RULE #2): on a
  // well-overlapped, excited window the photometric information matrix is full-rank and
  // well-conditioned, so the orchestrator emits with a finite confidence and no weak
  // directions. NOTE: we deliberately do NOT assert the refined extrinsic is closer to truth
  // — the coarse BEV photometric minimum is biased (precision != accuracy), which is exactly
  // why the gate scores confidence rather than trusting the low cost.
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  Eigen::VectorXd intr(4);
  intr << 300.0, 300.0, 160.0, 120.0;
  CameraFactory mk = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };

  const SurroundScene s = makeOverlappingScene(cam);
  OnlineSurroundRigOptions opts = overlappingOpts();
  opts.min_confidence = 1e-6;  // the documented healthy floor (observability.hpp)

  OnlineSurroundRig orch({mk, mk}, {intr, intr}, {s.T_c1_c0_ref}, opts);
  for (std::size_t i = 0; i < s.poses.size(); ++i) orch.addFrame(s.per_frame[i], s.poses[i]);
  const SurroundRigEmission e = orch.tryEmit();

  CF_EXPECT_TRUE(!e.refused_for_motion);
  CF_EXPECT_TRUE(e.overlap_count >= opts.min_overlap_pairs);
  CF_EXPECT_TRUE(e.emitted);
  CF_EXPECT_TRUE(e.observable);
  CF_EXPECT_TRUE(e.confidence >= opts.min_confidence);
  CF_EXPECT_TRUE(e.weak_parameters.empty());
  CF_EXPECT_TRUE(e.extrinsics.size() == 1);
  // The search accepted at least one cost-reducing move (cost_reduction cleared the pre-filter),
  // so the emitted extrinsics differ from the reference and drift is strictly positive. (`>= 0`
  // is a tautology: drift is a sum of norms.)
  CF_EXPECT_TRUE(e.drift > 0.0);
}

CF_TEST(online_surround_rig_refuses_for_observability_when_information_collapses) {
  // RULE #2 teeth: with the SAME well-excited, well-overlapped geometry but FLAT (textureless)
  // images, every BEV agreement residual is identically zero, so the photometric information
  // matrix H = J^T J collapses to ~0 and is NOT observable. The orchestrator must reach the
  // observability gate (we relax the cheap cost/coverage pre-filters so it does) and REFUSE for
  // observability — never emit. A stubbed identity-H gate (the exact failure this test exists to
  // catch) would instead report observable and EMIT garbage extrinsics from a flat image.
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  Eigen::VectorXd intr(4);
  intr << 300.0, 300.0, 160.0, 120.0;
  CameraFactory mk = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };

  const SurroundScene s = makeOverlappingScene(cam);  // excited poses (motion gate passes)
  Image8 flat(320, 240, 128);                          // textureless
  std::vector<const Image8*> flat_imgs = {&flat, &flat};

  OnlineSurroundRigOptions opts = overlappingOpts();
  opts.min_cost_reduction = -1.0;  // a flat image yields zero cost reduction; let it through to
                                   // the observability gate instead of the cost pre-filter.
  opts.min_overlap_pairs = 50;     // flat images still overlap geometrically (~hundreds of pairs)

  OnlineSurroundRig orch({mk, mk}, {intr, intr}, {s.T_c1_c0_ref}, opts);
  for (std::size_t i = 0; i < s.poses.size(); ++i) orch.addFrame(flat_imgs, s.poses[i]);
  const SurroundRigEmission e = orch.tryEmit();

  CF_EXPECT_TRUE(!e.refused_for_motion);          // motion is excited...
  CF_EXPECT_TRUE(!e.refused_for_no_signal);       // ...and there IS geometric overlap...
  CF_EXPECT_TRUE(!e.emitted);                      // ...but the gate refuses:
  CF_EXPECT_TRUE(e.refused_for_observability);     // no photometric information => not observable
  CF_EXPECT_TRUE(!e.observable);
  CF_EXPECT_TRUE(e.confidence == 0.0);
}

CF_TEST(online_surround_rig_refuses_when_confidence_below_threshold) {
  // Same well-conditioned data, but an unreachable min_confidence: the orchestrator must
  // REFUSE for observability (never emit when the estimate cannot clear the required
  // confidence) and surface the weak directions — RULE #2, precision != accuracy.
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  Eigen::VectorXd intr(4);
  intr << 300.0, 300.0, 160.0, 120.0;
  CameraFactory mk = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };

  const SurroundScene s = makeOverlappingScene(cam);
  OnlineSurroundRigOptions opts = overlappingOpts();
  opts.min_confidence = 10.0;  // > 1 (rcond is in [0,1]) => never satisfiable

  OnlineSurroundRig orch({mk, mk}, {intr, intr}, {s.T_c1_c0_ref}, opts);
  for (std::size_t i = 0; i < s.poses.size(); ++i) orch.addFrame(s.per_frame[i], s.poses[i]);
  const SurroundRigEmission e = orch.tryEmit();

  CF_EXPECT_TRUE(!e.emitted);
  CF_EXPECT_TRUE(e.refused_for_observability);
  CF_EXPECT_TRUE(e.observable);   // the matrix is full-rank (rcond > min_reciprocal_condition)...
  CF_EXPECT_TRUE(e.confidence > 0.0);
  CF_EXPECT_TRUE(e.confidence < opts.min_confidence);  // ...but below the required confidence
}

CF_TEST(online_surround_rig_partial_degeneracy_flags_only_rotation_axes) {
  // Translation excited on all 3 axes, rotation held fixed: the per-axis motion gate must flag
  // exactly the 3 rotation axes (not all 6, not the translations) — exercising the per-axis
  // range logic rather than the trivial all-static case.
  PinholeCamera cam(300.0, 300.0, 160.0, 120.0);
  Eigen::VectorXd intr(4);
  intr << 300.0, 300.0, 160.0, 120.0;
  CameraFactory mk = [](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    return std::make_unique<PinholeCamera>(q[0], q[1], q[2], q[3]);
  };

  const Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Vector3d(1.2, 0.0, 0.0));  // fixed rotation
  std::vector<Sophus::SE3d> poses = {
      Sophus::SE3d(R, Eigen::Vector3d(0.0, 0.0, 1.5)),
      Sophus::SE3d(R, Eigen::Vector3d(0.3, 0.0, 1.5)),
      Sophus::SE3d(R, Eigen::Vector3d(0.0, 0.3, 1.5)),
      Sophus::SE3d(R, Eigen::Vector3d(0.0, 0.0, 1.8))};  // all trans ranges 0.3 > 0.10
  const Sophus::SE3d T_c1_c0(Sophus::SO3d(), Eigen::Vector3d(-0.3, 0.0, 0.0));
  Image8 im0 = makeGroundImage(cam, poses[0], 160, 120);
  Image8 im1 = makeGroundImage(cam, T_c1_c0 * poses[0], 160, 120);
  std::vector<const Image8*> frame_imgs = {&im0, &im1};

  OnlineSurroundRig orch({mk, mk}, {intr, intr}, {T_c1_c0});
  for (const Sophus::SE3d& p : poses) orch.addFrame(frame_imgs, p);
  const SurroundRigEmission e = orch.tryEmit();

  CF_EXPECT_TRUE(e.refused_for_motion);
  CF_EXPECT_TRUE(!e.emitted);
  CF_EXPECT_TRUE(e.unexcited_axes.size() == 3);
  int rot = 0;
  for (const std::string& a : e.unexcited_axes)
    if (a == "rot_x" || a == "rot_y" || a == "rot_z") ++rot;
  CF_EXPECT_TRUE(rot == 3);
}
