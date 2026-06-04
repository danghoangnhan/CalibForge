// Tests for detect/triangulate.hpp — multi-view linear DLT triangulation + FeatureTrack pack.

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"  // View
#include "calibforge/double_sphere_camera.hpp"
#include "calibforge/feature_tracker.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/triangulate.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::CameraModel;
using calibforge::DoubleSphereCamera;
using calibforge::PinholeCamera;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::View;
using calibforge::detect::FeatureObservation;
using calibforge::detect::FeatureTrack;
using calibforge::detect::TriangulatedTrack;
using calibforge::detect::TriangulationResult;

CF_TEST(triangulate_linear_recovers_world_point_from_3_views) {
  PinholeCamera cam(500.0, 500.0, 320.0, 240.0);
  const Vec3 X_world{0.5, -0.2, 4.0};

  // Three viewing poses with translation baselines, looking roughly toward +Z.
  std::vector<Sophus::SE3d> T_world_cam = {
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.0)),
                   Eigen::Vector3d(0.3, 0.0, 0.05)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(-0.01, 0.03, 0.005)),
                   Eigen::Vector3d(-0.25, 0.1, -0.05))};

  std::vector<Vec2> uvs;
  std::vector<const CameraModel*> cams;
  for (const Sophus::SE3d& T : T_world_cam) {
    const Eigen::Vector3d Xc =
        T.inverse() * Eigen::Vector3d(X_world[0], X_world[1], X_world[2]);
    uvs.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    cams.push_back(&cam);
  }

  const TriangulationResult r =
      calibforge::detect::triangulateLinear(uvs, cams, T_world_cam);
  CF_EXPECT_TRUE(r.ok);
  CF_EXPECT_NEAR(r.point_world[0], X_world[0], 1e-6);
  CF_EXPECT_NEAR(r.point_world[1], X_world[1], 1e-6);
  CF_EXPECT_NEAR(r.point_world[2], X_world[2], 1e-6);
}

CF_TEST(triangulate_tracks_packs_into_per_frame_views) {
  PinholeCamera cam(500.0, 500.0, 320.0, 240.0);
  const std::vector<Vec3> Xs_world = {{0.4, -0.2, 4.0}, {-0.3, 0.1, 5.0}, {0.0, 0.0, 6.0}};

  std::vector<Sophus::SE3d> T_world_cam = {
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0)),
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.2, 0.0, 0.0)),
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.4, 0.05, 0.0)),
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.6, 0.0, -0.05))};

  // Build one FeatureTrack per landmark, each observed across all 4 frames.
  std::vector<FeatureTrack> tracks;
  for (int i = 0; i < static_cast<int>(Xs_world.size()); ++i) {
    FeatureTrack t;
    t.id = i;
    for (int f = 0; f < static_cast<int>(T_world_cam.size()); ++f) {
      const Eigen::Vector3d Xc = T_world_cam[f].inverse() *
                                  Eigen::Vector3d(Xs_world[i][0], Xs_world[i][1], Xs_world[i][2]);
      t.obs.push_back({f, cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()})});
    }
    tracks.push_back(std::move(t));
  }

  const std::vector<TriangulatedTrack> tt =
      calibforge::detect::triangulateTracks(tracks, &cam, T_world_cam, 3);
  CF_EXPECT_TRUE(tt.size() == 3);
  for (std::size_t i = 0; i < tt.size(); ++i) {
    CF_EXPECT_NEAR(tt[i].point_world[0], Xs_world[i][0], 1e-5);
    CF_EXPECT_NEAR(tt[i].point_world[1], Xs_world[i][1], 1e-5);
    CF_EXPECT_NEAR(tt[i].point_world[2], Xs_world[i][2], 1e-5);
  }

  const std::vector<View> views = calibforge::detect::packMonocularViews(
      tt, static_cast<int>(T_world_cam.size()));
  CF_EXPECT_TRUE(views.size() == T_world_cam.size());
  for (const View& v : views) {
    CF_EXPECT_TRUE(v.object_points.size() == Xs_world.size());
    CF_EXPECT_TRUE(v.image_points.size() == Xs_world.size());
  }
}

CF_TEST(triangulate_linear_rejects_rank_deficient_zero_baseline) {
  PinholeCamera cam(500.0, 500.0, 320.0, 240.0);
  const Vec3 X_world{0.5, -0.2, 4.0};

  // Two views at the same translation (zero baseline) — same ray-system; rank deficient.
  std::vector<Sophus::SE3d> T_world_cam = {
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0)),
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0))};
  std::vector<Vec2> uvs;
  std::vector<const CameraModel*> cams;
  for (const Sophus::SE3d& T : T_world_cam) {
    const Eigen::Vector3d Xc =
        T.inverse() * Eigen::Vector3d(X_world[0], X_world[1], X_world[2]);
    uvs.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    cams.push_back(&cam);
  }
  const TriangulationResult r =
      calibforge::detect::triangulateLinear(uvs, cams, T_world_cam, /*cond=*/1e-3);
  CF_EXPECT_TRUE(!r.ok);
}

CF_TEST(triangulate_linear_accepts_wide_fov_landmark_with_negative_camera_z) {
  // The model-agnostic cheirality (X_cam . unproject_ray > 0) must ACCEPT a landmark genuinely
  // in front of a >180deg-FOV lens even when its camera-frame z is NEGATIVE — the case the old
  // X_cam.z()>0 check wrongly rejected. A PinholeCamera cannot exercise this (its unproject rays
  // always have z>0, so dot>0 <=> z>0); use a wide-FOV double-sphere lens whose unproject can
  // return a backward-tilted ray.
  DoubleSphereCamera cam(200.0, 200.0, 160.0, 120.0, /*xi=*/0.3, /*alpha=*/0.6);
  // Landmark in cam0's frame: in front of the double-sphere surface (projectValid) yet z<0.
  const Eigen::Vector3d X_c0 = 4.0 * Eigen::Vector3d(0.9, 0.0, -0.2).normalized();
  CF_EXPECT_TRUE(cam.projectValid(Vec3{X_c0.x(), X_c0.y(), X_c0.z()}));
  CF_EXPECT_TRUE(X_c0.z() < 0.0);
  const Vec3 X_world{X_c0.x(), X_c0.y(), X_c0.z()};  // world == cam0 frame (identity pose)

  std::vector<Sophus::SE3d> T_world_cam = {
      Sophus::SE3d(),                                                            // cam0 = identity
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.4, 0.0, 0.0)),              // x baseline
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.3, 0.0))};             // y baseline
  std::vector<Vec2> uvs;
  std::vector<const CameraModel*> cams;
  for (const Sophus::SE3d& T : T_world_cam) {
    const Eigen::Vector3d Xc = T.inverse() * X_c0;
    CF_EXPECT_TRUE(Xc.z() < 0.0);  // negative camera-frame z in EVERY view (old z>0 would reject)
    uvs.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    cams.push_back(&cam);
  }

  const TriangulationResult r =
      calibforge::detect::triangulateLinear(uvs, cams, T_world_cam);
  CF_EXPECT_TRUE(r.ok);  // accepted via the model-agnostic dot-product cheirality
  CF_EXPECT_NEAR(r.point_world[0], X_world[0], 1e-4);
  CF_EXPECT_NEAR(r.point_world[1], X_world[1], 1e-4);
  CF_EXPECT_NEAR(r.point_world[2], X_world[2], 1e-4);
}

CF_TEST(triangulate_linear_conditioning_gate_rejects_when_threshold_exceeds_condition) {
  // The zero-baseline test now returns at the PARALLAX gate (parallax 0 < 1deg) BEFORE the
  // conditioning gate, leaving `out.condition < condition_threshold` uncovered with parallax
  // passing. Exercise it directly: a well-parallaxed point has a finite condition in (0,1];
  // raising condition_threshold ABOVE that value (with the parallax floor wide open) must reject
  // via the conditioning gate — catching a flipped comparison or an sv(0)/sv(last) inversion the
  // parallax-shadowed zero-baseline test cannot.
  PinholeCamera cam(500.0, 500.0, 320.0, 240.0);
  const Vec3 X_world{0.5, -0.2, 4.0};
  std::vector<Sophus::SE3d> T_world_cam = {
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.01, 0.0)),
                   Eigen::Vector3d(0.3, 0.0, 0.05)),
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(-0.01, 0.03, 0.005)),
                   Eigen::Vector3d(-0.25, 0.1, -0.05))};
  std::vector<Vec2> uvs;
  std::vector<const CameraModel*> cams;
  for (const Sophus::SE3d& T : T_world_cam) {
    const Eigen::Vector3d Xc = T.inverse() * Eigen::Vector3d(X_world[0], X_world[1], X_world[2]);
    uvs.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    cams.push_back(&cam);
  }

  // Parallax floor wide open (0) so ONLY the conditioning gate can reject.
  const TriangulationResult ok = calibforge::detect::triangulateLinear(
      uvs, cams, T_world_cam, /*condition_threshold=*/1e-9, /*min_parallax_rad=*/0.0);
  CF_EXPECT_TRUE(ok.ok);
  CF_EXPECT_TRUE(ok.condition > 0.0 && ok.condition <= 1.0);  // valid rcond range (catches inversion)

  // Same geometry, threshold raised above the measured condition -> conditioning gate rejects.
  const TriangulationResult rej = calibforge::detect::triangulateLinear(
      uvs, cams, T_world_cam, /*condition_threshold=*/ok.condition * 10.0, /*min_parallax_rad=*/0.0);
  CF_EXPECT_TRUE(!rej.ok);
}

CF_TEST(triangulate_linear_rejects_low_parallax_even_when_conditioning_passes) {
  // A far point seen across a tiny baseline: the depth is essentially unconstrained, but the
  // reciprocal-condition number alone does NOT catch this. With the conditioning gate set
  // wide open (1e-12) the geometric parallax gate must still reject the near-parallel rays.
  PinholeCamera cam(500.0, 500.0, 320.0, 240.0);
  const Vec3 X_world{0.1, -0.05, 50.0};  // 50 m away

  std::vector<Sophus::SE3d> T_world_cam = {
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.0, 0.0, 0.0)),
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(0.01, 0.0, 0.0))};  // 1 cm baseline => ~0.01 deg
  std::vector<Vec2> uvs;
  std::vector<const CameraModel*> cams;
  for (const Sophus::SE3d& T : T_world_cam) {
    const Eigen::Vector3d Xc =
        T.inverse() * Eigen::Vector3d(X_world[0], X_world[1], X_world[2]);
    uvs.push_back(cam.project(Vec3{Xc.x(), Xc.y(), Xc.z()}));
    cams.push_back(&cam);
  }
  const TriangulationResult r = calibforge::detect::triangulateLinear(
      uvs, cams, T_world_cam, /*cond=*/1e-12, /*min_parallax_rad=*/0.017453292519943295);
  CF_EXPECT_TRUE(!r.ok);
  CF_EXPECT_TRUE(r.parallax_rad < 0.017453292519943295);  // below the 1 deg floor
}
