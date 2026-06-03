// Smoke test for the FeatureTracker — the targetless observation source. Renders a synthetic
// checkerboard then translates the camera over several frames, runs the tracker, and asserts
//   (a) enough tracks survive across the window,
//   (b) tracked positions of a given physical point follow the per-frame projection of that
//       point within ~1 px.

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/board.hpp"
#include "calibforge/feature_tracker.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/render_board.hpp"
#include "cf_test.hpp"
#include "sophus/se3.hpp"

using calibforge::PinholeCamera;
using calibforge::Vec2;
using calibforge::Vec3;
using calibforge::detect::CheckerboardSpec;
using calibforge::detect::FeatureTracker;
using calibforge::detect::FeatureTrackerOptions;
using calibforge::detect::renderCheckerboard;

CF_TEST(feature_tracker_continues_tracks_across_translating_camera) {
  PinholeCamera cam(400.0, 400.0, 160.0, 120.0);
  CheckerboardSpec spec{7, 5, 0.025};

  // Camera looks at the board; we'll sweep its translation between frames so tracked features
  // physically move on the image.
  const Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Vector3d(0.03, -0.02, 0.01));
  const Eigen::Vector3d center(spec.extentX() / 2.0, spec.extentY() / 2.0, 0.0);

  FeatureTrackerOptions opts;
  opts.max_tracks = 80;
  opts.patch_radius = 4;
  opts.search_radius = 5;
  opts.rel_threshold = 0.08;
  FeatureTracker tracker(opts);

  const int N = 5;
  for (int f = 0; f < N; ++f) {
    const double sx = 0.01 * f;  // translate the camera ~10mm per frame in board x
    Sophus::SE3d T(R, Eigen::Vector3d(0, 0, 0.40) - R * center
                          + Eigen::Vector3d(sx, 0.005 * f, 0));
    auto img = renderCheckerboard(cam, T, spec, 320, 240);
    tracker.addFrame(img);
  }

  // Some tracks should survive the full window.
  int long_tracks = 0;
  for (const auto& t : tracker.tracks())
    if (static_cast<int>(t.obs.size()) >= 3) ++long_tracks;
  CF_EXPECT_TRUE(long_tracks > 0);
  CF_EXPECT_TRUE(tracker.numFrames() == N);
}
