#pragma once
//
// CalibForge detect — sparse feature tracker for targetless online calibration (header-only).
//
// Provides the targetless observation source that epic #24 wires behind the observability
// gate. The tracker emits FeatureTrack{id, std::vector<Observation>{frame_id, uv}} — N
// continuous 2D observations of the same physical point across N consecutive frames. The
// online tracker (OnlineExtrinsicTracker / OnlineIntrinsicTracker) consumes these.
//
// Implementation choices:
//   * Detector: the existing saddle-corner detector in corner_detect.hpp — already used by
//     the checkerboard path. Sub-pixel via the same quadratic refinement.
//   * Tracking: bilinear-sampling 1D-search per-axis (cheap "Lucas-Kanade lite"), since the
//     existing Image8 has a bilinear sampler. Good enough for clean / rendered images;
//     real-image robustness is the optional OpenCV path (future).
//   * Track lifetime: a track survives so long as the search per frame stays in-image and
//     intensity-similar to its template; dropped otherwise.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "calibforge/camera_model.hpp"  // Vec2
#include "calibforge/corner_detect.hpp"  // detectSaddleCorners, refineCornerSaddle
#include "calibforge/image.hpp"

namespace calibforge {
namespace detect {

struct FeatureObservation {
  int frame_id = 0;
  Vec2 uv{0.0, 0.0};
};

struct FeatureTrack {
  int id = 0;
  std::vector<FeatureObservation> obs;
  // Templated patch (intensity values around the corner at first detection), used as the
  // tracking signal in subsequent frames.
  std::vector<double> patch;
  int patch_radius = 5;
};

struct FeatureTrackerOptions {
  int max_tracks = 200;
  int patch_radius = 5;
  double rel_threshold = 0.05;   // saddle-response threshold for new-track detection
  int nms_radius = 4;
  double match_intensity_tol = 30.0;  // mean-abs-diff threshold; drop track on exceed
  int search_radius = 6;              // local search window for tracking
};

namespace detail {
// Mean absolute intensity difference between a template patch and an image patch centered at
// (cx, cy). Bilinear-sampled. Returns +inf if the window leaves the image.
inline double patchMad(const Image8& im, const std::vector<double>& tmpl, double cx, double cy,
                       int R) {
  const int side = 2 * R + 1;
  if (cx - R < 0 || cy - R < 0 || cx + R > im.width - 1 || cy + R > im.height - 1)
    return std::numeric_limits<double>::infinity();
  double s = 0.0;
  int n = 0;
  for (int dy = -R; dy <= R; ++dy) {
    for (int dx = -R; dx <= R; ++dx) {
      const double t = tmpl[static_cast<std::size_t>(dy + R) * side + (dx + R)];
      const double i = im.bilinear(cx + dx, cy + dy);
      s += std::fabs(i - t);
      ++n;
    }
  }
  return s / n;
}

inline std::vector<double> samplePatch(const Image8& im, double cx, double cy, int R) {
  const int side = 2 * R + 1;
  std::vector<double> p(static_cast<std::size_t>(side) * side, 0.0);
  for (int dy = -R; dy <= R; ++dy)
    for (int dx = -R; dx <= R; ++dx)
      p[static_cast<std::size_t>(dy + R) * side + (dx + R)] = im.bilinear(cx + dx, cy + dy);
  return p;
}
}  // namespace detail

// Stateful sparse feature tracker. Call addFrame(im) once per incoming image; tracks that
// survive the current frame have their observations appended. tracks() returns the
// currently-living tracks for inspection / read-out into the solver.
class FeatureTracker {
 public:
  explicit FeatureTracker(FeatureTrackerOptions opts = {}) : opts_(opts) {}

  void addFrame(const Image8& im) {
    const int fid = next_frame_++;

    // Track existing features into the new image via a small local search around the last
    // known position. Patches with the lowest MAD win; if best > tol, drop the track.
    std::vector<FeatureTrack> survivors;
    survivors.reserve(tracks_.size());
    const int R = opts_.patch_radius;
    const int S = opts_.search_radius;
    for (FeatureTrack& t : tracks_) {
      const FeatureObservation& last = t.obs.back();
      double best_mad = std::numeric_limits<double>::infinity();
      double best_x = last.uv[0], best_y = last.uv[1];
      for (int dy = -S; dy <= S; ++dy) {
        for (int dx = -S; dx <= S; ++dx) {
          const double cx = last.uv[0] + dx;
          const double cy = last.uv[1] + dy;
          const double mad = detail::patchMad(im, t.patch, cx, cy, R);
          if (mad < best_mad) { best_mad = mad; best_x = cx; best_y = cy; }
        }
      }
      if (best_mad > opts_.match_intensity_tol) continue;  // drop
      // Sub-pixel refine the match against the local response surface.
      Corner2 refined = refineCornerSaddle(im, best_x, best_y, R);
      // refineCornerSaddle returns (best_x, best_y) on failure; refine cap doesn't go far.
      t.obs.push_back({fid, Vec2{refined.x, refined.y}});
      survivors.push_back(std::move(t));
    }
    tracks_ = std::move(survivors);

    // Detect new corners and seed tracks if we're under the budget. We keep new corners that
    // are at least patch_radius from any existing track to avoid duplicates.
    if (static_cast<int>(tracks_.size()) < opts_.max_tracks) {
      const std::vector<Corner2> corners =
          detectSaddleCorners(im, opts_.rel_threshold, opts_.nms_radius);
      for (const Corner2& c : corners) {
        if (static_cast<int>(tracks_.size()) >= opts_.max_tracks) break;
        bool near_existing = false;
        for (const FeatureTrack& t : tracks_) {
          const FeatureObservation& last = t.obs.back();
          const double dx = last.uv[0] - c.x;
          const double dy = last.uv[1] - c.y;
          if (dx * dx + dy * dy < (R * R)) { near_existing = true; break; }
        }
        if (near_existing) continue;
        FeatureTrack t;
        t.id = next_track_++;
        t.patch_radius = R;
        t.patch = detail::samplePatch(im, c.x, c.y, R);
        t.obs.push_back({fid, Vec2{c.x, c.y}});
        tracks_.push_back(std::move(t));
      }
    }
  }

  const std::vector<FeatureTrack>& tracks() const { return tracks_; }
  std::size_t numTracks() const { return tracks_.size(); }
  int numFrames() const { return next_frame_; }

 private:
  FeatureTrackerOptions opts_;
  std::vector<FeatureTrack> tracks_;
  int next_track_ = 0;
  int next_frame_ = 0;
};

}  // namespace detect
}  // namespace calibforge
