#pragma once
//
// CalibForge pipeline — N-camera rig calibration (header-only, CPU).
//
// Generalizes stereo (v0.2): a reference camera cam0, N-1 shared rigid extrinsics T_ck_c0
// (Xck = T_ck_c0 * Xc0), per-view world→cam0 poses, and per-camera intrinsics. cam0
// observations are ordinary ReprojectionResiduals; every other camera reuses the
// StereoReprojectionResidual (which already composes one extrinsic through the pose). The
// known board fixes the gauge. A camera that doesn't see the board in a view contributes
// no residuals for that view (leave its image_points empty).

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"  // CameraFactory, View
#include "calibforge/camera_model.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/reprojection_residual.hpp"
#include "calibforge/stereo_reprojection_residual.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

struct RigView {
  std::vector<Vec3> object_points;               // shared board points (world frame)
  // Per-camera observations, indexed [camera][point], aligned with object_points. A camera
  // that doesn't see the board this view has an empty vector.
  std::vector<std::vector<Vec2>> image_points;   // size = num_cameras
};

struct RigResult {
  std::vector<Eigen::VectorXd> intrinsics;       // per camera (size N)
  std::vector<Sophus::SE3d> extrinsics;          // T_ck_c0 for k=1..N-1 (size N-1)
  std::vector<Sophus::SE3d> poses;               // world->cam0, per view
  LmSummary summary;
  Eigen::MatrixXd information;
  int num_residuals = 0;
};

inline RigResult calibrateRig(
    const std::vector<RigView>& views,
    const std::vector<Eigen::VectorXd>& intrinsics_init,        // size N
    const std::vector<Sophus::SE3d>& extrinsics_init,           // size N-1, T_ck_c0
    const std::vector<Sophus::SE3d>& poses_init,                // size num_views, world->cam0
    const std::vector<CameraFactory>& make_cameras,             // size N
    const LmOptions& opts = LmOptions{},
    bool intrinsics_fixed = false) {
  const int ncam = static_cast<int>(intrinsics_init.size());
  const int nv = static_cast<int>(views.size());

  // Stable parameter-block storage.
  std::vector<std::vector<double>> intr(ncam);
  for (int c = 0; c < ncam; ++c)
    intr[c].assign(intrinsics_init[c].data(), intrinsics_init[c].data() + intrinsics_init[c].size());
  std::vector<std::array<double, 7>> extr(static_cast<std::size_t>(ncam - 1));
  for (int k = 0; k < ncam - 1; ++k) SE3Param::store(extrinsics_init[k], extr[k].data());
  std::vector<std::array<double, 7>> pose(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) SE3Param::store(poses_init[i], pose[i].data());

  DenseProblem problem;
  for (int c = 0; c < ncam; ++c) {
    problem.addParameterBlock(intr[c].data(),
                              std::make_shared<EuclideanParam>(static_cast<int>(intr[c].size())));
    if (intrinsics_fixed) problem.setParameterBlockConstant(intr[c].data());
  }
  for (int k = 0; k < ncam - 1; ++k)
    problem.addParameterBlock(extr[k].data(), std::make_shared<SE3Param>());
  for (int i = 0; i < nv; ++i)
    problem.addParameterBlock(pose[i].data(), std::make_shared<SE3Param>());

  for (int i = 0; i < nv; ++i) {
    const RigView& v = views[i];
    for (int c = 0; c < ncam && c < static_cast<int>(v.image_points.size()); ++c) {
      const std::vector<Vec2>& obs = v.image_points[c];
      const int nin = static_cast<int>(intr[c].size());
      for (std::size_t j = 0; j < obs.size() && j < v.object_points.size(); ++j) {
        if (c == 0) {
          problem.addResidualBlock(
              std::make_unique<ReprojectionResidual>(make_cameras[0], nin, v.object_points[j],
                                                     obs[j]),
              {intr[0].data(), pose[i].data()});
        } else {
          problem.addResidualBlock(
              std::make_unique<StereoReprojectionResidual>(make_cameras[c], nin,
                                                           v.object_points[j], obs[j]),
              {intr[c].data(), extr[c - 1].data(), pose[i].data()});
        }
      }
    }
  }

  const LmSummary s = problem.solveLm(opts);

  RigResult res;
  res.intrinsics.resize(static_cast<std::size_t>(ncam));
  for (int c = 0; c < ncam; ++c)
    res.intrinsics[c] = Eigen::Map<const Eigen::VectorXd>(intr[c].data(),
                                                          static_cast<int>(intr[c].size()));
  res.extrinsics.resize(static_cast<std::size_t>(ncam - 1));
  for (int k = 0; k < ncam - 1; ++k) res.extrinsics[k] = SE3Param::load(extr[k].data());
  res.poses.resize(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) res.poses[i] = SE3Param::load(pose[i].data());
  res.summary = s;
  res.information = problem.informationMatrix();
  res.num_residuals = problem.numResiduals();
  return res;
}

}  // namespace calibforge
