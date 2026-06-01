#pragma once
//
// CalibForge pipeline — stereo (two-camera) calibration (header-only, CPU).
//
// Built on the Problem/ResidualBlock interface (issue #7): per-view world->cam0 poses, two
// intrinsics blocks, and a SHARED cam0->cam1 extrinsic T_c1c0 (Xc1 = T_c1c0 * Xc0). cam0
// observations are ordinary ReprojectionResiduals; cam1 observations are
// StereoReprojectionResiduals composing the extrinsic with the pose. The known board fixes
// the gauge, so no extra datum constraint is needed for target-based stereo.

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"  // CameraFactory
#include "calibforge/camera_model.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/reprojection_residual.hpp"
#include "calibforge/stereo_reprojection_residual.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

struct StereoView {
  std::vector<Vec3> object_points;  // shared board points (world frame)
  std::vector<Vec2> image_points0;  // observed in cam0 (same order)
  std::vector<Vec2> image_points1;  // observed in cam1 (same order)
};

struct StereoResult {
  Eigen::VectorXd intrinsics0;
  Eigen::VectorXd intrinsics1;
  Sophus::SE3d T_cam1_cam0;          // Xc1 = T_cam1_cam0 * Xc0
  std::vector<Sophus::SE3d> poses;   // world -> cam0, per view
  LmSummary summary;
  Eigen::MatrixXd information;
  int num_residuals = 0;
};

inline StereoResult calibrateStereo(
    const std::vector<StereoView>& views,
    const Eigen::VectorXd& intrinsics0_init,
    const Eigen::VectorXd& intrinsics1_init,
    const Sophus::SE3d& T_cam1_cam0_init,
    const std::vector<Sophus::SE3d>& poses_init,
    const CameraFactory& make_camera0,
    const CameraFactory& make_camera1,
    const LmOptions& opts = LmOptions{}) {
  const int nin0 = static_cast<int>(intrinsics0_init.size());
  const int nin1 = static_cast<int>(intrinsics1_init.size());
  const int nv = static_cast<int>(views.size());

  // Stable parameter-block storage.
  std::vector<double> intr0(intrinsics0_init.data(), intrinsics0_init.data() + nin0);
  std::vector<double> intr1(intrinsics1_init.data(), intrinsics1_init.data() + nin1);
  std::array<double, 7> extr{};
  SE3Param::store(T_cam1_cam0_init, extr.data());
  std::vector<std::array<double, 7>> pose(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) SE3Param::store(poses_init[i], pose[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr0.data(), std::make_shared<EuclideanParam>(nin0));
  problem.addParameterBlock(intr1.data(), std::make_shared<EuclideanParam>(nin1));
  problem.addParameterBlock(extr.data(), std::make_shared<SE3Param>());
  for (int i = 0; i < nv; ++i)
    problem.addParameterBlock(pose[i].data(), std::make_shared<SE3Param>());

  for (int i = 0; i < nv; ++i) {
    const StereoView& v = views[i];
    for (std::size_t j = 0; j < v.object_points.size(); ++j) {
      problem.addResidualBlock(
          std::make_unique<ReprojectionResidual>(make_camera0, nin0, v.object_points[j],
                                                 v.image_points0[j]),
          {intr0.data(), pose[i].data()});
      problem.addResidualBlock(
          std::make_unique<StereoReprojectionResidual>(make_camera1, nin1, v.object_points[j],
                                                       v.image_points1[j]),
          {intr1.data(), extr.data(), pose[i].data()});
    }
  }

  const LmSummary s = problem.solveLm(opts);

  StereoResult res;
  res.intrinsics0 = Eigen::Map<const Eigen::VectorXd>(intr0.data(), nin0);
  res.intrinsics1 = Eigen::Map<const Eigen::VectorXd>(intr1.data(), nin1);
  res.T_cam1_cam0 = SE3Param::load(extr.data());
  res.poses.resize(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) res.poses[i] = SE3Param::load(pose[i].data());
  res.summary = s;
  res.information = problem.informationMatrix();
  res.num_residuals = problem.numResiduals();
  return res;
}

}  // namespace calibforge
