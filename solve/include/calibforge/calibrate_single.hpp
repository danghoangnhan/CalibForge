#pragma once
//
// CalibForge pipeline — single-camera calibration (header-only, CPU).
//
// Re-expressed over the solver-agnostic Problem/ResidualBlock interface (issue #7):
// intrinsics are one Euclidean parameter block, each view's pose is one SE(3) block,
// and every observed point is a ReprojectionResidual. A DenseProblem assembles them and
// the manifold Levenberg-Marquardt solves — intrinsics update Euclidean, poses retract on
// SE(3) via T <- T*exp(delta). Jacobians are ANALYTIC (docs/RESEARCH.md Theme 4).
//
// CPU path: per docs/RESEARCH.md Theme 1, best for a single small calibration.

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"          // Vec2, Vec3, CameraModel
#include "calibforge/dense_problem.hpp"          // DenseProblem
#include "calibforge/least_squares.hpp"          // LmOptions, LmSummary
#include "calibforge/manifold.hpp"               // EuclideanParam, SE3Param
#include "calibforge/reprojection_residual.hpp"  // ReprojectionResidual
#include "sophus/se3.hpp"

namespace calibforge {

using Vec6 = Eigen::Matrix<double, 6, 1>;

struct View {
  std::vector<Vec3> object_points;  // 3D board points (world frame)
  std::vector<Vec2> image_points;   // observed pixels (same order)
};

// Builds a camera model from the intrinsic parameter vector.
using CameraFactory = std::function<std::unique_ptr<CameraModel>(const Eigen::VectorXd&)>;

struct SingleCameraResult {
  Eigen::VectorXd intrinsics;
  std::vector<Sophus::SE3d> poses;  // world->camera
  LmSummary summary;
  Eigen::MatrixXd information;  // J^T J at the solution; feed to assessObservability()
  int num_residuals = 0;        // total residual rows (for parameterUncertainty(), #6)
};

inline SingleCameraResult calibrateSingleCamera(
    const std::vector<View>& views,
    const Eigen::VectorXd& intrinsics_init,
    const std::vector<Sophus::SE3d>& poses_init,
    const CameraFactory& make_camera,
    const LmOptions& opts = LmOptions{}) {
  const int nin = static_cast<int>(intrinsics_init.size());
  const int nv = static_cast<int>(views.size());

  // Parameter-block storage (stable addresses for the problem's lifetime):
  // intrinsics (nin doubles) + one 7-double SE(3) per view.
  std::vector<double> intr_store(intrinsics_init.data(), intrinsics_init.data() + nin);
  std::vector<std::array<double, 7>> pose_store(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) SE3Param::store(poses_init[i], pose_store[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr_store.data(), std::make_shared<EuclideanParam>(nin));
  for (int i = 0; i < nv; ++i)
    problem.addParameterBlock(pose_store[i].data(), std::make_shared<SE3Param>());

  for (int i = 0; i < nv; ++i) {
    const View& v = views[i];
    for (std::size_t j = 0; j < v.image_points.size(); ++j) {
      problem.addResidualBlock(
          std::make_unique<ReprojectionResidual>(make_camera, nin, v.object_points[j],
                                                 v.image_points[j]),
          {intr_store.data(), pose_store[i].data()});
    }
  }

  const LmSummary s = problem.solveLm(opts);

  SingleCameraResult res;
  res.intrinsics = Eigen::Map<const Eigen::VectorXd>(intr_store.data(), nin);
  res.poses.resize(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) res.poses[i] = SE3Param::load(pose_store[i].data());
  res.summary = s;
  res.information = problem.informationMatrix();
  res.num_residuals = problem.numResiduals();
  return res;
}

}  // namespace calibforge
