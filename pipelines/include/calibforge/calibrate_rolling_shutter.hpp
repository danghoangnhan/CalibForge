#pragma once
//
// CalibForge pipeline — rolling-shutter readout-time calibration (header-only, CPU).
//
// Jointly estimates the shared readout time t_r, the per-frame pose and first-order velocity
// twist, and the (shared) intrinsics, by reprojecting each point at its row-specific capture
// time (RollingShutterResidual). t_r=0 recovers the global-shutter calibration. UAV CMOS
// sensors are line-by-line, so this is mandatory there (docs/DESIGN.md §7).

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"  // CameraFactory
#include "calibforge/camera_model.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/rolling_shutter_residual.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

struct RollingShutterView {
  std::vector<Vec3> object_points;
  std::vector<Vec2> image_points;
  std::vector<double> row_fracs;  // capture-time fraction per point = observed_row / image_height
};

struct RollingShutterResult {
  Eigen::VectorXd intrinsics;
  std::vector<Sophus::SE3d> poses;                 // world->cam (frame start), per view
  std::vector<Eigen::Matrix<double, 6, 1>> velocities;  // (rho, omega) per view
  double readout_time = 0.0;                        // t_r (shared)
  LmSummary summary;
  Eigen::MatrixXd information;
  int num_residuals = 0;
};

inline RollingShutterResult calibrateRollingShutter(
    const std::vector<RollingShutterView>& views, const Eigen::VectorXd& intrinsics_init,
    const std::vector<Sophus::SE3d>& poses_init,
    const std::vector<Eigen::Matrix<double, 6, 1>>& velocities_init, double readout_time_init,
    const CameraFactory& make_camera, const LmOptions& opts = LmOptions{},
    bool optimize_velocities = true) {
  // NOTE on observability: in the first-order model only the product t_r * velocity enters, so
  // t_r is unobservable jointly with the velocity scale. The realistic UAV path supplies the
  // per-frame velocity from the IMU (optimize_velocities=false) — then t_r is observable.
  const int nin = static_cast<int>(intrinsics_init.size());
  const int nv = static_cast<int>(views.size());

  std::vector<double> intr(intrinsics_init.data(), intrinsics_init.data() + nin);
  std::array<double, 1> tr{readout_time_init};
  std::vector<std::array<double, 7>> pose(static_cast<std::size_t>(nv));
  std::vector<std::array<double, 6>> vel(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) {
    SE3Param::store(poses_init[i], pose[i].data());
    for (int k = 0; k < 6; ++k) vel[i][k] = velocities_init[i][k];
  }

  DenseProblem problem;
  problem.addParameterBlock(intr.data(), std::make_shared<EuclideanParam>(nin));
  problem.addParameterBlock(tr.data(), std::make_shared<EuclideanParam>(1));
  for (int i = 0; i < nv; ++i) {
    problem.addParameterBlock(pose[i].data(), std::make_shared<SE3Param>());
    problem.addParameterBlock(vel[i].data(), std::make_shared<EuclideanParam>(6));
    if (!optimize_velocities) problem.setParameterBlockConstant(vel[i].data());  // known from IMU
  }

  for (int i = 0; i < nv; ++i) {
    const RollingShutterView& v = views[i];
    for (std::size_t j = 0; j < v.object_points.size(); ++j) {
      problem.addResidualBlock(
          std::make_unique<RollingShutterResidual>(make_camera, nin, v.object_points[j],
                                                   v.image_points[j], v.row_fracs[j]),
          {intr.data(), pose[i].data(), vel[i].data(), tr.data()});
    }
  }

  const LmSummary s = problem.solveLm(opts);

  RollingShutterResult res;
  res.intrinsics = Eigen::Map<const Eigen::VectorXd>(intr.data(), nin);
  res.poses.resize(static_cast<std::size_t>(nv));
  res.velocities.resize(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) {
    res.poses[i] = SE3Param::load(pose[i].data());
    for (int k = 0; k < 6; ++k) res.velocities[i][k] = vel[i][k];
  }
  res.readout_time = tr[0];
  res.summary = s;
  res.information = problem.informationMatrix();
  res.num_residuals = problem.numResiduals();
  return res;
}

}  // namespace calibforge
