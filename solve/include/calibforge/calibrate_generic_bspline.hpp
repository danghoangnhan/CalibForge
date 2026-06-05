#pragma once
//
// CalibForge pipeline — generic per-pixel B-spline calibration (header-only, CPU).
//
// Wires the GenericBSplineCamera (core/generic_bspline_camera.hpp, Schöps CVPR 2020) into the
// SAME solver-agnostic DenseProblem the parametric models use: the 3*Nx*Ny control-point
// directions ARE the intrinsic parameter block, each view pose is an SE(3) block, and every
// observation is a ReprojectionResidual that calls the model's analytic project + param/point
// Jacobians. This is the v1.0 follow-up that takes the B-spline model from "header-only, not
// wired into any pipeline" to a real estimator (issue #25).
//
// GAUGE NOTE: unproject() NORMALIZES the blended ray, so the radial magnitude of each control
// point is unobservable (a per-control-point gauge), and when poses are free a global rotation
// trades against all rays. The Marquardt diagonal damping regularizes these null directions, so
// the solve still drives the REPROJECTION error down and lands on a gauge representative — the
// model is validated by functional equivalence (it reprojects like the source), not by matching
// control points pointwise. Use optimize_poses=false (poses known from a prior parametric solve)
// for a clean functional fit; optimize_poses=true does the full joint refinement.
//
// Typical init: GenericBSplineCamera(grid).fitFromParametricCamera(source) then pass its
// params() as control_init — see makeGenericBSplineInit() below.

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/calibrate_single.hpp"        // View, CameraFactory
#include "calibforge/camera_model.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/generic_bspline_camera.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/manifold.hpp"
#include "calibforge/reprojection_residual.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

struct GenericBSplineCalibrateOptions {
  LmOptions lm;
  bool optimize_poses = true;  // false => hold per-view poses constant (functional fit to a field)
};

struct GenericBSplineResult {
  GenericBSplineGrid grid;
  std::vector<double> control_points;  // optimized flat param vector (3*nx*ny)
  std::vector<Sophus::SE3d> poses;     // world->camera per view
  LmSummary summary;
  Eigen::MatrixXd information;         // J^T J at the solution (control + pose tangents)
  int num_residuals = 0;
  double rms_reprojection_px = 0.0;
};

// CameraFactory that rebuilds a GenericBSplineCamera over `grid` from a flat param vector — the
// adapter that lets a dense B-spline ride the standard ReprojectionResidual.
inline CameraFactory makeGenericBSplineFactory(const GenericBSplineGrid& grid) {
  return [grid](const Eigen::VectorXd& q) -> std::unique_ptr<CameraModel> {
    auto cam = std::make_unique<GenericBSplineCamera>(grid);
    cam->setParams(std::vector<double>(q.data(), q.data() + q.size()));
    return cam;
  };
}

// Convenience: fit a grid to a parametric source camera (the standard initial guess).
inline std::vector<double> makeGenericBSplineInit(const GenericBSplineGrid& grid,
                                                  const CameraModel& source) {
  GenericBSplineCamera cam(grid);
  cam.fitFromParametricCamera(source);
  return cam.params();
}

inline GenericBSplineResult calibrateGenericBSpline(
    const std::vector<View>& views, const GenericBSplineGrid& grid,
    const std::vector<double>& control_init,
    const std::vector<Sophus::SE3d>& poses_init,
    const GenericBSplineCalibrateOptions& opts = GenericBSplineCalibrateOptions{}) {
  const int nin = 3 * grid.nx * grid.ny;
  const int nv = static_cast<int>(views.size());

  std::vector<double> intr = control_init;
  std::vector<std::array<double, 7>> pose_store(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) SE3Param::store(poses_init[i], pose_store[i].data());

  DenseProblem problem;
  problem.addParameterBlock(intr.data(), std::make_shared<EuclideanParam>(nin));
  for (int i = 0; i < nv; ++i) {
    problem.addParameterBlock(pose_store[i].data(), std::make_shared<SE3Param>());
    if (!opts.optimize_poses) problem.setParameterBlockConstant(pose_store[i].data());
  }

  const CameraFactory factory = makeGenericBSplineFactory(grid);
  for (int i = 0; i < nv; ++i) {
    const View& v = views[i];
    for (std::size_t j = 0; j < v.image_points.size(); ++j)
      problem.addResidualBlock(
          std::make_unique<ReprojectionResidual>(factory, nin, v.object_points[j],
                                                 v.image_points[j]),
          {intr.data(), pose_store[i].data()});
  }

  GenericBSplineResult res;
  res.summary = problem.solveLm(opts.lm);
  res.grid = grid;
  res.control_points = intr;
  res.poses.resize(static_cast<std::size_t>(nv));
  for (int i = 0; i < nv; ++i) res.poses[i] = SE3Param::load(pose_store[i].data());
  res.information = problem.informationMatrix();
  res.num_residuals = problem.numResiduals();
  res.rms_reprojection_px =
      (res.num_residuals > 0) ? std::sqrt(2.0 * res.summary.final_cost / res.num_residuals) : 0.0;
  return res;
}

}  // namespace calibforge
