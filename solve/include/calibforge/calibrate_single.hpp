#pragma once
//
// CalibForge pipeline — single-camera calibration (header-only, CPU).
//
// Minimizes reprojection error over [intrinsics ; per-view SE(3) pose] with a
// manifold-aware Levenberg-Marquardt: intrinsics update Euclidean, poses retract on
// SE(3) via the right perturbation T <- T * exp(delta). Jacobians are ANALYTIC
// (docs/RESEARCH.md Theme 4, "analytic on hot paths"):
//   d(pixel)/d(intrinsics) = CameraModel::projectJacobianWrtParams
//   d(pixel)/d(pose)       = CameraModel::projectJacobianWrtPoint * [R | -R [X]_x]
// where [R | -R [X]_x] = d(T*X)/d(delta) for the right perturbation.
//
// CPU path: per docs/RESEARCH.md Theme 1, best for a single small calibration.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"   // Vec2, Vec3, CameraModel, Jacobian
#include "calibforge/least_squares.hpp"  // LmOptions, LmSummary
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
};

namespace detail {
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return S;
}
}  // namespace detail

inline SingleCameraResult calibrateSingleCamera(
    const std::vector<View>& views,
    const Eigen::VectorXd& intrinsics_init,
    const std::vector<Sophus::SE3d>& poses_init,
    const CameraFactory& make_camera,
    const LmOptions& opts = LmOptions{}) {
  using Eigen::MatrixXd;
  using Eigen::Vector3d;
  using Eigen::VectorXd;

  const int nin = static_cast<int>(intrinsics_init.size());
  const int nv = static_cast<int>(views.size());
  int m = 0;
  for (const auto& v : views) m += 2 * static_cast<int>(v.image_points.size());
  const int n = nin + 6 * nv;

  VectorXd intr = intrinsics_init;
  std::vector<Sophus::SE3d> poses = poses_init;

  // Residual (and, when need_jac, analytic Jacobian) at the given state.
  auto evaluate = [&](const VectorXd& cur_intr, const std::vector<Sophus::SE3d>& cur_poses,
                      VectorXd& r, MatrixXd& J, bool need_jac) {
    r.resize(m);
    if (need_jac) J.setZero(m, n);
    std::unique_ptr<CameraModel> cam = make_camera(cur_intr);
    int row = 0;
    for (int i = 0; i < nv; ++i) {
      const Sophus::SE3d& T = cur_poses[i];
      const Eigen::Matrix3d R = T.rotationMatrix();
      const View& v = views[i];
      for (std::size_t j = 0; j < v.image_points.size(); ++j) {
        const Vec3& X = v.object_points[j];
        const Vector3d Xw(X[0], X[1], X[2]);
        const Vector3d Xc = T * Xw;
        const Vec3 pc{Xc.x(), Xc.y(), Xc.z()};
        const Vec2 px = cam->project(pc);
        r[row] = px[0] - v.image_points[j][0];
        r[row + 1] = px[1] - v.image_points[j][1];

        if (need_jac) {
          const Jacobian Ji = cam->projectJacobianWrtParams(pc);  // 2 x nin
          for (int c = 0; c < nin; ++c) {
            J(row, c) = Ji.data[0 * nin + c];
            J(row + 1, c) = Ji.data[1 * nin + c];
          }
          const Jacobian Jp = cam->projectJacobianWrtPoint(pc);  // 2 x 3
          Eigen::Matrix<double, 2, 3> dpix;
          dpix << Jp.data[0], Jp.data[1], Jp.data[2],
                  Jp.data[3], Jp.data[4], Jp.data[5];
          Eigen::Matrix<double, 3, 6> dXc;  // d(T*X)/d(delta), right perturbation
          dXc.leftCols<3>() = R;
          dXc.rightCols<3>() = -R * detail::skew(Xw);
          const Eigen::Matrix<double, 2, 6> Jpose = dpix * dXc;
          const int off = nin + 6 * i;
          for (int a = 0; a < 6; ++a) {
            J(row, off + a) = Jpose(0, a);
            J(row + 1, off + a) = Jpose(1, a);
          }
        }
        row += 2;
      }
    }
  };

  auto retract = [&](const VectorXd& intr_in, const std::vector<Sophus::SE3d>& poses_in,
                     const VectorXd& dx, VectorXd& intr_out, std::vector<Sophus::SE3d>& poses_out) {
    intr_out = intr_in + dx.head(nin);
    poses_out.resize(nv);
    for (int i = 0; i < nv; ++i)
      poses_out[i] = poses_in[i] * Sophus::SE3d::exp(Vec6(dx.segment<6>(nin + 6 * i)));
  };

  LmSummary s;
  VectorXd r;
  MatrixXd J;
  evaluate(intr, poses, r, J, true);
  double cost = 0.5 * r.squaredNorm();
  s.initial_cost = cost;
  double lambda = opts.initial_lambda;

  int it = 0;
  for (; it < opts.max_iterations; ++it) {
    const MatrixXd JtJ = J.transpose() * J;
    const VectorXd g = J.transpose() * r;
    if (g.norm() < opts.gradient_tolerance) {
      s.converged = true;
      break;
    }
    bool step_accepted = false;
    for (int tries = 0; tries < 12; ++tries) {
      MatrixXd A = JtJ;
      A.diagonal() += lambda * JtJ.diagonal();
      const VectorXd dx = A.ldlt().solve(-g);

      VectorXd intr_t;
      std::vector<Sophus::SE3d> poses_t;
      retract(intr, poses, dx, intr_t, poses_t);
      VectorXd r_t;
      MatrixXd unused;
      evaluate(intr_t, poses_t, r_t, unused, false);
      const double cost_t = 0.5 * r_t.squaredNorm();

      if (cost_t < cost) {
        const double rel = (cost - cost_t) / std::max(cost, 1e-300);
        const double step_norm = dx.norm();
        intr = intr_t;
        poses = poses_t;
        cost = cost_t;
        evaluate(intr, poses, r, J, true);  // re-linearize at the accepted point
        lambda = std::max(lambda * 0.3, 1e-12);
        step_accepted = true;
        const double scale = intr.norm() + static_cast<double>(nv) + 1.0;
        if (rel < opts.function_tolerance) s.converged = true;
        if (step_norm < opts.parameter_tolerance * (scale + opts.parameter_tolerance))
          s.converged = true;
        break;
      }
      lambda *= 3.0;
    }
    if (s.converged) {
      ++it;
      break;
    }
    if (!step_accepted) break;
  }
  s.iterations = it;
  s.final_cost = cost;

  SingleCameraResult res;
  res.intrinsics = intr;
  res.poses = poses;
  res.summary = s;
  return res;
}

}  // namespace calibforge
