#pragma once
//
// CalibForge pipeline — hand-eye calibration (robot-world / hand-eye, header-only, CPU).
//
// A camera (known intrinsics) on a moving robot observes a fixed target. Given the robot's
// gripper-in-base pose per view (forward kinematics) and the target observations, estimate
// X = T_gripper_cam (hand-eye) and Z = T_base_target via reprojection on the DenseProblem.
// The known target board fixes the gauge; degenerate motion (too few / parallel-axis views)
// is caught by the observability gate.
//
// Note: a closed-form Tsai/Park AX=XB initializer would widen the convergence basin; for
// well-conditioned motion a perturbed initial guess converges directly (added as future work).

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/camera_model.hpp"
#include "calibforge/dense_problem.hpp"
#include "calibforge/hand_eye_residual.hpp"
#include "calibforge/least_squares.hpp"
#include "calibforge/manifold.hpp"
#include "sophus/se3.hpp"

namespace calibforge {

struct HandEyeView {
  Sophus::SE3d T_base_gripper;        // known gripper pose in base (robot forward kinematics)
  std::vector<Vec3> object_points;    // target points (target frame)
  std::vector<Vec2> image_points;     // observed pixels (same order)
};

struct HandEyeResult {
  Sophus::SE3d X;  // T_gripper_cam (hand-eye)
  Sophus::SE3d Z;  // T_base_target
  LmSummary summary;
  Eigen::MatrixXd information;  // tangent-space J^T J over [X, Z]
  int num_residuals = 0;
};

inline HandEyeResult calibrateHandEye(const std::vector<HandEyeView>& views,
                                      const Sophus::SE3d& X_init, const Sophus::SE3d& Z_init,
                                      const CameraModel& camera,
                                      const LmOptions& opts = LmOptions{}) {
  std::array<double, 7> X{}, Z{};
  SE3Param::store(X_init, X.data());
  SE3Param::store(Z_init, Z.data());

  DenseProblem problem;
  problem.addParameterBlock(X.data(), std::make_shared<SE3Param>());
  problem.addParameterBlock(Z.data(), std::make_shared<SE3Param>());

  for (const auto& v : views) {
    const Sophus::SE3d A = v.T_base_gripper.inverse();  // T_gripper_base
    for (std::size_t j = 0; j < v.image_points.size() && j < v.object_points.size(); ++j) {
      problem.addResidualBlock(
          std::make_unique<HandEyeResidual>(&camera, A, v.object_points[j], v.image_points[j]),
          {X.data(), Z.data()});
    }
  }

  const LmSummary s = problem.solveLm(opts);

  HandEyeResult res;
  res.X = SE3Param::load(X.data());
  res.Z = SE3Param::load(Z.data());
  res.summary = s;
  res.information = problem.informationMatrix();
  res.num_residuals = problem.numResiduals();
  return res;
}

}  // namespace calibforge
