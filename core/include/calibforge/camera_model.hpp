#pragma once
//
// CalibForge core — the model-agnostic camera contract (INTERFACE STUB).
//
// Every camera model implements this one interface so the solver, the detector,
// and the apply-path are all model-independent. Adding a lens model = writing one
// class, not touching the pipeline.
//
// Borrow map: adopt nvTorchCam's project_to_pixel / pixel_to_ray design
// (Apache-2.0). nvTorchCam ships pinhole, Brown-Conrady, Kannala-Brandt fisheye,
// polynomial fisheye, KITTI-360, cubemap, equirect, orthographic — but NOT
// double-sphere or EUCM, which CalibForge must add (ref: Usenko 3DV 2018,
// basalt-headers). See docs/RESEARCH.md Theme 4.

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace calibforge {

using Vec2 = std::array<double, 2>;  // pixel (u, v)
using Vec3 = std::array<double, 3>;  // ray / 3D point (x, y, z)

// Row-major Jacobian of a projection w.r.t. a parameter group, d(pixel)/d(params).
struct Jacobian {
  std::size_t rows = 2;          // residual dim (u, v)
  std::size_t cols = 0;          // parameter dim
  std::vector<double> data;      // rows*cols, row-major
};

// Abstract differentiable camera model. Implementations live in core/src.
class CameraModel {
 public:
  virtual ~CameraModel() = default;

  // 3D point/ray in camera frame -> pixel.
  virtual Vec2 project(const Vec3& point_cam) const = 0;

  // Pixel -> unit-length observation ray in camera frame. Models without a
  // closed-form inverse (Brown-Conrady, KB fisheye) use a differentiable Newton
  // iteration (implicit-function-theorem differentiation), as in nvTorchCam.
  virtual Vec3 unproject(const Vec2& pixel) const = 0;

  // RULE (docs/RESEARCH.md Theme 4): prefer ANALYTIC Jacobians on hot paths
  // (~30% faster, ~40% less memory than autodiff on GPU); autodiff is for
  // prototyping new models. Implementations override the analytic version.
  virtual Jacobian projectJacobianWrtParams(const Vec3& point_cam) const = 0;

  virtual std::size_t numParams() const = 0;
  virtual std::string name() const = 0;
};

}  // namespace calibforge
