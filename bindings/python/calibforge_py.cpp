//
// CalibForge — Python bindings (pybind11). Exposes the camera models, the observability /
// uncertainty gate, and single-camera calibration. Built only when CALIBFORGE_PYTHON=ON
// (pybind11 is BSD-3, fetched via CMake); the core C++ suite is unaffected.

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "calibforge/brown_conrady_camera.hpp"
#include "calibforge/calibrate_single.hpp"
#include "calibforge/double_sphere_camera.hpp"
#include "calibforge/eucm_camera.hpp"
#include "calibforge/kannala_brandt_camera.hpp"
#include "calibforge/observability.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "sophus/se3.hpp"

namespace py = pybind11;
using namespace calibforge;

// Build a CameraFactory from a model name (intrinsic length must match).
static CameraFactory factoryFor(const std::string& model) {
  if (model == "pinhole")
    return [](const Eigen::VectorXd& q) {
      return std::unique_ptr<CameraModel>(new PinholeCamera(q[0], q[1], q[2], q[3]));
    };
  if (model == "brown_conrady")
    return [](const Eigen::VectorXd& q) {
      return std::unique_ptr<CameraModel>(
          new BrownConradyCamera(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]));
    };
  if (model == "kannala_brandt")
    return [](const Eigen::VectorXd& q) {
      return std::unique_ptr<CameraModel>(
          new KannalaBrandtCamera(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]));
    };
  if (model == "double_sphere")
    return [](const Eigen::VectorXd& q) {
      return std::unique_ptr<CameraModel>(
          new DoubleSphereCamera(q[0], q[1], q[2], q[3], q[4], q[5]));
    };
  if (model == "eucm")
    return [](const Eigen::VectorXd& q) {
      return std::unique_ptr<CameraModel>(new EUCMCamera(q[0], q[1], q[2], q[3], q[4], q[5]));
    };
  throw std::runtime_error("unknown camera model: " + model);
}

template <typename Cam, typename PyCls>
static void addCameraMethods(PyCls& c) {
  c.def("project", [](const Cam& cam, std::array<double, 3> p) { return cam.project(p); })
      .def("unproject", [](const Cam& cam, std::array<double, 2> px) { return cam.unproject(px); })
      .def("name", &Cam::name)
      .def("num_params", &Cam::numParams);
}

PYBIND11_MODULE(calibforge, m) {
  m.doc() = "CalibForge — NVIDIA-accelerated geometric camera calibration (Python bindings)";

  auto pin = py::class_<PinholeCamera>(m, "PinholeCamera")
                 .def(py::init<double, double, double, double>())
                 .def("params", &PinholeCamera::params);
  addCameraMethods<PinholeCamera>(pin);

  auto bc = py::class_<BrownConradyCamera>(m, "BrownConradyCamera")
                .def(py::init<double, double, double, double, double, double, double, double, double>())
                .def("params", &BrownConradyCamera::params);
  addCameraMethods<BrownConradyCamera>(bc);

  auto kb = py::class_<KannalaBrandtCamera>(m, "KannalaBrandtCamera")
                .def(py::init<double, double, double, double, double, double, double, double>());
  addCameraMethods<KannalaBrandtCamera>(kb);

  auto ds = py::class_<DoubleSphereCamera>(m, "DoubleSphereCamera")
                .def(py::init<double, double, double, double, double, double>())
                .def("project_valid", &DoubleSphereCamera::projectValid);
  addCameraMethods<DoubleSphereCamera>(ds);

  auto eu = py::class_<EUCMCamera>(m, "EUCMCamera")
                .def(py::init<double, double, double, double, double, double>())
                .def("project_valid", &EUCMCamera::projectValid);
  addCameraMethods<EUCMCamera>(eu);

  py::class_<ObservabilityReport>(m, "ObservabilityReport")
      .def_readonly("observable", &ObservabilityReport::observable)
      .def_readonly("confidence", &ObservabilityReport::confidence)
      .def_readonly("min_eigenvalue", &ObservabilityReport::min_eigenvalue);
  m.def("assess_observability",
        [](const Eigen::MatrixXd& H) { return assessObservability(H); }, py::arg("information"));

  py::class_<ParameterUncertainty>(m, "ParameterUncertainty")
      .def_readonly("sigma", &ParameterUncertainty::sigma)
      .def_readonly("weak_parameters", &ParameterUncertainty::weak_parameters)
      .def_readonly("sigma0_sq", &ParameterUncertainty::sigma0_sq);
  m.def("parameter_uncertainty",
        [](const Eigen::MatrixXd& H, double final_cost, int m_resid,
           const std::vector<std::string>& names) {
          return parameterUncertainty(H, final_cost, m_resid, names);
        },
        py::arg("information"), py::arg("final_cost"), py::arg("num_residuals"),
        py::arg("names") = std::vector<std::string>{});

  m.def(
      "calibrate_single",
      [](const std::string& model,
         const std::vector<std::vector<std::array<double, 3>>>& object_points,
         const std::vector<std::vector<std::array<double, 2>>>& image_points,
         const std::vector<double>& intrinsics_init,
         const std::vector<Eigen::Matrix4d>& poses_init, int max_iter) {
        std::vector<View> views;
        for (std::size_t i = 0; i < object_points.size(); ++i) {
          View v;
          v.object_points = object_points[i];
          v.image_points = image_points[i];
          views.push_back(v);
        }
        Eigen::VectorXd intr =
            Eigen::Map<const Eigen::VectorXd>(intrinsics_init.data(),
                                              static_cast<Eigen::Index>(intrinsics_init.size()));
        std::vector<Sophus::SE3d> poses;
        for (const auto& P : poses_init) poses.push_back(Sophus::SE3d(P));
        LmOptions opts;
        opts.max_iterations = max_iter;
        SingleCameraResult res = calibrateSingleCamera(views, intr, poses, factoryFor(model), opts);

        py::dict d;
        d["intrinsics"] = std::vector<double>(res.intrinsics.data(),
                                              res.intrinsics.data() + res.intrinsics.size());
        d["final_cost"] = res.summary.final_cost;
        d["converged"] = res.summary.converged;
        d["num_residuals"] = res.num_residuals;
        d["information"] = res.information;
        return d;
      },
      py::arg("model"), py::arg("object_points"), py::arg("image_points"),
      py::arg("intrinsics_init"), py::arg("poses_init"), py::arg("max_iter") = 100);
}
