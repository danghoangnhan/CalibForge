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
#include "calibforge/calibrate_generic_bspline.hpp"
#include "calibforge/calibrate_single.hpp"
#include "calibforge/double_sphere_camera.hpp"
#include "calibforge/eucm_camera.hpp"
#include "calibforge/generic_bspline_camera.hpp"
#include "calibforge/generic_bspline_yaml.hpp"
#include "calibforge/kannala_brandt_camera.hpp"
#include "calibforge/observability.hpp"
#include "calibforge/pinhole_camera.hpp"
#include "calibforge/rectify_stereo.hpp"
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

  // Generic per-pixel B-spline model (Schöps CVPR 2020): the control GRID + the dense ray field.
  py::class_<GenericBSplineGrid>(m, "GenericBSplineGrid")
      .def(py::init<>())
      .def_readwrite("nx", &GenericBSplineGrid::nx)
      .def_readwrite("ny", &GenericBSplineGrid::ny)
      .def_readwrite("image_w", &GenericBSplineGrid::image_w)
      .def_readwrite("image_h", &GenericBSplineGrid::image_h)
      .def_readwrite("margin", &GenericBSplineGrid::margin);

  auto gb = py::class_<GenericBSplineCamera>(m, "GenericBSplineCamera")
                .def(py::init<const GenericBSplineGrid&>())
                .def("grid", &GenericBSplineCamera::grid)
                .def("params", &GenericBSplineCamera::params)
                .def("set_params", &GenericBSplineCamera::setParams)
                .def("fit_from_parametric",
                     [](GenericBSplineCamera& cam, const std::string& model,
                        const std::vector<double>& q) {
                       const std::unique_ptr<CameraModel> src = factoryFor(model)(
                           Eigen::Map<const Eigen::VectorXd>(
                               q.data(), static_cast<Eigen::Index>(q.size())));
                       cam.fitFromParametricCamera(*src);
                     },
                     py::arg("model"), py::arg("intrinsics"));
  addCameraMethods<GenericBSplineCamera>(gb);

  // B-spline YAML serialization (round-trips the grid + control points).
  m.def("generic_bspline_to_yaml", [](const GenericBSplineCamera& cam) {
    return io::toGenericBSplineYaml(io::toGenericBSplineIntrinsics(cam));
  });
  m.def("generic_bspline_from_yaml", [](const std::string& text) {
    return io::fromGenericBSplineIntrinsics(io::parseGenericBSplineYaml(text));
  });

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

  // Generic B-spline calibration: fit the control grid to a parametric source, then refine
  // against observations. `optimize_poses=false` does the functional fit (poses known).
  m.def(
      "calibrate_generic_bspline",
      [](const GenericBSplineGrid& grid, const std::string& init_model,
         const std::vector<double>& init_intrinsics,
         const std::vector<std::vector<std::array<double, 3>>>& object_points,
         const std::vector<std::vector<std::array<double, 2>>>& image_points,
         const std::vector<Eigen::Matrix4d>& poses_init, int max_iter, bool optimize_poses) {
        std::vector<View> views;
        for (std::size_t i = 0; i < object_points.size(); ++i) {
          View v;
          v.object_points = object_points[i];
          v.image_points = image_points[i];
          views.push_back(v);
        }
        const std::unique_ptr<CameraModel> src = factoryFor(init_model)(
            Eigen::Map<const Eigen::VectorXd>(init_intrinsics.data(),
                                              static_cast<Eigen::Index>(init_intrinsics.size())));
        const std::vector<double> control_init = makeGenericBSplineInit(grid, *src);
        std::vector<Sophus::SE3d> poses;
        for (const auto& P : poses_init) poses.push_back(Sophus::SE3d(P));
        GenericBSplineCalibrateOptions opts;
        opts.lm.max_iterations = max_iter;
        opts.optimize_poses = optimize_poses;
        GenericBSplineResult res = calibrateGenericBSpline(views, grid, control_init, poses, opts);

        py::dict d;
        d["control_points"] = res.control_points;
        d["final_cost"] = res.summary.final_cost;
        d["converged"] = res.summary.converged;
        d["num_residuals"] = res.num_residuals;
        d["rms_reprojection_px"] = res.rms_reprojection_px;
        return d;
      },
      py::arg("grid"), py::arg("init_model"), py::arg("init_intrinsics"),
      py::arg("object_points"), py::arg("image_points"), py::arg("poses_init"),
      py::arg("max_iter") = 100, py::arg("optimize_poses") = true);

  // Stereo rectification: from a calibrated pair (two intrinsics + extrinsic T_cam1_cam0) produce
  // the rectifying rotations R0/R1, shared rectified K, the ROS projection matrices P0/P1, and the
  // disparity->depth Q. Returns a dict (Eigen marshals to numpy via pybind11/eigen.h).
  m.def(
      "compute_stereo_rectification",
      [](const std::string& model0, const std::vector<double>& intrinsics0,
         const std::string& model1, const std::vector<double>& intrinsics1,
         const Eigen::Matrix4d& T_cam1_cam0, int width, int height) {
        const Eigen::VectorXd i0 = Eigen::Map<const Eigen::VectorXd>(
            intrinsics0.data(), static_cast<Eigen::Index>(intrinsics0.size()));
        const Eigen::VectorXd i1 = Eigen::Map<const Eigen::VectorXd>(
            intrinsics1.data(), static_cast<Eigen::Index>(intrinsics1.size()));
        const StereoRectification rect = computeStereoRectification(
            i0, i1, Sophus::SE3d(T_cam1_cam0), width, height, factoryFor(model0),
            factoryFor(model1));
        py::dict d;
        d["R0"] = rect.R0;
        d["R1"] = rect.R1;
        d["K_rect"] = std::vector<double>(rect.K_rect.begin(), rect.K_rect.end());
        d["P0"] = std::vector<double>(rect.P0.begin(), rect.P0.end());
        d["P1"] = std::vector<double>(rect.P1.begin(), rect.P1.end());
        d["Q"] = std::vector<double>(rect.Q.begin(), rect.Q.end());
        d["baseline"] = rect.baseline;
        return d;
      },
      py::arg("model0"), py::arg("intrinsics0"), py::arg("model1"), py::arg("intrinsics1"),
      py::arg("T_cam1_cam0"), py::arg("width"), py::arg("height"));
}
