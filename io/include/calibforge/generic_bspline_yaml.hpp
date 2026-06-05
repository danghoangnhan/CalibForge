#pragma once
//
// CalibForge io — generic per-pixel B-spline camera serialization (header-only, dependency-free).
//
// The parametric YAML (opencv_yaml.hpp) carries a fixed K + a short distortion vector; the
// generic B-spline model instead has a control GRID (nx, ny, margin, image size) plus 3*nx*ny
// control-point ray directions. ROS CameraInfo / OpenCV's K-matrix dialect cannot express that,
// so we emit a sibling document: the same %YAML:1.0 / !!opencv-matrix conventions (so the no-
// OpenCV CI suite still exercises it) with a `camera_model: generic_bspline` tag, the grid
// header keys, and the control points as one 1 x (3*nx*ny) row-major matrix. Round-trips losslessly
// (17 significant digits). This is the io half of wiring the B-spline model into the stack (#25).

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "calibforge/generic_bspline_camera.hpp"

namespace calibforge {
namespace io {

struct GenericBSplineIntrinsics {
  GenericBSplineGrid grid;
  std::vector<double> control_points;  // flat, row-major control[i*ny+j] = (dx,dy,dz); size 3*nx*ny
};

// Build the serializable struct straight from a model (or vice-versa).
inline GenericBSplineIntrinsics toGenericBSplineIntrinsics(const GenericBSplineCamera& cam) {
  return GenericBSplineIntrinsics{cam.grid(), cam.params()};
}
inline GenericBSplineCamera fromGenericBSplineIntrinsics(const GenericBSplineIntrinsics& in) {
  GenericBSplineCamera cam(in.grid);
  cam.setParams(in.control_points);
  return cam;
}

namespace detail {
inline std::string bsfmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.16e", v);  // lossless for double
  return buf;
}
}  // namespace detail

inline std::string toGenericBSplineYaml(const GenericBSplineIntrinsics& in) {
  std::ostringstream os;
  os << "%YAML:1.0\n---\n";
  os << "camera_model: generic_bspline\n";
  os << "image_width: " << in.grid.image_w << "\n";
  os << "image_height: " << in.grid.image_h << "\n";
  os << "grid_nx: " << in.grid.nx << "\n";
  os << "grid_ny: " << in.grid.ny << "\n";
  os << "grid_margin: " << detail::bsfmt(in.grid.margin) << "\n";
  const int n = static_cast<int>(in.control_points.size());
  os << "control_points: !!opencv-matrix\n   rows: 1\n   cols: " << n
     << "\n   dt: d\n   data: [ ";
  for (int i = 0; i < n; ++i) {
    os << detail::bsfmt(in.control_points[static_cast<std::size_t>(i)]);
    if (i + 1 < n) os << ", ";
  }
  os << " ]\n";
  return os.str();
}

// Tolerant line-oriented parser (same dialect as parseOpenCvYaml): scalar grid keys + a
// multi-line control_points !!opencv-matrix block.
inline GenericBSplineIntrinsics parseGenericBSplineYaml(const std::string& text) {
  GenericBSplineIntrinsics in;
  std::istringstream is(text);
  std::string line;
  bool reading_data = false;

  auto parseNumbers = [](const std::string& s, std::vector<double>& out) {
    std::string tok;
    for (char c : s) {
      if (c == ',' || c == '[' || c == ']' || std::isspace(static_cast<unsigned char>(c))) {
        if (!tok.empty()) { out.push_back(std::strtod(tok.c_str(), nullptr)); tok.clear(); }
      } else {
        tok += c;
      }
    }
    if (!tok.empty()) out.push_back(std::strtod(tok.c_str(), nullptr));
  };

  while (std::getline(is, line)) {
    if (reading_data) {
      const bool end = line.find(']') != std::string::npos;
      parseNumbers(line, in.control_points);
      if (end) reading_data = false;
      continue;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = line.substr(0, colon);
    key.erase(0, key.find_first_not_of(" \t"));
    const std::string rest = line.substr(colon + 1);

    if (key == "control_points") {  // matrix block follows (rows/cols/dt/data lines)
      continue;
    }
    if (key == "data") {
      in.control_points.clear();
      parseNumbers(rest, in.control_points);
      if (rest.find(']') == std::string::npos) reading_data = true;
      continue;
    }
    if (key == "image_width") in.grid.image_w = std::atoi(rest.c_str());
    else if (key == "image_height") in.grid.image_h = std::atoi(rest.c_str());
    else if (key == "grid_nx") in.grid.nx = std::atoi(rest.c_str());
    else if (key == "grid_ny") in.grid.ny = std::atoi(rest.c_str());
    else if (key == "grid_margin") in.grid.margin = std::strtod(rest.c_str(), nullptr);
  }
  return in;
}

inline void writeGenericBSplineYaml(const std::string& path, const GenericBSplineIntrinsics& in) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot open for write: " + path);
  f << toGenericBSplineYaml(in);
}

inline GenericBSplineIntrinsics readGenericBSplineYaml(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open for read: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parseGenericBSplineYaml(ss.str());
}

}  // namespace io
}  // namespace calibforge
