#pragma once
//
// CalibForge io — OpenCV YAML (cv::FileStorage) read/write for intrinsics + distortion
// (header-only, dependency-free). The OpenCV calibration dialect is a tiny fixed subset
// (%YAML:1.0, !!opencv-matrix, rows/cols/dt/data), so we hand-roll it rather than pull a
// compiled YAML dependency — the no-OpenCV CI suite still exercises the format. An optional
// cross-check against real cv::FileStorage lives in the OpenCV-gated tests.
//
// to/parse work on strings (testable with no file I/O); write/read wrap them with a file.

#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace calibforge {
namespace io {

struct Intrinsics {
  std::array<double, 9> camera_matrix = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // K, row-major 3x3
  std::vector<double> distortion_coeffs;                               // k1,k2,p1,p2[,k3...]
  int image_width = 0;
  int image_height = 0;
  std::string distortion_model = "plumb_bob";
};

// Convenience: assemble K from focal/principal point.
inline Intrinsics makeIntrinsics(double fx, double fy, double cx, double cy,
                                 std::vector<double> dist, int w, int h,
                                 std::string model = "plumb_bob") {
  Intrinsics in;
  in.camera_matrix = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  in.distortion_coeffs = std::move(dist);
  in.image_width = w;
  in.image_height = h;
  in.distortion_model = std::move(model);
  return in;
}

namespace detail {
inline std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.16e", v);  // 17 significant digits: lossless for double
  return buf;
}
inline void emitMatrix(std::ostringstream& os, const std::string& key, int rows, int cols,
                       const double* data) {
  os << key << ": !!opencv-matrix\n   rows: " << rows << "\n   cols: " << cols
     << "\n   dt: d\n   data: [ ";
  for (int i = 0; i < rows * cols; ++i) {
    os << fmt(data[i]);
    if (i + 1 < rows * cols) os << ", ";
  }
  os << " ]\n";
}
}  // namespace detail

inline std::string toOpenCvYaml(const Intrinsics& in) {
  std::ostringstream os;
  os << "%YAML:1.0\n---\n";
  os << "image_width: " << in.image_width << "\n";
  os << "image_height: " << in.image_height << "\n";
  detail::emitMatrix(os, "camera_matrix", 3, 3, in.camera_matrix.data());
  os << "distortion_model: " << in.distortion_model << "\n";
  detail::emitMatrix(os, "distortion_coefficients", 1,
                     static_cast<int>(in.distortion_coeffs.size()),
                     in.distortion_coeffs.data());
  return os.str();
}

// Tolerant line-oriented parser for the OpenCV calibration dialect: handles OpenCV's
// collapsed integer tokens ("0.", "1."), scientific notation, and multi-line data arrays.
inline Intrinsics parseOpenCvYaml(const std::string& text) {
  Intrinsics in;
  std::istringstream is(text);
  std::string line;
  std::string current_key;       // matrix currently being read
  std::vector<double> acc;       // accumulated data values for the current matrix
  bool reading_data = false;

  auto flush = [&](const std::string& key, std::vector<double>& vals) {
    if (key == "camera_matrix") {
      if (vals.size() != 9) throw std::runtime_error("camera_matrix must have 9 entries");
      for (int i = 0; i < 9; ++i) in.camera_matrix[i] = vals[i];
    } else if (key == "distortion_coefficients") {
      in.distortion_coeffs = vals;
    }
    vals.clear();
  };

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
      parseNumbers(line, acc);
      if (end) { flush(current_key, acc); reading_data = false; current_key.clear(); }
      continue;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = line.substr(0, colon);
    // trim leading whitespace from key
    key.erase(0, key.find_first_not_of(" \t"));
    const std::string rest = line.substr(colon + 1);

    if (rest.find("!!opencv-matrix") != std::string::npos) {
      current_key = key;
      continue;
    }
    if (key == "data") {
      acc.clear();
      parseNumbers(rest, acc);
      if (rest.find(']') != std::string::npos) {
        flush(current_key, acc);
        current_key.clear();
      } else {
        reading_data = true;
      }
      continue;
    }
    if (key == "image_width") in.image_width = std::atoi(rest.c_str());
    else if (key == "image_height") in.image_height = std::atoi(rest.c_str());
    else if (key == "distortion_model") {
      std::string m = rest;
      m.erase(0, m.find_first_not_of(" \t"));
      m.erase(m.find_last_not_of(" \t\r\n") + 1);
      if (!m.empty()) in.distortion_model = m;
    }
  }
  return in;
}

inline void writeOpenCvYaml(const std::string& path, const Intrinsics& in) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot open for write: " + path);
  f << toOpenCvYaml(in);
}

inline Intrinsics readOpenCvYaml(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open for read: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return parseOpenCvYaml(ss.str());
}

}  // namespace io
}  // namespace calibforge
