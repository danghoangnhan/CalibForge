#pragma once
//
// CalibForge io — Kalibr camchain.yaml read/write for multi-camera rigs (header-only,
// dependency-free). Kalibr stores each camera as camN with intrinsics + distortion + the
// CONSECUTIVE extrinsic T_cn_cnm1 (maps cam(n-1) coords -> cam(n) coords); cam0 has none.
// Our rig stores T_ck_c0 (cam0 -> camk); consecutiveFromReference() converts between them.

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "sophus/se3.hpp"

namespace calibforge {
namespace io {

struct KalibrCamera {
  std::string camera_model = "pinhole";       // pinhole
  std::array<double, 4> intrinsics = {0, 0, 0, 0};  // fx, fy, cx, cy
  std::string distortion_model = "none";      // radtan / equidistant / none
  std::vector<double> distortion_coeffs;      // radtan: k1,k2,p1,p2 ; equidistant: k1..k4
  int width = 0, height = 0;
  bool has_extrinsic = false;                 // false for cam0 (the reference)
  Eigen::Matrix4d T_cn_cnm1 = Eigen::Matrix4d::Identity();  // cam(n-1) -> cam(n)
};

struct KalibrCamchain {
  std::vector<KalibrCamera> cameras;
};

// Convert reference extrinsics T_ck_c0 (k=1..N-1) to Kalibr's consecutive T_cn_cnm1.
//   T_cn_cnm1 = T_cn_c0 * T_c(n-1)_c0^{-1}   (with T_c0_c0 = identity)
inline std::vector<Eigen::Matrix4d> consecutiveFromReference(
    const std::vector<Sophus::SE3d>& T_ck_c0) {
  std::vector<Eigen::Matrix4d> out;
  Sophus::SE3d prev;  // identity = cam0
  for (const auto& T_ck : T_ck_c0) {
    out.push_back((T_ck * prev.inverse()).matrix());
    prev = T_ck;
  }
  return out;
}

namespace detail {
inline std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.12g", v);
  return b;
}
inline void parseNums(const std::string& s, std::vector<double>& out) {
  std::string tok;  // delimiters are , [ ] and whitespace; -, +, ., e/E accumulate into a number
  for (char c : s) {
    if (c == ',' || c == '[' || c == ']' || std::isspace(static_cast<unsigned char>(c))) {
      if (!tok.empty()) { out.push_back(std::strtod(tok.c_str(), nullptr)); tok.clear(); }
    } else {
      tok += c;
    }
  }
  if (!tok.empty()) out.push_back(std::strtod(tok.c_str(), nullptr));
}
}  // namespace detail

inline std::string toKalibrYaml(const KalibrCamchain& chain) {
  std::ostringstream os;
  for (std::size_t i = 0; i < chain.cameras.size(); ++i) {
    const KalibrCamera& c = chain.cameras[i];
    os << "cam" << i << ":\n";
    os << "  camera_model: " << c.camera_model << "\n";
    os << "  intrinsics: [" << detail::num(c.intrinsics[0]) << ", " << detail::num(c.intrinsics[1])
       << ", " << detail::num(c.intrinsics[2]) << ", " << detail::num(c.intrinsics[3]) << "]\n";
    os << "  distortion_model: " << c.distortion_model << "\n";
    os << "  distortion_coeffs: [";
    for (std::size_t k = 0; k < c.distortion_coeffs.size(); ++k) {
      if (k) os << ", ";
      os << detail::num(c.distortion_coeffs[k]);
    }
    os << "]\n";
    os << "  resolution: [" << c.width << ", " << c.height << "]\n";
    if (c.has_extrinsic) {
      os << "  T_cn_cnm1:\n";
      for (int r = 0; r < 4; ++r) {
        os << "  - [" << detail::num(c.T_cn_cnm1(r, 0)) << ", " << detail::num(c.T_cn_cnm1(r, 1))
           << ", " << detail::num(c.T_cn_cnm1(r, 2)) << ", " << detail::num(c.T_cn_cnm1(r, 3))
           << "]\n";
      }
    }
  }
  return os.str();
}

inline KalibrCamchain parseKalibrYaml(const std::string& text) {
  KalibrCamchain chain;
  std::istringstream is(text);
  std::string line;
  KalibrCamera* cur = nullptr;
  int trow = -1;  // current T_cn_cnm1 row index, -1 = not reading the matrix
  while (std::getline(is, line)) {
    if (!line.empty() && line[0] == 'c' && line.rfind("cam", 0) == 0 &&
        line.find(':') != std::string::npos && line.find_first_not_of(" \t") == 0) {
      chain.cameras.emplace_back();
      cur = &chain.cameras.back();
      trow = -1;
      continue;
    }
    if (!cur) continue;
    std::string key = line;
    key.erase(0, key.find_first_not_of(" \t"));
    if (trow >= 0 && trow < 4 && key.rfind("- [", 0) == 0) {
      std::vector<double> row;
      detail::parseNums(key.substr(key.find('[')), row);  // skip the "- " list marker
      for (int j = 0; j < 4 && j < static_cast<int>(row.size()); ++j) cur->T_cn_cnm1(trow, j) = row[j];
      ++trow;
      continue;
    }
    const std::size_t colon = key.find(':');
    if (colon == std::string::npos) continue;
    const std::string name = key.substr(0, colon);
    const std::string rest = key.substr(colon + 1);
    if (name == "camera_model") {
      std::string v = rest; v.erase(0, v.find_first_not_of(" \t")); v.erase(v.find_last_not_of(" \t\r\n") + 1);
      cur->camera_model = v;
    } else if (name == "distortion_model") {
      std::string v = rest; v.erase(0, v.find_first_not_of(" \t")); v.erase(v.find_last_not_of(" \t\r\n") + 1);
      cur->distortion_model = v;
    } else if (name == "intrinsics") {
      std::vector<double> v; detail::parseNums(rest, v);
      for (int j = 0; j < 4 && j < static_cast<int>(v.size()); ++j) cur->intrinsics[j] = v[j];
    } else if (name == "distortion_coeffs") {
      cur->distortion_coeffs.clear(); detail::parseNums(rest, cur->distortion_coeffs);
    } else if (name == "resolution") {
      std::vector<double> v; detail::parseNums(rest, v);
      if (v.size() >= 2) { cur->width = static_cast<int>(v[0]); cur->height = static_cast<int>(v[1]); }
    } else if (name == "T_cn_cnm1") {
      cur->has_extrinsic = true; trow = 0;
    }
  }
  return chain;
}

}  // namespace io
}  // namespace calibforge
