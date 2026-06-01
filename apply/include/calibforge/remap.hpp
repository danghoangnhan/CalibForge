#pragma once
//
// CalibForge apply — CPU reference bilinear remap (header-only). Samples the input image
// through a WarpMap (output->source table from warp_map.hpp). This is the parity reference
// the server GPU path (remap_cuda.hpp, CV-CUDA / CUDA kernel) must reproduce.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "calibforge/image.hpp"
#include "calibforge/warp_map.hpp"

namespace calibforge {
namespace apply {

inline Image8 remapBilinear(const Image8& input, const WarpMap& map) {
  Image8 out(map.width, map.height);
  for (int y = 0; y < map.height; ++y) {
    for (int x = 0; x < map.width; ++x) {
      const Vec2 s = map.at(x, y);
      const double v = input.bilinear(s[0], s[1]);  // constant (zero) border
      out.at(x, y) = static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
    }
  }
  return out;
}

}  // namespace apply
}  // namespace calibforge
