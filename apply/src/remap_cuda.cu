//
// CalibForge apply — CUDA bilinear remap (server undistort path, #12).
//
// Compiled ONLY when a CUDA toolkit is found (see the if(CMAKE_CUDA_COMPILER) block in
// CMakeLists.txt), built across the single-source -gencode arch matrix. Reproduces the CPU
// reference remapBilinear() (apply/remap.hpp) to within +/-1 LSB (the map is uploaded as
// float). On a CUDA-less host this translation unit is not built.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "calibforge/image.hpp"
#include "calibforge/warp_map.hpp"
#include "cuda_check.cuh"  // CF_CUDA_CHECK / CF_CUDA_LAST (cuda-samples helper_cuda idiom)

namespace calibforge {
namespace apply {

__global__ void cfRemapKernel(const unsigned char* in, int iw, int ih, const float* mapx,
                              const float* mapy, int ow, int oh, unsigned char* out) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ow || y >= oh) return;
  const float u = mapx[y * ow + x];
  const float v = mapy[y * ow + x];
  float val = 0.0f;  // constant (zero) border, matching the CPU reference
  if (u >= 0.0f && v >= 0.0f && u <= iw - 1.0f && v <= ih - 1.0f) {
    const int x0 = static_cast<int>(floorf(u));
    const int y0 = static_cast<int>(floorf(v));
    const int x1 = min(x0 + 1, iw - 1);
    const int y1 = min(y0 + 1, ih - 1);
    const float fx = u - x0;
    const float fy = v - y0;
    const float a = in[y0 * iw + x0];
    const float b = in[y0 * iw + x1];
    const float c = in[y1 * iw + x0];
    const float d = in[y1 * iw + x1];
    val = (a * (1.0f - fx) + b * fx) * (1.0f - fy) + (c * (1.0f - fx) + d * fx) * fy;
  }
  int iv = static_cast<int>(lroundf(val));
  iv = iv < 0 ? 0 : (iv > 255 ? 255 : iv);
  out[y * ow + x] = static_cast<unsigned char>(iv);
}

Image8 remapBilinearCuda(const Image8& input, const WarpMap& map) {
  const int iw = input.width, ih = input.height, ow = map.width, oh = map.height;
  std::vector<float> mx(static_cast<std::size_t>(ow) * oh);
  std::vector<float> my(static_cast<std::size_t>(ow) * oh);
  for (std::size_t i = 0; i < mx.size(); ++i) {
    mx[i] = static_cast<float>(map.src[i][0]);
    my[i] = static_cast<float>(map.src[i][1]);
  }

  unsigned char *d_in = nullptr, *d_out = nullptr;
  float *d_mx = nullptr, *d_my = nullptr;
  const std::size_t in_bytes = static_cast<std::size_t>(iw) * ih;
  const std::size_t out_bytes = static_cast<std::size_t>(ow) * oh;
  CF_CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
  CF_CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CF_CUDA_CHECK(cudaMalloc(&d_mx, mx.size() * sizeof(float)));
  CF_CUDA_CHECK(cudaMalloc(&d_my, my.size() * sizeof(float)));
  CF_CUDA_CHECK(cudaMemcpy(d_in, input.data.data(), in_bytes, cudaMemcpyHostToDevice));
  CF_CUDA_CHECK(cudaMemcpy(d_mx, mx.data(), mx.size() * sizeof(float), cudaMemcpyHostToDevice));
  CF_CUDA_CHECK(cudaMemcpy(d_my, my.data(), my.size() * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 block(16, 16);
  const dim3 grid((ow + block.x - 1) / block.x, (oh + block.y - 1) / block.y);
  cfRemapKernel<<<grid, block>>>(d_in, iw, ih, d_mx, d_my, ow, oh, d_out);
  CF_CUDA_LAST("cfRemapKernel");                  // getLastCudaError equivalent
  CF_CUDA_CHECK(cudaDeviceSynchronize());

  Image8 out(ow, oh);
  CF_CUDA_CHECK(cudaMemcpy(out.data.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
  CF_CUDA_CHECK(cudaFree(d_in));
  CF_CUDA_CHECK(cudaFree(d_out));
  CF_CUDA_CHECK(cudaFree(d_mx));
  CF_CUDA_CHECK(cudaFree(d_my));
  return out;
}

}  // namespace apply
}  // namespace calibforge
