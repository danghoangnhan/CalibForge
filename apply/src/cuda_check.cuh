#pragma once
//
// CalibForge apply — CUDA error-checking helpers in the style of NVIDIA cuda-samples
// `Common/helper_cuda.h` (checkCudaErrors / getLastCudaError). BSD-3-Clause pattern, see
// docs/DEPENDENCIES.md — the idiom is adopted, the header is NOT vendored. Library-adapted:
// throws std::runtime_error on failure instead of cudaDeviceReset()+exit(), so CalibForge
// stays embeddable.

#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace calibforge {
namespace apply {

inline void cfCudaCheck(cudaError_t result, const char* expr, const char* file, int line) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                             " code=" + std::to_string(static_cast<int>(result)) + " \"" +
                             cudaGetErrorString(result) + "\" in: " + expr);
  }
}

inline void cfGetLastCudaError(const char* what, const char* file, int line) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error after ") + what + " at " + file + ":" +
                             std::to_string(line) + ": " + cudaGetErrorString(e));
  }
}

}  // namespace apply
}  // namespace calibforge

// checkCudaErrors / getLastCudaError equivalents (cuda-samples idiom).
#define CF_CUDA_CHECK(val) ::calibforge::apply::cfCudaCheck((val), #val, __FILE__, __LINE__)
#define CF_CUDA_LAST(what) ::calibforge::apply::cfGetLastCudaError((what), __FILE__, __LINE__)
