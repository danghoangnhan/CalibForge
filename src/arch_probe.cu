// Trivial CUDA translation unit.
//
// Its only job at the scaffold stage is to prove that the multi-arch -gencode
// matrix in the top-level CMakeLists actually compiles end-to-end on a CUDA host
// (one source -> server + Jetson). It is built only when a CUDA toolkit is found.

#include <cstdio>

__global__ void calibforge_probe_kernel(int* out) {
  *out = blockIdx.x + threadIdx.x;  // nothing meaningful — just exercises codegen
}

int main() {
  std::printf("CalibForge arch probe: CUDA translation unit compiled and linked.\n");
  // Note: we do NOT launch the kernel here — this probe must succeed at build time
  // even on machines with no physical GPU (CI). Reference the symbol so it is emitted.
  (void)&calibforge_probe_kernel;
  return 0;
}
