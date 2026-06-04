#pragma once
//
// CalibForge solve — native CUDA dense linear-solver back-end for the Levenberg-Marquardt
// inner step (#25, the GPU solver back-end). One LM trial solves the damped normal equations
//
//     (J^T J + lambda * diag(J^T J)) dx = - J^T r
//
// on the GPU: J^T J via cuBLAS SYRK, J^T r via cuBLAS GEMV, and the SPD factor+solve via
// cuSOLVER dense Cholesky (potrf/potrs). The analytic Jacobians are still assembled on the
// host (RULE #4) — only the dense linear algebra moves to the device.
//
// ⚠️ RULE #1 (docs/RESEARCH.md Theme 1): the GPU is NOT automatically faster. For a single
// small calibration the host Eigen path wins (host<->device transfer + kernel-launch overhead
// dwarf an O(n^3) solve for n in the tens). This back-end earns its keep on large rigs /
// batched / generic 10^4-param problems. The CPU path stays the DEFAULT; this back-end is
// selected explicitly (SolverBackend::GpuCuda) and its crossover is MEASURED, never assumed
// (tools/benchmark/calibforge_bench.cpp GPU rows).
//
// cudaSolverAvailable() mirrors apply/cudaRemapAvailable(): true exactly when this build linked
// the CUDA back-end (CALIBFORGE_HAS_CUDA), so callers (DenseProblem) degrade cleanly to the CPU
// path on a CUDA-less host. The implementation lives in solve/src/cuda_linear_solver.cu,
// compiled into calibforge_cuda only when a CUDA toolkit is present.

namespace calibforge {

inline bool cudaSolverAvailable() {
#ifdef CALIBFORGE_HAS_CUDA
  return true;
#else
  return false;
#endif
}

#ifdef CALIBFORGE_HAS_CUDA
// Solve one damped-normal-equations LM step on the GPU and write the n-vector
//     dx = -(J^T J + lambda * diag(J^T J))^-1 (J^T r)
// to delta_out (must hold n doubles). J_colmajor is the m x n Jacobian in COLUMN-MAJOR order
// (Eigen's default storage — pass Eigen::MatrixXd::data() directly); r is the m-residual.
// Returns false if the damped normal matrix is not symmetric-positive-definite (cuSOLVER potrf
// reports a non-zero leading minor) or any CUDA/cuBLAS/cuSOLVER call fails — the caller should
// then increase lambda and retry, exactly like the host reject path. On a false return delta_out
// is left UNSPECIFIED (it is written only on success) — the caller must discard it. Numerically
// matches the host Eigen LDLT solve to FP64 round-off (cross-implementation, not bit-exact).
bool cudaSolveLmStep(const double* J_colmajor, int m, int n, const double* r, double lambda,
                     double* delta_out);
#endif

}  // namespace calibforge
