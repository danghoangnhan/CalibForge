//
// CalibForge solve — native CUDA dense LM linear-solver back-end (#25 GPU solver back-end).
//
// Implements cudaSolveLmStep(): one damped-normal-equations Levenberg-Marquardt step on the GPU
//     dx = -(J^T J + lambda * diag(J^T J))^-1 (J^T r)
// using cuBLAS (SYRK for J^T J, GEMV for J^T r) + cuSOLVER (dense Cholesky potrf/potrs). Built
// only with a CUDA toolkit (CALIBFORGE_HAS_CUDA), across the single-source -gencode arch matrix,
// exactly like apply/src/remap_cuda.cu. Returns false on a non-SPD damped matrix or any device
// error so the host LM loop can damp harder and retry (never silently emits a bad step).
//
// All matrices are COLUMN-MAJOR (Eigen default == cuBLAS native), so J (m x n) has leading
// dimension m and no host-side transpose is needed.

#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "calibforge/cuda_linear_solver.hpp"  // declaration cross-check (CALIBFORGE_HAS_CUDA on)

namespace calibforge {
namespace {

// Multiply the diagonal of a column-major n x n matrix A in place by `factor`
// (Marquardt scale-invariant damping: A_ii = J^TJ_ii * (1 + lambda)). Element (i,i) of a
// column-major n x n matrix is at A[i + i*n].
__global__ void scaleDiagonalKernel(double* A, int n, double factor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[static_cast<long long>(i) * n + i] *= factor;
}

// Lazily-created, reused cuBLAS + cuSOLVER handles (handle creation is ~ms; recreating per LM
// step would dominate small-problem timing and unfairly penalize the GPU in the benchmark).
// Intentionally leaked at process exit — destroying them during static teardown races the CUDA
// context teardown. Single-threaded use (the LM loop is sequential).
struct Handles {
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  bool ok = false;
};
Handles& handles() {
  static Handles h = [] {
    Handles hh;
    hh.ok = (cublasCreate(&hh.blas) == CUBLAS_STATUS_SUCCESS) &&
            (cusolverDnCreate(&hh.solver) == CUSOLVER_STATUS_SUCCESS);
    return hh;
  }();
  return h;
}

}  // namespace

bool cudaSolveLmStep(const double* J_colmajor, int m, int n, const double* r, double lambda,
                     double* delta_out) {
  if (m <= 0 || n <= 0 || J_colmajor == nullptr || r == nullptr || delta_out == nullptr)
    return false;
  Handles& h = handles();
  if (!h.ok) return false;

  double *dJ = nullptr, *dr = nullptr, *dA = nullptr, *dg = nullptr, *dwork = nullptr;
  int* dinfo = nullptr;
  bool ok = true;
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  auto check = [&](bool cond) { if (!cond) ok = false; return ok; };

  // Device buffers.
  check(cudaMalloc(&dJ, sizeof(double) * static_cast<size_t>(m) * n) == cudaSuccess);
  check(ok && cudaMalloc(&dr, sizeof(double) * m) == cudaSuccess);
  check(ok && cudaMalloc(&dA, sizeof(double) * static_cast<size_t>(n) * n) == cudaSuccess);
  check(ok && cudaMalloc(&dg, sizeof(double) * n) == cudaSuccess);
  check(ok && cudaMalloc(&dinfo, sizeof(int)) == cudaSuccess);

  if (ok) {
    check(cudaMemcpy(dJ, J_colmajor, sizeof(double) * static_cast<size_t>(m) * n,
                     cudaMemcpyHostToDevice) == cudaSuccess);
    check(ok && cudaMemcpy(dr, r, sizeof(double) * m, cudaMemcpyHostToDevice) == cudaSuccess);
  }

  const double one = 1.0, zero = 0.0, neg_one = -1.0;
  // A = J^T J (lower triangle, n x n). With trans=CUBLAS_OP_T, DSYRK computes C = A^T A for the
  // m x n input A=J (lda=m) — i.e. J^T J; trans=CUBLAS_OP_N would instead give J J^T (m x m, wrong
  // shape). potrf/potrs below read only the LOWER triangle, so the upper triangle is left unset.
  if (ok)
    check(cublasDsyrk(h.blas, uplo, CUBLAS_OP_T, n, m, &one, dJ, m, &zero, dA, n) ==
          CUBLAS_STATUS_SUCCESS);
  // g = J^T r   (n-vector).
  if (ok)
    check(cublasDgemv(h.blas, CUBLAS_OP_T, m, n, &one, dJ, m, dr, 1, &zero, dg, 1) ==
          CUBLAS_STATUS_SUCCESS);
  // Marquardt damping: scale the diagonal by (1 + lambda).
  if (ok) {
    const int threads = 128;
    scaleDiagonalKernel<<<(n + threads - 1) / threads, threads>>>(dA, n, 1.0 + lambda);
    check(cudaGetLastError() == cudaSuccess);
  }
  // RHS = -g  (so potrs yields dx = A^{-1} (-g)).
  if (ok) check(cublasDscal(h.blas, n, &neg_one, dg, 1) == CUBLAS_STATUS_SUCCESS);

  // Cholesky factor + solve A dx = -g (in place on dg).
  int lwork = 0;
  if (ok)
    check(cusolverDnDpotrf_bufferSize(h.solver, uplo, n, dA, n, &lwork) ==
          CUSOLVER_STATUS_SUCCESS);
  if (ok && lwork > 0)
    check(cudaMalloc(&dwork, sizeof(double) * lwork) == cudaSuccess);
  if (ok)
    check(cusolverDnDpotrf(h.solver, uplo, n, dA, n, dwork, lwork, dinfo) ==
          CUSOLVER_STATUS_SUCCESS);
  // info != 0 => leading minor not positive-definite (rank-deficient / too little damping).
  int info = -1;
  if (ok) check(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  if (ok && info != 0) ok = false;  // not SPD: caller damps harder and retries
  if (ok)
    check(cusolverDnDpotrs(h.solver, uplo, n, 1, dA, n, dg, n, dinfo) == CUSOLVER_STATUS_SUCCESS);
  // potrs writes its own info (illegal-argument => <0); check it too, mirroring potrf, so a
  // device-side failure can never slip a garbage step past the host loop.
  int solve_info = -1;
  if (ok) check(cudaMemcpy(&solve_info, dinfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  if (ok && solve_info != 0) ok = false;
  if (ok) check(cudaMemcpy(delta_out, dg, sizeof(double) * n, cudaMemcpyDeviceToHost) ==
                cudaSuccess);

  cudaFree(dJ);
  cudaFree(dr);
  cudaFree(dA);
  cudaFree(dg);
  cudaFree(dwork);
  cudaFree(dinfo);
  return ok;
}

}  // namespace calibforge
