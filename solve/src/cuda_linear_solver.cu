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
// The damped solve is TEMPLATED over the device scalar type so the SAME code path serves the FP64
// default (the accuracy oracle, matches host Eigen LDLT to round-off) AND an FP32 path used to
// MEASURE edge<->server precision parity (FP32/bf16 vs FP64 is called out as unproven in
// docs/SPIKES.md §D.3 / CLAUDE.md rule 6). cudaSolveLmStepF32() casts the host FP64 J/r down to
// float, solves entirely in single precision, and casts dx back up — so the parity test compares
// a genuine single-precision GPU solve against the FP64 GPU + host solves on real silicon.
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
template <typename T>
__global__ void scaleDiagonalKernel(T* A, int n, T factor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) A[static_cast<long long>(i) * n + i] *= factor;
}

// --- Precision-overloaded thin wrappers over the cuBLAS/cuSOLVER S* / D* variants ----------
// (cuBLAS/cuSOLVER expose separate functions per scalar type, not templates; these overloads let
// the solve body below be written once and instantiated for float and double.)
inline cublasStatus_t cfSyrk(cublasHandle_t h, cublasFillMode_t uplo, int n, int k,
                             const double* alpha, const double* A, int lda, const double* beta,
                             double* C, int ldc) {
  return cublasDsyrk(h, uplo, CUBLAS_OP_T, n, k, alpha, A, lda, beta, C, ldc);
}
inline cublasStatus_t cfSyrk(cublasHandle_t h, cublasFillMode_t uplo, int n, int k,
                             const float* alpha, const float* A, int lda, const float* beta,
                             float* C, int ldc) {
  return cublasSsyrk(h, uplo, CUBLAS_OP_T, n, k, alpha, A, lda, beta, C, ldc);
}
inline cublasStatus_t cfGemvT(cublasHandle_t h, int m, int n, const double* alpha,
                              const double* A, int lda, const double* x, const double* beta,
                              double* y) {
  return cublasDgemv(h, CUBLAS_OP_T, m, n, alpha, A, lda, x, 1, beta, y, 1);
}
inline cublasStatus_t cfGemvT(cublasHandle_t h, int m, int n, const float* alpha, const float* A,
                              int lda, const float* x, const float* beta, float* y) {
  return cublasSgemv(h, CUBLAS_OP_T, m, n, alpha, A, lda, x, 1, beta, y, 1);
}
inline cublasStatus_t cfScal(cublasHandle_t h, int n, const double* a, double* x) {
  return cublasDscal(h, n, a, x, 1);
}
inline cublasStatus_t cfScal(cublasHandle_t h, int n, const float* a, float* x) {
  return cublasSscal(h, n, a, x, 1);
}
inline cusolverStatus_t cfPotrfBuffer(cusolverDnHandle_t h, cublasFillMode_t uplo, int n,
                                      double* A, int lda, int* lwork) {
  return cusolverDnDpotrf_bufferSize(h, uplo, n, A, lda, lwork);
}
inline cusolverStatus_t cfPotrfBuffer(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, float* A,
                                      int lda, int* lwork) {
  return cusolverDnSpotrf_bufferSize(h, uplo, n, A, lda, lwork);
}
inline cusolverStatus_t cfPotrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, double* A,
                                int lda, double* work, int lwork, int* info) {
  return cusolverDnDpotrf(h, uplo, n, A, lda, work, lwork, info);
}
inline cusolverStatus_t cfPotrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, float* A,
                                int lda, float* work, int lwork, int* info) {
  return cusolverDnSpotrf(h, uplo, n, A, lda, work, lwork, info);
}
inline cusolverStatus_t cfPotrs(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, int nrhs,
                                const double* A, int lda, double* B, int ldb, int* info) {
  return cusolverDnDpotrs(h, uplo, n, nrhs, A, lda, B, ldb, info);
}
inline cusolverStatus_t cfPotrs(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, int nrhs,
                                const float* A, int lda, float* B, int ldb, int* info) {
  return cusolverDnSpotrs(h, uplo, n, nrhs, A, lda, B, ldb, info);
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

// Templated damped-normal-equations solve over device scalar T (float or double). `dJ`/`dr` are
// already device-resident; writes the n-vector dx into `dout` (device). Returns false on non-SPD
// or any device error.
template <typename T>
bool solveLmStepDevice(Handles& h, const T* dJ, int m, int n, const T* dr, T lambda, T* dout) {
  T *dA = nullptr, *dg = nullptr, *dwork = nullptr;
  int* dinfo = nullptr;
  bool ok = true;
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  auto check = [&](bool cond) { if (!cond) ok = false; return ok; };

  check(cudaMalloc(&dA, sizeof(T) * static_cast<size_t>(n) * n) == cudaSuccess);
  check(ok && cudaMalloc(&dg, sizeof(T) * n) == cudaSuccess);
  check(ok && cudaMalloc(&dinfo, sizeof(int)) == cudaSuccess);

  const T one = T(1), zero = T(0), neg_one = T(-1);
  // A = J^T J (lower triangle, n x n) via SYRK with trans=T over the m x n input J (lda=m).
  if (ok) check(cfSyrk(h.blas, uplo, n, m, &one, dJ, m, &zero, dA, n) == CUBLAS_STATUS_SUCCESS);
  // g = J^T r (n-vector).
  if (ok) check(cfGemvT(h.blas, m, n, &one, dJ, m, dr, &zero, dg) == CUBLAS_STATUS_SUCCESS);
  // Marquardt damping: scale the diagonal by (1 + lambda).
  if (ok) {
    const int threads = 128;
    scaleDiagonalKernel<T><<<(n + threads - 1) / threads, threads>>>(dA, n, T(1) + lambda);
    check(cudaGetLastError() == cudaSuccess);
  }
  // RHS = -g (so potrs yields dx = A^{-1}(-g)).
  if (ok) check(cfScal(h.blas, n, &neg_one, dg) == CUBLAS_STATUS_SUCCESS);

  int lwork = 0;
  if (ok) check(cfPotrfBuffer(h.solver, uplo, n, dA, n, &lwork) == CUSOLVER_STATUS_SUCCESS);
  if (ok && lwork > 0) check(cudaMalloc(&dwork, sizeof(T) * lwork) == cudaSuccess);
  if (ok) check(cfPotrf(h.solver, uplo, n, dA, n, dwork, lwork, dinfo) == CUSOLVER_STATUS_SUCCESS);
  int info = -1;
  if (ok) check(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  if (ok && info != 0) ok = false;  // not SPD: caller damps harder and retries
  if (ok) check(cfPotrs(h.solver, uplo, n, 1, dA, n, dg, n, dinfo) == CUSOLVER_STATUS_SUCCESS);
  int solve_info = -1;
  if (ok) check(cudaMemcpy(&solve_info, dinfo, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  if (ok && solve_info != 0) ok = false;
  if (ok) check(cudaMemcpy(dout, dg, sizeof(T) * n, cudaMemcpyDeviceToDevice) == cudaSuccess);

  cudaFree(dA);
  cudaFree(dg);
  cudaFree(dwork);
  cudaFree(dinfo);
  return ok;
}

}  // namespace

bool cudaSolveLmStep(const double* J_colmajor, int m, int n, const double* r, double lambda,
                     double* delta_out) {
  if (m <= 0 || n <= 0 || J_colmajor == nullptr || r == nullptr || delta_out == nullptr)
    return false;
  Handles& h = handles();
  if (!h.ok) return false;

  double *dJ = nullptr, *dr = nullptr, *dx = nullptr;
  bool ok = true;
  auto check = [&](bool cond) { if (!cond) ok = false; return ok; };
  check(cudaMalloc(&dJ, sizeof(double) * static_cast<size_t>(m) * n) == cudaSuccess);
  check(ok && cudaMalloc(&dr, sizeof(double) * m) == cudaSuccess);
  check(ok && cudaMalloc(&dx, sizeof(double) * n) == cudaSuccess);
  if (ok)
    check(cudaMemcpy(dJ, J_colmajor, sizeof(double) * static_cast<size_t>(m) * n,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  if (ok) check(cudaMemcpy(dr, r, sizeof(double) * m, cudaMemcpyHostToDevice) == cudaSuccess);
  if (ok) ok = solveLmStepDevice<double>(h, dJ, m, n, dr, lambda, dx);
  if (ok)
    check(cudaMemcpy(delta_out, dx, sizeof(double) * n, cudaMemcpyDeviceToHost) == cudaSuccess);
  cudaFree(dJ);
  cudaFree(dr);
  cudaFree(dx);
  return ok;
}

bool cudaSolveLmStepF32(const double* J_colmajor, int m, int n, const double* r, double lambda,
                        double* delta_out) {
  if (m <= 0 || n <= 0 || J_colmajor == nullptr || r == nullptr || delta_out == nullptr)
    return false;
  Handles& h = handles();
  if (!h.ok) return false;

  // Cast the host FP64 inputs down to float, solve entirely in single precision on the device,
  // cast dx back up to double. This is the genuine FP32 edge-precision path the parity test
  // compares against the FP64 GPU + host solves.
  std::vector<float> Jf(static_cast<size_t>(m) * n), rf(static_cast<size_t>(m));
  for (size_t i = 0; i < Jf.size(); ++i) Jf[i] = static_cast<float>(J_colmajor[i]);
  for (int i = 0; i < m; ++i) rf[static_cast<size_t>(i)] = static_cast<float>(r[i]);

  float *dJ = nullptr, *dr = nullptr, *dx = nullptr;
  bool ok = true;
  auto check = [&](bool cond) { if (!cond) ok = false; return ok; };
  check(cudaMalloc(&dJ, sizeof(float) * static_cast<size_t>(m) * n) == cudaSuccess);
  check(ok && cudaMalloc(&dr, sizeof(float) * m) == cudaSuccess);
  check(ok && cudaMalloc(&dx, sizeof(float) * n) == cudaSuccess);
  if (ok)
    check(cudaMemcpy(dJ, Jf.data(), sizeof(float) * Jf.size(), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  if (ok)
    check(cudaMemcpy(dr, rf.data(), sizeof(float) * rf.size(), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  if (ok) ok = solveLmStepDevice<float>(h, dJ, m, n, dr, static_cast<float>(lambda), dx);
  std::vector<float> xf(static_cast<size_t>(n));
  if (ok)
    check(cudaMemcpy(xf.data(), dx, sizeof(float) * n, cudaMemcpyDeviceToHost) == cudaSuccess);
  if (ok)
    for (int i = 0; i < n; ++i) delta_out[i] = static_cast<double>(xf[static_cast<size_t>(i)]);
  cudaFree(dJ);
  cudaFree(dr);
  cudaFree(dx);
  return ok;
}

}  // namespace calibforge
