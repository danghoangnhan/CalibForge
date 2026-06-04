#pragma once
//
// CalibForge solve — the SOLVER-AGNOSTIC problem interface (INTERFACE STUB).
//
// ⚠️ RULE #1 (docs/RESEARCH.md Theme 1): GPU is NOT automatically faster. A single
// small calibration (few cameras) is typically faster on a CPU solver (Ceres) than
// on a batched GPU solver (~25x in Theseus's benchmarks). The GPU path earns its
// keep on batched fleet / large rigs / generic 10^4-param models / online work.
// => CPU is the DEFAULT backend for a single small calibration. Keep this interface
//    backend-agnostic so we can route per-problem and swap solvers freely.

#include <cstddef>
#include <memory>
#include <vector>

#include "calibforge/residual_block.hpp"

namespace calibforge {

enum class SolverBackend {
  Auto,         // size-based routing (small/single -> CPU, batched/large -> GPU) is the INTENT;
                // currently Auto == CPU until a measured crossover threshold is wired (RULE #1).
  CpuCeres,     // BSD; fastest on small single problems; also the accuracy oracle
  GpuCuda,      // native CalibForge CUDA dense LM (cuBLAS SYRK/GEMV + cuSOLVER Cholesky); built
                // when CALIBFORGE_HAS_CUDA, falls back to CPU otherwise. See cuda_linear_solver.hpp.
  GpuPyPose,    // Apache-2.0; sparse-Jacobian LM + FastTriggs; batched/online (not yet wired)
  GpuGraphite,  // MIT (WIP); implicit-Schur + bf16; edge / low-VRAM (not yet wired)
};

struct SolveOptions {
  SolverBackend backend = SolverBackend::Auto;
  int max_iterations = 100;
  double function_tolerance = 1e-6;
  bool deterministic = false;  // require CPU<->GPU and FP32/bf16<->FP64 parity (unproven; must test)
};

struct SolveSummary {
  bool converged = false;
  int iterations = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  double rms_reprojection_px = 0.0;
};

// Backend-agnostic nonlinear-least-squares problem. Implementations dispatch to
// the chosen SolverBackend.
class Problem {
 public:
  virtual ~Problem() = default;

  virtual void addResidualBlock(std::unique_ptr<ResidualBlock> block,
                                std::vector<double*> parameter_blocks) = 0;

  // Gauge freedom / datum fixing: hold a parameter block constant. (One of the
  // hard problems CalibForge must implement itself — see docs/RESEARCH.md Theme 1.)
  virtual void setParameterBlockConstant(double* parameter_block) = 0;

  virtual SolveSummary solve(const SolveOptions& options) = 0;
};

}  // namespace calibforge
