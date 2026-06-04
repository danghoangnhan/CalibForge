#pragma once
//
// CalibForge solve — concrete dense Problem over the manifold LM (header-only, CPU).
//
// Realizes the SOLVER-AGNOSTIC Problem/ResidualBlock interface: each parameter block
// carries a LocalParameterization (Euclidean or SE3); residual blocks emit minimal
// analytic Jacobians; the SAME Marquardt loop calibrate_single proved assembles and
// solves the nonlinear least squares. CPU is the default path for a single small
// calibration (docs/RESEARCH.md Theme 1: GPU is not automatically faster).
//
// Usage: addParameterBlock() every block's pointer + manifold, addResidualBlock() the
// residuals, then solveLm()/solve(). informationMatrix() returns J^T J at the solution.

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "calibforge/cuda_linear_solver.hpp"  // cudaSolverAvailable / cudaSolveLmStep (GPU backend)
#include "calibforge/least_squares.hpp"  // LmOptions, LmSummary
#include "calibforge/manifold.hpp"
#include "calibforge/problem.hpp"
#include "calibforge/residual_block.hpp"

namespace calibforge {

class DenseProblem : public Problem {
 public:
  // Register a parameter block (the doubles at `ptr`) with its manifold. Must be called
  // for every block a residual references, before solving. Pointers must stay valid and
  // stable (don't reallocate the backing storage) for the problem's lifetime.
  void addParameterBlock(double* ptr, std::shared_ptr<LocalParameterization> param) {
    if (index_.count(ptr)) return;
    index_[ptr] = static_cast<int>(blocks_.size());
    blocks_.push_back(Block{ptr, std::move(param), false, -1});
  }

  void addResidualBlock(std::unique_ptr<ResidualBlock> block,
                        std::vector<double*> parameter_blocks) override {
    std::vector<int> ids;
    ids.reserve(parameter_blocks.size());
    for (double* p : parameter_blocks) {
      const auto it = index_.find(p);
      ids.push_back(it == index_.end() ? -1 : it->second);  // -1 = unregistered (caller error)
    }
    residuals_.push_back(Res{std::move(block), std::move(ids), 0});
  }

  void setParameterBlockConstant(double* parameter_block) override {
    const auto it = index_.find(parameter_block);
    if (it != index_.end()) blocks_[it->second].constant = true;
  }

  // Direct LM entry — calibrate_single uses this to preserve v0.1 numeric behavior.
  // `backend` selects the per-iteration dense linear solver: CpuCeres/Auto keep the host Eigen
  // path (byte-identical, default); GpuCuda offloads (J^T J + lambda diag) dx = -J^T r to the
  // GPU (cuBLAS/cuSOLVER) when available, else transparently falls back to the host path.
  LmSummary solveLm(const LmOptions& opts = LmOptions{},
                    SolverBackend backend = SolverBackend::CpuCeres) {
    const bool use_gpu = (backend == SolverBackend::GpuCuda) && cudaSolverAvailable();
    // Tangent-column offsets for non-constant blocks; row offsets for residuals.
    int n = 0;
    for (auto& b : blocks_) {
      if (b.constant) { b.col = -1; continue; }
      b.col = n;
      n += b.param->tangentSize();
    }
    int m = 0;
    for (auto& r : residuals_) {
      r.row = m;
      m += static_cast<int>(r.block->residualDim());
    }
    n_tangent_ = n;
    m_ = m;

    // Robust loss (FastTriggs IRLS): weights are computed per residual block and held
    // fixed across each inner trial loop, then refreshed at the accepted point.
    robust_ = opts.robust;
    robust_active_ = (robust_.kernel != RobustKernel::None);
    block_weight_.assign(residuals_.size(), 1.0);
    computeWeights();

    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    evaluate(true, r, J);
    double cost = 0.5 * r.squaredNorm();

    LmSummary s;
    s.initial_cost = cost;
    double lambda = opts.initial_lambda;
    int it = 0;
    for (; it < opts.max_iterations; ++it) {
      const Eigen::VectorXd g = J.transpose() * r;
      if (g.norm() < opts.gradient_tolerance) { s.converged = true; break; }
      // J^T J is only needed for the HOST damped solve; the GPU path forms it on the device.
      Eigen::MatrixXd JtJ;
      if (!use_gpu) JtJ = J.transpose() * J;

      bool step_accepted = false;
      for (int tries = 0; tries < 12; ++tries) {
        Eigen::VectorXd dx;
        if (use_gpu) {
#ifdef CALIBFORGE_HAS_CUDA
          // (J^T J + lambda diag(J^T J)) dx = -J^T r solved on the GPU (cuBLAS + cuSOLVER).
          dx.resize(n_tangent_);
          if (!cudaSolveLmStep(J.data(), m_, n_tangent_, r.data(), lambda, dx.data())) {
            lambda *= 3.0;  // damped matrix not SPD on device: damp harder, retry (host reject path)
            continue;
          }
#endif
        } else {
          Eigen::MatrixXd A = JtJ;
          A.diagonal() += lambda * JtJ.diagonal();  // Marquardt (scale-invariant) damping
          dx = A.ldlt().solve(-g);
        }

        // Apply the retraction in place, saving originals so a reject can restore.
        std::vector<std::vector<double>> saved(blocks_.size());
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi) {
          Block& b = blocks_[bi];
          if (b.constant) continue;
          const int a = b.param->ambientSize();
          saved[bi].assign(b.ptr, b.ptr + a);
          std::vector<double> xo(static_cast<std::size_t>(a));
          b.param->retract(b.ptr, &dx[b.col], xo.data());
          std::copy(xo.begin(), xo.end(), b.ptr);
        }

        Eigen::VectorXd r_t;
        Eigen::MatrixXd unused;
        evaluate(false, r_t, unused);
        const double cost_t = 0.5 * r_t.squaredNorm();

        if (cost_t < cost) {
          const double rel = (cost - cost_t) / std::max(cost, 1e-300);
          const double step_norm = dx.norm();
          computeWeights();      // IRLS: refresh robust weights at the accepted point
          evaluate(true, r, J);  // re-linearize (with refreshed weights)
          cost = 0.5 * r.squaredNorm();  // == cost_t when non-robust (weights all 1)
          lambda = std::max(lambda * 0.3, 1e-12);
          step_accepted = true;
          const double scale = 1.0 + static_cast<double>(n_tangent_);
          if (rel < opts.function_tolerance) s.converged = true;
          if (step_norm < opts.parameter_tolerance * (scale + opts.parameter_tolerance))
            s.converged = true;
          break;
        }
        // Reject: restore originals, damp harder, retry.
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi) {
          if (blocks_[bi].constant) continue;
          std::copy(saved[bi].begin(), saved[bi].end(), blocks_[bi].ptr);
        }
        lambda *= 3.0;
      }

      if (s.converged) { ++it; break; }
      if (!step_accepted) break;  // could not decrease even with heavy damping
    }
    s.iterations = it;
    s.final_cost = cost;

    evaluate(true, r, J);  // re-linearize at the final solution for the information matrix
    information_ = J.transpose() * J;
    return s;
  }

  SolveSummary solve(const SolveOptions& options) override {
    LmOptions lm;
    lm.max_iterations = options.max_iterations;
    lm.function_tolerance = options.function_tolerance;
    const LmSummary s = solveLm(lm, options.backend);
    SolveSummary out;
    out.converged = s.converged;
    out.iterations = s.iterations;
    out.initial_cost = s.initial_cost;
    out.final_cost = s.final_cost;
    out.rms_reprojection_px = (m_ > 0) ? std::sqrt(2.0 * s.final_cost / m_) : 0.0;
    return out;
  }

  const Eigen::MatrixXd& informationMatrix() const { return information_; }
  int numResiduals() const { return m_; }
  int tangentSize() const { return n_tangent_; }

 private:
  struct Block {
    double* ptr;
    std::shared_ptr<LocalParameterization> param;
    bool constant;
    int col;  // tangent column offset, or -1 if constant
  };
  struct Res {
    std::unique_ptr<ResidualBlock> block;
    std::vector<int> ids;  // indices into blocks_
    int row;               // residual row offset
  };

  void evaluate(bool need_jac, Eigen::VectorXd& r, Eigen::MatrixXd& J) {
    r.resize(m_);
    if (need_jac) J.setZero(m_, n_tangent_);
    for (std::size_t bidx = 0; bidx < residuals_.size(); ++bidx) {
      Res& res = residuals_[bidx];
      const int rd = static_cast<int>(res.block->residualDim());
      std::vector<const double*> ps(res.ids.size());
      for (std::size_t k = 0; k < res.ids.size(); ++k) ps[k] = blocks_[res.ids[k]].ptr;

      std::vector<double*> jptrs(res.ids.size(), nullptr);
      std::vector<std::vector<double>> jstore(res.ids.size());
      if (need_jac) {
        for (std::size_t k = 0; k < res.ids.size(); ++k) {
          const Block& b = blocks_[res.ids[k]];
          if (b.constant) continue;  // constant block: don't request its Jacobian
          jstore[k].assign(static_cast<std::size_t>(rd) * b.param->tangentSize(), 0.0);
          jptrs[k] = jstore[k].data();
        }
      }
      res.block->evaluate(ps.data(), &r[res.row], need_jac ? jptrs.data() : nullptr);

      // FastTriggs: scale the block's residual + Jacobian rows by w = sqrt(rho'(s)).
      const double w = robust_active_ ? block_weight_[bidx] : 1.0;
      if (robust_active_ && w != 1.0)
        for (int i = 0; i < rd; ++i) r[res.row + i] *= w;

      if (need_jac) {
        for (std::size_t k = 0; k < res.ids.size(); ++k) {
          const Block& b = blocks_[res.ids[k]];
          if (b.constant || jptrs[k] == nullptr) continue;
          const int ts = b.param->tangentSize();
          for (int i = 0; i < rd; ++i)
            for (int c = 0; c < ts; ++c)
              J(res.row + i, b.col + c) = w * jstore[k][static_cast<std::size_t>(i) * ts + c];
        }
      }
    }
  }

  // Recompute per-residual-block robust weights from the current (unweighted) residuals.
  void computeWeights() {
    if (!robust_active_) return;
    block_weight_.assign(residuals_.size(), 1.0);
    std::vector<double> rbuf;
    for (std::size_t bidx = 0; bidx < residuals_.size(); ++bidx) {
      Res& res = residuals_[bidx];
      const int rd = static_cast<int>(res.block->residualDim());
      std::vector<const double*> ps(res.ids.size());
      for (std::size_t k = 0; k < res.ids.size(); ++k) ps[k] = blocks_[res.ids[k]].ptr;
      if (static_cast<int>(rbuf.size()) < rd) rbuf.resize(static_cast<std::size_t>(rd));
      res.block->evaluate(ps.data(), rbuf.data(), nullptr);
      double s = 0.0;
      for (int i = 0; i < rd; ++i) s += rbuf[i] * rbuf[i];
      block_weight_[bidx] = robustWeightSqrt(robust_, s);
    }
  }

  std::map<double*, int> index_;
  std::vector<Block> blocks_;
  std::vector<Res> residuals_;
  Eigen::MatrixXd information_;
  int n_tangent_ = 0;
  int m_ = 0;

  RobustLoss robust_ = {};
  bool robust_active_ = false;
  std::vector<double> block_weight_;  // one per residual block (FastTriggs sqrt(rho'(s)))
};

}  // namespace calibforge
