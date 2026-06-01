#!/usr/bin/env python3
"""Dependency spike: PyPose (primary GPU bundle-adjustment borrow candidate).

Confirms, firsthand, the Theme-1 research claims that matter for adopting PyPose:
  1. The Levenberg-Marquardt optimizer runs and CONVERGES on CPU.
  2. Its default robust-loss corrector is FastTriggs (the GPU-stable, 1st-order
     corrector that motivated choosing PyPose over a Ceres-Triggs path).

This is a minimal API + convergence smoke. The research-critical PyPose-vs-Ceres
timing on the calibration-sized regime is DEFERRED to a CUDA host with Ceres
installed (this box has no GPU and no Ceres) — see docs/SPIKES.md.

CPU-only. Run inside the spike venv:
    .spikes/.venv/bin/python tools/spikes/spike_pypose_lm.py
"""
import sys
import torch
import pypose as pp


class PoseInv(torch.nn.Module):
    """Canonical PyPose LM smoke: recover the pose whose Exp inverts the input."""
    def __init__(self, *dim):
        super().__init__()
        self.pose = pp.Parameter(pp.randn_se3(*dim))

    def forward(self, input):
        return (self.pose.Exp() @ input).Log().tensor()


def main():
    from pypose.optim.kernel import Huber
    torch.manual_seed(0)
    print(f"pypose {pp.__version__}, torch {torch.__version__}")

    # 1) Plain LM convergence (no robust kernel => Trivial corrector, by design).
    model = PoseInv(2, 2)
    inp = pp.randn_SE3(2, 2)
    optimizer = pp.optim.LM(model)
    print(f"  no-kernel corrector = {type(optimizer.corrector[0]).__name__} (expected Trivial)")
    last = None
    for i in range(10):
        err = optimizer.step(inp)
        last = float(err)
        print(f"  it {i}: error = {last:.3e}")
        if last < 1e-5:
            break
    converged = last is not None and last < 1e-5

    # 2) With a robust kernel and no explicit corrector, PyPose defaults to FastTriggs
    #    (optimizer.py: `self.corrector = [FastTriggs(k) for k in kernel] if corrector is None`).
    #    This is the GPU-stable 1st-order corrector the research highlighted (Theme 1).
    robust = pp.optim.LM(PoseInv(2, 2), kernel=Huber())
    corr = type(robust.corrector[0]).__name__
    print(f"  robust-kernel(Huber) corrector = {corr} (expected FastTriggs)")
    fasttriggs = corr == "FastTriggs"

    ok = converged and fasttriggs
    print(f"\nPyPose spike: LM converged = {converged} (final {last:.2e}); "
          f"robust-kernel default corrector is FastTriggs = {fasttriggs}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
