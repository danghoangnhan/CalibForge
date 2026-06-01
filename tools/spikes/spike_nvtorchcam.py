#!/usr/bin/env python3
"""Dependency spike: nvTorchCam (camera-model core borrow candidate).

Confirms, firsthand, the Theme-4 research claims that matter for adopting nvTorchCam:
  1. project_to_pixel / pixel_to_ray round-trip is consistent for a PINHOLE camera.
  2. The same round-trip holds for a KANNALA-BRANDT fisheye, exercising the
     differentiable Newton unprojection (the numerically interesting path).
  3. No DoubleSphere / EUCM model ships (the gap CalibForge must fill).

CPU-only; no GPU required. Run inside the spike venv:
    .spikes/.venv/bin/python tools/spikes/spike_nvtorchcam.py
"""
import sys
import torch
from nvtorchcam import cameras


def _cosine(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(-1)


def roundtrip(cam, pts, label, tol=1e-4):
    pix, depth, valid = cam.project_to_pixel(pts)
    origin, dirs, valid2 = cam.pixel_to_ray(pix, unit_vec=True)
    cos = _cosine(pts, dirs)                      # rays should be parallel to original points
    worst = (1.0 - cos[valid & valid2]).abs().max().item() if (valid & valid2).any() else float("nan")
    ok = worst < tol
    print(f"  [{ 'PASS' if ok else 'FAIL'}] {label}: worst |1-cos(angle)| = {worst:.2e} "
          f"(valid {int((valid & valid2).sum())}/{pts.shape[0]})")
    return ok


def main():
    torch.manual_seed(0)
    print(f"nvtorchcam {getattr(__import__('nvtorchcam'), '__version__', '?')}, torch {torch.__version__}")
    results = []

    # 1) Pinhole round-trip
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    pin = cameras.PinholeCamera.make(K)
    pts = torch.tensor([[0.10, 0.00, 1.0], [-0.20, 0.10, 2.0], [0.05, -0.15, 1.5],
                        [0.30, 0.20, 3.0], [0.00, 0.00, 1.0]])
    results.append(roundtrip(pin, pts, "pinhole project->unproject"))

    # 2) Kannala-Brandt fisheye round-trip (exercises differentiable Newton inverse)
    Kf = torch.tensor([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    dist = torch.tensor([0.1, 0.05, 0.0, 0.0])    # k1..k4
    fish = cameras.OpenCVFisheyeCamera.make(Kf, dist, theta_max=1.4)
    ptf = torch.tensor([[0.2, 0.0, 1.0], [-0.3, 0.2, 1.0], [0.1, -0.4, 1.2],
                        [0.5, 0.3, 1.0], [0.0, 0.0, 1.0]])
    results.append(roundtrip(fish, ptf, "KB-fisheye project->unproject (Newton)"))

    # 3) Confirm DoubleSphere / EUCM are absent (the gap)
    cam_classes = [n for n in dir(cameras) if n.endswith("Camera")]
    missing = [m for m in ("DoubleSphereCamera", "EUCMCamera", "UnifiedCamera") if m not in cam_classes]
    gap_ok = len(missing) == 3
    print(f"  [{'PASS' if gap_ok else 'NOTE'}] DS/EUCM gap confirmed: "
          f"none of DoubleSphere/EUCM/Unified present. Models shipped = {cam_classes}")
    results.append(gap_ok)

    ok = all(results)
    print(f"\nnvTorchCam spike: {'ALL PASS' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
