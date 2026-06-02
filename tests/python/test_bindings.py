"""pytest for the CalibForge pybind11 bindings (built with -DCALIBFORGE_PYTHON=ON).

Cross-checks the Python API against the same synthetic recovery the C++ suite uses.
"""
import numpy as np
import calibforge as cf


def se3(axis_angle, t):
    """4x4 SE(3) from an axis-angle rotation (Rodrigues) and a translation."""
    w = np.asarray(axis_angle, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        R = np.eye(3)
    else:
        k = w / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float)
    return T


def test_pinhole_project_matches_formula():
    cam = cf.PinholeCamera(500, 500, 320, 240)
    u, v = cam.project([0.1, 0.2, 2.0])
    assert abs(u - (500 * 0.1 / 2.0 + 320)) < 1e-9
    assert abs(v - (500 * 0.2 / 2.0 + 240)) < 1e-9
    assert cam.name() == "pinhole"
    assert cam.num_params() == 4


def test_generic_models_construct_and_project():
    for cam in (
        cf.KannalaBrandtCamera(350, 350, 320, 240, -0.05, 0.01, 0.0, 0.0),
        cf.DoubleSphereCamera(320, 320, 320, 240, 0.2, 0.55),
        cf.EUCMCamera(320, 320, 320, 240, 0.6, 1.1),
    ):
        u, v = cam.project([0.1, 0.2, 1.0])
        assert np.isfinite(u) and np.isfinite(v)


def test_assess_observability_identity():
    rep = cf.assess_observability(np.eye(5))
    assert rep.observable
    assert abs(rep.confidence - 1.0) < 1e-9


def test_calibrate_single_recovers_pinhole():
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    gt = cf.PinholeCamera(fx, fy, cx, cy)
    board = [[c * 0.1, r * 0.1, 0.0] for r in range(4) for c in range(4)]
    gt_poses = [
        se3([0.10, -0.05, 0.02], [-0.15, -0.15, 1.5]),
        se3([-0.20, 0.15, -0.05], [-0.20, -0.10, 1.2]),
        se3([0.05, 0.25, 0.10], [-0.10, -0.20, 1.8]),
        se3([0.15, 0.10, -0.15], [-0.25, -0.05, 1.4]),
    ]
    obj, img = [], []
    for T in gt_poses:
        o, im = [], []
        for X in board:
            Xc = (T @ np.array([X[0], X[1], X[2], 1.0]))[:3]
            o.append([X[0], X[1], X[2]])
            im.append(list(gt.project([float(Xc[0]), float(Xc[1]), float(Xc[2])])))
        obj.append(o)
        img.append(im)

    dpert = se3([0.02, -0.01, 0.01], [0.02, -0.02, 0.02])
    poses0 = [T @ dpert for T in gt_poses]
    intr0 = [470.0, 530.0, 305.0, 255.0]

    res = cf.calibrate_single("pinhole", obj, img, intr0, poses0, max_iter=100)
    assert res["converged"]
    assert res["final_cost"] < 1e-7
    intr = res["intrinsics"]
    assert abs(intr[0] - fx) < 1e-2
    assert abs(intr[2] - cx) < 1e-2

    rep = cf.assess_observability(np.asarray(res["information"]))
    assert rep.observable
