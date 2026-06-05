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


def test_generic_bspline_model_and_yaml_roundtrip():
    grid = cf.GenericBSplineGrid()
    grid.nx, grid.ny, grid.image_w, grid.image_h, grid.margin = 14, 10, 640, 480, -25.0
    cam = cf.GenericBSplineCamera(grid)
    # fit the dense ray field to a wide-FOV double-sphere source
    cam.fit_from_parametric("double_sphere", [320.0, 320.0, 320.0, 240.0, 0.2, 0.55])
    assert cam.name() == "generic_bspline"
    assert cam.num_params() == 3 * 14 * 10
    u, v = cam.project([0.1, 0.05, 1.0])
    assert np.isfinite(u) and np.isfinite(v)
    # YAML round-trip preserves the grid + control points
    text = cf.generic_bspline_to_yaml(cam)
    back = cf.generic_bspline_from_yaml(text)
    assert back.grid().nx == 14 and back.grid().ny == 10
    assert back.grid().image_w == 640 and back.grid().image_h == 480
    assert abs(back.grid().margin - (-25.0)) < 1e-12
    assert np.max(np.abs(np.asarray(cam.params()) - np.asarray(back.params()))) < 1e-12


def test_calibrate_generic_bspline_fits_wide_fov():
    grid = cf.GenericBSplineGrid()
    grid.nx, grid.ny, grid.image_w, grid.image_h, grid.margin = 14, 10, 640, 480, -25.0
    gt = cf.DoubleSphereCamera(320.0, 320.0, 320.0, 240.0, 0.2, 0.55)
    board = [[c * 0.08 - 0.24, r * 0.08 - 0.24, 0.0] for r in range(7) for c in range(7)]
    poses = [
        se3([0.25, -0.10, 0.05], [-0.05, -0.05, 0.7]),
        se3([-0.30, 0.20, -0.10], [-0.08, -0.04, 0.65]),
        se3([0.10, 0.35, 0.15], [-0.04, -0.08, 0.8]),
        se3([0.20, 0.15, -0.25], [-0.10, -0.03, 0.6]),
        se3([-0.15, -0.25, 0.20], [-0.06, -0.10, 0.75]),
        se3([0.30, -0.20, -0.10], [-0.03, -0.05, 0.55]),
    ]
    obj, img = [], []
    for T in poses:
        o, im = [], []
        for X in board:
            Xc = (T @ np.array([X[0], X[1], X[2], 1.0]))[:3]
            px = gt.project([float(Xc[0]), float(Xc[1]), float(Xc[2])])
            if not (np.isfinite(px[0]) and 0 <= px[0] <= 640 and 0 <= px[1] <= 480):
                continue
            o.append([X[0], X[1], X[2]])
            im.append(list(px))
        obj.append(o)
        img.append(im)
    res = cf.calibrate_generic_bspline(
        grid, "double_sphere", [320.0, 320.0, 320.0, 240.0, 0.2, 0.55],
        obj, img, poses, max_iter=120, optimize_poses=False)
    assert res["converged"]
    assert res["num_residuals"] > 0
    assert res["rms_reprojection_px"] < 0.05
    # Functional equivalence (mirrors the C++ twin): rebuild the fitted B-spline from the returned
    # control points and confirm it reproduces the double-sphere SOURCE on the calibrated rays —
    # not merely that it drove its own residuals down (which `converged` already implies).
    fitted = cf.GenericBSplineCamera(grid)
    fitted.set_params(res["control_points"])
    errs = []
    for X in board:
        for T in poses:
            Xc = (T @ np.array([X[0], X[1], X[2], 1.0]))[:3]
            p = [float(Xc[0]), float(Xc[1]), float(Xc[2])]
            ref = gt.project(p)
            if not (np.isfinite(ref[0]) and 0 <= ref[0] <= 640 and 0 <= ref[1] <= 480):
                continue
            got = fitted.project(p)
            errs.append(((got[0] - ref[0]) ** 2 + (got[1] - ref[1]) ** 2) ** 0.5)
    assert len(errs) > 0
    assert max(errs) < 0.1  # sub-0.1px functional reproduction of the source field


def test_stereo_rectification():
    # Calibrated pair: small relative rotation + ~10 cm x-baseline (Xc1 = R*Xc0 + t).
    T = se3([0.02, -0.01, 0.015], [-0.10, 0.005, 0.002])
    rect = cf.compute_stereo_rectification(
        "pinhole", [500.0, 500.0, 320.0, 240.0],
        "pinhole", [510.0, 508.0, 322.0, 238.0],
        T, 640, 480)
    b = (0.10 ** 2 + 0.005 ** 2 + 0.002 ** 2) ** 0.5
    assert abs(rect["baseline"] - b) < 1e-9
    f = rect["K_rect"][0]
    assert abs(rect["P1"][3] - (-f * b)) < 1e-6   # right-camera baseline term
    R0 = np.asarray(rect["R0"])
    assert R0.shape == (3, 3)
    assert np.max(np.abs(R0 @ R0.T - np.eye(3))) < 1e-9   # orthonormal
    assert abs(np.linalg.det(R0) - 1.0) < 1e-9
    # Q maps disparity back to depth Z = f*b/disp.
    Q = np.asarray(rect["Q"]).reshape(4, 4)
    Z = 2.5
    xyzw = Q @ np.array([350.0, 250.0, f * b / Z, 1.0])
    assert abs(xyzw[2] / xyzw[3] - Z) < 1e-9
