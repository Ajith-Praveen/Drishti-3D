"""Tests for drishti3d.geometry.depth_fit -- per-view depth fit to BA points."""

from __future__ import annotations

import numpy as np

from drishti3d.geometry.depth_fit import ba_points_in_camera, fit_view_depth
from drishti3d.types import CameraIntrinsics, Pose

W, H = 200, 150
INTR = CameraIntrinsics(fx=180.0, fy=180.0, cx=100.0, cy=75.0, width=W, height=H)


def _true_depth_ramp() -> np.ndarray:
    """A sloped scene: depth runs 60 m -> 140 m across the image (enough spread for scale+shift)."""
    return np.tile(np.linspace(60.0, 140.0, W), (H, 1))


def _cam_points(depth: np.ndarray, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, W, n)
    iy = rng.integers(0, H, n)
    z = depth[iy, ix]
    x = (ix + 0.5 - INTR.cx) / INTR.fx * z
    y = (iy + 0.5 - INTR.cy) / INTR.fy * z
    return np.stack([x, y, z], axis=1)


def test_recovers_scale_and_shift_despite_outliers() -> None:
    truth = _true_depth_ramp()
    pred = (truth - 4.0) / 6.0  # backbone 6x shallow with an offset: truth = 6 * pred + 4
    cam = _cam_points(truth, 400)
    cam[:40, 2] *= 1.8  # 10% gross outliers

    fit = fit_view_depth(pred, INTR, cam)

    assert fit.applied, fit.failure
    assert fit.mode == "scale_shift"
    assert abs(fit.scale - 6.0) < 0.05
    assert abs(fit.shift - 4.0) < 1.0
    assert fit.inliers >= 340


def test_flat_ground_fits_scale_only() -> None:
    truth = np.full((H, W), 120.0)
    pred = truth / 6.0
    fit = fit_view_depth(pred, INTR, _cam_points(truth, 200))

    assert fit.applied
    assert fit.mode == "scale"
    assert fit.shift == 0.0
    assert abs(fit.scale - 6.0) < 1e-6


def test_too_few_points_is_refused_not_applied() -> None:
    truth = _true_depth_ramp()
    fit = fit_view_depth(truth, INTR, _cam_points(truth, 10))

    assert not fit.applied
    assert "only 10" in fit.failure


def test_implausible_scale_is_refused() -> None:
    truth = np.full((H, W), 120.0)
    fit = fit_view_depth(truth / 100.0, INTR, _cam_points(truth, 200))

    assert not fit.applied


def test_ba_points_in_camera_uses_world_from_camera_pose() -> None:
    # Camera at (10, 0, 100) looking straight down: camera Z = world -Z.
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    pose = Pose(R=R, t=np.array([10.0, 0.0, 100.0]))
    points = np.array([[10.0, 0.0, 0.0], [12.0, 3.0, 5.0], [0.0, 0.0, 0.0]])
    cam_idx = np.array([0, 0, 1])
    pt_idx = np.array([0, 1, 2])

    cam = ba_points_in_camera(points, cam_idx, pt_idx, 0, pose)

    assert cam.shape == (2, 3)
    np.testing.assert_allclose(cam[0], [0.0, 0.0, 100.0])
    np.testing.assert_allclose(cam[1], [2.0, -3.0, 95.0])


def test_course_yaw_points_image_top_along_travel() -> None:
    from drishti3d.geometry.bundle import gimbal_to_R
    from drishti3d.pipeline.stages import _course_yaw_by_keyframe

    # Flying east, then a hover (no motion) at the end.
    enu = {0: np.array([0.0, 0, 100]), 1: np.array([10.0, 0, 100]), 2: np.array([20.0, 0, 100]), 3: np.array([20.2, 0, 100])}
    yaws = _course_yaw_by_keyframe(enu, 4)

    for i in range(4):
        R = gimbal_to_R(yaws[i], -90.0, 0.0)
        np.testing.assert_allclose(-R[:, 1], [1.0, 0.0, 0.0], atol=1e-9)  # image up = east
        np.testing.assert_allclose(R[:, 2], [0.0, 0.0, -1.0], atol=1e-9)  # looking down


def test_scale_field_removes_a_tilt_the_affine_fit_cannot() -> None:
    truth = np.full((H, W), 280.0)
    u, v = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    pred = truth / (1.0 + 0.04 * u - 0.03 * v + 0.02 * u * u)  # smooth tilt/bowl error, ~4%
    cam = _cam_points(truth, 1300)

    affine = fit_view_depth(pred, INTR, cam, spatial=False)
    field = fit_view_depth(pred, INTR, cam)

    assert field.mode == "scale_field"
    assert field.residual_m < 0.3 < affine.residual_m
    corrected = field.apply(pred)
    assert np.abs(corrected - truth).max() < 1.0


def test_field_apply_uses_same_pixel_centres_as_fit():
    from drishti3d.geometry.depth_fit import DepthFit

    fit = DepthFit(applied=True, field=np.array([1., 0.1, 0.2, 0., 0., 0.]))
    depth = np.ones((2, 2)) * 100
    np.testing.assert_allclose(fit.apply(depth), [[85, 95], [105, 115]])


def test_field_refuses_unsafe_interior_extremum():
    from drishti3d.geometry.depth_fit import DepthFit, _fit_field

    rng = np.random.default_rng(10)
    u, v = rng.uniform(-1, 1, (2, 500))
    zp = np.ones(500) * 10
    # Corners and centre are positive, but this bowl inverts at (0.5, 0).
    zt = zp * ((u - 0.5) ** 2 + v ** 2 - 0.1)
    fit = DepthFit(applied=True, scale=1, residual_m=100)
    _fit_field(fit, zt, zp, u, v, 0.05, 50)
    assert fit.field is None


def test_world_frame_depth_is_refined_after_metric_fit(monkeypatch):
    from types import SimpleNamespace

    from drishti3d.geometry.backbone import BackboneResult
    from drishti3d.geometry.depth_fit import DepthFit
    from drishti3d.pipeline.stages import _fit_window_to_ba

    pose = Pose(R=np.eye(3), t=np.array([10., 0., 0.]))
    intr = CameraIntrinsics(fx=2, fy=2, cx=1, cy=1, width=2, height=2)
    result = BackboneResult(
        poses=[Pose(R=np.eye(3), t=np.zeros(3))],
        points=np.zeros((1, 2, 2, 3)), depth=np.ones((1, 2, 2)),
        confidence=np.ones((1, 2, 2)), intrinsics=[intr], is_metric=True,
        images=np.zeros((1, 2, 2, 3), np.uint8),
    )
    state = SimpleNamespace(ba_points=np.array([[10., 0., 2.]]),
        ba_obs_camera_idx=np.array([0]), ba_obs_point_idx=np.array([0]),
        poses=[pose], geometry_world_frame=True, depth_anchor_diags={},
        fit_points=np.zeros((0, 3)))  # empty densification must fall back to BA
    monkeypatch.setattr("drishti3d.geometry.depth_fit.fit_view_depth",
                        lambda *a, **kw: DepthFit(applied=True, scale=2))
    called = []

    def sweep(images, depth, poses, intrinsics, **kwargs):
        np.testing.assert_array_equal(depth, 2)
        assert poses[0] is pose
        called.append(True)
        return SimpleNamespace(depth=depth + 1, stats={"refined_pct": 100})

    monkeypatch.setattr("drishti3d.geometry.plane_sweep.refine_depth_by_plane_sweep", sweep)
    cfg = SimpleNamespace(ba_depth_fit=True, plane_sweep=True)
    output = _fit_window_to_ba(state, result, SimpleNamespace(keyframe_indices=lambda: [0]), cfg, 0)
    assert called
    np.testing.assert_array_equal(output.depth, 3)
    np.testing.assert_allclose(output.points[0, 0, 0], [9.25, -0.75, 3])
    assert state.depth_anchor_diags[0]["plane_sweep"]["refined_pct"] == 100

    from drishti3d.pipeline.stages import _points_from_depth

    invalid_depth = output.depth.copy()
    invalid_depth[0, 0, 0] = 0
    invalid_depth[0, 0, 1] = np.inf
    invalid = _points_from_depth(output, invalid_depth)
    assert np.isnan(invalid.points[0, 0]).all()
    assert (invalid.confidence[0, 0] == 0).all()
