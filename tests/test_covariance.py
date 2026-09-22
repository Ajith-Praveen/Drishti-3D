"""Tests for drishti3d.geometry.covariance and drishti3d.geometry.observability.

``observability.py`` tests live here (not in a separate file) because they
build directly on ``point_covariances``/``covariance_to_ellipsoid`` and
this repository's file-ownership split for this task did not include a
dedicated ``test_observability.py`` -- see this module's own docstring
comments at each observability test for what they cover.

The single most important test in this file is
``test_anisotropy_dramatically_higher_for_collinear_single_pass`` -- it is
the empirical claim the project's whole "novelty" pitch rests on: BA
covariance for a single-pass collinear flight is anisotropic in a way a
well-conditioned multi-view configuration's is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.geometry.bundle import BAConfig, BAProblem, bundle_adjust
from drishti3d.geometry.covariance import (
    ConfidenceThresholds,
    anisotropy_ratio,
    camera_covariances,
    confidence_from_covariance,
    covariance_to_ellipsoid,
    point_covariances,
)
from drishti3d.geometry.observability import (
    PlanConfig,
    observability_field,
    plan_corrective_flight,
    required_viewing_direction,
    uncertainty_heatmap,
    worst_constrained_direction,
)
from drishti3d.types import CameraIntrinsics, Confidence, Pose

_INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def _project_all(
    poses: list[Pose], intrinsics: list[CameraIntrinsics], points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_cam: list[int] = []
    obs_pt: list[int] = []
    obs_uv: list[list[float]] = []
    for ci, (pose, intr) in enumerate(zip(poses, intrinsics, strict=True)):
        Xc = (pose.R.T @ (points - pose.t).T).T
        z = Xc[:, 2]
        valid = z > 1.0
        u = intr.fx * Xc[:, 0] / z + intr.cx
        v = intr.fy * Xc[:, 1] / z + intr.cy
        inb = valid & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
        for pi in np.where(inb)[0]:
            obs_cam.append(ci)
            obs_pt.append(int(pi))
            obs_uv.append([u[pi], v[pi]])
    return np.array(obs_cam), np.array(obs_pt), np.array(obs_uv)


def _well_conditioned_poses(n_cams: int, radius: float, altitude: float) -> list[Pose]:
    """Cameras spread around a hemisphere looking at the origin -- wide triangulation angles."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_cams, endpoint=False)
    centres = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_cams, altitude)], axis=1)
    poses = []
    for c in centres:
        fwd = -c / np.linalg.norm(c)
        up_guess = np.array([0.0, 0.0, 1.0])
        x = np.cross(up_guess, fwd)
        x /= np.linalg.norm(x)
        y = np.cross(fwd, x)
        R = np.stack([x, y, fwd], axis=1)
        poses.append(Pose(R=R, t=c))
    return poses


def _collinear_nadir_poses(n_cams: int, span: float, altitude: float) -> list[Pose]:
    centres = np.stack([np.linspace(-span / 2, span / 2, n_cams), np.zeros(n_cams), np.full(n_cams, altitude)], axis=1)
    R_nadir = np.diag([1.0, -1.0, -1.0])
    return [Pose(R=R_nadir.copy(), t=c) for c in centres]


def _run_ba(poses: list[Pose], pts: np.ndarray, pixel_sigma: float = 1.0, robust_loss=None):
    intrinsics = [_INTRINSICS] * len(poses)
    obs_cam, obs_pt, obs_uv = _project_all(poses, intrinsics, pts)
    problem = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
    )
    cfg = BAConfig(max_iterations=100, robust_loss=robust_loss, pixel_sigma_px=pixel_sigma)
    result = bundle_adjust(problem, cfg)
    return result, obs_cam, obs_pt


# ---------------------------------------------------------------------------
# covariance_to_ellipsoid / anisotropy_ratio basic sanity
# ---------------------------------------------------------------------------


def test_covariance_to_ellipsoid_isotropic():
    cov = np.eye(3) * 0.04
    semi_axes, rotation = covariance_to_ellipsoid(cov)
    np.testing.assert_allclose(semi_axes, [0.2, 0.2, 0.2])
    # Rotation must still be a valid orthonormal basis.
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert anisotropy_ratio(cov) == pytest.approx(1.0)


def test_covariance_to_ellipsoid_batched_matches_single():
    covs = np.stack([np.diag([0.01, 0.04, 0.09]), np.eye(3) * 0.25], axis=0)
    semi_axes, rotation = covariance_to_ellipsoid(covs)
    assert semi_axes.shape == (2, 3)
    for i in range(2):
        single_axes, single_rot = covariance_to_ellipsoid(covs[i])
        np.testing.assert_allclose(semi_axes[i], single_axes)
        np.testing.assert_allclose(rotation[i], single_rot)


def test_anisotropy_ratio_known_elongation():
    cov = np.diag([0.0001, 0.0001, 1.0])  # elongated 100x along Z
    ratio = anisotropy_ratio(cov)
    assert ratio == pytest.approx(100.0, rel=1e-6)


def test_confidence_from_covariance_thresholds():
    thresholds = ConfidenceThresholds(measured_max_m=0.05, low_confidence_max_m=0.5)
    tight = np.eye(3) * (0.01**2)
    medium = np.eye(3) * (0.2**2)
    loose = np.eye(3) * (2.0**2)
    assert confidence_from_covariance(tight, thresholds) == Confidence.MEASURED
    assert confidence_from_covariance(medium, thresholds) == Confidence.LOW_CONFIDENCE
    assert confidence_from_covariance(loose, thresholds) == Confidence.INFERRED

    batch = np.stack([tight, medium, loose])
    tiers = confidence_from_covariance(batch, thresholds)
    np.testing.assert_array_equal(tiers, [Confidence.MEASURED, Confidence.LOW_CONFIDENCE, Confidence.INFERRED])


# ---------------------------------------------------------------------------
# Well-conditioned vs collinear: THE key empirical test
# ---------------------------------------------------------------------------


def test_anisotropy_dramatically_higher_for_collinear_single_pass():
    """Same point set, two different camera configurations:

    - well-conditioned: cameras spread around a hemisphere -> wide
      triangulation angles -> small, near-isotropic covariance.
    - collinear single-pass: cameras nadir along one straight line at
      altitude far exceeding the baseline -> narrow triangulation angles
      -> covariance is dramatically elongated along the (near-vertical)
      viewing direction.

    This is the empirical claim the project's novelty rests on --
    anisotropy_ratio must be *far* higher in the collinear case.
    """
    rng = np.random.default_rng(3)
    pts = np.stack(
        [rng.uniform(-3.0, 3.0, 30), rng.uniform(-3.0, 3.0, 30), rng.uniform(-0.5, 0.5, 30)], axis=1
    )

    well_poses = _well_conditioned_poses(n_cams=12, radius=10.0, altitude=8.0)
    result_well, _obs_cam_w, _obs_pt_w = _run_ba(well_poses, pts)
    assert result_well.converged
    assert result_well.rmse_after_px < 1e-6
    cov_well = point_covariances(result_well)
    ratio_well = anisotropy_ratio(cov_well)

    collinear_poses = _collinear_nadir_poses(n_cams=12, span=16.0, altitude=150.0)
    result_collinear, _obs_cam_c, _obs_pt_c = _run_ba(collinear_poses, pts)
    assert result_collinear.converged
    assert result_collinear.rmse_after_px < 1e-6
    cov_collinear = point_covariances(result_collinear)
    ratio_collinear = anisotropy_ratio(cov_collinear)

    median_well = float(np.median(ratio_well))
    median_collinear = float(np.median(ratio_collinear))

    # Well-conditioned: close to isotropic.
    assert median_well < 5.0
    # Collinear: dramatically elongated.
    assert median_collinear > 30.0
    # And the core comparative claim: an order of magnitude difference.
    assert median_collinear > median_well * 10.0


def test_worst_direction_is_near_vertical_for_nadir_collinear_flight():
    """For a nadir single-strip flight, the classic photogrammetric "poor
    vertical accuracy" failure mode should show up directly as the
    worst-constrained direction being close to the world vertical (Z) axis.
    """
    rng = np.random.default_rng(3)
    pts = np.stack(
        [rng.uniform(-3.0, 3.0, 20), rng.uniform(-3.0, 3.0, 20), rng.uniform(-0.5, 0.5, 20)], axis=1
    )
    poses = _collinear_nadir_poses(n_cams=12, span=16.0, altitude=150.0)
    result, _, _ = _run_ba(poses, pts)
    cov = point_covariances(result)
    worst = worst_constrained_direction(cov)  # (N, 3)
    # abs() because eigenvectors have arbitrary sign.
    vertical_alignment = np.abs(worst[:, 2])
    assert np.median(vertical_alignment) > 0.9


# ---------------------------------------------------------------------------
# Covariance scales correctly with observation noise
# ---------------------------------------------------------------------------


def test_covariance_scales_quadratically_with_pixel_sigma():
    rng = np.random.default_rng(5)
    pts = np.stack(
        [rng.uniform(-3.0, 3.0, 20), rng.uniform(-3.0, 3.0, 20), rng.uniform(-0.5, 0.5, 20)], axis=1
    )
    poses = _collinear_nadir_poses(n_cams=10, span=16.0, altitude=20.0)

    result_small, _, _ = _run_ba(poses, pts, pixel_sigma=0.5, robust_loss=None)
    result_large, _, _ = _run_ba(poses, pts, pixel_sigma=1.0, robust_loss=None)

    trace_small = np.trace(point_covariances(result_small), axis1=1, axis2=2).mean()
    trace_large = np.trace(point_covariances(result_large), axis1=1, axis2=2).mean()

    ratio = trace_large / trace_small
    # Doubling sigma should roughly quadruple covariance (trace of a 3x3
    # covariance scales the same way each individual variance does).
    assert ratio == pytest.approx(4.0, rel=0.05)


# ---------------------------------------------------------------------------
# camera_covariances
# ---------------------------------------------------------------------------


def test_camera_covariances_zero_for_fixed_cameras():
    rng = np.random.default_rng(6)
    pts = np.stack(
        [rng.uniform(-3.0, 3.0, 25), rng.uniform(-3.0, 3.0, 25), rng.uniform(-0.5, 0.5, 25)], axis=1
    )
    poses = _well_conditioned_poses(n_cams=10, radius=10.0, altitude=8.0)
    intrinsics = [_INTRINSICS] * len(poses)
    obs_cam, obs_pt, obs_uv = _project_all(poses, intrinsics, pts)
    problem = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
    )
    result = bundle_adjust(problem, BAConfig(max_iterations=100))
    cam_cov = camera_covariances(result)
    assert cam_cov.shape[0] == len(poses)
    # Cameras 0, 1 are the default gauge anchors -> exactly zero covariance.
    np.testing.assert_array_equal(cam_cov[0], np.zeros_like(cam_cov[0]))
    np.testing.assert_array_equal(cam_cov[1], np.zeros_like(cam_cov[1]))
    # A free camera should have nonzero (finite, positive semi-definite) covariance.
    assert np.all(np.linalg.eigvalsh(cam_cov[5]) >= -1e-12)
    assert np.trace(cam_cov[5]) > 0.0


# ---------------------------------------------------------------------------
# observability_field / required_viewing_direction / plan_corrective_flight
# ---------------------------------------------------------------------------


def _collinear_field(rng, n_cams=12, n_pts=30, altitude=150.0, span=16.0):
    pts = np.stack(
        [rng.uniform(-3.0, 3.0, n_pts), rng.uniform(-3.0, 3.0, n_pts), rng.uniform(-0.5, 0.5, n_pts)], axis=1
    )
    poses = _collinear_nadir_poses(n_cams=n_cams, span=span, altitude=altitude)
    result, obs_cam, obs_pt = _run_ba(poses, pts)
    cov = point_covariances(result)
    field = observability_field(result.points, result.poses, result.intrinsics, cov, obs_cam, obs_pt)
    return field, result


def test_observability_field_basic_shapes_and_ranges():
    rng = np.random.default_rng(7)
    field, _result = _collinear_field(rng)
    n = field.points.shape[0]
    assert field.covariances.shape == (n, 3, 3)
    assert field.semi_axes.shape == (n, 3)
    assert field.worst_direction.shape == (n, 3)
    assert field.required_direction.shape == (n, 3)
    assert np.all(field.n_observing_cameras > 0)
    # Every required/worst direction should be (numerically) a unit vector.
    np.testing.assert_allclose(np.linalg.norm(field.required_direction, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(field.worst_direction, axis=1), 1.0, atol=1e-6)


def test_required_viewing_direction_perpendicular_to_view_ray_for_known_axis():
    """A covariance elongated purely along world Z, observed by cameras
    directly overhead (mean viewing ray ~ -Z / nadir): the required
    direction should come out perpendicular to that nadir ray (i.e.
    horizontal), which is exactly "perpendicular to the current mean
    viewing ray, within the plane containing the worst-constrained axis"
    collapsing to any horizontal direction when the worst axis is the
    boresight itself (see observability.py's module docstring).
    """
    cov = np.diag([0.0001, 0.0001, 1.0])  # elongated along Z only
    point = np.array([0.0, 0.0, 0.0])
    poses = [
        Pose(R=np.eye(3), t=np.array([0.0, 0.0, 20.0])),
        Pose(R=np.eye(3), t=np.array([1.0, 0.0, 20.0])),
        Pose(R=np.eye(3), t=np.array([2.0, 0.0, 20.0])),
    ]
    direction = required_viewing_direction(point, cov, poses)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-9)
    # Perpendicular to nadir viewing ray [0, 0, -1] -> near-zero Z component.
    assert abs(direction[2]) < 1e-6


def test_required_viewing_direction_non_degenerate_case_lies_in_plane():
    """When the worst axis is *not* aligned with the mean viewing ray, the
    required direction should be the projection of that axis onto the
    plane orthogonal to the ray -- i.e. genuinely "in the plane containing
    the worst-constrained axis," not just an arbitrary fallback direction.
    """
    # Worst axis mostly along X, with a bit of Z -- not aligned with a
    # straight-down viewing ray.
    w = np.array([0.9, 0.0, 0.1])
    w = w / np.linalg.norm(w)
    # Build a covariance whose largest eigenvector is exactly `w`.
    other1 = np.array([0.0, 1.0, 0.0])
    other2 = np.cross(w, other1)
    other2 /= np.linalg.norm(other2)
    other1 = np.cross(other2, w)
    rot = np.stack([w, other1, other2], axis=1)
    cov = rot @ np.diag([1.0, 0.01, 0.01]) @ rot.T

    point = np.array([0.0, 0.0, 0.0])
    poses = [Pose(R=np.eye(3), t=np.array([0.0, 0.0, 20.0]))]  # mean ray = [0,0,-1] (nadir)
    direction = required_viewing_direction(point, cov, poses)

    v_mean = np.array([0.0, 0.0, -1.0])
    assert abs(np.dot(direction, v_mean)) < 1e-6  # perpendicular to viewing ray
    # Should align with w's component perpendicular to v_mean (here, w
    # itself is already perpendicular to v_mean since w has no Z... but we
    # gave it a small Z component, so check the projected version).
    w_perp = w - np.dot(w, v_mean) * v_mean
    w_perp /= np.linalg.norm(w_perp)
    assert abs(np.dot(direction, w_perp)) > 0.99


def test_plan_corrective_flight_heading_differs_from_original():
    """A parallel repeat pass would resolve nothing (zero new triangulation
    angle) -- the planned heading must differ meaningfully from the
    original flight heading.
    """
    rng = np.random.default_rng(3)
    field, result = _collinear_field(rng, n_cams=12, n_pts=30, altitude=150.0, span=16.0)

    plan = plan_corrective_flight(field, result.poses, PlanConfig())

    # Original flight ran along world X -> heading 90 (East) or 270 (West).
    original_heading_candidates = [90.0, 270.0]
    diffs = [min(abs(plan.heading_deg - h), 360 - abs(plan.heading_deg - h)) for h in original_heading_candidates]
    assert min(diffs) > 30.0  # meaningfully different from a parallel repeat

    assert -90.0 <= plan.gimbal_pitch_deg <= -PlanConfig().min_obliquity_deg + 1e-6
    assert plan.altitude_agl_m > 0
    assert 0.0 <= plan.predicted_resolved_fraction <= 1.0
    assert "predicted" in plan.recommendation.lower() or "%" in plan.recommendation
    assert len(plan.cluster_point_indices) > 0
    assert plan.start_enu.shape == (3,)
    assert plan.end_enu.shape == (3,)


def test_uncertainty_heatmap_shape_and_values():
    rng = np.random.default_rng(3)
    field, _ = _collinear_field(rng, n_cams=10, n_pts=25, altitude=100.0, span=16.0)
    heatmap = uncertainty_heatmap(field, resolution_m=1.0)
    assert heatmap.raster.ndim == 2
    assert heatmap.resolution_m == 1.0
    finite_vals = heatmap.raster[~np.isnan(heatmap.raster)]
    assert finite_vals.size > 0
    assert np.all(finite_vals >= 0.0)


def test_uncertainty_heatmap_rejects_bad_resolution():
    rng = np.random.default_rng(3)
    field, _ = _collinear_field(rng, n_cams=8, n_pts=15, altitude=80.0, span=12.0)
    with pytest.raises(ValueError):
        uncertainty_heatmap(field, resolution_m=0.0)
