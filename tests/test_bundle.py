"""Tests for drishti3d.geometry.bundle: sparse bundle adjustment.

All synthetic, no external data, no torch/GPU. Every scene here is a
small nadir-camera "drone survey": a straight (or near-straight) line of
cameras flying over a patch of ground points, matching the single-pass
geometry the whole project targets.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drishti3d.geometry.bundle import (
    GCP,
    BAConfig,
    BAProblem,
    CameraPrior,
    bundle_adjust,
    windowed_bundle_adjust,
)
from drishti3d.types import CameraIntrinsics, Pose

# ---------------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------------

_NADIR_R = np.diag([1.0, -1.0, -1.0])
_INTRINSICS = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def _nadir_flight(n_cams: int, length_m: float, altitude_m: float) -> list[Pose]:
    centres = np.stack(
        [np.linspace(0.0, length_m, n_cams), np.zeros(n_cams), np.full(n_cams, altitude_m)], axis=1
    )
    return [Pose(R=_NADIR_R.copy(), t=c) for c in centres]


def _ground_points(rng: np.random.Generator, n_pts: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> np.ndarray:
    return np.stack(
        [
            rng.uniform(*x_range, n_pts),
            rng.uniform(*y_range, n_pts),
            np.zeros(n_pts),
        ],
        axis=1,
    )


def _project_all(
    poses: list[Pose], intrinsics: list[CameraIntrinsics], points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact pinhole projection of every point into every camera; returns only in-bounds, in-front observations."""
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


def _measure_tilt_deg(poses: list[Pose], reference_R: np.ndarray = _NADIR_R) -> np.ndarray:
    """Angle (degrees) of each pose's rotation away from ``reference_R``."""
    tilts = []
    for p in poses:
        rel = reference_R.T @ p.R
        rv = Rotation.from_matrix(rel).as_rotvec()
        tilts.append(np.degrees(np.linalg.norm(rv)))
    return np.array(tilts)


def _make_recovery_scene(
    rng: np.random.Generator, n_cams: int = 10, n_pts: int = 60
) -> tuple[list[Pose], list[CameraIntrinsics], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_poses = _nadir_flight(n_cams, length_m=40.0, altitude_m=20.0)
    pts = _ground_points(rng, n_pts, (-5.0, 45.0), (-6.0, 6.0))
    intrinsics = [_INTRINSICS] * n_cams
    obs_cam, obs_pt, obs_uv = _project_all(true_poses, intrinsics, pts)
    return true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv


# ---------------------------------------------------------------------------
# BA recovers a known solution
# ---------------------------------------------------------------------------


def test_bundle_adjust_recovers_known_solution():
    """Perturb a known-exact scene, run BA, and recover ground truth to tight tolerance.

    The two gauge-anchor cameras (0, 1) are seeded at their *true* values
    (anchors are assumed trusted -- see ``bundle._default_gauge_fix``'s
    docstring), everything else gets real perturbation noise.
    """
    rng = np.random.default_rng(0)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=10, n_pts=60)
    assert len(obs_cam) > 200  # sanity: the synthetic scene actually has plenty of observations

    noisy_poses = []
    for i, pose in enumerate(true_poses):
        if i < 2:
            noisy_poses.append(Pose(R=pose.R.copy(), t=pose.t.copy()))
            continue
        dt = rng.normal(scale=0.5, size=3)
        drot = rng.normal(scale=0.02, size=3)
        Rn = Rotation.from_rotvec(drot).as_matrix() @ pose.R
        noisy_poses.append(Pose(R=Rn, t=pose.t + dt))
    noisy_pts = pts + rng.normal(scale=0.3, size=pts.shape)

    problem = BAProblem(
        cameras=noisy_poses, intrinsics=intrinsics, points=noisy_pts.copy(),
        obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
    )
    result = bundle_adjust(problem, BAConfig(max_iterations=200, robust_loss=None))

    assert result.converged
    assert result.rmse_before_px > 1.0  # perturbation actually broke reprojection
    # Orders of magnitude improvement, and tight in absolute terms.
    assert result.rmse_after_px < result.rmse_before_px / 1000.0
    assert result.rmse_after_px < 1e-4

    point_err = np.linalg.norm(result.points - pts, axis=1)
    assert point_err.max() < 1e-5

    cam_err = np.linalg.norm(np.array([p.t for p in result.poses]) - np.array([p.t for p in true_poses]), axis=1)
    assert cam_err.max() < 1e-5

    for est, true in zip(result.poses, true_poses, strict=True):
        rel = true.R.T @ est.R
        angle_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec()))
        assert angle_deg < 1e-3


def test_bundle_adjust_reports_jacobian_and_layout_for_covariance():
    rng = np.random.default_rng(1)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=6, n_pts=20)
    problem = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
    )
    result = bundle_adjust(problem, BAConfig(max_iterations=50))
    assert result.jacobian is not None
    assert result.jacobian.shape[1] == result.param_layout.n_params
    assert result.param_layout.n_points == 20
    assert result.param_layout.points_base_col == result.jacobian.shape[1] - 3 * 20


# ---------------------------------------------------------------------------
# max_nfev: the fix for bundle adjustment's runaway-iteration cost on real,
# noisy data (see bundle.py's module docstring / BAConfig.max_nfev
# docstring for the full profiling story: the old ``max_iterations *
# n_params`` formula let a real 8-keyframe BA problem run 41,000+ residual
# evaluations chasing a sub-1%, practically-invisible cost improvement).
# ---------------------------------------------------------------------------


def test_default_max_nfev_does_not_scale_with_param_count():
    """A large, never-exactly-zero-residual problem (like real triangulated
    tracks, unlike this file's other noiseless-recovery scenes) must not be
    allowed to run for a parameter-count-scaled number of evaluations --
    the old formula would have permitted ~94,800 here (100 * 948 params).
    """
    rng = np.random.default_rng(9)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=10, n_pts=300)
    noisy_pts = pts + rng.normal(scale=0.4, size=pts.shape)
    noisy_uv = obs_uv + rng.normal(scale=0.4, size=obs_uv.shape)

    problem = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses], intrinsics=intrinsics,
        points=noisy_pts, obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=noisy_uv,
    )
    config = BAConfig(max_iterations=100)  # max_nfev left at its default
    result = bundle_adjust(problem, config)

    assert result.param_layout.n_params > 900
    # The default is max_iterations * 30, independent of param count; allow
    # scipy a small overshoot margin (it checks the budget between, not
    # within, an evaluation batch).
    assert result.n_iterations <= 100 * 30 + 5


def test_max_nfev_explicit_override_respected():
    rng = np.random.default_rng(10)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=8, n_pts=30)
    problem = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
    )
    result = bundle_adjust(problem, BAConfig(max_nfev=5))
    assert result.n_iterations <= 8


# ---------------------------------------------------------------------------
# GPS prior pulls a biased solution back toward truth
# ---------------------------------------------------------------------------


def test_gps_prior_corrects_biased_solution():
    """A uniformly-biased initial guess (as if GPS-derived positions had a systematic offset
    in the *reconstruction*, e.g. from a bad prior window) gets pulled back toward the
    GPS-prior positions, which here are exact -- reprojection alone cannot tell a
    uniformly-translated block from the truth (translation is part of the gauge freedom),
    so the GPS prior is what actually fixes it.
    """
    rng = np.random.default_rng(2)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=8, n_pts=40)

    bias = np.array([3.0, 2.0, -1.5])  # a deliberate, large, uniform translation bias
    biased_poses = [Pose(R=p.R.copy(), t=p.t.copy() + bias) for p in true_poses]
    biased_pts = pts + bias

    gps_priors = [
        CameraPrior(camera_idx=i, gps_position=true_poses[i].t.copy(), gps_sigma_m=0.5) for i in range(8)
    ]

    # No camera is gauge-fixed here on purpose: translation gauge comes
    # entirely from the GPS priors (applied to *every* camera, so the
    # prior can correct the bias everywhere, including what would
    # otherwise be an anchor).
    problem = BAProblem(
        cameras=biased_poses, intrinsics=intrinsics, points=biased_pts.copy(),
        obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
        camera_priors=gps_priors, fixed_camera_indices=set(),
    )
    result = bundle_adjust(problem, BAConfig(max_iterations=200, robust_loss=None, fixed_gauge=False))

    cam_err_after = np.linalg.norm(np.array([p.t for p in result.poses]) - np.array([p.t for p in true_poses]), axis=1)
    initial_err = np.linalg.norm(bias)
    assert cam_err_after.max() < initial_err / 10.0


# ---------------------------------------------------------------------------
# Gravity prior fixes the tilt (the test that justifies the whole design)
# ---------------------------------------------------------------------------


def test_gravity_prior_fixes_tilt_in_collinear_flight():
    """Near-collinear flight, gauge-anchored to a *tilted* reference (as
    ``windowed_bundle_adjust`` anchors to a previous window's -- possibly
    imperfect -- result): without a gravity prior, reprojection + GPS alone
    give essentially zero gradient pressure to correct the anchor's tilt
    (weakly observable in a near-collinear strip, per this project's core
    claim) and the tilt survives unchanged; with a gravity prior pinning
    roll/pitch to the known-level truth, every free camera's tilt is pulled
    back down. This is the test that justifies building the gravity prior
    at all -- see ``geometry.bundle``'s module docstring.
    """
    rng = np.random.default_rng(42)
    n_cams = 8
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=n_cams, n_pts=25)

    tilt_deg = 6.0
    delta_r = Rotation.from_rotvec(np.radians(tilt_deg) * np.array([1.0, 0.0, 0.0])).as_matrix()
    # Tilt every camera's *orientation* about the flight axis (world X);
    # positions stay at true values (as if GPS-derived and untouched).
    tilted_poses = [Pose(R=delta_r @ p.R.copy(), t=p.t.copy()) for p in true_poses]

    gps_priors = [
        CameraPrior(camera_idx=i, gps_position=true_poses[i].t.copy(), gps_sigma_m=0.1) for i in range(n_cams)
    ]
    cfg = BAConfig(max_iterations=100, robust_loss=None, fixed_gauge=False)

    # Camera 0 anchors gauge at its (tilted) value -- simulating an
    # inter-window anchor inherited from an upstream, already-tilted result.
    problem_no_gravity = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in tilted_poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
        camera_priors=gps_priors, fixed_camera_indices={0},
    )
    result_no_gravity = bundle_adjust(problem_no_gravity, cfg)
    tilt_no_gravity = _measure_tilt_deg(result_no_gravity.poses)[1:]  # exclude the fixed anchor

    tilt_priors = [
        CameraPrior(camera_idx=i, gimbal_pitch_deg=-90.0, gimbal_roll_deg=0.0, tilt_sigma_deg=1.0)
        for i in range(n_cams)
    ]
    problem_gravity = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in tilted_poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
        camera_priors=gps_priors + tilt_priors, fixed_camera_indices={0},
    )
    result_gravity = bundle_adjust(problem_gravity, cfg)
    tilt_gravity = _measure_tilt_deg(result_gravity.poses)[1:]

    # Without a gravity prior, reprojection + GPS alone barely touch the
    # (weakly-observable) tilt -- it survives close to its initial value.
    assert np.mean(tilt_no_gravity) > tilt_deg - 0.1

    # With a gravity prior, every free camera's tilt is measurably pulled
    # down relative to the no-gravity case.
    assert np.all(tilt_gravity < tilt_no_gravity)
    assert np.mean(tilt_gravity) < np.mean(tilt_no_gravity) - 0.05


# ---------------------------------------------------------------------------
# Gauge fixing: rank-deficient without it, full rank with it
# ---------------------------------------------------------------------------


def test_gauge_fixing_removes_rank_deficiency():
    """Without gauge fixing, ``J^T J`` is rank-deficient by exactly the 7-DOF
    similarity gauge (3 rotation + 3 translation + 1 scale); fixing two
    cameras (see ``bundle._default_gauge_fix``) removes exactly that
    deficiency, leaving a full-rank system -- required for covariance.py's
    Schur complement/pinv reasoning to be meaningful at all.
    """
    rng = np.random.default_rng(5)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=8, n_pts=25)

    problem_no_gauge = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
        fixed_camera_indices=set(),
    )
    result_no_gauge = bundle_adjust(problem_no_gauge, BAConfig(max_iterations=1, robust_loss=None, fixed_gauge=False))
    H_no_gauge = (result_no_gauge.jacobian.T @ result_no_gauge.jacobian).toarray()
    n_params_no_gauge = H_no_gauge.shape[0]
    rank_no_gauge = np.linalg.matrix_rank(H_no_gauge)
    assert rank_no_gauge == n_params_no_gauge - 7

    problem_gauge = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses], intrinsics=intrinsics,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
    )
    result_gauge = bundle_adjust(problem_gauge, BAConfig(max_iterations=1, robust_loss=None, fixed_gauge=True))
    H_gauge = (result_gauge.jacobian.T @ result_gauge.jacobian).toarray()
    rank_gauge = np.linalg.matrix_rank(H_gauge)
    assert rank_gauge == H_gauge.shape[0]


# ---------------------------------------------------------------------------
# GCP factors
# ---------------------------------------------------------------------------


def test_gcp_factor_pulls_point_to_surveyed_position():
    rng = np.random.default_rng(6)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=8, n_pts=30)

    noisy_pts = pts.copy()
    noisy_pts[5] += np.array([2.0, -1.5, 0.5])  # perturb one point far from its true (and GCP) position

    gcp = GCP(point_idx=5, xyz=pts[5].copy(), sigma_m=0.01)
    problem = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses], intrinsics=intrinsics,
        points=noisy_pts, obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv, gcps=[gcp],
    )
    result = bundle_adjust(problem, BAConfig(max_iterations=100, robust_loss=None))
    assert np.linalg.norm(result.points[5] - pts[5]) < 0.01


# ---------------------------------------------------------------------------
# Robust loss / refine_intrinsics options don't crash and behave sanely
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("robust_loss", ["huber", "cauchy", None])
def test_robust_loss_options_run_and_reduce_rmse(robust_loss):
    rng = np.random.default_rng(3)
    true_poses, intrinsics, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=8, n_pts=30)

    noisy_poses = [Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses]
    noisy_poses[3] = Pose(R=true_poses[3].R.copy(), t=true_poses[3].t + np.array([0.3, 0.2, -0.1]))
    noisy_pts = pts + rng.normal(scale=0.1, size=pts.shape)

    problem = BAProblem(
        cameras=noisy_poses, intrinsics=intrinsics, points=noisy_pts,
        obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
    )
    result = bundle_adjust(problem, BAConfig(max_iterations=100, robust_loss=robust_loss))
    assert result.rmse_after_px < result.rmse_before_px


def test_refine_intrinsics_recovers_focal_length():
    """Intrinsics are refined *per camera* (see ``BAConfig.refine_intrinsics``
    docstring), which is only observable when each camera's pose is
    otherwise well pinned down -- without that, a camera's own focal
    length and its depth along the viewing axis trade off against each
    other (the classic photogrammetric focal-length/depth ambiguity) and
    reprojection alone cannot tell them apart. GPS position priors (tight,
    as if from a well-surveyed rig) remove that escape valve here, exactly
    as they would need to in a real per-camera self-calibration setup.

    Gauge-anchor cameras (0, 1 -- see ``bundle._default_gauge_fix``) have
    their *entire* parameter block held constant, intrinsics included, so
    they are seeded with correct intrinsics here (consistent with "anchors
    are assumed trusted", the same convention used elsewhere in this file);
    only the free cameras' (2..7) wrong initial focal length is expected to
    be corrected by the optimizer.
    """
    rng = np.random.default_rng(4)
    true_poses, _, pts, obs_cam, obs_pt, obs_uv = _make_recovery_scene(rng, n_cams=8, n_pts=40)

    wrong_intr = [
        _INTRINSICS if i < 2 else CameraIntrinsics(fx=480.0, fy=480.0, cx=320.0, cy=240.0, width=640, height=480)
        for i in range(8)
    ]
    gps_priors = [
        CameraPrior(camera_idx=i, gps_position=true_poses[i].t.copy(), gps_sigma_m=0.02) for i in range(8)
    ]
    problem = BAProblem(
        cameras=[Pose(R=p.R.copy(), t=p.t.copy()) for p in true_poses], intrinsics=wrong_intr,
        points=pts.copy(), obs_camera_idx=obs_cam, obs_point_idx=obs_pt, obs_uv=obs_uv,
        camera_priors=gps_priors,
    )
    result = bundle_adjust(problem, BAConfig(max_iterations=150, robust_loss=None, refine_intrinsics=True))
    assert result.rmse_after_px < 1e-3
    for k in result.intrinsics[2:]:
        assert k.fx == pytest.approx(500.0, abs=0.5)
        assert k.fy == pytest.approx(500.0, abs=0.5)


# ---------------------------------------------------------------------------
# windowed_bundle_adjust
# ---------------------------------------------------------------------------


def test_windowed_bundle_adjust_anchors_prevent_drift():
    """Two overlapping windows over the same synthetic flight; anchoring the
    shared cameras to window 0's refined poses should keep window 1's
    result in the same frame (no independent gauge drift).
    """
    rng = np.random.default_rng(8)
    n_cams = 14
    true_poses = _nadir_flight(n_cams, length_m=50.0, altitude_m=25.0)
    pts = _ground_points(rng, 80, (-5.0, 55.0), (-8.0, 8.0))
    intrinsics = [_INTRINSICS] * n_cams
    obs_cam, obs_pt, obs_uv = _project_all(true_poses, intrinsics, pts)

    # Window 0: cameras 0..8, Window 1: cameras 6..13 (shared: 6,7,8).
    w0_cams = list(range(9))
    w1_cams = list(range(6, 14))
    shared_global = [6, 7, 8]

    def _subproblem(cam_list: list[int], exact_indices: set[int] = frozenset()) -> BAProblem:
        """``exact_indices`` (local indices into ``cam_list``) are seeded at their
        *true* value with no noise -- used for window 0's own gauge-anchor
        cameras (0, 1), consistent with this file's "anchors are assumed
        trusted" convention (see ``test_bundle_adjust_recovers_known_solution``):
        window 0 has no external anchor of its own, so if its *first*
        gauge reference were itself noisy, that error would lever-arm into
        a growing drift across the window (and then propagate into every
        downstream window) -- a real and separate phenomenon from the
        inter-window-anchoring this test is actually about.
        """
        local_of_global = {g: i for i, g in enumerate(cam_list)}
        mask = np.isin(obs_cam, cam_list)
        sub_obs_cam = np.array([local_of_global[c] for c in obs_cam[mask]])
        sub_obs_pt_global = obs_pt[mask]
        unique_pts = sorted(set(sub_obs_pt_global.tolist()))
        local_of_pt = {g: i for i, g in enumerate(unique_pts)}
        sub_obs_pt = np.array([local_of_pt[p] for p in sub_obs_pt_global])
        sub_uv = obs_uv[mask]

        rng_local = np.random.default_rng(hash(tuple(cam_list)) % (2**32))
        cams = []
        for local_i, g in enumerate(cam_list):
            if local_i in exact_indices:
                cams.append(Pose(R=true_poses[g].R.copy(), t=true_poses[g].t.copy()))
                continue
            dt = rng_local.normal(scale=0.3, size=3)
            drot = rng_local.normal(scale=0.01, size=3)
            Rn = Rotation.from_rotvec(drot).as_matrix() @ true_poses[g].R
            cams.append(Pose(R=Rn, t=true_poses[g].t + dt))
        sub_pts = pts[unique_pts] + rng_local.normal(scale=0.2, size=(len(unique_pts), 3))

        return BAProblem(
            cameras=cams, intrinsics=[_INTRINSICS] * len(cam_list), points=sub_pts,
            obs_camera_idx=sub_obs_cam, obs_point_idx=sub_obs_pt, obs_uv=sub_uv,
        ), unique_pts

    problem0, _ = _subproblem(w0_cams, exact_indices={0, 1})
    problem1, _ = _subproblem(w1_cams)

    shared_pairs = [[(w0_cams.index(g), w1_cams.index(g)) for g in shared_global]]

    results = windowed_bundle_adjust(
        [problem0, problem1], BAConfig(max_iterations=150, robust_loss=None), shared_camera_pairs=shared_pairs, n_anchor=3
    )
    assert len(results) == 2
    assert results[0].converged
    # Window 1's anchored cameras must exactly match window 0's refined poses.
    for prev_local, curr_local in shared_pairs[0]:
        np.testing.assert_allclose(results[1].poses[curr_local].t, results[0].poses[prev_local].t, atol=1e-8)
        np.testing.assert_allclose(results[1].poses[curr_local].R, results[0].poses[prev_local].R, atol=1e-8)

    # And window 1's *own* (non-anchored) cameras should still end up close
    # to true world positions -- the anchoring didn't just freeze garbage.
    for local_idx, g in enumerate(w1_cams):
        if local_idx in {li for _, li in shared_pairs[0]}:
            continue
        err = np.linalg.norm(results[1].poses[local_idx].t - true_poses[g].t)
        assert err < 0.5
