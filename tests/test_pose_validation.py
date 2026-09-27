from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drishti3d.geometry.bundle import _reprojection_loss
from drishti3d.geometry.pose_validation import (
    ReconstructionRejected,
    validate_cameras,
    validate_sparse_depth,
)
from drishti3d.ingest.telemetry import resample_telemetry
from drishti3d.types import CameraIntrinsics, GeoPoint, Pose, TelemetrySample


def scene():
    poses = [Pose(R=np.diag([1., -1., -1.]), t=np.array([i * 10., 0., 100.])) for i in range(5)]
    intr = [CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480) for _ in poses]
    return poses, intr


def test_accepts_small_physical_correction():
    poses, intr = scene()
    updated = [Pose(R=p.R, t=p.t + [1, 0, 0]) for p in poses]
    assert validate_cameras(updated, poses, intr, intr)["median_position_shift_m"] == 1


def test_saved_run_magnitude_is_rejected_despite_low_reprojection():
    poses, intr = scene()
    rotation = Rotation.from_euler("z", 24.08, degrees=True).as_matrix()
    updated = [Pose(R=rotation @ p.R, t=p.t + [45.81, 0, 0]) for p in poses]
    with pytest.raises(ReconstructionRejected, match="45.810"):
        validate_cameras(updated, poses, intr, intr)


def test_single_outlier_not_hidden_by_median():
    poses, intr = scene()
    updated = list(poses)
    updated[-1] = Pose(R=poses[-1].R, t=poses[-1].t + [31, 0, 0])
    with pytest.raises(ReconstructionRejected, match="max_position"):
        validate_cameras(updated, poses, intr, intr)


def test_baseline_scale_collapse_rejected_with_small_absolute_motion():
    poses, intr = scene()
    updated = [Pose(R=p.R, t=np.array([20 + 0.5 * (p.t[0] - 20), 0, 100])) for p in poses]
    with pytest.raises(ReconstructionRejected, match="baseline scale"):
        validate_cameras(updated, poses, intr, intr)


@pytest.mark.parametrize("change", [{"fx": -1}, {"fx": float("nan")}, {"cx": 400}, {"fy": 700}])
def test_rejects_invalid_lens(change):
    from dataclasses import replace
    poses, intr = scene()
    changed = [replace(k, **change) for k in intr]
    with pytest.raises(ReconstructionRejected):
        validate_cameras(poses, poses, changed, intr)


def test_positive_depth_required():
    poses, _ = scene()
    result = SimpleNamespace(poses=poses, points=np.array([[0., 0., 120.]]), rmse_after_px=0.)
    problem = SimpleNamespace(obs_camera_idx=np.array([0]), obs_point_idx=np.array([0]))
    with pytest.raises(ReconstructionRejected, match="non-positive depth"):
        validate_sparse_depth(result, problem)


@pytest.mark.parametrize("loss", ["huber", "soft_l1", "cauchy", "arctan"])
def test_robust_loss_cannot_discard_gps_or_gravity(loss):
    rho = _reprojection_loss(loss, 2)(np.array([100., 100., 100., 100.]))
    assert np.all(rho[0, :2] < 100)
    np.testing.assert_array_equal(rho[:, 2:], [[100, 100], [1, 1], [0, 0]])


@pytest.mark.parametrize("headings", [(359., 1.), (179., -179.)])
def test_heading_interpolation_takes_short_arc(headings):
    samples = [TelemetrySample(timestamp=i, gimbal_yaw=h) for i, h in enumerate(headings)]
    yaw = resample_telemetry(samples, [0.5])[0].gimbal_yaw
    expected = 0 if headings[0] == 359 else 180
    assert abs((yaw - expected + 180) % 360 - 180) < 1e-8


def test_compass_east_maps_to_enu_east():
    from drishti3d.pipeline.stages import _poses_from_telemetry
    kf = SimpleNamespace(telemetry=TelemetrySample(timestamp=0, geo=GeoPoint(12, 77, 100),
                                                  gimbal_yaw=90, gimbal_pitch=0, gimbal_roll=0))
    pose = _poses_from_telemetry([kf])[0]
    np.testing.assert_allclose(pose.R[:, 2], [1, 0, 0], atol=1e-8)


def test_unverified_clock_blocks_geometry_before_backbone():
    from drishti3d.pipeline.stages import GeometryStage
    state = SimpleNamespace(telemetry_stats={"format": "csv", "offset_source": "assumed_zero"})
    with pytest.raises(ReconstructionRejected, match="synchronization"):
        GeometryStage().run(state, None, None)


def test_pose_rejection_stops_runner_before_dense_and_export(monkeypatch, tmp_path):
    from drishti3d.pipeline import runner
    calls = []

    class Stage:
        def __init__(self, name):
            self.name = name

        def run(self, *args, **kwargs):
            calls.append(self.name)
            raise ReconstructionRejected("unsafe camera solution")

    monkeypatch.setattr(runner, "_build_stages", lambda: [Stage(n) for n in ("pose_prior", "geometry", "export")])
    result = runner.run_pipeline(tmp_path / "unused.mp4")
    assert calls == ["pose_prior"]
    assert [s.status for s in result.stage_results] == ["failed", "skipped", "skipped"]


@pytest.mark.parametrize("bad,provenance", [(False, "default_guess"), (True, "default_guess"), (False, "exif")])
@pytest.mark.parametrize("second_pass", [False, True])
def test_bundle_commits_only_validated_consistent_calibration(monkeypatch, bad, provenance, second_pass):
    from dataclasses import replace

    from drishti3d.config import Config
    from drishti3d.geometry import bundle
    from drishti3d.pipeline import stages

    _, native = scene()
    kfs = [SimpleNamespace(intrinsics=native[i], telemetry=TelemetrySample(
        timestamp=i, geo=GeoPoint(12 + i * 0.0001, 77, 100), gimbal_pitch=-90, gimbal_yaw=0)) for i in range(5)]
    poses = stages._poses_from_telemetry(kfs)
    small = [replace(k, fx=k.fx / 2, fy=k.fy / 2, cx=k.cx / 2, cy=k.cy / 2,
                     width=k.width // 2, height=k.height // 2) for k in native]
    problem = SimpleNamespace(cameras=poses, intrinsics=small, points=np.array([[0., 0., -50.]]),
                              obs_camera_idx=np.array([0]), obs_point_idx=np.array([0]), obs_uv=np.array([[0., 0.]]))
    state = SimpleNamespace(config=Config(), keyframes=kfs, poses=poses, intrinsics=native[0],
                            matching_problem=problem, intrinsics_provenance=provenance)
    monkeypatch.setattr(stages.MatchingStage, "run", lambda *a: ({}, "fixture tracks"))
    refined = [replace(k, fx=k.fx * 1.03, fy=k.fy * 1.03, cx=k.cx + 1, cy=k.cy + 1) for k in small]
    triangulated = []
    if second_pass:
        from drishti3d.geometry import triangulate

        state.matching_trackset = SimpleNamespace(tracks=[SimpleNamespace(observations=[(0, 0, [0, 0])]) for _ in range(20)])
        state.matching_intrinsics = small

        def retriangulate(tracks, cameras, intrinsics, **kwargs):
            expected = refined if provenance == "default_guess" else small
            assert intrinsics == expected
            triangulated.append(True)
            return np.tile(problem.points, (20, 1)), np.ones(20, dtype=bool), np.ones(20)

        monkeypatch.setattr(triangulate, "triangulate_tracks", retriangulate)
        monkeypatch.setattr(triangulate, "filter_by_reprojection", lambda pts, tracks, *a, **kw: (pts, tracks))
        monkeypatch.setattr(triangulate, "build_ba_problem", lambda tracks, cameras, intrinsics, pts: SimpleNamespace(
            cameras=cameras, intrinsics=intrinsics, points=pts, obs_camera_idx=np.zeros(20, dtype=int),
            obs_point_idx=np.arange(20), obs_uv=np.zeros((20, 2))))

    def solve(problem, config):
        assert config.refine_shared_focal == (provenance == "default_guess")
        assert not config.refine_intrinsics
        return SimpleNamespace(poses=[Pose(R=p.R, t=p.t + [45 if bad else 0, 0, 0]) for p in poses],
                               intrinsics=refined if config.refine_shared_focal else small,
                               points=problem.points, rmse_before_px=2., rmse_after_px=0.5, converged=True)

    monkeypatch.setattr(bundle, "bundle_adjust", solve)
    fitted = []
    monkeypatch.setattr(stages, "_densify_fit_points", lambda state, poses: fitted.append(state.matching_intrinsics))
    if bad:
        with pytest.raises(ReconstructionRejected):
            stages.BundleAdjustmentStage().run(state, None, None)
        assert state.poses is poses
        assert kfs[0].intrinsics is native[0]
        assert not fitted
        assert not triangulated
    else:
        stages.BundleAdjustmentStage().run(state, None, None)
        solved = refined if provenance == "default_guess" else small
        assert fitted == [solved]
        assert bool(triangulated) == second_pass
        for kf, k in zip(kfs, solved):
            np.testing.assert_allclose([kf.intrinsics.fx, kf.intrinsics.fy, kf.intrinsics.cx, kf.intrinsics.cy],
                                       2 * np.array([k.fx, k.fy, k.cx, k.cy]))


def test_rotation_change_is_measured_against_the_closer_attitude_source():
    """flight01: flow-yaw seeded camera 0 29 deg off a correct gimbal log.

    A solution that agrees with the log but not the seed is supported by a
    measurement; one that agrees with neither is not.
    """
    poses, intr = scene()
    turn = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    seeds = list(poses)
    seeds[0] = Pose(R=turn @ poses[0].R, t=poses[0].t)  # bad flow-yaw seed
    logged = [Pose(R=Rotation.from_euler("z", 3, degrees=True).as_matrix() @ p.R, t=p.t) for p in poses]
    with pytest.raises(ReconstructionRejected, match="max_rotation"):
        validate_cameras(poses, seeds, intr, intr)
    diag = validate_cameras(poses, seeds, intr, intr, attitude_reference=logged)
    assert diag["max_rotation_change_deg"] == pytest.approx(3.0)
    wrong_everywhere = list(poses)
    wrong_everywhere[1] = Pose(R=Rotation.from_euler("x", 40, degrees=True).as_matrix() @ poses[1].R, t=poses[1].t)
    with pytest.raises(ReconstructionRejected, match="max_rotation"):
        validate_cameras(wrong_everywhere, seeds, intr, intr, attitude_reference=logged)


def test_lens_model_must_not_fold_over_inside_the_image():
    from dataclasses import replace

    poses, intr = scene()
    wide = [replace(k, dist_coeffs=np.array([-0.2, 0.036, 0, 0, 0])) for k in intr]
    diag = validate_cameras(poses, poses, wide, intr)
    assert 0.05 < diag["lens_corner_distortion"] < 0.2
    folded = [replace(k, dist_coeffs=np.array([-0.9, 0.0, 0, 0, 0])) for k in intr]
    with pytest.raises(ReconstructionRejected, match="folds over"):
        validate_cameras(poses, poses, folded, intr)


def test_solved_lens_commits_undistorted_frames_tracks_and_pinhole_intrinsics(monkeypatch):
    """Pass 1 solves (k1, k2); only after BOTH passes validate do frames, tracks and intrinsics change."""
    from drishti3d.config import Config
    from drishti3d.geometry import bundle, triangulate
    from drishti3d.pipeline import stages

    _, native = scene()
    kfs = [SimpleNamespace(intrinsics=native[i], telemetry=TelemetrySample(
        timestamp=i, geo=GeoPoint(12 + i * 0.0001, 77, 100), gimbal_pitch=-90, gimbal_yaw=0)) for i in range(5)]
    poses = stages._poses_from_telemetry(kfs)
    raw_uv = np.array([600.0, 50.0])
    from drishti3d.geometry.tracks import Track, TrackSet

    tracks = [Track(observations=[(0, 0, raw_uv.copy()), (1, 0, raw_uv.copy())]) for _ in range(20)]
    trackset = TrackSet(tracks=tracks)
    problem = SimpleNamespace(cameras=poses, intrinsics=list(native), points=np.array([[0., 0., -50.]]),
                              obs_camera_idx=np.array([0]), obs_point_idx=np.array([0]), obs_uv=np.array([raw_uv]))
    calls = []

    class Video:
        def set_undistortion(self, K, dist):
            calls.append(("video", np.asarray(dist)[:2].tolist()))

    class Cache:
        def undistort(self, K, dist):
            calls.append(("cache", np.asarray(dist)[:2].tolist()))

    def run(bad_second_pass):
        calls.clear()
        state = SimpleNamespace(config=Config(), keyframes=[SimpleNamespace(**vars(k)) for k in kfs], poses=poses,
                                intrinsics=native[0], matching_problem=problem, intrinsics_provenance="measured_from_flow",
                                matching_trackset=trackset, matching_intrinsics=list(native),
                                _matching_cache={"trackset": trackset}, video=Video(), keyframe_cache=Cache())
        monkeypatch.setattr(stages.MatchingStage, "run", lambda *a: ({}, "fixture tracks"))
        solves = []

        def solve(problem, config):
            solves.append(config)
            lens = {"focal_factor": 1.0, "k1": -0.2, "k2": 0.03} if config.refine_distortion else None
            dist = np.array([-0.2, 0.03, 0, 0, 0]) if config.refine_distortion else None
            intr = [CameraIntrinsics(k.fx, k.fy, k.cx, k.cy, k.width, k.height, dist) for k in problem.intrinsics]
            moved = 45 if (bad_second_pass and len(solves) == 2) else 0
            return SimpleNamespace(poses=[Pose(R=p.R, t=p.t + [moved, 0, 0]) for p in poses], intrinsics=intr,
                                   points=problem.points, rmse_before_px=2., rmse_after_px=0.5, converged=True,
                                   lens=lens)

        seen_uv = []

        def retriangulate(tracks_, cameras, intrinsics, **kw):
            seen_uv.append(np.array(tracks_.tracks[0].observations[0][2]))
            assert all(k.dist_coeffs is None for k in intrinsics)
            return np.tile(problem.points, (20, 1)), np.ones(20, dtype=bool), np.ones(20)

        monkeypatch.setattr(bundle, "bundle_adjust", solve)
        monkeypatch.setattr(triangulate, "triangulate_tracks", retriangulate)
        monkeypatch.setattr(triangulate, "filter_by_reprojection", lambda pts, tr, *a, **kw: (pts, tr))
        monkeypatch.setattr(triangulate, "build_ba_problem", lambda tr, cameras, intrinsics, pts: SimpleNamespace(
            cameras=cameras, intrinsics=intrinsics, points=pts, obs_camera_idx=np.array([0, 1] * 10),
            obs_point_idx=np.repeat(np.arange(20), 1)[:20], obs_uv=np.zeros((20, 2))))
        monkeypatch.setattr(stages, "_densify_fit_points", lambda state, poses: None)
        return state, solves, seen_uv

    state, solves, seen_uv = run(bad_second_pass=True)
    with pytest.raises(ReconstructionRejected):
        stages.BundleAdjustmentStage().run(state, None, None)
    assert solves[0].refine_distortion and not solves[1].refine_distortion
    # pass 2 triangulated undistorted copies; the stored tracks and frames are untouched
    assert not np.allclose(seen_uv[0], raw_uv)
    np.testing.assert_array_equal(tracks[0].observations[0][2], raw_uv)
    assert calls == []
    assert getattr(state, "lens_distortion", None) is None

    state, solves, seen_uv = run(bad_second_pass=False)
    stages.BundleAdjustmentStage().run(state, None, None)
    assert sorted(c[0] for c in calls) == ["cache", "video"]
    assert state.lens_distortion["dist_coeffs"][:2] == pytest.approx([-0.2, 0.03])
    assert all(kf.intrinsics.dist_coeffs is None for kf in state.keyframes)
    assert state.intrinsics.dist_coeffs is None
    # every stored observation moved exactly once (tracks shared by both track lists)
    np.testing.assert_allclose(tracks[0].observations[0][2], seen_uv[0])
    np.testing.assert_allclose(tracks[5].observations[1][2], seen_uv[0])


@pytest.mark.parametrize("strategy", ["world_frame", "telemetry_rotation"])
def test_georeferencing_keeps_world_frame_rotation_on_a_straight_strip(strategy):
    """BA poses already sit in the GPS frame; a Umeyama re-fit on a straight
    strip's camera centres adds an arbitrary roll about the flight line
    (flight01's 20 s model came out rolled 14 deg)."""
    from drishti3d.config import Config
    from drishti3d.pipeline import stages
    from drishti3d.types import PointCloud

    rng = np.random.default_rng(3)
    lat0, lon0 = 41.7745, -0.7413
    enu = np.c_[np.arange(12) * 20.0, np.zeros(12), np.full(12, 100.0)] + np.c_[np.zeros(12), rng.normal(0, 1, 12), np.zeros(12)]
    geo = [GeoPoint(lat0 + np.degrees(n / 6378137.0), lon0 + np.degrees(e / (6378137.0 * np.cos(np.radians(lat0)))), 400 + u)
           for e, n, u in enu]
    kfs = [SimpleNamespace(telemetry=TelemetrySample(timestamp=i, geo=g, gimbal_pitch=-90, gimbal_yaw=90),
                           intrinsics=None, frame_index=i, timestamp=float(i)) for i, g in enumerate(geo)]
    # BA leaves cameras metres from their fixes; on a straight strip that
    # noise alone decides the re-fitted roll.
    poses = [Pose(R=p.R, t=p.t + rng.normal(0, 1.5, 3)) for p in stages._poses_from_telemetry(kfs)]
    ground = np.c_[rng.uniform(0, 220, 500), rng.uniform(-60, 60, 500), np.zeros(500)]
    state = SimpleNamespace(config=Config(), telemetry_path="log.csv", keyframes=kfs, poses=poses,
                            point_cloud=PointCloud(xyz=ground.copy()), geometry_merge_strategy=strategy)
    stages._apply_georeferencing(state)
    for before, after in zip(poses, state.poses, strict=True):
        np.testing.assert_allclose(after.R, before.R, atol=1e-9)
    tilt = np.polyfit(state.point_cloud.xyz[:, 1], state.point_cloud.xyz[:, 2], 1)[0]
    assert abs(np.degrees(np.arctan(tilt))) < 0.05


def _stage_fixture(n=10, obs_per_cam=None):
    """Minimal BundleAdjustmentStage state over a straight GPS strip, with a stub matching problem."""
    from drishti3d.config import Config
    from drishti3d.pipeline import stages

    _, native = scene()
    kfs = [SimpleNamespace(intrinsics=native[0], telemetry=TelemetrySample(
        timestamp=i, geo=GeoPoint(12 + i * 0.0002, 77, 100), gimbal_pitch=-90, gimbal_yaw=0)) for i in range(n)]
    poses = stages._poses_from_telemetry(kfs)
    obs_per_cam = obs_per_cam or [40] * n
    cam = np.concatenate([np.full(k, i) for i, k in enumerate(obs_per_cam)]).astype(int)
    pts = np.arange(len(cam)) // 2
    problem = SimpleNamespace(cameras=poses, intrinsics=[native[0]] * n, points=np.tile([0.0, 0.0, -50.0], (pts.max() + 1, 1)),
                              obs_camera_idx=cam, obs_point_idx=pts, obs_uv=np.zeros((len(cam), 2)))
    state = SimpleNamespace(config=Config(), keyframes=kfs, poses=poses, intrinsics=native[0],
                            matching_problem=problem, intrinsics_provenance="exif")
    return state, poses, problem


def test_cameras_the_images_cannot_pose_keep_their_telemetry_pose(monkeypatch):
    """flight01: turn-boundary keyframes with 1-14 observations, and one camera the
    images pulled 38 m from GPS, used to fail a 208-camera solution outright."""
    from drishti3d.geometry import bundle
    from drishti3d.pipeline import stages

    state, poses, problem = _stage_fixture(obs_per_cam=[40, 40, 3, 40, 40, 40, 40, 40, 40, 40])
    monkeypatch.setattr(stages.MatchingStage, "run", lambda *a: ({}, "fixture tracks"))
    monkeypatch.setattr(stages, "_densify_fit_points", lambda state, poses: None)
    seen = []

    def solve(p, config):
        observed = set(np.unique(p.obs_camera_idx).tolist())
        seen.append(observed)
        # camera 6 is pulled 40 m off GPS while its observations are in the problem
        moved = [Pose(R=q.R, t=q.t + ([40, 0, 0] if (i == 6 and 6 in observed) else [0.3, 0, 0]))
                 for i, q in enumerate(p.cameras)]
        return SimpleNamespace(poses=moved, intrinsics=list(p.intrinsics), points=p.points,
                               rmse_before_px=2., rmse_after_px=0.5, converged=True, lens=None)

    monkeypatch.setattr(bundle, "bundle_adjust", solve)
    artifacts, message = stages.BundleAdjustmentStage().run(state, None, None)
    assert 2 not in seen[0]  # under-observed: dropped before the first solve
    assert 6 in seen[0] and 6 not in seen[1]  # outlier: dropped, then re-solved
    assert state.unrefined_keyframes == [2, 6]
    assert artifacts["unrefined_cameras"] == [2, 6]
    assert np.linalg.norm(state.poses[6].t - poses[6].t) < 1.0
    assert state.camera_validation["max_position_shift_m"] < 1.0


def test_too_many_unposed_cameras_rejects_the_solution(monkeypatch):
    from drishti3d.geometry import bundle
    from drishti3d.pipeline import stages

    state, poses, problem = _stage_fixture()
    monkeypatch.setattr(stages.MatchingStage, "run", lambda *a: ({}, "fixture tracks"))
    monkeypatch.setattr(bundle, "bundle_adjust", lambda p, config: SimpleNamespace(
        poses=[Pose(R=q.R, t=q.t + [0, 45 if i % 2 else 0, 0]) for i, q in enumerate(p.cameras)],
        intrinsics=list(p.intrinsics), points=p.points, rmse_before_px=2., rmse_after_px=0.5, converged=True, lens=None))
    with pytest.raises(ReconstructionRejected, match="could not be posed"):
        stages.BundleAdjustmentStage().run(state, None, None)
    assert state.poses is poses
    assert not getattr(state, "unrefined_keyframes", None)
