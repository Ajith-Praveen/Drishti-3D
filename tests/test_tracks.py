"""Tests for drishti3d.geometry.tracks (and, by necessity, triangulate.py/features.py
downstream of it -- see the end-to-end test at the bottom of this file).

Synthetic, deterministic, no external data, no torch/GPU -- consistent with the
rest of the test suite.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drishti3d.geometry.bundle import BAConfig, BAProblem, bundle_adjust
from drishti3d.geometry.features import (
    Features,
    Matches,
    detect_and_describe,
    geometric_verify,
    match_features,
)
from drishti3d.geometry.tracks import (
    Track,
    TrackSet,
    build_tracks,
    filter_tracks,
    select_pairs,
    track_statistics,
)
from drishti3d.geometry.triangulate import (
    build_ba_problem,
    filter_by_reprojection,
    triangulate_track,
    triangulate_tracks,
)
from drishti3d.types import (
    CameraIntrinsics,
    FrameMetrics,
    GeoPoint,
    Keyframe,
    Pose,
    TelemetrySample,
)

_NADIR_R = np.diag([1.0, -1.0, -1.0])
_INTR = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480)


def _kf(index: int, geo: GeoPoint | None = None) -> Keyframe:
    metrics = FrameMetrics(index=index, timestamp=float(index), blur_score=1.0, exposure_score=1.0, mean_luma=1.0, estimated_parallax=0.0)
    telemetry = TelemetrySample(timestamp=float(index), geo=geo) if geo is not None else None
    return Keyframe(frame_index=index, timestamp=float(index), metrics=metrics, telemetry=telemetry)


def _feat(coords: list[list[float]]) -> Features:
    coords_arr = np.asarray(coords, dtype=np.float64)
    return Features(keypoints=coords_arr, descriptors=None, scores=np.ones(len(coords_arr)), method="synthetic")


def _dummy_image(size: int = 8) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _match(qi: int, ti: int) -> Matches:
    return Matches(
        query_idx=np.array([qi]), train_idx=np.array([ti]), distances=np.array([0.1]), inlier_mask=np.array([True])
    )


# ---------------------------------------------------------------------------
# build_tracks: the union-find consistency bug
# ---------------------------------------------------------------------------


def _naive_union_groups(pair_list) -> dict:
    """A textbook union-find with NO consistency check -- used only to prove the fixture below
    actually triggers the classic bug (i.e. that a naive implementation gets it wrong)."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    for i, j, matches in pair_list:
        for qi, ti, keep in zip(matches.query_idx, matches.train_idx, matches.inlier_mask, strict=True):
            if keep:
                union((i, int(qi)), (j, int(ti)))

    groups: dict[tuple[int, int], set] = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return groups


def test_build_tracks_rejects_inconsistent_chain():
    """A bad match chain tries to fold two DIFFERENT frame-0 keypoints into one track; build_tracks must not allow it."""
    features_per_frame = [
        _feat([[10.0, 10.0], [50.0, 50.0]]),  # frame 0: keypoints 0 and 1
        _feat([[11.0, 11.0]]),  # frame 1: keypoint 0
        _feat([[12.0, 12.0]]),  # frame 2: keypoint 0
    ]
    images = [_dummy_image() for _ in features_per_frame]

    pair_list = [
        (0, 1, _match(0, 0)),  # frame0.kp0 <-> frame1.kp0
        (1, 2, _match(0, 0)),  # frame1.kp0 <-> frame2.kp0 (chains frame0.kp0 <-> frame2.kp0 transitively)
        (0, 2, _match(1, 0)),  # frame0.kp1 <-> frame2.kp0 -- CONFLICT: frame2.kp0 is already frame0.kp0's partner
    ]

    # Sanity check: prove this fixture really does corrupt a naive
    # (no-consistency-check) union-find, so the assertion below is testing
    # something real and not vacuous.
    naive_groups = _naive_union_groups(pair_list)
    corrupted = [g for g in naive_groups.values() if sum(1 for frame, _kp in g if frame == 0) > 1]
    assert corrupted, "fixture did not reproduce the classic union-find bug against a naive implementation"

    trackset = build_tracks(images, features_per_frame, pair_list)

    for track in trackset.tracks:
        frame_idxs = track.frame_indices()
        assert len(frame_idxs) == len(set(frame_idxs)), f"track has two observations in the same frame: {track.observations}"

    # The good part of the chain (frame0.kp0 <-> frame1.kp0 <-> frame2.kp0)
    # should survive as its own consistent track -- the bad edge is
    # excised, not the whole chain discarded.
    assert len(trackset.tracks) == 1
    survivor = trackset.tracks[0]
    assert survivor.frame_indices() == [0, 1, 2]


def test_build_tracks_requires_matching_image_and_feature_counts():
    with pytest.raises(ValueError):
        build_tracks([_dummy_image()], [_feat([[0.0, 0.0]]), _feat([[0.0, 0.0]])], [])


def test_build_tracks_ignores_unverified_matches():
    """A Matches with inlier_mask=None (never geometrically verified) contributes no edges."""
    features_per_frame = [_feat([[1.0, 1.0]]), _feat([[2.0, 2.0]])]
    images = [_dummy_image(), _dummy_image()]
    unverified = Matches(query_idx=np.array([0]), train_idx=np.array([0]), distances=np.array([0.1]))
    trackset = build_tracks(images, features_per_frame, [(0, 1, unverified)])
    assert len(trackset) == 0


# ---------------------------------------------------------------------------
# filter_tracks
# ---------------------------------------------------------------------------


def _track_of_length(n: int, frame_offset: int = 0) -> Track:
    return Track(observations=[(frame_offset + f, 0, np.array([1.0, 2.0])) for f in range(n)])


def test_filter_tracks_min_length():
    short = _track_of_length(2)
    long_ = _track_of_length(3)
    trackset = TrackSet(tracks=[short, long_])
    filtered = filter_tracks(trackset, min_length=3)
    assert len(filtered) == 1
    assert filtered.tracks[0] is long_


def test_filter_tracks_reprojection_requires_populated_field():
    track = _track_of_length(3)
    trackset = TrackSet(tracks=[track])
    with pytest.raises(ValueError):
        filter_tracks(trackset, min_length=1, max_reprojection_px=2.0)

    track.reprojection_error_px = 1.0
    assert len(filter_tracks(trackset, min_length=1, max_reprojection_px=2.0)) == 1

    track.reprojection_error_px = 10.0
    assert len(filter_tracks(trackset, min_length=1, max_reprojection_px=2.0)) == 0


# ---------------------------------------------------------------------------
# select_pairs
# ---------------------------------------------------------------------------


def test_select_pairs_sequential_is_linear_not_quadratic():
    n = 200
    window = 3
    keyframes = [_kf(i) for i in range(n)]
    pairs = select_pairs(keyframes, strategy="sequential", window=window)

    max_quadratic = n * (n - 1) // 2
    assert len(pairs) <= n * window
    assert len(pairs) < max_quadratic / 10
    for i, j in pairs:
        assert 0 < j - i <= window


def test_select_pairs_gps_loop_closure_adds_pairs_beyond_window():
    """A flight that heads out and returns near its own start should gain a loop-closure pair."""
    n = 20
    window = 2
    keyframes = []
    for i in range(n):
        offset = i if i < 10 else (19 - i)
        geo = GeoPoint(lat=10.0 + offset * 0.0001, lon=20.0, alt_msl=100.0)
        keyframes.append(_kf(i, geo=geo))

    pairs_plain = select_pairs(keyframes, strategy="sequential", window=window)
    pairs_loop = select_pairs(keyframes, strategy="sequential+loop", window=window, gps_radius_m=5.0)

    assert set(pairs_plain).issubset(set(pairs_loop))
    loop_only = [(i, j) for i, j in pairs_loop if j - i > window]
    assert loop_only, "expected at least one GPS loop-closure pair beyond the sequential window"
    assert (0, 19) in pairs_loop  # frame 0 and frame 19 sit at the same GPS position


def test_select_pairs_rejects_bad_strategy():
    with pytest.raises(ValueError):
        select_pairs([_kf(0), _kf(1)], strategy="all_pairs")


# ---------------------------------------------------------------------------
# track_statistics
# ---------------------------------------------------------------------------


def test_track_statistics_basic():
    tracks = [_track_of_length(3), _track_of_length(2)]
    trackset = TrackSet(tracks=tracks)
    stats = track_statistics(trackset, n_frames=5)

    assert stats["count"] == 2
    assert stats["mean_length"] == pytest.approx(2.5)
    assert stats["observations_per_frame"].get(0, 0) == 2
    assert stats["observations_per_frame"].get(3, 0) == 0
    assert stats["frac_frames_too_few_tracks"] > 0.0  # every frame here is far below the warning floor


# ---------------------------------------------------------------------------
# triangulate.py: DLT accuracy and degenerate-angle rejection
# ---------------------------------------------------------------------------


def _project(pose: Pose, intrinsics: CameraIntrinsics, point: np.ndarray) -> np.ndarray:
    xc = pose.R.T @ (point - pose.t)
    u = intrinsics.fx * xc[0] / xc[2] + intrinsics.cx
    v = intrinsics.fy * xc[1] / xc[2] + intrinsics.cy
    return np.array([u, v])


def test_triangulate_track_recovers_known_point_to_high_precision():
    true_point = np.array([3.0, -1.5, 0.0])
    poses = [
        Pose(R=_NADIR_R.copy(), t=np.array([0.0, 0.0, 20.0])),
        Pose(R=_NADIR_R.copy(), t=np.array([6.0, 0.0, 20.0])),
        Pose(R=_NADIR_R.copy(), t=np.array([-5.0, 3.0, 22.0])),
    ]
    intrinsics = [_INTR, _INTR, _INTR]
    observations = [(cam_idx, _project(pose, _INTR, true_point)) for cam_idx, pose in enumerate(poses)]

    point, angle = triangulate_track(observations, poses, intrinsics)

    # "Sub-millimetre" was the ask; exact synthetic projections plus exact
    # linear algebra recovers far tighter than that.
    assert np.allclose(point, true_point, atol=1e-6)
    assert angle > 1.5


def test_triangulate_tracks_rejects_narrow_baseline_accepts_wide_baseline():
    true_point = np.array([3.0, -1.5, 0.0])
    intrinsics = [_INTR, _INTR]

    # Near-collinear cameras (tiny baseline relative to depth) -> a tiny
    # triangulation angle; this is the single-pass degeneracy the module
    # docstring describes, exaggerated here for a clean test.
    poses_narrow = [
        Pose(R=_NADIR_R.copy(), t=np.array([0.0, 0.0, 500.0])),
        Pose(R=_NADIR_R.copy(), t=np.array([0.01, 0.0, 500.0])),
    ]
    # Same point, but a wide baseline relative to altitude -> a large angle.
    poses_wide = [
        Pose(R=_NADIR_R.copy(), t=np.array([0.0, 0.0, 20.0])),
        Pose(R=_NADIR_R.copy(), t=np.array([15.0, 0.0, 20.0])),
    ]

    track_narrow = TrackSet(tracks=[Track(observations=[
        (0, 0, _project(poses_narrow[0], _INTR, true_point)),
        (1, 0, _project(poses_narrow[1], _INTR, true_point)),
    ])])
    track_wide = TrackSet(tracks=[Track(observations=[
        (0, 0, _project(poses_wide[0], _INTR, true_point)),
        (1, 0, _project(poses_wide[1], _INTR, true_point)),
    ])])

    _points_n, valid_n, angles_n = triangulate_tracks(track_narrow, poses_narrow, intrinsics, min_angle_deg=1.5)
    assert not valid_n[0]
    assert angles_n[0] < 1.5

    points_w, valid_w, angles_w = triangulate_tracks(track_wide, poses_wide, intrinsics, min_angle_deg=1.5)
    assert valid_w[0]
    assert angles_w[0] >= 1.5
    assert np.allclose(points_w[0], true_point, atol=1e-3)


# ---------------------------------------------------------------------------
# filter_by_reprojection: max_points cap (the "cap points entering BA" lever)
# ---------------------------------------------------------------------------


def test_filter_by_reprojection_max_points_keeps_longest_lowest_error_tracks():
    """With more surviving tracks than ``max_points``, keep the longest
    (best-constrained) ones, breaking ties by lowest reprojection error --
    see that function's docstring for why this is the right selection
    criterion for capping bundle adjustment's point budget without hurting
    pose accuracy.
    """
    poses = [
        Pose(R=_NADIR_R.copy(), t=np.array([0.0, 0.0, 20.0])),
        Pose(R=_NADIR_R.copy(), t=np.array([6.0, 0.0, 20.0])),
        Pose(R=_NADIR_R.copy(), t=np.array([-5.0, 3.0, 20.0])),
        Pose(R=_NADIR_R.copy(), t=np.array([3.0, -4.0, 20.0])),
    ]
    intrinsics = [_INTR] * 4

    def _track(true_point, cams, noise=0.0):
        obs = []
        for c in cams:
            uv = _project(poses[c], _INTR, true_point)
            obs.append((c, 0, uv + np.array([noise, 0.0])))
        return Track(observations=obs), true_point

    # A long (4-view), exact track -- should always survive a cap.
    long_exact, p_long = _track(np.array([1.0, 0.5, 0.0]), [0, 1, 2, 3])
    # Two short (3-view) tracks: one exact, one with a small reprojection error.
    short_exact, p_short_exact = _track(np.array([2.0, -1.0, 0.0]), [0, 1, 2])
    short_noisy, p_short_noisy = _track(np.array([-1.0, 2.0, 0.0]), [0, 1, 2], noise=2.0)

    trackset = TrackSet(tracks=[long_exact, short_exact, short_noisy])
    points = np.stack([p_long, p_short_exact, p_short_noisy], axis=0)

    kept_points, kept_tracks = filter_by_reprojection(points, trackset, poses, intrinsics, max_px=50.0, max_points=2)

    kept_ids = [id(t) for t in kept_tracks.tracks]
    assert len(kept_tracks) == 2
    assert id(long_exact) in kept_ids  # longest track always kept
    assert id(short_exact) in kept_ids  # tiebreak: lower reprojection error than short_noisy
    assert id(short_noisy) not in kept_ids
    assert kept_points.shape == (2, 3)


def test_filter_by_reprojection_max_points_none_keeps_everything_passing_max_px():
    poses = [Pose(R=_NADIR_R.copy(), t=np.array([0.0, 0.0, 20.0])), Pose(R=_NADIR_R.copy(), t=np.array([6.0, 0.0, 20.0]))]
    intrinsics = [_INTR, _INTR]
    true_point = np.array([1.0, 0.5, 0.0])
    obs = [(0, 0, _project(poses[0], _INTR, true_point)), (1, 0, _project(poses[1], _INTR, true_point))]
    trackset = TrackSet(tracks=[Track(observations=obs)])
    points = np.array([true_point])

    _kept_points, kept_tracks = filter_by_reprojection(points, trackset, poses, intrinsics, max_px=5.0, max_points=None)
    assert len(kept_tracks) == 1


# ---------------------------------------------------------------------------
# End-to-end: real detect -> match -> verify -> tracks -> triangulate -> BAProblem -> bundle_adjust
#
# This is the integration test that proves the gap this workstream exists
# to close is actually closed: geometry.bundle.bundle_adjust could already
# run in isolation (tests/test_bundle.py exercises it with hand-fed
# BAProblem instances), but nothing produced a BAProblem from real 2D
# observations. Here, a synthetic textured flyover scene is rendered from
# known ground-truth poses, real SIFT features are detected and matched
# across frames, and the entire chain this workstream built --
# build_tracks -> triangulate_tracks -> build_ba_problem -- turns those
# real detections into a BAProblem that a real bundle_adjust call then
# refines, recovering deliberately-perturbed poses/points and driving
# reprojection RMSE down substantially.
# ---------------------------------------------------------------------------

_FLIGHT_ALT_M = 20.0
_FLIGHT_SPACING_M = 4.0
_BOX_MIN = np.array([4.0, -3.0, 0.0])
_BOX_MAX = np.array([9.0, 3.0, 5.0])
_HFOV_DEG = 70.0
_IMG_W, _IMG_H = 480, 360


def _ground_texture(seed: int = 7, tile: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 216, size=(tile, tile)).astype(np.uint8)
    return cv2.GaussianBlur(base, (3, 3), 0)


def _sample_texture(texture: np.ndarray, x: np.ndarray, y: np.ndarray, px_per_m: float = 30.0) -> np.ndarray:
    tile = texture.shape[0]
    xi = np.mod(np.floor(x * px_per_m).astype(np.int64), tile)
    yi = np.mod(np.floor(y * px_per_m).astype(np.int64), tile)
    return texture[yi, xi]


def _ray_box_hit(origin: np.ndarray, directions: np.ndarray, box_min: np.ndarray, box_max: np.ndarray):
    eps = 1e-12
    d = np.where(np.abs(directions) < eps, eps, directions)
    t1 = (box_min - origin) / d
    t2 = (box_max - origin) / d
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)
    t_enter = np.max(t_near, axis=-1)
    t_exit = np.min(t_far, axis=-1)
    hit = (t_exit >= t_enter) & (t_exit >= 1e-6) & (t_enter >= 1e-6)
    return t_enter, hit


def _render_view(pose: Pose, intrinsics: CameraIntrinsics, texture: np.ndarray, height: int, width: int) -> np.ndarray:
    """Ray-cast a textured ground plane + a box (same synthetic layout as ``geometry.backbone.NullBackbone``),
    but sample real grayscale texture at each hit point instead of returning depth/points -- i.e. an actual
    renderable image, so real feature detection has genuine multi-view-consistent texture to find and match."""
    k_inv = np.linalg.inv(intrinsics.K())
    us = np.arange(width, dtype=np.float64) + 0.5
    vs = np.arange(height, dtype=np.float64) + 0.5
    gu, gv = np.meshgrid(us, vs)
    pix = np.stack([gu, gv, np.ones_like(gu)], axis=-1)
    dir_cam = pix @ k_inv.T
    dir_world = dir_cam @ pose.R.T
    origin = pose.t

    dz = dir_world[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = np.where(np.abs(dz) > 1e-9, -origin[2] / dz, np.inf)
    ground_hit = np.isfinite(t_ground) & (t_ground > 1e-6)
    t_ground = np.where(ground_hit, t_ground, np.inf)

    t_box, box_hit = _ray_box_hit(origin, dir_world, _BOX_MIN, _BOX_MAX)
    t_box = np.where(box_hit, t_box, np.inf)

    use_box = box_hit & (t_box <= t_ground)
    t_final = np.where(use_box, t_box, t_ground)
    valid = np.isfinite(t_final)
    t_final = np.where(valid, t_final, 0.0)

    hit_world = origin.reshape(1, 1, 3) + t_final[..., None] * dir_world
    gray = _sample_texture(texture, hit_world[..., 0], hit_world[..., 1])
    gray = np.where(valid, gray, 20).astype(np.uint8)
    return gray


def _synthetic_flight(n_cams: int) -> tuple[list[Pose], CameraIntrinsics, list[np.ndarray]]:
    poses = [Pose(R=_NADIR_R.copy(), t=np.array([_FLIGHT_SPACING_M * i, 0.0, _FLIGHT_ALT_M])) for i in range(n_cams)]
    intrinsics = CameraIntrinsics.from_hfov(_HFOV_DEG, _IMG_W, _IMG_H)
    texture = _ground_texture()
    images = [_render_view(p, intrinsics, texture, _IMG_H, _IMG_W) for p in poses]
    return poses, intrinsics, images


def test_end_to_end_matching_triangulation_bundle_adjustment_chain():
    n_cams = 6
    poses_true, intrinsics, images = _synthetic_flight(n_cams)
    intrinsics_list = [intrinsics] * n_cams

    features_per_frame = [detect_and_describe(img, method="sift", max_features=2500) for img in images]
    assert all(len(f) > 100 for f in features_per_frame), "synthetic scene should be feature-rich"

    keyframes = [_kf(i) for i in range(n_cams)]
    pairs = select_pairs(keyframes, strategy="sequential", window=3)
    assert pairs  # sanity

    verified_pairs = []
    for i, j in pairs:
        matches = match_features(features_per_frame[i], features_per_frame[j], ratio=0.8, cross_check=True)
        if len(matches) < 8:
            continue
        verified = geometric_verify(features_per_frame[i], features_per_frame[j], matches, intrinsics=intrinsics, method="essential")
        if verified.inlier_ratio() > 0.5:
            verified_pairs.append((i, j, verified))
    assert verified_pairs, "expected at least some well-verified frame pairs"

    color_images = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images]
    trackset = build_tracks(color_images, features_per_frame, verified_pairs)
    trackset = filter_tracks(trackset, min_length=3)
    assert len(trackset) > 15, f"expected plenty of multi-view tracks from a textured scene, got {len(trackset)}"

    points3d, valid_mask, _angles = triangulate_tracks(trackset, poses_true, intrinsics_list, min_angle_deg=1.5)
    assert valid_mask.sum() > 10, "expected several tracks to triangulate with a healthy angle"

    kept_tracks = TrackSet(tracks=[t for t, keep in zip(trackset.tracks, valid_mask, strict=True) if keep])
    kept_points = points3d[valid_mask]

    problem = build_ba_problem(kept_tracks, poses_true, intrinsics_list, kept_points)
    assert problem.obs_uv.shape[0] == sum(len(t) for t in kept_tracks.tracks)
    assert problem.points.shape[0] == len(kept_tracks.tracks)

    # Perturb cameras (except the two gauge anchors) and points, then check
    # bundle_adjust recovers a low-reprojection-error solution against the
    # REAL detected 2D observations captured in `problem.obs_uv`.
    rng = np.random.default_rng(3)
    noisy_cameras = []
    for i, pose in enumerate(problem.cameras):
        if i < 2:
            noisy_cameras.append(Pose(R=pose.R.copy(), t=pose.t.copy()))
            continue
        dt = rng.normal(scale=0.6, size=3)
        drot = rng.normal(scale=0.03, size=3)
        r_noisy = Rotation.from_rotvec(drot).as_matrix() @ pose.R
        noisy_cameras.append(Pose(R=r_noisy, t=pose.t + dt))
    noisy_points = problem.points + rng.normal(scale=0.4, size=problem.points.shape)

    perturbed = BAProblem(
        cameras=noisy_cameras,
        intrinsics=problem.intrinsics,
        points=noisy_points,
        obs_camera_idx=problem.obs_camera_idx,
        obs_point_idx=problem.obs_point_idx,
        obs_uv=problem.obs_uv,
    )

    result = bundle_adjust(perturbed, BAConfig(fix_intrinsics=True, robust_loss="huber"))

    assert result.rmse_before_px > 5.0, "perturbation should have produced a clearly-broken initial guess"
    assert result.rmse_after_px < result.rmse_before_px * 0.25, "BA should substantially reduce reprojection RMSE"
    assert result.rmse_after_px < 5.0
