"""Tests for drishti3d.geometry: the swappable backbone, window planning, and submap merging.

Everything here must pass with no torch, no CUDA, and no model weights
installed -- the NullBackbone and the alignment/window-planning math are
pure numpy, and the MapAnything adapter's *conversion* helpers (resize,
intrinsics scaling, pose conversion) are pure numpy/opencv too; only
``MapAnythingBackbone.load()``/``predict()`` themselves need the real
optional dependencies, and this file never calls those.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from drishti3d.geometry.backbone import NullBackbone, get_backbone
from drishti3d.geometry.mapanything import (
    crop_to_patch_multiple,
    resize_preserving_aspect,
    scale_intrinsics,
)
from drishti3d.geometry.submap import (
    Sim3,
    alignment_residuals,
    merge_submaps,
    strategy_report,
    umeyama_alignment,
    umeyama_fixed_rotation,
)
from drishti3d.geometry.windows import (
    Window,
    estimate_memory,
    max_window_for_budget,
    plan_window_size,
    plan_windows,
)
from drishti3d.types import (
    CameraIntrinsics,
    FrameMetrics,
    Keyframe,
    PointCloud,
    Pose,
    Submap,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_keyframes(n: int) -> list[Keyframe]:
    return [
        Keyframe(
            frame_index=i,
            timestamp=float(i),
            metrics=FrameMetrics(index=i, timestamp=float(i), blur_score=1.0, exposure_score=1.0, mean_luma=1.0, estimated_parallax=0.0),
        )
        for i in range(n)
    ]


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _empty_cloud(n: int) -> PointCloud:
    return PointCloud(xyz=np.zeros((n, 3), dtype=np.float64))


# ---------------------------------------------------------------------------
# NullBackbone
# ---------------------------------------------------------------------------


def test_null_backbone_is_always_available_and_needs_no_load():
    backbone = NullBackbone()
    assert backbone.is_available() is True


def test_null_backbone_produces_documented_shapes_and_straight_line_poses():
    backbone = NullBackbone()
    backbone.load("cpu")
    n, h, w = 5, 48, 64
    images = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]

    result = backbone.predict(images)

    assert result.points.shape == (n, h, w, 3)
    assert result.depth.shape == (n, h, w)
    assert result.confidence.shape == (n, h, w)
    assert len(result.poses) == n
    assert len(result.intrinsics) == n
    assert result.is_metric is True
    assert np.all(result.confidence >= 0.0) and np.all(result.confidence <= 1.0)

    # Poses lie on a straight line: every camera centre's perpendicular
    # offset from the line through the first and last centre is ~0.
    centres = np.array([p.t for p in result.poses])
    direction = centres[-1] - centres[0]
    direction = direction / np.linalg.norm(direction)
    for centre in centres:
        offset = centre - centres[0]
        perpendicular = offset - np.dot(offset, direction) * direction
        assert np.linalg.norm(perpendicular) < 1e-9

    backbone.unload()


def test_null_backbone_respects_given_intrinsics():
    backbone = NullBackbone()
    backbone.load("cpu")
    h, w = 40, 50
    images = [np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8)]
    given = [
        CameraIntrinsics(fx=500.0, fy=500.0, cx=25.0, cy=20.0, width=w, height=h),
        CameraIntrinsics(fx=600.0, fy=600.0, cx=25.0, cy=20.0, width=w, height=h),
    ]

    result = backbone.predict(images, intrinsics=given)

    assert result.intrinsics[0].fx == 500.0
    assert result.intrinsics[1].fx == 600.0


def test_null_backbone_rejects_empty_input():
    backbone = NullBackbone()
    backbone.load("cpu")
    with pytest.raises(ValueError):
        backbone.predict([])


# ---------------------------------------------------------------------------
# get_backbone registry
# ---------------------------------------------------------------------------


def test_get_backbone_null():
    backbone = get_backbone("null")
    assert backbone.name == "null"
    assert backbone.is_available() is True


def test_get_backbone_unknown_raises():
    with pytest.raises(ValueError):
        get_backbone("not-a-real-backbone")


def test_get_backbone_mapanything_is_available_matches_real_importability():
    # torch and the `mapanything` package live behind the `ml` extra / an
    # external checkout (per pyproject.toml) and are not present on every
    # machine this test suite runs on -- but on a machine where they *have*
    # been installed (e.g. to test the adapter against real weights),
    # is_available() must report that too. Either way it must report
    # cleanly (a bool) instead of raising ImportError, and it must agree
    # with whether the imports it depends on actually succeed.
    backbone = get_backbone("mapanything")
    assert backbone.name == "mapanything"

    try:
        import mapanything  # noqa: F401
        import torch  # noqa: F401

        expected = True
    except ImportError:
        expected = False

    assert backbone.is_available() is expected


# ---------------------------------------------------------------------------
# windows.plan_windows
# ---------------------------------------------------------------------------


def test_plan_windows_covers_every_keyframe_ordered_no_drop():
    keyframes = _make_keyframes(23)
    windows = plan_windows(keyframes, window_size=8, overlap=3)

    assert windows == sorted(windows, key=lambda w: w.start)

    covered: set[int] = set()
    for w in windows:
        covered.update(range(w.start, w.end))
    assert covered == set(range(len(keyframes)))


def test_plan_windows_respects_overlap():
    keyframes = _make_keyframes(30)
    windows = plan_windows(keyframes, window_size=10, overlap=4)

    assert len(windows) > 1
    for prev, curr in pairwise(windows):
        assert len(curr.shared_with_previous) == 4
        assert set(curr.shared_with_previous) == set(range(prev.start, prev.end)) & set(range(curr.start, curr.end))


def test_plan_windows_fewer_keyframes_than_window_size():
    keyframes = _make_keyframes(5)
    windows = plan_windows(keyframes, window_size=10, overlap=3)

    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == 5
    assert windows[0].shared_with_previous == []


def test_plan_windows_exactly_one_window():
    keyframes = _make_keyframes(10)
    windows = plan_windows(keyframes, window_size=10, overlap=3)

    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == 10


def test_plan_windows_overlap_floor_enforced():
    keyframes = _make_keyframes(30)
    with pytest.raises(ValueError):
        plan_windows(keyframes, window_size=10, overlap=1)


def test_plan_windows_empty_keyframes():
    assert plan_windows([], window_size=8, overlap=3) == []


def test_plan_windows_max_windows_caps_output():
    keyframes = _make_keyframes(100)
    windows = plan_windows(keyframes, window_size=8, overlap=3, max_windows=2)
    assert len(windows) == 2


# ---------------------------------------------------------------------------
# windows.estimate_memory / max_window_for_budget
# ---------------------------------------------------------------------------


def test_estimate_memory_increases_with_window_size_and_image_size():
    small = estimate_memory(window_size=2, image_size=256)
    larger_views = estimate_memory(window_size=16, image_size=256)
    larger_res = estimate_memory(window_size=2, image_size=518)
    assert larger_views > small
    assert larger_res > small


def test_max_window_for_budget_inverts_estimate_memory():
    budget_gb = 6.0
    image_size = 384
    best = max_window_for_budget(budget_gb, image_size)
    assert best >= 1
    assert estimate_memory(best, image_size) <= budget_gb
    assert estimate_memory(best + 1, image_size) > budget_gb


def test_max_window_for_budget_zero_for_tiny_budget():
    assert max_window_for_budget(0.01, 518) == 0


# ---------------------------------------------------------------------------
# submap.umeyama_alignment -- the single most important test in this file
# ---------------------------------------------------------------------------


def test_umeyama_alignment_recovers_known_similarity_transform():
    rng = np.random.default_rng(42)
    src = rng.uniform(-10.0, 10.0, size=(30, 3))
    true_R = _random_rotation(rng)
    true_scale = 2.35
    true_t = np.array([5.0, -3.0, 1.5])
    dst = true_scale * (src @ true_R.T) + true_t

    transform, degenerate, condition_number = umeyama_alignment(src, dst)

    assert degenerate is False
    assert condition_number < 1e4
    assert transform.scale == pytest.approx(true_scale, abs=1e-6)
    np.testing.assert_allclose(transform.R, true_R, atol=1e-6)
    np.testing.assert_allclose(transform.t, true_t, atol=1e-6)

    # And applying the recovered transform to src should reproduce dst.
    np.testing.assert_allclose(transform.apply(src), dst, atol=1e-6)


def test_umeyama_alignment_rejects_too_few_points():
    rng = np.random.default_rng(0)
    src = rng.uniform(-1, 1, size=(2, 3))
    dst = rng.uniform(-1, 1, size=(2, 3))
    with pytest.raises(ValueError):
        umeyama_alignment(src, dst)


def test_umeyama_degeneracy_guard_triggers_on_collinear_points():
    line_param = np.linspace(0.0, 10.0, 6)
    src = np.stack([line_param, np.zeros(6), np.zeros(6)], axis=1)
    true_R = np.eye(3)
    dst = 1.5 * (src @ true_R.T) + np.array([1.0, 2.0, 3.0])

    _, degenerate, condition_number = umeyama_alignment(src, dst)

    assert degenerate is True
    assert condition_number > 1e4


# ---------------------------------------------------------------------------
# submap.umeyama_fixed_rotation -- Task 2's rotation-fixed Umeyama variant
# ---------------------------------------------------------------------------


def test_umeyama_fixed_rotation_recovers_scale_translation_from_collinear_points():
    # Exactly the configuration umeyama_alignment cannot solve (see the
    # degeneracy test above): collinear source points. With rotation fixed
    # externally (as the "telemetry_rotation" merge strategy does from
    # telemetry, see geometry.submap's module docstring), scale and
    # translation are still perfectly recoverable from them.
    line_param = np.linspace(0.0, 10.0, 6)
    src = np.stack([line_param, np.zeros(6), np.zeros(6)], axis=1)
    rng = np.random.default_rng(3)
    true_R = _random_rotation(rng)
    true_scale = 2.5
    true_t = np.array([1.0, 2.0, 3.0])
    dst = true_scale * (src @ true_R.T) + true_t

    transform = umeyama_fixed_rotation(src, dst, true_R)

    assert transform.scale == pytest.approx(true_scale, abs=1e-8)
    np.testing.assert_allclose(transform.R, true_R, atol=1e-12)
    np.testing.assert_allclose(transform.t, true_t, atol=1e-8)
    np.testing.assert_allclose(transform.apply(src), dst, atol=1e-8)


def test_umeyama_fixed_rotation_identity_matches_translation_only_fit():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dst = 3.0 * src + np.array([5.0, -2.0, 0.5])

    transform = umeyama_fixed_rotation(src, dst, np.eye(3))

    assert transform.scale == pytest.approx(3.0, abs=1e-8)
    np.testing.assert_allclose(transform.t, [5.0, -2.0, 0.5], atol=1e-8)


def test_umeyama_fixed_rotation_rejects_too_few_points():
    with pytest.raises(ValueError):
        umeyama_fixed_rotation(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(3))


# ---------------------------------------------------------------------------
# submap.merge_submaps / alignment_residuals
# ---------------------------------------------------------------------------


def test_merge_submaps_yields_low_residual_on_overlapping_synthetic_submaps():
    rng = np.random.default_rng(7)
    # "True" global camera centres for 10 keyframes, well-spread (not
    # collinear) so this exercises the well-posed case.
    true_centres = rng.uniform(-5.0, 5.0, size=(10, 3))
    true_poses = [Pose(R=np.eye(3), t=true_centres[i]) for i in range(10)]

    def random_sim3() -> Sim3:
        return Sim3(scale=float(rng.uniform(0.5, 2.0)), R=_random_rotation(rng), t=rng.uniform(-3.0, 3.0, size=3))

    transform_a = random_sim3()
    transform_b = random_sim3()

    kf_a = list(range(6))
    kf_b = list(range(3, 10))  # shared with a: {3, 4, 5}

    window_a = Window(index=0, start=0, end=6, shared_with_previous=[])
    window_b = Window(index=1, start=3, end=10, shared_with_previous=[3, 4, 5])

    poses_a = [transform_a.apply_pose(true_poses[i]) for i in kf_a]
    poses_b = [transform_b.apply_pose(true_poses[i]) for i in kf_b]

    submap_a = Submap(
        window=window_a, poses=poses_a, points=_empty_cloud(4), confidence=np.ones(4), keyframe_indices=kf_a, local_origin=poses_a[0]
    )
    submap_b = Submap(
        window=window_b, poses=poses_b, points=_empty_cloud(4), confidence=np.ones(4), keyframe_indices=kf_b, local_origin=poses_b[0]
    )

    residuals = alignment_residuals([submap_a, submap_b])
    assert len(residuals) == 1
    assert residuals[0]["junction"] == (0, 1)
    assert residuals[0]["degenerate"] is False
    assert residuals[0]["rmse_m"] < 1e-6

    cloud, poses = merge_submaps([submap_a, submap_b])
    assert isinstance(cloud, PointCloud)
    assert len(poses) == len(set(kf_a) | set(kf_b))


def test_merge_submaps_single_submap_is_identity():
    poses = [Pose(R=np.eye(3), t=np.array([float(i), 0.0, 0.0])) for i in range(4)]
    window = Window(index=0, start=0, end=4, shared_with_previous=[])
    submap = Submap(window=window, poses=poses, points=_empty_cloud(3), confidence=np.ones(3), keyframe_indices=[0, 1, 2, 3], local_origin=poses[0])

    _cloud, merged_poses = merge_submaps([submap])

    assert alignment_residuals([submap]) == []
    assert len(merged_poses) == 4
    for original, merged in zip(poses, merged_poses, strict=True):
        np.testing.assert_allclose(merged.t, original.t)


def test_merge_submaps_degenerate_guard_triggers_on_collinear_shared_cameras():
    # Shared cameras between the two submaps sit on a straight line -- the
    # canonical single-flight-strip degeneracy this module's docstring
    # warns about.
    line_param = np.linspace(0.0, 9.0, 10)
    true_centres = np.stack([line_param, np.zeros(10), np.full(10, 20.0)], axis=1)
    true_poses = [Pose(R=np.eye(3), t=true_centres[i]) for i in range(10)]

    kf_a = list(range(6))
    kf_b = list(range(3, 10))
    window_a = Window(index=0, start=0, end=6, shared_with_previous=[])
    window_b = Window(index=1, start=3, end=10, shared_with_previous=[3, 4, 5])

    submap_a = Submap(
        window=window_a, poses=[true_poses[i] for i in kf_a], points=_empty_cloud(2), confidence=np.ones(2), keyframe_indices=kf_a, local_origin=true_poses[0]
    )
    submap_b = Submap(
        window=window_b, poses=[true_poses[i] for i in kf_b], points=_empty_cloud(2), confidence=np.ones(2), keyframe_indices=kf_b, local_origin=true_poses[3]
    )

    residuals = alignment_residuals([submap_a, submap_b])
    assert residuals[0]["degenerate"] is True


def test_merge_submaps_raises_on_insufficient_shared_cameras():
    poses_a = [Pose(R=np.eye(3), t=np.array([float(i), 0.0, 0.0])) for i in range(4)]
    poses_b = [Pose(R=np.eye(3), t=np.array([float(i), 0.0, 0.0])) for i in range(2, 6)]
    window_a = Window(index=0, start=0, end=4, shared_with_previous=[])
    # Only one shared keyframe (index 3) -- below the minimum of 3.
    window_b = Window(index=1, start=2, end=6, shared_with_previous=[3])

    submap_a = Submap(window=window_a, poses=poses_a, points=_empty_cloud(1), confidence=np.ones(1), keyframe_indices=[0, 1, 2, 3], local_origin=poses_a[0])
    submap_b = Submap(window=window_b, poses=poses_b, points=_empty_cloud(1), confidence=np.ones(1), keyframe_indices=[2, 3, 4, 5], local_origin=poses_b[0])

    with pytest.raises(ValueError):
        merge_submaps([submap_a, submap_b])


# ---------------------------------------------------------------------------
# submap.merge_submaps -- strategy="telemetry_rotation" / "gps_anchored"
# (Task 2: the actual submap-merge rotation bug fix)
# ---------------------------------------------------------------------------


def _collinear_two_submap_setup(true_R: np.ndarray, true_scale: float, true_t: np.ndarray):
    """Two overlapping submaps over a collinear (single-flight-strip) local track.

    Local-frame poses are identity-rotation cameras on a straight line --
    exactly the configuration ``umeyama_alignment`` cannot recover rotation
    from (see ``test_merge_submaps_degenerate_guard_triggers_on_collinear_shared_cameras``).
    Returns everything needed to call ``merge_submaps``/``alignment_residuals``
    with ``strategy="telemetry_rotation"``: the two submaps, a dense
    ``camera_gps_enu`` (every keyframe, not just the 3 shared ones) in the
    known true world frame, and ``conditioned_R`` (every keyframe's true
    world rotation -- trivially ``true_R`` here since every local pose is
    identity).
    """
    n = 10
    line_param = np.linspace(0.0, 45.0, n)
    local_centres = np.stack([line_param, np.zeros(n), np.zeros(n)], axis=1)
    local_poses = [Pose(R=np.eye(3), t=local_centres[i]) for i in range(n)]

    world_positions = {i: true_scale * (true_R @ local_centres[i]) + true_t for i in range(n)}
    conditioned_R = {i: true_R.copy() for i in range(n)}

    kf_a = list(range(6))
    kf_b = list(range(3, 10))
    window_a = Window(index=0, start=0, end=6, shared_with_previous=[])
    window_b = Window(index=1, start=3, end=10, shared_with_previous=[3, 4, 5])

    submap_a = Submap(
        window=window_a, poses=[local_poses[i] for i in kf_a], points=_empty_cloud(2),
        confidence=np.ones(2), keyframe_indices=kf_a, local_origin=local_poses[0],
    )
    submap_b = Submap(
        window=window_b, poses=[local_poses[i] for i in kf_b], points=_empty_cloud(2),
        confidence=np.ones(2), keyframe_indices=kf_b, local_origin=local_poses[3],
    )
    return submap_a, submap_b, world_positions, conditioned_R


def test_merge_submaps_telemetry_rotation_recovers_orientation_on_collinear_cameras():
    rng = np.random.default_rng(11)
    true_R = _random_rotation(rng)
    true_scale = 1.7
    true_t = np.array([10.0, -5.0, 100.0])
    submap_a, submap_b, world_positions, conditioned_R = _collinear_two_submap_setup(true_R, true_scale, true_t)

    # Sanity: this really is the degenerate case the plain/chained path
    # cannot handle -- the bug being fixed.
    plain_residuals = alignment_residuals([submap_a, submap_b])
    assert plain_residuals[0]["degenerate"] is True

    cloud, poses = merge_submaps(
        [submap_a, submap_b],
        camera_gps_enu=world_positions,
        conditioned_R=conditioned_R,
        strategy="telemetry_rotation",
    )

    assert isinstance(cloud, PointCloud)
    for i, pose in enumerate(poses):
        np.testing.assert_allclose(pose.t, world_positions[i], atol=1e-6)
        np.testing.assert_allclose(pose.R, true_R, atol=1e-6)


def test_strategy_report_reports_full_anchoring_under_telemetry_rotation():
    rng = np.random.default_rng(12)
    true_R = _random_rotation(rng)
    submap_a, submap_b, world_positions, conditioned_R = _collinear_two_submap_setup(
        true_R, true_scale=1.0, true_t=np.zeros(3)
    )

    report = strategy_report(
        [submap_a, submap_b], camera_gps_enu=world_positions, conditioned_R=conditioned_R, strategy="telemetry_rotation"
    )

    assert report["strategy"] == "telemetry_rotation"
    assert report["n_submaps"] == 2
    assert report["n_anchored"] == 2
    assert report["n_fallback_to_chaining"] == 0
    assert report["fully_used_requested_strategy"] is True


def test_strategy_report_falls_back_when_telemetry_rotation_lacks_conditioning():
    # No conditioned_R at all -- telemetry_rotation cannot fix a rotation
    # for any submap, so every submap (after submap 0) must fall back to
    # chaining. This must be reported, never silent (see module docstring).
    rng = np.random.default_rng(13)
    true_R = _random_rotation(rng)
    submap_a, submap_b, world_positions, _conditioned_R = _collinear_two_submap_setup(
        true_R, true_scale=1.0, true_t=np.zeros(3)
    )

    report = strategy_report(
        [submap_a, submap_b], camera_gps_enu=world_positions, conditioned_R=None, strategy="telemetry_rotation"
    )

    assert report["n_anchored"] == 0
    assert report["fully_used_requested_strategy"] is False
    assert report["n_fallback_to_chaining"] == 1


def test_merge_submaps_gps_anchored_matches_chained_sim3_with_no_camera_gps_enu():
    # strategy="gps_anchored" with no camera_gps_enu given has nothing to
    # anchor on and must degrade to plain chaining, not crash or silently
    # produce a different (wrong) answer.
    rng = np.random.default_rng(7)
    true_centres = rng.uniform(-5.0, 5.0, size=(10, 3))
    true_poses = [Pose(R=np.eye(3), t=true_centres[i]) for i in range(10)]

    kf_a = list(range(6))
    kf_b = list(range(3, 10))
    window_a = Window(index=0, start=0, end=6, shared_with_previous=[])
    window_b = Window(index=1, start=3, end=10, shared_with_previous=[3, 4, 5])
    submap_a = Submap(window=window_a, poses=[true_poses[i] for i in kf_a], points=_empty_cloud(2), confidence=np.ones(2), keyframe_indices=kf_a, local_origin=true_poses[0])
    submap_b = Submap(window=window_b, poses=[true_poses[i] for i in kf_b], points=_empty_cloud(2), confidence=np.ones(2), keyframe_indices=kf_b, local_origin=true_poses[3])

    _cloud_default, poses_default = merge_submaps([submap_a, submap_b])
    _cloud_gps, poses_gps = merge_submaps([submap_a, submap_b], strategy="gps_anchored")

    for p_default, p_gps in zip(poses_default, poses_gps, strict=True):
        np.testing.assert_allclose(p_default.t, p_gps.t, atol=1e-8)


def test_merge_submaps_gps_anchored_corrects_degenerate_submap_rotation_via_telemetry():
    """Real-footage bug (see ``_gps_full_anchor_transform``'s docstring): a flight
    can be non-collinear overall (so ``"gps_anchored"`` is the right strategy)
    while one particular window still lands on a locally-straight leg,
    making that submap's own full-Sim(3) fit degenerate even though it
    still returns *some* rotation. Without a fix, that submap's dense
    points get rotated by an unreliable, essentially arbitrary rotation and
    can land far from where they belong -- exactly what was observed on
    real 16-keyframe drone footage (2 of 3 submaps degenerate, ending up as
    several small blobs tens of metres away from an otherwise-continuous
    model). This verifies the fix: when telemetry conditioning is
    available, the degenerate submap's rotation comes from that instead.
    """
    rng = np.random.default_rng(21)
    true_R = _random_rotation(rng)
    true_scale = 1.3
    true_t = np.array([5.0, 20.0, 50.0])

    # submap_a: a genuinely non-collinear local track (spread over 2 axes)
    # -- its own full-Sim(3) GPS fit should be well-conditioned on its own.
    local_a = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 5.0, 0.0], [5.0, 15.0, 0.0], [15.0, 20.0, 0.0], [0.0, 10.0, 0.0]]
    )
    poses_a = [Pose(R=np.eye(3), t=local_a[i]) for i in range(6)]
    kf_a = list(range(6))
    window_a = Window(index=0, start=0, end=6, shared_with_previous=[])
    submap_a = Submap(
        window=window_a, poses=poses_a, points=_empty_cloud(2), confidence=np.ones(2),
        keyframe_indices=kf_a, local_origin=poses_a[0],
    )

    # submap_b: a collinear local track (this window happened to land on a
    # straight leg of an otherwise-curved flight) -- its own full-Sim(3)
    # GPS fit is degenerate.
    line_param = np.linspace(0.0, 45.0, 7)
    local_b = np.stack([line_param, np.zeros(7), np.zeros(7)], axis=1)
    poses_b = [Pose(R=np.eye(3), t=local_b[i]) for i in range(7)]
    kf_b = list(range(3, 10))
    window_b = Window(index=1, start=3, end=10, shared_with_previous=[3, 4, 5])
    submap_b = Submap(
        window=window_b, poses=poses_b, points=_empty_cloud(2), confidence=np.ones(2),
        keyframe_indices=kf_b, local_origin=poses_b[0],
    )

    world_positions = {}
    for i, p in zip(kf_a, local_a, strict=True):
        world_positions[i] = true_scale * (true_R @ p) + true_t
    for i, p in zip(kf_b, local_b, strict=True):
        world_positions[i] = true_scale * (true_R @ p) + true_t

    conditioned_R = {i: true_R.copy() for i in kf_b}  # telemetry conditioning available for submap B's keyframes only

    # Sanity: submap B's own gps_anchored fit really is degenerate without
    # telemetry conditioning to correct it.
    report_uncorrected = strategy_report(
        [submap_a, submap_b], camera_gps_enu=world_positions, conditioned_R=None, strategy="gps_anchored"
    )
    _anchored_b, diag_b_uncorrected = report_uncorrected["submap_diagnostics"][1]
    assert diag_b_uncorrected["degenerate"] is True

    cloud, poses = merge_submaps(
        [submap_a, submap_b], camera_gps_enu=world_positions, conditioned_R=conditioned_R, strategy="gps_anchored"
    )
    assert isinstance(cloud, PointCloud)

    pose_by_kf = dict(zip(sorted(set(kf_a) | set(kf_b)), poses, strict=True))
    # Every keyframe -- including submap B's, whose own fit was degenerate
    # -- should land at its true world position AND with the true world
    # rotation, not an arbitrary/degenerate one.
    for i in kf_b:
        np.testing.assert_allclose(pose_by_kf[i].t, world_positions[i], atol=1e-6)
        np.testing.assert_allclose(pose_by_kf[i].R, true_R, atol=1e-6)

    report_corrected = strategy_report(
        [submap_a, submap_b], camera_gps_enu=world_positions, conditioned_R=conditioned_R, strategy="gps_anchored"
    )
    _anchored_b2, diag_b_corrected = report_corrected["submap_diagnostics"][1]
    assert diag_b_corrected["degenerate"] is False
    assert diag_b_corrected["rotation_source"] == "telemetry_conditioned"


def test_merge_submaps_rejects_unknown_strategy():
    poses = [Pose(R=np.eye(3), t=np.array([float(i), 0.0, 0.0])) for i in range(4)]
    window = Window(index=0, start=0, end=4, shared_with_previous=[])
    submap = Submap(window=window, poses=poses, points=_empty_cloud(3), confidence=np.ones(3), keyframe_indices=[0, 1, 2, 3], local_origin=poses[0])

    with pytest.raises(ValueError):
        merge_submaps([submap, submap], strategy="not-a-real-strategy")


# ---------------------------------------------------------------------------
# windows.plan_window_size -- Task 3: derive window_size from memory budget
# and keyframe count
# ---------------------------------------------------------------------------


def test_plan_window_size_never_exceeds_keyframe_count():
    assert plan_window_size(n_keyframes=5, image_size=518, requested=14) == 5


def test_plan_window_size_uses_requested_when_it_fits():
    # A generous budget (comparable to unified-memory headroom, not the
    # module default's conservative ~6GB discrete-GPU target) should let
    # the requested size through unmodified.
    size = plan_window_size(n_keyframes=100, image_size=518, vram_budget_gb=20.0, requested=14)
    assert size == 14


def test_plan_window_size_respects_a_tight_memory_budget():
    size = plan_window_size(n_keyframes=100, image_size=518, vram_budget_gb=0.01, requested=14)
    assert size < 14


def test_plan_window_size_floors_when_budget_limited_not_when_requested_is_small():
    # A tight *memory budget* (with plenty of keyframes and a generous
    # request) shouldn't be allowed to plan an unusably tiny window --
    # this floors back up.
    budget_limited = plan_window_size(n_keyframes=100, image_size=518, vram_budget_gb=3.0, requested=14)
    assert budget_limited == 8  # min(_MIN_PLANNED_WINDOW_SIZE, requested) -- budget_cap(3.0GB) is 1

    # An explicit small *request* (e.g. a test fixture deliberately using a
    # small window), with an ample budget, must NOT be silently bumped up
    # to that same floor -- the caller asked for exactly this.
    explicitly_requested = plan_window_size(n_keyframes=6, image_size=518, vram_budget_gb=20.0, requested=3)
    assert explicitly_requested == 3


# ---------------------------------------------------------------------------
# mapanything.py conversion helpers (pure numpy/opencv -- no torch needed)
# ---------------------------------------------------------------------------


def test_resize_preserving_aspect_keeps_aspect_ratio():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    resized, scale = resize_preserving_aspect(img, max_size=320)

    assert max(resized.shape[:2]) == 320
    assert scale == pytest.approx(320 / 640)
    np.testing.assert_allclose(resized.shape[0] / resized.shape[1], img.shape[0] / img.shape[1], rtol=1e-2)


def test_resize_preserving_aspect_scales_intrinsics_by_exact_factor():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    intr = CameraIntrinsics(fx=1000.0, fy=900.0, cx=320.0, cy=240.0, width=640, height=480)

    _, scale = resize_preserving_aspect(img, max_size=320)
    scaled = scale_intrinsics(intr, scale)

    assert scaled.fx == pytest.approx(intr.fx * scale)
    assert scaled.fy == pytest.approx(intr.fy * scale)
    assert scaled.cx == pytest.approx(intr.cx * scale)
    assert scaled.cy == pytest.approx(intr.cy * scale)


def test_scale_intrinsics_arbitrary_factor():
    intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0, width=800, height=600)
    scale = 0.37
    scaled = scale_intrinsics(intr, scale)

    assert scaled.fx == pytest.approx(intr.fx * scale)
    assert scaled.fy == pytest.approx(intr.fy * scale)
    assert scaled.cx == pytest.approx(intr.cx * scale)
    assert scaled.cy == pytest.approx(intr.cy * scale)


def test_crop_to_patch_multiple_produces_exact_multiple_of_patch_size():
    # Real-world regression: MapAnything's patch embedding hard-asserts the
    # input shape is divisible by the encoder's patch size (14 for the
    # installed checkpoint) -- resize_preserving_aspect's max-side scaling
    # (e.g. --sizes 256,384 in scripts/benchmark_vram.py) has no reason to
    # land on a multiple of 14, so this crop step is required.
    img = np.zeros((256, 384, 3), dtype=np.uint8)
    cropped, top, left = crop_to_patch_multiple(img, patch_size=14)

    assert cropped.shape[0] % 14 == 0
    assert cropped.shape[1] % 14 == 0
    assert cropped.shape[0] == 252  # 18 * 14
    assert cropped.shape[1] == 378  # 27 * 14
    # Centered crop.
    assert top == (256 - 252) // 2
    assert left == (384 - 378) // 2


def test_crop_to_patch_multiple_is_noop_when_already_aligned():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cropped, top, left = crop_to_patch_multiple(img, patch_size=14)

    assert cropped.shape == img.shape
    assert (top, left) == (0, 0)


def test_crop_to_patch_multiple_shifted_principal_point_stays_centered():
    # The crop is centered, so a principal point at the image center should
    # land back near the (smaller) cropped image's center once shifted by
    # (crop_top, crop_left) -- verifying the offset convention this
    # adapter's predict() uses to correct cx/cy after cropping.
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=192.0, cy=128.0, width=384, height=256)
    img = np.zeros((256, 384, 3), dtype=np.uint8)
    cropped, top, left = crop_to_patch_multiple(img, patch_size=14)

    shifted_cx = intr.cx - left
    shifted_cy = intr.cy - top
    assert shifted_cx == pytest.approx(cropped.shape[1] / 2, abs=1.0)
    assert shifted_cy == pytest.approx(cropped.shape[0] / 2, abs=1.0)


# ---------------------------------------------------------------------------
# autocast device policy
# ---------------------------------------------------------------------------


def test_autocast_dtype_per_device():
    """bf16 on the accelerators, fp32 on CPU -- including indexed CUDA devices.

    The indexed case is the bug this encodes: ``device == "cuda"`` answers
    False for "cuda:1", which silently ran the model in fp32 on every GPU
    but the first. MPS is bf16 on measured evidence (1.03 mm median point
    displacement vs fp32, 2.78x faster) -- see _autocast_dtype's docstring.
    """
    from drishti3d.geometry.mapanything import _autocast_dtype

    assert _autocast_dtype("cuda") == "bf16"
    assert _autocast_dtype("cuda:0") == "bf16"
    assert _autocast_dtype("cuda:1") == "bf16"
    assert _autocast_dtype("mps") == "bf16"
    assert _autocast_dtype("cpu") is None
    assert _autocast_dtype(None) is None


def test_confidence_decoding_matches_the_source_convention():
    """Two confidence conventions; applying the wrong decoder zeroes everything.

    MapAnything's learned head emits 1 + exp(x) in [1, inf), decoded by
    1 - 1/conf. Its multi-view depth-consistency confidence is documented
    as already being in [0, 1]. Decoding the latter with the former's
    formula flattens every value to exactly 0.0 -- measured: the
    confidence filter then dropped all 4,099,909 points and fusion
    produced no vertices.
    """
    import numpy as np

    from drishti3d.geometry.mapanything import _predictions_to_result

    def pred(conf):
        return {
            "camera_poses": np.eye(4)[None],
            "pts3d": np.zeros((1, 2, 2, 3)),
            "depth_z": np.ones((1, 2, 2)),
            "conf": np.asarray(conf, dtype=np.float64)[None],
        }

    # Multi-view confidence: already [0, 1], must survive unchanged.
    mv = np.array([[0.0, 0.25], [0.75, 1.0]])
    out = _predictions_to_result([pred(mv)], "ckpt", None, confidence_is_unit_range=True)
    np.testing.assert_allclose(out.confidence[0], mv, atol=1e-9)

    # The learned decoder applied to the same input destroys it.
    wrong = _predictions_to_result([pred(mv)], "ckpt", None, confidence_is_unit_range=False)
    assert float(wrong.confidence[0].max()) == 0.0, "this is the failure mode being guarded"

    # And the learned head's own encoding still decodes monotonically.
    learned = np.array([[1.0, 2.0], [5.0, 101.0]])
    ok = _predictions_to_result([pred(learned)], "ckpt", None, confidence_is_unit_range=False)
    vals = ok.confidence[0].ravel()
    assert vals[0] == pytest.approx(0.0) and vals[-1] == pytest.approx(0.990099, abs=1e-5)
    assert np.all(np.diff(vals) > 0)


# ---------------------------------------------------------------------------
# extent-capped window planning
# ---------------------------------------------------------------------------


def _kf_at(positions):
    """Keyframes carrying only the poses plan_windows reads."""
    from drishti3d.types import FrameMetrics, Keyframe, Pose

    out = []
    for i, p in enumerate(positions):
        out.append(
            Keyframe(
                frame_index=i,
                timestamp=float(i),
                metrics=FrameMetrics(index=i, timestamp=float(i), blur_score=1.0,
                                     exposure_score=1.0, mean_luma=1.0, estimated_parallax=0.0),
                pose=Pose(R=np.eye(3), t=np.asarray(p, dtype=np.float64)),
            )
        )
    return out


def test_extent_cap_shrinks_windows_where_keyframes_are_far_apart():
    """The sample flight's shape: tight at the start, wide later.

    A fixed count spans 48 m early and 148 m later, which is exactly the
    split that made windows 0-3 return good depth and 5-18 return depth
    7x too shallow. The cap has to hold the extent steady instead.
    """
    from drishti3d.geometry.windows import plan_windows

    tight = [[i * 6.5, 0.0, 120.0] for i in range(20)]
    wide = [[130.0 + i * 33.0, 0.0, 120.0] for i in range(20)]
    kfs = _kf_at(tight + wide)

    windows = plan_windows(kfs, window_size=8, overlap=4, max_extent_m=60.0)

    extents = []
    for w in windows:
        pts = np.array([kfs[i].pose.t for i in w.keyframe_indices()])
        extents.append(float(np.linalg.norm(pts - pts[0], axis=1).max()))

    # Every window stays near the cap; without it the later ones are ~230 m.
    assert max(extents) <= 60.0 + 33.0, extents
    # Early windows keep their full size, later ones shrink.
    assert windows[0].size > windows[-1].size, [w.size for w in windows]


def test_extent_cap_still_covers_every_keyframe():
    from drishti3d.geometry.windows import plan_windows

    kfs = _kf_at([[i * 40.0, 0.0, 120.0] for i in range(30)])
    windows = plan_windows(kfs, window_size=8, overlap=4, max_extent_m=60.0)

    covered = set()
    for w in windows:
        covered.update(w.keyframe_indices())
    assert covered == set(range(30))
    # And it must terminate with a sane number of windows, not thousands.
    assert len(windows) < 30


def test_extent_cap_keeps_enough_cameras_to_align():
    """A window below 3 cameras cannot be Sim(3)-aligned at all."""
    from drishti3d.geometry.windows import plan_windows

    # Keyframes 500 m apart: every pair already busts a 60 m cap.
    kfs = _kf_at([[i * 500.0, 0.0, 120.0] for i in range(12)])
    windows = plan_windows(kfs, window_size=8, overlap=4, max_extent_m=60.0)
    assert all(w.size >= 3 for w in windows), [w.size for w in windows]


def test_no_extent_cap_reproduces_the_original_plan():
    from drishti3d.geometry.windows import plan_windows

    kfs = _kf_at([[i * 20.0, 0.0, 120.0] for i in range(40)])
    assert plan_windows(kfs, window_size=8, overlap=4) == plan_windows(
        kfs, window_size=8, overlap=4, max_extent_m=None
    )


# ---------------------------------------------------------------------------
# merge strategy must be chosen for the window, not the flight
# ---------------------------------------------------------------------------


def test_window_collinearity_sees_straight_passes_inside_a_grid():
    """A grid flight is made of straight passes; the merge fits one pass.

    The whole-track measure calls this scene non-collinear, because the
    passes spread out in a second direction. Every individual window is
    still a straight line, and rotation about a straight line is exactly
    what the merge's camera-centre fit cannot recover.
    """
    from drishti3d.geometry.windows import Window
    from drishti3d.pipeline.stages import _median_window_collinearity

    # Four parallel passes: a grid overall.
    enu = {}
    i = 0
    for pass_y in (0.0, 60.0, 120.0, 180.0):
        for step in range(10):
            enu[i] = np.array([step * 20.0, pass_y, 120.0])
            i += 1

    track = np.array([enu[k] for k in sorted(enu)])
    singular = np.linalg.svd(track - track.mean(axis=0), compute_uv=False)
    flight_collinearity = 1.0 - singular[1] / singular[0]
    assert flight_collinearity < 0.85, "fixture should look like a grid overall"

    windows = [Window(index=w, start=w * 4, end=w * 4 + 6, shared_with_previous=[]) for w in range(8)]
    assert _median_window_collinearity(windows, enu) >= 0.85


def test_window_collinearity_is_low_for_a_genuinely_spread_window():
    from drishti3d.geometry.windows import Window
    from drishti3d.pipeline.stages import _median_window_collinearity

    enu = {
        0: np.array([0.0, 0.0, 120.0]),
        1: np.array([50.0, 0.0, 120.0]),
        2: np.array([50.0, 50.0, 120.0]),
        3: np.array([0.0, 50.0, 120.0]),
    }
    windows = [Window(index=0, start=0, end=4, shared_with_previous=[])]
    assert _median_window_collinearity(windows, enu) < 0.5


def test_a_window_too_small_to_measure_counts_as_a_line():
    """Fewer than 3 cameras is the worst case for a rotation fit."""
    from drishti3d.geometry.windows import Window
    from drishti3d.pipeline.stages import _median_window_collinearity

    enu = {0: np.zeros(3), 1: np.array([10.0, 0.0, 0.0])}
    windows = [Window(index=0, start=0, end=2, shared_with_previous=[])]
    assert _median_window_collinearity(windows, enu) == 1.0


def test_single_inference_plans_one_window_for_the_whole_flight():
    """Window boundaries, not the backbone, are what wreck the surface.

    Measured roughness inside a 2 m cell: 2.40 m from one inference over
    32 views, against 8-12 m from 19 merged windows and 72 m from 64.
    So the planner must produce ONE window when it can.
    """
    from drishti3d.geometry.windows import plan_windows

    kfs = _kf_at([[i * 20.0, 0.0, 120.0] for i in range(40)])
    windows = plan_windows(kfs, window_size=len(kfs), overlap=4)
    assert len(windows) == 1
    assert windows[0].size == 40
    assert windows[0].shared_with_previous == []


def test_single_inference_falls_back_to_fewest_windows_under_a_ceiling():
    """A ceiling means as few boundaries as possible, not many small ones."""
    from drishti3d.geometry.windows import plan_windows

    kfs = _kf_at([[i * 20.0, 0.0, 120.0] for i in range(40)])
    windows = plan_windows(kfs, window_size=24, overlap=4)
    assert len(windows) <= 3, [w.size for w in windows]
    covered = set()
    for w in windows:
        covered.update(w.keyframe_indices())
    assert covered == set(range(40))
