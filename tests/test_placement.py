"""Tests for both placement checks: before geometry, and before fusion.

The pre-fusion half tests the property the module exists for: a
reconstruction whose windows are stacked at different heights must be
reported as FAILED, and the identical scene with the windows correctly
placed must PASS -- even though the two are indistinguishable from above.
One test asserts that indistinguishability directly, because it is the
reason every top-down preview this pipeline ever wrote was blind to the
error that mattered.

The pre-geometry half tests frame footprints, which need no depth at all:
where a frame lands is fixed by its pose, its intrinsics and a ground
height. Those tests check the placement against the pinhole arithmetic
rather than against a stored image, so they say whether the geometry is
right and not merely whether it changed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drishti3d.export.placement import (
    ground_z_from_telemetry,
    placement_metrics,
    render_elevation,
    render_topdown_by_submap,
    write_placement_check,
)


def _ground(n_submaps: int = 6, per: int = 4000, z_offsets=None, seed: int = 0):
    """A flat field imaged by ``n_submaps`` overlapping windows.

    Every window sees the same 60 x 60 m patch, so a correct merge puts
    them all at the same height. ``z_offsets`` injects the merge error.
    """
    rng = np.random.default_rng(seed)
    offsets = z_offsets if z_offsets is not None else [0.0] * n_submaps
    xyz, labels = [], []
    for s in range(n_submaps):
        pts = rng.uniform(-30, 30, (per, 2))
        z = rng.normal(0.0, 0.02, per) + offsets[s]
        xyz.append(np.c_[pts, z])
        labels.append(np.full(per, s))
    return np.vstack(xyz), np.concatenate(labels)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def test_correctly_placed_windows_pass() -> None:
    xyz, labels = _ground()
    report = placement_metrics(xyz, labels)
    assert report.passed
    assert report.ground_spread_m < 0.5
    assert report.metrics["flat_thickness_median_m"] < 0.5


def test_stacked_windows_fail_with_the_stack_height() -> None:
    offsets = [0.0, 3.0, -2.0, 7.0, 11.0, -5.0]
    xyz, labels = _ground(z_offsets=offsets)
    report = placement_metrics(xyz, labels)

    assert not report.passed
    # The spread must report the actual stack height, not merely be large.
    expected = max(offsets) - min(offsets)
    assert abs(report.ground_spread_m - expected) < 0.5, report.metrics


def test_top_down_cannot_tell_the_two_apart() -> None:
    """The reason this module exists, asserted rather than claimed."""
    good_xyz, labels = _ground()
    bad_xyz, _ = _ground(z_offsets=[0.0, 3.0, -2.0, 7.0, 11.0, -5.0])

    good = render_topdown_by_submap(good_xyz, labels)
    bad = render_topdown_by_submap(bad_xyz, labels)
    assert good is not None and bad is not None
    assert good.shape == bad.shape
    # Same XY, same colours, same splat: the top-down rasters are equal
    # while the scenes differ by 16 m of vertical error.
    assert np.array_equal(good, bad)

    # And the elevation view is emphatically not equal.
    elev_good = render_elevation(good_xyz, labels, axis="x")
    elev_bad = render_elevation(bad_xyz, labels, axis="x")
    assert elev_good is not None and elev_bad is not None
    assert not np.array_equal(elev_good, elev_bad)


def test_thickness_separates_stacked_ground_from_real_structure() -> None:
    """A tall building is not a placement failure."""
    rng = np.random.default_rng(3)
    ground = np.c_[rng.uniform(-30, 30, (20000, 2)), rng.normal(0, 0.02, 20000)]
    # A 10 m building over a small footprint, seen by every window.
    wall = np.c_[rng.uniform(-4, 4, (4000, 2)), rng.uniform(0, 10, 4000)]
    xyz = np.vstack([ground, wall])
    labels = np.concatenate([rng.integers(0, 6, 20000), rng.integers(0, 6, 4000)])

    report = placement_metrics(xyz, labels)
    assert report.passed, report.metrics
    # The building's cells are excluded from the flat measurement, so the
    # thickness reflects the ground and not the wall.
    assert report.metrics["flat_thickness_median_m"] < 0.5
    assert report.metrics["cells_flat"] < report.metrics["cells_measured"]


def test_gps_ground_reference_reports_absolute_error() -> None:
    """Windows can agree with each other and still be in the wrong place."""
    xyz, labels = _ground()
    xyz[:, 2] += 12.0  # whole reconstruction floating, internally consistent

    report = placement_metrics(xyz, labels, gps_ground_z=0.0)
    # Relative agreement is fine...
    assert report.ground_spread_m < 0.5
    # ...but the absolute placement is 12 m out, and that is reported.
    assert abs(report.metrics["gps_ground_error_median_m"] - 12.0) < 0.2
    assert abs(report.metrics["gps_ground_error_max_abs_m"] - 12.0) < 0.2


def test_ground_z_from_telemetry_is_camera_height_minus_altitude() -> None:
    cams = np.array([[0.0, 0.0, 118.0], [1.0, 0.0, 122.0], [2.0, 0.0, 120.0]])
    assert ground_z_from_telemetry(cams, [120.0, 120.0, 120.0]) == 0.0
    # A single bad sample must not move the reference line.
    assert abs(ground_z_from_telemetry(cams, [120.0, 120.0, 20.0]) - 0.0) < 2.1
    assert ground_z_from_telemetry(cams, [None, None, None]) is None


# ---------------------------------------------------------------------------
# the writer -- must never take the pipeline down with it
# ---------------------------------------------------------------------------


def test_writes_every_panel_and_the_metrics(tmp_path) -> None:
    xyz, labels = _ground(z_offsets=[0, 4, -3, 8, 12, -6])
    report = write_placement_check(tmp_path, xyz, labels, gps_ground_z=0.0)

    for name in (
        "placement_topdown.png",
        "placement_elevation_x.png",
        "placement_elevation_y.png",
        "placement_check.png",
    ):
        assert (tmp_path / name).exists(), name

    saved = json.loads((tmp_path / "placement.json").read_text())
    assert saved["passed"] is False
    assert saved["ground_spread_m"] == report.ground_spread_m


def test_empty_and_mismatched_input_do_not_raise(tmp_path) -> None:
    empty = write_placement_check(tmp_path, np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
    assert empty.metrics["n_submaps"] == 0

    # Labels that do not line up with the points are a caller bug, but the
    # placement check is a diagnostic: it must report nothing rather than
    # abort a reconstruction that is otherwise fine.
    mismatched = write_placement_check(tmp_path, np.zeros((10, 3)), np.zeros(4, dtype=np.int64))
    assert mismatched.metrics["n_submaps"] == 0


def test_unwritable_directory_is_survivable(tmp_path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    xyz, labels = _ground(n_submaps=2, per=500)
    # Metrics still come back; only the rendering failed.
    report = write_placement_check(blocker, xyz, labels)
    assert report.metrics["n_submaps"] == 2


def test_subsampling_is_deterministic(tmp_path) -> None:
    xyz, labels = _ground(n_submaps=4, per=3000)
    a = write_placement_check(tmp_path / "a", xyz, labels, max_points=2000)
    b = write_placement_check(tmp_path / "b", xyz, labels, max_points=2000)
    assert a.metrics["ground_z_by_submap"] == b.metrics["ground_z_by_submap"]


# ---------------------------------------------------------------------------
# the merge must hand back labels that line up with its own cloud
# ---------------------------------------------------------------------------


def test_merge_submaps_returns_labels_parallel_to_its_cloud() -> None:
    from drishti3d.geometry.submap import merge_submaps
    from drishti3d.types import Pose, PointCloud, Submap
    from drishti3d.geometry.windows import Window

    rng = np.random.default_rng(1)

    def _submap(index: int, start: int, size: int, shared: list[int]) -> Submap:
        idx = list(range(start, start + size))
        pts = rng.uniform(-10, 10, (400, 3))
        return Submap(
            window=Window(index=index, start=start, end=start + size, shared_with_previous=shared),
            poses=[Pose(R=np.eye(3), t=np.array([float(i), 0.0, 50.0])) for i in idx],
            points=PointCloud(xyz=pts),
            confidence=np.ones(400),
            keyframe_indices=idx,
            local_origin=np.zeros(3),
        )

    # Four shared keyframes per junction: Umeyama needs >= 3 non-collinear
    # camera centres, and the poses above lie along the X axis, so the
    # merge falls back to its translation-only fit rather than failing.
    submaps = [
        _submap(0, 0, 6, []),
        _submap(1, 2, 6, [2, 3, 4, 5]),
        _submap(2, 4, 6, [4, 5, 6, 7]),
    ]
    cloud, _poses, labels = merge_submaps(submaps, return_labels=True)

    assert labels.shape[0] == cloud.xyz.shape[0]
    assert set(np.unique(labels)).issubset({0, 1, 2})
    # Without return_labels the signature is unchanged for every existing caller.
    two_tuple = merge_submaps(submaps)
    assert len(two_tuple) == 2


# ---------------------------------------------------------------------------
# BEFORE geometry: footprints from poses alone
# ---------------------------------------------------------------------------


def _nadir(x: float, y: float, z: float = 120.0):
    from drishti3d.types import Pose

    # World-from-camera for a camera looking straight down: camera +Z
    # (forward) maps to world -Z, camera +Y (down in image) to world -Y.
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return Pose(R=R, t=np.array([x, y, z]))


def _intr(w: int = 4000, h: int = 3000, fx: float = 2700.0):
    from drishti3d.types import CameraIntrinsics

    return CameraIntrinsics(fx=fx, fy=fx, cx=w / 2, cy=h / 2, width=w, height=h)


def test_nadir_footprint_matches_the_pinhole_arithmetic() -> None:
    from drishti3d.export.placement import frame_footprints

    intr = _intr()
    height = 120.0
    (footprint,) = frame_footprints([_nadir(0.0, 0.0, height)], [intr], 0.0)

    assert footprint is not None
    # Ground width = sensor width * height / fx, straight from similar triangles.
    expected = intr.width * height / intr.fx
    assert np.ptp(footprint[:, 0]) == pytest.approx(expected, rel=1e-9)
    assert np.ptp(footprint[:, 1]) == pytest.approx(intr.height * height / intr.fx, rel=1e-9)
    # Centred under the camera.
    assert footprint.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_footprint_scales_with_height_above_ground() -> None:
    from drishti3d.export.placement import frame_footprints

    intr = _intr()
    low, high = frame_footprints([_nadir(0, 0, 60.0), _nadir(0, 0, 120.0)], [intr, intr], 0.0)
    assert np.ptp(high[:, 0]) == pytest.approx(2.0 * np.ptp(low[:, 0]), rel=1e-9)


def test_a_camera_below_or_at_the_ground_is_refused() -> None:
    from drishti3d.export.placement import frame_footprints

    assert frame_footprints([_nadir(0, 0, 120.0)], [_intr()], 200.0) == [None]
    assert frame_footprints([None], [_intr()], 0.0) == [None]


def test_upward_facing_camera_never_meets_the_ground() -> None:
    from drishti3d.export.placement import frame_footprints
    from drishti3d.types import Pose

    up = Pose(R=np.eye(3), t=np.array([0.0, 0.0, 120.0]))  # camera +Z forward = world +Z
    assert frame_footprints([up], [_intr()], 0.0) == [None]


def test_overlap_metrics_track_how_far_the_drone_moved() -> None:
    from drishti3d.export.placement import frame_footprints, frame_placement_metrics

    intr = _intr()
    # Footprint is 4000 * 120 / 2700 = 177.8 m wide. Stepping 17.8 m is
    # 90% overlap; stepping 177.8 m is none.
    for step, expected in ((17.78, 90.0), (177.8, 0.0)):
        poses = [_nadir(i * step, 0.0) for i in range(8)]
        cams = np.array([p.t for p in poses])
        m = frame_placement_metrics(frame_footprints(poses, [intr] * 8, 0.0), cams)
        assert m["consecutive_overlap_pct"] == pytest.approx(expected, abs=1.0), (step, m)


def test_a_gap_in_the_flight_shows_as_single_coverage() -> None:
    from drishti3d.export.placement import frame_footprints, frame_placement_metrics

    intr = _intr()
    # Two dense clusters far apart: the middle is covered by nobody and
    # each cluster's edges by only one frame.
    poses = [_nadir(i * 20.0, 0.0) for i in range(5)] + [_nadir(2000.0 + i * 20.0, 0.0) for i in range(5)]
    cams = np.array([p.t for p in poses])
    m = frame_placement_metrics(frame_footprints(poses, [intr] * len(poses), 0.0), cams)
    assert m["cells_seen_once"] > 0
    assert m["overlap_min"] == 1


def test_pose_vs_gps_residual_is_reported() -> None:
    from drishti3d.export.placement import frame_footprints, frame_placement_metrics

    intr = _intr()
    poses = [_nadir(i * 20.0, 0.0) for i in range(6)]
    cams = np.array([p.t for p in poses])
    gps = cams + np.array([3.0, 0.0, 0.0])  # a uniform 3 m offset
    m = frame_placement_metrics(frame_footprints(poses, [intr] * 6, 0.0), cams, gps_positions=gps)
    assert m["pose_vs_gps_median_m"] == pytest.approx(3.0, rel=1e-6)


def test_write_frame_placement_produces_the_sheet(tmp_path) -> None:
    from drishti3d.export.placement import write_frame_placement

    poses = [_nadir(i * 20.0, (i % 3) * 20.0) for i in range(12)]
    metrics = write_frame_placement(tmp_path, poses, [_intr()] * 12, ground_z=0.0)

    assert metrics["applied"]
    assert metrics["frames_placed"] == 12
    for name in ("frames_topdown.png", "frames_elevation.png", "frame_placement.png", "frame_placement.json"):
        assert (tmp_path / name).exists(), name
    assert json.loads((tmp_path / "frame_placement.json").read_text())["frames_placed"] == 12


def test_write_frame_placement_refuses_without_altitude(tmp_path) -> None:
    from drishti3d.export.placement import write_frame_placement

    metrics = write_frame_placement(tmp_path, [_nadir(0, 0)], [_intr()], ground_z=None)
    assert metrics["applied"] is False
    assert "altitude" in metrics["reason"]

    metrics = write_frame_placement(tmp_path, [None, None], [_intr()], ground_z=0.0)
    assert metrics["applied"] is False
    assert "poses" in metrics["reason"]


# ---------------------------------------------------------------------------
# attributing the error: anchor scale vs merge scale
# ---------------------------------------------------------------------------


def test_merge_scale_is_measured_from_the_cameras() -> None:
    """The merge's own rescale, recovered without touching merge_submaps.

    A similarity scales every baseline by the same factor, so comparing
    a submap's local camera baselines against the same cameras' merged
    positions recovers it exactly.
    """
    from drishti3d.geometry.windows import Window
    from drishti3d.pipeline.stages import _merge_scale_per_submap
    from drishti3d.types import PointCloud, Pose, Submap

    def submap(index: int, kfs: list[int], step: float) -> Submap:
        poses = [Pose(R=np.eye(3), t=np.array([i * step, 0.0, 100.0])) for i, _ in enumerate(kfs)]
        return Submap(
            window=Window(index=index, start=kfs[0], end=kfs[-1] + 1, shared_with_previous=[]),
            poses=poses,
            points=PointCloud(xyz=np.zeros((10, 3))),
            confidence=np.ones(10),
            keyframe_indices=kfs,
            local_origin=poses[0],
        )

    # Merged cameras step 20 m apart. Submap 0's local cameras step 20 m
    # (merge scale 1.0), submap 1's step 4 m (merge scale 5.0).
    merged = [Pose(R=np.eye(3), t=np.array([i * 20.0, 0.0, 120.0])) for i in range(8)]
    scales = _merge_scale_per_submap([submap(0, [0, 1, 2, 3], 20.0), submap(1, [4, 5, 6, 7], 4.0)], merged)

    assert scales[0] == pytest.approx(1.0, rel=1e-9)
    assert scales[1] == pytest.approx(5.0, rel=1e-9)


def test_merge_scale_skips_submaps_it_cannot_measure() -> None:
    from drishti3d.geometry.windows import Window
    from drishti3d.pipeline.stages import _merge_scale_per_submap
    from drishti3d.types import PointCloud, Pose, Submap

    stationary = Submap(
        window=Window(index=0, start=0, end=2, shared_with_previous=[]),
        # Both cameras at the same place: no baseline, so no scale.
        poses=[Pose(R=np.eye(3), t=np.zeros(3)), Pose(R=np.eye(3), t=np.zeros(3))],
        points=PointCloud(xyz=np.zeros((4, 3))),
        confidence=np.ones(4),
        keyframe_indices=[0, 1],
        local_origin=Pose(R=np.eye(3), t=np.zeros(3)),
    )
    merged = [Pose(R=np.eye(3), t=np.array([i * 20.0, 0.0, 120.0])) for i in range(2)]
    assert _merge_scale_per_submap([stationary], merged) == {}
