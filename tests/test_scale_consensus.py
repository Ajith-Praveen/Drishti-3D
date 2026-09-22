"""Tests for putting every window on one depth scale -- and for knowing when not to.

On the sample flight the per-window depth-anchor ratios came out between
1.16 and 6.79, for windows of the same field, at a constant 119.9 m
altitude, through the same backbone. The first version of this module
took the median of that and snapped every window onto it, which made the
run worse: the windows near the median landed within 3-7 m of the GPS
ground, and the four windows that had measured ~1.2 were pushed 32-41 m
below it.

So these tests cover both halves. Scatter around a single scale is
measurement noise and must be harmonised away. Two separated clusters
are not noise -- they are the backbone genuinely disagreeing with itself
-- and a median of them is just the bigger cluster wearing a disguise,
so the only correct action is to refuse, leave every window alone, and
report it.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.geometry.scale_consensus import consensus_ratio, harmonise_submap_scales
from drishti3d.geometry.windows import Window
from drishti3d.types import PointCloud, Pose, Submap


def _diag(window_ratio: float, gps: float | None = None, parallax: float | None = None) -> dict:
    d: dict = {"applied": True, "window_ratio": window_ratio}
    if gps is not None:
        d["gps_altitude"] = {"ratio": gps, "views": 4}
    if parallax is not None:
        d["parallax"] = {"ratio": parallax, "samples": 200}
    return d


def _submap(index: int, *, n: int = 200, origin_z: float = 100.0) -> Submap:
    rng = np.random.default_rng(index)
    origin = Pose(R=np.eye(3), t=np.array([0.0, 0.0, origin_z]))
    # Ground 100 m below the camera, which is what a correct scale means.
    pts = np.c_[rng.uniform(-20, 20, (n, 2)), np.zeros(n)]
    return Submap(
        window=Window(index=index, start=index * 2, end=index * 2 + 4, shared_with_previous=[]),
        poses=[origin, Pose(R=np.eye(3), t=np.array([5.0, 0.0, origin_z]))],
        points=PointCloud(xyz=pts),
        confidence=np.ones(n),
        keyframe_indices=[index * 2, index * 2 + 1],
        local_origin=origin,
        # Which camera saw each point. Required: the correction is
        # applied along each point's own view ray, so that the merge's
        # camera-baseline scale fit cannot cancel it.
        view_index=np.zeros(n, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# the consensus itself
# ---------------------------------------------------------------------------


def test_consensus_ignores_outliers() -> None:
    """A single stray window among a tight cluster is overruled by it."""
    diags = {
        0: _diag(5.8, gps=5.8), 1: _diag(6.0, gps=6.0), 2: _diag(5.9, gps=5.9),
        3: _diag(4.6, gps=4.6), 4: _diag(6.7, gps=6.7), 5: _diag(6.1, gps=6.1),
    }
    c = consensus_ratio(diags)
    assert c is not None
    assert 5.5 < c.ratio < 6.5, c.as_dict()
    assert c.source == "gps_altitude"


def test_consensus_refuses_two_clusters() -> None:
    """The sample flight's real shape: 1.16-1.33 and 5.3-6.8, no middle.

    A median of this is 6.07, which is not a compromise -- it is simply
    the larger cluster. Applying it pushed the small-ratio windows tens
    of metres below the ground. Refusing is the only correct answer.
    """
    measured = [1.159, 1.239, 1.202, 1.329, 6.369, 5.805, 6.353, 6.084, 6.320, 6.525]
    diags = {i: _diag(r, gps=r) for i, r in enumerate(measured)}
    assert consensus_ratio(diags) is None

    submaps = [_submap(i) for i in range(len(measured))]
    originals = [sm.points.xyz.copy() for sm in submaps]
    diagnostics = harmonise_submap_scales(submaps, diags)

    assert diagnostics["applied"] is False
    assert "span" in diagnostics["reason"]
    assert diagnostics["input_spread"] == pytest.approx(6.525 / 1.159, rel=1e-3)
    # Every window keeps exactly the scale it measured for itself.
    for sm, original in zip(submaps, originals, strict=True):
        assert np.array_equal(sm.points.xyz, original)


def test_consensus_prefers_gps_but_falls_back_to_parallax() -> None:
    gps_only = {i: _diag(3.0, gps=3.0) for i in range(4)}
    assert consensus_ratio(gps_only).source == "gps_altitude"

    par_only = {i: _diag(3.0, parallax=3.0) for i in range(4)}
    assert consensus_ratio(par_only).source == "parallax"


def test_consensus_refuses_when_too_few_windows_measured() -> None:
    assert consensus_ratio({0: _diag(5.0, gps=5.0), 1: _diag(5.1, gps=5.1)}) is None
    assert consensus_ratio({}) is None


# ---------------------------------------------------------------------------
# applying it
# ---------------------------------------------------------------------------


def _stacked(ratios: list[float], consensus: float = 6.0):
    """Submaps carrying the scale error their own ratio implies."""
    submaps = [_submap(i) for i in range(len(ratios))]
    for i, r in enumerate(ratios):
        origin = np.asarray(submaps[i].local_origin.t)
        submaps[i].points = PointCloud(xyz=origin + (r / consensus) * (submaps[i].points.xyz - origin))
    return submaps


def _ground_spread(submaps) -> float:
    return float(np.ptp([np.median(sm.points.xyz[:, 2]) for sm in submaps]))


def test_scatter_around_one_scale_is_harmonised_away() -> None:
    """Measurement scatter within a single cluster is what this can fix."""
    ratios = [5.8, 6.0, 5.9, 5.1, 6.7, 6.1]
    diags = {i: _diag(r, gps=r) for i, r in enumerate(ratios)}
    submaps = _stacked(ratios)

    before = _ground_spread(submaps)
    diagnostics = harmonise_submap_scales(submaps, diags)
    after = _ground_spread(submaps)

    assert diagnostics["applied"]
    # Every window lands on the consensus by default, so every window
    # whose own ratio differed at all is listed as corrected.
    assert {c["window"] for c in diagnostics["corrections"]} == set(range(6))
    # Ground heights now agree to well under the metre this project
    # targets; before, they spanned tens of metres.
    assert after < 0.01 < 1.0 < before, (before, after)


def test_a_tolerance_is_paid_for_in_metres() -> None:
    """Why the default snaps everything rather than keeping close-enough windows.

    The same six windows, harmonised with a 10% tolerance, still end up
    metres apart -- because 10% of a 100 m ground distance is 10 m.
    """
    ratios = [5.8, 6.0, 5.9, 5.1, 6.7, 6.1]
    diags = {i: _diag(r, gps=r) for i, r in enumerate(ratios)}
    submaps = _stacked(ratios)

    harmonise_submap_scales(submaps, diags, tolerance=1.10)
    assert _ground_spread(submaps) > 1.0


def test_two_clusters_are_left_alone_rather_than_averaged() -> None:
    """The sample flight, end to end: refusing must not move anything.

    Snapping these onto their median would put the four small-ratio
    windows five times too deep -- measurably worse than leaving them
    where their own measurement put them.
    """
    ratios = [1.16, 1.24, 1.20, 1.33, 6.37, 5.81, 6.35, 6.08, 6.32, 6.53]
    diags = {i: _diag(r, gps=r) for i, r in enumerate(ratios)}
    submaps = _stacked(ratios)
    before = _ground_spread(submaps)

    diagnostics = harmonise_submap_scales(submaps, diags)

    assert diagnostics["applied"] is False
    assert _ground_spread(submaps) == pytest.approx(before)


def test_windows_already_on_the_consensus_are_left_untouched() -> None:
    """Snapping must be a no-op when there is nothing to snap."""
    diags = {i: _diag(6.0, gps=6.0) for i in range(4)}
    submaps = [_submap(i) for i in range(4)]
    originals = [sm.points.xyz.copy() for sm in submaps]

    diagnostics = harmonise_submap_scales(submaps, diags)
    assert diagnostics["windows_rescaled"] == 0
    for sm, original in zip(submaps, originals, strict=True):
        assert np.array_equal(sm.points.xyz, original)


def test_an_explicit_tolerance_still_keeps_close_windows() -> None:
    diags = {i: _diag(r, gps=r) for i, r in enumerate([5.8, 6.0, 5.9, 6.05])}
    submaps = [_submap(i) for i in range(4)]
    originals = [sm.points.xyz.copy() for sm in submaps]

    diagnostics = harmonise_submap_scales(submaps, diags, tolerance=1.10)
    assert diagnostics["windows_rescaled"] == 0
    for sm, original in zip(submaps, originals, strict=True):
        assert np.array_equal(sm.points.xyz, original)


def test_rescale_is_a_similarity_so_shape_is_preserved() -> None:
    """Scaling must not deform the window -- the merge relies on that."""
    diags = {i: _diag(r, gps=r) for i, r in enumerate([6.0, 6.0, 6.0, 4.5])}
    submaps = [_submap(i) for i in range(4)]
    target = submaps[3]
    before = target.points.xyz.copy()

    harmonise_submap_scales(submaps, diags)

    # Every pairwise distance scaled by exactly one factor.
    d_before = np.linalg.norm(before[1:] - before[0], axis=1)
    d_after = np.linalg.norm(target.points.xyz[1:] - target.points.xyz[0], axis=1)
    factors = d_after / np.maximum(d_before, 1e-12)
    assert np.allclose(factors, factors[0], rtol=1e-9)
    assert factors[0] == pytest.approx(6.0 / 4.5, rel=1e-6)


def test_camera_translations_do_not_move() -> None:
    """The correction must survive the merge, so baselines cannot change.

    ``merge_submaps`` fits each submap's scale from its camera baselines.
    A correction that moved the cameras too would be a pure similarity
    and that fit would cancel it exactly -- measured: two runs differing
    only in whether this ran produced byte-identical output.
    """
    diags = {i: _diag(r, gps=r) for i, r in enumerate([4.0, 4.0, 4.0, 3.0])}
    submaps = [_submap(i) for i in range(4)]
    target = submaps[3]
    baseline_before = float(np.linalg.norm(np.asarray(target.poses[1].t) - np.asarray(target.poses[0].t)))

    harmonise_submap_scales(submaps, diags)

    baseline_after = float(np.linalg.norm(np.asarray(target.poses[1].t) - np.asarray(target.poses[0].t)))
    assert baseline_after == pytest.approx(baseline_before, rel=1e-12)


def test_covariance_scales_quadratically() -> None:
    diags = {i: _diag(r, gps=r) for i, r in enumerate([3.0, 3.0, 3.0, 2.4])}
    submaps = [_submap(i) for i in range(4)]
    n = submaps[3].points.xyz.shape[0]
    submaps[3].points = PointCloud(
        xyz=submaps[3].points.xyz, covariance=np.tile(np.eye(3), (n, 1, 1))
    )

    harmonise_submap_scales(submaps, diags)
    assert submaps[3].points.covariance[0, 0, 0] == pytest.approx((3.0 / 2.4) ** 2, rel=1e-9)


def test_unanchored_window_receives_the_consensus_outright() -> None:
    diags = {i: _diag(6.0, gps=6.0) for i in range(3)}
    diags[3] = {"applied": False, "failure": "too few samples"}
    submaps = [_submap(i) for i in range(4)]
    before = submaps[3].points.xyz.copy()

    diagnostics = harmonise_submap_scales(submaps, diags)
    correction = next(c for c in diagnostics["corrections"] if c["window"] == 3)
    assert correction["reason"] == "unanchored"
    assert correction["correction"] == pytest.approx(6.0, rel=1e-6)
    assert not np.array_equal(submaps[3].points.xyz, before)


def test_no_consensus_leaves_everything_alone() -> None:
    submaps = [_submap(i) for i in range(2)]
    originals = [sm.points.xyz.copy() for sm in submaps]
    diagnostics = harmonise_submap_scales(submaps, {0: _diag(5.0, gps=5.0)})
    assert diagnostics["applied"] is False
    for sm, original in zip(submaps, originals, strict=True):
        assert np.array_equal(sm.points.xyz, original)


# ---------------------------------------------------------------------------
# the anchor must no longer deform a window per view
# ---------------------------------------------------------------------------


def test_anchor_applies_one_scale_to_every_view() -> None:
    """Per-view scaling is a deformation no Sim(3) merge can undo.

    Two views look straight down at the same flat ground from the same
    height. Whatever ratio is measured, both views' ground must end up at
    the same Z -- otherwise the window is internally thick before it
    reaches the merge.
    """
    from dataclasses import dataclass, field

    from drishti3d.geometry.depth_anchor import anchor_depth_fused
    from drishti3d.types import CameraIntrinsics

    h = w = 32

    @dataclass
    class _Result:
        depth: np.ndarray
        points: np.ndarray
        poses: list
        intrinsics: list
        images: np.ndarray | None = None
        confidence: np.ndarray | None = None
        metadata: dict = field(default_factory=dict)

    # Both cameras at 100 m; the backbone thinks the ground is 25 m down.
    poses = [
        Pose(R=np.eye(3), t=np.array([0.0, 0.0, 100.0])),
        Pose(R=np.eye(3), t=np.array([8.0, 0.0, 100.0])),
    ]
    intr = [CameraIntrinsics(fx=500.0, fy=500.0, cx=w / 2, cy=h / 2, width=w, height=h)] * 2
    depth = np.full((2, h, w), 25.0)
    points = np.zeros((2, h, w, 3))
    for v, p in enumerate(poses):
        points[v, ..., 0] = p.t[0]
        points[v, ..., 1] = p.t[1]
        points[v, ..., 2] = p.t[2] - 25.0

    images = np.zeros((2, h, w, 3), dtype=np.uint8)
    result = _Result(depth=depth, points=points, poses=poses, intrinsics=intr, images=images)

    anchored, diag = anchor_depth_fused(images, result, poses, intr, [100.0, 100.0], min_samples=10)

    assert diag.get("applied"), diag
    # One scale for the window: every view's ground lands at one height.
    ground_z = [float(np.median(anchored.points[v, ..., 2])) for v in range(2)]
    assert abs(ground_z[0] - ground_z[1]) < 1e-6, ground_z
    # And the scale is the one the altitude implies: 100 m / 25 m = 4.
    assert diag["window_ratio"] == pytest.approx(4.0, rel=0.05)


def test_correction_survives_a_merge_that_refits_scale() -> None:
    """The regression that byte-identical v15/v16 output should have caught.

    ``merge_submaps`` fits each submap's scale from its camera baselines.
    An earlier version scaled points AND cameras about a common origin --
    a pure similarity -- so that fit cancelled the whole correction and
    two runs differing only in whether this module applied produced
    identical output. Scaling about each point's own camera changes the
    ground's depth below the cameras while leaving every baseline alone,
    which a camera-based scale fit cannot undo.
    """
    diags = {i: _diag(r, gps=r) for i, r in enumerate([6.0, 6.0, 6.0, 4.5])}
    submaps = [_submap(i) for i in range(4)]
    target = submaps[3]

    def depth_below_cameras(sm) -> float:
        camera_z = float(np.median([p.t[2] for p in sm.poses]))
        return camera_z - float(np.median(sm.points.xyz[:, 2]))

    before = depth_below_cameras(target)
    harmonise_submap_scales(submaps, diags)
    after = depth_below_cameras(target)

    # Depth below the cameras is what a depth-scale error actually is,
    # and it is the quantity that must change.
    assert after == pytest.approx(before * (6.0 / 4.5), rel=1e-9)

    # Simulate the merge's own scale fit: it solves scale from baselines,
    # which this correction left untouched, so it finds 1.0 and cancels
    # nothing.
    baselines = np.linalg.norm(np.asarray(target.poses[1].t) - np.asarray(target.poses[0].t))
    assert baselines == pytest.approx(
        np.linalg.norm(np.asarray(submaps[0].poses[1].t) - np.asarray(submaps[0].poses[0].t))
    )


def test_submap_without_a_view_index_is_refused_not_guessed() -> None:
    """A wrong centre turns a depth correction into a translation."""
    diags = {i: _diag(r, gps=r) for i, r in enumerate([6.0, 6.0, 6.0, 4.5])}
    submaps = [_submap(i) for i in range(4)]
    submaps[3].view_index = None
    before = submaps[3].points.xyz.copy()

    diagnostics = harmonise_submap_scales(submaps, diags)

    assert np.array_equal(submaps[3].points.xyz, before)
    assert 3 not in {c["window"] for c in diagnostics["corrections"]}
