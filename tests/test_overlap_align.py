"""Tests for geometry.overlap_align.

The scenario is the measured failure: two windows that reconstruct the
same ground correctly but are placed at different heights, because the
camera-centre Sim(3) that placed them is nearly unconstrained vertically.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.geometry.overlap_align import refine_overlap_translations
from drishti3d.geometry.submap import Sim3
from drishti3d.geometry.windows import Window
from drishti3d.types import PointCloud, Pose, Submap

_NADIR_R = np.diag([1.0, -1.0, -1.0])


def _submap(index, kfs, shared, x0, n_per_view=800, seed=0, ground_z=0.0):
    """A window over flat ground, with per-point view indices."""
    rng = np.random.default_rng(seed)
    poses = [Pose(R=_NADIR_R, t=np.array([x0 + 25.0 * j, 0.0, 120.0])) for j in range(len(kfs))]
    xyz, vidx = [], []
    for v, p in enumerate(poses):
        pts = np.column_stack([
            rng.uniform(p.t[0] - 40, p.t[0] + 40, n_per_view),
            rng.uniform(-40, 40, n_per_view),
            np.full(n_per_view, ground_z) + rng.normal(0, 0.05, n_per_view),
        ])
        xyz.append(pts)
        vidx.append(np.full(n_per_view, v, dtype=np.int32))
    xyz = np.concatenate(xyz)
    w = Window(index=index, start=kfs[0], end=kfs[-1] + 1)
    w.shared_with_previous = list(shared)
    return Submap(
        window=w,
        poses=poses,
        points=PointCloud(xyz=xyz, confidence=np.full(len(xyz), 2, np.uint8)),
        confidence=np.ones(len(xyz)),
        keyframe_indices=list(kfs),
        local_origin=poses[0],
        view_index=np.concatenate(vidx),
    )


def _identity(n):
    return [Sim3(scale=1.0, R=np.eye(3), t=np.zeros(3)) for _ in range(n)]


def test_closes_a_vertical_offset_between_windows():
    """The measured failure: same ground, two heights.

    This is what a near-coplanar camera-centre Sim(3) cannot see, and what
    left the fused cloud a 14.8 m-thick slab of duplicated surfaces.
    """
    a = _submap(0, [0, 1, 2, 3], [], x0=0.0, seed=1)
    b = _submap(1, [2, 3, 4, 5], [2, 3], x0=50.0, seed=2)
    t = _identity(2)
    t[1] = Sim3(scale=1.0, R=np.eye(3), t=np.array([0.0, 0.0, 9.0]))  # 9 m too high

    out, diag = refine_overlap_translations([a, b], t)

    assert diag["junctions_corrected"] == 1, diag
    # The correction must cancel the 9 m, not merely reduce it.
    assert out[1].t[2] == pytest.approx(0.0, abs=0.5), f"residual {out[1].t[2]:.2f} m"
    assert diag["median_vertical_offset_m"] == pytest.approx(9.0, abs=0.5)


def test_leaves_an_already_aligned_pair_alone():
    a = _submap(0, [0, 1, 2, 3], [], x0=0.0, seed=1)
    b = _submap(1, [2, 3, 4, 5], [2, 3], x0=50.0, seed=2)
    out, diag = refine_overlap_translations([a, b], _identity(2))
    assert np.linalg.norm(out[1].t) < 0.5, f"moved an aligned pair by {np.linalg.norm(out[1].t):.2f} m"


def test_refuses_an_implausibly_large_correction():
    """Beyond a sane range the windows are not the same place."""
    a = _submap(0, [0, 1, 2, 3], [], x0=0.0, seed=1)
    b = _submap(1, [2, 3, 4, 5], [2, 3], x0=50.0, seed=2)
    t = _identity(2)
    t[1] = Sim3(scale=1.0, R=np.eye(3), t=np.array([0.0, 0.0, 300.0]))
    out, diag = refine_overlap_translations([a, b], t)
    entry = diag["per_junction"][0]
    assert "refused" in entry or entry.get("failure"), entry
    assert out[1].t[2] == pytest.approx(300.0)  # untouched


def test_offsets_accumulate_along_the_chain():
    """Correcting window i only helps if i-1 is already corrected."""
    subs = [_submap(0, [0, 1, 2, 3], [], x0=0.0, seed=1)]
    subs.append(_submap(1, [2, 3, 4, 5], [2, 3], x0=50.0, seed=2))
    subs.append(_submap(2, [4, 5, 6, 7], [4, 5], x0=100.0, seed=3))
    t = _identity(3)
    t[1] = Sim3(scale=1.0, R=np.eye(3), t=np.array([0.0, 0.0, 5.0]))
    t[2] = Sim3(scale=1.0, R=np.eye(3), t=np.array([0.0, 0.0, 11.0]))

    out, diag = refine_overlap_translations(subs, t)

    assert diag["junctions_corrected"] == 2, diag
    grounds = [float(np.median(tr.apply(s.points.xyz.reshape(-1, 3))[:, 2])) for s, tr in zip(subs, out)]
    assert max(grounds) - min(grounds) < 1.0, f"windows still span {max(grounds)-min(grounds):.2f} m"


def test_skips_submaps_without_view_index():
    a = _submap(0, [0, 1, 2, 3], [], x0=0.0, seed=1)
    b = _submap(1, [2, 3, 4, 5], [2, 3], x0=50.0, seed=2)
    b.view_index = None
    out, diag = refine_overlap_translations([a, b], _identity(2))
    assert diag["junctions_corrected"] == 0
    assert "view index" in diag["per_junction"][0]["skipped"]


# ---------------------------------------------------------------------------
# Global solve
# ---------------------------------------------------------------------------


def _chain(n, drifts, seed0=1):
    """n windows over flat ground, each placed with a given vertical drift."""
    subs, ts = [], []
    for i in range(n):
        kfs = list(range(i * 2, i * 2 + 4))
        shared = list(range(i * 2, i * 2 + 2)) if i else []
        subs.append(_submap(i, kfs, shared, x0=50.0 * i, seed=seed0 + i))
        ts.append(Sim3(scale=1.0, R=np.eye(3), t=np.array([0.0, 0.0, drifts[i]])))
    return subs, ts


def test_global_solve_distributes_error_instead_of_accumulating_it():
    """The point of solving globally rather than walking the chain.

    Chained correction carries every junction's residual forward, so the
    far end of a flight inherits the sum of them. A global least-squares
    fit spreads that residual across the whole network.
    """
    from drishti3d.geometry.overlap_align import solve_overlap_translations

    drifts = [0.0, 4.0, 9.0, 15.0, 22.0, 30.0]
    subs, ts = _chain(6, drifts)

    out, diag = solve_overlap_translations(subs, ts)

    assert diag["junctions_measured"] >= 5, diag
    grounds = [float(np.median(t.apply(s.points.xyz.reshape(-1, 3))[:, 2])) for s, t in zip(subs, out)]
    span = max(grounds) - min(grounds)
    assert span < 1.0, f"windows still span {span:.2f} m (drifts were {drifts})"
    # Submap 0 is the gauge and must not move.
    assert out[0].t[2] == pytest.approx(0.0, abs=1e-6)


def test_global_solve_survives_one_unmeasurable_junction():
    """A broken link must not strand everything after it.

    Chaining cannot do this: an unmeasurable junction leaves its
    neighbour and every later window on a stale offset. A global solve
    routes around it through the other overlaps.
    """
    from drishti3d.geometry.overlap_align import solve_overlap_translations

    subs, ts = _chain(5, [0.0, 5.0, 11.0, 18.0, 26.0])
    # Break one window's per-point view index so its pairings vanish.
    subs[2] = _submap(2, [4, 5, 6, 7], [4, 5], x0=100.0, seed=99)
    subs[2].points.xyz[:] += np.array([0.0, 0.0, 11.0])
    ts[2] = Sim3(scale=1.0, R=np.eye(3), t=np.zeros(3))

    out, diag = solve_overlap_translations(subs, ts)
    grounds = [float(np.median(t.apply(s.points.xyz.reshape(-1, 3))[:, 2])) for s, t in zip(subs, out)]
    assert max(grounds) - min(grounds) < 2.0, f"span {max(grounds)-min(grounds):.2f} m"


def test_global_solve_leaves_an_aligned_network_alone():
    from drishti3d.geometry.overlap_align import solve_overlap_translations

    subs, ts = _chain(4, [0.0, 0.0, 0.0, 0.0])
    out, diag = solve_overlap_translations(subs, ts)
    assert diag["max_correction_m"] < 0.5, diag


def test_gps_levelling_puts_ground_at_camera_height_minus_altitude():
    """The merge's only ABSOLUTE vertical reference.

    Measured on real footage: GPS says the ground spans 5.1 m across the
    flight (level terrain, constant altitude); the reconstruction spanned
    50.0 m, drifting -20 m to +25 m. Overlap alignment ties windows to
    each other but cannot see that the whole network has drifted.
    """
    from drishti3d.geometry.overlap_align import level_submaps_to_gps

    alt = 120.0
    drifts = [0.0, 12.0, -9.0, 21.0]
    subs, ts = _chain(4, drifts)
    gps = {}
    alts = {}
    for s in subs:
        for kf, p in zip(s.keyframe_indices, s.poses):
            gps[kf] = np.array([p.t[0], p.t[1], alt])  # camera at z=alt over ground z=0
            alts[kf] = alt

    out, diag = level_submaps_to_gps(subs, ts, gps, alts)

    assert diag["applied"], diag
    assert diag["submaps_levelled"] == 4
    grounds = [float(np.percentile(t.apply(s.points.xyz.reshape(-1, 3))[:, 2], 10))
               for s, t in zip(subs, out)]
    # Every submap's ground now sits at ~0 (= camera_z - alt), so the
    # span collapses regardless of how each one had drifted.
    assert max(grounds) - min(grounds) < 1.0, f"span {max(grounds)-min(grounds):.2f} m from drifts {drifts}"
    assert abs(np.median(grounds)) < 1.0


def test_gps_levelling_refuses_an_implausible_correction():
    from drishti3d.geometry.overlap_align import level_submaps_to_gps

    subs, ts = _chain(2, [0.0, 400.0])
    gps, alts = {}, {}
    for s in subs:
        for kf, p in zip(s.keyframe_indices, s.poses):
            gps[kf] = np.array([p.t[0], p.t[1], 120.0]); alts[kf] = 120.0
    out, diag = level_submaps_to_gps(subs, ts, gps, alts)
    entry = [e for e in diag["per_submap"] if e["submap"] == 1][0]
    assert "refused" in entry, entry
    assert out[1].t[2] == pytest.approx(400.0)


def test_gps_levelling_needs_altitudes():
    from drishti3d.geometry.overlap_align import level_submaps_to_gps

    subs, ts = _chain(2, [0.0, 5.0])
    out, diag = level_submaps_to_gps(subs, ts, {}, {})
    assert not diag["applied"] and "no GPS" in diag["failure"]


def test_joint_solve_satisfies_overlap_AND_gps_together():
    """Applying the two corrections in sequence makes the spread worse.

    Measured when GPS levelling ran after the overlap solve: median error
    improved (8.5 -> 2.9 m) while the SPREAD degraded (50.0 -> 56.6 m),
    because each submap was moved independently and the overlap agreement
    it had just been given was destroyed. Solved jointly, both hold.
    """
    from drishti3d.geometry.overlap_align import solve_overlap_translations

    alt = 120.0
    drifts = [0.0, 9.0, -6.0, 15.0, 24.0]
    subs, ts = _chain(5, drifts)
    gps, alts = {}, {}
    for s in subs:
        for kf, p in zip(s.keyframe_indices, s.poses):
            gps[kf] = np.array([p.t[0], p.t[1], alt])
            alts[kf] = alt

    out, diag = solve_overlap_translations(
        subs, ts, camera_gps_enu=gps, keyframe_altitude_m=alts
    )

    assert diag["gps_constraints"] >= 4, diag
    grounds = [float(np.percentile(t.apply(s.points.xyz.reshape(-1, 3))[:, 2], 10))
               for s, t in zip(subs, out)]
    # Relative agreement (overlap) AND absolute placement (GPS) both hold.
    assert max(grounds) - min(grounds) < 1.5, f"span {max(grounds)-min(grounds):.2f} m"
    assert abs(np.median(grounds)) < 1.5, f"ground at {np.median(grounds):.2f} m, expected ~0"


def test_joint_solve_still_works_without_gps():
    """No GPS: falls back to pinning submap 0, as before."""
    from drishti3d.geometry.overlap_align import solve_overlap_translations

    subs, ts = _chain(4, [0.0, 7.0, 14.0, 20.0])
    out, diag = solve_overlap_translations(subs, ts)
    assert diag["gps_constraints"] == 0
    grounds = [float(np.median(t.apply(s.points.xyz.reshape(-1, 3))[:, 2])) for s, t in zip(subs, out)]
    assert max(grounds) - min(grounds) < 1.0
    assert out[0].t[2] == pytest.approx(0.0, abs=1e-6)
