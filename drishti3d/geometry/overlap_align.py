"""Refine submap placement using the ground they reconstruct in common.

The constraint the merge is missing
------------------------------------
``merge_submaps`` places each submap by fitting a Sim(3) to its camera
centres against GPS. On a survey flight those centres are a nearly flat
sheet: measured on real footage, 182 m x 141 m horizontally but only
8.3 m vertically -- a planarity ratio of 0.045. A similarity transform
fitted to near-coplanar points is well determined *in* the plane and
nearly unconstrained *perpendicular* to it, so every window's horizontal
placement is good (which is why top-down views look plausible) and its
vertical placement is free to drift.

The measured consequence: the same ground reconstructed by different
windows lands at different heights, and the fused cloud becomes a slab of
duplicated surfaces 14.8 m thick where a single ground surface should be
~0.1 m. That thickness is what forces a coarse TSDF voxel, which is what
leaves roofs as bumps.

Fixing the per-window depth scale (``geometry.depth_anchor``) does not fix
this. Depth anchoring makes the ground the right distance from *its own
camera*; two windows can each be internally perfect and still sit at
different heights relative to each other.

The constraint that is available
---------------------------------
Consecutive windows share keyframes -- ``Window.shared_with_previous``,
four of them at this project's overlap setting. They therefore reconstruct
the *same ground twice*. Those duplicate points are a direct measurement
of the offset between the two windows, and crucially they are strong in
exactly the axis the camera-centre fit is weak in: a horizontal ground
surface pins vertical offset extremely well, because sliding one copy up
relative to the other immediately separates them.

So: after the Sim(3) fit, measure the residual offset between each pair of
overlapping submaps from their shared points, and correct it.

Why translation only
--------------------
Rotation and scale are already well determined -- by the camera centres
horizontally, and by ``depth_anchor`` for scale. Re-solving them here
would let a noisy overlap region undo constraints that were measured
properly. Translation is the degree of freedom the camera-centre fit
genuinely fails to pin, so translation is the only one corrected.

Offsets are accumulated along the chain, because correcting window i
against window i-1 only helps if i-1 is itself already corrected.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from drishti3d.geometry.submap import Sim3
from drishti3d.types import Submap

logger = logging.getLogger(__name__)

__all__ = ["level_submaps_to_gps", "refine_overlap_translations", "solve_overlap_translations"]

#: Points sampled from each side of an overlap when measuring its offset.
#: The measurement is a robust median over correspondences; tens of
#: thousands buy nothing over a few thousand and cost KD-tree time.
_MAX_SAMPLES = 20000

#: A correspondence further apart than this (after the Sim(3) fit) is not
#: the same piece of ground. Generous, because the offsets being corrected
#: are metres in size, but finite so that genuinely unmatched geometry
#: does not drag the median.
_MAX_PAIR_DISTANCE_M = 25.0

#: Correspondences needed before an offset is believed.
_MIN_PAIRS = 200

#: Offsets larger than this are refused. At that size the two submaps are
#: not describing the same place, and "correcting" by it would move a
#: window somewhere arbitrary rather than onto its neighbour.
_MAX_OFFSET_M = 40.0


def _sample(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=limit, replace=False)]


def _measure_offset(
    fixed_pts: np.ndarray,
    moving_pts: np.ndarray,
    *,
    max_distance: float,
    min_pairs: int,
    iterations: int = 3,
) -> tuple[np.ndarray | None, dict]:
    """Translation carrying ``moving_pts`` onto ``fixed_pts``.

    A few rounds of nearest-neighbour matching and median offset --
    translation-only ICP. The median (not the mean) is what makes this
    robust: an overlap region always contains some geometry only one side
    reconstructed, and those points have no true correspondence.
    """
    if len(fixed_pts) < min_pairs or len(moving_pts) < min_pairs:
        return None, {"failure": f"too few points ({len(fixed_pts)} vs {len(moving_pts)})"}

    tree = cKDTree(fixed_pts)
    offset = np.zeros(3)
    pairs = 0
    for _ in range(iterations):
        dist, idx = tree.query(moving_pts + offset, k=1, distance_upper_bound=max_distance)
        good = np.isfinite(dist)
        pairs = int(good.sum())
        if pairs < min_pairs:
            return None, {"failure": f"only {pairs} correspondences within {max_distance} m"}
        delta = np.median(fixed_pts[idx[good]] - (moving_pts[good] + offset), axis=0)
        offset = offset + delta
        if float(np.linalg.norm(delta)) < 0.01:
            break

    dist, idx = tree.query(moving_pts + offset, k=1, distance_upper_bound=max_distance)
    good = np.isfinite(dist)
    residual = float(np.median(dist[good])) if good.any() else float("inf")
    return offset, {"pairs": pairs, "residual_m": round(residual, 3)}


def refine_overlap_translations(
    submaps: list[Submap],
    transforms: list[Sim3],
    *,
    max_samples: int = _MAX_SAMPLES,
    max_distance: float = _MAX_PAIR_DISTANCE_M,
    min_pairs: int = _MIN_PAIRS,
    max_offset_m: float = _MAX_OFFSET_M,
) -> tuple[list[Sim3], dict]:
    """Correct each submap's translation so its overlap matches its predecessor.

    Returns ``(transforms, diag)`` with new ``Sim3`` objects; rotation and
    scale are carried through untouched. A junction that cannot be
    measured leaves its submap on the chain's accumulated offset rather
    than inventing a correction, and says so.
    """
    if len(submaps) < 2:
        return transforms, {"junctions": 0, "note": "nothing to align"}

    out = list(transforms)
    cumulative = np.zeros(3)
    per_junction: list[dict] = []
    applied = 0

    for i in range(1, len(submaps)):
        prev, curr = submaps[i - 1], submaps[i]
        entry: dict = {"junction": (i - 1, i)}

        # Only the keyframes the two windows genuinely share. Using the
        # whole submap would match unrelated ground that merely happens to
        # be nearby and drag the offset toward zero.
        shared = list(curr.window.shared_with_previous or [])
        vidx_prev = getattr(prev, "view_index", None)
        vidx_curr = getattr(curr, "view_index", None)
        if not shared or vidx_prev is None or vidx_curr is None:
            entry["skipped"] = "no shared keyframes or no per-point view index"
            per_junction.append(entry)
            out[i] = Sim3(scale=out[i].scale, R=out[i].R, t=out[i].t + cumulative)
            continue

        prev_local = {kf: j for j, kf in enumerate(prev.keyframe_indices)}
        curr_local = {kf: j for j, kf in enumerate(curr.keyframe_indices)}
        prev_views = [prev_local[kf] for kf in shared if kf in prev_local]
        curr_views = [curr_local[kf] for kf in shared if kf in curr_local]
        if not prev_views or not curr_views:
            entry["skipped"] = "shared keyframes not present in both submaps"
            per_junction.append(entry)
            out[i] = Sim3(scale=out[i].scale, R=out[i].R, t=out[i].t + cumulative)
            continue

        prev_mask = np.isin(vidx_prev, prev_views)
        curr_mask = np.isin(vidx_curr, curr_views)
        # Both sides in the SAME global frame, including everything the
        # chain has corrected so far.
        fixed = out[i - 1].apply(prev.points.xyz.reshape(-1, 3)[prev_mask])
        moving_transform = Sim3(scale=out[i].scale, R=out[i].R, t=out[i].t + cumulative)
        moving = moving_transform.apply(curr.points.xyz.reshape(-1, 3)[curr_mask])

        offset, info = _measure_offset(
            _sample(fixed, max_samples, seed=i),
            _sample(moving, max_samples, seed=1000 + i),
            max_distance=max_distance,
            min_pairs=min_pairs,
        )
        entry.update(info)

        if offset is None:
            out[i] = moving_transform
            per_junction.append(entry)
            continue

        magnitude = float(np.linalg.norm(offset))
        entry["offset_m"] = round(magnitude, 3)
        entry["offset_z_m"] = round(float(offset[2]), 3)
        if magnitude > max_offset_m:
            entry["refused"] = f"offset {magnitude:.1f} m exceeds {max_offset_m} m"
            logger.warning(
                "overlap align: junction (%d, %d) wants a %.1f m correction -- refusing. "
                "At that size the two windows are not describing the same place.",
                i - 1,
                i,
                magnitude,
            )
            out[i] = moving_transform
            per_junction.append(entry)
            continue

        cumulative = cumulative + offset
        out[i] = Sim3(scale=out[i].scale, R=out[i].R, t=out[i].t + cumulative)
        applied += 1
        per_junction.append(entry)

    measured = [e["offset_m"] for e in per_junction if "offset_m" in e and "refused" not in e]
    z_offsets = [e["offset_z_m"] for e in per_junction if "offset_z_m" in e and "refused" not in e]
    diag = {
        "junctions": len(submaps) - 1,
        "junctions_corrected": applied,
        "median_offset_m": round(float(np.median(measured)), 3) if measured else None,
        "max_offset_m": round(float(np.max(measured)), 3) if measured else None,
        "median_vertical_offset_m": round(float(np.median(np.abs(z_offsets))), 3) if z_offsets else None,
        "per_junction": per_junction,
    }
    if measured:
        logger.info(
            "overlap align: corrected %d/%d junctions from shared geometry "
            "(median offset %.2f m, vertical %.2f m, max %.2f m). The camera-centre Sim(3) "
            "cannot see these -- the centres are a near-flat sheet, so vertical is its weak axis.",
            applied,
            len(submaps) - 1,
            diag["median_offset_m"],
            diag["median_vertical_offset_m"],
            diag["max_offset_m"],
        )
    else:
        logger.warning("overlap align: no junction could be measured from shared geometry")
    return out, diag


# ---------------------------------------------------------------------------
# Global solve
# ---------------------------------------------------------------------------

#: Windows further apart than this in index are not tested for overlap.
#: Consecutive windows share keyframes by construction; a survey flight
#: that loops can also overlap windows a few apart, and those constraints
#: are what stop error accumulating along the chain. Beyond this the pairs
#: cost KD-tree time and almost never share ground.
_MAX_JUNCTION_SPAN = 3


#: Weight on a GPS ground-height constraint relative to an overlap
#: constraint measured from one correspondence. Overlap rows are weighted
#: by sqrt(pairs) -- typically ~140 for 20,000 correspondences -- so this
#: makes one GPS row worth roughly a third of one well-measured overlap.
#: Deliberately not dominant: alt_rel is height above TAKEOFF, so on
#: sloping ground it is biased, and it should inform the solve without
#: overriding geometry that was measured directly.
_GPS_CONSTRAINT_WEIGHT = 50.0


def solve_overlap_translations(
    submaps: list[Submap],
    transforms: list[Sim3],
    *,
    camera_gps_enu: dict | None = None,
    keyframe_altitude_m: dict | None = None,
    max_samples: int = _MAX_SAMPLES,
    max_distance: float = _MAX_PAIR_DISTANCE_M,
    min_pairs: int = _MIN_PAIRS,
    max_offset_m: float = _MAX_OFFSET_M,
    max_span: int = _MAX_JUNCTION_SPAN,
) -> tuple[list[Sim3], dict]:
    """Solve ALL submap translations at once against every overlap constraint.

    Why not chain pairwise corrections
    -----------------------------------
    ``refine_overlap_translations`` walks the chain, correcting window i
    against window i-1 and carrying the offset forward. Measured on real
    footage that took surface thickness from 14.8 m to 6.9 m -- real
    progress, but it leaves two structural problems:

    - **Error accumulates.** Every junction's residual (~2.6 m median)
      propagates to every later window, so the far end of the flight
      inherits the sum of all of them.
    - **A junction that cannot be measured breaks the chain**, stranding
      its neighbour and everything after it.

    Solving globally fixes both. Each overlap gives one equation::

        t_j - t_i = offset_ij

    over all submap translations at once. That is a sparse linear
    least-squares problem -- no iteration, no ordering, and residual error
    is distributed evenly across the network instead of piling up at one
    end. Non-consecutive overlaps (a flight that loops back over ground it
    already covered) add constraints that tie distant parts of the chain
    together, which a sequential walk cannot use at all.

    Gauge: submap 0 is held fixed. The camera-centre Sim(3) already placed
    the whole network against GPS; this only removes the RELATIVE
    disagreement between windows, and pinning one window keeps the
    absolute georeferencing that fit established.

    Each equation is weighted by ``sqrt(pairs)`` -- an offset measured
    from 20,000 correspondences deserves more say than one from 200.
    """
    n = len(submaps)
    if n < 2:
        return transforms, {"junctions": 0, "note": "nothing to align"}

    rows: list[tuple[int, int, np.ndarray, float, dict]] = []
    per_junction: list[dict] = []

    for i in range(n):
        for j in range(i + 1, min(i + 1 + max_span, n)):
            entry: dict = {"junction": (i, j)}
            a, b = submaps[i], submaps[j]
            va, vb = getattr(a, "view_index", None), getattr(b, "view_index", None)
            if va is None or vb is None:
                continue
            # Keyframes the two windows genuinely have in common.
            common = sorted(set(a.keyframe_indices) & set(b.keyframe_indices))
            if not common:
                continue
            a_local = {kf: k for k, kf in enumerate(a.keyframe_indices)}
            b_local = {kf: k for k, kf in enumerate(b.keyframe_indices)}
            a_views = [a_local[kf] for kf in common]
            b_views = [b_local[kf] for kf in common]

            fixed = transforms[i].apply(a.points.xyz.reshape(-1, 3)[np.isin(va, a_views)])
            moving = transforms[j].apply(b.points.xyz.reshape(-1, 3)[np.isin(vb, b_views)])
            offset, info = _measure_offset(
                _sample(fixed, max_samples, seed=i * 97 + j),
                _sample(moving, max_samples, seed=i * 131 + j + 7),
                max_distance=max_distance,
                min_pairs=min_pairs,
            )
            entry.update(info)
            if offset is None:
                per_junction.append(entry)
                continue
            magnitude = float(np.linalg.norm(offset))
            entry["offset_m"] = round(magnitude, 3)
            entry["offset_z_m"] = round(float(offset[2]), 3)
            if magnitude > max_offset_m:
                entry["refused"] = f"offset {magnitude:.1f} m exceeds {max_offset_m} m"
                per_junction.append(entry)
                continue
            rows.append((i, j, offset, float(np.sqrt(max(info.get("pairs", 1), 1))), entry))
            per_junction.append(entry)

    if not rows:
        logger.warning("overlap solve: no measurable overlap; keeping the camera-centre fit")
        return transforms, {"junctions": 0, "per_junction": per_junction, "failure": "no measurable overlap"}

    # Sparse least squares: one row per constraint per axis, plus a gauge
    # row pinning submap 0.
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr

    # Absolute GPS ground-height rows, solved JOINTLY with the overlap
    # rows rather than applied afterwards.
    #
    # The two constraints answer different questions -- overlaps say where
    # windows sit relative to EACH OTHER, GPS says where the ground
    # actually IS -- and applying them in sequence means the second
    # silently undoes the first. Measured when they were sequential: GPS
    # levelling improved the median error (8.5 -> 2.9 m) while making the
    # spread WORSE (50.0 -> 56.6 m), because every submap was moved
    # independently and the overlap agreement was destroyed.
    #
    # As rows in one system, least squares balances them: a submap with
    # strong overlap evidence is held by its neighbours, one with weak
    # overlap is pulled to its GPS height, and no correction can undo
    # another.
    gps_rows: list[tuple[int, float, dict]] = []
    if camera_gps_enu and keyframe_altitude_m:
        for i, (sm, tf) in enumerate(zip(submaps, transforms)):
            pts = tf.apply(sm.points.xyz.reshape(-1, 3))
            if len(pts) < 500:
                continue
            tree_i = cKDTree(pts[:, :2])
            errs = []
            for kf in sm.keyframe_indices:
                cam = camera_gps_enu.get(kf)
                alt = keyframe_altitude_m.get(kf)
                if cam is None or alt is None:
                    continue
                idx = tree_i.query_ball_point(np.asarray(cam)[:2], _GROUND_SAMPLE_RADIUS_M)
                if len(idx) < 100:
                    continue
                errs.append((float(cam[2]) - float(alt)) - float(np.percentile(pts[idx, 2], _GROUND_PERCENTILE)))
            if errs:
                dz = float(np.median(errs))
                if abs(dz) <= _MAX_LEVEL_CORRECTION_M:
                    gps_rows.append((i, dz, {"submap": i, "gps_dz_m": round(dz, 2), "cameras": len(errs)}))

    n_rows = len(rows) + len(gps_rows) + 1
    A = sp.lil_matrix((n_rows, n), dtype=np.float64)
    rhs = np.zeros((n_rows, 3), dtype=np.float64)
    for r, (i, j, offset, weight, _e) in enumerate(rows):
        A[r, j] = -weight
        A[r, i] = weight
        rhs[r] = -offset * weight
    for k, (i, dz, _e) in enumerate(gps_rows):
        r = len(rows) + k
        A[r, i] = _GPS_CONSTRAINT_WEIGHT
        rhs[r, 2] = dz * _GPS_CONSTRAINT_WEIGHT  # vertical only
    # Gauge: pin submap 0 only when GPS gives no absolute reference.
    # With GPS rows present the system is already anchored, and pinning
    # would fight them.
    A[n_rows - 1, 0] = 0.0 if gps_rows else 1.0
    A = A.tocsr()

    corrections = np.zeros((n, 3))
    for axis in range(3):
        corrections[:, axis] = lsqr(A, rhs[:, axis], atol=1e-10, btol=1e-10)[0]

    out = [Sim3(scale=t.scale, R=t.R, t=t.t + corrections[k]) for k, t in enumerate(transforms)]

    # Residual after the solve: how far each constraint is still violated.
    residuals = [
        float(np.linalg.norm((corrections[j] - corrections[i]) + offset))
        for i, j, offset, _w, _e in rows
    ]
    measured = [e["offset_m"] for e in per_junction if "offset_m" in e and "refused" not in e]
    z = [abs(e["offset_z_m"]) for e in per_junction if "offset_z_m" in e and "refused" not in e]
    diag = {
        "method": "global_least_squares",
        "junctions_measured": len(rows),
        "junctions_tried": len(per_junction),
        "median_offset_m": round(float(np.median(measured)), 3) if measured else None,
        "max_offset_m": round(float(np.max(measured)), 3) if measured else None,
        "median_vertical_offset_m": round(float(np.median(z)), 3) if z else None,
        "gps_constraints": len(gps_rows),
        "gps_dz_span_m": round(float(np.ptp([g[1] for g in gps_rows])), 2) if len(gps_rows) > 1 else None,
        "median_residual_m": round(float(np.median(residuals)), 3),
        "max_correction_m": round(float(np.abs(corrections).max()), 3),
        "per_junction": per_junction,
    }
    logger.info(
        "overlap solve: %d constraints over %d submaps solved globally. "
        "Measured offsets: median %.2f m (vertical %.2f m, max %.2f m). "
        "Residual after solve: median %.2f m -- error is now distributed across the "
        "network rather than accumulated along a chain.",
        len(rows),
        n,
        diag["median_offset_m"] or 0.0,
        diag["median_vertical_offset_m"] or 0.0,
        diag["max_offset_m"] or 0.0,
        diag["median_residual_m"],
    )
    return out, diag


# ---------------------------------------------------------------------------
# GPS vertical levelling
# ---------------------------------------------------------------------------

#: Radius around a camera's ground track used to sample the reconstructed
#: ground beneath it. Small enough to be "under this camera", large enough
#: to contain thousands of points.
_GROUND_SAMPLE_RADIUS_M = 10.0

#: Percentile of Z taken as ground beneath a camera. Low, because the
#: column also contains canopy and roofs; the 10th percentile tracks the
#: ground while ignoring the few points below it.
_GROUND_PERCENTILE = 10.0

#: Vertical corrections larger than this are refused -- at that size the
#: submap is not merely drifted, and shifting it would move it somewhere
#: arbitrary.
_MAX_LEVEL_CORRECTION_M = 60.0


def level_submaps_to_gps(
    submaps,
    transforms,
    camera_gps_enu: dict,
    keyframe_altitude_m: dict,
    *,
    radius_m: float = _GROUND_SAMPLE_RADIUS_M,
    max_correction_m: float = _MAX_LEVEL_CORRECTION_M,
):
    """Shift each submap vertically so the ground sits at ``camera_Z - altitude``.

    The constraint every other step misses
    ---------------------------------------
    A nadir camera at GPS position ``C`` flying ``alt_rel`` above the
    ground is looking at ground whose elevation is ``C.z - alt_rel``. That
    is a direct measurement, available for every keyframe, and it pins the
    one axis the camera-centre Sim(3) cannot (survey camera centres are a
    near-flat sheet, planarity ratio 0.045).

    Nothing upstream enforces it. ``depth_anchor`` makes the ground the
    right distance from its OWN camera, which is a per-window statement;
    two windows can each satisfy it and still sit at different absolute
    heights. ``solve_overlap_translations`` ties windows to each other but
    has no absolute reference, so the whole network can drift together.

    Measured on real footage: GPS says the ground across this flight spans
    5.1 m (level terrain, constant altitude). The reconstruction spanned
    **50.0 m**, drifting smoothly from -20 m at the start to +25 m in the
    middle. Flat fields were 17.3 m thick with duplicated surfaces.

    Why per submap and vertical only
    ---------------------------------
    A single offset per submap is rigid -- it cannot bend the geometry
    inside a window, so nothing measured within a window is disturbed.
    Vertical only, because horizontal placement is already correct (98% of
    reconstructed points fall inside the camera footprint GPS predicts).

    The known weakness: ``alt_rel`` is height above the TAKEOFF POINT, not
    above the ground being imaged, so on sloping terrain this imposes the
    takeoff elevation where the true ground differs. That error is the
    site's relief -- metres -- against a 50 m drift. It is reported per
    submap so a reader can see which correction was large.
    """
    from scipy.spatial import cKDTree

    if not camera_gps_enu or not keyframe_altitude_m:
        return transforms, {"applied": False, "failure": "no GPS positions or altitudes"}

    out = list(transforms)
    per_submap = []
    applied = 0
    for i, (sm, tf) in enumerate(zip(submaps, transforms)):
        entry = {"submap": i}
        pts = tf.apply(sm.points.xyz.reshape(-1, 3))
        if len(pts) < 500:
            entry["skipped"] = "too few points"
            per_submap.append(entry)
            continue
        tree = cKDTree(pts[:, :2])
        errors = []
        for kf in sm.keyframe_indices:
            cam = camera_gps_enu.get(kf)
            alt = keyframe_altitude_m.get(kf)
            if cam is None or alt is None:
                continue
            idx = tree.query_ball_point(np.asarray(cam)[:2], radius_m)
            if len(idx) < 100:
                continue
            actual = float(np.percentile(pts[idx, 2], _GROUND_PERCENTILE))
            expected = float(cam[2]) - float(alt)
            errors.append(expected - actual)
        if not errors:
            entry["skipped"] = "no camera had enough ground beneath it"
            per_submap.append(entry)
            continue
        correction = float(np.median(errors))
        entry["cameras"] = len(errors)
        entry["correction_m"] = round(correction, 2)
        if abs(correction) > max_correction_m:
            entry["refused"] = f"|{correction:.1f}| m exceeds {max_correction_m} m"
            per_submap.append(entry)
            continue
        out[i] = Sim3(scale=tf.scale, R=tf.R, t=tf.t + np.array([0.0, 0.0, correction]))
        applied += 1
        per_submap.append(entry)

    corrections = [e["correction_m"] for e in per_submap if "correction_m" in e and "refused" not in e]
    diag = {
        "applied": applied > 0,
        "submaps_levelled": applied,
        "submaps_total": len(submaps),
        "median_correction_m": round(float(np.median(corrections)), 2) if corrections else None,
        "correction_span_m": round(float(np.ptp(corrections)), 2) if len(corrections) > 1 else None,
        "per_submap": per_submap,
    }
    if corrections:
        logger.info(
            "gps levelling: shifted %d/%d submaps vertically so the ground sits at "
            "camera_Z - alt_rel (median %.2f m, span %.2f m). This is the only step that "
            "gives the merge an ABSOLUTE vertical reference; camera centres are too "
            "coplanar for the Sim(3) to supply one.",
            applied,
            len(submaps),
            diag["median_correction_m"],
            diag["correction_span_m"] or 0.0,
        )
    return out, diag
