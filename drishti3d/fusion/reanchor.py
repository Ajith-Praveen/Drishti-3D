"""Re-anchor each view's dense depth to the bundle adjustment's sparse points.

Why a second anchoring pass
---------------------------
``geometry.depth_anchor`` rescales every view inside GeometryStage using
depth it triangulates itself from feature pairs. That is the right thing
to do at that point -- nothing better exists yet -- and on the straight
transect it lands within a few percent. On the loop it does not: where
the backbone's depth was six times too shallow the pair triangulation
gets 2-14k usable samples instead of 40k, its spread is 3-6x wider, and
the resulting per-window ratios (3.7, 6.1, 5.8 on adjacent windows) left
the ground 20% high in some windows and 40% deep in others. Bundle
adjustment then ran with BA-grade rotations and it made no difference,
which rules rotation out: the limit is the anchor's own measurement.

By the time fusion runs, a strictly better measurement exists. The bundle
adjustment has ~7,700 points seen from several cameras each, refined to
~1.7 px reprojection with GPS priors. Those points are the parallax truth
this pipeline has, and they are already in the world frame with the
refined cameras. This module compares each view's dense depth against
them and rescales the view -- the same ``p' = C + r (p - C)`` correction
as the first pass, driven by a far better ``r``.

How the comparison avoids the frame problem
--------------------------------------------
Submap points live in the submap's local frame; BA points live in the
world frame. Rather than transform one into the other (which needs the
merge transform this stage has not computed yet), both are expressed in
*normalised image coordinates* of their own camera: ``(x/z, y/z)`` in
camera axes. That quantity is the pixel's viewing ray and does not depend
on where the camera sits in any frame, so a BA point and a dense point
that come from the same pixel land at the same normalised coordinate
whichever frame each is described in. Nearest neighbours in that space
pair each sparse point with the dense point behind the same pixel, and
the depth ratio between them is the correction.

What it refuses
---------------
A view with fewer than ``min_obs`` matched points is left as-is. A ratio
outside ``[min_ratio, max_ratio]`` is refused as a failure, not applied:
this pass corrects a first pass that was already roughly right, so a
correction of 3x means something else is broken and hiding it would be
worse than reporting it.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from drishti3d.types import Pose, Submap

logger = logging.getLogger(__name__)

__all__ = ["reanchor_submaps_to_ba"]

#: Radius in normalised image coordinates within which a BA point and a
#: dense point are taken to be the same pixel. 0.004 is ~2 px at the 956 px
#: working resolution (fx ~530): tight enough that a roof point is not
#: paired with the street beside it.
_MATCH_RADIUS_NORM = 0.004

#: Matched observations a view needs before its ratio is believed.
_MIN_OBS = 20


def _normalised(points: np.ndarray, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """``(x/z, y/z)`` and ``z`` of world/local points in ``pose``'s camera frame."""
    q = (np.asarray(points, dtype=np.float64) - pose.t) @ pose.R
    z = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        n = q[:, :2] / z[:, None]
    return n, z


def reanchor_submaps_to_ba(
    submaps: list[Submap],
    ba_points: np.ndarray,
    obs_camera_idx: np.ndarray,
    obs_point_idx: np.ndarray,
    world_poses: list[Pose],
    *,
    min_obs: int = _MIN_OBS,
    min_ratio: float = 0.3,
    max_ratio: float = 3.0,
    match_radius: float = _MATCH_RADIUS_NORM,
) -> tuple[list[Submap], dict]:
    """Rescale each submap view's points to agree with the BA points it observed.

    Returns ``(submaps, diag)``. Submaps whose ``view_index`` is missing
    are returned untouched (older caches); every skipped or refused view
    is listed in ``diag`` with its reason.
    """
    ba_points = np.asarray(ba_points, dtype=np.float64)
    obs_camera_idx = np.asarray(obs_camera_idx)
    obs_point_idx = np.asarray(obs_point_idx)

    out: list[Submap] = []
    per_view: list[dict] = []
    applied = 0
    inherited = 0
    ratios_applied: list[float] = []

    for si, sm in enumerate(submaps):
        vidx = getattr(sm, "view_index", None)
        if vidx is None or len(vidx) != sm.points.xyz.shape[0]:
            out.append(sm)
            per_view.append({"submap": si, "skipped": "no per-point view index"})
            continue

        xyz = np.array(sm.points.xyz, dtype=np.float64, copy=True)
        for v, k in enumerate(sm.keyframe_indices):
            entry = {"submap": si, "view": v, "keyframe": int(k)}
            if k >= len(world_poses) or world_poses[k] is None:
                entry["skipped"] = "no refined world pose"
                per_view.append(entry)
                continue
            sel = obs_point_idx[obs_camera_idx == k]
            if len(sel) < min_obs:
                entry["skipped"] = f"only {len(sel)} BA observations"
                per_view.append(entry)
                continue
            n_ba, z_ba = _normalised(ba_points[sel], world_poses[k])
            good = np.isfinite(n_ba).all(axis=1) & (z_ba > 1e-6)
            n_ba, z_ba = n_ba[good], z_ba[good]

            mask = vidx == v
            if mask.sum() < min_obs:
                entry["skipped"] = "view has too few dense points"
                per_view.append(entry)
                continue
            dense = xyz[mask]
            n_bb, z_bb = _normalised(dense, sm.poses[v])
            ok = np.isfinite(n_bb).all(axis=1) & (z_bb > 1e-6)
            if ok.sum() < min_obs:
                entry["skipped"] = "dense points behind the camera"
                per_view.append(entry)
                continue
            tree = cKDTree(n_bb[ok])
            dist, nn = tree.query(n_ba, k=1, distance_upper_bound=match_radius)
            hit = np.isfinite(dist)
            if hit.sum() < min_obs:
                entry["skipped"] = f"only {int(hit.sum())} BA points matched a dense pixel"
                per_view.append(entry)
                continue
            r = z_ba[hit] / z_bb[ok][nn[hit]]
            r = r[np.isfinite(r) & (r > 0)]
            ratio = float(np.median(r))
            entry.update({"matched": int(len(r)), "ratio": round(ratio, 4), "mad": round(float(np.median(np.abs(r - ratio))), 4)})
            if not (min_ratio <= ratio <= max_ratio):
                entry["refused"] = f"ratio {ratio:.3g} outside [{min_ratio}, {max_ratio}]"
                per_view.append(entry)
                continue
            C = np.asarray(sm.poses[v].t, dtype=np.float64)
            xyz[mask] = C + ratio * (dense - C)
            applied += 1
            ratios_applied.append(ratio)
            per_view.append(entry)

        # A view the BA points could not measure must NOT be left at the
        # old scale while its neighbours move: that is what splits a mesh.
        # It inherits the submap's median measured ratio instead -- the
        # same fallback geometry.depth_anchor uses for thin views, and for
        # the same reason. Measured: with 2,000 BA points over 77 cameras
        # (~26 each, under the 20-observation floor) 45 of 149 views went
        # unmeasured and the reconstructed ground spread over 368 m.
        measured_here = [e["ratio"] for e in per_view if e.get("submap") == si and "ratio" in e and "refused" not in e]
        if measured_here:
            fallback = float(np.median(measured_here))
            for v, _k in enumerate(sm.keyframe_indices):
                entry = next((e for e in per_view if e.get("submap") == si and e.get("view") == v), None)
                if entry is None or "ratio" in entry and "refused" not in entry:
                    continue  # already rescaled above
                mask = vidx == v
                if not mask.any():
                    continue
                C = np.asarray(sm.poses[v].t, dtype=np.float64)
                xyz[mask] = C + fallback * (xyz[mask] - C)
                entry["inherited_submap_ratio"] = round(fallback, 4)
                inherited += 1

        out.append(
            Submap(
                window=sm.window,
                poses=sm.poses,
                points=type(sm.points)(
                    xyz=xyz,
                    rgb=sm.points.rgb,
                    covariance=getattr(sm.points, "covariance", None),
                    confidence=getattr(sm.points, "confidence", None),
                ),
                confidence=sm.confidence,
                keyframe_indices=sm.keyframe_indices,
                local_origin=sm.local_origin,
                view_index=vidx,
            )
        )

    total_views = sum(len(sm.keyframe_indices) for sm in submaps)
    # Why views were NOT rescaled, as a histogram: "too few BA observations"
    # on a whole stretch of the flight means the sparse refinement never
    # reached it, which is a different problem from a refused ratio.
    reasons: dict[str, int] = {}
    for e in per_view:
        key = e.get("refused") or e.get("skipped")
        if key:
            key = key.split(" BA ")[0] if "BA observations" in key else key
            key = "ratio outside bounds" if key.startswith("ratio ") else key
            key = "only N BA observations" if key.startswith("only ") and "observations" in key else key
            key = "only N BA points matched a dense pixel" if "matched a dense pixel" in key else key
            reasons[key] = reasons.get(key, 0) + 1
    diag = {
        "views_total": total_views,
        "views_reanchored": applied,
        # Views rescaled by their submap's median rather than their own
        # measurement. Reported separately: it is a weaker claim.
        "views_inherited_submap_ratio": inherited,
        "views_left_unscaled": total_views - applied - inherited,
        "skip_reasons": reasons,
        "ratio_median": float(np.median(ratios_applied)) if ratios_applied else None,
        "ratio_min": float(min(ratios_applied)) if ratios_applied else None,
        "ratio_max": float(max(ratios_applied)) if ratios_applied else None,
        "per_view": per_view,
    }
    if ratios_applied:
        logger.info(
            "ba re-anchor: rescaled %d/%d views (+%d inherited submap median) against %d BA points; "
            "residual ratio median %.3f (min %.3f, max %.3f)",
            applied,
            total_views,
            inherited,
            len(ba_points),
            diag["ratio_median"],
            diag["ratio_min"],
            diag["ratio_max"],
        )
    else:
        logger.warning("ba re-anchor: no view could be re-anchored (%d views)", total_views)
    return out, diag
