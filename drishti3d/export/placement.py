"""Frame-placement check: prove the windows land in the right place before meshing.

Why this stage exists
---------------------
Every progress snapshot this pipeline writes is *top-down*, and a top-down
raster cannot show a vertical error. When 21 windows each reconstruct the
same field at a slightly different height, the top-down view of the stack
is indistinguishable from the top-down view of a correct reconstruction --
so a merge that is metres wrong in Z renders as a plausible-looking map,
gets fused into a mesh, and only then looks bad, forty minutes later, with
no way to tell placement error apart from meshing error.

This module separates those two questions. It runs immediately after
``merge_submaps`` and before any fusion, and answers one thing:

    are the windows in the right place?

It answers it in the two ways a vertical error is actually visible:

- **Elevation renders** (X-Z and Y-Z), coloured *per submap*. A correct
  merge draws one ground line with every colour interleaved along it. A
  stacked merge draws the same terrain profile repeated at several
  heights, one colour per copy -- the defect is unmistakable, and it is
  invisible in every top-down view.
- **Numbers against GPS.** Telemetry says where the ground is:
  ``camera_Z - alt_rel``. Each submap's own ground height is compared to
  that, and the spread across submaps is the merge error in metres.

Nothing here modifies the reconstruction. It measures, renders and (when
``halt_on_failure``) refuses to spend fusion time on a placement that is
already known to be wrong.

What "ground height" means here
-------------------------------
The 10th percentile of Z within a submap, not the minimum: a handful of
outlier points below the terrain would otherwise set the level. On a
flight over buildings the true ground is still the low tail of the
distribution, so p10 tracks it; over dense canopy it sits inside the
canopy instead, which biases every submap in the *same* direction and so
leaves the cross-submap spread -- the number that matters -- intact.

Thickness is measured per horizontal cell, as the p5-p95 span of Z inside
that cell. Cells are restricted to the flat part of the scene (see
``_flat_cell_mask``) because a cell straddling a roof edge legitimately
contains a 10 m span and would otherwise be indistinguishable from a cell
containing two copies of the same ground.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PlacementReport",
    "frame_footprints",
    "frame_placement_metrics",
    "ground_z_from_telemetry",
    "placement_metrics",
    "render_elevation",
    "render_frame_placement",
    "render_topdown_by_submap",
    "write_frame_placement",
    "write_placement_check",
]

#: Longest-edge pixels for each rendered panel.
_PANEL_SIZE = 900

#: Vertical exaggeration on the elevation panels. A survey strip is wide
#: and thin; at true aspect the whole reconstruction is a few pixels tall
#: and a 10 m stack is invisible. Panels are titled with this factor.
_ELEVATION_EXAGGERATION = 6.0

#: Percentile taken as a submap's ground level. See module docstring.
_GROUND_PCTL = 10.0

#: Horizontal cell size, metres, for the thickness measurement.
_CELL_M = 2.0

#: Cells with fewer points than this are ignored -- a thickness computed
#: from three points is noise, not a measurement.
_MIN_CELL_POINTS = 30
#: Points one window needs inside an overlap cell for its median height
#: there to count (a shared cell splits its points between windows).
_MIN_OVERLAP_POINTS = 8

#: A cell counts as flat when its own p5-p95 Z span, measured from the
#: single best-placed submap in it, is below this. Cells failing it may
#: contain real vertical structure (a wall, a roof edge) and cannot be
#: used to distinguish real relief from duplicate surfaces.
_FLAT_CELL_SPAN_M = 1.5

#: Placement is reported as FAILED above this cross-submap ground spread.
#: One metre is the problem statement's own accuracy requirement, so a
#: merge that scatters submaps further than this cannot meet it no matter
#: how good the fusion is.
_SPREAD_FAIL_M = 1.0

#: With a single submap there is no cross-window spread to measure, so the
#: verdict falls back to where the ground sits against the GPS ground and
#: how thick flat ground came out. The GPS threshold is looser than the
#: spread one because the GPS ground itself carries standalone-GPS
#: vertical error (a few metres); a flat-ground slab thicker than two
#: metres is a depth problem whatever the GPS says.
_GPS_GROUND_FAIL_M = 5.0
_FLAT_THICKNESS_FAIL_M = 2.0

#: The "GPS ground" is the takeoff height (``camera_Z - alt_rel``), which is
#: only where the ground is while the terrain stays level with the launch
#: point: on PinPoint flight01 the ground under a 20 s clip sits 4-15 m
#: below takeoff, so a correctly placed window "failed" by 8.5 m. When the
#: image-derived bundle-adjustment points exist in the dense cloud's own
#: frame they are the reference instead, compared cell by cell on this
#: grid (BA points are sparse, so the cells are coarse), with at least
#: this many reference points per cell.
_REFERENCE_CELL_M = 10.0
_REFERENCE_MIN_POINTS = 5
_REFERENCE_GROUND_FAIL_M = 5.0

#: Without that reference the takeoff height is all there is, and it is an
#: assumption, not a measurement: terrain inside a survey routinely sits
#: 10-20 m off the launch point. Its tolerance therefore grows with the
#: flight height -- still catching a window whose depth scale is wrong by
#: tens of percent (the failure this check exists for), no longer failing
#: correct geometry over hilly ground.
_TAKEOFF_RELIEF_FRACTION = 0.25

#: Distinct hues for per-submap colouring, BGR. Chosen to stay
#: distinguishable at one-pixel splat size.
_PALETTE = np.array(
    [
        (60, 60, 255), (60, 200, 60), (255, 120, 60), (60, 220, 255),
        (255, 60, 200), (255, 220, 60), (140, 60, 255), (60, 255, 180),
        (200, 200, 200), (30, 120, 255), (180, 255, 60), (255, 60, 100),
    ],
    dtype=np.uint8,
)


class PlacementReport:
    """Placement measurements, plus whether they pass."""

    def __init__(self, metrics: dict) -> None:
        self.metrics = metrics

    @property
    def ground_spread_m(self) -> float | None:
        return self.metrics.get("ground_spread_m")

    @property
    def verdict(self) -> str:
        """``"PASS"``, ``"FAIL"`` or ``"UNMEASURED"``.

        Several submaps: they must agree on the ground to within a metre.
        One submap: its ground must sit within ``_GPS_GROUND_FAIL_M`` of the
        GPS ground and flat ground must be thinner than
        ``_FLAT_THICKNESS_FAIL_M``. Nothing measurable is ``UNMEASURED`` --
        not a failure, and never silently a pass.
        """
        if self.metrics.get("forward_view"):
            # A level, forward camera sees the ground at grazing angles and its
            # solved points sit on canopy and walls, not ground: the test is
            # whether the dense surface passes through those points in 3D.
            d, lim = self.metrics.get("ba_to_dense_median_m"), self.metrics.get("ba_to_dense_limit_m")
            if d is None or lim is None:
                return "UNMEASURED"
            return "PASS" if d <= lim else "FAIL"
        spread = self.ground_spread_m
        ref_err = self.metrics.get("reference_ground_error_median_m")
        if spread is not None:
            if spread > _SPREAD_FAIL_M:
                return "FAIL"
            # Windows can agree with each other and still all sit off the
            # image-derived ground (one shared depth-scale error).
            if ref_err is not None and abs(ref_err) > _REFERENCE_GROUND_FAIL_M:
                return "FAIL"
            return "PASS"
        if ref_err is not None:
            err, limit = ref_err, _REFERENCE_GROUND_FAIL_M
        else:
            err, limit = self.metrics.get("gps_ground_error_median_m"), self.gps_ground_threshold_m
        if err is None:
            return "UNMEASURED"
        thick = self.metrics.get("flat_thickness_median_m")
        if abs(err) > limit or (thick is not None and thick > _FLAT_THICKNESS_FAIL_M):
            return "FAIL"
        return "PASS"

    @property
    def gps_ground_threshold_m(self) -> float:
        """Tolerance against the takeoff-height ground: 5 m, or a quarter of the height above it."""
        agl = self.metrics.get("gps_agl_m")
        if agl:
            return max(_GPS_GROUND_FAIL_M, _TAKEOFF_RELIEF_FRACTION * float(agl))
        return _GPS_GROUND_FAIL_M

    @property
    def passed(self) -> bool:
        """True only for a measured PASS (see ``verdict``)."""
        return self.verdict == "PASS"

    def summary(self) -> str:
        m = self.metrics
        spread = m.get("ground_spread_m")
        thick = m.get("flat_thickness_median_m")
        err = m.get("gps_ground_error_median_m")
        ref_err = m.get("reference_ground_error_median_m")
        parts = [
            f"{m.get('n_submaps', 0)} submaps",
            f"ground spread {spread:.2f} m" if spread is not None else "ground spread n/a",
            f"flat thickness {thick:.2f} m" if thick is not None else "flat thickness n/a",
        ]
        if m.get("forward_view") and m.get("ba_to_dense_median_m") is not None:
            parts.append(
                f"forward view: solved points to dense surface median {m['ba_to_dense_median_m']:.2f} m "
                f"(limit {m.get('ba_to_dense_limit_m', float('nan')):.2f} m)"
            )
        elif ref_err is not None:
            parts.append(f"vs {m.get('reference_ground_source', 'reference')} ground {ref_err:+.2f} m")
        elif err is not None:
            parts.append(f"vs GPS ground {err:+.2f} m")
        parts.append(self.verdict)
        return "; ".join(parts)

    def as_dict(self) -> dict:
        return {
            **self.metrics,
            "passed": self.passed,
            "verdict": self.verdict,
            "spread_threshold_m": _SPREAD_FAIL_M,
            "gps_ground_threshold_m": round(self.gps_ground_threshold_m, 3),
            "reference_ground_threshold_m": _REFERENCE_GROUND_FAIL_M,
            "flat_thickness_threshold_m": _FLAT_THICKNESS_FAIL_M,
        }


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _cell_ids(xy: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Map XY to flat cell ids. Returns (ids, inverse-index into unique)."""
    grid = np.floor(xy / max(cell_m, 1e-6)).astype(np.int64)
    # Pack the 2D cell coordinate into one integer key so np.unique works
    # on a 1D array -- cheaper than unique(axis=0) by a wide margin at
    # several million points.
    key = (grid[:, 0].astype(np.int64) << 32) ^ (grid[:, 1].astype(np.int64) & 0xFFFFFFFF)
    _uniq, inverse = np.unique(key, return_inverse=True)
    return key, inverse


def _per_group_percentiles(
    values: np.ndarray, groups: np.ndarray, n_groups: int, pctls: tuple[float, ...]
) -> np.ndarray:
    """Percentiles of ``values`` within each group id. Shape (n_groups, len(pctls)).

    Done by sorting once on (group, value) and indexing, because calling
    np.percentile per group costs a Python-level loop over hundreds of
    thousands of cells.
    """
    order = np.lexsort((values, groups))
    sorted_groups = groups[order]
    sorted_values = values[order]
    counts = np.bincount(sorted_groups, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    out = np.full((n_groups, len(pctls)), np.nan, dtype=np.float64)
    nonempty = counts > 0
    for j, p in enumerate(pctls):
        # Nearest-rank percentile: index into each group's own sorted run.
        offset = np.clip(np.round((counts - 1) * (p / 100.0)).astype(np.int64), 0, None)
        idx = starts + offset
        out[nonempty, j] = sorted_values[idx[nonempty]]
    return out


def _flat_cell_mask(spans: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Cells usable for the thickness measurement.

    A cell is kept when it holds enough points to measure and its span is
    small enough that a large span there means duplicate surfaces rather
    than genuine vertical structure. The span threshold is applied to the
    *lower quartile* of observed spans when that is already large -- on a
    badly stacked reconstruction every cell is thick, and a fixed 1.5 m
    cut would keep nothing and report "no flat cells" instead of "your
    cells are all 10 m thick".
    """
    enough = counts >= _MIN_CELL_POINTS
    if not enough.any():
        return enough
    kept = spans[enough]
    threshold = _FLAT_CELL_SPAN_M
    if np.nanpercentile(kept, 25) > _FLAT_CELL_SPAN_M:
        # Everything is thick. Keep the flattest quarter so the reported
        # number describes the best-behaved ground in the scene, and note
        # it in the metrics rather than silently widening the definition.
        threshold = float(np.nanpercentile(kept, 25))
    return enough & (spans <= threshold)


def _overlap_disagreement(xyz: np.ndarray, labels: np.ndarray, cell_m: float) -> dict:
    """How far apart windows put the ground WHERE THEY OVERLAP.

    Per cell, every window with enough points gets its own median height;
    the cell's disagreement is the range of those medians. Only cells two
    or more windows reach count, so terrain relief between windows that
    never see the same ground is not mistaken for placement error (a 4 km
    loop spans tens of metres of real relief). The cell grows with the
    area/point ratio so a sparse cloud still gets enough points per cell.
    """
    n = xyz.shape[0]
    area = float(np.ptp(xyz[:, 0]) * np.ptp(xyz[:, 1])) if n else 0.0
    cell = float(max(cell_m, min(25.0, np.sqrt(area * 4 * _MIN_CELL_POINTS / max(n, 1)))))
    empty = {"cell_m": round(cell, 2), "cells": 0, "median_m": None, "p90_m": None}
    if n == 0 or labels.max(initial=0) < 1:
        return empty
    _key, cell_idx = _cell_ids(xyz[:, :2], cell)
    group = cell_idx.astype(np.int64) * (int(labels.max()) + 1) + labels
    uniq, inv, counts = np.unique(group, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    z_sorted = xyz[order, 2]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    medians = np.array([np.median(z_sorted[a : a + c]) for a, c in zip(starts, counts, strict=True)])
    ok = counts >= _MIN_OVERLAP_POINTS
    cells_of_group = uniq // (int(labels.max()) + 1)
    cu, cinv = np.unique(cells_of_group[ok], return_inverse=True)
    if cu.size == 0:
        return empty
    zmax = np.full(cu.size, -np.inf)
    zmin = np.full(cu.size, np.inf)
    np.maximum.at(zmax, cinv, medians[ok])
    np.minimum.at(zmin, cinv, medians[ok])
    nwin = np.bincount(cinv)
    shared = nwin >= 2
    if not shared.any():
        return empty
    ranges = (zmax - zmin)[shared]
    return {
        "cell_m": round(cell, 2),
        "cells": int(shared.sum()),
        "median_m": float(np.median(ranges)),
        "p90_m": round(float(np.percentile(ranges, 90)), 3),
    }


def _reference_ground_errors(
    xyz: np.ndarray, labels: np.ndarray, reference: np.ndarray, cell_m: float = _REFERENCE_CELL_M
) -> tuple[dict[int, float], int]:
    """Per submap: median over shared cells of (its ground - the reference's ground).

    Both grounds are the same low percentile of height inside each cell, so
    trees and roofs bias the two the same way. Returns the per-submap
    errors and how many reference cells were usable.
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    ref = ref[np.isfinite(ref).all(axis=1)]
    if ref.shape[0] < _REFERENCE_MIN_POINTS:
        return {}, 0
    ref_key, ref_idx = _cell_ids(ref[:, :2], cell_m)
    n_ref = int(ref_idx.max()) + 1
    ref_ground = _per_group_percentiles(ref[:, 2], ref_idx, n_ref, (_GROUND_PCTL,))[:, 0]
    ref_counts = np.bincount(ref_idx, minlength=n_ref)
    uniq_ref = np.unique(ref_key)
    usable = ref_counts >= _REFERENCE_MIN_POINTS
    lookup = dict(zip(uniq_ref[usable].tolist(), ref_ground[usable].tolist(), strict=True))
    if not lookup:
        return {}, 0

    errors: dict[int, float] = {}
    for s in np.unique(labels):
        sel = labels == s
        if int(sel.sum()) < _MIN_CELL_POINTS:
            continue
        key, idx = _cell_ids(xyz[sel, :2], cell_m)
        n = int(idx.max()) + 1
        ground = _per_group_percentiles(xyz[sel, 2], idx, n, (_GROUND_PCTL,))[:, 0]
        counts = np.bincount(idx, minlength=n)
        diffs = [
            g - lookup[k]
            for k, g, c in zip(np.unique(key).tolist(), ground.tolist(), counts.tolist(), strict=True)
            if c >= _MIN_CELL_POINTS and k in lookup
        ]
        if diffs:
            errors[int(s)] = float(np.median(diffs))
    return errors, len(lookup)


def placement_metrics(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    gps_ground_z: float | None = None,
    gps_agl_m: float | None = None,
    reference_ground: np.ndarray | None = None,
    reference_source: str = "reference",
    cell_m: float = _CELL_M,
) -> PlacementReport:
    """Measure whether the per-submap placements agree with each other and the ground.

    ``labels`` is a per-point submap index, parallel to ``xyz``. Passing
    an all-zero label array measures the cloud as a whole (thickness only,
    no cross-submap spread). ``reference_ground`` is an optional (M, 3)
    point set in the SAME frame (the bundle-adjustment points), compared
    cell by cell; when it yields a measurement it takes over from the
    takeoff-height ``gps_ground_z`` in the verdict.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if xyz.shape[0] == 0 or labels.shape[0] != xyz.shape[0]:
        return PlacementReport({"n_points": int(xyz.shape[0]), "n_submaps": 0})

    n_submaps = int(labels.max()) + 1 if labels.size else 0
    z = xyz[:, 2]

    # Per-submap ground height, and how far apart those are. This is the
    # merge error: every submap images overlapping ground, so a correct
    # merge puts these within the terrain's own relief of each other.
    ground_by_submap: dict[int, float] = {}
    for s in range(n_submaps):
        sel = labels == s
        if int(sel.sum()) < _MIN_CELL_POINTS:
            continue
        ground_by_submap[s] = float(np.percentile(z[sel], _GROUND_PCTL))

    grounds = np.array(list(ground_by_submap.values()), dtype=np.float64)
    # Whole-flight spread of per-window ground: only meaningful when every
    # window images the same ground (a short strip). Kept for reference;
    # the verdict uses the overlap measure below.
    global_spread = float(grounds.max() - grounds.min()) if grounds.size >= 2 else None
    overlap = _overlap_disagreement(xyz, labels, cell_m)
    # Windows that never share a cell leave only the whole-flight number.
    ground_spread = overlap["median_m"] if overlap["cells"] else global_spread

    # Per-cell vertical span. On a correct reconstruction of flat ground
    # this is a few centimetres of noise; on a stacked one it is the gap
    # between the copies.
    _key, cell_idx = _cell_ids(xyz[:, :2], cell_m)
    n_cells = int(cell_idx.max()) + 1 if cell_idx.size else 0
    pct = _per_group_percentiles(z, cell_idx, n_cells, (5.0, 95.0))
    counts = np.bincount(cell_idx, minlength=n_cells)
    spans = pct[:, 1] - pct[:, 0]

    flat = _flat_cell_mask(spans, counts)
    all_cells = counts >= _MIN_CELL_POINTS
    flat_thickness = float(np.nanmedian(spans[flat])) if flat.any() else None
    all_thickness = float(np.nanmedian(spans[all_cells])) if all_cells.any() else None

    # How many distinct submaps contribute to a typical cell -- a cell
    # seen by six windows whose span is 10 m is six stacked copies; a cell
    # seen by one window cannot be stacked at all, and its span is real
    # structure. Without this the thickness number cannot distinguish the
    # two cases.
    order = np.lexsort((labels, cell_idx))
    c_sorted, l_sorted = cell_idx[order], labels[order]
    boundary = np.ones(len(c_sorted), dtype=bool)
    boundary[1:] = (c_sorted[1:] != c_sorted[:-1]) | (l_sorted[1:] != l_sorted[:-1])
    submaps_per_cell = np.bincount(c_sorted[boundary], minlength=n_cells)

    metrics: dict = {
        "n_points": int(xyz.shape[0]),
        "n_submaps": n_submaps,
        "n_submaps_measured": len(ground_by_submap),
        # String keys, not int: these metrics are serialised into the
        # pipeline result and read back, and JSON has no integer keys --
        # a round trip would otherwise silently change 0 into "0" and
        # make a saved run unequal to the run that produced it.
        "ground_z_by_submap": {str(k): round(v, 3) for k, v in sorted(ground_by_submap.items())},
        "ground_spread_m": None if ground_spread is None else round(ground_spread, 3),
        "ground_spread_global_m": None if global_spread is None else round(global_spread, 3),
        "overlap_cell_m": overlap["cell_m"],
        "overlap_cells": overlap["cells"],
        "overlap_p90_m": overlap["p90_m"],
        "cell_m": cell_m,
        "cells_measured": int(all_cells.sum()),
        "cells_flat": int(flat.sum()),
        "flat_thickness_median_m": None if flat_thickness is None else round(flat_thickness, 3),
        "all_thickness_median_m": None if all_thickness is None else round(all_thickness, 3),
        "submaps_per_cell_median": float(np.median(submaps_per_cell[all_cells])) if all_cells.any() else None,
        "extent_m": {
            "x": round(float(np.ptp(xyz[:, 0])), 1),
            "y": round(float(np.ptp(xyz[:, 1])), 1),
            "z": round(float(np.ptp(z)), 1),
        },
    }

    if gps_ground_z is not None:
        metrics["gps_ground_z_m"] = round(float(gps_ground_z), 3)
        if gps_agl_m:
            metrics["gps_agl_m"] = round(float(gps_agl_m), 2)
        if grounds.size:
            errors = grounds - float(gps_ground_z)
            metrics["gps_ground_error_median_m"] = round(float(np.median(errors)), 3)
            metrics["gps_ground_error_max_abs_m"] = round(float(np.max(np.abs(errors))), 3)
            metrics["gps_ground_error_by_submap"] = {
                str(k): round(v - float(gps_ground_z), 3) for k, v in sorted(ground_by_submap.items())
            }

    if reference_ground is not None:
        ref_errors, ref_cells = _reference_ground_errors(xyz, labels, reference_ground)
        if ref_errors:
            values = np.array(list(ref_errors.values()), dtype=np.float64)
            metrics["reference_ground_source"] = reference_source
            metrics["reference_ground_cells"] = ref_cells
            metrics["reference_ground_error_median_m"] = round(float(np.median(values)), 3)
            metrics["reference_ground_error_max_abs_m"] = round(float(np.max(np.abs(values))), 3)
            metrics["reference_ground_error_by_submap"] = {
                str(k): round(v, 3) for k, v in sorted(ref_errors.items())
            }

    return PlacementReport(metrics)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _colours_for(labels: np.ndarray) -> np.ndarray:
    return _PALETTE[np.asarray(labels).reshape(-1).astype(np.int64) % len(_PALETTE)]


def _splat(
    h_world: np.ndarray,
    v_world: np.ndarray,
    colours: np.ndarray,
    *,
    size: int,
    exaggerate_v: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Orthographic splat of (horizontal, vertical) world coords into BGR.

    ``exaggerate_v`` > 1 stretches the vertical axis relative to the
    horizontal one. An elevation view of a survey is a thin strip --
    hundreds of metres wide, tens tall -- and at true aspect it renders a
    few pixels high, hiding exactly what the panel exists to show. The
    panel title states the exaggeration so nobody reads a stretched view
    as metric.

    Returns the image and the (hmin, hmax, vmin, vmax) extent it used, so
    overlays can be drawn in the same frame afterwards. Rows always
    increase downward, so larger ``v_world`` is higher in the image.
    """
    hmin, hmax = float(h_world.min()), float(h_world.max())
    vmin, vmax = float(v_world.min()), float(v_world.max())
    span_h = max(hmax - hmin, 1e-6)
    span_v = max(vmax - vmin, 1e-6)

    if exaggerate_v == 1.0:
        # True aspect: one scale for both axes, so the panel is metric.
        scale_h = scale_v = size / max(span_h, span_v)
    else:
        scale_h = size / span_h
        scale_v = min(scale_h * exaggerate_v, size / span_v)
    width = max(1, min(size, round(span_h * scale_h)))
    height = max(1, min(size, round(span_v * scale_v)))

    col = np.clip(((h_world - hmin) * scale_h).astype(np.int32), 0, width - 1)
    row = np.clip((height - 1 - (v_world - vmin) * (height / span_v)).astype(np.int32), 0, height - 1)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[row, col] = colours
    return image, (hmin, hmax, vmin, vmax)


def render_topdown_by_submap(xyz: np.ndarray, labels: np.ndarray, *, size: int = _PANEL_SIZE) -> np.ndarray | None:
    """Top-down raster with one colour per submap.

    Shows horizontal coverage and which window contributed what. It cannot
    show vertical error -- that is what ``render_elevation`` is for.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return None
    image, _ = _splat(xyz[:, 0], xyz[:, 1], _colours_for(labels), size=size)
    return image


def render_elevation(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    axis: str = "x",
    size: int = _PANEL_SIZE,
    gps_ground_z: float | None = None,
    exaggerate_v: float = _ELEVATION_EXAGGERATION,
) -> np.ndarray | None:
    """Side view: ``axis`` ("x" or "y") against Z, coloured per submap.

    This is the panel that makes a stacked merge obvious. Correct
    placement draws a single terrain profile with every submap's colour
    mixed along it. Duplicate surfaces draw the same profile several
    times at different heights, each in its own colour.

    ``gps_ground_z`` draws a white reference line at the height telemetry
    says the ground is, so absolute placement error is readable off the
    panel and not only relative disagreement.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return None
    h = xyz[:, 0] if axis == "x" else xyz[:, 1]
    z = xyz[:, 2]
    image, (_hmin, _hmax, vmin, vmax) = _splat(h, z, _colours_for(labels), size=size, exaggerate_v=exaggerate_v)

    if gps_ground_z is not None and vmax > vmin:
        height = image.shape[0]
        row = int(round((height - 1) * (1.0 - (float(gps_ground_z) - vmin) / (vmax - vmin))))
        if 0 <= row < height:
            cv2.line(image, (0, row), (image.shape[1] - 1, row), (255, 255, 255), 1)
    return image


def _label_panel(image: np.ndarray, title: str) -> np.ndarray:
    bar = np.zeros((26, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def _text_panel(lines: list[str], width: int) -> np.ndarray:
    height = 26 + 18 * len(lines) + 8
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, "PLACEMENT CHECK", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        colour = (120, 120, 255) if line.startswith("FAIL") else (210, 210, 210)
        cv2.putText(panel, line, (6, 40 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    return panel


def _stack(panels: list[np.ndarray]) -> np.ndarray:
    width = max(p.shape[1] for p in panels)
    padded = [
        np.pad(p, ((0, 0), (0, width - p.shape[1]), (0, 0))) if p.shape[1] < width else p
        for p in panels
    ]
    return np.vstack(padded)


# ---------------------------------------------------------------------------
# the stage entry point
# ---------------------------------------------------------------------------


def write_placement_check(
    out_dir: Path | str,
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    gps_ground_z: float | None = None,
    gps_agl_m: float | None = None,
    reference_ground: np.ndarray | None = None,
    reference_source: str = "reference",
    camera_positions: np.ndarray | None = None,
    cell_m: float = _CELL_M,
    max_points: int = 1_500_000,
    size: int = _PANEL_SIZE,
) -> PlacementReport:
    """Measure and render placement into ``out_dir``. Never raises.

    Writes ``placement_topdown.png``, ``placement_elevation_x.png``,
    ``placement_elevation_y.png``, a combined ``placement_check.png``
    sheet, and ``placement.json`` with the metrics.

    Measurements retain complete spatial cells when subsampling. Striding
    individual mesh vertices removes the ground cells' statistical support
    and biases the thickness test toward densely sampled vertical walls.
    """
    out_dir = Path(out_dir)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    if xyz.shape[0] == 0 or labels.shape[0] != xyz.shape[0]:
        return PlacementReport({"n_points": int(xyz.shape[0]), "n_submaps": 0})

    original_count = len(xyz)
    if xyz.shape[0] > max_points:
        step = int(np.ceil(xyz.shape[0] / max_points))
        grid = np.floor(xyz[:, :2] / cell_m).astype(np.int64)
        cell_hash = (grid[:, 0] * 73856093) ^ (grid[:, 1] * 19349663)
        keep = np.remainder(cell_hash, step) == 0
        if not keep.any():
            keep = cell_hash == cell_hash[0]
        xyz, labels = xyz[keep], labels[keep]

    report = placement_metrics(
        xyz,
        labels,
        gps_ground_z=gps_ground_z,
        gps_agl_m=gps_agl_m,
        reference_ground=reference_ground,
        reference_source=reference_source,
        cell_m=cell_m,
    )
    report.metrics["input_points"] = original_count
    report.metrics["sampling"] = "complete_spatial_cells"

    # Rendering may use a point stride after the measurements are complete.
    if len(xyz) > max_points:
        step = int(np.ceil(len(xyz) / max_points))
        xyz, labels = xyz[::step], labels[::step]

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        topdown = render_topdown_by_submap(xyz, labels, size=size)
        elev_x = render_elevation(xyz, labels, axis="x", size=size, gps_ground_z=gps_ground_z)
        elev_y = render_elevation(xyz, labels, axis="y", size=size, gps_ground_z=gps_ground_z)

        if camera_positions is not None and topdown is not None and len(camera_positions):
            _draw_camera_track(topdown, xyz, np.asarray(camera_positions, dtype=np.float64))

        for name, image in (
            ("placement_topdown.png", topdown),
            ("placement_elevation_x.png", elev_x),
            ("placement_elevation_y.png", elev_y),
        ):
            if image is not None:
                cv2.imwrite(str(out_dir / name), image)

        m = report.metrics
        lines = [report.summary()]
        if m.get("ground_spread_m") is not None and not report.passed:
            lines.append(
                f"FAIL: submaps disagree on ground height by {m['ground_spread_m']:.2f} m "
                f"(needs <= {_SPREAD_FAIL_M:.1f} m for metre accuracy)"
            )
        lines.append(
            f"flat cells {m.get('cells_flat')}/{m.get('cells_measured')} at {m.get('cell_m')} m; "
            f"median {m.get('submaps_per_cell_median')} submaps per cell"
        )
        if m.get("gps_ground_z_m") is not None:
            lines.append(
                f"GPS says ground z = {m['gps_ground_z_m']:.2f} m (white line); "
                f"submaps median {m.get('gps_ground_error_median_m', float('nan')):+.2f} m, "
                f"worst {m.get('gps_ground_error_max_abs_m', float('nan')):.2f} m"
            )
        if m.get("reference_ground_error_median_m") is not None:
            lines.append(
                f"vs {m['reference_ground_source']} ground ({m['reference_ground_cells']} cells of "
                f"{_REFERENCE_CELL_M:g} m): submaps median {m['reference_ground_error_median_m']:+.2f} m, "
                f"worst {m['reference_ground_error_max_abs_m']:.2f} m -- this, not the takeoff height, "
                "decides the verdict"
            )
        lines.append("colour = submap index. one ground line = correct. repeated lines = stacked windows.")

        panels = [p for p in (
            _label_panel(
                elev_x,
                f"ELEVATION X-Z  (side view -- vertical error is visible HERE; Z exaggerated {_ELEVATION_EXAGGERATION:.0f}x)",
            ) if elev_x is not None else None,
            _label_panel(elev_y, f"ELEVATION Y-Z  (side view; Z exaggerated {_ELEVATION_EXAGGERATION:.0f}x)")
            if elev_y is not None else None,
            _label_panel(topdown, "TOP-DOWN  (coverage only -- cannot show vertical error)") if topdown is not None else None,
        ) if p is not None]
        if panels:
            width = max(p.shape[1] for p in panels)
            sheet = _stack([_text_panel(lines, width), *panels])
            cv2.imwrite(str(out_dir / "placement_check.png"), sheet)

        (out_dir / "placement.json").write_text(json.dumps(report.as_dict(), indent=2))
    except Exception:
        logger.warning("placement: writing the placement check failed; continuing", exc_info=True)

    level = logger.warning if report.verdict == "FAIL" else logger.info
    level("=" * 78)
    level("PLACEMENT CHECK: %s", report.summary())
    level("  renders in %s (see placement_check.png -- elevation panels show vertical error)", out_dir)
    level("=" * 78)
    return report


def _draw_camera_track(image: np.ndarray, xyz: np.ndarray, cameras: np.ndarray) -> None:
    """Overlay the camera track on a top-down panel drawn from ``xyz``'s extent."""
    xmin, xmax = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    ymin, ymax = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    height, width = image.shape[:2]
    sx = width / max(xmax - xmin, 1e-6)
    sy = height / max(ymax - ymin, 1e-6)
    pts = []
    for c in cameras.reshape(-1, 3):
        col = int(np.clip((c[0] - xmin) * sx, 0, width - 1))
        row = int(np.clip(height - 1 - (c[1] - ymin) * sy, 0, height - 1))
        pts.append((col, row))
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(image, a, b, (255, 255, 255), 1, cv2.LINE_AA)


def ground_z_from_telemetry(camera_positions: np.ndarray, altitudes_rel: list[float | None]) -> float | None:
    """Where telemetry says the ground is: median of ``camera_Z - alt_rel``.

    Returns ``None`` when no keyframe carries a relative altitude. Uses the
    median rather than the mean because a single bad altitude sample would
    otherwise move the reference line every render.
    """
    cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    samples = [
        float(cams[i, 2]) - float(alt)
        for i, alt in enumerate(altitudes_rel)
        if i < len(cams) and alt is not None and float(alt) > 1.0
    ]
    if not samples:
        return None
    return float(np.median(samples))


# ---------------------------------------------------------------------------
# BEFORE geometry: where will each frame land?
# ---------------------------------------------------------------------------
#
# Everything above measures a reconstruction that already exists. That is
# the right check for the merge, but it is an expensive place to learn
# that the inputs were wrong: on this project the dense backbone spends
# about eight minutes on the windows before the first surface can be
# looked at, and a pose or altitude problem has by then been baked into
# every one of them.
#
# The functions below need no depth and no inference. A camera pose, the
# intrinsics and a ground height are enough to say exactly which patch of
# ground a frame sees -- cast the four image corners onto the ground
# plane. Rendering those footprints answers, in seconds and before any
# geometry is built:
#
#   - do consecutive frames overlap enough to be reconstructed together?
#   - does the flight actually cover the area, or are there holes?
#   - do the poses agree with GPS, or has a frame been placed somewhere
#     the drone never flew?
#
# A footprint is only as good as the flat-ground assumption behind it
# (terrain relief moves the corners), so these are a coverage and
# consistency check, not a measurement of the ground itself.


def frame_footprints(
    poses: list,
    intrinsics: list,
    ground_z: float,
    *,
    max_extent_m: float = 2000.0,
) -> list[np.ndarray | None]:
    """Ground footprint of each frame: 4 corners, or ``None`` when undefined.

    Each image corner is a ray from the camera centre through that pixel;
    the footprint corner is where the ray meets the plane ``z = ground_z``.
    A frame whose ray points up, runs parallel to the ground, or lands
    absurdly far away (an oblique view near the horizon does all three)
    returns ``None`` rather than a corner at infinity.
    """
    out: list[np.ndarray | None] = []
    for i, pose in enumerate(poses):
        intr = intrinsics[i] if i < len(intrinsics) else (intrinsics[0] if intrinsics else None)
        if pose is None or intr is None:
            out.append(None)
            continue

        centre = np.asarray(pose.t, dtype=np.float64).reshape(3)
        rotation = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
        height_above = centre[2] - ground_z
        if height_above <= 1.0:
            out.append(None)
            continue

        corners_px = [(0.0, 0.0), (intr.width, 0.0), (intr.width, intr.height), (0.0, intr.height)]
        ground = []
        for u, v in corners_px:
            # Pixel to a direction in the camera frame, then to world.
            ray_cam = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
            ray = rotation @ ray_cam
            if ray[2] >= -1e-6:
                # Pointing level or upward: this corner never meets the ground.
                ground = []
                break
            t = (ground_z - centre[2]) / ray[2]
            point = centre + t * ray
            if np.linalg.norm(point[:2] - centre[:2]) > max_extent_m:
                ground = []
                break
            ground.append(point[:2])
        out.append(np.asarray(ground) if len(ground) == 4 else None)
    return out


def frame_placement_metrics(
    footprints: list[np.ndarray | None],
    camera_positions: np.ndarray,
    *,
    gps_positions: np.ndarray | None = None,
    cell_m: float = 10.0,
) -> dict:
    """Coverage, overlap and pose-vs-GPS agreement, from footprints alone.

    ``overlap_median`` is how many frames see a typical covered cell. It
    is the number that decides whether windowed reconstruction can work
    at all: a cell seen once can be reconstructed but never cross-checked,
    and neighbouring windows that share no ground cannot be aligned to
    each other by any method.
    """
    placed = [f for f in footprints if f is not None]
    metrics: dict = {
        "frames_total": len(footprints),
        "frames_placed": len(placed),
        "frames_unplaced": len(footprints) - len(placed),
        "cell_m": cell_m,
    }
    if not placed:
        return metrics

    # Rasterise each footprint's axis-aligned extent onto a coarse grid
    # and count how many frames cover each cell. The extent overstates a
    # rotated footprint slightly; at 10 m cells against a footprint
    # hundreds of metres across that is not worth a polygon fill.
    all_xy = np.vstack(placed)
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    nx = max(1, int(np.ceil((xmax - xmin) / cell_m)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_m)))
    counts = np.zeros((ny, nx), dtype=np.int32)
    for f in placed:
        c0 = int(np.clip((f[:, 0].min() - xmin) / cell_m, 0, nx - 1))
        c1 = int(np.clip((f[:, 0].max() - xmin) / cell_m, 0, nx - 1))
        r0 = int(np.clip((f[:, 1].min() - ymin) / cell_m, 0, ny - 1))
        r1 = int(np.clip((f[:, 1].max() - ymin) / cell_m, 0, ny - 1))
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1

    covered = counts > 0
    metrics.update(
        {
            "area_covered_m2": float(covered.sum()) * cell_m * cell_m,
            "extent_m": {"x": round(float(xmax - xmin), 1), "y": round(float(ymax - ymin), 1)},
            "overlap_median": float(np.median(counts[covered])) if covered.any() else 0.0,
            "overlap_min": int(counts[covered].min()) if covered.any() else 0,
            "cells_seen_once": int((counts == 1).sum()),
            "cells_covered": int(covered.sum()),
            "footprint_median_m": round(
                float(np.median([np.ptp(f[:, 0]) for f in placed])), 1
            ),
        }
    )

    # Consecutive-frame overlap, the quantity windowing actually depends
    # on. Estimated from how far the camera moved against how wide a
    # frame sees -- exact polygon intersection would be more precise and
    # would not change any decision made from this number.
    cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    if len(cams) >= 2 and metrics.get("footprint_median_m"):
        steps = np.linalg.norm(np.diff(cams[:, :2], axis=0), axis=1)
        width = float(metrics["footprint_median_m"])
        metrics["baseline_median_m"] = round(float(np.median(steps)), 2)
        metrics["consecutive_overlap_pct"] = round(
            100.0 * max(0.0, 1.0 - float(np.median(steps)) / max(width, 1e-6)), 1
        )

    if gps_positions is not None and len(gps_positions) == len(cams) and len(cams):
        gps = np.asarray(gps_positions, dtype=np.float64).reshape(-1, 3)
        residual = np.linalg.norm(cams - gps, axis=1)
        metrics["pose_vs_gps_median_m"] = round(float(np.median(residual)), 3)
        metrics["pose_vs_gps_max_m"] = round(float(residual.max()), 3)

    return metrics


def render_frame_placement(
    footprints: list[np.ndarray | None],
    camera_positions: np.ndarray,
    *,
    size: int = _PANEL_SIZE,
    gps_positions: np.ndarray | None = None,
) -> np.ndarray | None:
    """Top-down: every frame's ground footprint, plus the camera track.

    Footprints are drawn as outlines rather than filled, so overlap shows
    up as a denser weave instead of one flat slab -- the whole point is
    to see how much neighbouring frames share.
    """
    placed = [f for f in footprints if f is not None]
    cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
    if not placed and not len(cams):
        return None

    pts = np.vstack(placed) if placed else cams[:, :2]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span_x = max(float(xmax - xmin), 1e-6)
    span_y = max(float(ymax - ymin), 1e-6)
    scale = size / max(span_x, span_y)
    width = max(1, round(span_x * scale))
    height = max(1, round(span_y * scale))
    image = np.zeros((height, width, 3), dtype=np.uint8)

    def to_px(xy: np.ndarray) -> np.ndarray:
        col = np.clip((xy[:, 0] - xmin) * scale, 0, width - 1)
        row = np.clip(height - 1 - (xy[:, 1] - ymin) * scale, 0, height - 1)
        return np.stack([col, row], axis=1).astype(np.int32)

    for i, f in enumerate(footprints):
        if f is None:
            continue
        colour = tuple(int(c) for c in _PALETTE[i % len(_PALETTE)])
        cv2.polylines(image, [to_px(f)], isClosed=True, color=colour, thickness=1, lineType=cv2.LINE_AA)

    if gps_positions is not None and len(gps_positions):
        track = to_px(np.asarray(gps_positions, dtype=np.float64).reshape(-1, 3)[:, :2])
        cv2.polylines(image, [track], isClosed=False, color=(120, 120, 120), thickness=1, lineType=cv2.LINE_AA)
    if len(cams):
        track = to_px(cams[:, :2])
        cv2.polylines(image, [track], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        for p in track:
            cv2.circle(image, (int(p[0]), int(p[1])), 2, (255, 255, 255), -1)
    return image


def write_frame_placement(
    out_dir: Path | str,
    poses: list,
    intrinsics: list,
    *,
    ground_z: float | None,
    gps_positions: np.ndarray | None = None,
    size: int = _PANEL_SIZE,
) -> dict:
    """Place every frame on the ground and render it, before any geometry runs.

    Writes ``frames_topdown.png``, ``frames_elevation.png``, a labelled
    ``frame_placement.png`` sheet and ``frame_placement.json``. Never
    raises: this is a diagnostic in front of the expensive stage, and it
    must not be the thing that stops a reconstruction.
    """
    out_dir = Path(out_dir)
    usable = [p for p in poses if p is not None]
    if not usable or ground_z is None:
        return {
            "applied": False,
            "reason": "no camera poses" if not usable else "no telemetry altitude to place a ground plane",
        }

    cams = np.array([p.t for p in poses if p is not None], dtype=np.float64)
    footprints = frame_footprints(poses, intrinsics, ground_z)
    metrics = frame_placement_metrics(footprints, cams, gps_positions=gps_positions)
    metrics["ground_z_m"] = round(float(ground_z), 3)
    metrics["applied"] = True

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        topdown = render_frame_placement(footprints, cams, size=size, gps_positions=gps_positions)

        # Elevation: camera heights against the ground plane they were
        # placed on. A frame whose altitude disagrees with its neighbours
        # shows up here as a camera off the track, before its depth has
        # been regressed into the model.
        labels = np.zeros(len(cams), dtype=np.int64)
        elevation = render_elevation(
            np.c_[cams[:, 0], cams[:, 1], cams[:, 2]], labels, axis="x", size=size, gps_ground_z=ground_z
        )

        for name, image in (("frames_topdown.png", topdown), ("frames_elevation.png", elevation)):
            if image is not None:
                cv2.imwrite(str(out_dir / name), image)

        lines = [
            f"{metrics['frames_placed']}/{metrics['frames_total']} frames placed on ground z={ground_z:.2f} m",
            f"covers {metrics.get('area_covered_m2', 0) / 1e4:.2f} ha; "
            f"typical cell seen by {metrics.get('overlap_median')} frames, worst {metrics.get('overlap_min')}",
            f"frame footprint {metrics.get('footprint_median_m')} m wide; "
            f"camera moves {metrics.get('baseline_median_m')} m between frames "
            f"= {metrics.get('consecutive_overlap_pct')}% overlap",
        ]
        if metrics.get("pose_vs_gps_median_m") is not None:
            lines.append(
                f"poses vs GPS: median {metrics['pose_vs_gps_median_m']:.2f} m, "
                f"max {metrics['pose_vs_gps_max_m']:.2f} m"
            )
        if metrics.get("cells_seen_once"):
            lines.append(
                f"{metrics['cells_seen_once']}/{metrics['cells_covered']} cells are seen by only one "
                "frame -- nothing can cross-check those"
            )
        lines.append("colour = frame index; white = camera track; grey = raw GPS track")

        panels = [p for p in (
            _label_panel(topdown, "FRAME FOOTPRINTS  (where each frame lands, before any geometry)")
            if topdown is not None else None,
            _label_panel(
                elevation,
                f"CAMERA HEIGHTS vs GROUND PLANE  (Z exaggerated {_ELEVATION_EXAGGERATION:.0f}x)",
            ) if elevation is not None else None,
        ) if p is not None]
        if panels:
            width = max(p.shape[1] for p in panels)
            cv2.imwrite(str(out_dir / "frame_placement.png"), _stack([_text_panel(lines, width), *panels]))

        (out_dir / "frame_placement.json").write_text(json.dumps(metrics, indent=2))
    except Exception:
        logger.warning("placement: writing the frame placement failed; continuing", exc_info=True)

    logger.info("=" * 78)
    logger.info(
        "FRAME PLACEMENT (before geometry): %d/%d frames placed; typical ground cell seen by %s "
        "frames; %s%% consecutive overlap",
        metrics["frames_placed"],
        metrics["frames_total"],
        metrics.get("overlap_median"),
        metrics.get("consecutive_overlap_pct"),
    )
    logger.info("  renders in %s (frame_placement.png)", out_dir)
    logger.info("=" * 78)
    return metrics
