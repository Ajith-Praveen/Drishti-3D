"""Stage-by-stage previews of what the pipeline knows, before it builds anything.

Why the pipeline is staged this way
-----------------------------------
Reconstruction failures on this project were repeatedly diagnosed from the
finished mesh, forty minutes after the cause. That does not work, because
by then a pose error, a coverage hole and a depth error all look like the
same thing: a bad model.

So the expensive stages are now gated behind cheap ones that each answer a
single question, in the order the answers become available, and each one
renders what it concluded:

1. **Path** -- where did the drone start, and how did it move?
   Needs only telemetry. Answers it in seconds.
2. **Coverage** -- which ground did the camera actually see?
   Needs the path plus intrinsics. Still no depth, still seconds.
3. **Placement** -- do the frames land where the flight says, and do
   neighbours overlap enough to be reconstructed together?
4. **Depth** -- only now is the backbone worth running.
5. **Geometry**, then **Fusion**.

A stage that fails is worth seeing *as a picture*, not as a number,
because the failure modes are spatial: a track that jumps, a strip of
ground nothing covers, a frame placed where the drone never flew.

What these renders can and cannot tell you
-------------------------------------------
Everything here is computed on a flat ground plane at the telemetry's own
height (``camera_z - alt_rel``). Terrain relief moves a real footprint
away from the drawn one by roughly ``relief / altitude`` of its width --
6 m of relief at 120 m is a 5% error. That is fine for judging coverage
and overlap, and it is NOT a measurement of the ground itself. The
surface only exists after depth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "coverage_metrics",
    "path_metrics",
    "render_coverage",
    "render_drone_path",
    "write_coverage_preview",
    "write_path_preview",
]

_PANEL = 900

#: Ground cell size, metres, for the coverage raster. Small enough to see
#: a real gap between passes, large enough that a 700 x 900 m survey fits
#: in a raster that renders instantly.
_COVERAGE_CELL_M = 5.0


# ---------------------------------------------------------------------------
# 1. the drone path
# ---------------------------------------------------------------------------


def path_metrics(positions: np.ndarray, timestamps: list[float] | None = None) -> dict:
    """Shape of the flight: extent, spacing, altitude, and where it starts.

    ``spacing_ratio`` is the one number worth watching. Keyframe spacing
    that varies several-fold across a flight makes every downstream
    window a different size in metres even when it holds the same number
    of frames, and a multi-view depth backbone's metric output degrades
    as its cameras spread out -- so a 5x spacing spread silently becomes
    a 5x spread in depth error.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] < 2:
        return {"n_points": int(pos.shape[0])}

    steps = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    nonzero = steps[steps > 1e-6]
    metrics = {
        "n_points": int(pos.shape[0]),
        "start_enu": [round(float(x), 2) for x in pos[0]],
        "end_enu": [round(float(x), 2) for x in pos[-1]],
        "extent_m": {
            "x": round(float(np.ptp(pos[:, 0])), 1),
            "y": round(float(np.ptp(pos[:, 1])), 1),
            "z": round(float(np.ptp(pos[:, 2])), 1),
        },
        "path_length_m": round(float(steps.sum()), 1),
        "spacing_m": {
            "min": round(float(nonzero.min()), 2) if nonzero.size else None,
            "median": round(float(np.median(nonzero)), 2) if nonzero.size else None,
            "max": round(float(nonzero.max()), 2) if nonzero.size else None,
        },
        "altitude_m": {
            "min": round(float(pos[:, 2].min()), 1),
            "median": round(float(np.median(pos[:, 2])), 1),
            "max": round(float(pos[:, 2].max()), 1),
        },
    }
    if nonzero.size:
        lo, hi = np.percentile(nonzero, [5, 95])
        metrics["spacing_ratio"] = round(float(hi / max(lo, 1e-6)), 2)
    if timestamps and len(timestamps) == pos.shape[0]:
        duration = float(timestamps[-1] - timestamps[0])
        metrics["duration_s"] = round(duration, 1)
        if duration > 0:
            metrics["ground_speed_mps"] = round(float(steps.sum() / duration), 2)
    return metrics


def render_drone_path(
    positions: np.ndarray,
    *,
    size: int = _PANEL,
    gps_positions: np.ndarray | None = None,
) -> np.ndarray | None:
    """Top-down flight track, coloured along time, with START and END marked.

    Direction is drawn as colour (dark at the start, bright at the end)
    rather than as arrowheads, which at survey scale are either invisible
    or cover the track. The raw GPS track is drawn underneath in grey
    when it differs from the refined one, so a pose refinement that has
    dragged a camera somewhere the drone never flew is visible as a
    divergence rather than having to be inferred from an RMSE.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] < 2:
        return None

    pts = pos[:, :2] if gps_positions is None else np.vstack([pos[:, :2], np.asarray(gps_positions)[:, :2]])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    margin = 0.04 * max(xmax - xmin, ymax - ymin, 1.0)
    xmin, ymin, xmax, ymax = xmin - margin, ymin - margin, xmax + margin, ymax + margin
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    scale = size / span
    w = max(1, round((xmax - xmin) * scale))
    h = max(1, round((ymax - ymin) * scale))
    img = np.zeros((h, w, 3), dtype=np.uint8)

    def px(xy):
        c = int(np.clip((xy[0] - xmin) * scale, 0, w - 1))
        r = int(np.clip(h - 1 - (xy[1] - ymin) * scale, 0, h - 1))
        return c, r

    if gps_positions is not None:
        g = np.asarray(gps_positions, dtype=np.float64).reshape(-1, 3)
        for a, b in zip(g[:-1], g[1:], strict=False):
            cv2.line(img, px(a), px(b), (90, 90, 90), 1, cv2.LINE_AA)

    n = pos.shape[0]
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        colour = cv2.applyColorMap(np.uint8([[int(60 + 195 * t)]]), cv2.COLORMAP_VIRIDIS)[0, 0]
        cv2.line(img, px(pos[i]), px(pos[i + 1]), tuple(int(c) for c in colour), 2, cv2.LINE_AA)

    for p in pos:
        cv2.circle(img, px(p), 2, (200, 200, 200), -1)

    s, e = px(pos[0]), px(pos[-1])
    cv2.circle(img, s, 9, (80, 255, 80), 2)
    cv2.putText(img, "START", (s[0] + 12, s[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
    cv2.circle(img, e, 9, (80, 80, 255), 2)
    cv2.putText(img, "END", (e[0] + 12, e[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
    return img


def _altitude_strip(positions: np.ndarray, width: int, height: int = 150) -> np.ndarray:
    """Altitude against distance flown -- a climb or a drop is obvious here."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if pos.shape[0] < 2:
        return img
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))])
    z = pos[:, 2]
    zlo, zhi = float(z.min()), float(z.max())
    if zhi - zlo < 1.0:
        zlo, zhi = zlo - 0.5, zhi + 0.5
    pts = [
        (
            int(np.clip(d / max(dist[-1], 1e-6) * (width - 1), 0, width - 1)),
            int(np.clip(height - 1 - (zi - zlo) / (zhi - zlo) * (height - 20) - 10, 0, height - 1)),
        )
        for d, zi in zip(dist, z, strict=True)
    ]
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        cv2.line(img, a, b, (255, 200, 80), 1, cv2.LINE_AA)
    cv2.putText(img, f"altitude {zlo:.0f}-{zhi:.0f} m over {dist[-1]:.0f} m flown",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def _text_panel(lines: list[str], width: int, title: str) -> np.ndarray:
    h = 30 + 18 * len(lines) + 8
    p = np.zeros((h, width, 3), dtype=np.uint8)
    cv2.putText(p, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        colour = (120, 120, 255) if line.startswith(("FAIL", "WARNING")) else (210, 210, 210)
        cv2.putText(p, line, (6, 44 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    return p


def _stack(panels: list[np.ndarray]) -> np.ndarray:
    width = max(p.shape[1] for p in panels)
    return np.vstack([np.pad(p, ((0, 0), (0, width - p.shape[1]), (0, 0))) if p.shape[1] < width else p for p in panels])


def write_path_preview(
    out_dir: Path | str,
    positions: np.ndarray,
    *,
    timestamps: list[float] | None = None,
    gps_positions: np.ndarray | None = None,
) -> dict:
    """Render the flight path and write ``1_drone_path.png``. Never raises."""
    out_dir = Path(out_dir)
    metrics = path_metrics(positions, timestamps)
    if metrics.get("n_points", 0) < 2:
        return {**metrics, "applied": False, "reason": "fewer than two camera positions"}
    metrics["applied"] = True

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        track = render_drone_path(positions, gps_positions=gps_positions)
        if track is not None:
            lines = [
                f"{metrics['n_points']} cameras over {metrics['path_length_m']:.0f} m flown"
                + (f" in {metrics['duration_s']:.0f} s" if metrics.get("duration_s") else ""),
                f"area {metrics['extent_m']['x']:.0f} x {metrics['extent_m']['y']:.0f} m; "
                f"altitude {metrics['altitude_m']['median']:.0f} m "
                f"({metrics['altitude_m']['min']:.0f}-{metrics['altitude_m']['max']:.0f})",
                f"spacing between cameras: {metrics['spacing_m']['min']:.1f} / "
                f"{metrics['spacing_m']['median']:.1f} / {metrics['spacing_m']['max']:.1f} m "
                f"(min/median/max)",
            ]
            ratio = metrics.get("spacing_ratio")
            if ratio and ratio > 2.0:
                lines.append(
                    f"WARNING: spacing varies {ratio:.1f}x across the flight. Windows holding the same"
                )
                lines.append(
                    "  number of frames will span very different distances, and a multi-view depth"
                )
                lines.append(
                    "  backbone's metric output degrades as its cameras spread out."
                )
            lines.append("colour runs dark->bright with time; grey = raw GPS track")
            sheet = _stack([
                _text_panel(lines, max(track.shape[1], 640), "1. DRONE PATH"),
                track,
                _altitude_strip(positions, max(track.shape[1], 640)),
            ])
            cv2.imwrite(str(out_dir / "1_drone_path.png"), sheet)
        (out_dir / "1_drone_path.json").write_text(json.dumps(metrics, indent=2))
    except Exception:
        logger.warning("flightview: writing the path preview failed; continuing", exc_info=True)

    logger.info(
        "DRONE PATH: %d cameras, %.0f m flown, %.0f x %.0f m area, altitude %.0f m, spacing %.1f m median",
        metrics["n_points"],
        metrics["path_length_m"],
        metrics["extent_m"]["x"],
        metrics["extent_m"]["y"],
        metrics["altitude_m"]["median"],
        metrics["spacing_m"]["median"] or float("nan"),
    )
    return metrics


# ---------------------------------------------------------------------------
# 2. what the camera actually saw
# ---------------------------------------------------------------------------


def coverage_metrics(footprints: list[np.ndarray | None], *, cell_m: float = _COVERAGE_CELL_M) -> dict:
    """Which ground the camera saw, and how many times.

    ``seen_once_pct`` is the number that decides whether a reconstruction
    can be checked at all: ground seen by one frame can be reconstructed
    but never cross-checked, so any error there is invisible and
    uncorrectable. ``holes`` is ground enclosed by the survey that no
    frame covers.
    """
    placed = [f for f in footprints if f is not None]
    out: dict = {"frames_total": len(footprints), "frames_placed": len(placed), "cell_m": cell_m}
    if not placed:
        return out

    allxy = np.vstack(placed)
    xmin, ymin = allxy.min(axis=0)
    xmax, ymax = allxy.max(axis=0)
    nx = max(1, int(np.ceil((xmax - xmin) / cell_m)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_m)))
    counts = np.zeros((ny, nx), dtype=np.int32)

    for f in placed:
        poly = np.stack(
            [
                np.clip((f[:, 0] - xmin) / cell_m, 0, nx - 1),
                np.clip((f[:, 1] - ymin) / cell_m, 0, ny - 1),
            ],
            axis=1,
        ).astype(np.int32)
        # Fill the real quadrilateral, not its bounding box: an oblique or
        # rotated frame's box overstates coverage by up to 2x, which would
        # hide exactly the gaps this is meant to find.
        mask = np.zeros((ny, nx), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 1)
        counts += mask

    covered = counts > 0
    total_cells = int(covered.sum())
    out.update(
        {
            "area_covered_ha": round(total_cells * cell_m * cell_m / 1e4, 3),
            "overlap_median": float(np.median(counts[covered])) if total_cells else 0.0,
            "overlap_min": int(counts[covered].min()) if total_cells else 0,
            "seen_once_pct": round(100.0 * float((counts == 1).sum()) / max(total_cells, 1), 2),
            "seen_once_cells": int((counts == 1).sum()),
            "cells_covered": total_cells,
            "grid": [int(ny), int(nx)],
            "origin_enu": [round(float(xmin), 2), round(float(ymin), 2)],
        }
    )

    # Holes: uncovered ground fully enclosed by covered ground. Flood-fill
    # from the border marks the outside, so whatever uncovered area is
    # left is interior -- ground the survey flew around but never saw.
    free = (~covered).astype(np.uint8)
    ff = free.copy()
    mask = np.zeros((ny + 2, nx + 2), np.uint8)
    for seed in ((0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)):
        if ff[seed[1], seed[0]]:
            cv2.floodFill(ff, mask, seed, 2)
    interior = int((ff == 1).sum())
    out["hole_cells"] = interior
    out["hole_area_ha"] = round(interior * cell_m * cell_m / 1e4, 3)
    return out


def render_coverage(
    footprints: list[np.ndarray | None],
    *,
    size: int = _PANEL,
    cell_m: float = _COVERAGE_CELL_M,
    camera_positions: np.ndarray | None = None,
) -> np.ndarray | None:
    """Coverage heatmap: how many frames see each patch of ground.

    Black is ground nothing saw. The colour ramp runs from one frame
    (dark) to many (bright), so a thin dark seam between two passes --
    the thing that becomes a hole in the mesh -- reads at a glance.
    """
    placed = [f for f in footprints if f is not None]
    if not placed:
        return None
    allxy = np.vstack(placed)
    xmin, ymin = allxy.min(axis=0)
    xmax, ymax = allxy.max(axis=0)
    nx = max(1, int(np.ceil((xmax - xmin) / cell_m)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_m)))
    counts = np.zeros((ny, nx), dtype=np.int32)
    for f in placed:
        poly = np.stack(
            [np.clip((f[:, 0] - xmin) / cell_m, 0, nx - 1), np.clip((f[:, 1] - ymin) / cell_m, 0, ny - 1)],
            axis=1,
        ).astype(np.int32)
        m = np.zeros((ny, nx), dtype=np.uint8)
        cv2.fillConvexPoly(m, poly, 1)
        counts += m

    hi = max(1, int(np.percentile(counts[counts > 0], 95))) if (counts > 0).any() else 1
    norm = np.clip(counts / hi, 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[counts == 0] = (0, 0, 0)
    img = np.flipud(img)  # north up

    scale = size / max(nx, ny)
    img = cv2.resize(img, (max(1, int(nx * scale)), max(1, int(ny * scale))), interpolation=cv2.INTER_NEAREST)

    if camera_positions is not None and len(camera_positions):
        cams = np.asarray(camera_positions, dtype=np.float64).reshape(-1, 3)
        h, w = img.shape[:2]
        pts = [
            (
                int(np.clip((c[0] - xmin) / max(xmax - xmin, 1e-6) * (w - 1), 0, w - 1)),
                int(np.clip(h - 1 - (c[1] - ymin) / max(ymax - ymin, 1e-6) * (h - 1), 0, h - 1)),
            )
            for c in cams
        ]
        for a, b in zip(pts[:-1], pts[1:], strict=False):
            cv2.line(img, a, b, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_coverage_preview(
    out_dir: Path | str,
    footprints: list[np.ndarray | None],
    *,
    camera_positions: np.ndarray | None = None,
    cell_m: float = _COVERAGE_CELL_M,
) -> dict:
    """Render the coverage mask and write ``2_coverage.png``. Never raises."""
    out_dir = Path(out_dir)
    metrics = coverage_metrics(footprints, cell_m=cell_m)
    if not metrics.get("cells_covered"):
        return {**metrics, "applied": False, "reason": "no frame could be placed on the ground"}
    metrics["applied"] = True

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        heat = render_coverage(footprints, cell_m=cell_m, camera_positions=camera_positions)
        if heat is not None:
            lines = [
                f"{metrics['frames_placed']}/{metrics['frames_total']} frames placed; "
                f"{metrics['area_covered_ha']:.2f} ha covered at {cell_m:.0f} m cells",
                f"a typical patch is seen by {metrics['overlap_median']:.0f} frames "
                f"(worst {metrics['overlap_min']})",
                f"{metrics['seen_once_pct']:.1f}% of covered ground is seen by only ONE frame "
                f"-- nothing can cross-check it",
            ]
            if metrics.get("hole_cells"):
                lines.append(
                    f"WARNING: {metrics['hole_area_ha']:.2f} ha of ground INSIDE the survey was "
                    "never seen (black patches enclosed by coverage)"
                )
            lines.append("dark = seen once, bright = seen many times, black = not seen; white = flight track")
            cv2.imwrite(
                str(out_dir / "2_coverage.png"),
                _stack([_text_panel(lines, max(heat.shape[1], 640), "2. GROUND COVERAGE (camera FOV)"), heat]),
            )
        (out_dir / "2_coverage.json").write_text(json.dumps(metrics, indent=2))
    except Exception:
        logger.warning("flightview: writing the coverage preview failed; continuing", exc_info=True)

    level = logger.warning if metrics.get("hole_cells") else logger.info
    level(
        "COVERAGE: %.2f ha, typical patch seen by %.0f frames, %.1f%% seen once, %.2f ha unseen holes",
        metrics["area_covered_ha"],
        metrics["overlap_median"],
        metrics["seen_once_pct"],
        metrics.get("hole_area_ha", 0.0),
    )
    return metrics
