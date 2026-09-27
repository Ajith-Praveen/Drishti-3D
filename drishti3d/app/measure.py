"""Measurements on a reconstructed surface, and their export to GIS formats.

Everything here works on plain arrays in the model's local frame (x east,
y north, z up, metres), so it is testable without a window:

- ``HeightGrid``: the top surface rasterised from the mesh vertices (highest
  vertex per cell), sampled bilinearly. Profiles and volumes read heights
  from it rather than ray-casting a many-million-face mesh per sample.
- ``elevation_profile``: heights along a polyline, with climb, descent and
  the steepest slope.
- ``cut_fill_volume``: material above and below a base plane fitted through
  the outline's own corner heights -- a stockpile, debris or a pit measured
  against the ground around it.
- ``to_geojson`` / ``to_kml``: every measurement in WGS84 longitude,
  latitude and elevation, for QGIS/ArcGIS and Google Earth.

A measurement is a plain dict (see ``Measurement``) so the viewer, the
exports and the tests share one shape.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from xml.sax.saxutils import escape

import numpy as np

#: Measurement kinds, in the order the Tools menu offers them.
KINDS = ("point", "distance", "area", "volume", "profile")


@dataclass
class Measurement:
    """One finished measurement: its surface points (local metres) and what was read off them."""

    kind: str
    points: np.ndarray
    value: float
    unit: str
    label: str
    warning: str = ""
    extra: dict = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def polyline_length(points: np.ndarray) -> float:
    """3D length of a polyline."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def polygon_area_xy(points: np.ndarray) -> float:
    """Planimetric (map) area of a closed polygon: shoelace on x, y."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


@dataclass
class HeightGrid:
    """Top-surface heights on a regular grid: ``z[row, col]`` at the centre of each cell, NaN where empty."""

    z: np.ndarray
    xmin: float
    ymax: float
    cell: float

    @classmethod
    def from_points(cls, xyz: np.ndarray, cell: float | None = None, max_cells: int = 16_000_000) -> HeightGrid:
        """Highest point per cell. ``cell`` defaults to ~1.2x the mean point spacing."""
        p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        p = p[np.isfinite(p).all(axis=1)]
        if len(p) == 0:
            raise ValueError("no surface points to measure on")
        lo, hi = p[:, :2].min(axis=0), p[:, :2].max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        if cell is None:
            cell = 1.2 * math.sqrt(float(span[0] * span[1]) / len(p))
        cell = max(float(cell), math.sqrt(float(span[0] * span[1]) / max_cells), 1e-3)
        nx, ny = int(np.ceil(span[0] / cell)) + 1, int(np.ceil(span[1] / cell)) + 1
        col = np.clip(((p[:, 0] - lo[0]) / cell).astype(np.int64), 0, nx - 1)
        row = np.clip(((hi[1] - p[:, 1]) / cell).astype(np.int64), 0, ny - 1)
        z = np.full((ny, nx), -np.inf)
        np.maximum.at(z, (row, col), p[:, 2])
        z[~np.isfinite(z)] = np.nan
        return cls(z=z, xmin=float(lo[0]), ymax=float(hi[1]), cell=float(cell))

    def _frac_index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return (xy[:, 0] - self.xmin) / self.cell - 0.5, (self.ymax - xy[:, 1]) / self.cell - 0.5

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """Bilinear height at each (x, y); where a corner is empty, the nearest filled cell; NaN off the surface."""
        c, r = self._frac_index(xy)
        ny, nx = self.z.shape
        c0 = np.clip(np.floor(c).astype(np.int64), 0, nx - 1)
        r0 = np.clip(np.floor(r).astype(np.int64), 0, ny - 1)
        c1, r1 = np.clip(c0 + 1, 0, nx - 1), np.clip(r0 + 1, 0, ny - 1)
        fc, fr = np.clip(c - c0, 0.0, 1.0), np.clip(r - r0, 0.0, 1.0)
        z00, z01, z10, z11 = self.z[r0, c0], self.z[r0, c1], self.z[r1, c0], self.z[r1, c1]
        out = (z00 * (1 - fc) * (1 - fr) + z01 * fc * (1 - fr) + z10 * (1 - fc) * fr + z11 * fc * fr)
        missing = ~np.isfinite(out)
        if missing.any():
            near = self.z[np.clip(np.rint(r[missing]).astype(np.int64), 0, ny - 1),
                          np.clip(np.rint(c[missing]).astype(np.int64), 0, nx - 1)]
            out[missing] = near
        inside = (c >= -0.5) & (c <= nx - 0.5) & (r >= -0.5) & (r <= ny - 0.5)
        out[~inside] = np.nan
        return out

    def cells_in_polygon(self, polygon_xy: np.ndarray) -> np.ndarray:
        """(ny, nx) bool: cells whose centre lies inside the polygon."""
        import cv2

        c, r = self._frac_index(np.asarray(polygon_xy, dtype=np.float64)[:, :2])
        mask = np.zeros(self.z.shape, dtype=np.uint8)
        # fillPoly works in pixel units with 4 fractional bits of precision.
        pts = np.round(np.stack([c, r], axis=1) * 16).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1, lineType=cv2.LINE_8, shift=4)
        return mask.astype(bool)


def elevation_profile(grid: HeightGrid, points: np.ndarray, step_m: float | None = None) -> dict:
    """Surface heights along a polyline, every ``step_m`` (default: one grid cell)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) < 2:
        raise ValueError("a profile needs at least two points")
    step = float(step_m or grid.cell)
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    samples, dist = [p[0, :2]], [0.0]
    for k, length in enumerate(seg):
        n = max(1, int(np.ceil(length / step)))
        t = np.arange(1, n + 1) / n
        samples.extend(p[k, :2] + t[:, None] * (p[k + 1, :2] - p[k, :2]))
        dist.extend(dist[-1] + t * length)
    xy = np.asarray(samples)
    d = np.asarray(dist, dtype=np.float64)
    z = grid.sample(xy)
    ok = np.isfinite(z)
    dz = np.diff(z)
    dd = np.diff(d)
    good = np.isfinite(dz) & (dd > 1e-9)
    slopes = np.degrees(np.arctan2(np.abs(dz[good]), dd[good])) if good.any() else np.zeros(0)
    return {
        "distance_m": d,
        "elevation_m": z,
        "xy": xy,
        "length_m": float(d[-1]),
        "climb_m": float(np.nansum(np.where(dz > 0, dz, 0.0))),
        "descent_m": float(np.nansum(np.where(dz < 0, -dz, 0.0))),
        "min_m": float(np.nanmin(z)) if ok.any() else float("nan"),
        "max_m": float(np.nanmax(z)) if ok.any() else float("nan"),
        "max_slope_deg": float(slopes.max()) if slopes.size else 0.0,
        "valid_fraction": float(ok.mean()),
    }


def cut_fill_volume(grid: HeightGrid, polygon: np.ndarray) -> dict:
    """Volume above and below a base plane fitted to the outline's corners.

    The base is the least-squares plane through the polygon's vertices (the
    ground around a stockpile, the rim of a pit). "Above" is material over
    that plane (a stockpile, debris); "below" is the void under it (a pit,
    a cutting). Cells inside the outline with no surface are excluded and
    counted in ``coverage``.
    """
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 3)
    if len(p) < 3:
        raise ValueError("a volume needs an outline of at least three points")
    A = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
    (a, b, c), *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
    inside = grid.cells_in_polygon(p)
    rows, cols = np.nonzero(inside)
    x = grid.xmin + (cols + 0.5) * grid.cell
    y = grid.ymax - (rows + 0.5) * grid.cell
    z = grid.z[rows, cols]
    ok = np.isfinite(z)
    dz = z[ok] - (a * x[ok] + b * y[ok] + c)
    cell_area = grid.cell * grid.cell
    above = float(np.clip(dz, 0, None).sum() * cell_area)
    below = float(np.clip(-dz, 0, None).sum() * cell_area)
    return {
        "above_m3": above,
        "below_m3": below,
        "net_m3": above - below,
        "area_m2": polygon_area_xy(p),
        "max_above_m": float(dz.max()) if dz.size else 0.0,
        "max_below_m": float(-dz.min()) if dz.size else 0.0,
        "coverage": float(ok.mean()) if ok.size else 0.0,
        "base_plane": [float(a), float(b), float(c)],
    }


# ---------------------------------------------------------------------------
# GIS export
# ---------------------------------------------------------------------------


def _geodetic(points: np.ndarray, origin) -> np.ndarray:
    """(N, 3) local metres -> (N, 3) [lon, lat, elevation m]."""
    from drishti3d.geometry.georef import enu_to_wgs84

    return enu_to_wgs84(np.asarray(points, dtype=np.float64).reshape(-1, 3), origin)


def _properties(m: Measurement) -> dict:
    props = {"kind": m.kind, "label": m.label, "value": round(float(m.value), 4), "unit": m.unit, "created": m.created}
    if m.warning:
        props["warning"] = m.warning
    for key, value in m.extra.items():
        if isinstance(value, (int, float, str, bool)) and not (isinstance(value, float) and not math.isfinite(value)):
            props[key] = round(value, 4) if isinstance(value, float) else value
    return props


def _profile_line(m: Measurement, origin) -> np.ndarray | None:
    """A profile exports its sampled surface line (so the elevations travel with it), not just the clicks."""
    xy, z = m.extra.get("_xy"), m.extra.get("_z")
    if xy is None or z is None:
        return None
    pts = np.c_[np.asarray(xy, dtype=np.float64), np.asarray(z, dtype=np.float64)]
    pts = pts[np.isfinite(pts).all(axis=1)]
    return _geodetic(pts, origin) if len(pts) >= 2 else None


def to_geojson(measurements: list[Measurement], origin) -> dict:
    """A GeoJSON FeatureCollection (RFC 7946: WGS84 lon, lat, elevation)."""
    features = []
    for m in measurements:
        llh = _geodetic(m.points, origin)
        coords = [[round(float(x), 8), round(float(y), 8), round(float(h), 3)] for x, y, h in llh]
        if m.kind == "point":
            geometry = {"type": "Point", "coordinates": coords[0]}
        elif m.kind in ("area", "volume"):
            geometry = {"type": "Polygon", "coordinates": [coords + [coords[0]]]}
        else:
            line = _profile_line(m, origin) if m.kind == "profile" else None
            if line is not None:
                coords = [[round(float(x), 8), round(float(y), 8), round(float(h), 3)] for x, y, h in line]
            geometry = {"type": "LineString", "coordinates": coords}
        features.append({"type": "Feature", "geometry": geometry, "properties": _properties(m)})
    return {"type": "FeatureCollection", "features": features}


def to_kml(measurements: list[Measurement], origin, name: str = "DRISHTI-3D measurements") -> str:
    """KML 2.2 for Google Earth: absolute (sea-level) elevations, one Placemark per measurement."""
    styles = {
        "point": ("ff00d7ff", 1.0), "distance": ("ff2bb3ff", 3.0), "area": ("ff66cc33", 2.0),
        "volume": ("ff3366ff", 2.0), "profile": ("ffff66cc", 3.0),
    }
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"<name>{escape(name)}</name>",
    ]
    for kind, (colour, width) in styles.items():
        out.append(
            f'<Style id="{kind}"><LineStyle><color>{colour}</color><width>{width}</width></LineStyle>'
            f"<PolyStyle><color>{'55' + colour[2:]}</color></PolyStyle>"
            f"<IconStyle><color>{colour}</color></IconStyle></Style>"
        )
    for m in measurements:
        llh = _geodetic(m.points, origin)
        if m.kind == "profile":
            line = _profile_line(m, origin)
            llh = line if line is not None else llh
        coords = " ".join(f"{x:.8f},{y:.8f},{h:.3f}" for x, y, h in llh)
        desc = "; ".join(f"{k}: {v}" for k, v in _properties(m).items() if k not in ("label", "kind"))
        out.append(f"<Placemark><name>{escape(m.label)}</name><description>{escape(desc)}</description>"
                   f"<styleUrl>#{m.kind}</styleUrl>")
        if m.kind == "point":
            out.append(f"<Point><altitudeMode>absolute</altitudeMode><coordinates>{coords}</coordinates></Point>")
        elif m.kind in ("area", "volume"):
            first = f"{llh[0][0]:.8f},{llh[0][1]:.8f},{llh[0][2]:.3f}"
            out.append("<Polygon><altitudeMode>absolute</altitudeMode><outerBoundaryIs><LinearRing>"
                       f"<coordinates>{coords} {first}</coordinates></LinearRing></outerBoundaryIs></Polygon>")
        else:
            out.append(f"<LineString><altitudeMode>absolute</altitudeMode><coordinates>{coords}</coordinates></LineString>")
        out.append("</Placemark>")
    out += ["</Document>", "</kml>"]
    return "\n".join(out)


def write_measurements(path, measurements: list[Measurement], origin) -> None:
    """``.geojson``/``.json`` or ``.kml`` by extension."""
    from pathlib import Path

    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".geojson", ".json"):
        path.write_text(json.dumps(to_geojson(measurements, origin), indent=1))
    elif ext == ".kml":
        path.write_text(to_kml(measurements, origin))
    else:
        raise ValueError(f"unsupported measurement export {ext!r}: use .geojson or .kml")
