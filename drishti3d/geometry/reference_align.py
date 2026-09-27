"""Align the georeferenced model to a reference orthophoto and elevation model.

Why
---
GPS georeferencing leaves the whole model offset by the GPS error: a few
metres for standalone GNSS, and no amount of bundle adjustment removes it
because every camera shares it. A surveyed reference of the same ground --
a national orthophoto (Spain's PNOA is 25 cm), and an elevation model --
fixes that offset without RTK or ground control points.

This is the render-and-match idea from UAVD4L (3DV 2024), specialised to
nadir survey footage: for a straight-down camera the "rendered view" of
our model that corresponds to the reference is simply its top-down
orthographic render. So:

1. **Render** the model top-down on the reference's own grid (``gsd``
   metres per pixel): the highest point per cell, in colour.
2. **Match** that render against the reference orthophoto with DISK +
   LightGlue (``geometry.learned_matching``), fit a 2-D similarity with
   RANSAC in metres, and fall back to phase correlation of edge maps when
   too few matches survive (textureless farmland in a bad season).
3. **Vertical**: the median difference between our surface and the
   reference elevation model over the model's footprint.
4. **Apply** the correction to every point exactly (ENU -> reference CRS ->
   corrected -> back to ENU), and to the cameras.

What it refuses
---------------
A horizontal fix is only applied with at least ``min_inliers`` RANSAC
inliers, a scale within ``max_scale_dev`` of 1 and a rotation under
``max_rot_deg`` -- GPS error is a translation, so a large rotation or
scale means the match is wrong, not the model. A correction larger than
``max_shift_m`` is refused too. Refusals are reported, never silent.

Honest limits
-------------
The result can only be as good as the reference: PNOA's stated
planimetric accuracy is ~0.5 m RMSE; Copernicus GLO-30 is a 30 m surface
model with a few metres of vertical error, so the vertical fix is coarse.
ETRS89 (the reference's datum) and WGS84 differ by ~0.9 m in 2026; pyproj
treats them as identical by default and this module does not model it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ReferenceAlignment", "align_to_reference", "render_topdown"]


@dataclass
class ReferenceAlignment:
    applied: bool
    horizontal: dict = field(default_factory=dict)
    vertical: dict = field(default_factory=dict)
    failure: str | None = None
    # Similarity in the reference CRS: p' = s * R(theta) @ (p - c) + c + t
    scale: float = 1.0
    rot_deg: float = 0.0
    t: tuple[float, float] = (0.0, 0.0)
    centre: tuple[float, float] = (0.0, 0.0)
    dz: float = 0.0

    def as_dict(self) -> dict:
        return {
            "applied": self.applied,
            "failure": self.failure,
            "shift_east_m": round(self.t[0], 3),
            "shift_north_m": round(self.t[1], 3),
            "shift_up_m": round(self.dz, 3),
            "rotation_deg": round(self.rot_deg, 4),
            "scale": round(self.scale, 6),
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }

    def apply_xy(self, xy: np.ndarray) -> np.ndarray:
        c = np.asarray(self.centre)
        th = np.radians(self.rot_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return self.scale * (np.asarray(xy) - c) @ R.T + c + np.asarray(self.t)


def render_topdown(xy: np.ndarray, z: np.ndarray, rgb: np.ndarray, x0: float, y1: float, gsd: float, w: int, h: int):
    """Highest point per cell, north-up, row 0 at ``y1``. Returns ``(image uint8 HxWx3, filled mask)``."""
    import cv2

    col = np.floor((xy[:, 0] - x0) / gsd).astype(np.int64)
    row = np.floor((y1 - xy[:, 1]) / gsd).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    col, row, z, rgb = col[ok], row[ok], z[ok], rgb[ok]
    cell = row * w + col
    order = np.lexsort((z, cell))  # by cell, then height ascending -> last per cell is the top
    last = np.ones(order.size, dtype=bool)
    last[:-1] = cell[order][1:] != cell[order][:-1]
    top = order[last]
    img = np.zeros((h * w, 3), dtype=np.uint8)
    img[cell[top]] = rgb[top]
    mask = np.zeros(h * w, dtype=np.uint8)
    mask[cell[top]] = 1
    img, mask = img.reshape(h, w, 3), mask.reshape(h, w)
    # Close pinholes between splatted points so features see surfaces, not dots.
    holes = (mask == 0).astype(np.uint8)
    img = cv2.inpaint(img, holes, 3, cv2.INPAINT_TELEA) if holes.any() else img
    filled = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    img[~filled] = 0
    return img, filled


def _bare_ground_mask(xy, z, x0, y1, gsd, w, h, dem_path, crs, bounds, tol_m: float = 1.5) -> np.ndarray | None:
    """Our render's cells whose top surface agrees with the reference terrain model (after its median offset)."""
    dem, _, _ = _read_reference(dem_path, crs, bounds, gsd)
    col = np.floor((xy[:, 0] - x0) / gsd).astype(np.int64)
    row = np.floor((y1 - xy[:, 1]) / gsd).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    top = np.full((h, w), np.nan)
    np.fmax.at(top, (row[ok], col[ok]), np.asarray(z, dtype=np.float64)[ok])
    d = top - dem[0]
    known = np.isfinite(d)
    if known.sum() < 100:
        return None
    # The model-to-DEM vertical offset, measured on the ground: the mode of
    # the LOWER half of the differences. The median would land between ground
    # and canopy in woodland and select neither.
    lower = d[known][d[known] <= np.median(d[known])]
    hist, edges = np.histogram(lower, bins=max(10, int(np.ptp(lower) / 0.25) + 1))
    k = int(np.argmax(hist))
    offset = 0.5 * (edges[k] + edges[k + 1])
    return known & (np.abs(d - offset) < tol_m)


def _read_reference(path, crs, bounds, gsd):
    """Reference raster resampled onto our grid (``bounds`` in ``crs``). Returns (array, w, h)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.transform import from_origin

    x0, y0, x1, y1 = bounds
    w, h = int(np.ceil((x1 - x0) / gsd)), int(np.ceil((y1 - y0) / gsd))
    with rasterio.open(path) as src:
        out = np.zeros((src.count, h, w), dtype=np.float32)
        for b in range(src.count):
            reproject(
                rasterio.band(src, b + 1), out[b], dst_transform=from_origin(x0, y1, gsd, gsd), dst_crs=crs,
                resampling=Resampling.bilinear, dst_nodata=np.nan,
            )
    return out, w, h


def _match_translation(
    ours: np.ndarray, ref: np.ndarray, gsd: float, keep: np.ndarray | None = None, tol_m: float = 1.5, seed: int = 0
) -> tuple[np.ndarray | None, dict]:
    """Translation (2x3, pixels) taking OUR render onto the reference, from DISK+LightGlue matches.

    One-point RANSAC over the match displacements, refined as the median of
    the inliers' displacements. ``keep`` (our render's pixels) restricts the
    evidence to bare ground when it leaves enough matches: the reference is
    rectified on a terrain model, so tree crowns and roofs in it are shifted
    by relief displacement our true orthophoto does not have.
    """
    from drishti3d.geometry.learned_matching import LightGlueMatcher, detect_disk

    diag: dict = {"method": "lightglue_translation"}
    fa = detect_disk(ours, max_features=4096, detect_scale=1.0)
    fb = detect_disk(ref, max_features=4096, detect_scale=1.0)
    m = LightGlueMatcher().match(fa, fb)
    src = fa.keypoints[m.query_idx].astype(np.float64)
    dst = fb.keypoints[m.train_idx].astype(np.float64)
    diag["matches"] = int(len(src))
    if keep is not None and len(src):
        h, w = keep.shape
        on = keep[np.clip(src[:, 1].astype(int), 0, h - 1), np.clip(src[:, 0].astype(int), 0, w - 1)]
        diag["bare_ground_matches"] = int(on.sum())
        if on.sum() >= 40:
            src, dst = src[on], dst[on]
            diag["evidence"] = "bare_ground"
        else:
            diag["evidence"] = "all (too few bare-ground matches)"
    if len(src) < 8:
        return None, diag
    disp = dst - src
    tol = tol_m / gsd
    rng = np.random.default_rng(seed)
    best, best_inl = None, None
    for i in rng.choice(len(disp), size=min(len(disp), 500), replace=False):
        inl = np.linalg.norm(disp - disp[i], axis=1) <= tol
        if best_inl is None or inl.sum() > best_inl.sum():
            best, best_inl = disp[i], inl
    t = np.median(disp[best_inl], axis=0)
    inl = np.linalg.norm(disp - t, axis=1) <= tol
    # Mean, not median, of the final inliers: keypoints sit on the pixel grid,
    # so a median of their displacements is quantised to whole pixels (0.5 m).
    t = disp[inl].mean(axis=0)
    res = np.linalg.norm(disp[inl] - t, axis=1) * gsd
    # How much of the area the evidence comes from (3x3 tiles holding inliers).
    ys, xs = src[inl, 1], src[inl, 0]
    h_, w_ = ours.shape[:2]
    tiles = {(int(3 * y / h_), int(3 * x / w_)) for y, x in zip(ys, xs, strict=True)}
    diag.update(inliers=int(inl.sum()), residual_rms_m=round(float(np.sqrt(np.mean(res**2))), 3), tiles_with_inliers=len(tiles))
    return np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]]]), diag


def _match_similarity(ours: np.ndarray, ref: np.ndarray, gsd: float) -> tuple[np.ndarray | None, dict]:
    """2x3 similarity (pixels) taking OUR render onto the reference, via DISK+LightGlue, else phase correlation."""
    import cv2

    diag: dict = {}
    try:
        from drishti3d.geometry.learned_matching import LightGlueMatcher, detect_disk

        fa = detect_disk(ours, max_features=4096, detect_scale=1.0)
        fb = detect_disk(ref, max_features=4096, detect_scale=1.0)
        m = LightGlueMatcher().match(fa, fb)
        diag["matches"] = int(len(m))
        if len(m) >= 8:
            src = fa.keypoints[m.query_idx].astype(np.float32)
            dst = fb.keypoints[m.train_idx].astype(np.float32)
            A, inl = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=max(2.0, 1.5 / gsd), maxIters=5000, confidence=0.999
            )
            if A is not None and inl is not None:
                inl = inl.ravel().astype(bool)
                res = np.linalg.norm((src[inl] @ A[:, :2].T + A[:, 2]) - dst[inl], axis=1) * gsd
                diag.update(method="lightglue", inliers=int(inl.sum()), residual_rms_m=round(float(np.sqrt(np.mean(res**2))), 3))
                return A, diag
    except Exception as exc:  # kornia missing, weights unavailable offline, ...
        diag["lightglue_error"] = str(exc)[:200]

    # Fallback: translation only, phase correlation on gradient magnitude.
    def grad(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    (dx, dy), resp = cv2.phaseCorrelate(grad(ours), grad(ref))
    diag.update(method="phase_correlation", response=round(float(resp), 4))
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]]), diag


def align_to_reference(
    xyz_enu: np.ndarray,
    rgb: np.ndarray,
    origin,
    ortho_path: str,
    dem_path: str | None = None,
    *,
    gsd: float = 0.5,
    margin_m: float = 60.0,
    min_inliers: int = 25,
    min_phase_response: float = 0.05,
    max_rot_deg: float = 3.0,
    max_scale_dev: float = 0.02,
    max_shift_m: float = 30.0,
    fit: str = "translation",
) -> ReferenceAlignment:
    """Measure the model's offset against a reference orthophoto (+ DEM). See the module docstring.

    ``fit="translation"`` (default) solves the horizontal shift only. GPS
    error is a translation, and the GPS/bundle-adjustment fit already fixes
    scale to ~0.1% and rotation; a free similarity fitted to matches that
    crowd into one textured corner let the scale swing 1.3% between two runs
    of one model on flight01 -- 5 m at 400 m from the centre.
    ``"similarity"`` restores the 4-DoF fit.
    """
    import pyproj
    import rasterio

    from drishti3d.geometry.georef import enu_to_wgs84

    with rasterio.open(ortho_path) as src:
        ref_crs = src.crs.to_string()
    to_ref = pyproj.Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
    llh = enu_to_wgs84(np.asarray(xyz_enu, dtype=np.float64).reshape(-1, 3), origin)
    ex, ny = to_ref.transform(llh[:, 0], llh[:, 1])
    xy = np.c_[ex, ny]
    z = llh[:, 2]

    x0, y0 = xy.min(0) - margin_m
    x1, y1 = xy.max(0) + margin_m
    out = ReferenceAlignment(applied=False, centre=(float(xy[:, 0].mean()), float(xy[:, 1].mean())))

    ref, w, h = _read_reference(ortho_path, ref_crs, (x0, y0, x1, y1), gsd)
    ref_rgb = np.nan_to_num(ref[:3]).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    ours, filled = render_topdown(xy, z, np.asarray(rgb, dtype=np.uint8).reshape(-1, 3), x0, y1, gsd, w, h)
    out.horizontal["render_coverage"] = round(float(filled.mean()), 3)

    if fit == "translation":
        keep = _bare_ground_mask(xy, z, x0, y1, gsd, w, h, dem_path, ref_crs, (x0, y0, x1, y1)) if dem_path else None
        try:
            A, diag = _match_translation(ours, ref_rgb, gsd, keep=keep)
        except Exception as exc:  # kornia missing, weights unavailable offline, ...
            A, diag = _match_similarity(ours, ref_rgb, gsd)
            diag["translation_error"] = str(exc)[:200]
    else:
        A, diag = _match_similarity(ours, ref_rgb, gsd)
    out.horizontal.update(diag)
    if A is None:
        out.failure = "no match"
        return out
    scale = float(np.hypot(A[0, 0], A[1, 0]))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    # Pixel similarity -> metric similarity about the model centre. Rows grow
    # southward, so image rotation is the negative of map rotation.
    cx, cy = out.centre
    pc = np.array([(cx - x0) / gsd, (y1 - cy) / gsd])
    moved = A[:, :2] @ pc + A[:, 2]
    tx, ty = (moved[0] - pc[0]) * gsd, -(moved[1] - pc[1]) * gsd

    reasons = []
    if str(diag.get("method", "")).startswith("lightglue") and diag.get("inliers", 0) < min_inliers:
        reasons.append(f"only {diag.get('inliers', 0)} RANSAC inliers (< {min_inliers})")
    if diag.get("method") == "phase_correlation" and diag.get("response", 0) < min_phase_response:
        reasons.append(f"phase-correlation response {diag.get('response')} below {min_phase_response}")
    if abs(rot) > max_rot_deg:
        reasons.append(f"rotation {rot:.2f} deg exceeds {max_rot_deg}")
    if abs(scale - 1.0) > max_scale_dev:
        reasons.append(f"scale {scale:.4f} deviates more than {max_scale_dev:.0%}")
    if np.hypot(tx, ty) > max_shift_m:
        reasons.append(f"shift {np.hypot(tx, ty):.1f} m exceeds {max_shift_m} m")
    if reasons:
        out.failure = "; ".join(reasons)
        return out
    out.scale, out.rot_deg, out.t = scale, -rot, (float(tx), float(ty))

    if dem_path:
        dem, _, _ = _read_reference(dem_path, ref_crs, (x0, y0, x1, y1), gsd)
        corrected = out.apply_xy(xy)
        col = np.clip(((corrected[:, 0] - x0) / gsd).astype(int), 0, w - 1)
        row = np.clip(((y1 - corrected[:, 1]) / gsd).astype(int), 0, h - 1)
        ref_z = dem[0][row, col]
        ok = np.isfinite(ref_z)
        if ok.sum() > 1000:
            # Compare like with like per 30 m cell: GLO-30 is a radar surface
            # model that averages its cell, and a national terrain model is
            # the ground -- over bare fields both are the cell's typical
            # height, i.e. OUR median there. The upper envelope used before
            # (p90) measured shrubs and embankments against that average and
            # left the aligned bare ground 1.7 m low on PinPoint flight01.
            key = (row[ok] // int(30 / gsd)) * 100000 + (col[ok] // int(30 / gsd))
            order = np.argsort(key)
            ks, zs, rs = key[order], z[ok][order], ref_z[ok][order]
            bounds = np.r_[0, np.nonzero(np.diff(ks))[0] + 1, ks.size]
            diffs = [np.median(zs[a:b]) - np.median(rs[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 20]
            if diffs:
                out.dz = -float(np.median(diffs))
                out.vertical = {"cells": len(diffs), "offset_m": round(-out.dz, 3), "spread_m": round(float(np.std(diffs)), 3)}

    out.applied = True
    logger.info(
        "reference align: shift E %+.2f m, N %+.2f m, U %+.2f m, rot %+.3f deg (%s)",
        out.t[0], out.t[1], out.dz, out.rot_deg, out.horizontal.get("method"),
    )
    # Remember how to go back to ENU for the caller.
    out._ctx = (to_ref, ref_crs, origin)
    return out


def apply_to_enu(alignment: ReferenceAlignment, xyz_enu: np.ndarray) -> np.ndarray:
    """Apply a measured alignment to ENU points exactly (through the reference CRS and back)."""
    import pyproj

    from drishti3d.geometry.georef import enu_to_wgs84, wgs84_to_enu

    to_ref, ref_crs, origin = alignment._ctx
    from_ref = pyproj.Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
    xyz = np.asarray(xyz_enu, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] == 0:
        return xyz.copy()
    llh = enu_to_wgs84(xyz, origin)
    ex, ny = to_ref.transform(llh[:, 0], llh[:, 1])
    moved = alignment.apply_xy(np.c_[ex, ny])
    lon, lat = from_ref.transform(moved[:, 0], moved[:, 1])
    return wgs84_to_enu(np.c_[lon, lat, llh[:, 2] + alignment.dz], origin)
