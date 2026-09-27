"""Planimetric check of an exported orthomosaic against a reference orthophoto, tile by tile.

Independent of cameras: both images are resampled onto one map grid, split
into tiles, and each tile's shift is measured by phase correlation of edge
maps. Tiles with too little texture or a weak correlation peak are skipped
and counted. The distribution of the remaining shifts is how far features
in the model sit from the same features in the reference -- the model's
own horizontal accuracy (relative to the reference's; PNOA states ~0.5 m
RMSE for its orthophotos).

Usage
-----
    .venv/bin/python scripts/ortho_check.py RUN/output/orthomosaic.tif flight01/reference/ortho.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _grad(gray: np.ndarray) -> np.ndarray:
    import cv2

    g = gray.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def main() -> int:
    import cv2
    import rasterio
    from rasterio.warp import Resampling, reproject

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ortho", type=Path, help="the model's orthomosaic GeoTIFF")
    ap.add_argument("reference", type=Path, help="reference orthophoto GeoTIFF")
    ap.add_argument("--gsd", type=float, default=0.5, help="comparison grid, metres per pixel")
    ap.add_argument("--tile-m", type=float, default=60.0)
    ap.add_argument("--min-response", type=float, default=0.08, help="phase-correlation peak needed to trust a tile")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    with rasterio.open(args.ortho) as src:
        crs, b = src.crs, src.bounds
        gsd = args.gsd
        w, h = int((b.right - b.left) / gsd), int((b.top - b.bottom) / gsd)
        dst_t = rasterio.transform.from_origin(b.left, b.top, gsd, gsd)
        ours = np.zeros((3, h, w), np.uint8)
        for k in range(3):
            reproject(rasterio.band(src, k + 1), ours[k], dst_transform=dst_t, dst_crs=crs, resampling=Resampling.average)
    ref = np.zeros((3, h, w), np.uint8)
    with rasterio.open(args.reference) as src:
        for k in range(3):
            reproject(rasterio.band(src, k + 1), ref[k], dst_transform=dst_t, dst_crs=crs, resampling=Resampling.average)
    a = cv2.cvtColor(np.moveaxis(ours, 0, -1), cv2.COLOR_RGB2GRAY)
    r = cv2.cvtColor(np.moveaxis(ref, 0, -1), cv2.COLOR_RGB2GRAY)
    covered = (ours.max(axis=0) > 0) & (ref.max(axis=0) > 0)

    tn = int(args.tile_m / gsd)
    win = cv2.createHanningWindow((tn, tn), cv2.CV_32F)
    shifts, skipped = [], 0
    for ty in range(0, h - tn + 1, tn):
        for tx in range(0, w - tn + 1, tn):
            if covered[ty : ty + tn, tx : tx + tn].mean() < 0.95:
                continue
            ga, gr = _grad(a[ty : ty + tn, tx : tx + tn]), _grad(r[ty : ty + tn, tx : tx + tn])
            if ga.std() < 2.0 or gr.std() < 2.0:
                skipped += 1
                continue
            (dx, dy), resp = cv2.phaseCorrelate(gr * win, ga * win)
            if resp < args.min_response:
                skipped += 1
                continue
            shifts.append((dx * gsd, -dy * gsd, resp))  # metres the model is east/north of the reference
    if not shifts:
        sys.exit("no tile produced a trustworthy correlation")
    s = np.array(shifts)
    mag = np.hypot(s[:, 0], s[:, 1])
    med = np.median(s[:, :2], axis=0)
    rel = np.hypot(s[:, 0] - med[0], s[:, 1] - med[1])
    out = {
        "tiles_measured": len(s),
        "tiles_skipped_low_texture_or_weak_peak": skipped,
        "tile_m": args.tile_m,
        "median_offset_east_m": round(float(med[0]), 3),
        "median_offset_north_m": round(float(med[1]), 3),
        "horizontal_error_median_m": round(float(np.median(mag)), 3),
        "horizontal_error_p90_m": round(float(np.percentile(mag, 90)), 3),
        "within_1m_pct": round(float(np.mean(mag <= 1.0) * 100), 1),
        "internal_distortion_median_m": round(float(np.median(rel)), 3),
    }
    print(
        f"{out['tiles_measured']} tiles of {args.tile_m:g} m measured ({skipped} skipped): model vs reference "
        f"median {out['horizontal_error_median_m']:.2f} m, p90 {out['horizontal_error_p90_m']:.2f} m, "
        f"<=1 m {out['within_1m_pct']:.0f}% of tiles; common offset E {med[0]:+.2f} N {med[1]:+.2f} m; "
        f"internal distortion (after removing it) median {out['internal_distortion_median_m']:.2f} m"
    )
    if args.json is not None:
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
