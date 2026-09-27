"""Fetch reference data for ``geometry.reference_align``: an orthophoto and an elevation model.

Two sources, both open:

- **Orthophoto** from any OGC WMS that serves JPEG/PNG in a projected CRS
  (default: Spain's IGN PNOA, 25 cm, CC-BY 4.0). Requested in tiles no
  larger than the server's limit and stitched into one GeoTIFF whose
  georeferencing comes from the request bbox, so no world files are needed.
- **Elevation** from the Copernicus GLO-30 DEM on AWS (global, 30 m,
  free), the tile(s) covering the bbox, cropped and reprojected to the
  orthophoto's CRS.

For an air-gapped deployment skip this script: hand ``reference_align``
your own orthophoto/DEM GeoTIFFs (e.g. national survey products).

Usage
-----
    .venv/bin/python scripts/fetch_reference.py --crs EPSG:25830 \\
        --bbox 686880,4626459,688089,4627729 --gsd 0.3 --out flight01/reference
"""

from __future__ import annotations

import argparse
import io
import math
import urllib.request
from pathlib import Path

import numpy as np

PNOA_WMS = "https://www.ign.es/wms-inspire/pnoa-ma"
PNOA_LAYER = "OI.OrthoimageCoverage"
COP_DEM = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "drishti3d-reference-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
        if r.headers.get_content_type().endswith("xml"):
            raise RuntimeError(f"server returned an error document for {url}:\n{data[:400]!r}")
        return data


def fetch_ortho(wms: str, layer: str, crs: str, bbox, gsd: float, out: Path, max_px: int = 2048) -> Path:
    import rasterio
    from PIL import Image
    from rasterio.transform import from_origin

    x0, y0, x1, y1 = bbox
    width, height = int(math.ceil((x1 - x0) / gsd)), int(math.ceil((y1 - y0) / gsd))
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    for r0 in range(0, height, max_px):
        for c0 in range(0, width, max_px):
            w, h = min(max_px, width - c0), min(max_px, height - r0)
            tb = (x0 + c0 * gsd, y1 - (r0 + h) * gsd, x0 + (c0 + w) * gsd, y1 - r0 * gsd)
            url = (
                f"{wms}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES=&CRS={crs}"
                f"&BBOX={tb[0]:.3f},{tb[1]:.3f},{tb[2]:.3f},{tb[3]:.3f}&WIDTH={w}&HEIGHT={h}&FORMAT=image/jpeg"
            )
            tile = np.asarray(Image.open(io.BytesIO(_get(url))).convert("RGB"))
            mosaic[r0 : r0 + h, c0 : c0 + w] = tile[:h, :w]
            print(f"  ortho tile {r0 // max_px},{c0 // max_px}: {w}x{h}")
    path = out / "ortho.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=crs,
        transform=from_origin(x0, y1, gsd, gsd), compress="jpeg", photometric="ycbcr",
    ) as dst:
        dst.write(mosaic.transpose(2, 0, 1))
    return path


def _cop_tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def fetch_dem(crs: str, bbox, out: Path, res: float = 10.0) -> Path:
    import pyproj
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    x0, y0, x1, y1 = bbox
    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = to_ll.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    width, height = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    dem = np.full((height, width), np.nan, dtype=np.float32)
    dst_t = from_origin(x0, y1, res, res)
    for la in range(math.floor(min(lats)), math.floor(max(lats)) + 1):
        for lo in range(math.floor(min(lons)), math.floor(max(lons)) + 1):
            name = _cop_tile_name(la, lo)
            print(f"  DEM tile {name}")
            with rasterio.open(COP_DEM.format(name=name)) as src:
                part = np.full_like(dem, np.nan)
                reproject(
                    rasterio.band(src, 1), part, dst_transform=dst_t, dst_crs=crs, resampling=Resampling.bilinear,
                    dst_nodata=np.nan,
                )
                dem = np.where(np.isnan(dem), part, dem)
    path = out / "dem.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32", crs=crs,
        transform=dst_t, nodata=np.nan,
    ) as dst:
        dst.write(dem, 1)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crs", required=True, help="projected CRS of the bbox, e.g. EPSG:25830")
    ap.add_argument("--bbox", required=True, help="xmin,ymin,xmax,ymax in --crs")
    ap.add_argument("--gsd", type=float, default=0.3, help="orthophoto ground sample distance (m)")
    ap.add_argument("--wms", default=PNOA_WMS)
    ap.add_argument("--layer", default=PNOA_LAYER)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    bbox = [float(v) for v in args.bbox.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    print("orthophoto:", fetch_ortho(args.wms, args.layer, args.crs, bbox, args.gsd, args.out))
    print("elevation:", fetch_dem(args.crs, bbox, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
