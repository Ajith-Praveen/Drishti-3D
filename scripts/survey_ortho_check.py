"""Camera-free survey accuracy: where each surveyed feature actually sits in an orthomosaic.

``score_flight01.py`` casts each surveyed pixel's ray from a camera at the
survey frame. Survey frames fall between keyframes, so that camera is
interpolated, and its error (metres, in banking flight) is part of every
score. This check does not trust that camera. It only uses it to cut an
approximately rectified patch of the survey frame around the surveyed
pixel, then finds that patch in the orthomosaic by normalised
cross-correlation. The feature's position in the orthomosaic minus its
surveyed position is the model's own horizontal error at that point:
what someone measuring on the delivered orthomosaic gets.

Run it on the IGN reference orthophoto too (``--ortho flight01/reference/ortho.tif``):
that is the method's floor (the reference's own accuracy plus matching).

Usage
-----
    .venv/bin/python scripts/survey_ortho_check.py RUN/output --clip-start 120.46
    .venv/bin/python scripts/survey_ortho_check.py RUN/output --clip-start 120.46 --ortho flight01/reference/ortho.tif
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLIGHT = ROOT / "flight01"


class _Raster:
    """A single-band view of a GeoTIFF, sampled bilinearly at map coordinates (NaN outside)."""

    def __init__(self, path: Path):
        import cv2
        import rasterio

        with rasterio.open(path) as src:
            data = src.read()
            self.transform, nodata = src.transform, src.nodata
        if data.shape[0] >= 3:
            img = cv2.cvtColor(np.ascontiguousarray(np.moveaxis(data[:3], 0, -1)), cv2.COLOR_RGB2GRAY).astype(np.float32)
            img[(data[:3] == 0).all(axis=0)] = np.nan  # black = no data in an orthomosaic
        else:
            img = data[0].astype(np.float32)
            if nodata is not None:
                img[data[0] == nodata] = np.nan
        self.img = img

    def sample(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        import cv2

        col, row = ~self.transform * (e, n)
        return cv2.remap(
            self.img, (col - 0.5).astype(np.float32), (row - 0.5).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
        )


def _match(search: np.ndarray, tpl: np.ndarray, ambiguity_px: float = 1e9, ambiguity_ncc: float = 0.05):
    """(dx, dy) pixels of ``tpl``'s best match relative to the centre of ``search``, the peak NCC and the
    distance (px) to the farthest position scoring within ``ambiguity_ncc`` of the peak.

    A feature on a hedge, a road or a field edge matches equally well
    anywhere along the line (the aperture problem): its match is a guess
    along the line, and that spread is reported so the caller can drop it.
    """
    import cv2

    if not np.isfinite(tpl).all():
        return None
    s = np.where(np.isfinite(search), search, np.nanmean(search) if np.isfinite(search).any() else 0.0)
    res = cv2.matchTemplate(s.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, peak, _, (x, y) = cv2.minMaxLoc(res)
    fx = fy = 0.0
    if 0 < x < res.shape[1] - 1:  # parabolic sub-pixel refinement
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
        fx = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
    if 0 < y < res.shape[0] - 1:
        a, b, c = res[y - 1, x], res[y, x], res[y + 1, x]
        fy = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) < 0 else 0.0
    near_y, near_x = np.nonzero(res >= peak - ambiguity_ncc)
    spread = float(np.hypot(near_x - x, near_y - y).max()) if near_x.size else 0.0
    cx, cy = (res.shape[1] - 1) / 2.0, (res.shape[0] - 1) / 2.0
    return x + fx - cx, y + fy - cy, float(peak), spread


def main() -> int:
    import cv2
    from scipy.spatial.transform import Rotation, Slerp

    from drishti3d.export.bundle_export import _enu_to_map_fn
    from drishti3d.geometry.georef import wgs84_to_enu
    from drishti3d.ingest.video import VideoSource
    from drishti3d.types import GeoPoint

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_output", type=Path, help="the run's output/ directory")
    ap.add_argument("--clip-start", type=float, required=True, help="mkv time of the clip's first frame (flight01: 120.46)")
    ap.add_argument("--ortho", type=Path, default=None, help="orthophoto to test (default: RUN/output/orthomosaic.tif)")
    ap.add_argument("--video", type=Path, default=FLIGHT / "flight01_120_480.mp4")
    ap.add_argument("--points", type=Path, default=FLIGHT / "points.csv")
    ap.add_argument("--subset", choices=("all", "nadir"), default="nadir")
    ap.add_argument("--gsd", type=float, default=0.2, help="comparison grid, metres per pixel")
    ap.add_argument("--template-m", type=float, default=6.0, help="half-size of the frame patch matched")
    ap.add_argument("--search-m", type=float, default=30.0, help="half-size of the searched orthophoto window")
    ap.add_argument("--min-ncc", type=float, default=0.5)
    ap.add_argument(
        "--max-ambiguity-m", type=float, default=1.5,
        help="drop a match whose correlation stays within 0.05 of its peak this far away (linear features)",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = args.run_output
    geo = json.loads((out / "georef.json").read_text())
    o = geo["origin"]
    origin = GeoPoint(lat=o["lat"], lon=o["lon"], alt_msl=o["alt_msl"])
    to_map = _enu_to_map_fn(origin, geo["map_crs"])
    ortho = _Raster(args.ortho or out / "orthomosaic.tif")
    dsm = _Raster(out / "dsm.tif")
    # Heights only rectify the frame patch: fill unobserved cells from the nearest observed one.
    from scipy import ndimage

    holes = ~np.isfinite(dsm.img)
    if holes.any() and (~holes).any():
        _, (ir, ic) = ndimage.distance_transform_edt(holes, return_indices=True)
        dsm.img = dsm.img[ir, ic]

    cams = sorted(json.loads((out / "cameras.json").read_text())["cameras"], key=lambda c: c["timestamp_s"])
    times = np.array([c["timestamp_s"] for c in cams])
    frames = np.array([c["frame_index"] for c in cams], dtype=np.float64)
    pos = np.array([c["t_world"] for c in cams], dtype=np.float64)
    slerp = Slerp(times, Rotation.from_matrix(np.array([c["R_world_from_cam"] for c in cams])))
    video = VideoSource(args.video)

    rows = list(csv.DictReader(args.points.open()))
    if args.subset == "nadir":
        rows = [r for r in rows if r.get("nadir_ok") == "1"]
    g, T, S = args.gsd, args.template_m, args.search_m
    results, skipped = [], {"outside": 0, "no_ground": 0, "off_frame": 0, "weak": 0, "ambiguous": 0}
    for r in rows:
        tv = float(r["tv_s"]) - args.clip_start
        if not (times[0] <= tv <= times[-1]):
            skipped["outside"] += 1
            continue
        C = np.array([np.interp(tv, times, a) for a in pos.T])
        R = slerp([tv]).as_matrix()[0]
        k = int(np.argmin(np.abs(times - tv)))
        intr = cams[k]["intrinsics"]
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
        dist = np.asarray(intr.get("dist_coeffs") or [0, 0, 0, 0, 0], dtype=np.float64)
        sx, sy = intr["width"] / float(r["img_w"]), intr["height"] / float(r["img_h"])
        u = (float(r["img_w"]) / 2 + float(r["off_x_px"])) * sx
        v = (float(r["img_h"]) / 2 + float(r["off_y_px"])) * sy
        und = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, dist)[0, 0]
        d = R @ np.array([und[0], und[1], 1.0])
        # Ray onto the model's DSM: a few fixed-point steps from the median ground.
        z = float(np.nanmedian(dsm.img)) - float(to_map(C[None])[0][2] - C[2])  # map height -> ENU z
        hit = None
        for _ in range(8):
            t = (z - C[2]) / d[2]
            hit = C + t * d
            em = to_map(hit[None])[0]
            zs = dsm.sample(np.array([[em[0]]]), np.array([[em[1]]]))[0, 0]
            if not np.isfinite(zs):
                hit = None
                break
            z = float(zs) - float(em[2] - hit[2])  # DSM heights are ellipsoidal map heights
        if hit is None:
            skipped["no_ground"] += 1
            continue

        # Local north-up grid around the hit, in ENU; its map coordinates for sampling.
        ax = np.arange(-S, S + g / 2, g)
        X, Y = np.meshgrid(hit[0] + ax, hit[1] - ax)
        enu = np.stack([X.ravel(), Y.ravel(), np.full(X.size, hit[2])], axis=1)
        m = to_map(enu)
        E, N = m[:, 0].reshape(X.shape), m[:, 1].reshape(X.shape)
        search = ortho.sample(E, N)
        zg = dsm.sample(E, N) - (m[:, 2].reshape(X.shape) - hit[2])
        zg = np.where(np.isfinite(zg), zg, hit[2])
        P = np.stack([X, Y, zg], axis=-1).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(R.T)
        uv, _ = cv2.projectPoints((P - C).astype(np.float64), rvec, np.zeros(3), K, dist)
        uv = uv.reshape(X.shape + (2,)).astype(np.float32)
        frame = video.read_frames([int(round(np.interp(tv, times, frames)))])[0].image
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
        patch = cv2.remap(grey, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        c, h = patch.shape[0] // 2, int(round(T / g))
        tpl = patch[c - h : c + h + 1, c - h : c + h + 1]
        if not np.isfinite(tpl).all():
            skipped["off_frame"] += 1
            continue
        found = _match(search, tpl)
        if found is None or found[2] < args.min_ncc:
            skipped["weak"] += 1
            continue
        dx, dy, peak, spread = found
        if spread * g > args.max_ambiguity_m:
            skipped["ambiguous"] += 1
            continue
        feature = hit[:2] + np.array([dx * g, -dy * g])  # where the frame's feature sits in the orthophoto
        truth = wgs84_to_enu(np.array([[float(r["truth_lng"]), float(r["truth_lat"]), o["alt_msl"]]]), origin)[0]
        results.append(
            {
                "point": f"{r['frame_id']}{r['point_id']}",
                "err_e_m": float(feature[0] - truth[0]),
                "err_n_m": float(feature[1] - truth[1]),
                "ray_err_m": float(np.hypot(hit[0] - truth[0], hit[1] - truth[1])),
                "ncc": peak,
            }
        )

    if not results:
        sys.exit(f"no point matched ({skipped})")
    e = np.array([[p["err_e_m"], p["err_n_m"]] for p in results])
    mag = np.hypot(e[:, 0], e[:, 1])
    bias = np.median(e, axis=0)
    rel = np.hypot(*(e - bias).T)
    ray = np.array([p["ray_err_m"] for p in results])
    summary = {
        "ortho": str(args.ortho or out / "orthomosaic.tif"),
        "points_matched": len(results),
        "points_total": len(rows),
        "skipped": skipped,
        "median_m": round(float(np.median(mag)), 3),
        "rmse_m": round(float(np.sqrt(np.mean(mag**2))), 3),
        "p90_m": round(float(np.percentile(mag, 90)), 3),
        "within_1m_pct": round(float(np.mean(mag <= 1.0) * 100), 1),
        "bias_e_m": round(float(bias[0]), 3),
        "bias_n_m": round(float(bias[1]), 3),
        "median_after_bias_m": round(float(np.median(rel)), 3),
        "ray_scorer_median_same_points_m": round(float(np.median(ray)), 3),
        "points": results,
    }
    print(
        f"{len(results)}/{len(rows)} surveyed features found in {Path(summary['ortho']).name} (skipped {skipped}): "
        f"horizontal error median {summary['median_m']:.2f} m, RMSE {summary['rmse_m']:.2f} m, "
        f"p90 {summary['p90_m']:.2f} m, <=1 m {summary['within_1m_pct']:.0f}%; "
        f"common offset E {bias[0]:+.2f} N {bias[1]:+.2f} m, median after removing it {summary['median_after_bias_m']:.2f} m; "
        f"ray scorer on the same points {summary['ray_scorer_median_same_points_m']:.2f} m"
    )
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
