"""Score a DRISHTI-3D run on PinPoint flight01 against its 100 surveyed ground points.

What is measured
----------------
``flight01/points.csv`` pairs a pixel in a video frame with the true
latitude/longitude of the ground feature at that pixel. For each row this
script:

1. interpolates the camera at the row's video time from the run's
   ``cameras.json`` (rotation slerp, position lerp between the two nearest
   keyframes);
2. undistorts the pixel with the nearest camera's ``dist_coeffs`` when the
   run recorded them (points.csv pixels are raw video pixels), casts its
   ray and takes the first place it meets the reconstructed surface -- an
   exact ray/triangle hit when the model has faces, otherwise the median of
   the model points inside a narrow cone around the ray;
3. converts the hit to latitude/longitude and reports the horizontal
   distance to the surveyed truth.

So the number is end to end: camera pose, surface and georeferencing
together, exactly what an operator measuring on the model would get.

Georeferencing is recovered here, not trusted: the model frame is mapped
to local East-North-Up with a similarity fitted from every camera's
position to its own GPS fix (Umeyama). That is the same information the
pipeline had, so this does not flatter it. On a near-straight track a full
3-D rotation is unobservable about the flight line (the fit can roll the
whole model sideways), so there the fit keeps gravity and solves only yaw,
scale and translation (``--fit auto``, the default).

Clip start
----------
points.csv times are ORIGINAL-video times; cameras.json times are clip
times. A wrong ``--clip-start`` casts every ray from the wrong frame and
produces errors of hundreds of metres that look like a reconstruction
failure. The script therefore measures the clip start the run's own
camera GPS implies (the shift that lines the cameras' GPS track up with
points.csv's synced drone positions). Without ``--clip-start`` that value
is used; with it, a disagreement over 0.5 s is reported, because it means
either the flag or the run's telemetry offset is wrong. The check is only
relative: points.csv's drone positions and the run's camera GPS both come
from the flight log, so a clip start that is wrong in the SAME way as the
run's telemetry offset passes it. For ``flight01_120_480.mp4`` the clip's
frame 0 is mkv PTS 120.464 s (found by frame matching), so the right values
are ``--clip-start 120.46`` here and ``--telemetry-offset 121.66`` for the
run.

``--subset nadir`` restricts scoring to PinPoint's ``nadir_ok`` rows. The
flight is a fixed-wing grid that banks 24-46 deg in its turns while the
logged gimbal stays constant, so the non-nadir rows are frames in turns,
where keyframe interpolation of the camera is not meaningful.

The baseline in ``flight01/estadisticas.json`` (``err_att_m``: project the
pixel with the logged attitude onto a terrain model) is printed alongside.

Usage
-----
    .venv/bin/python scripts/score_flight01.py output/flight01/run_*/output --clip-start 120
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
FLIGHT = ROOT / "flight01"
_EARTH_R = 6378137.0
_CLIP_START_TOLERANCE_S = 0.5


def _enu_from_latlon(lat, lon, alt, lat0, lon0, alt0):
    """Local tangent-plane ENU (metres); plenty accurate over a ~1 km site."""
    e = np.radians(np.asarray(lon) - lon0) * _EARTH_R * np.cos(np.radians(lat0))
    n = np.radians(np.asarray(lat) - lat0) * _EARTH_R
    return np.stack([e, n, np.asarray(alt, dtype=float) - alt0], axis=-1)


def _latlon_from_enu(enu, lat0, lon0):
    lat = lat0 + np.degrees(enu[..., 1] / _EARTH_R)
    lon = lon0 + np.degrees(enu[..., 0] / (_EARTH_R * np.cos(np.radians(lat0))))
    return lat, lon


def _umeyama(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d.T @ s / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (s**2).sum(1).mean()
    return scale, R, mu_d - scale * R @ mu_s


def _similarity_yaw(src: np.ndarray, dst: np.ndarray):
    """Least-squares scale, rotation about Z only, and translation: gravity is kept."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    h = d[:, :2].T @ s[:, :2]
    theta = np.arctan2(h[1, 0] - h[0, 1], h[0, 0] + h[1, 1])
    c, n = np.cos(theta), np.sin(theta)
    R = np.array([[c, -n, 0.0], [n, c, 0.0], [0.0, 0.0, 1.0]])
    scale = float(np.sum(d * (s @ R.T)) / np.sum(s**2))
    return scale, R, mu_d - scale * R @ mu_s


def _track_is_collinear(points: np.ndarray, ratio: float = 0.1) -> bool:
    """True when the camera track's horizontal spread is essentially one line."""
    xy = points[:, :2] - points[:, :2].mean(0)
    sv = np.linalg.svd(xy, compute_uv=False)
    return sv.size < 2 or sv[1] < ratio * max(sv[0], 1e-9)


def _read_ply_xyz(path: Path) -> np.ndarray:
    from drishti3d.export.formats import read_ply

    geo = read_ply(path)
    xyz = getattr(geo, "xyz", None)
    if xyz is None:
        xyz = geo[0] if isinstance(geo, tuple) else np.asarray(geo)
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)


def _ray_hit(tree, xyz, origin, direction, cone_rad=0.004, max_range=2000.0, step=1.0):
    """Median of model points in a thin cone around the ray, first surface along it."""
    ts = np.arange(5.0, max_range, step)
    samples = origin + ts[:, None] * direction
    radii = np.maximum(0.5, ts * cone_rad)
    for t, p, r in zip(ts, samples, radii, strict=True):
        idx = tree.query_ball_point(p, r)
        if len(idx) >= 3:
            return np.median(xyz[idx], axis=0), t
    return None, None


def _mesh_scene(path: Path, scale: float, R: np.ndarray, t: np.ndarray):
    """Open3D raycasting scene over the model's triangles in scoring ENU, or None without faces."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        return None
    v = np.asarray(mesh.vertices, dtype=np.float64)
    v = (scale * (R @ v.T)).T + t
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(v.astype(np.float32))
    tmesh.triangle.indices = o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def _mesh_hit(scene, origin: np.ndarray, direction: np.ndarray):
    """First ray/triangle intersection, or (None, None)."""
    import open3d as o3d

    rays = o3d.core.Tensor(np.concatenate([origin, direction])[None].astype(np.float32))
    t_hit = float(scene.cast_rays(rays)["t_hit"].numpy()[0])
    if not np.isfinite(t_hit):
        return None, None
    return origin + t_hit * direction, t_hit


def _undistort_pixel(u: float, v: float, intr: dict) -> tuple[float, float]:
    """Map a raw video pixel to the pinhole pixel that the camera model's rays are cast from."""
    coeffs = intr.get("dist_coeffs")
    if coeffs is None or not np.any(np.asarray(coeffs, dtype=np.float64)):
        return u, v
    import cv2

    K = np.array([[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]])
    out = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, np.asarray(coeffs, dtype=np.float64), P=K)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def _gps_implied_clip_start(cams: list[dict], rows: list[dict], lat0: float, lon0: float):
    """Clip start (s) at which the cameras' GPS track best matches points.csv's drone positions.

    Returns ``(clip_start_s, median_mismatch_m)``, or ``(None, None)`` when too
    few rows overlap the keyframe span for any shift.
    """
    times = np.array([c["timestamp_s"] for c in cams])
    cam_en = _enu_from_latlon(
        [c["gps"]["lat"] for c in cams], [c["gps"]["lon"] for c in cams], np.zeros(len(cams)), lat0, lon0, 0.0
    )
    tv = np.array([float(r["tv_s"]) for r in rows])
    drone_en = _enu_from_latlon(
        [float(r["drone_lat"]) for r in rows], [float(r["drone_lng"]) for r in rows], np.zeros(len(rows)), lat0, lon0, 0.0
    )
    min_rows = max(5, len(rows) // 2)

    def mismatch(shift: float) -> float:
        t = tv - shift
        ok = (t >= times[0]) & (t <= times[-1])
        if ok.sum() < min_rows:
            return np.inf
        e = np.interp(t[ok], times, cam_en[:, 0])
        n = np.interp(t[ok], times, cam_en[:, 1])
        return float(np.median(np.hypot(e - drone_en[ok, 0], n - drone_en[ok, 1])))

    coarse = np.arange(tv.min() - times[-1], tv.max() - times[0] + 0.5, 0.5)
    best = coarse[int(np.argmin([mismatch(s) for s in coarse]))]
    fine = np.arange(best - 0.5, best + 0.5, 0.02)
    vals = [mismatch(s) for s in fine]
    i = int(np.argmin(vals))
    if not np.isfinite(vals[i]):
        return None, None
    return float(fine[i]), float(vals[i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_output", type=Path, help="the run's output/ directory (model.ply + cameras.json)")
    ap.add_argument(
        "--clip-start",
        type=float,
        default=None,
        help="seconds of the original video at the clip's frame 0 (points.csv times are original-video "
        "times; 120 for flight01_120_480.mp4). Default: the value the run's camera GPS implies",
    )
    ap.add_argument(
        "--trust-georef",
        action="store_true",
        help="use the run's own georeferencing (georef.json) instead of re-fitting cameras to GPS -- "
        "required to measure reference alignment, which a GPS re-fit would undo",
    )
    ap.add_argument("--points", type=Path, default=FLIGHT / "points.csv", help="surveyed points CSV")
    ap.add_argument(
        "--subset",
        choices=("all", "nadir"),
        default="all",
        help="nadir: only PinPoint's nadir_ok rows (frames outside the fixed-wing turns)",
    )
    ap.add_argument("--model", default="model.ply", help="surface file inside run_output (PLY)")
    ap.add_argument(
        "--surface",
        choices=("auto", "mesh", "points"),
        default="auto",
        help="auto: exact ray/triangle hits when the model has faces, else the point cone",
    )
    ap.add_argument(
        "--fit",
        choices=("auto", "sim3", "yaw"),
        default="auto",
        help="camera-to-GPS fit: full similarity, or yaw+scale+translation keeping gravity; "
        "auto picks yaw for a near-straight track, where roll about the flight line is unobservable",
    )
    ap.add_argument("--json", type=Path, default=None, help="also write the scores to this JSON file")
    args = ap.parse_args()

    cams = json.loads((args.run_output / "cameras.json").read_text())["cameras"]
    cams = [c for c in cams if c["gps"] is not None and c["intrinsics"] is not None]
    cams.sort(key=lambda c: c["timestamp_s"])
    if len(cams) < 3:
        sys.exit("need >= 3 cameras with GPS in cameras.json")

    t_model = np.array([c["t_world"] for c in cams])
    if args.trust_georef:
        o = json.loads((args.run_output / "georef.json").read_text())["origin"]
        lat0, lon0, alt0 = o["lat"], o["lon"], o["alt_msl"]
        t_gps = _enu_from_latlon(
            [c["gps"]["lat"] for c in cams], [c["gps"]["lon"] for c in cams], [c["gps"]["alt_msl"] for c in cams], lat0, lon0, alt0
        )
        scale, Rg, tg = 1.0, np.eye(3), np.zeros(3)
        fit = "trusted georef.json"
    else:
        lat0, lon0, alt0 = cams[0]["gps"]["lat"], cams[0]["gps"]["lon"], cams[0]["gps"]["alt_msl"]
        t_gps = _enu_from_latlon(
            [c["gps"]["lat"] for c in cams], [c["gps"]["lon"] for c in cams], [c["gps"]["alt_msl"] for c in cams], lat0, lon0, alt0
        )
        fit = args.fit
        if fit == "auto":
            fit = "yaw" if _track_is_collinear(t_gps) else "sim3"
        scale, Rg, tg = _similarity_yaw(t_model, t_gps) if fit == "yaw" else _umeyama(t_model, t_gps)
    cam_fit = np.linalg.norm((scale * (Rg @ t_model.T)).T + tg - t_gps, axis=1)

    model_path = args.run_output / args.model
    scene = None if args.surface == "points" else _mesh_scene(model_path, scale, Rg, tg)
    if args.surface == "mesh" and scene is None:
        sys.exit(f"{model_path} has no faces; use --surface points")
    tree = xyz = None
    if scene is None:
        xyz_model = _read_ply_xyz(model_path)
        xyz = (scale * (Rg @ xyz_model.T)).T + tg
        tree = cKDTree(xyz)
    surface = "mesh ray/triangle" if scene is not None else "point cone"
    # Tier of the surface each ray lands on (nearest vertex): measured by
    # stereo or INFERRED (the visible-scene fill), so the fill's accuracy is
    # reported separately instead of hiding inside one number.
    tier_tree = tier_of = None
    try:
        from drishti3d.export.formats import read_ply

        geo = read_ply(model_path)
        if isinstance(geo, tuple) and len(geo) >= 4 and geo[3] is not None:
            vx = (scale * (Rg @ np.asarray(geo[0], dtype=np.float64).T)).T + tg
            tier_tree, tier_of = cKDTree(vx), np.asarray(geo[3])
    except Exception:  # noqa: BLE001 - tiers are a breakdown, never a reason to fail scoring
        tier_tree = None

    times = np.array([c["timestamp_s"] for c in cams])
    rots = Rotation.from_matrix(np.array([Rg @ np.array(c["R_world_from_cam"]) for c in cams]))
    slerp = Slerp(times, rots)
    cam_pos = (scale * (Rg @ t_model.T)).T + tg

    rows = list(csv.DictReader(args.points.open()))
    if args.subset == "nadir":
        rows = [r for r in rows if r.get("nadir_ok") == "1"]
    implied, implied_mismatch = _gps_implied_clip_start(cams, rows, cams[0]["gps"]["lat"], cams[0]["gps"]["lon"])
    if args.clip_start is None:
        if implied is None:
            sys.exit("cannot infer --clip-start from camera GPS (too few overlapping rows); pass it explicitly")
        clip_start, clip_source = implied, "inferred from camera GPS"
        print(
            f"NOTE: --clip-start not given; using {implied:.2f} s implied by the cameras' GPS "
            f"(median track mismatch {implied_mismatch:.2f} m). This is the true clip start only if the "
            "run's telemetry offset was right."
        )
    else:
        clip_start, clip_source = args.clip_start, "given"
        if implied is not None and abs(implied - clip_start) > _CLIP_START_TOLERANCE_S:
            print(
                f"WARNING: camera GPS implies clip start {implied:.2f} s, not the given {clip_start:.2f} s "
                f"({implied - clip_start:+.2f} s). Either --clip-start is wrong, or the run's telemetry "
                "offset was off by that much and its cameras carry GPS from the wrong instants."
            )

    errors, baseline, tiers, outside, no_hit = [], [], [], 0, 0
    for r in rows:
        tv = float(r["tv_s"]) - clip_start
        if not (times[0] <= tv <= times[-1]):
            outside += 1
            continue
        pos = np.array([np.interp(tv, times, axis) for axis in cam_pos.T])
        R = slerp([tv]).as_matrix()[0]
        intr = cams[int(np.argmin(np.abs(times - tv)))]["intrinsics"]
        # points.csv offsets are from the image centre, in the 1280x720 frame.
        sx = intr["width"] / float(r["img_w"])
        sy = intr["height"] / float(r["img_h"])
        u = (float(r["img_w"]) / 2 + float(r["off_x_px"])) * sx
        v = (float(r["img_h"]) / 2 + float(r["off_y_px"])) * sy
        u, v = _undistort_pixel(u, v, intr)
        ray_cam = np.array([(u - intr["cx"]) / intr["fx"], (v - intr["cy"]) / intr["fy"], 1.0])
        d = R @ ray_cam
        d /= np.linalg.norm(d)
        hit, _ = _mesh_hit(scene, pos, d) if scene is not None else _ray_hit(tree, xyz, pos, d)
        if hit is None:
            no_hit += 1
            continue
        truth = _enu_from_latlon(float(r["truth_lat"]), float(r["truth_lng"]), 0.0, lat0, lon0, 0.0)
        errors.append(float(np.hypot(hit[0] - truth[0], hit[1] - truth[1])))
        baseline.append(float(r["err_att_m"]))
        tiers.append(int(tier_of[tier_tree.query(hit)[1]]) if tier_tree is not None else -1)

    e = np.array(errors)
    b = np.array(baseline)
    print(
        f"cameras: {len(cams)}; camera-to-GPS fit ({fit}) residual median {np.median(cam_fit):.2f} m (scale {scale:.4f})"
    )
    print(f"clip start: {clip_start:.2f} s ({clip_source}); surface: {surface} ({model_path.name})")
    print(f"points scored: {len(e)}/{len(rows)} (outside keyframe time span {outside}, no surface hit {no_hit})")
    summary = {
        "run_output": str(args.run_output),
        "clip_start_s": clip_start,
        "clip_start_source": clip_source,
        "gps_implied_clip_start_s": implied,
        "surface": surface,
        "cameras": len(cams),
        "fit": fit,
        "camera_gps_fit_median_m": float(np.median(cam_fit)),
        "subset": args.subset,
        "points_total": len(rows),
        "points_scored": len(e),
        "outside_span": outside,
        "no_hit": no_hit,
    }
    if len(e):
        summary.update(
            median_m=float(np.median(e)),
            rmse_m=float(np.sqrt(np.mean(e**2))),
            p90_m=float(np.percentile(e, 90)),
            within_1m_pct=float(np.mean(e <= 1.0) * 100),
            within_3m_pct=float(np.mean(e <= 3.0) * 100),
            baseline_median_m=float(np.median(b)),
            baseline_rmse_m=float(np.sqrt(np.mean(b**2))),
        )
        print(
            f"DRISHTI-3D horizontal error: median {np.median(e):.2f} m, RMSE {np.sqrt(np.mean(e**2)):.2f} m, "
            f"p90 {np.percentile(e, 90):.2f} m, <=1 m {np.mean(e <= 1.0) * 100:.0f}%, "
            f"<=3 m {np.mean(e <= 3.0) * 100:.0f}%"
        )
        print(
            f"baseline (logged attitude onto terrain, same points): median {np.median(b):.2f} m, "
            f"RMSE {np.sqrt(np.mean(b**2)):.2f} m"
        )
        tv = np.array(tiers)
        for code, name in ((2, "measured"), (1, "low confidence"), (0, "inferred (visible-scene fill)")):
            m = tv == code
            if m.any():
                print(f"  on {name} surface: {int(m.sum())} points, median {np.median(e[m]):.2f} m")
                summary[f"median_m_{name.split()[0]}"] = float(np.median(e[m]))
                summary[f"points_{name.split()[0]}"] = int(m.sum())
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
