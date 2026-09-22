"""One-command assessment of a finished run: is the model one piece, at one scale, with relief?

Prints the handful of numbers that actually decide whether a run is
shippable, all reference-free (nothing here needs ground truth):

- mesh connectivity (components, largest fraction) -- a fragmented model
  is unusable regardless of its RMSE
- ground elevation per major component -- the 50 m split that made the
  ground land in two slabs shows up here as a large span
- camera-to-ground distance vs telemetry altitude per keyframe -- the
  backbone's depth-scale error, before/after anchoring
- local relief above ground -- whether structures stand up
- the report card's own headline metrics and the depth-anchor summary

Usage: python scripts/assess_run.py results/<run> [results/<baseline_run>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from colmap_probe import _read_ply_xyz, local_relief  # noqa: E402


def mesh_connectivity(ply: Path) -> dict:
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(str(ply))
    ids, cnt, _ = m.cluster_connected_triangles()
    ids, cnt = np.asarray(ids), np.asarray(cnt)
    V, T = np.asarray(m.vertices), np.asarray(m.triangles)
    order = np.argsort(cnt)[::-1]
    grounds = []
    for c in order[:8]:
        P = V[np.unique(T[ids == c])]
        grounds.append(float(np.percentile(P[:, 2], 10)))
    return {
        "vertices": int(len(V)),
        "faces": int(len(T)),
        "components": int(len(cnt)),
        "largest_pct": round(100.0 * cnt[order[0]] / max(len(ids), 1), 1),
        "z_spread_m": round(float(np.ptp(V[:, 2])), 1),
        "top8_ground_z": [round(g, 1) for g in grounds],
        "top8_ground_span_m": round(max(grounds) - min(grounds), 1) if grounds else None,
    }


def altitude_check(run: Path) -> dict:
    """Reconstructed camera-to-ground vs telemetry alt_rel, per keyframe."""
    from scipy.spatial import cKDTree

    d = np.load(run / "arrays.npz", allow_pickle=True)
    P, X = d["poses_t"], d["pc_xyz"]
    meta = json.loads((run / "meta.json").read_text())
    alt = np.array([k["telemetry"]["geo"]["alt_rel"] for k in meta["keyframes"]])
    tree = cKDTree(X[:, :2])
    ratios = []
    for i in range(len(P)):
        idx = tree.query_ball_point(P[i, :2], 25.0)
        if len(idx) < 50:
            continue
        ground = np.percentile(X[idx, 2], 10)
        ratios.append((P[i, 2] - ground) / alt[i])
    r = np.asarray(ratios)
    return {
        "keyframes_checked": int(len(r)),
        "cam_to_ground_over_alt_rel": {
            "median": round(float(np.median(r)), 3) if len(r) else None,
            "p10": round(float(np.percentile(r, 10)), 3) if len(r) else None,
            "p90": round(float(np.percentile(r, 90)), 3) if len(r) else None,
        },
        "note": "1.0 = backbone depth matches height above takeoff; alt_rel != AGL on sloping terrain",
    }


def report_headline(run: Path) -> dict:
    out = {}
    txt = run / "output" / "report.txt"
    if txt.exists():
        for line in txt.read_text().splitlines():
            for key in ("Relative RMSE", "Mean reprojection error", "Coverage", "Keyframes used", "windows anchored", "depth ratio"):
                if line.strip().startswith(key):
                    out[key] = line.split(":", 1)[1].strip()
    meta = json.loads((run / "meta.json").read_text())
    for s in meta.get("stage_results", []):
        a = s.get("artifacts", {})
        if s["name"] == "pose_prior":
            out["pose_prior"] = {k: a[k] for k in a if k in ("tracks_triangulated", "rmse_after_px", "ba_pass2_points", "median_position_shift_m", "median_rotation_change_deg", "converged")}
        if s["name"] == "bundle_adjustment":
            out["ba"] = {k: a[k] for k in a if k.startswith(("ba_pass", "tracks_", "rmse_", "converged"))}
        if s["name"] == "geometry":
            da = a.get("depth_anchor") or {}
            out["depth_anchor"] = {k: da.get(k) for k in ("windows_anchored", "windows_total", "ratio_median", "ratio_min", "ratio_max")}
            out["depth_anchor_refused"] = da.get("windows_refused")
            pw = da.get("per_window") or []
            mf = [w.get("masked_fraction") for w in pw if w.get("masked_fraction") is not None]
            out["masked_fraction_mean"] = round(float(np.mean(mf)), 3) if mf else None
        out.setdefault("timings_s", {})[s["name"]] = round(s.get("elapsed_s", 0))
    return out


def assess(run: Path) -> dict:
    out = {"run": str(run)}
    mesh = run / "output" / "model.ply"
    if mesh.exists():
        out["mesh"] = mesh_connectivity(mesh)
        out["relief"] = local_relief(_read_ply_xyz(mesh))
    if (run / "arrays.npz").exists():
        out["altitude"] = altitude_check(run)
    out["report"] = report_headline(run)
    return out


def main() -> int:
    runs = [Path(p) for p in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 2
    for run in runs:
        print("=" * 78)
        print(json.dumps(assess(run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
