"""Settle one question: is the missing building relief the backbone's fault?

The defect
----------
The 956 px MapAnything run reconstructs buildings only 0.17 m above the
surrounding terrain. Real buildings in that scene are 3-4 m. Two
explanations are consistent with that observation and they lead to
completely different remedies:

1. **The backbone smooths.** MapAnything regresses a dense depth *field*,
   and a regressed field is biased toward smoothness -- a 3 m roof over a
   10 m footprint is a small, high-frequency deviation that a network
   trained with an L1/L2 depth loss is rewarded for flattening. If this is
   the cause, swapping in a per-pixel photometric matcher recovers the
   roofs and the fix is a new backend.

2. **The footage has no signal.** At 119.6 m AGL with a measured 6.75 m
   baseline (B/H 0.056), the triangulation geometry may simply not resolve
   3 m of relief above the noise floor. If this is the cause, *no* backend
   change helps, and the honest answer is that single-pass nadir video at
   this altitude cannot deliver building heights.

Why COLMAP's sparse model is the right instrument
--------------------------------------------------
This script deliberately does NOT run dense MVS. It runs COLMAP's sparse
SfM -- SIFT matching, incremental mapping, bundle adjustment -- and reads
the triangulated feature points.

That is the point. Sparse SfM triangulates each point from *matched image
observations* with no learned smoothness prior anywhere in the loop. It is
the cleanest available measurement of how much vertical structure the
image geometry alone supports. If COLMAP's sparse points sit 3-4 m above
local ground on the rooftops, the parallax is demonstrably there and
explanation (1) holds. If COLMAP's roofs are also flat, explanation (2)
holds and the altitude, not the backbone, is the ceiling.

A dense run would answer the same question with prettier output and a
CUDA requirement macOS cannot satisfy (``patch_match_stereo`` is
CUDA-only). Sparse runs on CPU, uses the same 16 keyframes, and is
decisive for the thing actually in dispute.

Comparability
-------------
Two choices keep this an apples-to-apples test rather than a
demonstration:

- **The same 16 keyframes**, by frame index, read from the 956 px run's
  own ``meta.json``. Not a fresh triage pass -- a different frame set
  would confound backbone differences with sampling differences.
- **The same relief metric**, ``local_relief()``, computed identically on
  COLMAP's cloud and on ours. Height above the local ground surface, not
  raw Z spread: raw spread is dominated by terrain slope across the site
  and by outliers, and says nothing about whether *buildings* stand up.

Scale
-----
COLMAP's reconstruction is up to an unknown similarity. Metres are
recovered by Umeyama-aligning COLMAP's camera centres onto the GPS-derived
ENU positions of the same keyframes -- the same ``umeyama_alignment`` the
pipeline already uses for submap merging. The fitted scale is reported;
a wildly wrong scale invalidates the height numbers and should be read
before the relief figures are.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drishti3d.geometry.georef import wgs84_to_enu  # noqa: E402
from drishti3d.geometry.submap import umeyama_alignment  # noqa: E402
from drishti3d.types import GeoPoint  # noqa: E402

#: Half-width (metres) of the neighbourhood used to define "local ground"
#: when measuring relief. Must be wider than a building footprint -- a
#: window narrower than the roof finds its ground reference ON the roof and
#: reports zero relief for a building that is plainly there. 12 m clears a
#: typical suburban house; the scene in question is residential.
_GROUND_WINDOW_M = 12.0

#: Percentile of Z taken as the local ground level inside each window.
#: Not the minimum: a single low outlier (a triangulation blunder under
#: the surface) would drag the reference down and inflate every relief
#: number in the cell. The 5th percentile is robust to that while still
#: tracking the ground rather than the mean surface.
_GROUND_PERCENTILE = 5.0

#: Relief (metres) above local ground at which a point is counted as
#: "raised structure". Set below the 3-4 m real building height so that a
#: partially-recovered roof still registers -- the question is whether
#: relief exists at all, not whether it is exact.
_STRUCTURE_THRESHOLD_M = 2.0


# ---------------------------------------------------------------------------
# COLMAP sparse model readers (binary format)
# ---------------------------------------------------------------------------


def _read_next_bytes(fid, num_bytes: int, format_char_sequence: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_points3d_bin(path: Path) -> np.ndarray:
    """Return ``(N, 3)`` XYZ of every triangulated point in the model."""
    xyz = []
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary = _read_next_bytes(fid, 43, "QdddBBBd")
            xyz.append(binary[1:4])
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_length)
    return np.asarray(xyz, dtype=np.float64)


def read_images_bin(path: Path) -> dict[str, np.ndarray]:
    """Return ``{image_name: camera_centre_xyz}`` for every registered image.

    COLMAP stores world-from-camera as ``(qvec, tvec)`` mapping world into
    camera, so the centre is ``-R^T t`` -- not ``t``. Getting this backwards
    silently produces a mirrored trajectory that Umeyama will happily fit
    with a plausible-looking scale, which is why it is spelled out.
    """
    centres: dict[str, np.ndarray] = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            name = ""
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name += char.decode("utf-8")
            num_p2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_p2d)
            centres[name] = -_qvec_to_rotmat(qvec).T @ tvec
    return centres


def _qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


def local_relief(xyz: np.ndarray, window_m: float = _GROUND_WINDOW_M) -> dict:
    """Height of each point above its own local ground level.

    Bins points into ``window_m`` cells in XY, takes a low percentile of Z
    per cell as that cell's ground, and returns the distribution of
    ``z - ground``. This is the reference-free stand-in for "are the
    buildings there": terrain contributes ~0 regardless of how steeply the
    site slopes, while any raised structure contributes its height.

    Returns the percentiles plus the share of points above
    ``_STRUCTURE_THRESHOLD_M``, which is the single number worth comparing
    between two reconstructions of the same scene.
    """
    if len(xyz) < 100:
        return {"error": f"only {len(xyz)} points"}

    xy = xyz[:, :2]
    z = xyz[:, 2]
    origin = xy.min(axis=0)
    cell = np.floor((xy - origin) / window_m).astype(np.int64)
    ncols = int(cell[:, 0].max()) + 1
    flat = cell[:, 1] * ncols + cell[:, 0]

    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    z_sorted = z[order]
    boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(flat_sorted)]])

    ground = np.empty(len(z), dtype=np.float64)
    populated = 0
    for s, e in zip(starts, ends):
        # A cell with too few points cannot define a percentile that means
        # anything; its own minimum is used and it is not counted as a
        # populated cell in the stats.
        block = z_sorted[s:e]
        ground[order[s:e]] = np.percentile(block, _GROUND_PERCENTILE) if (e - s) >= 20 else block.min()
        if (e - s) >= 20:
            populated += 1

    relief = z - ground
    return {
        "points": int(len(xyz)),
        "cells": int(len(starts)),
        "cells_with_stats": populated,
        "relief_p50_m": round(float(np.percentile(relief, 50)), 3),
        "relief_p90_m": round(float(np.percentile(relief, 90)), 3),
        "relief_p99_m": round(float(np.percentile(relief, 99)), 3),
        "relief_max_m": round(float(relief.max()), 3),
        "pct_above_2m": round(100.0 * float((relief > _STRUCTURE_THRESHOLD_M).mean()), 3),
    }


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames(video: Path, indices: list[int], out_dir: Path) -> list[str]:
    """Decode exactly ``indices`` from ``video`` at full resolution."""
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(indices)
    names: list[str] = []
    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    for i, frame in enumerate(container.decode(stream)):
        if i in wanted:
            name = f"frame_{i:06d}.jpg"
            frame.to_image().save(out_dir / name, quality=95)
            names.append(name)
            wanted.discard(i)
            if not wanted:
                break
    container.close()
    return names


# ---------------------------------------------------------------------------
# COLMAP driver
# ---------------------------------------------------------------------------


def run_colmap(workspace: Path, fx: float, cx: float, cy: float, width: int, height: int) -> Path:
    """Sparse SfM only. Returns the sparse model directory.

    Camera model is fixed to PINHOLE with the intrinsics the pipeline
    already derived from the camera database, and ``--Mapper.ba_refine_*``
    is left ON: letting COLMAP refine focal length is what makes its
    triangulation independent of our intrinsics being right, which matters
    because a wrong focal length is itself a candidate explanation for
    flattened relief.
    """
    images = workspace / "images"
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse.mkdir(exist_ok=True)

    common = ["--database_path", str(database)]

    subprocess.run(
        [
            "colmap", "feature_extractor",
            *common,
            "--image_path", str(images),
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", f"{fx},{fx},{cx},{cy}",
            # COLMAP 4.x moved the backend-agnostic knobs out of
            # SiftExtraction into FeatureExtraction (SIFT is now one of
            # several detectors). The old --SiftExtraction.max_image_size /
            # .use_gpu names hard-error rather than being ignored, so these
            # are version-sensitive and deliberately not "fixed up" silently.
            "--FeatureExtraction.use_gpu", "0",
            "--FeatureExtraction.max_image_size", "3200",
            # 8192, not COLMAP's 8192-default-raised-to-16384: matching is
            # brute-force on this CPU-only build (O(n^2) per pair, 120 pairs
            # exhaustive), so doubling features quadruples a cost that is
            # already the bottleneck. 8192 over a 3200 px frame is dense
            # enough that rooftops carry plenty of keypoints.
            "--SiftExtraction.max_num_features", "8192",
        ],
        check=True,
    )

    # Exhaustive, not sequential: 16 images is small enough that every pair
    # is affordable, and exhaustive matching removes "the matcher never
    # tried that pair" as an explanation for anything missing downstream.
    subprocess.run(
        ["colmap", "exhaustive_matcher", *common, "--FeatureMatching.use_gpu", "0"],
        check=True,
    )

    subprocess.run(
        [
            "colmap", "mapper",
            *common,
            "--image_path", str(images),
            "--output_path", str(sparse),
        ],
        check=True,
    )

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise RuntimeError("COLMAP mapper produced no model: reconstruction failed outright")
    # Largest model wins; a fragmented run leaves several partial ones.
    return max(models, key=lambda m: (m / "points3D.bin").stat().st_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=Path("/Users/ajith/Desktop/sih_data_samples/DJI_0753.MP4"))
    ap.add_argument("--reference-run", type=Path, default=REPO / "results/local_956")
    ap.add_argument("--workspace", type=Path, default=Path("/tmp/colmap_probe"))
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-colmap", action="store_true")
    args = ap.parse_args()

    if shutil.which("colmap") is None and not args.skip_colmap:
        print("colmap not on PATH -- `brew install colmap` first", file=sys.stderr)
        return 2

    meta = json.loads((args.reference_run / "meta.json").read_text())
    keyframes = meta["keyframes"]
    indices = [k["frame_index"] for k in keyframes]
    intr = keyframes[0]["intrinsics"]
    print(f"reference run: {len(indices)} keyframes, indices {indices[0]}..{indices[-1]}")

    args.workspace.mkdir(parents=True, exist_ok=True)
    if not args.skip_extract:
        names = extract_frames(args.video, indices, args.workspace / "images")
        print(f"extracted {len(names)} frames at {intr['width']}x{intr['height']}")

    if not args.skip_colmap:
        model = run_colmap(
            args.workspace,
            fx=intr["fx"], cx=intr["cx"], cy=intr["cy"],
            width=intr["width"], height=intr["height"],
        )
    else:
        model = max(
            (p for p in (args.workspace / "sparse").iterdir() if p.is_dir()),
            key=lambda m: (m / "points3D.bin").stat().st_size,
        )
    print(f"model: {model}")

    xyz_colmap = read_points3d_bin(model / "points3D.bin")
    centres = read_images_bin(model / "images.bin")
    print(f"colmap: {len(xyz_colmap)} sparse points, {len(centres)}/{len(indices)} images registered")

    # --- metric scale, from GPS ------------------------------------------
    geo0 = keyframes[0]["telemetry"]["geo"]
    origin = GeoPoint(lat=geo0["lat"], lon=geo0["lon"], alt_msl=geo0["alt_msl"])
    gps_enu, colmap_c = [], []
    for k in keyframes:
        name = f"frame_{k['frame_index']:06d}.jpg"
        if name not in centres:
            continue
        g = k["telemetry"]["geo"]
        gps_enu.append(wgs84_to_enu(np.array([[g["lon"], g["lat"], g["alt_msl"]]]), origin)[0])
        colmap_c.append(centres[name])

    if len(colmap_c) < 3:
        print(f"only {len(colmap_c)} images registered -- cannot recover scale; COLMAP failed on this footage")
        return 1

    sim3, degenerate, cond = umeyama_alignment(np.asarray(colmap_c), np.asarray(gps_enu))
    residual = np.linalg.norm(sim3.apply(np.asarray(colmap_c)) - np.asarray(gps_enu), axis=1)
    print(
        f"gps alignment: scale={sim3.scale:.4f} residual_median={np.median(residual):.3f} m "
        f"degenerate={degenerate} cond={cond:.1f}"
    )

    xyz_metric = sim3.apply(xyz_colmap)

    # --- the comparison ---------------------------------------------------
    print("\n--- COLMAP sparse (no learned prior) ---")
    colmap_stats = local_relief(xyz_metric)
    for k, v in colmap_stats.items():
        print(f"  {k}: {v}")

    ours_path = args.reference_run / "output" / "point_cloud.ply"
    if ours_path.exists():
        xyz_ours = _read_ply_xyz(ours_path)
        print(f"\n--- MapAnything 956 px ({ours_path.name}) ---")
        for k, v in local_relief(xyz_ours).items():
            print(f"  {k}: {v}")
    else:
        print(f"\n(no reference cloud at {ours_path}; skipping side-by-side)")

    out = args.workspace / "relief_comparison.json"
    out.write_text(json.dumps({"colmap": colmap_stats, "scale": sim3.scale}, indent=2))
    print(f"\nwrote {out}")
    return 0


def _read_ply_xyz(path: Path) -> np.ndarray:
    """Minimal PLY reader for the x/y/z properties of a binary or ASCII cloud."""
    with open(path, "rb") as f:
        fmt = None
        count = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                count = int(line.split()[2])
                in_vertex = True
            elif line.startswith("element "):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        sizes = {"float": 4, "float32": 4, "double": 8, "float64": 8, "uchar": 1, "uint8": 1,
                 "char": 1, "int8": 1, "short": 2, "ushort": 2, "int": 4, "uint": 4}
        np_of = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
                 "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                 "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4"}

        if fmt == "ascii":
            names = [n for _, n in props]
            arr = np.loadtxt(f, max_rows=count)
            idx = [names.index(c) for c in ("x", "y", "z")]
            return arr[:, idx].astype(np.float64)

        dtype = np.dtype([(n, np_of[t]) for t, n in props])
        assert dtype.itemsize == sum(sizes[t] for t, _ in props)
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)
        return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
