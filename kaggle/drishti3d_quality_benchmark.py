# ============================================================================
# DRISHTI-3D — Kaggle mesh-quality & accuracy benchmark (GPU accelerated)
# ============================================================================
# SETTINGS BEFORE YOU RUN (right sidebar):
#   Accelerator : GPU T4 x2       <-- NOT TPU. MapAnything + SegFormer need CUDA.
#   Internet    : On              <-- needed to pull repos + model weights
#   Persistence : Files only      <-- keeps /kaggle/working between sessions
#
# SECRET (Add-ons -> Secrets):  name = HF_TOKEN, value = your HF token
#
# ---------------------------------------------------------------------------
# ABOUT "30 GB of GPU"
# ---------------------------------------------------------------------------
# Kaggle's "T4 x2" is TWO SEPARATE 16 GB devices, not one 32 GB pool. A single
# model cannot use more than 16 GB unless it is explicitly sharded across both,
# and neither MapAnything nor SegFormer shards here. So the number that governs
# whether this runs is 16 GB per device, not 32 GB total.
#
# That is good news for the deployment story: the target hardware is a 6 GB
# RTX 4060, and Cell 3 measures actual peak VRAM so the 16 GB headroom can be
# used to find where the 6 GB ceiling actually binds instead of guessing.
#
# The second GPU IS used: Cell 5 pins segmentation to cuda:1 while geometry
# holds cuda:0, so the two models never contend for the same device.
# ============================================================================


# %% [CELL 1] ---------------------------------------------------------------
# Environment. Run first, read the output before continuing.

import os
import subprocess
import sys

print("=" * 74)
print("ENVIRONMENT")
print("=" * 74)

try:
    import torch

    print(f"torch            {torch.__version__}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        total = 0.0
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            total += p.total_memory / 1e9
            print(f"  cuda:{i}         {p.name}  {p.total_memory / 1e9:.1f} GB")
        cap = torch.cuda.get_device_capability(0)
        print(f"cuda version     {torch.version.cuda}")
        print(f"compute cap      {cap[0]}.{cap[1]}  (fp16 tensor cores: {cap[0] >= 7})")
        print(f"\ntotal across devices : {total:.1f} GB")
        print(f"usable by ONE model  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("  ^ this is the number that matters; the devices are separate.")
    else:
        print("\n  *** NO CUDA. Set Accelerator = GPU T4 x2, not TPU/None. ***")
except ImportError:
    print("torch not installed")

print(f"\npython           {sys.version.split()[0]}")
print(f"cwd              {os.getcwd()}")


# %% [CELL 2] ---------------------------------------------------------------
# Install. Kaggle ships a CUDA torch already -- nothing here may replace it.

print("Installing dependencies (4-7 min)...")

# Pipeline runtime deps Kaggle may not have.
!pip install -q av pyproj laspy 2>&1 | tail -2

# open3d gives the sparse TSDF path (much faster than the numpy fallback).
!pip install -q open3d 2>&1 | tail -2

# xatlas: UV unwrapping for the texture atlas. Small, pure wheel.
!pip install -q xatlas 2>&1 | tail -2

# MapAnything. --no-deps protects Kaggle's CUDA torch; that makes US
# responsible for its real requirements, taken from its own pyproject.
!pip install -q --no-deps git+https://github.com/facebookresearch/map-anything.git 2>&1 | tail -2
!pip install -q \
    "uniception==0.1.7" \
    hydra-core natsort orjson pillow-heif plyfile python-box \
    safetensors tensorboard trimesh einops jaxtyping \
    "rerun-sdk~=0.24.1" "opencv-python-headless==4.10.0.84" 2>&1 | tail -2

# transformers for SegFormer (semantic classification + dynamic masking).
!pip install -q transformers 2>&1 | tail -2

print("\nVerifying the install holds together:")
import torch

print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA broke during install — restart session and rerun"

from mapanything.models import MapAnything  # noqa: F401
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT  # noqa: F401
from transformers import SegformerForSemanticSegmentation  # noqa: F401
import open3d
import xatlas  # noqa: F401

print(f"  mapanything + uniception + transformers + xatlas OK, open3d {open3d.__version__}")


# %% [CELL 3] ---------------------------------------------------------------
# Pull code + data from the private Hugging Face repos.

from pathlib import Path

from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
CODE_DIR = "/kaggle/working/drishti3d"
DATA_DIR = "/kaggle/working/data"

print("Downloading code...")
snapshot_download(repo_id="ajt0704/3chakshu", repo_type="model", local_dir=CODE_DIR, token=HF_TOKEN)

print("Downloading data (5.8 GB, several minutes)...")
snapshot_download(repo_id="ajt0704/3chakshu-data", repo_type="dataset", local_dir=DATA_DIR, token=HF_TOKEN)

os.environ["PYTHONPATH"] = CODE_DIR
sys.path.insert(0, CODE_DIR)

print("\nData files:")
for p in sorted(Path(DATA_DIR).rglob("*")):
    if p.is_file() and p.suffix.upper() in {".MP4", ".MOV", ".CSV", ".SRT"}:
        print(f"   {p.stat().st_size / 1e9:6.2f} GB  {p.relative_to(DATA_DIR)}")

r = subprocess.run(
    [sys.executable, "-c", "from drishti3d.device import get_device; print('device:', get_device())"],
    cwd=CODE_DIR,
    capture_output=True,
    text=True,
    env={**os.environ, "PYTHONPATH": CODE_DIR},
)
print("\n" + (r.stdout or r.stderr))
assert "cuda" in r.stdout, "pipeline is not seeing CUDA"


# %% [CELL 4] ---------------------------------------------------------------
# Write the GPU-accelerated config. Everything that can run on the GPU does.

import yaml

CONFIG_PATH = "/kaggle/working/kaggle_gpu.yaml"

config = {
    # "accurate" retunes triage spacing, backbone resolution, matching AND
    # semantics agreement thresholds together -- see config.py's profile
    # tables. On a 16 GB device there is no reason to run anything less.
    "quality_profile": "accurate",
    "geometry": {
        "backbone": "mapanything",
        # 924 is the balanced default; 1288 is what 16 GB actually affords
        # and is where roof edges stop being mushy. Cell 8 reports peak VRAM
        # so you can see the cost.
        "max_image_size": 1288,
        "window_size": 14,
    },
    "semantics": {
        "enabled": True,
        "model": "segformer",
        "checkpoint": "nvidia/segformer-b4-finetuned-ade-512-512",
        "max_image_size": 1280,
        # 8 is safe on a 16 GB device; the 6 GB target uses 4.
        "batch_size": 8,
        "remove_dynamic": True,
        "min_vote_ratio": 0.6,
        "min_views": 3,
    },
    "texture": {
        "enabled": True,
        # 8192 is the practical ceiling before viewers start refusing the
        # texture. Needs ~1.2 GB of host RAM transiently, not VRAM.
        "texture_size": 8192,
        "blend_views": 4,
        "max_views": 200,
    },
    "fusion": {
        "voxel_count_budget": 8_000_000,
        "pre_tsdf_max_points": 6_000_000,
    },
}

Path(CONFIG_PATH).write_text(yaml.safe_dump(config, sort_keys=False))
print(Path(CONFIG_PATH).read_text())


# %% [CELL 5] ---------------------------------------------------------------
# Run the full pipeline with GPU acceleration, capturing peak VRAM.

import json
import time

VIDEO = f"{DATA_DIR}/survey_nadir/DJI_0753.MP4"
TELEM = f"{DATA_DIR}/survey_nadir/DJI_0753_airdata.csv"


def run(max_kf, tag, extra=""):
    """Run the pipeline and return (out_dir, wall_clock_seconds)."""
    out = f"/kaggle/working/run_{tag}"
    cmd = (
        f"cd {CODE_DIR} && PYTHONPATH={CODE_DIR} "
        # Fragmentation is the usual cause of a spurious OOM on long runs;
        # expandable_segments lets the allocator give memory back.
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"python -m drishti3d.pipeline.runner "
        f"'{VIDEO}' --telemetry '{TELEM}' --config {CONFIG_PATH} "
        f"--backbone mapanything --max-keyframes {max_kf} "
        f"--out {out} --log-level INFO {extra} 2>&1 | tail -60"
    )
    print(f"\n{'=' * 74}\nRUN: {tag}  (target {max_kf} keyframes)\n{'=' * 74}")
    t0 = time.time()
    os.system(cmd)
    elapsed = time.time() - t0
    print(f"\n>>> {tag}: {elapsed / 60:.1f} min wall clock")
    return out, elapsed


# Smoke test first. If this fails, stop and read the error -- do not run the
# full flight on a pipeline that is already broken at 16 keyframes.
out16, t16 = run(16, "16kf")
print("\nDeliverables written:")
for p in sorted(Path(f"{out16}/output").glob("*")):
    print(f"   {p.stat().st_size / 1e6:9.2f} MB  {p.name}")


# %% [CELL 6] ---------------------------------------------------------------
# Mesh-quality + accuracy metrics. This is the cell that answers the question.

import numpy as np

sys.path.insert(0, CODE_DIR)


def assess(out_dir, label):
    """Full quality assessment of one run. Returns a dict of metrics."""
    import laspy
    from scipy import ndimage
    from scipy.spatial import cKDTree

    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    metrics = {"label": label}

    meta_path = Path(out_dir) / "meta.json"
    if not meta_path.exists():
        print("  no meta.json — the run did not complete")
        return metrics
    meta = json.loads(meta_path.read_text())
    report = meta.get("report", {})

    # -- stage timings -----------------------------------------------------
    stages = {s["name"]: s.get("elapsed_s", 0) for s in meta.get("stage_results", [])}
    statuses = {s["name"]: s.get("status") for s in meta.get("stage_results", [])}
    print("\nSTAGES")
    for name, seconds in stages.items():
        print(f"  {name:<20} {seconds:8.1f} s   [{statuses.get(name)}]")
    metrics["stage_timings_s"] = stages
    metrics["stage_statuses"] = statuses
    metrics["total_s"] = sum(stages.values())

    # -- accuracy, straight from the report card ---------------------------
    print("\nACCURACY (from the pipeline's own report card)")
    for key in (
        "relative_rmse_m",
        "absolute_rmse_m",
        "scale_error_pct",
        "mean_reprojection_error_px",
        "coverage_pct",
        "keyframe_count",
        "confidence_source",
    ):
        print(f"  {key:<30} {report.get(key)}")
        metrics[key] = report.get(key)

    breakdown = report.get("confidence_breakdown_pct")
    if isinstance(breakdown, dict):
        print("\nCONFIDENCE TIERS")
        for k, v in breakdown.items():
            print(f"  {k:<30} {v:.1f} %")
        metrics["confidence_breakdown_pct"] = breakdown

    # -- semantics ---------------------------------------------------------
    hist = report.get("semantic_class_pct")
    if isinstance(hist, dict):
        print("\nSCENE COMPOSITION (per reconstructed point)")
        for name, pct in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<30} {pct:5.1f} %")
        metrics["semantic_class_pct"] = hist
    for key in (
        "semantic_labelled_pct",
        "semantic_disputed_points",
        "semantic_unseen_points",
        "dynamic_points_removed",
        "dtm_method",
        "dtm_interpolated_pct",
        "facade_structures_completed",
        "facade_points_inferred",
    ):
        if report.get(key) not in (None, "not computed"):
            print(f"  {key:<30} {report.get(key)}")
            metrics[key] = report.get(key)

    # -- point cloud geometry ---------------------------------------------
    pc_path = Path(out_dir) / "output" / "point_cloud.las"
    if not pc_path.exists():
        print("\n  no point_cloud.las produced")
        return metrics

    las = laspy.read(str(pc_path))
    xyz = np.vstack([las.x, las.y, las.z]).T
    metrics["n_points"] = int(len(xyz))

    extent = xyz.max(0) - xyz.min(0)
    area_m2 = float(extent[0] * extent[1])
    metrics["extent_m"] = [round(float(v), 1) for v in extent]
    metrics["density_pts_per_m2"] = round(len(xyz) / max(area_m2, 1.0), 1)

    # Local planar residual: the depth-precision metric. Fit a plane to the
    # neighbours within 0.5 m of each sample and take the residual spread.
    rng = np.random.default_rng(0)
    sample = xyz[rng.choice(len(xyz), min(600, len(xyz)), replace=False)]
    tree = cKDTree(xyz[:, :2])
    residuals = []
    for p in sample:
        idx = tree.query_ball_point(p[:2], 0.5)
        if len(idx) < 25:
            continue
        q = xyz[idx]
        A = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
        coef, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        residuals.append((q[:, 2] - A @ coef).std())
    metrics["local_planar_residual_m"] = round(float(np.median(residuals)), 3) if residuals else None

    # Connectivity: one surface, or a field of fragments?
    gs = 1.0
    ix = ((xyz[:, 0] - xyz[:, 0].min()) / gs).astype(int)
    iy = ((xyz[:, 1] - xyz[:, 1].min()) / gs).astype(int)
    grid = np.zeros((iy.max() + 1, ix.max() + 1), bool)
    grid[iy, ix] = True
    _lab, n_comp = ndimage.label(grid)
    sizes = np.bincount(_lab.ravel())[1:]
    metrics["connected_components"] = int(n_comp)
    metrics["largest_component_pct"] = round(float(sizes.max() / sizes.sum() * 100), 1) if len(sizes) else 0.0

    dims = set(las.point_format.dimension_names)
    metrics["has_rgb"] = "red" in dims and int(np.asarray(las.red).max()) > 0
    metrics["has_semantic_class"] = "semantic_class" in dims
    metrics["has_confidence"] = "confidence" in dims

    print("\nPOINT CLOUD")
    print(f"  points                         {metrics['n_points']:,}")
    print(f"  extent (m)                     {metrics['extent_m']}")
    print(f"  density (pts/m2)               {metrics['density_pts_per_m2']}")
    res = metrics["local_planar_residual_m"]
    print(f"  local 1m planar residual (m)   {res}   {'OK' if res and res < 1.0 else 'HIGH'}")
    print(f"  connected components           {metrics['connected_components']}")
    frac = metrics["largest_component_pct"]
    print(f"  largest component              {frac} %   {'OK' if frac > 80 else 'FRAGMENTED'}")
    print(f"  RGB / semantic_class / conf    {metrics['has_rgb']} / {metrics['has_semantic_class']} / {metrics['has_confidence']}")

    # -- mesh quality ------------------------------------------------------
    obj = Path(out_dir) / "output" / "model_textured.obj"
    png = Path(out_dir) / "output" / "model_textured.png"
    if obj.exists():
        import trimesh

        mesh = trimesh.load(str(obj), process=False)
        metrics["mesh_vertices"] = int(len(mesh.vertices))
        metrics["mesh_faces"] = int(len(mesh.faces))
        metrics["mesh_watertight"] = bool(mesh.is_watertight)
        metrics["mesh_winding_consistent"] = bool(mesh.is_winding_consistent)
        metrics["mesh_euler_number"] = int(mesh.euler_number)
        metrics["has_uv"] = bool(getattr(mesh.visual, "uv", None) is not None)
        print("\nTEXTURED MESH")
        print(f"  vertices / faces               {metrics['mesh_vertices']:,} / {metrics['mesh_faces']:,}")
        print(f"  watertight                     {metrics['mesh_watertight']}")
        print(f"  winding consistent             {metrics['mesh_winding_consistent']}")
        print(f"  has UV coordinates             {metrics['has_uv']}")
        if png.exists():
            import cv2

            atlas = cv2.imread(str(png))
            nonblack = float((atlas.sum(axis=2) > 0).mean() * 100)
            metrics["atlas_size"] = list(atlas.shape[:2])
            metrics["atlas_nonblack_pct"] = round(nonblack, 1)
            print(f"  atlas                          {atlas.shape[1]}x{atlas.shape[0]}, {nonblack:.1f}% filled")
    else:
        print("\n  no model_textured.obj — texture bake did not run (check xatlas + mesh)")

    return metrics


m16 = assess(out16, "16 KEYFRAMES")


# %% [CELL 7] ---------------------------------------------------------------
# Depth / normal / class maps -- the visual evidence the geometry is real.

import matplotlib.pyplot as plt

sys.path.insert(0, CODE_DIR)
from drishti3d.export.depthviz import render_keyframe_maps  # noqa: E402
from drishti3d.fusion.mesh import estimate_normals  # noqa: E402
from drishti3d.pipeline.result import PipelineResult  # noqa: E402

result = PipelineResult.load(out16)
viz_dir = Path(out16) / "output" / "maps"

normals = None
try:
    normals = estimate_normals(result.point_cloud, k=30)
except Exception as exc:
    print(f"normals unavailable: {exc}")

written = render_keyframe_maps(
    result.point_cloud,
    result.poses,
    result.keyframes,
    viz_dir,
    max_frames=6,
    normals=normals,
)
print(f"wrote {len(written)} map images to {viz_dir}")
if (viz_dir / "depth_ranges.txt").exists():
    print((viz_dir / "depth_ranges.txt").read_text())

depths = sorted(viz_dir.glob("depth_*.png"))[:3]
normals_png = sorted(viz_dir.glob("normal_*.png"))[:3]
classes = sorted(viz_dir.glob("class_*.png"))[:3]
rows = [r for r in (depths, normals_png, classes) if r]
if rows:
    import cv2

    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(15, 4 * len(rows)))
    axes = np.atleast_2d(axes)
    titles = ["depth", "normal", "class"]
    for r, row in enumerate(rows):
        for c, path in enumerate(row):
            axes[r, c].imshow(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
            axes[r, c].set_title(f"{titles[r]} {path.stem.split('_')[-1]}")
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.show()


# %% [CELL 8] ---------------------------------------------------------------
# Peak VRAM, measured -- and what it means for the 6 GB RTX 4060 target.

import torch

print("=" * 74)
print("VRAM")
print("=" * 74)
for i in range(torch.cuda.device_count()):
    peak = torch.cuda.max_memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  cuda:{i}  peak allocated in THIS process {peak:.2f} GB / {total:.1f} GB")
print(
    "\nNOTE: the pipeline runs in a subprocess (see run()), so the figures above\n"
    "are this notebook's own, not the pipeline's. For the real number run\n"
    "scripts/benchmark_vram.py below, which measures inside the process that\n"
    "actually loads the backbone."
)

!cd {CODE_DIR} && PYTHONPATH={CODE_DIR} python scripts/benchmark_vram.py \
    --backend mapanything \
    --views 2,4,8,16,24,32 \
    --sizes 518,924,1288 \
    --target-vram-gb 6 \
    --out /kaggle/working/vram_t4.json


# %% [CELL 9] ---------------------------------------------------------------
# Scale up. Only run once Cell 6 looks healthy.

out64, t64 = run(64, "64kf")
m64 = assess(out64, "64 KEYFRAMES")


# %% [CELL 10] --------------------------------------------------------------
# FULL FLIGHT. 600 keyframes = the 10-minute-video budget the VRAM benchmark
# extrapolates against. See the note in the other notebook on why an absurd
# --max-keyframes is NOT "no cap".

outfull, tfull = run(600, "full")
mfull = assess(outfull, "FULL FLIGHT")


# %% [CELL 11] --------------------------------------------------------------
# Does a 10-minute video finish inside 15 minutes?

meta = json.loads((Path(outfull) / "meta.json").read_text())
stages = {s["name"]: s.get("elapsed_s", 0) for s in meta.get("stage_results", [])}

nkf = meta.get("report", {}).get("keyframe_count")
if not isinstance(nkf, int) or nkf <= 0:
    nkf = len(meta.get("keyframes", []))
assert nkf > 0, "no keyframes recorded — the run did not get past triage"

VIDEO_S = 242.0  # DJI_0753 duration

print("=" * 74)
print("PER-STAGE TIMING (full flight)")
print("=" * 74)
for name, seconds in stages.items():
    print(f"  {name:<20} {seconds:8.1f} s")
total = sum(stages.values())
print(f"  {'TOTAL':<20} {total:8.1f} s  ({total / 60:.1f} min)")
print(f"\n  keyframes selected   {nkf}")

# Triage and ingest scale with video length; everything else scales with the
# keyframe count.
per_kf = (total - stages.get("triage", 0) - stages.get("ingest", 0)) / max(nkf, 1)
triage_600 = (stages.get("triage", 0) + stages.get("ingest", 0)) * 600 / VIDEO_S
kf_600 = nkf * 600 / VIDEO_S
projected = triage_600 + per_kf * kf_600

print("\n" + "=" * 74)
print("PROJECTION TO A 10-MINUTE VIDEO")
print("=" * 74)
print(f"  keyframes             {kf_600:.0f}")
print(f"  ingest + triage       {triage_600 / 60:.1f} min")
print(f"  per-keyframe cost     {per_kf:.1f} s")
print(f"  PROJECTED TOTAL       {projected / 60:.1f} min")
print(f"  REQUIREMENT           15.0 min")
print(f"  VERDICT               {'PASS' if projected < 900 else 'FAIL'}")


# %% [CELL 12] --------------------------------------------------------------
# Consolidated quality table across all three runs + JSON for the report.

import pandas as pd

rows = [m for m in (m16, m64, mfull) if m.get("n_points")]
if rows:
    keep = [
        "label",
        "n_points",
        "density_pts_per_m2",
        "local_planar_residual_m",
        "largest_component_pct",
        "semantic_labelled_pct",
        "mesh_faces",
        "mesh_watertight",
        "atlas_nonblack_pct",
        "total_s",
    ]
    df = pd.DataFrame(rows)
    df = df[[c for c in keep if c in df.columns]]
    display(df)  # noqa: F821 -- Kaggle/IPython builtin

Path("/kaggle/working/quality_metrics.json").write_text(json.dumps(rows, indent=2, default=str))
print("\nwrote /kaggle/working/quality_metrics.json")


# %% [CELL 13] --------------------------------------------------------------
# Package results for download (excluding the multi-GB clouds).

!cd /kaggle/working && zip -qr results.zip \
    run_*/output/report.txt run_*/output/report.html run_*/meta.json \
    run_*/output/maps/*.png run_*/output/maps/*.txt \
    run_*/output/model_textured.png \
    vram_t4.json quality_metrics.json \
    -x "*.las" "*.ply" "*.obj" "*.xyz"
!ls -lh /kaggle/working/results.zip

# Optionally push full outputs back to HF:
# from huggingface_hub import HfApi
# HfApi().upload_folder(folder_path="/kaggle/working/run_full",
#                       path_in_repo="results/kaggle_quality_run",
#                       repo_id="ajt0704/3chakshu-data", repo_type="dataset",
#                       token=HF_TOKEN)
