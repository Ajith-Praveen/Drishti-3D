# ============================================================================
# 3CHAKSHU — Kaggle headless test notebook
# ============================================================================
# SETTINGS BEFORE YOU RUN (right sidebar):
#   Accelerator : GPU T4 x2       <-- NOT TPU. MapAnything needs CUDA.
#   Internet    : On              <-- needed to pull repos + model weights
#   Persistence : Files only      <-- keeps /kaggle/working between sessions
#
# SECRET (Add-ons -> Secrets):  name = HF_TOKEN, value = your HF token
#   Both repos are private, so this is required. Never paste the token inline.
# ============================================================================


# %% [CELL 1] ---------------------------------------------------------------
# Environment check. Run this FIRST and read the output before continuing.

import subprocess, sys, os

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

try:
    import torch
    print(f"torch            {torch.__version__}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}          {p.name}  {p.total_memory/1e9:.1f} GB")
        print(f"cuda version     {torch.version.cuda}")
        # fp16 tensor cores exist on T4 (sm_75) and above
        cap = torch.cuda.get_device_capability(0)
        print(f"compute cap      {cap[0]}.{cap[1]}  (fp16 tensor cores: {cap[0] >= 7})")
    else:
        print("\n  *** NO CUDA. Check Accelerator = GPU T4 x2, not TPU/None. ***")
except ImportError:
    print("torch not installed")

print(f"\npython           {sys.version.split()[0]}")
print(f"cwd              {os.getcwd()}")


# %% [CELL 2] ---------------------------------------------------------------
# Install dependencies.
# Kaggle ships torch with CUDA already — do NOT let anything reinstall it.
# `--no-deps` on mapanything prevents it dragging in a CPU-only torch.

print("Installing dependencies (3-6 min)...")

# Runtime deps our pipeline needs that Kaggle may not have
!pip install -q av pyproj laspy 2>&1 | tail -2

# open3d gives the SPARSE TSDF path (much faster than our numpy fallback).
# It installs cleanly on Kaggle's Linux image (unlike Apple Silicon).
!pip install -q open3d 2>&1 | tail -2

# MapAnything itself. --no-deps keeps Kaggle's CUDA torch intact, but that
# means WE are now responsible for every one of its real requirements.
# Taken verbatim from map-anything/pyproject.toml's `dependencies`, minus
# torch/torchvision (Kaggle's CUDA build must survive) and minus
# huggingface_hub/tqdm/requests (already on the Kaggle image).
#
# `uniception` is NOT optional: geometry/mapanything.py imports
# `uniception.models.encoders.image_normalizations` at predict() time, so
# omitting it fails the run at inference, not at install.
!pip install -q --no-deps git+https://github.com/facebookresearch/map-anything.git 2>&1 | tail -2
!pip install -q \
    "uniception==0.1.7" \
    hydra-core \
    natsort \
    orjson \
    pillow-heif \
    plyfile \
    python-box \
    safetensors \
    tensorboard \
    trimesh \
    einops \
    jaxtyping \
    "rerun-sdk~=0.24.1" \
    "opencv-python-headless==4.10.0.84" 2>&1 | tail -2

print("\nVerifying the install actually holds together:")
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA broke during install — restart session and rerun"

# Import the two modules that actually matter. If either of these raises,
# the pipeline WILL fail later in Cell 5/6 — better to know now.
from mapanything.models import MapAnything  # noqa: F401
from uniception.models.encoders.image_normalizations import (  # noqa: F401
    IMAGE_NORMALIZATION_DICT,
)
print("  mapanything + uniception import OK")

import open3d
print(f"  open3d {open3d.__version__}  <-- sparse TSDF available")


# %% [CELL 3] ---------------------------------------------------------------
# Pull code + data from your private Hugging Face repos.

from kaggle_secrets import UserSecretsClient
from huggingface_hub import snapshot_download
import os

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

CODE_DIR = "/kaggle/working/drishti3d"
DATA_DIR = "/kaggle/working/data"

print("Downloading code...")
snapshot_download(
    repo_id="ajt0704/3chakshu",
    repo_type="model",
    local_dir=CODE_DIR,
    token=HF_TOKEN,
)

print("Downloading data (5.8 GB, several minutes)...")
snapshot_download(
    repo_id="ajt0704/3chakshu-data",
    repo_type="dataset",
    local_dir=DATA_DIR,
    token=HF_TOKEN,
)

from pathlib import Path

print("\nCode tree:")
for p in sorted(Path(CODE_DIR).iterdir()):
    print("  ", p.name)

print("\nData files:")
for p in sorted(Path(DATA_DIR).rglob("*")):
    if p.is_file() and p.suffix.upper() in {".MP4", ".MOV", ".CSV", ".SRT"}:
        print(f"   {p.stat().st_size/1e9:6.2f} GB  {p.relative_to(DATA_DIR)}")

os.environ["PYTHONPATH"] = CODE_DIR
sys.path.insert(0, CODE_DIR)


# %% [CELL 4] ---------------------------------------------------------------
# Sanity check: does the package import and see the GPU?

import subprocess
r = subprocess.run(
    [sys.executable, "-c",
     "from drishti3d.device import get_device, device_report;"
     "print('device:', get_device());"
     "print('report:', device_report())"],
    cwd=CODE_DIR, capture_output=True, text=True,
    env={**os.environ, "PYTHONPATH": CODE_DIR},
)
print(r.stdout or r.stderr)
assert "cuda" in r.stdout, "pipeline is not seeing CUDA"


# %% [CELL 5] ---------------------------------------------------------------
# VRAM benchmark. Gives the real CUDA numbers and extrapolates to a 6 GB 4060.
# ~5 min. First run downloads MapAnything weights (~4.6 GB).

!cd {CODE_DIR} && PYTHONPATH={CODE_DIR} python scripts/benchmark_vram.py \
    --backend mapanything \
    --views 2,4,8,16,24,32 \
    --sizes 518,924,1288 \
    --target-vram-gb 6 \
    --out /kaggle/working/vram_t4.json


# %% [CELL 6] ---------------------------------------------------------------
# Staged pipeline runs. Start small, confirm it works, then scale up.
# Each writes to its own output dir so nothing is overwritten.

VIDEO = f"{DATA_DIR}/survey_nadir/DJI_0753.MP4"
TELEM = f"{DATA_DIR}/survey_nadir/DJI_0753_airdata.csv"

import time, json

def run(max_kf, tag, extra=""):
    out = f"/kaggle/working/run_{tag}"
    cmd = (
        f"cd {CODE_DIR} && PYTHONPATH={CODE_DIR} python -m drishti3d.pipeline.runner "
        f"'{VIDEO}' --telemetry '{TELEM}' --backbone mapanything "
        f"--max-keyframes {max_kf} --out {out} --log-level ERROR {extra}"
    )
    print(f"\n{'='*70}\nRUN: {tag}  ({max_kf} keyframes)\n{'='*70}")
    t0 = time.time()
    os.system(cmd)
    el = time.time() - t0
    print(f"\n>>> {tag}: {el/60:.1f} min wall clock")
    return out, el

# Step 1 — smoke test. If this fails, stop and read the error.
out16, t16 = run(16, "16kf")


# %% [CELL 7] ---------------------------------------------------------------
# Inspect the 16-keyframe result before spending time on a bigger run.

import numpy as np, laspy
from scipy.spatial import cKDTree

def inspect(out_dir, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    rpt = f"{out_dir}/output/report.txt"
    if os.path.exists(rpt):
        print(open(rpt).read()[:900])

    pc = f"{out_dir}/output/point_cloud.las"
    if not os.path.exists(pc):
        print("no point_cloud.las produced")
        return
    las = laspy.read(pc)
    xyz = np.vstack([las.x, las.y, las.z]).T
    rgb_max = int(np.asarray(las.red).max())

    # local planar residual — the depth-precision metric
    rng = np.random.default_rng(0)
    centres = xyz[rng.choice(len(xyz), min(400, len(xyz)), replace=False)]
    tree = cKDTree(xyz[:, :2])
    res = []
    for p in centres:
        idx = tree.query_ball_point(p[:2], 0.5)
        if len(idx) < 25:
            continue
        q = xyz[idx]
        A = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
        co, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        res.append((q[:, 2] - A @ co).std())

    # connectivity — is the model one surface or many fragments?
    gs = 1.0
    ix = ((xyz[:, 0] - xyz[:, 0].min()) / gs).astype(int)
    iy = ((xyz[:, 1] - xyz[:, 1].min()) / gs).astype(int)
    grid = np.zeros((iy.max() + 1, ix.max() + 1), bool)
    grid[iy, ix] = True
    from scipy import ndimage
    lab, n = ndimage.label(grid)
    sizes = np.bincount(lab.ravel())[1:]
    frac = sizes.max() / sizes.sum() if len(sizes) else 0

    print(f"points                {len(xyz):,}")
    print(f"extent                {np.round(xyz.max(0)-xyz.min(0),1)} m")
    print(f"RGB max               {rgb_max}   {'OK' if rgb_max>0 else 'MISSING COLOUR'}")
    print(f"local 1m residual     {np.median(res):.2f} m   {'OK' if np.median(res)<1 else 'TOO HIGH'}")
    print(f"connected components  {n}")
    print(f"largest component     {frac*100:.1f}%   {'OK' if frac>0.8 else 'FRAGMENTED'}")

inspect(out16, "16 KEYFRAMES")


# %% [CELL 8] ---------------------------------------------------------------
# Scale up. Only run this once Cell 7 looks healthy.
# 64 keyframes first, then the full flight.

out64, t64 = run(64, "64kf")
inspect(out64, "64 KEYFRAMES")


# %% [CELL 9] ---------------------------------------------------------------
# FULL FLIGHT — this is the 15-minute-budget test.
#
# NOTE: --max-keyframes sets config.triage.target_keyframes, which is a
# TARGET, not a hard ceiling: triage's adaptive threshold (selector.py's
# _ADAPT_OVERSHOOT_FACTOR path) scales itself to hit that target rather
# than truncating the video early. Passing an absurd number therefore
# disables the throttle entirely and can select thousands of keyframes —
# a session-timeout and VRAM risk, not a "no cap" flag. 600 is the honest
# full-flight number: the 10-minute-video keyframe budget the VRAM
# benchmark already extrapolates against (--extrapolate-keyframes 600).
outfull, tfull = run(600, "full")
inspect(outfull, "FULL FLIGHT")


# %% [CELL 10] --------------------------------------------------------------
# The answer: does a 10-minute video finish within 15 minutes?

import json

meta = json.load(open(f"{outfull}/meta.json"))
stages = {s["name"]: s.get("elapsed_s", 0) for s in meta.get("stage_results", [])}

# build_report (export/report.py:_SCALAR_METRICS) emits `keyframe_count`.
# There is no `triage_keyframes_selected` key — reading one silently
# yields 0, which collapses the projection below to triage-time-only and
# prints PASS unconditionally. Fall back to the saved keyframe list,
# which result.save() always round-trips.
nkf = meta.get("report", {}).get("keyframe_count")
if not isinstance(nkf, int) or nkf <= 0:
    nkf = len(meta.get("keyframes", []))
assert nkf > 0, "no keyframes recorded — the run did not get past triage"

vid_s = 242.0          # DJI_0753 duration

print("=" * 70)
print("PER-STAGE TIMING (full flight)")
print("=" * 70)
for k, v in stages.items():
    print(f"  {k:<20} {v:8.1f} s")
total = sum(stages.values())
print(f"  {'TOTAL':<20} {total:8.1f} s  ({total/60:.1f} min)")
print(f"\n  keyframes selected   {nkf}")

# triage scales with video length; the rest scales with keyframe count
per_kf = (total - stages.get("triage", 0) - stages.get("ingest", 0)) / max(nkf, 1)
triage_600 = stages.get("triage", 0) * 600 / vid_s
kf_600 = nkf * 600 / vid_s
proj = triage_600 + per_kf * kf_600

print("\n" + "=" * 70)
print("PROJECTION TO A 10-MINUTE VIDEO")
print("=" * 70)
print(f"  keyframes             {kf_600:.0f}")
print(f"  triage                {triage_600/60:.1f} min")
print(f"  per-keyframe cost     {per_kf:.1f} s")
print(f"  PROJECTED TOTAL       {proj/60:.1f} min")
print(f"  REQUIREMENT           15.0 min")
print(f"  VERDICT               {'PASS' if proj < 900 else 'FAIL'}")


# %% [CELL 11] --------------------------------------------------------------
# Optional: COLMAP baseline on the same input. Strong comparison evidence.
# Slow — run only if you have session time left.

# !apt-get -qq install -y colmap > /dev/null 2>&1
# !colmap --help | head -3


# %% [CELL 12] --------------------------------------------------------------
# Package results for download (exclude the huge point clouds).

!cd /kaggle/working && zip -qr results.zip \
    run_*/output/report.txt run_*/output/report.html run_*/meta.json \
    vram_t4.json -x "*.las" "*.ply" "*.obj"
!ls -lh /kaggle/working/results.zip

# Optionally push the full outputs back to HF:
# from huggingface_hub import HfApi
# HfApi().upload_folder(folder_path="/kaggle/working/run_full",
#                       path_in_repo="results/kaggle_full_run",
#                       repo_id="ajt0704/3chakshu-data", repo_type="dataset",
#                       token=HF_TOKEN)
