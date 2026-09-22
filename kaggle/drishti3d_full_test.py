# ============================================================================
# DRISHTI-3D — FULL SYSTEM TEST on Kaggle (GPU, private HF repos)
# ============================================================================
# SETTINGS BEFORE YOU RUN (right sidebar):
#   Accelerator : GPU T4 x2       <-- NOT TPU
#   Internet    : On
#   Persistence : Files only
#
# SECRET (Add-ons -> Secrets):  name = HF_TOKEN
#   Both repos are private. Never paste the token inline -- Kaggle notebooks
#   are trivially shareable and a pasted token leaks with the first share.
#
# ---------------------------------------------------------------------------
# BEFORE YOU RUN THIS: push your current code
# ---------------------------------------------------------------------------
#   HF_TOKEN=hf_xxx python scripts/push_to_hf.py
#
# The HF code repo is a SNAPSHOT. If you have edited anything locally since
# the last push, this notebook would benchmark code you are no longer
# writing. Cell 4 refuses to continue if the snapshot is missing modules
# that should exist -- but it cannot detect a snapshot that is merely OLD,
# so push first, every time.
#
# ---------------------------------------------------------------------------
# ABOUT THE "30 GB"
# ---------------------------------------------------------------------------
# Kaggle's T4 x2 is TWO SEPARATE 16 GB devices, not one 32 GB pool. No model
# here shards across devices, so 16 GB per device is the ceiling that
# governs whether anything fits -- a second card does NOT raise it.
#
# The main runs use ONE GPU on purpose (GeometryConfig.max_devices=1): the
# deployment target has one, only the geometry stage parallelises anyway
# (~19% off total wall clock, not 2x), and the multi-device path is the
# least-exercised code here. Cell 10 measures the 1-vs-2 difference so the
# trade is evidence rather than assumption.
# ============================================================================


# %% [CELL 1] ---------------------------------------------------------------
# Environment.

import os
import subprocess
import sys

print("=" * 76)
print("ENVIRONMENT")
print("=" * 76)

try:
    import torch

    print(f"torch            {torch.__version__}")
    print(f"cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i}         {p.name}  {p.total_memory / 1e9:.1f} GB")
        cap = torch.cuda.get_device_capability(0)
        print(f"cuda version     {torch.version.cuda}")
        print(f"compute cap      {cap[0]}.{cap[1]}  (fp16 tensor cores: {cap[0] >= 7})")
        print(f"\ndevice count     {torch.cuda.device_count()}  <-- windows run one per device")
        print(f"per-device VRAM  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB  <-- the real ceiling")
    else:
        print("\n  *** NO CUDA. Set Accelerator = GPU T4 x2, not TPU/None. ***")
except ImportError:
    print("torch not installed")

print(f"\npython           {sys.version.split()[0]}")


# %% [CELL 2] ---------------------------------------------------------------
# Install. Kaggle's CUDA torch must survive -- nothing here may replace it.

print("Installing (5-8 min)...")

!pip install -q av pyproj laspy 2>&1 | tail -2
!pip install -q open3d 2>&1 | tail -2
!pip install -q xatlas 2>&1 | tail -2

# MapAnything with --no-deps to protect Kaggle's CUDA torch, which makes US
# responsible for its real requirements. Taken from its own pyproject.
!pip install -q --no-deps git+https://github.com/facebookresearch/map-anything.git 2>&1 | tail -2
!pip install -q \
    "uniception==0.1.7" \
    hydra-core natsort orjson pillow-heif plyfile python-box \
    safetensors tensorboard trimesh einops jaxtyping \
    "rerun-sdk~=0.24.1" "opencv-python-headless==4.10.0.84" 2>&1 | tail -2

# SegFormer, for semantic classification + dynamic-object masking.
!pip install -q transformers 2>&1 | tail -2

print("\nVerifying every import the pipeline actually needs:")
import torch

assert torch.cuda.is_available(), "CUDA broke during install — restart session and rerun"
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")

from mapanything.models import MapAnything  # noqa: F401
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT  # noqa: F401
from transformers import SegformerForSemanticSegmentation  # noqa: F401
import open3d
import xatlas  # noqa: F401

print(f"  mapanything + uniception + transformers + xatlas OK, open3d {open3d.__version__}")


# %% [CELL 3] ---------------------------------------------------------------
# Pull code + data from the PRIVATE Hugging Face repos.

from pathlib import Path

from huggingface_hub import snapshot_download
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

CODE_REPO = "ajt0704/3chakshu"          # private, repo_type="model"
DATA_REPO = "ajt0704/3chakshu-data"     # private, repo_type="dataset"
CODE_DIR = "/kaggle/working/drishti3d"
DATA_DIR = "/kaggle/working/data"

# Which flight to reconstruct. The data repo holds two, ~3.1 GB each:
#
#   "survey_nadir"     DJI_0753.MP4 + Airdata CSV -- 242 s nadir survey strip.
#                      The canonical single-pass case this whole system targets.
#   "vertical_ascent"  DJI_2026...D.MP4 + SRT     -- a vertical ascent, with
#                      genuinely different capture geometry (see
#                      geometry.flight_profile). Worth a SECOND run, because it
#                      exercises the oblique/facade path a nadir strip cannot.
#
# Only this one is downloaded. Pulling both costs several minutes of session
# time for footage no cell touches; change the string and re-run cell 3 to
# switch flights.
FLIGHT = "survey_nadir"

print(f"Downloading code from {CODE_REPO} ...")
snapshot_download(repo_id=CODE_REPO, repo_type="model", local_dir=CODE_DIR, token=HF_TOKEN)

print(f"Downloading {FLIGHT} from {DATA_REPO} (~3.1 GB, a few minutes) ...")
snapshot_download(
    repo_id=DATA_REPO,
    repo_type="dataset",
    local_dir=DATA_DIR,
    token=HF_TOKEN,
    allow_patterns=[f"{FLIGHT}/*"],
)

os.environ["PYTHONPATH"] = CODE_DIR
sys.path.insert(0, CODE_DIR)

print("\nData files:")
for p in sorted(Path(DATA_DIR).rglob("*")):
    if p.is_file() and p.suffix.upper() in {".MP4", ".MOV", ".CSV", ".SRT"}:
        print(f"   {p.stat().st_size / 1e9:6.2f} GB  {p.relative_to(DATA_DIR)}")

# Resolve the video and its telemetry from what actually landed on disk,
# rather than hard-coding a filename per flight. The two flights use
# different naming AND different telemetry formats (Airdata CSV vs DJI SRT),
# and `ingest.telemetry.load_telemetry` handles either -- so discovering them
# is all that is needed to make FLIGHT the single switch.
flight_dir = Path(DATA_DIR) / FLIGHT
videos = sorted(p for p in flight_dir.rglob("*") if p.suffix.upper() in {".MP4", ".MOV"})
sidecars = sorted(p for p in flight_dir.rglob("*") if p.suffix.upper() in {".CSV", ".SRT", ".GPX"})
assert videos, f"no video found under {flight_dir} — check FLIGHT and the data repo"

VIDEO = str(videos[0])
TELEM = str(sidecars[0]) if sidecars else None
print(f"\nvideo     : {VIDEO}")
print(f"telemetry : {TELEM or '(none — the run will have no GPS priors)'}")


# %% [CELL 4] ---------------------------------------------------------------
# FRESHNESS GATE. Refuse to benchmark a stale snapshot.

REQUIRED_MODULES = [
    "drishti3d/semantics/segmenter.py",
    "drishti3d/semantics/labelling.py",
    "drishti3d/semantics/classes.py",
    "drishti3d/fusion/texture.py",
    "drishti3d/fusion/completion.py",
    "drishti3d/export/terrain.py",
    "drishti3d/export/depthviz.py",
    "drishti3d/ingest/photometric.py",
]

missing = [m for m in REQUIRED_MODULES if not (Path(CODE_DIR) / m).exists()]
print("CODE FRESHNESS")
for m in REQUIRED_MODULES:
    print(f"  {'OK  ' if (Path(CODE_DIR) / m).exists() else 'MISS'}  {m}")

if missing:
    raise SystemExit(
        f"\nSTALE SNAPSHOT: {len(missing)} module(s) missing from {CODE_REPO}.\n"
        "Run this locally, then re-run this notebook:\n"
        "    HF_TOKEN=hf_xxx python scripts/push_to_hf.py\n"
    )

# Multi-GPU support is newer than the modules above; check it explicitly.
src = (Path(CODE_DIR) / "drishti3d/device.py").read_text()
assert "available_devices" in src, "stale snapshot: device.py has no available_devices() — re-push"
print("\n  multi-GPU support present")

r = subprocess.run(
    [
        sys.executable,
        "-c",
        "from drishti3d.device import available_devices, device_report;"
        "print('devices:', available_devices());"
        "print('report :', device_report())",
    ],
    cwd=CODE_DIR,
    capture_output=True,
    text=True,
    env={**os.environ, "PYTHONPATH": CODE_DIR},
)
print("\n" + (r.stdout or r.stderr))
assert "cuda" in r.stdout, "pipeline is not seeing CUDA"


# %% [CELL 5] ---------------------------------------------------------------
# Run the unit test suite on Linux + CUDA. 323 tests, none needing a GPU --
# this proves the snapshot is internally consistent before we spend an hour
# of session time on reconstruction runs.

!cd {CODE_DIR} && pip install -q pytest 2>&1 | tail -1
!cd {CODE_DIR} && PYTHONPATH={CODE_DIR} python -m pytest -q \
    --ignore=tests/test_app.py -p no:cacheprovider 2>&1 | tail -15


# %% [CELL 5B] --------------------------------------------------------------
# VRAM sweep -- BEFORE configuring, not after.
#
# This measures the real memory curve for this GPU across view counts and
# resolutions, and extrapolates to a 6 GB deployment target. Running it
# after the pipeline (where it used to live) meant the pipeline config was
# a guess; an earlier config asked for 1288 px x 14 views and OOM'd every
# window, which one look at this table would have prevented.
#
# Read the row matching the next cell's max_image_size / window_size before
# accepting that config.

!cd {CODE_DIR} && PYTHONPATH={CODE_DIR} python scripts/benchmark_vram.py \
    --backend mapanything \
    --views 2,4,8,14 \
    --sizes 518,924,1288 \
    --target-vram-gb 6 \
    --out /kaggle/working/vram_t4.json


# %% [CELL 6] ---------------------------------------------------------------
# GPU config. Single-device by choice -- see max_devices below.

import yaml

CONFIG_PATH = "/kaggle/working/kaggle_gpu.yaml"

config = {
    # "accurate" retunes triage baseline spacing, matching and the semantic
    # agreement thresholds together. It also sets geometry.max_image_size to
    # 1288, which the explicit geometry section below overrides -- see the
    # arithmetic in that comment for why 1288 cannot run on a T4.
    "quality_profile": "accurate",
    "geometry": {
        "backbone": "mapanything",
        # ---------------------------------------------------------------
        # 518 px -- MEASURED on this exact GPU, not estimated.
        #
        # The cell 5B sweep on a Tesla T4 (14.56 GB):
        #     518px x2 views ->  8.63 GB reserved   ok
        #     518px x4 views -> 14.00 GB reserved   ok, at the edge
        #     everything larger -> OOM
        #
        # MapAnything's encoder here is DINOv2 **ViT-g/14** -- the giant
        # variant, 4.91 GB of weights alone -- so activations cost ~2.3 GB
        # per view at 518 px and scale with pixel count. At 924 px that is
        # ~7.2 GB/view, so even a SINGLE view needs ~12.1 GB and two need
        # ~19.4 GB: a T4 cannot run 924 px at all.
        #
        # This matters for the accuracy claim: 518 px measured 1.447 m
        # relative RMSE locally, which FAILS the <= 1 m spec, while 924 px
        # measured 0.863 m and passes. A 16 GB T4 therefore cannot reach
        # the required accuracy with this backbone -- that is a hardware
        # finding, not a tuning problem, and it is why the accuracy numbers
        # in the report come from the higher-memory machine.
        #
        # 518/3 is chosen over 518/4 because 14.00 GB of 14.56 GB leaves no
        # room for fragmentation.
        # ---------------------------------------------------------------
        "max_image_size": 518,
        # 3 is a REQUEST, not a guarantee: plan_window_size caps it against
        # 80% of free VRAM (see _VRAM_BUDGET_FRACTION). Being given fewer
        # views than asked for is the planner working, not a failure.
        "window_size": 3,
        # 1 = single device, deliberately, even though Kaggle offers two.
        #
        # Two reasons. First, the deployment target is a single GPU, so a
        # dual-T4 timing describes hardware no user has -- the number worth
        # quoting is the one-GPU number. Second, only the geometry stage
        # parallelises across devices (~39% of runtime), so by Amdahl the
        # second card buys ~19% off total wall clock, not 2x -- a small
        # gain for the least-exercised code path in the pipeline.
        #
        # Cell 10 still measures both, so the difference is evidence rather
        # than an assumption. Set 0 there, not here.
        "max_devices": 1,
        # Decode the next window while the GPU works on the current one.
        "prefetch_frames": True,
    },
    "semantics": {
        "enabled": True,
        "model": "segformer",
        "checkpoint": "nvidia/segformer-b4-finetuned-ade-512-512",
        # Matches the geometry resolution; segmenting finer than the
        # geometry it is projected onto buys nothing.
        "max_image_size": 518,
        "batch_size": 4,
        "remove_dynamic": True,
        "min_vote_ratio": 0.6,
        # 2, not 3. A point is only labelled if this many segmented views
        # saw it; on a single-pass flight 3 leaves most of the cloud
        # UNLABELLED (measured: 78.9% at min_views=3 locally).
        "min_views": 2,
    },
    "texture": {
        "enabled": True,
        # 4096, not 8192. The bake is host-RAM bound and export measured
        # 17.2% of total runtime locally -- 8192 quadruples that cost for
        # detail beyond what a 924 px source can actually resolve.
        "texture_size": 4096,
        "blend_views": 4,
        "max_views": 120,
    },
    "fusion": {
        # open3d is a core dependency now, so this takes the SPARSE TSDF
        # path. The dense numpy fallback measured 33.5% of total runtime.
        "voxel_count_budget": 8_000_000,
        "pre_tsdf_max_points": 6_000_000,
        # Grade confidence against the SOURCE FRAMES rather than trusting
        # the backbone's self-report. A point several views agree about has
        # been checked against several photographs; one they disagree about
        # is wrong however confident the network was. This is what makes
        # `confidence_source` read `photometric_verification` instead of
        # `backbone_confidence_and_view_count`.
        "photometric_verify": True,
        "photometric_agree_threshold": 12.0,
        "photometric_min_views": 3,
        "photometric_min_contrast": 6.0,
        # Strip Poisson spike slivers and speck islands before texturing.
        # Only deletes faces -- never smooths or moves a vertex, because a
        # nudged surface is no longer one anybody measured.
        "mesh_cleanup": True,
        "mesh_max_edge_factor": 6.0,
    },
}

Path(CONFIG_PATH).write_text(yaml.safe_dump(config, sort_keys=False))
print(Path(CONFIG_PATH).read_text())


# %% [CELL 7] ---------------------------------------------------------------
# Run helper.

import json
import time

# VIDEO / TELEM were resolved in cell 3 from whatever FLIGHT downloaded.


def run(max_kf, tag, config_path=CONFIG_PATH, extra=""):
    """Run the pipeline; returns (out_dir, wall_clock_seconds)."""
    out = f"/kaggle/working/run_{tag}"
    # --telemetry is omitted entirely when the flight has no sidecar, rather
    # than passed as an empty string: the runner treats a missing path as an
    # error, but treats absent telemetry as a legitimate (if less accurate)
    # run with no GPS priors.
    telem_arg = f"--telemetry '{TELEM}' " if TELEM else ""
    cmd = (
        f"cd {CODE_DIR} && PYTHONPATH={CODE_DIR} "
        # Fragmentation is the usual cause of a spurious OOM on long runs.
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"python -m drishti3d.pipeline.runner "
        f"'{VIDEO}' {telem_arg}--config {config_path} "
        f"--backbone mapanything --max-keyframes {max_kf} "
        f"--out {out} --log-level INFO {extra} 2>&1 | tail -70"
    )
    print(f"\n{'=' * 76}\nRUN: {tag}  (target {max_kf} keyframes)\n{'=' * 76}")
    t0 = time.time()
    os.system(cmd)
    elapsed = time.time() - t0
    print(f"\n>>> {tag}: {elapsed / 60:.1f} min wall clock")
    return out, elapsed


# Smoke test. If this fails, stop and read the error rather than running the
# full flight on an already-broken pipeline.
out16, t16 = run(16, "16kf")

print("\nDeliverables written:")
for p in sorted(Path(f"{out16}/output").glob("*")):
    print(f"   {p.stat().st_size / 1e6:9.2f} MB  {p.name}")


# %% [CELL 7B] --------------------------------------------------------------
# LIVE PROGRESS. The runner writes a top-down snapshot after every completed
# geometry window into <out>/progress. Watching these grow is how you tell a
# working long run from a stalled one -- and `latest.png` is a stable path
# you can re-display from another cell WHILE a run is still going.

import matplotlib.pyplot as plt

import cv2

prog_dir = Path(out16) / "progress"
snaps = sorted(prog_dir.glob("snapshot_*.png"))
print(f"{len(snaps)} progress snapshots in {prog_dir}")

if snaps:
    # First, middle, last: the scene should visibly fill in across them. If
    # the last looks like the first, windows completed without adding
    # geometry -- which is a silent failure the stage table will not show.
    picks = [snaps[0], snaps[len(snaps) // 2], snaps[-1]]
    fig, axes = plt.subplots(1, len(picks), figsize=(16, 5.5))
    for ax, p in zip(np.atleast_1d(axes), picks):
        ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        ax.set_title(p.stem)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("none written — geometry produced no partial results (see the traceback above)")


# %% [CELL 8] ---------------------------------------------------------------
# Deliverable checklist: did every feature actually produce its artefact?

EXPECTED = {
    "point_cloud.las": "raw fused cloud (full detail)",
    "point_cloud.ply": "same, PLY",
    "model.las": "meshed/primary deliverable",
    "model.glb": "web-ready mesh",
    "model_textured.obj": "UV-mapped textured mesh",
    "model_textured.mtl": "material referencing the atlas",
    "model_textured.png": "photographic texture atlas",
    "dsm.tif": "digital surface model",
    "dtm.tif": "bare-earth terrain model",
    "dtm_interpolated.tif": "where the DTM is a guess, not a measurement",
    "height_above_ground.tif": "DSM - DTM (canopy/building height)",
    "orthomosaic.tif": "orthorectified imagery",
    "confidence.tif": "per-cell confidence raster",
    "facades_inferred.las": "extruded facades, tagged INFERRED",
    "report.txt": "accuracy report card",
    "report.html": "same, styled",
}

out_dir = Path(out16) / "output"
print(f"{'ARTEFACT':<26} {'STATUS':<8} WHAT IT IS")
print("-" * 76)
produced = 0
for name, desc in EXPECTED.items():
    hit = out_dir / name
    # GeoTIFF writers fall back to .npy+.tfw when neither rasterio nor
    # tifffile is present; count that as produced, not missing.
    alt = out_dir / (name.replace(".tif", ".npy"))
    ok = hit.exists() or alt.exists()
    produced += ok
    print(f"{name:<26} {'OK' if ok else 'MISSING':<8} {desc}")
print("-" * 76)
print(f"{produced}/{len(EXPECTED)} deliverables produced")


# %% [CELL 9] ---------------------------------------------------------------
# Quality + accuracy assessment.

import numpy as np

sys.path.insert(0, CODE_DIR)


def assess(out_dir_str, label):
    """Full quality assessment of one run. Returns a metrics dict."""
    import laspy
    from scipy import ndimage
    from scipy.spatial import cKDTree

    print(f"\n{'=' * 76}\n{label}\n{'=' * 76}")
    metrics = {"label": label}

    meta_path = Path(out_dir_str) / "meta.json"
    if not meta_path.exists():
        print("  no meta.json — the run did not complete")
        return metrics
    meta = json.loads(meta_path.read_text())
    report = meta.get("report", {})

    stages = {s["name"]: s.get("elapsed_s", 0) for s in meta.get("stage_results", [])}
    statuses = {s["name"]: s.get("status") for s in meta.get("stage_results", [])}
    print("\nSTAGES")
    for name, seconds in stages.items():
        print(f"  {name:<20} {seconds:8.1f} s   [{statuses.get(name)}]")
    metrics["stage_timings_s"] = stages
    metrics["stage_statuses"] = statuses
    metrics["total_s"] = sum(stages.values())

    # Print the FULL traceback for any failed stage. The stage table
    # truncates messages to one line, which for a traceback is the useless
    # line -- literally "Traceback (most recent call last):". Anything that
    # failed is the most important thing on this page, so it gets the space.
    failures = [s for s in meta.get("stage_results", []) if s.get("status") == "failed"]
    for s in failures:
        print(f"\n{'!' * 76}\nFAILED STAGE: {s['name']}\n{'!' * 76}")
        print(s.get("message", "(no message recorded)"))
    metrics["failed_stages"] = [s["name"] for s in failures]

    # Per-window geometry failures. A stage can be "ok" overall and still
    # have lost most of its windows, so these are printed whether or not
    # the stage itself failed.
    for s in meta.get("stage_results", []):
        window_errors = (s.get("artifacts") or {}).get("window_errors") or {}
        if window_errors:
            print(f"\n{'!' * 76}\n{s['name'].upper()}: {len(window_errors)} WINDOW FAILURE(S)\n{'!' * 76}")
            for idx, err in sorted(window_errors.items(), key=lambda kv: int(kv[0])):
                print(f"--- window {idx} ---")
                print(err)
            metrics["window_errors"] = window_errors
    if failures:
        print(
            f"\n{len(failures)} stage(s) failed -- the metrics below are from a partial run "
            "and should not be quoted as results."
        )

    print("\nACCURACY (the pipeline's own report card)")
    for key in (
        "relative_rmse_m",
        "absolute_rmse_m",
        "scale_error_pct",
        "mean_reprojection_error_px",
        "coverage_pct",
        "keyframe_count",
        "confidence_source",
    ):
        print(f"  {key:<32} {report.get(key)}")
        metrics[key] = report.get(key)

    breakdown = report.get("confidence_breakdown_pct")
    if isinstance(breakdown, dict):
        print("\nCONFIDENCE TIERS")
        for k, v in breakdown.items():
            print(f"  {k:<32} {v:.1f} %")
        metrics["confidence_breakdown_pct"] = breakdown
        measured = breakdown.get("measured_pct")
        if measured is not None:
            metrics["measured_pct"] = measured
            print(f"  {'-> MEASURED':<32} {measured:.1f} %   target >= 90%")

    # Photometric verification -- the evidence behind the tier above.
    photo = {k: v for k, v in report.items() if k.startswith("photometric_")}
    if photo:
        print("\nPHOTOMETRIC VERIFICATION (graded against the source frames)")
        for k in (
            "photometric_verified_pct",
            "photometric_verified_points",
            "photometric_inconsistent_points",
            "photometric_unverifiable_points",
            "photometric_median_error",
            "photometric_promoted",
            "photometric_demoted",
            "photometric_views_used",
        ):
            if report.get(k) not in (None, "not computed"):
                print(f"  {k.replace('photometric_', ''):<32} {report[k]}")
                metrics[k] = report[k]
        print("  (unverifiable = too few views or too little texture to judge --")
        print("   NOT the same as wrong, and never promoted on absent evidence)")

    hist = report.get("semantic_class_pct")
    if isinstance(hist, dict):
        print("\nSCENE COMPOSITION (per reconstructed point)")
        for name, pct in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<32} {pct:5.1f} %")
        metrics["semantic_class_pct"] = hist

    print("\nSEMANTICS / TERRAIN / INFERRED GEOMETRY")
    for key in (
        "semantic_labelled_pct",
        "semantic_disputed_points",
        "semantic_unseen_points",
        "semantic_model",
        "dynamic_points_removed",
        "dtm_method",
        "ground_point_pct",
        "dtm_interpolated_pct",
        "facade_structures_completed",
        "facade_points_inferred",
    ):
        value = report.get(key)
        if value not in (None, "not computed"):
            print(f"  {key:<32} {value}")
            metrics[key] = value

    pc_path = Path(out_dir_str) / "output" / "point_cloud.las"
    if not pc_path.exists():
        print("\n  no point_cloud.las produced")
        return metrics

    las = laspy.read(str(pc_path))
    xyz = np.vstack([las.x, las.y, las.z]).T
    metrics["n_points"] = int(len(xyz))
    extent = xyz.max(0) - xyz.min(0)
    metrics["extent_m"] = [round(float(v), 1) for v in extent]
    metrics["density_pts_per_m2"] = round(len(xyz) / max(float(extent[0] * extent[1]), 1.0), 1)

    # Local planar residual -- the depth-precision metric.
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
    ix = ((xyz[:, 0] - xyz[:, 0].min()) / 1.0).astype(int)
    iy = ((xyz[:, 1] - xyz[:, 1].min()) / 1.0).astype(int)
    grid = np.zeros((iy.max() + 1, ix.max() + 1), bool)
    grid[iy, ix] = True
    lab, n_comp = ndimage.label(grid)
    sizes = np.bincount(lab.ravel())[1:]
    metrics["connected_components"] = int(n_comp)
    metrics["largest_component_pct"] = round(float(sizes.max() / sizes.sum() * 100), 1) if len(sizes) else 0.0

    dims = set(las.point_format.dimension_names)
    metrics["has_rgb"] = "red" in dims and int(np.asarray(las.red).max()) > 0
    metrics["has_semantic_class"] = "semantic_class" in dims
    metrics["has_confidence"] = "confidence" in dims
    if "classification" in dims:
        codes = np.unique(np.asarray(las.classification))
        metrics["asprs_codes_present"] = codes.tolist()

    print("\nPOINT CLOUD")
    print(f"  points                          {metrics['n_points']:,}")
    print(f"  extent (m)                      {metrics['extent_m']}")
    print(f"  density (pts/m2)                {metrics['density_pts_per_m2']}")
    res = metrics["local_planar_residual_m"]
    print(f"  local 1m planar residual (m)    {res}   {'OK' if res and res < 1.0 else 'HIGH'}")
    frac = metrics["largest_component_pct"]
    print(f"  largest component               {frac} %   {'OK' if frac > 80 else 'FRAGMENTED'}")
    print(f"  RGB / semantic / confidence     {metrics['has_rgb']} / {metrics['has_semantic_class']} / {metrics['has_confidence']}")
    print(f"  ASPRS codes in LAS              {metrics.get('asprs_codes_present')}")

    obj = Path(out_dir_str) / "output" / "model_textured.obj"
    png = Path(out_dir_str) / "output" / "model_textured.png"
    if obj.exists():
        import trimesh

        mesh = trimesh.load(str(obj), process=False)
        metrics["mesh_vertices"] = int(len(mesh.vertices))
        metrics["mesh_faces"] = int(len(mesh.faces))
        metrics["mesh_watertight"] = bool(mesh.is_watertight)
        metrics["mesh_winding_consistent"] = bool(mesh.is_winding_consistent)
        metrics["has_uv"] = bool(getattr(mesh.visual, "uv", None) is not None)
        print("\nTEXTURED MESH")
        print(f"  vertices / faces                {metrics['mesh_vertices']:,} / {metrics['mesh_faces']:,}")
        print(f"  watertight / winding-consistent {metrics['mesh_watertight']} / {metrics['mesh_winding_consistent']}")
        print(f"  has UV coordinates              {metrics['has_uv']}")
        if png.exists():
            import cv2

            atlas = cv2.imread(str(png))
            metrics["atlas_size"] = list(atlas.shape[:2])
            metrics["atlas_nonblack_pct"] = round(float((atlas.sum(axis=2) > 0).mean() * 100), 1)
            print(f"  atlas                           {atlas.shape[1]}x{atlas.shape[0]}, {metrics['atlas_nonblack_pct']}% filled")
    else:
        print("\n  no model_textured.obj — texture bake did not run")

    return metrics


m16 = assess(out16, "16 KEYFRAMES")

# The report card only exists if ExportStage ran. On a partial run it will
# not, and reading it unconditionally replaces the diagnosis above with an
# unrelated FileNotFoundError.
report_txt = Path(out16) / "output" / "report.txt"
if report_txt.exists():
    print("\n\nREPORT CARD\n" + "=" * 76)
    print(report_txt.read_text())
else:
    print("\n\n(no report.txt — export did not run; see the failed stages above)")


# %% [CELL 10] --------------------------------------------------------------
# A/B: 1 GPU vs 2 GPUs. Measures what the multi-device path is actually worth.

import copy

# The main config is already single-device; this builds the DUAL variant so
# the comparison measures what the second card is worth on this machine.
dual = copy.deepcopy(config)
dual["geometry"]["max_devices"] = 0  # every GPU present
DUAL_PATH = "/kaggle/working/kaggle_2gpu.yaml"
Path(DUAL_PATH).write_text(yaml.safe_dump(dual, sort_keys=False))

out_1gpu, t_1gpu = run(32, "32kf_1gpu", config_path=CONFIG_PATH)
out_2gpu, t_2gpu = run(32, "32kf_2gpu", config_path=DUAL_PATH)


def geometry_seconds(out_dir_str):
    meta = json.loads((Path(out_dir_str) / "meta.json").read_text())
    return {s["name"]: s.get("elapsed_s", 0) for s in meta.get("stage_results", [])}.get("geometry", 0.0)


# Print any failed-stage traceback from these two runs before the timings,
# because a "speedup" computed from two failed runs is meaningless.
for _out, _label in ((out_1gpu, "32kf 1 GPU"), (out_2gpu, "32kf 2 GPU")):
    _meta = json.loads((Path(_out) / "meta.json").read_text())
    for _s in _meta.get("stage_results", []):
        if _s.get("status") == "failed":
            print(f"\n{'!' * 76}\nFAILED in {_label}: {_s['name']}\n{'!' * 76}")
            print(_s.get("message", "(no message)"))

g1, g2 = geometry_seconds(out_1gpu), geometry_seconds(out_2gpu)
print("\n" + "=" * 76)
print("MULTI-GPU SPEEDUP (32 keyframes)")
print("=" * 76)
print(f"  geometry stage, 1 GPU   {g1:8.1f} s")
print(f"  geometry stage, 2 GPUs  {g2:8.1f} s")
if g2 > 0:
    print(f"  geometry speedup        {g1 / g2:8.2f}x")
print(f"  end-to-end, 1 GPU       {t_1gpu:8.1f} s")
print(f"  end-to-end, 2 GPUs      {t_2gpu:8.1f} s")
if t_2gpu > 0:
    print(f"  end-to-end speedup      {t_1gpu / t_2gpu:8.2f}x")
print(
    "\nEnd-to-end speedup is necessarily smaller than the geometry speedup:\n"
    "only the geometry stage is parallelised across devices. Amdahl, not a defect."
)

# Sanity: parallelism must not change the result.
print("\nDoes 2-GPU produce the same reconstruction as 1-GPU?")
import laspy

pc_1 = Path(out_1gpu) / "output" / "point_cloud.las"
pc_2 = Path(out_2gpu) / "output" / "point_cloud.las"
if not (pc_1.exists() and pc_2.exists()):
    # A failed geometry stage produces no cloud. Say so plainly instead of
    # raising a FileNotFoundError on top of the real failure above.
    missing = [str(p) for p in (pc_1, pc_2) if not p.exists()]
    print("  cannot compare -- no point cloud was produced by:")
    for m in missing:
        print(f"    {m}")
    print("  Read the failed-stage traceback printed by assess() above.")
else:
    a = laspy.read(str(pc_1))
    b = laspy.read(str(pc_2))
    na, nb = len(a.x), len(b.x)
    print(f"  points: 1 GPU {na:,}   2 GPU {nb:,}   delta {abs(na - nb) / max(na, 1) * 100:.3f} %")
    print("  (small deltas are expected: float reduction order inside the merge can")
    print("   differ with completion order. Large deltas mean a real ordering bug.)")


# %% [CELL 11] --------------------------------------------------------------
# Depth / normal / class maps -- visual evidence the geometry is real.

import matplotlib.pyplot as plt

from drishti3d.export.depthviz import render_keyframe_maps  # noqa: E402
from drishti3d.fusion.mesh import estimate_normals  # noqa: E402
from drishti3d.pipeline.result import PipelineResult  # noqa: E402

result = PipelineResult.load(out16)
viz_dir = Path(out16) / "output" / "maps"

# A failed geometry stage leaves point_cloud as None. Say so plainly and
# stop, rather than raising a second traceback on top of the first.
if result.point_cloud is None or not len(getattr(result.point_cloud, "xyz", [])):
    print("no point cloud in this run -- geometry failed or produced nothing.")
    print("Fix the failed stage above before expecting maps here.")
    written = {}
else:
    normals = None
    try:
        normals = estimate_normals(result.point_cloud, k=30)
    except Exception as exc:
        print(f"normals unavailable: {exc}")

    written = render_keyframe_maps(
        result.point_cloud, result.poses, result.keyframes, viz_dir, max_frames=6, normals=normals
    )
    print(f"wrote {len(written)} map images to {viz_dir}")
if (viz_dir / "depth_ranges.txt").exists():
    print((viz_dir / "depth_ranges.txt").read_text())

import cv2

rows = [
    sorted(viz_dir.glob("depth_*.png"))[:3],
    sorted(viz_dir.glob("normal_*.png"))[:3],
    sorted(viz_dir.glob("class_*.png"))[:3],
]
rows = [r for r in rows if r]
if rows:
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(16, 4.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        for c, path in enumerate(row):
            axes[r, c].imshow(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
            axes[r, c].set_title(path.stem)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.show()

# The texture atlas itself.
atlas_path = Path(out16) / "output" / "model_textured.png"
if atlas_path.exists():
    atlas = cv2.imread(str(atlas_path))
    plt.figure(figsize=(9, 9))
    plt.imshow(cv2.cvtColor(atlas, cv2.COLOR_BGR2RGB))
    plt.title(f"texture atlas {atlas.shape[1]}x{atlas.shape[0]}")
    plt.axis("off")
    plt.show()



# %% [CELL 13] --------------------------------------------------------------
# Scale up, then the full flight.

out64, t64 = run(64, "64kf")
m64 = assess(out64, "64 KEYFRAMES")

# 600 keyframes == the 10-minute-video budget the VRAM benchmark extrapolates
# against. NOTE: --max-keyframes sets a TARGET, not a hard ceiling (it drives
# config.triage.target_keyframes), so an absurd value disables triage's
# adaptive throttle rather than meaning "no cap".
outfull, tfull = run(600, "full")
mfull = assess(outfull, "FULL FLIGHT")


# %% [CELL 14] --------------------------------------------------------------
# The requirement: does a 10-minute video finish inside 15 minutes?

meta = json.loads((Path(outfull) / "meta.json").read_text())
stages = {s["name"]: s.get("elapsed_s", 0) for s in meta.get("stage_results", [])}

nkf = meta.get("report", {}).get("keyframe_count")
if not isinstance(nkf, int) or nkf <= 0:
    nkf = len(meta.get("keyframes", []))
assert nkf > 0, "no keyframes recorded — the run did not get past triage"

# Read the duration from the file rather than hard-coding it: the projection
# below divides by it, so a stale constant silently scales every number here
# the moment FLIGHT changes.
from drishti3d.ingest.video import VideoSource  # noqa: E402

with VideoSource(VIDEO) as _v:
    VIDEO_S = float(_v.duration)
print(f"source video duration: {VIDEO_S:.1f} s")

print("=" * 76)
print("PER-STAGE TIMING (full flight)")
print("=" * 76)
for name, seconds in stages.items():
    print(f"  {name:<20} {seconds:8.1f} s")
total = sum(stages.values())
print(f"  {'TOTAL':<20} {total:8.1f} s  ({total / 60:.1f} min)")
print(f"\n  keyframes selected   {nkf}")

# Ingest+triage scale with video length; the rest scales with keyframe count.
per_kf = (total - stages.get("triage", 0) - stages.get("ingest", 0)) / max(nkf, 1)
fixed_600 = (stages.get("triage", 0) + stages.get("ingest", 0)) * 600 / VIDEO_S
kf_600 = nkf * 600 / VIDEO_S
projected = fixed_600 + per_kf * kf_600

print("\n" + "=" * 76)
print("PROJECTION TO A 10-MINUTE VIDEO")
print("=" * 76)
print(f"  keyframes             {kf_600:.0f}")
print(f"  ingest + triage       {fixed_600 / 60:.1f} min")
print(f"  per-keyframe cost     {per_kf:.1f} s")
print(f"  PROJECTED TOTAL       {projected / 60:.1f} min")
print(f"  REQUIREMENT           15.0 min")
print(f"  VERDICT               {'PASS' if projected < 900 else 'FAIL'}")


# %% [CELL 15] --------------------------------------------------------------
# Consolidated table + JSON, then package for download.

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
        "dtm_method",
        "facade_structures_completed",
        "mesh_faces",
        "mesh_watertight",
        "atlas_nonblack_pct",
        "total_s",
    ]
    df = pd.DataFrame(rows)
    display(df[[c for c in keep if c in df.columns]])  # noqa: F821 -- IPython builtin

Path("/kaggle/working/quality_metrics.json").write_text(json.dumps(rows, indent=2, default=str))
print("wrote /kaggle/working/quality_metrics.json")

!cd /kaggle/working && zip -qr results.zip \
    run_*/output/report.txt run_*/output/report.html run_*/meta.json \
    run_*/output/maps/*.png run_*/output/maps/*.txt \
    run_*/output/model_textured.png \
    vram_t4.json quality_metrics.json \
    -x "*.las" "*.ply" "*.obj" "*.xyz"
!ls -lh /kaggle/working/results.zip

# Optional: push full outputs back to the private dataset repo.
# from huggingface_hub import HfApi
# HfApi().upload_folder(folder_path="/kaggle/working/run_full",
#                       path_in_repo="results/kaggle_full_test",
#                       repo_id=DATA_REPO, repo_type="dataset", token=HF_TOKEN)
