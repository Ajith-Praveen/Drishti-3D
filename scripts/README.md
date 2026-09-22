# `benchmark_vram.py`

Standalone VRAM/timing sweep for a multi-view geometry backbone
(MapAnything / VGGT / a shape-only synthetic stand-in). It is **one file**
with no dependency on the `drishti3d` package — copy just
`scripts/benchmark_vram.py` to wherever you want to measure, and run it
there.

Why this matters for this project: the target deployment machine (a Lenovo
LOQ, RTX 4060 laptop GPU) has only **6 GB of VRAM**, which is below every
published data point we have for this class of model. This script exists
to answer, on the real hardware, three questions the literature can't
answer for us:

1. How many views (frames per reconstruction window) actually fit?
2. How fast is each view, once it fits?
3. Given that, can a full ~10-minute flight (~600 keyframes) finish within
   a 15-minute budget?

## 0. Sanity-check the harness first (no downloads, no GPU needed)

Before installing anything heavy, confirm the script itself runs and
produces a sane table/JSON, using shape-only synthetic tensors:

```bash
uv run python scripts/benchmark_vram.py --backend synthetic --out synthetic_smoke.json
```

This works even with no `torch` installed at all (it degrades to a
numpy-only control-flow check with a clear "not meaningful" warning) and
with no CUDA (it falls back to CPU/MPS, again with a clear warning that
VRAM numbers aren't meaningful there). Once this looks right, move to a
real backend.

## 1. On the Lenovo LOQ (RTX 4060, 6 GB) — the run that actually matters

### Windows (native, PowerShell)

```powershell
# Fresh venv (any Python 3.10-3.12 works; this script has no drishti3d dependency)
py -m venv .venv-bench
.venv-bench\Scripts\activate

# CUDA-enabled torch. Check your installed driver's CUDA version first
# (`nvidia-smi`, top-right corner) and match the cuXXX tag below to it;
# cu121 works with driver-reported CUDA 12.1+ (which covers essentially
# all recent driver releases on a 4060).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For --backend mapanything: clone and install the real package (do this
# on a machine with network access if the target box will run air-gapped
# later — see "Offline / air-gapped weights" below).
git clone https://github.com/facebookresearch/map-anything.git
pip install -e map-anything

python scripts/benchmark_vram.py --backend mapanything --out results_4060.json
```

### WSL (Ubuntu on the same laptop)

WSL2 passes the GPU through, but you need the WSL-specific CUDA path (not
a native Linux driver install — the Windows NVIDIA driver already provides
the WSL CUDA support):

```bash
python3 -m venv .venv-bench
source .venv-bench/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/facebookresearch/map-anything.git
pip install -e map-anything

python scripts/benchmark_vram.py --backend mapanything --out results_4060_wsl.json
```

If `torch.cuda.is_available()` is `False` inside WSL, the WSL CUDA
userspace libraries aren't wired up — update the Windows NVIDIA driver
(not a driver *inside* WSL; WSL has none of its own) and confirm
`nvidia-smi` runs inside the WSL shell before retrying.

### Recommended first real sweep on the 4060

Start narrow — a wide sweep at high resolution can sit in one giant OOM
retry loop if you're not careful. This script *does* catch OOM and keep
going, but starting small still gets you a usable partial result fastest:

```bash
python scripts/benchmark_vram.py \
  --backend mapanything \
  --views 2,4,6,8,12,16,24 \
  --sizes 256,384,518 \
  --out results_4060.json
```

Read the `CONCLUSION` section at the end of stdout (and in the JSON under
no separate key — it's derived from `results` + `environment` on every
run) for the largest window size that fits with a safety margin, and the
10-minute-video extrapolation with the arithmetic spelled out.

## 1.5. On a Mac (Apple Silicon / MPS) — dev-machine runs, NOT the target

If your dev machine is a MacBook (M-series), you don't have CUDA, but the
script now gets real numbers out of MPS instead of timing-only:

```bash
uv sync --extra ml   # installs torch/torchvision for this venv
uv run python scripts/benchmark_vram.py --backend synthetic --views 2,4,8,16,32,64 --sizes 256,384,518 --out results_mps.json
```

(Swap `--backend synthetic` for `--backend mapanything` or `--backend
vggt` once you want real-model numbers instead of the shape-only
harness check.)

**What this run *does* tell you:**

- The harness's control flow works end to end (sweeping, OOM/error/
  unsupported-op handling, table/JSON output) — a legitimate reason to
  do this before shipping the script to the 4060.
- Real memory deltas from `torch.mps.current_allocated_memory()` /
  `driver_allocated_memory()`, sampled before/after every cell (`empty_cache()`
  between cells, mirroring the CUDA path) — these track how memory scales
  with `--views` and image size, which is directionally useful.
- Real per-view timings on Apple's GPU (with a warmup iteration discarded
  per cell so cold-start cost doesn't pollute the measurement).
- Whether `PYTORCH_ENABLE_MPS_FALLBACK` is set (it silently pushes
  unsupported ops to CPU, which invalidates timings) and whether any op
  actually landed on CPU instead of MPS when it shouldn't have — both are
  checked and loudly flagged if triggered.
- A CONCLUSION section that fits a memory-vs-`views` curve (quadratic,
  with a linear fallback when there aren't enough distinct view counts)
  and predicts the largest window size that would fit in
  `--target-vram-gb` (default `6.0`, matching the 4060).

**What it does NOT tell you, no matter how clean the numbers look:**

- **The MPS memory figures are unified memory (shared CPU/GPU system
  RAM), not dedicated VRAM.** They are a different memory system from
  the 4060's 6 GB, sampled with different tooling (there's no MPS
  equivalent of `torch.cuda.max_memory_allocated` — these are
  point-in-time samples, not true peaks) on a machine with (for the
  MacBook Air M5 this was written for) **16 GB of total unified RAM** —
  nearly 3x the 4060's dedicated VRAM, shared with the OS and everything
  else running at the time.
- **A window size that comfortably fits in 16 GB of unified memory on
  this Mac WILL OOM on the 4060's 6 GB of dedicated VRAM.** The
  `--target-vram-gb` extrapolation in CONCLUSION is a curve-fit
  *prediction*, explicitly labeled as such, not a measurement — it must
  be confirmed by an actual run on the real 4060 (§1 above) before any
  window size is sized against it or shipped.
- MPS seconds/view does **not** transfer to CUDA — different
  architecture, kernels, and thermal/clock behavior. Only use MPS timing
  to compare configurations against each other on the same Mac.

## 2. On Colab / Kaggle (T4, 16 GB) — for heavier runs / cross-checking

Colab:

```python
!pip install torch torchvision  # Colab images ship a CUDA-matched torch already in most runtimes; only do this if it's missing/mismatched
!git clone https://github.com/facebookresearch/map-anything.git
!pip install -e map-anything

!python scripts/benchmark_vram.py --backend mapanything --views 4,8,16,24,32,48 --sizes 384,518 --out results_t4.json
```

(Upload just `benchmark_vram.py` via the Colab file browser, or `wget`
it from wherever you're hosting this repo — it needs nothing else from
this project.)

Kaggle notebooks: same commands, run in a code cell with a GPU accelerator
selected (Settings → Accelerator → GPU T4 x2 or P100). Kaggle's default
image usually already has a CUDA-matched torch; skip the `pip install
torch` line unless the import fails.

## 3. Offline / air-gapped weights (MapAnything)

The shipped desktop app is meant to run air-gapped. A
`MapAnything.from_pretrained("facebook/map-anything-apache")` call by
default reaches out to the Hugging Face Hub — fine for this benchmark
script on a machine with network access, but the *pipeline's* real adapter
(`drishti3d.geometry.mapanything.MapAnythingBackbone`) will refuse to
silently do that in production. To predownload weights once and reuse them
offline everywhere (including with this benchmark script's
`--local-weights` flag):

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download facebook/map-anything-apache --local-dir ./mapanything-weights
```

Then either:

```bash
python scripts/benchmark_vram.py --backend mapanything --local-weights ./mapanything-weights
```

or set the environment variable the real pipeline adapter also reads
(`--local-weights` defaults to it if not passed explicitly):

```bash
export DRISHTI3D_MAPANYTHING_WEIGHTS=/path/to/mapanything-weights   # Linux/macOS/WSL
setx DRISHTI3D_MAPANYTHING_WEIGHTS "C:\path\to\mapanything-weights"  # Windows, new shells
```

## 4. Reading the results

- **stdout**: an environment block (GPU/VRAM + CUDA/driver versions on
  CUDA; total unified RAM + `PYTORCH_ENABLE_MPS_FALLBACK` status on MPS),
  a results table (one row per `views` x `sizes` cell — `status` is
  `ok`, `oom`, `unsupported_op` (the op isn't implemented on this
  device — MPS mainly), or `error`; none of these ever abort the sweep),
  and the CONCLUSION section (a real measured-fit conclusion on CUDA; an
  explicitly-labeled cross-device *extrapolation* on MPS/CPU — see §1.5).
- **`--out results.json`**: the same data as structured JSON
  (`environment`, `args`, `results`) for feeding into a spreadsheet, a
  plot, or `drishti3d/geometry/windows.py`'s `_EST_*` calibration
  constants (see that module's "CALIBRATION NEEDED" comment block — this
  benchmark's output is exactly what should replace those placeholder
  numbers).

## 5. Flags

| Flag | Default | Meaning |
|---|---|---|
| `--views` | `2,4,8,12,16,24,32` | Comma-separated view counts to sweep. |
| `--sizes` | `256,384,518` | Comma-separated image long-side sizes (px). |
| `--backend` | `synthetic` | `mapanything`, `vggt`, or `synthetic`. |
| `--images DIR` | none | Real drone frames to benchmark on (cycled if fewer than the largest `--views`); else synthetic textured images are generated. |
| `--checkpoint` | `facebook/map-anything-apache` | MapAnything checkpoint id (Apache-2.0; pass `facebook/map-anything` for the stronger but CC-BY-NC checkpoint). |
| `--local-weights DIR` | `$DRISHTI3D_MAPANYTHING_WEIGHTS` | Local predownloaded weights directory (offline/air-gapped runs). |
| `--safety-margin` | `0.9` | Fraction of total VRAM budgeted for in the conclusion. |
| `--extrapolate-keyframes` | `600` | Assumed keyframe count for the "10-minute video" extrapolation. |
| `--extrapolate-minutes` | `15` | Wall-clock budget (minutes) for that extrapolation's pass/fail verdict. |
| `--target-vram-gb` | `6.0` | Dedicated-VRAM target (GB) for the cross-device memory-curve extrapolation printed when the run happened on MPS or CPU (not CUDA) — see §1.5. |
| `--out` | `vram_benchmark_results.json` | Where to write the full JSON results. |
