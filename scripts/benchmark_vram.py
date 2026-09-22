#!/usr/bin/env python3
"""Standalone VRAM/timing benchmark for a DRISHTI-3D-style geometry backbone.

This script is deliberately self-contained: copy this ONE file to the
target machine (the Lenovo LOQ's Windows/WSL install, or a Colab/Kaggle
notebook cell) and run it there. It does not import the ``drishti3d``
package -- only (optionally) ``torch``, ``numpy``, and whichever backbone
package you point ``--backend`` at.

Why this exists
----------------
The target deployment machine has an RTX 4060 laptop GPU with only 6 GB of
VRAM. Every public data point we have for this class of model (stock VGGT
fits ~60 frames on a 24 GB 4090; MASt3R-SLAM/VGGT-SLAM OOM outright on an
8 GB 4060) is for *more* VRAM than we have. 6 GB is below anything in the
published literature, so instead of guessing, this script measures: how
many views actually fit, at what resolution, how fast, and whether a full
video can finish in a reasonable time -- on the *actual* target hardware.

Quick start
-----------
Before downloading any model weights, sanity-check the harness itself
(shape-only synthetic tensors, no model, no download)::

    uv run python scripts/benchmark_vram.py --backend synthetic

Once that looks sane, point it at a real backbone (see scripts/README.md
for full setup instructions, including the CUDA torch install command and
how to point at local/offline weights)::

    uv run python scripts/benchmark_vram.py --backend mapanything --out results.json

Behavior
--------
- Sweeps ``--views`` x ``--sizes`` (both comma-separated, configurable).
- For each cell: runs the backend, measures peak VRAM
  (``torch.cuda.max_memory_allocated`` / ``max_memory_reserved``, reset
  before every cell) and wall-clock seconds.
- Catches OOM (and any other per-cell exception) and keeps going -- a
  benchmark that dies at the first OOM defeats the purpose of sweeping.
- Prints a plain-text results table (no external dependencies) and writes
  the full results as JSON to ``--out``.
- Prints a CONCLUSION section: the largest window size that fits with a
  safety margin, measured seconds/view, and an explicit-arithmetic
  extrapolation of whether a ~600-keyframe (10 minute) video can finish
  within 15 minutes on this machine.
- Falls back to MPS/CPU with a clear warning when no CUDA device is
  present. On MPS (Apple Silicon), it now reports real numbers where it
  can -- ``torch.mps.current_allocated_memory`` /
  ``driver_allocated_memory`` deltas, and total system unified memory as
  the effective ceiling -- but those are unified CPU/GPU memory, not the
  target's dedicated VRAM, and the CONCLUSION section extrapolates from
  them explicitly rather than presenting them as equivalent. On plain CPU
  there is still no memory accounting at all, only timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback as _traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# DINOv2-style ViT patch size, used only to size the synthetic backend's
# "token" tensor plausibly -- not load-bearing for anything real.
_SYNTHETIC_PATCH_SIZE = 14
_SYNTHETIC_FEATURE_DIM = 1024


# ---------------------------------------------------------------------------
# Image generation / loading
# ---------------------------------------------------------------------------


def make_synthetic_image(size: int, seed: int) -> np.ndarray:
    """A deterministic, mildly-textured ``(size, size, 3)`` uint8 RGB image (no files, no deps)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint16)
    gradient_x = np.linspace(0, 255, size, dtype=np.uint16)
    gradient_y = np.linspace(0, 255, size, dtype=np.uint16)
    img[:, :, 0] = (img[:, :, 0] + gradient_x[np.newaxis, :]) // 2
    img[:, :, 1] = (img[:, :, 1] + gradient_y[:, np.newaxis]) // 2
    return img.astype(np.uint8)


def _resize_long_side_nearest(img: np.ndarray, size: int) -> np.ndarray:
    """Dependency-free nearest-neighbour resize so the image's longer side is ``size``, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    row_idx = np.clip((np.arange(new_h) / scale).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(new_w) / scale).astype(int), 0, w - 1)
    return img[row_idx][:, col_idx]


def _load_image_file(path: Path) -> np.ndarray:
    """Load an image file as ``(H, W, 3)`` uint8 RGB, using whichever of opencv/Pillow is available."""
    try:
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"failed to decode image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass

    try:
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))
    except ImportError:
        raise RuntimeError(
            "--images requires either 'opencv-python' or 'Pillow' to be installed to decode "
            "image files (synthetic mode needs neither)."
        ) from None


def build_images(images_dir: str | None, n_views: int, size: int) -> list[np.ndarray]:
    """Build ``n_views`` images at ``size`` px (longer side), either from ``images_dir`` (cycled if too few) or synthetic."""
    if images_dir is None:
        return [make_synthetic_image(size, seed=i) for i in range(n_views)]

    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"--images {images_dir} contains no images with extensions {sorted(_IMAGE_EXTS)}")
    chosen = [paths[i % len(paths)] for i in range(n_views)]
    return [_resize_long_side_nearest(_load_image_file(p), size) for p in chosen]


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def _nvidia_driver_version() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _total_system_memory_gb() -> float | None:
    """Total physical RAM in GB, used on MPS as the effective (unified-memory) ceiling.

    Tries ``psutil`` only if it's already importable (never added as a
    dependency for this), then POSIX ``sysconf``, then ``sysctl`` (macOS).
    """
    try:
        import psutil  # optional, intentionally not a declared dependency

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass

    try:
        proc = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=False
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return int(proc.stdout.strip()) / (1024**3)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    return None


def _mps_current_allocated_gb() -> float | None:
    """Current MPS tensor allocation, GB. No MPS equivalent of a true 'peak' exists."""
    if TORCH_AVAILABLE and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024**3)
    return None


def _mps_driver_allocated_gb() -> float | None:
    """Total memory the MPS driver has claimed from the OS, GB (a proxy for 'reserved')."""
    if TORCH_AVAILABLE and hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        return torch.mps.driver_allocated_memory() / (1024**3)
    return None


def detect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {"torch_available": TORCH_AVAILABLE}

    if not TORCH_AVAILABLE:
        env["device"] = "cpu"
        env["warning"] = (
            "torch is not installed. Running in shape-only validation mode: "
            "no real tensors, no timing/VRAM realism. Install torch (see "
            "scripts/README.md) for a meaningful run."
        )
        return env

    env["torch_version"] = torch.__version__

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env["device"] = "cuda"
        env["gpu_name"] = props.name
        env["vram_total_gb"] = props.total_memory / (1024**3)
        env["cuda_version"] = torch.version.cuda
        env["driver_version"] = _nvidia_driver_version()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        env["device"] = "mps"
        env["is_unified_memory"] = True
        env["total_system_memory_gb"] = _total_system_memory_gb()
        fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        env["mps_fallback_env_value"] = fallback_env
        env["mps_fallback_env_set"] = bool(fallback_env) and fallback_env != "0"

        warning = (
            "No CUDA device found; falling back to MPS. The memory numbers below are "
            "Apple Silicon UNIFIED MEMORY (shared CPU/GPU system RAM), NOT dedicated VRAM "
            "like the target RTX 4060's 6 GB -- they are a different memory system and "
            "must not be used to size anything for the 4060 without re-validation on the "
            "real hardware. torch also has no MPS equivalent of "
            "torch.cuda.max_memory_allocated, so figures here are point-in-time samples "
            "(current_allocated_memory / driver_allocated_memory) around each cell, not "
            "true peaks."
        )
        if env["mps_fallback_env_set"]:
            warning += (
                " ALSO: PYTORCH_ENABLE_MPS_FALLBACK is set in this environment -- "
                "unsupported ops silently fall back to CPU, which can make a cell look "
                "like it ran on the GPU while it partly (or entirely) ran on the CPU. "
                "Unset it for a trustworthy MPS run."
            )
        env["warning"] = warning
    else:
        env["device"] = "cpu"
        env["warning"] = (
            "No CUDA or MPS device found; running on CPU. Peak-VRAM figures are not "
            "meaningful and timings will not reflect real GPU performance."
        )

    return env


def print_environment(env: dict[str, Any]) -> None:
    print("=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    print(f"  device            : {env.get('device')}")
    print(f"  torch available   : {env.get('torch_available')}")
    if env.get("torch_version"):
        print(f"  torch version     : {env['torch_version']}")
    if env.get("device") == "cuda":
        print(f"  GPU               : {env.get('gpu_name', 'unknown')}")
        print(f"  total VRAM        : {env.get('vram_total_gb', float('nan')):.2f} GB")
        print(f"  CUDA version      : {env.get('cuda_version', 'unknown')}")
        print(f"  driver version    : {env.get('driver_version', 'unknown')}")
    if env.get("device") == "mps":
        print(
            f"  total system RAM  : {_fmt(env.get('total_system_memory_gb'))} GB "
            "(UNIFIED memory, shared CPU/GPU -- NOT dedicated VRAM)"
        )
        fallback_note = "  <-- WARNING: silent CPU fallback possible" if env.get("mps_fallback_env_set") else " (unset)"
        print(f"  PYTORCH_ENABLE_MPS_FALLBACK : {env.get('mps_fallback_env_value') or 'not set'}{fallback_note}")
    if env.get("warning"):
        print(f"  WARNING           : {env['warning']}")
    print()


# ---------------------------------------------------------------------------
# Backend model loading
# ---------------------------------------------------------------------------


def load_mapanything_model(checkpoint: str, local_weights: str | None, device: str) -> Any:
    from mapanything.models import MapAnything

    weights_source = checkpoint
    if local_weights:
        local_path = Path(local_weights)
        if not local_path.exists():
            raise RuntimeError(
                f"--local-weights/$DRISHTI3D_MAPANYTHING_WEIGHTS points at '{local_path}', "
                "which does not exist. Predownload the checkpoint there (e.g. via "
                f"`huggingface-cli download {checkpoint} --local-dir {local_path}` on a "
                "machine with network access) before running on an air-gapped box."
            )
        weights_source = str(local_path)
    else:
        print(
            f"WARNING: no local weights directory given; "
            f"MapAnything.from_pretrained({checkpoint!r}) may attempt a network download "
            "from the Hugging Face Hub.",
            file=sys.stderr,
        )

    model = MapAnything.from_pretrained(weights_source).to(device)
    model.eval()
    return model


def load_vggt_model(device: str) -> Any:
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-cell backend runners
# ---------------------------------------------------------------------------


def run_synthetic_cell(images: list[np.ndarray], device: str, dtype: Any) -> Any:
    """Allocate tensors shaped like a real multi-view transformer pass, with no model weights.

    Sized after the same reasoning as ``drishti3d.geometry.windows
    .estimate_memory``: a per-view "token" activation tensor, plus a
    per-(view, view)-pair tensor standing in for cross-view attention.
    This exists purely to validate the benchmark harness's control flow
    (argument parsing, sweeping, OOM handling, table/JSON output) without
    downloading or running any real model.

    Returns the ``features`` tensor (or ``None`` on the numpy-only path)
    so callers can sanity-check which device the computation actually
    landed on.
    """
    n = len(images)
    h, w = images[0].shape[0], images[0].shape[1]

    if TORCH_AVAILABLE and device in ("cuda", "mps"):
        tokens = max(1, h // _SYNTHETIC_PATCH_SIZE) * max(1, w // _SYNTHETIC_PATCH_SIZE)
        imgs = torch.from_numpy(np.stack(images)).to(device=device, dtype=dtype)
        features = torch.randn(n, tokens, _SYNTHETIC_FEATURE_DIM, device=device, dtype=dtype)
        cross_attn = torch.randn(n, n, tokens, device=device, dtype=dtype) if n > 1 else None
        # A little real compute so wall-clock isn't purely allocation noise.
        _ = (features @ features.transpose(-1, -2)).mean()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        del imgs, cross_attn
        return features
    else:
        # No GPU tensor backend: numpy-only, so this still exercises every
        # other part of the harness even with zero ML dependencies.
        stacked = np.stack(images)
        time.sleep(0.005 * n)
        del stacked
        return None


def run_mapanything_cell(model: Any, images: list[np.ndarray], device: str, dtype: Any) -> Any:
    """Run one MapAnything ``infer()`` call.

    ``model.infer()`` requires each view dict to have an ``img`` tensor of
    shape ``(1, 3, H, W)`` *normalized* per the encoder's ``data_norm_type``
    (not raw 0-255 RGB), plus a ``data_norm_type`` key -- read from
    ``model.encoder.data_norm_type`` rather than hard-coded, since a
    different checkpoint's encoder could use a different normalization.
    Omitting either silently produces garbage (wrong img format) or a hard
    ``ValueError: ... missing required keys: {'data_norm_type'}`` (this is
    the bug that made every cell in this benchmark fail before this fix).
    """
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    norm_type = model.encoder.data_norm_type
    image_norm = IMAGE_NORMALIZATION_DICT[norm_type]
    mean = image_norm.mean.view(3, 1, 1).to(device, dtype=torch.float32)
    std = image_norm.std.view(3, 1, 1).to(device, dtype=torch.float32)

    # MapAnything's patch embedding conv hard-asserts the input spatial
    # dims are an exact multiple of the encoder's patch size (confirmed:
    # `AssertionError: Input shape must be divisible by patch size: 14`
    # for e.g. --sizes 256,384, neither of which is a multiple of 14).
    # Center-crop down to the nearest multiple rather than resizing again.
    patch_size = getattr(model.encoder, "patch_size", 14)

    views = []
    for img in images:
        h, w = img.shape[:2]
        aligned_h = max(patch_size, (h // patch_size) * patch_size)
        aligned_w = max(patch_size, (w // patch_size) * patch_size)
        top, left = (h - aligned_h) // 2, (w - aligned_w) // 2
        img = img[top : top + aligned_h, left : left + aligned_w]

        img_chw = torch.from_numpy(np.ascontiguousarray(img)).to(device).permute(2, 0, 1).float() / 255.0
        img_chw = (img_chw - mean) / std
        views.append({"img": img_chw.unsqueeze(0), "data_norm_type": [norm_type]})

    # bf16 autocast on CUDA only; fp32 (no autocast) everywhere else -- see
    # drishti3d/geometry/mapanything.py's module docstring for why MPS
    # doesn't get autocast by default.
    use_amp = device == "cuda"
    with torch.inference_mode():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=use_amp, amp_dtype="bf16")
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    return preds


def run_vggt_cell(model: Any, images: list[np.ndarray], device: str, dtype: Any) -> Any:
    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous().float() / 255.0
    batch = batch.to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = model(batch)
        else:
            output = model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    return output


def run_cell(backend: str, model: Any, images: list[np.ndarray], device: str, dtype: Any) -> Any:
    """Run one sweep cell and return its raw output (used for the MPS device sanity-check)."""
    if backend == "synthetic":
        return run_synthetic_cell(images, device, dtype)
    elif backend == "mapanything":
        return run_mapanything_cell(model, images, device, dtype)
    elif backend == "vggt":
        return run_vggt_cell(model, images, device, dtype)
    else:
        raise ValueError(f"unknown backend {backend!r}")


def _output_device_types(obj: Any, found: set[str] | None = None) -> set[str]:
    """Recursively collect ``.device.type`` of every torch.Tensor found in ``obj`` (tensor/dict/list/tuple)."""
    if found is None:
        found = set()
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        found.add(obj.device.type)
    elif isinstance(obj, dict):
        for v in obj.values():
            _output_device_types(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _output_device_types(v, found)
    return found


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _is_oom(exc: Exception) -> bool:
    if TORCH_AVAILABLE and hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def sweep(args: argparse.Namespace, env: dict[str, Any]) -> list[dict[str, Any]]:
    device = env["device"]
    dtype = None
    model: Any = None

    if TORCH_AVAILABLE:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.backend == "mapanything":
        model = load_mapanything_model(args.checkpoint, args.local_weights, device)
    elif args.backend == "vggt":
        model = load_vggt_model(device)

    results: list[dict[str, Any]] = []
    mps_peak_driver_gb_observed: float | None = None
    device_sanity_checked = False

    for size in args.sizes:
        for n_views in args.views:
            cell: dict[str, Any] = {"backend": args.backend, "size": size, "n_views": n_views}

            if TORCH_AVAILABLE and device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            elif TORCH_AVAILABLE and device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            try:
                images = build_images(args.images, n_views, size)

                # Warmup iteration (discarded): shakes out first-touch allocator
                # / lazy-compile overhead so the timed iteration below reflects
                # steady-state performance, not one-time setup cost.
                run_cell(args.backend, model, images, device, dtype)

                mps_alloc_before = _mps_current_allocated_gb() if device == "mps" else None

                start = time.perf_counter()
                output = run_cell(args.backend, model, images, device, dtype)
                elapsed = time.perf_counter() - start

                cell["status"] = "ok"
                cell["seconds"] = elapsed
                cell["seconds_per_view"] = elapsed / n_views
                if TORCH_AVAILABLE and device == "cuda":
                    cell["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
                    cell["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
                elif TORCH_AVAILABLE and device == "mps":
                    mps_alloc_after = _mps_current_allocated_gb()
                    mps_driver_after = _mps_driver_allocated_gb()
                    # Reused generic field names so the table/JSON schema stays
                    # uniform across devices -- but on MPS these are unified
                    # memory point-in-time samples, NOT dedicated VRAM peaks:
                    # "allocated" = current_allocated_memory() right after the
                    # cell; "reserved" = driver_allocated_memory() (closest MPS
                    # proxy for CUDA's reserved/cached pool).
                    cell["peak_allocated_gb"] = mps_alloc_after
                    cell["peak_reserved_gb"] = mps_driver_after
                    cell["mps_alloc_delta_gb"] = (
                        (mps_alloc_after - mps_alloc_before)
                        if (mps_alloc_after is not None and mps_alloc_before is not None)
                        else None
                    )
                    cell["is_unified_memory"] = True
                    if mps_driver_after is not None:
                        mps_peak_driver_gb_observed = max(mps_peak_driver_gb_observed or 0.0, mps_driver_after)
                else:
                    cell["peak_allocated_gb"] = None
                    cell["peak_reserved_gb"] = None

                # One-time sanity check: did this computation actually run on
                # the requested accelerator, or did PYTORCH_ENABLE_MPS_FALLBACK
                # (or some other silent fallback) route it to CPU instead?
                if device in ("cuda", "mps") and not device_sanity_checked and output is not None:
                    device_sanity_checked = True
                    seen = _output_device_types(output)
                    if seen and device not in seen:
                        msg = (
                            f"output tensor(s) report device(s) {sorted(seen)}, not the "
                            f"requested {device!r}. This usually means an op silently fell "
                            "back to CPU (check PYTORCH_ENABLE_MPS_FALLBACK) -- timings in "
                            "this run are NOT trustworthy as GPU timings."
                        )
                        print(f"  WARNING: {msg}", file=sys.stderr)
                        cell["device_fallback_detected"] = True
                del output

            except NotImplementedError as exc:
                # The typical shape of an unsupported MPS op -- kept distinct
                # from OOM/other errors so the results table shows *why* a
                # configuration failed.
                cell["status"] = "unsupported_op"
                cell["error"] = f"{type(exc).__name__}: {exc}"[:300]
                cell["seconds"] = None
                cell["seconds_per_view"] = None
                cell["peak_allocated_gb"] = None
                cell["peak_reserved_gb"] = None
                # Clear the failing frame BEFORE emptying the cache.
                #
                # Python keeps that frame alive through the exception's
                # traceback, and after a CUDA OOM its locals still hold
                # multi-gigabyte device tensors -- so empty_cache() frees
                # nothing and every LATER cell in the sweep starts with a
                # contaminated GPU. Observed: the 924/2 cell reported 13.97
                # GB already in use before allocating anything, which was
                # leftover from 518/4, making 924 look unusable when it had
                # not actually been measured on a clean device.
                if exc.__traceback__ is not None:
                    _traceback.clear_frames(exc.__traceback__)
                if TORCH_AVAILABLE and device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                elif TORCH_AVAILABLE and device == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

            except Exception as exc:  # noqa: BLE001 - one bad cell must never sink the whole sweep
                cell["status"] = "oom" if _is_oom(exc) else "error"
                cell["error"] = f"{type(exc).__name__}: {exc}"[:300]
                cell["seconds"] = None
                cell["seconds_per_view"] = None
                cell["peak_allocated_gb"] = None
                cell["peak_reserved_gb"] = None
                # Clear the failing frame BEFORE emptying the cache.
                #
                # Python keeps that frame alive through the exception's
                # traceback, and after a CUDA OOM its locals still hold
                # multi-gigabyte device tensors -- so empty_cache() frees
                # nothing and every LATER cell in the sweep starts with a
                # contaminated GPU. Observed: the 924/2 cell reported 13.97
                # GB already in use before allocating anything, which was
                # leftover from 518/4, making 924 look unusable when it had
                # not actually been measured on a clean device.
                if exc.__traceback__ is not None:
                    _traceback.clear_frames(exc.__traceback__)
                if TORCH_AVAILABLE and device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                elif TORCH_AVAILABLE and device == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

            results.append(cell)
            _print_cell_progress(cell)

    if mps_peak_driver_gb_observed is not None:
        env["mps_peak_driver_gb_observed"] = mps_peak_driver_gb_observed

    return results


def _print_cell_progress(cell: dict[str, Any]) -> None:
    status = cell["status"]
    if status == "ok":
        unit_note = " (unified mem)" if cell.get("is_unified_memory") else ""
        delta_note = f"  delta={_fmt(cell['mps_alloc_delta_gb'])}GB" if cell.get("mps_alloc_delta_gb") is not None else ""
        print(
            f"  [ok]    size={cell['size']:>4} views={cell['n_views']:>3}  "
            f"{cell['seconds']:.2f}s ({cell['seconds_per_view']:.3f} s/view)  "
            f"alloc={_fmt(cell['peak_allocated_gb'])}GB reserved={_fmt(cell['peak_reserved_gb'])}GB{unit_note}{delta_note}"
        )
    else:
        print(f"  [{status:>13}] size={cell['size']:>4} views={cell['n_views']:>3}  {cell.get('error', '')}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x: float | None, ndigits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{ndigits}f}"


def print_table(results: list[dict[str, Any]], env: dict[str, Any] | None = None) -> None:
    headers = ["backend", "size", "views", "status", "alloc_gb", "reserved_gb", "seconds", "s/view", "error"]
    rows = []
    for r in results:
        rows.append(
            [
                str(r["backend"]),
                str(r["size"]),
                str(r["n_views"]),
                str(r["status"]),
                _fmt(r.get("peak_allocated_gb")),
                _fmt(r.get("peak_reserved_gb")),
                _fmt(r.get("seconds"), 3),
                _fmt(r.get("seconds_per_view"), 4),
                (r.get("error") or "")[:36],
            ]
        )

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]) for i in range(len(headers))]

    def fmt_row(cols: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    if env is not None and env.get("device") == "mps":
        print(
            "NOTE: alloc_gb/reserved_gb below are Apple Silicon UNIFIED memory samples "
            "(current_allocated_memory / driver_allocated_memory), not dedicated VRAM -- "
            "see ENVIRONMENT and CONCLUSION above/below. status=unsupported_op means an op "
            "hit torch's not-implemented-on-MPS path (distinct from oom)."
        )
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for row in rows:
        print(fmt_row(row))
    print()


def _fit_memory_curve(points: list[tuple[float, float]]) -> tuple[str, tuple[float, float, float]] | None:
    """Fit ``memory = a + b*views + c*views**2`` to ``(views, memory_gb)`` points.

    Quadratic when there are >= 3 distinct view counts (global attention adds
    a roughly quadratic term); linear (``c = 0``) fallback with >= 2; ``None``
    if there's not enough data to fit anything.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(set(xs)) >= 3:
        c, b, a = (float(v) for v in np.polyfit(xs, ys, 2))
        return "quadratic (a + b*views + c*views^2)", (a, b, c)
    if len(set(xs)) >= 2:
        b, a = (float(v) for v in np.polyfit(xs, ys, 1))
        return "linear (a + b*views) -- too few distinct view counts for a quadratic fit", (a, b, 0.0)
    return None


def _predict_max_views(coeffs: tuple[float, float, float], target_gb: float) -> int | None:
    """Largest integer ``views`` with predicted memory <= target_gb, or None if unsolvable."""
    a, b, c = coeffs
    if abs(c) < 1e-12:
        if b <= 0:
            return None
        v = (target_gb - a) / b
    else:
        disc = b * b - 4 * c * (a - target_gb)
        if disc < 0:
            return None
        v = (-b + math.sqrt(disc)) / (2 * c)
    if not math.isfinite(v) or v < 1:
        return None
    return math.floor(v)


def _print_cross_device_extrapolation(results: list[dict[str, Any]], env: dict[str, Any], args: argparse.Namespace) -> None:
    device = env.get("device")
    print(
        f"This run happened on device={device!r}, NOT the target RTX 4060. Everything "
        "below is a PREDICTION extrapolated from a curve fit on a different memory "
        "system (Apple unified memory or plain CPU), not a measurement of the real "
        "deployment target. It MUST be confirmed by an actual run on the 4060 (see "
        "scripts/README.md) before any window size is shipped or relied upon."
    )
    if device == "mps":
        print(
            "Timing note: measured seconds/view above is an Apple GPU (MPS) number and "
            "does NOT transfer to CUDA -- different architecture, different kernels, "
            "different clock/thermal behavior. Use it only to compare configurations "
            "against each other on this Mac, not to predict 4060 wall-clock time."
        )
    print()

    if device == "mps":
        total_mem = env.get("total_system_memory_gb")
        peak_driver = env.get("mps_peak_driver_gb_observed")
        print(
            f"  Effective ceiling this run actually had: {_fmt(total_mem)} GB of total "
            "unified system RAM (shared with the OS and everything else on the Mac), "
            f"not the 6 GB of dedicated VRAM on the target 4060. Peak MPS driver "
            f"allocation observed during the sweep: {_fmt(peak_driver)} GB."
        )
        print()

    memory_key = "peak_allocated_gb" if device == "mps" else None
    if memory_key is None:
        print(
            f"No memory measurements exist for device={device!r} (CPU has no memory "
            "accounting in this script, only timing), so no memory-based window-size "
            "prediction -- extrapolated or otherwise -- can be made. Only timing data "
            "was collected; see the RESULTS table above."
        )
        print()
        return

    print(f"Target for extrapolation: --target-vram-gb = {args.target_vram_gb:.2f} GB (the 4060's dedicated VRAM)")
    print()

    any_prediction = False
    for size in sorted({r["size"] for r in results}):
        points = [
            (float(r["n_views"]), r[memory_key])
            for r in results
            if r["size"] == size and r["status"] == "ok" and r.get(memory_key) is not None
        ]
        fit = _fit_memory_curve(points)
        if fit is None:
            print(f"  size={size:>4}px: not enough successful cells with memory data to fit a curve (have {len(points)}).")
            continue
        kind, coeffs = fit
        a, b, c = coeffs
        predicted = _predict_max_views(coeffs, args.target_vram_gb)
        print(f"  size={size:>4}px: fit {kind}")
        print(f"    a={a:.4f}  b={b:.6f}  c={c:.8f}  (from {len(points)} point(s))")
        if predicted is None:
            print(
                f"    -> could not solve for a window size within {args.target_vram_gb:.2f} GB "
                "(fit never stays at/under the target, or is non-increasing)."
            )
        else:
            any_prediction = True
            print(
                f"    -> PREDICTED largest window at {args.target_vram_gb:.2f} GB "
                f"(unified-memory-derived, UNCONFIRMED on real VRAM): {predicted} views"
            )

    print()
    if any_prediction:
        print(
            "REMINDER: a window size validated against unified memory on this Mac can "
            "still OOM on the 4060's 6 GB of dedicated VRAM -- unified memory is shared "
            "with the OS/everything else and behaves differently from a fixed VRAM pool. "
            "Do not size anything for the 4060 from this run alone."
        )
        print()


def print_conclusion(results: list[dict[str, Any]], env: dict[str, Any], args: argparse.Namespace) -> None:
    print("=" * 78)
    print("CONCLUSION")
    print("=" * 78)

    vram_total = env.get("vram_total_gb")
    if vram_total is None:
        _print_cross_device_extrapolation(results, env, args)
        return

    budget_gb = vram_total * args.safety_margin
    print(f"Detected GPU: {env.get('gpu_name', 'unknown')} with {vram_total:.2f} GB total VRAM")
    print(f"Safety margin: {args.safety_margin:.0%} -> usable budget = {budget_gb:.2f} GB")
    print()

    best_per_size: dict[int, dict[str, Any]] = {}
    for size in sorted({r["size"] for r in results}):
        fitting = [
            r
            for r in results
            if r["size"] == size and r["status"] == "ok" and r["peak_reserved_gb"] is not None and r["peak_reserved_gb"] <= budget_gb
        ]
        if fitting:
            best = max(fitting, key=lambda r: r["n_views"])
            best_per_size[size] = best
            print(
                f"  size={size:>4}px: largest window that fits = {best['n_views']} views "
                f"({best['peak_reserved_gb']:.2f} GB reserved, {best['seconds_per_view']:.3f} s/view)"
            )
        else:
            print(f"  size={size:>4}px: no tested view count fit within the {budget_gb:.2f} GB budget")

    print()

    if not best_per_size:
        print("No configuration fit within the safety-margined budget -- try smaller --sizes/--views.")
        print()
        return

    headline_size = max(best_per_size)
    headline = best_per_size[headline_size]
    window_size = headline["n_views"]
    seconds_per_view = headline["seconds_per_view"]

    print(
        f"Headline: at {headline_size}px, window_size={window_size} fits "
        f"({headline['peak_reserved_gb']:.2f} GB reserved <= {budget_gb:.2f} GB budget), "
        f"measured {seconds_per_view:.3f} s/view."
    )
    print()

    n_keyframes = args.extrapolate_keyframes
    overlap = max(2, round(0.3 * window_size))
    step = max(1, window_size - overlap)
    n_windows = math.ceil(max(0, n_keyframes - window_size) / step) + 1
    total_view_passes = n_windows * window_size
    total_seconds = total_view_passes * seconds_per_view
    budget_seconds = args.extrapolate_minutes * 60
    verdict = "PASS" if total_seconds <= budget_seconds else "FAIL"

    print(
        f"Extrapolation: can a {n_keyframes}-keyframe (~10 min flight) video finish "
        f"within {args.extrapolate_minutes:.0f} minutes on this machine?"
    )
    print(f"  window_size (from above)         = {window_size} views")
    print(f"  overlap (30% default, floor 2)   = {overlap} views")
    print(f"  step = window_size - overlap     = {window_size} - {overlap} = {step} views")
    print(f"  n_windows = ceil(({n_keyframes} - {window_size}) / {step}) + 1 = {n_windows}")
    print(f"  total view-passes = n_windows * window_size = {n_windows} * {window_size} = {total_view_passes}")
    print(f"  measured seconds/view             = {seconds_per_view:.4f} s")
    print(
        f"  total_seconds = total view-passes * seconds/view = "
        f"{total_view_passes} * {seconds_per_view:.4f} = {total_seconds:.1f} s"
    )
    print(f"  budget = {args.extrapolate_minutes:.0f} min = {budget_seconds:.0f} s")
    comparison = "<=" if verdict == "PASS" else ">"
    print(f"  => {verdict}: {total_seconds:.1f}s {comparison} {budget_seconds:.0f}s")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone VRAM/timing sweep for a multi-view geometry backbone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--views", type=_parse_int_list, default=[2, 4, 8, 12, 16, 24, 32], help="Comma-separated view counts to sweep.")
    parser.add_argument("--sizes", type=_parse_int_list, default=[256, 384, 518], help="Comma-separated image long-side sizes (px) to sweep.")
    parser.add_argument("--backend", choices=["mapanything", "vggt", "synthetic"], default="synthetic")
    parser.add_argument("--images", type=str, default=None, help="Directory of real images to benchmark on (cycled if fewer than the largest --views). Default: generate synthetic textured images.")
    parser.add_argument("--out", type=str, default="vram_benchmark_results.json", help="Path to write full JSON results to.")
    parser.add_argument("--checkpoint", type=str, default="facebook/map-anything-apache", help="MapAnything checkpoint id (--backend mapanything only).")
    parser.add_argument(
        "--local-weights",
        type=str,
        default=os.environ.get("DRISHTI3D_MAPANYTHING_WEIGHTS"),
        help="Local directory of predownloaded MapAnything weights, for offline/air-gapped runs. Defaults to $DRISHTI3D_MAPANYTHING_WEIGHTS.",
    )
    parser.add_argument("--safety-margin", type=float, default=0.9, help="Fraction of total VRAM to budget for in the conclusion (headroom for fragmentation/other processes).")
    parser.add_argument("--extrapolate-keyframes", type=int, default=600, help="Assumed keyframe count for the 10-minute-video extrapolation.")
    parser.add_argument("--extrapolate-minutes", type=float, default=15.0, help="Wall-clock budget (minutes) for the extrapolation's pass/fail verdict.")
    parser.add_argument(
        "--target-vram-gb",
        type=float,
        default=6.0,
        help="Dedicated VRAM (GB) of the deployment target, used only for the cross-device memory-curve "
        "extrapolation printed when this run happened on MPS or CPU (i.e. not on CUDA).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = detect_environment()
    print_environment(env)

    print("=" * 78)
    print(f"SWEEP  backend={args.backend}  views={args.views}  sizes={args.sizes}")
    print("=" * 78)

    try:
        results = sweep(args, env)
    except Exception as exc:  # noqa: BLE001 - model load failures should be actionable, not a raw traceback-only exit
        print(f"\nFATAL: could not run the '{args.backend}' backend: {exc}", file=sys.stderr)
        print("Try `--backend synthetic` first to confirm the harness itself works.", file=sys.stderr)
        return 1

    print()
    print_table(results, env)
    print_conclusion(results, env, args)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"environment": env, "args": vars(args), "results": results}, indent=2, default=str))
    print(f"Wrote {len(results)} cell results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
