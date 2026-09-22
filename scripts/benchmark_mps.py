"""Measure what MapAnything actually costs on Apple Silicon, and what unthrottles it.

Two things hold the GPU back on MPS, and neither is a hardware limit:

1. ``MapAnythingBackbone.predict`` gates autocast on CUDA
   (``use_amp = _is_cuda(self._device)``), so every MPS run is fp32 --
   twice the memory traffic of half precision, on a device whose ceiling
   is memory bandwidth. torch 2.14 supports both fp16 and bf16 autocast
   on MPS.
2. MapAnything's own ``_compute_adaptive_minibatch_size`` sizes the dense
   prediction head's minibatch from free VRAM via ``torch.cuda.mem_get_info``
   and, finding no CUDA, returns a hardcoded **1** -- one view at a time
   through the memory bottleneck. ``infer(minibatch_size=N)`` overrides it.

This script times one real window under each combination so the choice is
made on measurement rather than on the assumption that "MPS is just slow".
Run it before changing the defaults; the numbers belong in the commit.

Unified memory is shared with the OS, so a too-large minibatch degrades
into swapping rather than failing cleanly. Each configuration is therefore
timed independently and a failure is recorded, not raised.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drishti3d.geometry.mapanything import (  # noqa: E402
    MapAnythingBackbone,
    crop_to_patch_multiple,
    resize_preserving_aspect,
    scale_intrinsics,
)
from drishti3d.types import CameraIntrinsics  # noqa: E402


def load_window(video: Path, meta: Path, n_views: int, size: int):
    """Decode the first ``n_views`` keyframes of a finished run, at ``size`` px."""
    import av

    kf = json.loads(meta.read_text())["keyframes"][:n_views]
    wanted = {k["frame_index"]: i for i, k in enumerate(kf)}
    images: list = [None] * len(kf)
    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    for i, frame in enumerate(container.decode(stream)):
        if i in wanted:
            images[wanted[i]] = frame.to_ndarray(format="bgr24")
            if all(im is not None for im in images):
                break
    container.close()

    intr = kf[0]["intrinsics"]
    base = CameraIntrinsics(
        fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"], width=intr["width"], height=intr["height"]
    )
    out_images, out_intr = [], []
    for img in images:
        resized, scale = resize_preserving_aspect(img, size)
        out_images.append(resized)
        out_intr.append(scale_intrinsics(base, scale))
    return out_images, out_intr


def time_predict(backbone, images, intrinsics, *, amp: str | None, minibatch: int | None, mem_efficient: bool) -> dict:
    """One timed ``predict`` under the given settings. Never raises."""
    import torch

    backbone._bench_amp = amp
    backbone._bench_minibatch = minibatch
    backbone._bench_mem_efficient = mem_efficient
    label = f"amp={amp or 'fp32':<4} minibatch={minibatch or 'auto':<4} mem_efficient={mem_efficient}"
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        t0 = time.perf_counter()
        result = backbone.predict(images, intrinsics=intrinsics)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        depth = np.asarray(result.depth)
        return {
            "label": label,
            "seconds": round(dt, 2),
            "median_depth": round(float(np.median(depth[depth > 0])), 3) if (depth > 0).any() else None,
        }
    except Exception as exc:  # noqa: BLE001 - a config that does not fit is a result, not a crash
        return {"label": label, "seconds": None, "error": f"{type(exc).__name__}: {exc}"[:160]}


def patch_backbone(backbone):
    """Make ``predict`` honour the benchmark's amp / minibatch / mem_efficient knobs.

    Monkeypatched here rather than added as constructor arguments: the
    point of this script is to decide what the defaults should be, and
    that decision should not require the production class to grow three
    fields before the measurement exists to justify them.
    """
    import torch
    from mapanything.utils.inference import preprocess_input_views_for_inference  # noqa: F401

    original = backbone._model.infer

    def patched(views, **kwargs):
        kwargs["use_amp"] = backbone._bench_amp is not None
        if backbone._bench_amp:
            kwargs["amp_dtype"] = backbone._bench_amp
        kwargs["memory_efficient_inference"] = backbone._bench_mem_efficient
        if backbone._bench_minibatch is not None:
            kwargs["minibatch_size"] = backbone._bench_minibatch
        return original(views, **kwargs)

    backbone._model.infer = patched
    # predict() computes use_amp from the device; the patch above overrides
    # whatever it passes, so the CUDA gate no longer decides anything here.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=Path("/Users/ajith/Desktop/sih_data_samples/DJI_0753.MP4"))
    ap.add_argument("--meta", type=Path, default=REPO / "results/local_final/meta.json")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--out", type=Path, default=Path("/tmp/mps_benchmark.json"))
    args = ap.parse_args()

    print(f"loading {args.views} views at {args.size}px ...")
    images, intrinsics = load_window(args.video, args.meta, args.views, args.size)
    print(f"  {images[0].shape[1]}x{images[0].shape[0]} after resize")

    backbone = MapAnythingBackbone(max_image_size=args.size, mask_edges=False)
    if not backbone.is_available():
        print("MapAnything not importable", file=sys.stderr)
        return 2
    print("loading weights onto mps ...")
    backbone.load(device="mps")
    patch_backbone(backbone)

    configs = [
        # (amp, minibatch, memory_efficient) -- baseline first
        (None, 1, True),
        ("fp16", 1, True),
        ("bf16", 1, True),
        ("fp16", 4, True),
        ("fp16", 8, True),
        ("fp16", None, False),
    ]
    results = []
    for amp, mb, mem in configs:
        r = time_predict(backbone, images, intrinsics, amp=amp, minibatch=mb, mem_efficient=mem)
        results.append(r)
        if r["seconds"] is None:
            print(f"  {r['label']}  FAILED  {r.get('error')}")
        else:
            print(f"  {r['label']}  {r['seconds']:>7.2f} s   median_depth={r['median_depth']}")

    ok = [r for r in results if r["seconds"] is not None]
    if ok:
        base = results[0]["seconds"]
        best = min(ok, key=lambda r: r["seconds"])
        print(f"\nbaseline (fp32, minibatch 1): {base} s")
        print(f"best: {best['label']} -> {best['seconds']} s ({base / best['seconds']:.2f}x faster)" if base else "")
        # Depth must not change materially: a faster setting that alters
        # the geometry is not a speedup, it is a different reconstruction.
        depths = [r["median_depth"] for r in ok if r["median_depth"] is not None]
        if len(depths) > 1:
            spread = (max(depths) - min(depths)) / max(abs(np.median(depths)), 1e-6)
            print(f"median-depth spread across configs: {spread * 100:.2f}% (must be ~0 for these to be equivalent)")

    args.out.write_text(json.dumps({"views": args.views, "size": args.size, "results": results}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
