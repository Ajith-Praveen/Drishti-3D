"""Compute-device selection helpers.

``torch`` is an optional dependency (installed via the ``ml`` extra), so it
is imported lazily inside functions here. This module must remain fully
importable and usable (returning "cpu") even when torch is not installed.
"""

from __future__ import annotations


def get_device(prefer: str | None = None) -> str:
    """Return the best available torch device string: "cuda", "mps", or "cpu".

    Parameters
    ----------
    prefer:
        Optional device family the caller would like ("cuda", "mps", "cpu").
        If given and available, it is returned as-is. If given but not
        available, falls back to auto-detection. If None, auto-detects.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if prefer == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if prefer == "cpu":
            return "cpu"
        # requested device unavailable -> fall through to auto-detection

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_devices(max_devices: int = 0, prefer: str | None = None) -> list[str]:
    """Every usable compute device, best first: ``["cuda:0", "cuda:1", ...]``.

    This is what lets one keyframe window run per GPU concurrently (see
    ``pipeline.stages.GeometryStage``). The list is explicit per-index
    (``"cuda:0"``, not bare ``"cuda"``) because a worker that says only
    ``"cuda"`` lands on whatever the ambient current device happens to be,
    which in a thread pool is a race, not a choice.

    Only CUDA multiplies: MPS is a single logical device no matter how many
    GPU cores the Mac has, and multiple CPU "devices" would contend for the
    same cores rather than adding throughput. Both therefore return exactly
    one entry, which is why the single-GPU and CPU paths stay byte-identical
    to how they behaved before this function existed.

    ``max_devices`` caps the count (``0`` means "use everything"). Useful
    when another process already holds VRAM on the higher indices, or to
    reproduce a single-device run on a multi-GPU box for comparison.
    """
    try:
        import torch
    except ImportError:
        return ["cpu"]

    base = get_device(prefer)
    if base != "cuda":
        return [base]

    count = torch.cuda.device_count()
    if count <= 1:
        return ["cuda:0"] if count == 1 else ["cpu"]

    devices = [f"cuda:{i}" for i in range(count)]
    if max_devices > 0:
        devices = devices[:max_devices]
    return devices


def device_report() -> dict:
    """Return a dict describing the selected device and, for CUDA, VRAM stats.

    Keys: "device", "name", and (CUDA only) "vram_total_gb", "vram_free_gb".
    """
    report: dict = {"device": get_device()}

    try:
        import torch
    except ImportError:
        report["name"] = "cpu"
        return report

    device = report["device"]

    if device == "cuda":
        idx = torch.cuda.current_device()
        report["name"] = torch.cuda.get_device_name(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        report["vram_total_gb"] = total_bytes / (1024**3)
        report["vram_free_gb"] = free_bytes / (1024**3)

        # Per-device detail. Reported as a list rather than a sum: two 16 GB
        # cards are not one 32 GB card, and a report that adds them up
        # invites planning a model that cannot fit on either.
        devices = available_devices()
        report["devices"] = devices
        report["device_count"] = len(devices)
        report["device_vram_gb"] = [
            torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(len(devices))
        ]
    elif device == "mps":
        report["name"] = "Apple MPS"
    else:
        report["name"] = "cpu"

    return report
