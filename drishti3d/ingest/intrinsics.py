"""Camera intrinsics for the ingest stage.

``intrinsics_from_config`` returns an operator-supplied calibration (the
problem statement's optional camera-intrinsics input) with provenance
``"user"``. Without one, everything returned here is a *prior*, not a
calibration: fx/fy/cx/cy are a best-effort guess from whatever is cheaply
available (embedded metadata, a small built-in camera database, or a
generic drone HFOV) -- good enough to seed structure-from-motion, which is
expected to *refine* these values (and possibly recover distortion) via
self-calibration once enough keyframes have been reconstructed. Do not
treat those guesses as ground truth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from drishti3d.types import CameraIntrinsics

if TYPE_CHECKING:
    from drishti3d.ingest.video import VideoSource

logger = logging.getLogger(__name__)

# sensor_width_mm + a typical (non-calibrated) focal length, for common
# consumer drones. These are nominal manufacturer specs, themselves a prior.
_CAMERA_DB: dict[str, dict[str, float]] = {
    "dji mini 3 pro": {"sensor_width_mm": 9.6, "focal_len_mm": 6.7},
    "dji mini 4 pro": {"sensor_width_mm": 9.6, "focal_len_mm": 6.7},
    "dji mavic 3": {"sensor_width_mm": 17.3, "focal_len_mm": 12.29},
    "dji air 2s": {"sensor_width_mm": 13.2, "focal_len_mm": 8.38},
    "dji phantom 4 pro": {"sensor_width_mm": 13.2, "focal_len_mm": 8.8},
    "autel evo ii": {"sensor_width_mm": 13.2, "focal_len_mm": 8.0},
}

_DEFAULT_HFOV_DEG = 84.0  # typical wide-ish drone camera horizontal FOV


def _square_intrinsics(fx: float, width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def _focal_len_mm_from_telemetry(telemetry: Any) -> float | None:
    """Pull a representative SRT-embedded focal length (mm) out of parsed telemetry, if any.

    ``TelemetrySample`` has no focal-length field (it's a per-clip camera
    setting, not a flight measurement), so ``load_telemetry`` surfaces it
    via its ``stats`` dict instead. We accept either a bare sample list (no
    focal length available, returns None) or the ``(samples, stats)`` tuple
    ``load_telemetry`` returns directly, so callers can pass through
    whichever they already have without re-deriving anything.
    """
    if telemetry is None:
        return None
    if isinstance(telemetry, tuple) and len(telemetry) == 2 and isinstance(telemetry[1], dict):
        value = telemetry[1].get("focal_len_mm")
        return float(value) if value is not None else None
    return None


def _focal_length_35mm_from_metadata(metadata: dict[str, str]) -> float | None:
    """Look for a 35mm-equivalent focal length in container/stream metadata.

    Video containers rarely carry per-frame EXIF the way still photos do,
    but some encoders copy a handful of EXIF-style tags into container
    metadata. A 35mm-equivalent value is self-contained (it implies a
    fixed 36mm reference sensor width), so it's usable without a
    device-specific sensor width -- unlike a raw ``focal_length`` tag.
    """
    for key, value in metadata.items():
        key_l = key.lower()
        if "focal" in key_l and "35" in key_l:
            try:
                return float(str(value).lower().replace("mm", "").strip())
            except ValueError:
                continue
    return None


def _autodetect_camera_model(metadata: dict[str, str]) -> str | None:
    blob = " ".join(str(v) for v in metadata.values()).lower()
    for name in _CAMERA_DB:
        if name in blob:
            return name
    return None


def intrinsics_from_video(
    video: VideoSource, telemetry: Any = None, camera_model: str | None = None
) -> tuple[CameraIntrinsics, str]:
    """Best-effort camera intrinsics for ``video``, plus a provenance string.

    Provenance, in the order they're tried (most-trusted first):

    - ``"exif"``: a 35mm-equivalent focal length found in container/stream
      metadata. Self-contained and camera-agnostic.
    - ``"srt_focal_len"``: focal length measured in the DJI SRT telemetry
      for *this specific clip*, combined with a matched camera database
      sensor width. Preferred over the DB's default focal length because
      it reflects the actual zoom/lens state during the shot.
    - ``"camera_db"``: a recognized camera model's typical sensor width +
      focal length. ``camera_model`` may be passed explicitly, or is
      auto-detected from the video's container/stream metadata (e.g. a
      device-model tag) when omitted.
    - ``"default_guess"``: ``CameraIntrinsics.from_hfov(84deg, ...)``, used
      when nothing more specific is available.

    ``telemetry`` accepts either a ``list[TelemetrySample]`` or the
    ``(samples, stats)`` tuple returned by ``ingest.telemetry.load_telemetry``
    (see ``_focal_len_mm_from_telemetry`` for why).
    """
    width, height = video.width, video.height
    metadata = getattr(video, "metadata", None) or {}

    f35 = _focal_length_35mm_from_metadata(metadata)
    if f35 is not None and f35 > 0:
        fx = width * f35 / 36.0
        return _square_intrinsics(fx, width, height), "exif"

    model = camera_model or _autodetect_camera_model(metadata)
    db_entry = _CAMERA_DB.get(model.strip().lower()) if model else None

    focal_mm = _focal_len_mm_from_telemetry(telemetry)
    if focal_mm is not None and focal_mm > 0 and db_entry is not None:
        fx = focal_mm / db_entry["sensor_width_mm"] * width
        return _square_intrinsics(fx, width, height), "srt_focal_len"

    if db_entry is not None:
        fx = db_entry["focal_len_mm"] / db_entry["sensor_width_mm"] * width
        return _square_intrinsics(fx, width, height), "camera_db"

    if focal_mm is not None:
        logger.debug(
            "SRT focal_len present (%.2f mm) but no matching camera_db sensor width; "
            "falling back to default HFOV guess",
            focal_mm,
        )

    return CameraIntrinsics.from_hfov(_DEFAULT_HFOV_DEG, width, height), "default_guess"


_ACCEPTED_DIST_LENGTHS = (4, 5, 8, 12, 14)  # the coefficient counts OpenCV's distortion models accept


def intrinsics_from_config(ingest_cfg: Any, width: int, height: int) -> tuple[CameraIntrinsics, str] | None:
    """Operator-supplied calibration from ``IngestConfig``, or ``None`` when none was given.

    ``camera_fx`` wins over ``camera_hfov_deg``; with neither set this
    returns ``None`` and the caller falls back to ``intrinsics_from_video``.
    Provenance is always ``"user"``: a calibration the operator vouches for,
    which focal-from-flow and bundle-adjustment focal refinement leave alone
    (they only replace ``"default_guess"``).

    Raises ``ValueError`` for values no real camera can have, so a typo
    fails at ingest instead of surfacing later as a warped model.
    """
    fx = getattr(ingest_cfg, "camera_fx", None)
    hfov = getattr(ingest_cfg, "camera_hfov_deg", None)
    if fx is None and hfov is None:
        if getattr(ingest_cfg, "camera_dist_coeffs", None) is not None:
            raise ValueError(
                "camera_dist_coeffs needs camera_fx or camera_hfov_deg: distortion coefficients are "
                "defined relative to a focal length, so they cannot be paired with a guessed one"
            )
        return None

    if fx is not None:
        fx = float(fx)
        fy = float(getattr(ingest_cfg, "camera_fy", None) or fx)
        cx = getattr(ingest_cfg, "camera_cx", None)
        cy = getattr(ingest_cfg, "camera_cy", None)
        calib_width = getattr(ingest_cfg, "camera_calibration_width", None)
        if calib_width is not None and calib_width <= 0:
            raise ValueError(f"camera_calibration_width must be positive, got {calib_width}")
        factor = width / float(calib_width) if calib_width else 1.0
        fx, fy = fx * factor, fy * factor
        cx = width / 2.0 if cx is None else float(cx) * factor
        cy = height / 2.0 if cy is None else float(cy) * factor
    else:
        hfov = float(hfov)
        if not 1.0 < hfov < 179.0:
            raise ValueError(f"camera_hfov_deg must be between 1 and 179 degrees, got {hfov}")
        base = CameraIntrinsics.from_hfov(hfov, width, height)
        fx, fy, cx, cy = base.fx, base.fy, base.cx, base.cy

    if fx <= 0 or fy <= 0:
        raise ValueError(f"camera focal lengths must be positive, got fx={fx}, fy={fy}")
    if not (0.0 <= cx <= width and 0.0 <= cy <= height):
        raise ValueError(f"principal point ({cx}, {cy}) lies outside the {width}x{height} image")

    dist = getattr(ingest_cfg, "camera_dist_coeffs", None)
    dist_coeffs = None
    if dist is not None:
        dist_coeffs = np.asarray(dist, dtype=np.float64).reshape(-1)
        if dist_coeffs.size not in _ACCEPTED_DIST_LENGTHS:
            raise ValueError(
                f"camera_dist_coeffs needs {', '.join(map(str, _ACCEPTED_DIST_LENGTHS))} values "
                f"(OpenCV order k1, k2, p1, p2[, k3...]), got {dist_coeffs.size}"
            )
        if not np.all(np.isfinite(dist_coeffs)):
            raise ValueError("camera_dist_coeffs must all be finite")

    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, dist_coeffs=dist_coeffs
    )
    return intrinsics, "user"
