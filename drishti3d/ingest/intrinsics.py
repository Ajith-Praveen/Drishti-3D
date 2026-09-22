"""Camera intrinsics priors for the ingest stage.

Everything returned here is a *prior*, not a calibration. We never have a
checkerboard for these clips, so fx/fy/cx/cy are always a best-effort guess
from whatever is cheaply available (embedded metadata, a small built-in
camera database, or a generic drone HFOV) -- good enough to seed
structure-from-motion, which is expected to *refine* these values (and
possibly recover distortion) via self-calibration once enough keyframes
have been reconstructed. Do not treat the values returned here as ground
truth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
