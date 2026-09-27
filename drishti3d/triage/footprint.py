"""Measure each image's along-track ground footprint from the video itself.

Why
---
Keyframe spacing used to come from a stereo rule, ``B / H`` (baseline over
altitude). That rule is about depth precision, not about how much of the
same ground consecutive images share -- and on flight01 (103 m AGL, image
top pointing along the track) its 0.15 put a keyframe every 16 m on a
~90 m footprint: 82% forward overlap, each ground point seen by five or
six keyframes. Every one of those views carries metres of regressed depth
error, so fusing them all stacks slightly different copies of the same
surface. Photogrammetry plans flights by OVERLAP instead; this module
supplies the number that needs.

How
---
The footprint is measured, not modelled, so it needs no altitude, focal
length or gimbal orientation (DJI_1001 has no height-above-ground at all):

1. At ~24 moments of steady flight (turns skipped), take two frames about
   ``target_move_m`` apart along the GPS track.
2. Phase-correlate them (downscaled, Hann-windowed): the ground moves by
   one rigid shift in a nadir view, and phase correlation measures it to
   sub-pixel.
3. Metres per pixel = GPS distance / shift. The image chord along the
   shift direction, times that, is the along-track footprint -- whatever
   way the camera is yawed relative to the track.

The result is a per-time profile (altitude changes over a flight change
the footprint), smoothed with a running median.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["FootprintProfile", "measure_footprint"]

_WORK_WIDTH = 640


@dataclass
class FootprintProfile:
    """Along-track footprint (metres) as a function of video time."""

    times: np.ndarray
    along_m: np.ndarray
    gsd_m_per_px: np.ndarray  # at the source video's native width
    samples_tried: int = 0
    diag: dict = field(default_factory=dict)

    def at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.along_m))

    @property
    def median_along_m(self) -> float:
        return float(np.median(self.along_m))

    def summary(self) -> dict:
        return {
            "samples_used": int(self.times.size),
            "samples_tried": int(self.samples_tried),
            "along_track_footprint_m_median": round(self.median_along_m, 2),
            "along_track_footprint_m_range": [round(float(self.along_m.min()), 2), round(float(self.along_m.max()), 2)],
            "gsd_m_per_px_median": round(float(np.median(self.gsd_m_per_px)), 4),
        }


def _gray_small(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = _WORK_WIDTH / g.shape[1]
    if scale < 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32)


def _speed_and_turn(lookup, t: float, half: float = 0.5):
    from drishti3d.triage.selector import _interp_enu_position, _turn_rate_deg_s

    a, _ = _interp_enu_position(lookup, t - half)
    b, _ = _interp_enu_position(lookup, t + half)
    if a is None or b is None:
        return None, None
    return float(np.linalg.norm((b - a)[:2]) / (2 * half)), _turn_rate_deg_s(lookup, t)


def measure_footprint(
    video,
    lookup,
    *,
    n_samples: int = 24,
    target_move_m: float = 12.0,
    min_speed_mps: float = 3.0,
    max_turn_deg_s: float = 5.0,
    min_response: float = 0.05,
) -> FootprintProfile | None:
    """Sample the video for its along-track footprint. ``None`` when too few samples measure."""
    from drishti3d.triage.selector import _interp_enu_position

    fps = float(getattr(video, "fps", 0.0) or 0.0)
    n_frames = int(getattr(video, "frame_count", 0) or 0)
    if fps <= 0 or n_frames < 10:
        return None
    t_arr = lookup[0]
    t_lo = max(1.0, float(t_arr[0]) + 1.0)
    t_hi = min(n_frames / fps - 4.0, float(t_arr[-1]) - 4.0)
    if t_hi <= t_lo:
        return None

    times, along, gsd = [], [], []
    tried = 0
    for t in np.linspace(t_lo, t_hi, n_samples):
        speed, turn = _speed_and_turn(lookup, t)
        if speed is None or speed < min_speed_mps:
            continue
        dt = float(np.clip(target_move_m / speed, 0.1, 3.0))
        _s2, turn2 = _speed_and_turn(lookup, t + dt)
        if (turn is not None and turn > max_turn_deg_s) or (turn2 is not None and turn2 > max_turn_deg_s):
            continue
        tried += 1
        i, j = int(round(t * fps)), int(round((t + dt) * fps))
        try:
            frames = video.read_frames([i, j])
        except Exception:
            logger.debug("footprint: could not read frames %d/%d", i, j, exc_info=True)
            continue
        if len(frames) != 2 or frames[0].image is None or frames[1].image is None:
            continue
        ga, gb = _gray_small(frames[0].image), _gray_small(frames[1].image)
        win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
        shift = float(np.hypot(dx, dy))
        if resp < min_response or shift < 3.0 or shift > 0.45 * min(ga.shape):
            continue
        pa, _ = _interp_enu_position(lookup, frames[0].timestamp)
        pb, _ = _interp_enu_position(lookup, frames[1].timestamp)
        if pa is None or pb is None:
            continue
        move = float(np.linalg.norm((pb - pa)[:2]))
        h, w = ga.shape
        phi = np.arctan2(abs(dy), abs(dx))
        chord = min(w / max(np.cos(phi), 1e-6), h / max(np.sin(phi), 1e-6))
        m_per_px = move / shift
        times.append(t)
        along.append(m_per_px * chord)
        gsd.append(m_per_px * w / frames[0].image.shape[1])

    if len(times) < 3:
        logger.info("footprint: only %d/%d samples measured; falling back to the altitude rule", len(times), tried)
        return None
    along_arr = np.asarray(along)
    # Running median over 5 samples: one bad correlation must not set a stretch's spacing.
    k = 2
    smooth = np.array([np.median(along_arr[max(0, n - k) : n + k + 1]) for n in range(along_arr.size)])
    prof = FootprintProfile(
        times=np.asarray(times), along_m=smooth, gsd_m_per_px=np.asarray(gsd), samples_tried=tried
    )
    logger.info(
        "footprint: along-track %.1f m median (%.1f-%.1f m) from %d samples; GSD %.3f m/px",
        prof.median_along_m, smooth.min(), smooth.max(), len(times), float(np.median(gsd)),
    )
    return prof
