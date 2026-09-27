"""Measure the video <-> telemetry clock offset from the keyframes' own image rotation.

A flight log rarely starts on the same clock as the video. DJI SRT sidecars
and per-frame CSVs are stamped on the video clock; an Airdata ``isVideo``
column gives the offset to ~0.1 s; an ArduPilot/PX4 log or a trimmed clip
gives nothing, and an operator-typed offset can simply be wrong. Every
telemetry value the reconstruction uses -- GPS position priors, gimbal
attitude -- is then attached to the wrong instant: at 15 m/s a 1.2 s error
puts every camera 18 m along-track from where it was, and in a turn rotates
it by tens of degrees.

The camera's heading is recorded twice: in the log (gimbal yaw, or the GPS
course when no attitude is logged) and in the images, which rotate about
the optical axis by exactly the heading change of a nadir camera. So for
consecutive keyframes ``a, b`` the measured image rotation must equal
``heading(t_b + lag) - heading(t_a + lag)`` at the one lag where the two
clocks agree. Turns make that lag sharply determined (a 0.1 s error in a
30 deg/s turn is a 3 deg mismatch, against sub-degree image rotation
precision); straight legs carry no information and are down-weighted by the
robust cost rather than trusted.

The estimate is only reported as confident when the flight actually turned
inside the keyframe span, the cost minimum is sharp, no second lag explains
the rotations nearly as well (repeated lawnmower legs can), and the
residual at the minimum is small.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from drishti3d.types import TelemetrySample

logger = logging.getLogger(__name__)

_MIN_INLIERS = 20
_TURN_DEG = 5.0  # a keyframe pair "turned" when its image rotated at least this much
_RESIDUAL_CLIP_DEG = 20.0  # robust cost: one bad rotation cannot dominate
_MIN_TURNING_PAIRS = 3
_MAX_MEDIAN_RESIDUAL_DEG = 3.0
_SHARPNESS_MIN = 2.0  # cost 1 s away / cost at the minimum
_UNIQUENESS_MIN = 1.5  # best cost more than 2 s away / cost at the minimum


@dataclass
class SyncEstimate:
    """Result of ``estimate_lag``.

    ``lag_s`` is how far the telemetry timestamps run AHEAD of the video: the
    sample stamped ``t + lag_s`` describes video time ``t``. Correcting means
    subtracting it from every telemetry timestamp, i.e. adding it to the
    video-start offset (``t' = t_raw - offset``).
    """

    lag_s: float
    sign: int
    confident: bool
    reason: str
    heading_source: str
    pairs: int
    turning_pairs: int
    median_residual_deg: float
    sharpness: float
    uniqueness: float
    search_s: tuple[float, float]

    def as_dict(self) -> dict:
        data = asdict(self)
        data["search_s"] = list(self.search_s)
        return data


def _gray(image: np.ndarray, max_side: int) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = max_side / float(max(gray.shape[:2]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def image_rotation_deg(image_a: np.ndarray, image_b: np.ndarray, max_side: int = 640) -> float | None:
    """In-plane rotation (degrees) of ``image_b`` relative to ``image_a``, or None when unmeasurable.

    ORB features, ratio-tested matches, and a RANSAC similarity fit: nadir
    footage between neighbouring keyframes is a rotation + translation + a
    little scale, which a similarity models exactly for flat ground.
    """
    a, b = _gray(image_a, max_side), _gray(image_b, max_side)
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < _MIN_INLIERS or len(kb) < _MIN_INLIERS:
        return None
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < _MIN_INLIERS:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000)
    if M is None or inliers is None or int(inliers.sum()) < _MIN_INLIERS:
        return None
    return math.degrees(math.atan2(M[1, 0], M[0, 0]))


def heading_track(samples: list[TelemetrySample]) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(timestamps, unwrapped heading in degrees, source) from telemetry, or None.

    Gimbal yaw when most samples carry it (the camera's own heading);
    otherwise the GPS course over ground, from samples moving faster than
    2 m/s (slower courses are dominated by position noise).
    """
    ordered = sorted(samples, key=lambda s: s.timestamp)
    with_yaw = [s for s in ordered if s.gimbal_yaw is not None and math.isfinite(s.gimbal_yaw)]
    if len(with_yaw) >= max(10, len(ordered) // 2):
        t = np.array([s.timestamp for s in with_yaw])
        h = np.degrees(np.unwrap(np.radians([s.gimbal_yaw for s in with_yaw])))
        return t, h, "gimbal_yaw"

    with_geo = [s for s in ordered if s.geo is not None]
    if len(with_geo) < 10:
        return None
    from drishti3d.ingest.telemetry import telemetry_to_enu

    enu, _ = telemetry_to_enu(with_geo)
    t = np.array([s.timestamp for s in with_geo])
    dt = np.diff(t)
    d = np.diff(enu[:, :2], axis=0)
    ok = dt > 0
    speed = np.zeros(len(d))
    speed[ok] = np.linalg.norm(d[ok], axis=1) / dt[ok]
    moving = ok & (speed > 2.0)
    if moving.sum() < 10:
        return None
    course = np.degrees(np.arctan2(d[moving, 0], d[moving, 1]))  # clockwise from North
    tm = 0.5 * (t[:-1] + t[1:])[moving]
    return tm, np.degrees(np.unwrap(np.radians(course))), "gps_course"


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def estimate_lag(
    pair_times: np.ndarray,
    pair_rotation_deg: np.ndarray,
    heading_t: np.ndarray,
    heading_deg: np.ndarray,
    lag_range: tuple[float, float],
    heading_source: str = "gimbal_yaw",
) -> SyncEstimate | None:
    """Lag that best explains the image rotations with the logged heading changes.

    ``pair_times`` is ``(n, 2)`` video times of each keyframe pair,
    ``pair_rotation_deg`` their measured image rotation. Returns None when
    no lag in ``lag_range`` keeps at least half the pairs inside the
    telemetry span.
    """
    pair_times = np.asarray(pair_times, dtype=np.float64).reshape(-1, 2)
    rot = np.asarray(pair_rotation_deg, dtype=np.float64)
    finite = np.isfinite(rot)
    pair_times, rot = pair_times[finite], rot[finite]
    n = len(rot)
    if n < 3:
        return None
    t0, t1 = float(heading_t[0]), float(heading_t[-1])
    min_valid = max(3, n // 2)

    def costs(lag: float) -> tuple[float, float]:
        ta, tb = pair_times[:, 0] + lag, pair_times[:, 1] + lag
        ok = (ta >= t0) & (tb <= t1)
        if ok.sum() < min_valid:
            return np.inf, np.inf
        dh = np.interp(tb[ok], heading_t, heading_deg) - np.interp(ta[ok], heading_t, heading_deg)
        out = []
        for sign in (1.0, -1.0):
            r = np.minimum(np.abs(_wrap_deg(rot[ok] - sign * dh)), _RESIDUAL_CLIP_DEG)
            out.append(float(np.mean(r**2)))
        return out[0], out[1]

    lo, hi = float(lag_range[0]), float(lag_range[1])
    coarse = np.arange(lo, hi + 1e-9, 0.1)
    table = np.array([costs(x) for x in coarse])  # (lags, 2 signs)
    if not np.isfinite(table).any():
        return None
    k, s_idx = np.unravel_index(int(np.nanargmin(np.where(np.isfinite(table), table, np.nan))), table.shape)
    sign = 1 if s_idx == 0 else -1
    fine = np.arange(coarse[k] - 0.15, coarse[k] + 0.15 + 1e-9, 0.01)
    fine_cost = np.array([costs(x)[s_idx] for x in fine])
    j = int(np.nanargmin(np.where(np.isfinite(fine_cost), fine_cost, np.nan)))
    lag = float(fine[j])
    if 0 < j < len(fine) - 1 and np.isfinite(fine_cost[j - 1 : j + 2]).all():
        c0, c1, c2 = fine_cost[j - 1], fine_cost[j], fine_cost[j + 1]
        denom = c0 - 2 * c1 + c2
        if denom > 0:
            lag += 0.01 * 0.5 * (c0 - c2) / denom
    best = max(costs(lag)[s_idx], 1e-6)

    # Diagnostics at the chosen lag and sign.
    ta, tb = pair_times[:, 0] + lag, pair_times[:, 1] + lag
    ok = (ta >= t0) & (tb <= t1)
    dh = np.interp(tb[ok], heading_t, heading_deg) - np.interp(ta[ok], heading_t, heading_deg)
    resid = np.abs(_wrap_deg(rot[ok] - sign * dh))
    turning = np.abs(rot[ok]) >= _TURN_DEG
    n_turning = int(turning.sum())
    median_resid = float(np.median(resid[turning])) if n_turning else float("nan")

    near = [costs(lag + d)[s_idx] for d in (-1.0, 1.0)]
    near = [c for c in near if np.isfinite(c)]
    sharpness = (min(near) / best) if near else 0.0
    far = np.abs(coarse - lag) > 2.0
    far_costs = table[far, s_idx] if far.any() else np.array([])
    far_costs = far_costs[np.isfinite(far_costs)]
    uniqueness = float(far_costs.min() / best) if far_costs.size else float("inf")

    reasons = []
    if n_turning < _MIN_TURNING_PAIRS:
        reasons.append(f"only {n_turning} keyframe pairs turned >= {_TURN_DEG:g} deg")
    if not (median_resid <= _MAX_MEDIAN_RESIDUAL_DEG):
        reasons.append(f"median turn residual {median_resid:.1f} deg > {_MAX_MEDIAN_RESIDUAL_DEG:g}")
    if sharpness < _SHARPNESS_MIN:
        reasons.append(f"cost minimum not sharp ({sharpness:.2f}x at +/-1 s)")
    if uniqueness < _UNIQUENESS_MIN:
        reasons.append(f"another lag >2 s away fits almost as well ({uniqueness:.2f}x)")
    confident = not reasons
    return SyncEstimate(
        lag_s=round(lag, 3),
        sign=sign,
        confident=confident,
        reason="; ".join(reasons) if reasons else "sharp, unique minimum",
        heading_source=heading_source,
        pairs=int(ok.sum()),
        turning_pairs=n_turning,
        median_residual_deg=round(median_resid, 3) if np.isfinite(median_resid) else float("nan"),
        sharpness=round(sharpness, 3),
        uniqueness=round(uniqueness, 3) if np.isfinite(uniqueness) else float("inf"),
        search_s=(lo, hi),
    )


def keyframe_pairs(timestamps: list[float], max_gap_s: float = 6.0, steps: tuple[int, ...] = (1, 2)) -> list[tuple[int, int]]:
    """Neighbouring keyframe index pairs close enough in time to still overlap."""
    pairs = []
    for step in steps:
        for i in range(len(timestamps) - step):
            j = i + step
            if 0 < timestamps[j] - timestamps[i] <= max_gap_s:
                pairs.append((i, j))
    return pairs


def estimate_from_keyframes(
    images: list[np.ndarray | None],
    timestamps: list[float],
    samples: list[TelemetrySample],
    lag_range: tuple[float, float],
    max_side: int = 640,
    progress=None,
) -> SyncEstimate | None:
    """``estimate_lag`` over every overlapping keyframe pair's measured rotation.

    ``progress(done, total)``, when given, is called once per pair.
    """
    track = heading_track(samples)
    if track is None:
        return None
    heading_t, heading_deg, source = track
    # Neighbours and next-but-one on short flights; neighbours only on long
    # ones, where one fit per pair is already plenty of turns.
    pairs = keyframe_pairs(timestamps, steps=(1, 2) if len(timestamps) <= 150 else (1,))
    times, rots = [], []
    for k, (i, j) in enumerate(pairs):
        if progress is not None:
            progress(k, len(pairs))
        if images[i] is None or images[j] is None:
            continue
        angle = image_rotation_deg(images[i], images[j], max_side=max_side)
        if angle is None:
            continue
        times.append((timestamps[i], timestamps[j]))
        rots.append(angle)
    if len(rots) < 3:
        return None
    return estimate_lag(np.array(times), np.array(rots), heading_t, heading_deg, lag_range, heading_source=source)
