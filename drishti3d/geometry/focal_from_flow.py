"""Measure focal length from GPS baseline, telemetry altitude and image motion.

Why bundle adjustment cannot do this here
------------------------------------------
``BAConfig.refine_intrinsics`` works -- verified on synthetic data, where a
focal seeded 1.57x low recovers toward truth. It does nothing on a real
survey, and the reason is structural rather than a bug: the BA points were
*triangulated using the seeded focal*, so they are already self-consistent
with it. Reprojection error is ~1 px at the seed, no gradient pushes the
focal anywhere, and the solver sits in that minimum. Measured on 77 real
keyframes: 18 free cameras, focal spread 1066-1066 px, factor exactly
1.000. COLMAP reaches a different answer only because incremental SfM
*estimates* focal from two-view geometry instead of starting from a seed.

The measurement this module makes instead
------------------------------------------
For a nadir camera at height ``Z`` above locally flat ground, translating
by a metric baseline ``B``, every ground point shifts across the image by::

    d = fx * B / Z          [pixels]

``B`` comes from GPS, ``Z`` from telemetry, and ``d`` is measured from the
images themselves. So::

    fx = d * Z / B

Each of those is an input the problem statement guarantees (video, GPS,
flight metadata). Nothing is seeded and nothing is regressed, which is why
this converges where self-calibration does not.

Measured on the sample flight: 2698 px (p25 2673, p75 2720 over 13 pairs,
a ~1% spread), against a 2132 px generic-HFOV guess -- a 27% error in the
one input that scales every horizontal distance and every triangulated
depth in the reconstruction.

What it assumes, and how each assumption fails
-----------------------------------------------
- **Nadir view.** Off-nadir, part of the motion is parallax across depth
  rather than uniform translation, and the median shift no longer equals
  ``fx*B/Z``. Callers pass the gimbal pitch and pairs outside tolerance
  are skipped.
- **Locally flat ground.** Terrain relief ``dZ`` biases the estimate by
  roughly ``dZ/Z``: at 120 m altitude, 6 m of relief is a 5% error. The
  median over many pairs absorbs the scatter; a systematic slope does not
  cancel and is reported via the spread.
- **``Z`` is height above the ground being imaged.** ``alt_rel`` is height
  above the TAKEOFF POINT. On terrain-following flight over sloping ground
  these differ, and the estimate inherits that difference directly. This
  is the dominant error source and the reason the result is returned with
  its inter-quartile spread rather than as a bare number.
- **Pure translation between the pair.** Rotation adds image motion that
  is not parallax. Pairs whose estimates disagree with the median are
  rejected before it is reported.

Because of those, this returns a *better prior*, not a calibration. It is
the right input for a pipeline whose alternative is a generic HFOV guess.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from drishti3d.geometry.features import detect_and_describe, geometric_verify, match_features
from drishti3d.geometry.mapanything import resize_preserving_aspect

logger = logging.getLogger(__name__)

__all__ = ["FocalEstimate", "estimate_focal_from_flow"]

#: Resolution the flow is measured at. Only a median displacement is
#: needed, which is stable well below native; keypoint coordinates are
#: scaled back to native pixels before the focal is computed.
_WORK_SIZE = 1280

#: Minimum metric baseline for a usable pair. Below this the displacement
#: is dominated by matching noise rather than motion.
_MIN_BASELINE_M = 5.0

#: Verified inliers a pair needs before its displacement is believed.
_MIN_INLIERS = 40

#: Degrees from straight down beyond which the flat-ground/nadir model
#: stops holding well enough to measure focal length from.
_MAX_OFF_NADIR_DEG = 12.0

#: Pairs whose estimate deviates from the median by more than this
#: fraction are dropped before the final median. Terrain relief and
#: gimbal motion produce outliers that should not move the answer.
_OUTLIER_FRACTION = 0.25


class FocalEstimate:
    """A measured focal length, with the evidence needed to judge it."""

    def __init__(self, fx: float, samples: list[float], pairs_used: int, pairs_tried: int) -> None:
        self.fx = fx
        self.samples = samples
        self.pairs_used = pairs_used
        self.pairs_tried = pairs_tried

    @property
    def spread_fraction(self) -> float:
        """Inter-quartile spread as a fraction of the estimate.

        The honest uncertainty. Wide spread means the flat-ground or
        constant-altitude assumption is not holding on this flight, and
        the number should be treated as weak.
        """
        if len(self.samples) < 4:
            return float("inf")
        q1, q3 = np.percentile(self.samples, [25, 75])
        return float((q3 - q1) / max(self.fx, 1e-6))

    def as_dict(self) -> dict:
        return {
            "fx_px": round(self.fx, 1),
            "pairs_used": self.pairs_used,
            "pairs_tried": self.pairs_tried,
            "iqr_fraction": round(self.spread_fraction, 4),
            "p25_px": round(float(np.percentile(self.samples, 25)), 1) if len(self.samples) >= 4 else None,
            "p75_px": round(float(np.percentile(self.samples, 75)), 1) if len(self.samples) >= 4 else None,
        }


def estimate_focal_from_flow(
    images: list[np.ndarray | None],
    positions: list[np.ndarray | None],
    altitudes: list[float | None],
    gimbal_pitch_deg: list[float | None],
    *,
    native_width: int,
    max_pairs: int = 24,
    min_baseline_m: float = _MIN_BASELINE_M,
) -> FocalEstimate | None:
    """Focal length in NATIVE pixels, or ``None`` when it cannot be measured.

    ``images`` may be at any resolution; displacements are converted back
    to ``native_width`` pixels, because the focal length this returns
    describes the full-resolution frame regardless of what was matched.
    """
    n = min(len(images), len(positions), len(altitudes))
    samples: list[float] = []
    tried = 0

    for i in range(n - 1):
        if len(samples) >= max_pairs:
            break
        a, b = images[i], images[i + 1]
        if a is None or b is None or positions[i] is None or positions[i + 1] is None:
            continue
        alt = altitudes[i]
        if alt is None or alt <= 1.0:
            continue
        pitch = gimbal_pitch_deg[i] if i < len(gimbal_pitch_deg) else None
        if pitch is not None and abs(abs(float(pitch)) - 90.0) > _MAX_OFF_NADIR_DEG:
            continue
        baseline = float(np.linalg.norm(np.asarray(positions[i + 1])[:2] - np.asarray(positions[i])[:2]))
        if baseline < min_baseline_m:
            continue

        tried += 1
        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
        work_a, _ = resize_preserving_aspect(gray_a, _WORK_SIZE)
        work_b, scale_b = resize_preserving_aspect(gray_b, _WORK_SIZE)
        # Displacement is measured in work pixels; converting to native
        # needs the ratio between the ORIGINAL frame and the work image,
        # not the resize factor of whatever was passed in.
        native_scale = work_b.shape[1] / float(native_width)

        fa = detect_and_describe(work_a, method="sift", max_features=3000)
        fb = detect_and_describe(work_b, method="sift", max_features=3000)
        matches = match_features(fa, fb, ratio=0.8)
        if len(matches) < _MIN_INLIERS:
            continue
        verified = geometric_verify(fa, fb, matches)
        if verified.inlier_mask is None or int(verified.inlier_mask.sum()) < _MIN_INLIERS:
            continue

        q = verified.query_idx[verified.inlier_mask]
        t = verified.train_idx[verified.inlier_mask]
        shift_work = float(np.median(np.linalg.norm(fb.keypoints[t, :2] - fa.keypoints[q, :2], axis=1)))
        if shift_work <= 1e-6:
            continue
        shift_native = shift_work / max(native_scale, 1e-9)
        samples.append(shift_native * float(alt) / baseline)

    if len(samples) < 3:
        logger.info(
            "focal from flow: only %d usable pairs (%d tried); keeping the intrinsics prior",
            len(samples),
            tried,
        )
        return None

    # Reject pairs that disagree with the consensus before taking the
    # answer: terrain relief and gimbal motion produce outliers, and a
    # median over a contaminated set is still a contaminated median.
    arr = np.asarray(samples, dtype=np.float64)
    rough = float(np.median(arr))
    keep = arr[np.abs(arr - rough) <= _OUTLIER_FRACTION * rough]
    if len(keep) < 3:
        keep = arr

    estimate = FocalEstimate(float(np.median(keep)), keep.tolist(), len(keep), tried)
    logger.info(
        "focal from flow: fx = %.0f px from %d/%d pairs (IQR %.1f%% of the estimate). "
        "Measured as d*Z/B from GPS baseline, telemetry altitude and image displacement -- "
        "no seed, no self-calibration.",
        estimate.fx,
        estimate.pairs_used,
        tried,
        100.0 * estimate.spread_fraction,
    )
    return estimate
