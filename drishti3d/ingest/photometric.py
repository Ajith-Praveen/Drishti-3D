"""Photometric normalisation: make frames comparable despite the sun.

Two different problems live here, and conflating them produces a worse
result than solving neither.

1. Within-frame contrast (``normalize_illumination``)
-----------------------------------------------------
A single drone frame can contain a sunlit roof and a deep building shadow.
Global auto-exposure picks one, and detail in the other is crushed. CLAHE
on the *lightness* channel only (never on RGB directly, which shifts hue)
recovers local contrast in both. This helps feature matching -- SIFT and
friends are contrast-dependent, and a shadowed facade with no usable
gradient contributes no tracks at all.

Crucially this is applied for *analysis*, not for export: the texture atlas
is baked from the original pixels, because a CLAHE'd texture is no longer
a photograph of the scene and should not be presented as one.

2. Between-frame exposure (``solve_exposure_gains``)
-----------------------------------------------------
Auto-exposure changes between keyframes as the drone turns relative to the
sun. Bake those frames into one atlas untouched and the result has visible
brightness discontinuities wherever the source view changes -- the single
most obvious "this is a stitched model" artifact there is.

The fix is the classic gain-compensation least squares (Brown & Lowe's
panorama formulation, applied here to 3D surface samples rather than image
overlaps): find a per-view scalar gain ``g_i`` minimising the disagreement
between views that observed the *same surface point*, with a prior pulling
the gains toward 1 so the system is not free to drive everything to zero.

Why a scalar gain and not a full colour transform: a per-channel or affine
model has enough freedom to "explain" genuine albedo differences between
surfaces as exposure, which bleaches real colour variation out of the
model. A single multiplicative gain per view can only correct what
auto-exposure actually does.

What this does NOT do
---------------------
It does not remove shadows. A cast shadow is a real radiometric fact about
the scene at capture time, and inventing the surface underneath it would be
fabricating data. Gain compensation makes the *seams* between views
disappear; the shadow stays, correctly, and the texture shows it.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "apply_gains",
    "normalize_illumination",
    "solve_exposure_gains",
]

#: Strength of the prior pulling every gain toward 1.0. Without it the
#: system is scale-degenerate (multiplying every gain by any constant
#: leaves pairwise disagreement unchanged), and the solution drifts toward
#: a globally darker or brighter model with each run.
_GAIN_PRIOR_WEIGHT = 1.0

#: Gains outside this range are refused. A view needing a 3x correction is
#: not slightly mis-exposed -- it is a frame the exposure model does not
#: describe (a lens flare, a cloud shadow crossing mid-frame), and applying
#: the fitted gain would spread that frame's problem across the atlas.
_MIN_GAIN, _MAX_GAIN = 0.4, 2.5


def normalize_illumination(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """CLAHE the lightness channel of a BGR frame, preserving hue.

    Converts to LAB, equalises ``L`` only, converts back. Running CLAHE on
    the three BGR channels independently would equalise them to different
    curves and visibly shift colour -- grass toward grey, brick toward
    orange -- which is why this costs a colour-space round trip instead.

    ``clip_limit`` bounds the contrast amplification: above ~4 it starts
    amplifying sensor noise in the shadows it is meant to be recovering,
    which for compressed drone video means amplifying compression blocks.
    """
    if image is None or image.size == 0:
        return image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a BGR (H, W, 3) frame, got shape {image.shape}")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def solve_exposure_gains(
    observations: list[tuple[int, int, float, float]],
    n_views: int,
    *,
    prior_weight: float = _GAIN_PRIOR_WEIGHT,
) -> np.ndarray:
    """Least-squares per-view exposure gains from co-observed surface points.

    ``observations`` is a list of ``(view_i, view_j, luma_i, luma_j)``:
    both views saw one surface point, and reported these two luminances.
    In a perfectly exposed pair ``g_i * luma_i == g_j * luma_j``.

    Returns ``(n_views,)`` gains, each clamped to a sane range. Views with
    no observations keep a gain of exactly 1.0 -- an unconstrained view
    must not be nudged by the prior into a correction nothing justifies.

    Solved in log space: the constraint is multiplicative, so
    ``log g_i - log g_j == log(luma_j) - log(luma_i)`` is linear, and the
    whole system becomes one ordinary least-squares problem rather than an
    iterative nonlinear fit.
    """
    if n_views <= 0:
        return np.ones(0, dtype=np.float64)
    if not observations:
        return np.ones(n_views, dtype=np.float64)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    constrained = np.zeros(n_views, dtype=bool)

    for i, j, luma_i, luma_j in observations:
        if luma_i <= 1e-3 or luma_j <= 1e-3:
            # Near-black samples carry no exposure information and their
            # log ratio is dominated by noise.
            continue
        row = np.zeros(n_views)
        row[i] = 1.0
        row[j] = -1.0
        rows.append(row)
        rhs.append(float(np.log(luma_j) - np.log(luma_i)))
        constrained[i] = True
        constrained[j] = True

    if not rows:
        return np.ones(n_views, dtype=np.float64)

    # Prior: every gain pulled toward log(1) == 0.
    for k in range(n_views):
        row = np.zeros(n_views)
        row[k] = prior_weight
        rows.append(row)
        rhs.append(0.0)

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)
    log_gains, *_ = np.linalg.lstsq(A, b, rcond=None)

    gains = np.exp(log_gains)
    gains[~constrained] = 1.0

    out_of_range = (gains < _MIN_GAIN) | (gains > _MAX_GAIN)
    if np.any(out_of_range):
        logger.info(
            "photometric: %d/%d view gains fell outside [%.1f, %.1f] and were clamped -- "
            "those frames are probably not just mis-exposed",
            int(out_of_range.sum()),
            n_views,
            _MIN_GAIN,
            _MAX_GAIN,
        )
    return np.clip(gains, _MIN_GAIN, _MAX_GAIN)


def apply_gains(samples: np.ndarray, gains: np.ndarray, view_indices: np.ndarray) -> np.ndarray:
    """Scale per-sample colours by their source view's gain, saturating at 255.

    ``samples`` is ``(N, 3)`` float colour, ``view_indices`` is ``(N,)``
    telling which view each sample came from. Clipped rather than
    normalised: a pixel that saturates after correction was already at or
    near the sensor's ceiling, and rescaling the whole image to preserve it
    would darken everything else to protect one blown highlight.
    """
    scaled = np.asarray(samples, dtype=np.float32) * gains[view_indices][:, None].astype(np.float32)
    return np.clip(scaled, 0.0, 255.0)
