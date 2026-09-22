"""Verify reconstructed geometry against the source frames that made it.

The idea
--------
A 3D point is only correct if every camera that can see it agrees about
what colour it is. Project a point into all the frames that observed it: if
the surface is really there, each view samples the same physical patch and
the colours match. If the geometry is wrong -- a point floating in front of
a wall, a Poisson bridge across a gap, a depth spike -- the views sample
*different* physical patches and the colours disagree.

That disagreement is a direct, measurable error signal, computed against
the actual imagery rather than against the model's opinion of itself.

Why this matters more than the backbone's own confidence
--------------------------------------------------------
Until now ``confidence_source`` was ``backbone_confidence_and_view_count``:
MapAnything's self-reported per-pixel confidence, plus how many views saw
the point. Both are useful, and both are the model grading its own
homework. Modern depth networks are systematically overconfident, and a
confidently-wrong depth produces a confidently-wrong point.

Photometric agreement is *independent evidence*. A point whose appearance
is consistent across four separate views has been checked against four
separate photographs. That is a stronger claim than "the network felt sure",
and it is why a photometrically-verified point can honestly be called
MEASURED even when the backbone was unsure -- and why a point the backbone
loved gets demoted when the images disagree.

What this deliberately does NOT do
-----------------------------------
It never moves a vertex. Photometric error could in principle drive an
optimiser that slides points until the views agree, and that is a real
technique (photometric bundle adjustment) -- but it optimises *appearance*,
and a surface nudged to look consistent is no longer a surface anybody
measured. For a system whose deliverable is metric accuracy, this signal is
used to **grade and to delete**, never to reshape.

Limits worth stating
--------------------
Photometric consistency fails honestly in three places, and the caller is
told rather than left to assume:

- **Textureless surfaces.** A blank road agrees across every view whether
  or not the geometry is right. Agreement there is weak evidence, so the
  local texture contrast is returned alongside the error and points below
  a contrast floor are reported as ``unverifiable``, not as verified.
- **Specular and water surfaces**, which legitimately change appearance
  between viewpoints and will score as inconsistent.
- **Exposure drift** between frames, which shifts colour globally. That is
  what ``ingest.photometric``'s gain compensation exists to remove, and
  this module compares in a normalised space so residual drift matters
  less.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from drishti3d.semantics.labelling import project_points, visible_mask
from drishti3d.types import CameraIntrinsics, Confidence, Pose

logger = logging.getLogger(__name__)

__all__ = ["PhotometricResult", "photometric_consistency", "verify_confidence"]

#: Points needing at least this many views before agreement means anything.
#: Two views can agree by coincidence far more easily than four.
_MIN_VIEWS = 3

#: Local texture contrast (std of the sampled patch, 0-255) below which a
#: point is called UNVERIFIABLE rather than verified. A blank surface agrees
#: across every view regardless of whether its geometry is right, so
#: agreement there carries almost no information.
_MIN_CONTRAST = 6.0

#: Mean absolute colour deviation across views, in 0-255 units, below which
#: the views are taken to agree. ~12/255 tolerates JPEG noise, mild exposure
#: drift and resampling error while still rejecting a point that projects
#: onto genuinely different surfaces.
_AGREE_THRESHOLD = 12.0


#: Half-width (pixels) of the neighbourhood local texture is measured over.
#: 2 (a 5x5 window) is wide enough to span a roof edge or a road marking at
#: typical survey GSD, and narrow enough that it measures the texture at the
#: point rather than the variation of the scene around it.
_TEXTURE_RADIUS_PX = 2


def _local_texture(image: np.ndarray) -> np.ndarray:
    """Per-pixel local intensity spread -- how much detail is there to match.

    Standard deviation over a square window, computed as
    ``sqrt(E[x^2] - E[x]^2)`` via box filters, which is O(1) per pixel
    regardless of window size.

    This is what the contrast floor in ``photometric_consistency`` needs: a
    blank surface agrees across every view whether or not its geometry is
    right, so agreement there is uninformative. Judging that from the
    *image* is the only way to know it -- the colour of a point says
    nothing about whether its neighbourhood had any detail to match on.
    """
    grey = image.mean(axis=2) if image.ndim == 3 else image.astype(np.float64)
    grey = grey.astype(np.float32)
    k = 2 * _TEXTURE_RADIUS_PX + 1
    try:
        import cv2

        mean = cv2.boxFilter(grey, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
        mean_sq = cv2.boxFilter(grey * grey, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    except ImportError:
        from scipy import ndimage

        mean = ndimage.uniform_filter(grey, size=k, mode="reflect")
        mean_sq = ndimage.uniform_filter(grey * grey, size=k, mode="reflect")
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))


@dataclass
class PhotometricResult:
    """Per-point photometric evidence, plus why each point got its verdict."""

    #: Mean absolute deviation of the sampled colour across views (0-255).
    #: ``inf`` where too few views saw the point to say anything.
    error: np.ndarray
    #: How many views actually sampled each point.
    view_count: np.ndarray
    #: Local texture contrast at the point, across views. Low contrast
    #: means agreement is uninformative, not that geometry is good.
    contrast: np.ndarray
    #: True where the point was seen enough, has enough texture to judge,
    #: and the views agreed.
    verified: np.ndarray
    #: True where there was not enough evidence either way -- too few
    #: views, or too little texture. NOT the same as "wrong".
    unverifiable: np.ndarray
    stats: dict = field(default_factory=dict)


def photometric_consistency(
    xyz: np.ndarray,
    views: list[tuple[Pose, CameraIntrinsics, np.ndarray]],
    *,
    min_views: int = _MIN_VIEWS,
    min_contrast: float = _MIN_CONTRAST,
    agree_threshold: float = _AGREE_THRESHOLD,
    occlusion_tolerance_m: float = 0.5,
) -> PhotometricResult:
    """Score every point by how well the views that see it agree.

    ``views`` is ``(pose, intrinsics, image_bgr)`` per keyframe. Occlusion
    is resolved with the same shared z-buffer the semantic vote and the
    texture baker use, so all three agree about what is visible from where.

    Agreement is measured as mean absolute deviation from the per-point
    mean colour, which is cheap, bounded, and does not assume the views are
    photometrically calibrated the way a correlation measure would.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n == 0 or not views:
        empty_f = np.zeros(n, dtype=np.float32)
        empty_b = np.zeros(n, dtype=bool)
        return PhotometricResult(
            error=np.full(n, np.inf, dtype=np.float32),
            view_count=np.zeros(n, dtype=np.uint16),
            contrast=empty_f,
            verified=empty_b,
            unverifiable=np.ones(n, dtype=bool),
            stats={"views_used": 0, "points": int(n)},
        )

    # Running sums, so no (points x views) matrix is ever materialised --
    # a 10M-point cloud over 60 views would otherwise need ~14 GB.
    # float32, not float64: with tens of millions of points these two
    # arrays dominate the stage's memory (48 B/point at float64), and the
    # values are sums of at most a few dozen 0-255 samples, which float32
    # carries exactly. Variance is formed below in float64 from the sums.
    total = np.zeros((n, 3), dtype=np.float32)
    total_sq = np.zeros((n, 3), dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    # Best (highest) local texture any view saw at this point. Max, not
    # mean: one sharp, well-exposed view of a textured patch is enough to
    # make agreement meaningful, and averaging would let a few blurred or
    # grazing views veto evidence the good view genuinely provides.
    texture = np.zeros(n, dtype=np.float32)

    views_used = 0
    for pose, intr, image in views:
        if image is None:
            continue
        uv, depth, in_view = project_points(xyz, pose, intr)
        vis = visible_mask(
            uv, depth, in_view, width=intr.width, height=intr.height, tolerance_m=occlusion_tolerance_m
        )
        if not np.any(vis):
            continue
        idx = np.flatnonzero(vis)
        px = np.clip(uv[idx, 0].astype(np.int32), 0, intr.width - 1)
        py = np.clip(uv[idx, 1].astype(np.int32), 0, intr.height - 1)

        sample = image[py, px].astype(np.float32)
        total[idx] += sample
        total_sq[idx] += sample * sample
        counts[idx] += 1.0

        # Local texture at the sampled pixel, accumulated alongside the
        # colour. See `contrast` below for why this is measured from the
        # image rather than inferred from the colour itself.
        tex = _local_texture(image)
        np.maximum.at(texture, idx, tex[py, px])
        views_used += 1

    view_count = counts.astype(np.uint16)
    enough = counts >= max(2, min_views)
    safe = np.where(counts > 0, counts, 1.0).astype(np.float64)[:, None]

    mean = total.astype(np.float64) / safe
    # Variance across views, per channel, then averaged. Clipped at 0
    # because catastrophic cancellation in the sum-of-squares form can
    # produce a tiny negative.
    variance = np.clip(total_sq.astype(np.float64) / safe - mean * mean, 0.0, None)
    error = np.sqrt(variance).mean(axis=1).astype(np.float32)
    error[~enough] = np.inf

    # Contrast: how much signal there was to disagree about.
    #
    # This used to be `mean.std(axis=1)` -- the spread across the R, G and B
    # channels of the mean colour -- described in the comment as a stand-in
    # for local texture. It is not one. That expression measures colour
    # SATURATION: a vividly coloured but perfectly flat surface scores high,
    # while a sharply textured grey roof scores zero. Aerial imagery is
    # largely desaturated, so on a real 77-keyframe run it put 96.5% of all
    # points below the contrast floor and reported them UNVERIFIABLE --
    # which silently disabled the entire photometric check on exactly the
    # imagery it was written for.
    #
    # `texture` is instead measured from the images themselves (local
    # intensity spread around the sampled pixel), which is what the floor
    # was always meant to test.
    contrast = texture.astype(np.float32)

    unverifiable = (~enough) | (contrast < min_contrast)
    verified = enough & (~unverifiable) & (error <= agree_threshold)

    stats = {
        "views_used": views_used,
        "points": int(n),
        "verified_points": int(verified.sum()),
        "unverifiable_points": int(unverifiable.sum()),
        "inconsistent_points": int((enough & (~unverifiable) & (error > agree_threshold)).sum()),
        "verified_pct": round(100.0 * float(verified.mean()), 2),
        "median_error": float(np.median(error[np.isfinite(error)])) if np.any(np.isfinite(error)) else None,
        "mean_views_per_point": round(float(counts.mean()), 2),
    }
    return PhotometricResult(
        error=error,
        view_count=view_count,
        contrast=contrast,
        verified=verified,
        unverifiable=unverifiable,
        stats=stats,
    )


def verify_confidence(
    confidence: np.ndarray,
    result: PhotometricResult,
    *,
    promote: bool = True,
    demote: bool = True,
) -> tuple[np.ndarray, dict]:
    """Re-grade confidence tiers using photometric evidence.

    This is the point of the whole module. ``confidence`` arrives as the
    backbone's self-assessment; this replaces it with a verdict backed by
    the source photographs:

    - **Promote to MEASURED** where several views independently agree about
      the point's appearance. That is a stronger claim than the network's
      own confidence, not a weaker one -- it is the difference between "the
      model felt sure" and "four photographs concur".
    - **Demote** points the views disagree about, however confident the
      backbone was. Confidently-wrong depth is the failure mode this exists
      to catch.
    - **Leave unverifiable points alone.** A textureless road or a point
      seen by one camera has no photometric evidence either way, and
      inventing a verdict for it would be exactly the fabrication this
      project refuses elsewhere.

    Returns ``(confidence, stats)``.
    """
    confidence = np.asarray(confidence, dtype=np.uint8).copy()
    before_measured = int((confidence == Confidence.MEASURED).sum())

    promoted = demoted = 0
    if promote:
        target = result.verified & (confidence != Confidence.MEASURED)
        promoted = int(target.sum())
        confidence[target] = Confidence.MEASURED

    if demote:
        # Seen by enough views, enough texture to judge, and they disagreed.
        inconsistent = (~result.unverifiable) & (~result.verified) & np.isfinite(result.error)
        target = inconsistent & (confidence == Confidence.MEASURED)
        demoted = int(target.sum())
        confidence[target] = Confidence.LOW_CONFIDENCE

    after_measured = int((confidence == Confidence.MEASURED).sum())
    n = max(len(confidence), 1)
    stats = {
        "photometric_promoted": promoted,
        "photometric_demoted": demoted,
        "measured_pct_before": round(100.0 * before_measured / n, 2),
        "measured_pct_after": round(100.0 * after_measured / n, 2),
    }
    logger.info(
        "photometric: promoted %d, demoted %d -> measured %.1f%% -> %.1f%%",
        promoted,
        demoted,
        stats["measured_pct_before"],
        stats["measured_pct_after"],
    )
    return confidence, stats
