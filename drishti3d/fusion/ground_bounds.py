"""Reject reconstructed points that cannot physically be terrain.

What this catches, and why the existing cleanup cannot
------------------------------------------------------
``fusion.mesh``'s cleanup finds *slivers* -- long thin triangles thrown
across depth discontinuities -- by comparing each face's longest edge
against the mesh's own median edge. That works for what Poisson leaves
behind, and it is blind to the failure actually seen on this project:

    a 174 m tall, 142 m long body of points standing on the ground,
    made of dense, normal-sized triangles

Its faces are not slivers. Its edges are ordinary. It is well-formed
geometry in an impossible place, so an edge-length test cannot see it --
measured on two consecutive runs, the spike filter removed exactly 0
faces while that column sat in the middle of the model.

The test that does see it
-------------------------
Telemetry says where the ground is: ``camera_z - alt_rel``, the same
reference the placement check uses. Terrain and everything on it lives in
a band around that. A drone survey contains buildings and trees, so the
band has to be generous -- but nothing in it is 174 m tall, and that is
enough to separate real structure from a misplaced reconstruction.

Why this is a backstop and not the fix
--------------------------------------
Points this far out of place are a symptom: on this project they came
from a submap whose Sim(3) fit was degenerate (junction flagged
``[DEGENERATE]``, condition number 7.5e4) on a near-collinear window.
Deleting them does not make that submap correct, it only stops one broken
window from dominating the render and from inflating the bounding box
that sets TSDF's voxel resolution.

So this reports what it removed, loudly, and a run that needs it to
remove more than a few percent is a run with a real upstream problem --
which is why ``filter_to_ground_band`` returns the fraction rejected
rather than quietly handing back a smaller cloud.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["GroundBand", "filter_to_ground_band"]

#: How far above the ground plane a point may sit and still be believable.
#: Tall buildings and mature trees reach ~60 m; this keeps them and the
#: headroom above them. It is deliberately not tight -- the artifact this
#: exists to catch was 174 m, and a bound that tries to also trim ordinary
#: depth noise would start deleting real roofs.
_DEFAULT_ABOVE_M = 80.0

#: How far below. Real terrain drops away (a valley, a quarry, water),
#: and the reconstruction's own noise scatters points downward too, so
#: this is looser than a flat-field survey strictly needs.
_DEFAULT_BELOW_M = 60.0

#: Rejecting more than this fraction means the band is not trimming
#: outliers, it is deleting the reconstruction -- so the filter declines
#: to act and says why, rather than returning a cloud with most of the
#: model missing.
_MAX_REJECT_FRACTION = 0.25


class GroundBand:
    """The plausible-terrain band, and what it rejected."""

    def __init__(self, ground_z: float, above_m: float, below_m: float) -> None:
        self.ground_z = ground_z
        self.above_m = above_m
        self.below_m = below_m

    @property
    def low(self) -> float:
        return self.ground_z - self.below_m

    @property
    def high(self) -> float:
        return self.ground_z + self.above_m

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        z = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[:, 2]
        return (z >= self.low) & (z <= self.high)

    def as_dict(self) -> dict:
        return {
            "ground_z_m": round(float(self.ground_z), 3),
            "band_low_m": round(float(self.low), 3),
            "band_high_m": round(float(self.high), 3),
            "above_m": self.above_m,
            "below_m": self.below_m,
        }


def filter_to_ground_band(
    xyz: np.ndarray,
    ground_z: float | None,
    *,
    above_m: float = _DEFAULT_ABOVE_M,
    below_m: float = _DEFAULT_BELOW_M,
    max_reject_fraction: float = _MAX_REJECT_FRACTION,
) -> tuple[np.ndarray, dict]:
    """Keep-mask for points within a plausible band around the ground.

    Returns ``(keep_mask, diagnostics)``. The mask is all-``True`` --
    i.e. the filter declines -- when there is no ground reference, or
    when applying it would reject more than ``max_reject_fraction`` of
    the cloud. In that second case the problem is upstream and cutting
    a quarter of the model would hide it rather than help.
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool), {"applied": False, "reason": "empty cloud"}
    if ground_z is None:
        return np.ones(n, dtype=bool), {
            "applied": False,
            "reason": "no telemetry ground height; nothing to judge plausibility against",
        }

    band = GroundBand(float(ground_z), above_m, below_m)
    keep = band.mask(pts)
    rejected = int((~keep).sum())
    fraction = rejected / float(n)

    diag = {
        **band.as_dict(),
        "points_in": n,
        "points_rejected": rejected,
        "rejected_fraction": round(fraction, 5),
    }

    if fraction > max_reject_fraction:
        logger.warning(
            "GROUND BAND: %.1f%% of the cloud lies outside %.1f..%.1f m -- that is not a handful of "
            "outliers, it is most of the reconstruction, so the filter is declining. The depth or "
            "the merge is wrong upstream; trimming here would hide that instead of fixing it.",
            100.0 * fraction,
            band.low,
            band.high,
        )
        diag.update(applied=False, reason=f"would reject {100.0 * fraction:.1f}% of the cloud")
        return np.ones(n, dtype=bool), diag

    diag["applied"] = True
    if rejected:
        z = pts[~keep][:, 2]
        diag["worst_above_m"] = round(float(z.max() - band.ground_z), 2)
        diag["worst_below_m"] = round(float(band.ground_z - z.min()), 2)
        logger.warning(
            "GROUND BAND: removed %d points (%.2f%%) outside %.1f..%.1f m around the telemetry "
            "ground at %.1f m; worst was %.0f m above and %.0f m below it. These are not terrain. "
            "They are a symptom -- see fusion.ground_bounds -- so treat a large number here as an "
            "upstream failure, not as cleanup that worked.",
            rejected,
            100.0 * fraction,
            band.low,
            band.high,
            band.ground_z,
            diag["worst_above_m"],
            diag["worst_below_m"],
        )
    else:
        logger.info(
            "GROUND BAND: every point is within %.1f..%.1f m of the telemetry ground; nothing removed",
            band.low,
            band.high,
        )
    return keep, diag
