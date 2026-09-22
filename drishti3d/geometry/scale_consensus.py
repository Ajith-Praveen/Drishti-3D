"""Force every window onto one depth scale before they are merged.

The defect this fixes
---------------------
``depth_anchor`` measures, per window, how far the backbone's depth is
from the truth, and rescales that window by the measured ratio. Each
window measures independently. On a real flight the measured ratios came
out between 1.16 and 6.79 -- a factor of 5.9 across 19 windows of the
same field, flown at the same altitude, by the same camera, through the
same backbone.

They cannot all be right. Every window looks at ground roughly 120 m
below the camera; a window whose depth ends up 1.16x its true scale puts
that ground about 23 m down and one at 6.79x puts it about 136 m down.
Merging aligns the *cameras* -- that is what GPS anchoring does, and it
fits scale from camera baselines -- so it pins the track and leaves the
ground surfaces at whatever depth each window decided on, a hundred
metres apart. What comes out is one terrain reconstructed at many
heights, which looks exactly like a merge failure and is not one.

So the disagreement has to be resolved before merging, and that is all
this module does.

When it should refuse
---------------------
Only when the per-window estimates form a single cluster. On the sample
flight they did not -- see ``_MAX_CONSENSUS_SPREAD`` for the measurement
and for what happened when an earlier version averaged the clusters
anyway.

Why a consensus is the right resolution
---------------------------------------
The quantity being measured -- how wrong the backbone's metric depth is
on this footage -- is a property of the backbone and the footage, not of
the window. Windows differ only in which few seconds of flight they
cover. So the *true* ratio is very nearly one number, every window is a
noisy measurement of it, and the robust consensus of those measurements
is a far better estimate than any single window's.

Preferring the GPS-altitude half
--------------------------------
``anchor_depth_fused`` reports two independent estimates per window:
parallax (needs feature matches and a decent baseline; fails quietly into
nonsense when either is thin) and GPS altitude (``alt_rel`` over the
reconstructed ground depth; needs neither). For a *consensus* the GPS
estimate is the better input even though it is individually cruder: its
failure mode is a bias shared by every window, which the merge can absorb
as one global scale, whereas parallax's failure mode is per-window
scatter, which is precisely what must not enter the consensus. Parallax
is used when a window has no GPS estimate, and to sanity-check the
consensus afterwards.

Why the correction is applied about the cameras, not about an origin
--------------------------------------------------------------------
This is the part the first version of this module got wrong, and it made
the whole thing a no-op. ``merge_submaps`` fits a Sim(3) **including
scale** for every submap, and it fits that scale from the camera centres
-- local baselines onto GPS baselines. So scaling a submap's points *and*
its camera translations together, about any common origin, is a pure
similarity of the window, and the merge's own scale fit cancels it
exactly. Measured: a run that snapped all 19 windows onto one scale and a
run that left every window at its own produced byte-identical output.

What actually has to change is the ground's depth *below its cameras*,
which is what a depth-scale error really is. Scaling each point about the
camera that saw it does that and leaves every baseline untouched, so the
merge has nothing to cancel -- exactly how ``depth_anchor`` applies its
own correction. Camera translations are therefore deliberately left
alone here.

What it does not do
-------------------
It does not touch rotation, translation or the camera track, and it does
not deform a window: within a view, every point is scaled by the same
factor about the same centre.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ScaleConsensus", "consensus_ratio", "harmonise_submap_scales"]

#: A window whose own ratio is within this factor of the consensus keeps
#: it, rather than being rescaled.
#:
#: This defaults to 1.0 -- every window is snapped to the consensus --
#: because any tolerance here is paid for in metres. A window kept at a
#: ratio ``t`` times the consensus puts its ground ``(t - 1) * D`` out,
#: where ``D`` is the distance to that ground: at 120 m survey altitude a
#: seemingly-tight 10% tolerance permits 12 m of disagreement between
#: neighbouring windows, which is an order of magnitude past the metre
#: accuracy this pipeline is built for. Measured directly: with a 10%
#: tolerance, six windows whose ratios spanned 1.16-6.7 came out 5.0 m
#: apart after harmonising; snapping all six put them under a centimetre.
#:
#: There is no accuracy left on the table by snapping, because the thing
#: being measured -- how wrong the backbone's metric depth is on this
#: footage -- is genuinely one number (see the module docstring). A
#: window's deviation from the consensus is its own measurement error,
#: not information about that window.
_KEEP_TOLERANCE = 1.0

#: Minimum windows with a usable estimate before a consensus is believed.
#: Below this there is no majority to be robust against, and overriding a
#: window with the median of two others is not an improvement.
_MIN_WINDOWS = 3

#: Largest max/min spread in the per-window estimates that may still be
#: collapsed to a single number.
#:
#: This is the guard the first version of this module lacked, and it
#: matters more than the consensus itself. A median summarises a sample
#: only when the sample is one cluster. Measured on the sample flight:
#: four windows estimated 1.16-1.33 and twelve estimated 5.3-6.8, at a
#: constant 119.9 m altitude -- two clusters a factor of five apart, with
#: each window's own estimate internally tight (parallax sigma ~2%). The
#: median of that is 6.07, which is not a compromise between the clusters
#: but simply the larger one; snapping the small-ratio windows onto it
#: pushed their ground 32-41 m below where GPS says it is, while the
#: windows already near 6.07 landed within 3-7 m.
#:
#: A spread this wide is a real, structural inconsistency in the
#: backbone's output that has to be reported and fixed at its source.
#: Averaging it away destroys the windows that were right.
_MAX_CONSENSUS_SPREAD = 1.5


class ScaleConsensus:
    """The agreed ratio, and the evidence for it."""

    def __init__(self, ratio: float, source: str, samples: list[float], spread: float) -> None:
        self.ratio = ratio
        self.source = source
        self.samples = samples
        self.spread = spread

    def as_dict(self) -> dict:
        return {
            "consensus_ratio": round(self.ratio, 4),
            "source": self.source,
            "windows_measured": len(self.samples),
            "input_spread": round(self.spread, 3),
        }


def _window_estimates(diags: dict[int, dict]) -> tuple[dict[int, float], dict[int, float], str]:
    """Per-window (gps, parallax) ratio estimates from the anchor diagnostics."""
    gps: dict[int, float] = {}
    par: dict[int, float] = {}
    for index, diag in (diags or {}).items():
        if not isinstance(diag, dict):
            continue
        g = (diag.get("gps_altitude") or {}).get("ratio")
        p = (diag.get("parallax") or {}).get("ratio")
        if g is not None:
            gps[int(index)] = float(g)
        if p is not None:
            par[int(index)] = float(p)
    source = "gps_altitude" if len(gps) >= _MIN_WINDOWS else "parallax"
    return gps, par, source


def consensus_ratio(diags: dict[int, dict]) -> ScaleConsensus | None:
    """One depth scale for the whole flight, or ``None`` when unmeasurable.

    Built from the GPS-altitude estimates when enough windows have one
    (see the module docstring for why those are preferred for this job),
    falling back to parallax otherwise.
    """
    gps, par, source = _window_estimates(diags)
    chosen = gps if source == "gps_altitude" else par
    if len(chosen) < _MIN_WINDOWS:
        # Try the other one before giving up -- a flight with no altitude
        # column still deserves a consensus if parallax measured enough
        # windows, and vice versa.
        other = par if source == "gps_altitude" else gps
        if len(other) < _MIN_WINDOWS:
            return None
        chosen, source = other, ("parallax" if source == "gps_altitude" else "gps_altitude")

    values = np.asarray(sorted(chosen.values()), dtype=np.float64)
    ratio = float(np.median(values))
    spread = float(values.max() / max(values.min(), 1e-9))

    # Refuse rather than average two clusters together. See
    # _MAX_CONSENSUS_SPREAD -- doing this anyway is the specific mistake
    # that made a run worse instead of better.
    if spread > _MAX_CONSENSUS_SPREAD:
        logger.warning(
            "SCALE CONSENSUS REFUSED: the %d windows' own depth-scale estimates span %.2fx "
            "(%.3f to %.3f) from %s. That is not measurement noise around one number, it is the "
            "backbone returning genuinely different scales for different windows of the same "
            "flight -- and the windows that measured it correctly would be destroyed by being "
            "pulled onto the median. Each window keeps its own scale; the inconsistency is "
            "reported instead of hidden.",
            len(values),
            spread,
            float(values.min()),
            float(values.max()),
            source,
        )
        return None

    return ScaleConsensus(ratio, source, values.tolist(), spread)


def harmonise_submap_scales(
    submaps: list,
    diags: dict[int, dict],
    *,
    tolerance: float = _KEEP_TOLERANCE,
) -> dict:
    """Rescale outlier submaps onto the consensus. Mutates ``submaps`` in place.

    Each submap is scaled about its own ``local_origin`` (points and
    camera translations alike), which keeps the operation a similarity
    transform of that window and leaves the merge free to adjust it. A
    window already within ``tolerance`` of the consensus is left exactly
    as it was.

    Returns a diagnostics dict for the report card: the consensus, its
    evidence, and which windows were corrected by how much.
    """
    gps, par, _source = _window_estimates(diags)
    measured = gps or par
    consensus = consensus_ratio(diags)
    if consensus is None:
        spread = None
        if len(measured) >= 2:
            values = np.asarray(list(measured.values()), dtype=np.float64)
            spread = round(float(values.max() / max(values.min(), 1e-9)), 3)
        reason = (
            f"per-window estimates span {spread}x, past the {_MAX_CONSENSUS_SPREAD}x a single "
            "number can describe"
            if spread is not None and spread > _MAX_CONSENSUS_SPREAD
            else "too few windows measured a depth ratio"
        )
        logger.info("scale consensus: not applied -- %s; leaving every window at its own scale", reason)
        return {"applied": False, "reason": reason, "input_spread": spread}

    applied_by_window = {
        int(i): float(d["window_ratio"])
        for i, d in (diags or {}).items()
        if isinstance(d, dict) and d.get("applied") and d.get("window_ratio")
    }

    corrections: list[dict] = []
    for submap in submaps:
        index = getattr(getattr(submap, "window", None), "index", None)
        if index is None:
            continue
        applied = applied_by_window.get(int(index))
        if applied is None or applied <= 0:
            # The window was never anchored, so it still carries the
            # backbone's raw scale and the consensus is exactly the
            # correction it missed.
            correction = consensus.ratio
            reason = "unanchored"
        else:
            correction = consensus.ratio / applied
            reason = "outlier"
            if 1.0 / tolerance <= correction <= tolerance:
                continue

        if not _scale_submap(submap, correction):
            continue
        corrections.append(
            {
                "window": int(index),
                "was": None if applied is None else round(applied, 4),
                "now": round(consensus.ratio, 4),
                "correction": round(correction, 4),
                "reason": reason,
            }
        )

    if corrections:
        kept = (
            "every window is snapped onto it"
            if tolerance <= 1.0
            else f"windows within {100.0 * (tolerance - 1.0):.0f}% of it keep their own"
        )
        logger.warning(
            "SCALE CONSENSUS: the flight agrees on a depth scale of %.3f (from %s over %d windows, "
            "whose own estimates spanned %.2fx); %d/%d windows have been rescaled onto it and %s. "
            "Windows at different scales reconstruct the same ground at different heights, which no "
            "merge can correct.",
            consensus.ratio,
            consensus.source,
            len(consensus.samples),
            consensus.spread,
            len(corrections),
            len(submaps),
            kept,
        )
        for c in corrections:
            logger.info("  window %d: %s -> %.3f (x%.3f, %s)", c["window"], c["was"], c["now"], c["correction"], c["reason"])
    else:
        logger.info("scale consensus: every window was already at %.3f; nothing rescaled", consensus.ratio)

    return {
        "applied": True,
        **consensus.as_dict(),
        "tolerance": tolerance,
        "windows_rescaled": len(corrections),
        "corrections": corrections,
    }


def _scale_submap(submap, factor: float) -> bool:
    """Scale a submap's ground depth about the cameras. Returns whether it applied.

    Each point moves along the ray from the camera that saw it, by
    ``factor``. Camera centres do not move, so every baseline -- and
    therefore the scale ``merge_submaps`` fits from those baselines --
    is untouched, and the correction survives the merge. See the module
    docstring for the measurement that forced this.

    Needs ``Submap.view_index`` to know which camera saw each point.
    Submaps built before that field existed carry ``None``, and are
    refused rather than scaled about some arbitrary stand-in: a wrong
    centre turns a depth correction into a translation of the window.
    """
    if not np.isfinite(factor) or factor <= 0:
        return False

    view_index = getattr(submap, "view_index", None)
    points = np.asarray(submap.points.xyz, dtype=np.float64)
    shape = points.shape
    flat = points.reshape(-1, 3)

    if view_index is None or len(submap.poses) == 0:
        logger.warning(
            "scale consensus: a submap has no per-point view index, so its points cannot be "
            "scaled about the cameras that saw them; leaving it at its own scale"
        )
        return False

    view_index = np.asarray(view_index).reshape(-1)
    if view_index.shape[0] != flat.shape[0]:
        logger.warning(
            "scale consensus: view index has %d entries for %d points; leaving this submap alone",
            view_index.shape[0],
            flat.shape[0],
        )
        return False

    centres = np.array([np.asarray(p.t, dtype=np.float64) for p in submap.poses])
    per_point_centre = centres[np.clip(view_index, 0, len(centres) - 1)]
    scaled = per_point_centre + factor * (flat - per_point_centre)
    submap.points = replace(submap.points, xyz=scaled.reshape(shape))

    # Covariance is a squared quantity: scaling by s takes Cov to
    # s^2 * Cov. Skipping this would leave a rescaled window claiming its
    # old, now-wrong uncertainty.
    if getattr(submap.points, "covariance", None) is not None:
        cov = np.asarray(submap.points.covariance, dtype=np.float64)
        submap.points = replace(submap.points, covariance=cov * (factor**2))
    return True
