"""Submap window planning over triage's keyframe list.

6 GB of VRAM cannot hold a whole flight's worth of frames through a
transformer backbone at once (see the module docstring in
``geometry.mapanything`` and the sweep results ``scripts/benchmark_vram.py``
produces). We instead slide a bounded, overlapping window over the
``list[Keyframe]`` triage produced, run the backbone once per window, and
stitch the resulting per-window "submaps" back together in
``geometry.submap.merge_submaps``.

Why overlap matters (and why it can't be tiny)
-----------------------------------------------
A single drone pass is a *strip*: consecutive keyframes' camera centres lie
close to one line (the flight track), and that near-collinearity is exactly
the configuration in which fitting a full Sim(3) transform (rotation +
translation + scale) is numerically unstable. Camera centres alone pin down
translation and scale well (they clearly move along the strip), but they
barely constrain rotation *about* the strip's own axis (roll) or rotation
*in* the direction of travel (pitch) -- a small rotation about either axis
moves the centres almost nowhere. The practical symptom, if this is not
respected, is a merged model whose camera centres line up with GPS
beautifully while the point cloud itself comes out visibly tilted: the
alignment "looks right" by the metric that's easy to check (centre
positions) and is wrong on the metric that's hard to check by eye (absolute
orientation).

Two things fight this:

1. **More shared cameras per junction** (this module): each extra shared
   keyframe between consecutive windows is another constraint on the Sim(3)
   fit. ``plan_windows`` refuses fewer than 2 frames of overlap outright,
   and defaults to ~30% of the window size, because 1-2 shared points is
   barely enough to fit anything (3 non-collinear points are the true
   minimum for Umeyama; RANSAC over more than that is what actually buys
   robustness -- see ``geometry.submap``).
2. **External orientation constraints** (not this module): IMU gravity
   (which pins down roll/pitch directly, independent of any camera-centre
   geometry) and/or GPS heading, applied downstream once submaps are
   merged. Overlap alone reduces but does not eliminate the degeneracy;
   ``geometry.submap.merge_submaps`` reports a ``degenerate`` flag rather
   than silently trusting a numerically unreliable rotation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drishti3d.types import Keyframe

# Overlap must cover at least this many shared keyframes -- Umeyama's Sim(3)
# fit is only well-posed with >= 3 non-collinear correspondences, and 2
# shared frames is not enough margin for RANSAC to do anything useful with
# (see geometry.submap). This is a hard floor, not a default.
_MIN_OVERLAP = 2

# Default overlap fraction of `window_size` when the caller doesn't specify
# one -- ~30%, per the module docstring's reasoning about the near-collinear
# strip degeneracy needing real alignment margin, not just the bare minimum.
_DEFAULT_OVERLAP_FRACTION = 0.3


@dataclass
class Window:
    """One planned reconstruction window over the keyframe list.

    ``start``/``end`` are a Python half-open slice into the ``list
    [Keyframe]`` ``plan_windows`` was given (``keyframes[start:end]``).
    ``shared_with_previous`` holds the *global* keyframe-list indices
    (i.e. values in ``range(start, end)``, directly comparable across
    windows) that this window has in common with the immediately
    preceding window -- empty for the first window. These are the anchor
    frames ``geometry.submap.merge_submaps`` aligns consecutive submaps on.
    """

    index: int
    start: int
    end: int
    shared_with_previous: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.end - self.start

    def keyframe_indices(self) -> list[int]:
        """The global keyframe-list indices this window covers, in order."""
        return list(range(self.start, self.end))


def _extent_capped_size(keyframes: list[Keyframe], start: int, window_size: int, max_extent_m: float) -> int:
    """How many keyframes from ``start`` fit inside ``max_extent_m`` of flight.

    Never returns fewer than ``_MIN_OVERLAP + 1``: a window has to keep
    enough cameras for the merge's Sim(3) fit to be posed at all, so a
    burst of very widely-spaced keyframes is allowed to exceed the cap
    rather than produce a window nothing can align.
    """
    n = len(keyframes)
    floor = _MIN_OVERLAP + 1
    positions = []
    for i in range(start, min(start + window_size, n)):
        pose = getattr(keyframes[i], "pose", None)
        positions.append(None if pose is None else np.asarray(pose.t, dtype=np.float64))
    if any(p is None for p in positions) or len(positions) < 2:
        return min(window_size, n - start)

    origin = positions[0]
    count = 1
    for p in positions[1:]:
        if float(np.linalg.norm(p - origin)) > max_extent_m:
            break
        count += 1
    return max(floor, min(count, window_size, n - start))


def plan_windows(
    keyframes: list[Keyframe],
    window_size: int,
    overlap: int | None = None,
    max_windows: int | None = None,
    max_extent_m: float | None = None,
) -> list[Window]:
    """Plan overlapping windows covering every keyframe in ``keyframes``.

    Parameters
    ----------
    keyframes:
        The full triage output. Its length sets the range to cover; its
        poses are read only when ``max_extent_m`` is given.
    max_extent_m:
        Cap on how far the camera may travel within one window, in metres.
        ``None`` keeps the original count-only behaviour.

        This exists because a feed-forward multi-view backbone's metric
        depth degrades badly as the cameras spread out. Measured on the
        sample flight, with real telemetry poses and the same 8 views,
        against a true ground distance of 120 m::

            camera extent  48 m  ->  depth 78 m
            camera extent 148 m  ->  depth 16 m

        Keyframes on that flight are ~6.5 m apart at the start and ~33 m
        apart later, so a fixed 8-keyframe window spans 48 m early and
        148 m later -- and the windows split exactly that way in the
        output: windows 0-3 returned 90-105 m of depth, windows 5-18
        returned 17-23 m.

        Depth that comes back 7x too shallow has to be scaled up 7x to
        become metric, which scales its *error* up 7x too. That is what
        made the reconstructed ground 12 m thick inside a single 2 m
        cell. Capping the extent keeps the backbone in the regime where
        its depth is roughly right, so the correction -- and therefore
        the amplification -- stays small.
    window_size:
        Target number of keyframes per window (the last window may be
        smaller if ``len(keyframes)`` doesn't divide evenly).
    overlap:
        Number of keyframes consecutive windows share. Must be
        ``>= 2`` (``_MIN_OVERLAP``) whenever it constrains anything;
        defaults to ``round(0.3 * window_size)``, floored at 2.
    max_windows:
        Optional cap on the number of windows returned (e.g. for a quick
        dev/test run over a long flight). When given and reached, the
        returned windows will not necessarily cover every keyframe -- this
        is a deliberate escape hatch, not the default behaviour.

    Returns
    -------
    A list of ``Window``, ordered by ``start`` (and by ``index``), whose
    union of ``[start, end)`` ranges covers ``range(len(keyframes))``
    exactly (unless truncated by ``max_windows``).
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    n = len(keyframes)
    if n == 0:
        return []

    if overlap is None:
        overlap = max(_MIN_OVERLAP, round(_DEFAULT_OVERLAP_FRACTION * window_size))
    if overlap < _MIN_OVERLAP:
        raise ValueError(
            f"overlap must be >= {_MIN_OVERLAP} frames (Sim(3) alignment in "
            "geometry.submap needs >= 3 shared cameras to be well-posed at "
            f"all, and 2 is already a thin margin for RANSAC) -- got {overlap}"
        )

    # Fewer keyframes than one window, or exactly one window's worth: a
    # single window covering everything, no overlap possible or needed.
    if n <= window_size:
        return [Window(index=0, start=0, end=n, shared_with_previous=[])]

    if overlap >= window_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than window_size ({window_size})"
        )

    windows: list[Window] = []
    start = 0
    idx = 0
    while True:
        # Each window is sized independently when an extent cap is in
        # force: keyframe spacing varies several-fold across one flight,
        # so a single count cannot hold the camera extent steady.
        size = window_size
        if max_extent_m is not None:
            size = _extent_capped_size(keyframes, start, window_size, max_extent_m)
        this_overlap = min(overlap, size - 1)
        end = min(start + size, n)
        prev = windows[-1] if windows else None
        shared = list(range(max(start, prev.start), min(end, prev.end))) if prev is not None else []
        windows.append(Window(index=idx, start=start, end=end, shared_with_previous=shared))

        if end >= n:
            break
        if max_windows is not None and len(windows) >= max_windows:
            break

        idx += 1
        # Always advance, even if the cap shrank this window to its floor:
        # a step of zero would plan windows forever.
        start += max(1, size - this_overlap)

    return windows


# ---------------------------------------------------------------------------
# VRAM estimation
#
# *** CALIBRATION NEEDED ***
# The constants below are analytically-motivated placeholders, not
# measurements -- there is no CUDA device on the development machine this
# module was written on, so nothing here has been fitted to a real run.
# The functional form (fixed model weights + a per-view linear term for
# per-view activations/point-map buffers + a per-(view, view)-pair
# quadratic term for cross-view attention) is what we expect a
# MapAnything-style multi-view transformer's memory profile to look like;
# the coefficients are guesses sized to be roughly the right order of
# magnitude for a ~1B-parameter model at 518px. Replace every constant in
# this block with numbers fitted to a real ``scripts/benchmark_vram.py``
# sweep on the target RTX 4060 (or a Colab/Kaggle T4 as a stand-in) before
# trusting `estimate_memory` / `max_window_for_budget` for anything beyond
# a rough planning order-of-magnitude.
# ---------------------------------------------------------------------------

_EST_MODEL_WEIGHTS_GB = 2.2
"""Resident weights + fixed inference overhead (~1B params, fp16/bf16). NEEDS CALIBRATION."""

_EST_PER_VIEW_GB_AT_REFERENCE = 0.35
"""Per-view activation/point-map memory at `_EST_REFERENCE_SIZE`px. NEEDS CALIBRATION."""

_EST_PER_PAIR_GB_AT_REFERENCE = 0.015
"""Per-(view, view)-pair cross-attention memory at `_EST_REFERENCE_SIZE`px. NEEDS CALIBRATION."""

_EST_REFERENCE_SIZE = 518
"""Image size (long side, px) the two constants above were guessed at."""

_EST_SAFETY_MARGIN = 1.15
"""Multiplicative headroom for allocator fragmentation / bookkeeping overhead not modeled above."""


def estimate_memory(window_size: int, image_size: int) -> float:
    """Rough estimated peak VRAM (GB) for a backbone pass over ``window_size`` views at ``image_size`` px.

    See the calibration warning above this function: this is a placeholder
    formula, not a measurement. Use ``scripts/benchmark_vram.py`` for real
    numbers and refit the module-level ``_EST_*`` constants from its
    output.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {image_size}")

    scale = (image_size / _EST_REFERENCE_SIZE) ** 2
    per_view = _EST_PER_VIEW_GB_AT_REFERENCE * scale
    per_pair = _EST_PER_PAIR_GB_AT_REFERENCE * scale
    n_pairs = window_size * (window_size - 1) / 2.0

    total = _EST_MODEL_WEIGHTS_GB + window_size * per_view + n_pairs * per_pair
    return total * _EST_SAFETY_MARGIN


def max_window_for_budget(vram_gb: float, image_size: int) -> int:
    """Largest ``window_size`` whose ``estimate_memory`` fits within ``vram_gb``.

    Inverts ``estimate_memory`` by linear search from 1 upward -- window
    sizes of interest are small (single/low-double digits), so there is no
    need for anything cleverer than a direct search. Returns 0 if even a
    single view doesn't fit.
    """
    if vram_gb <= 0:
        return 0

    best = 0
    window_size = 1
    while estimate_memory(window_size, image_size) <= vram_gb:
        best = window_size
        window_size += 1
        if window_size > 100_000:  # pathological input safety valve
            break
    return best


# ---------------------------------------------------------------------------
# Window-size planning (Task 3: derive window_size from the memory budget
# and keyframe count, rather than a fixed constant).
#
# The old default (``GeometryConfig.window_size = 5``, ``overlap = 3``) gave
# a 29-keyframe flight 13 submaps and 12 junctions, each with the bare
# minimum 3 shared cameras -- needlessly many poorly-conditioned junctions
# for a GPU that (see this module's own docstring) can comfortably hold far
# more than 5 views at once. Larger windows both reduce the number of
# junctions (each a Sim(3) fit with real estimation noise, whether or not
# it's collinear-degenerate -- see geometry.submap's module docstring on
# chained-error compounding) AND cut the number of per-window backbone
# invocations (the dominant per-window cost), so there is no accuracy/speed
# trade-off pushing window_size down -- only the VRAM ceiling does.
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_WINDOW_SIZE = 14
"""Default target window size before any memory-budget capping.

Chosen to land in the "12-16 window / 4-6 shared" range this project's
brief calls for: at ~35% overlap (``_DEFAULT_OVERLAP_FRACTION`` and
``pipeline.stages.GeometryStage``'s own overlap floor) that is ~5 shared
cameras per junction, and for a real ~29-keyframe flight it collapses 13
old-default submaps down to ~3.
"""

_MIN_PLANNED_WINDOW_SIZE = 8
"""Floor ``plan_window_size`` will not go below even under a tight memory
budget. Distinct from (and looser than) ``pipeline.stages.GeometryStage``'s
own ``_MIN_VIABLE_WINDOW_SIZE`` (the strict floor ``merge_submaps``' >= 3
-shared-camera correspondence requirement imposes) -- this is the separate
"still a useful window, not a degenerate sliver" floor for the planner
itself, applied before that stricter downstream floor.
"""

_DEFAULT_VRAM_BUDGET_GB = 6.0
"""Matches this module's docstring ("6 GB of VRAM cannot hold a whole
flight...") -- the reference GPU class this project targets. Callers with a
known real budget (``scripts/benchmark_vram.py`` output, or a queried
device) should pass their own ``vram_budget_gb`` instead of relying on this
default.
"""


def plan_window_size(
    n_keyframes: int,
    image_size: int,
    vram_budget_gb: float = _DEFAULT_VRAM_BUDGET_GB,
    requested: int | None = None,
) -> int:
    """Pick a ``window_size`` from the memory budget and how many keyframes there are.

    Replaces treating ``window_size`` as a fixed constant (the old
    ``GeometryConfig.window_size = 5`` default, unconditionally): the
    right window size is a function of (a) how much memory a single
    backbone call over ``window_size`` views at ``image_size`` px actually
    costs (``max_window_for_budget``) and (b) how many keyframes there are
    to cover at all -- planning a 14-keyframe window for an 8-keyframe
    flight is pointless (``plan_windows`` already collapses that to one
    window, but asking for less than the whole flight when the whole
    flight already fits is not "deriving from the keyframe count", it's
    ignoring it).

    Parameters
    ----------
    n_keyframes:
        Total keyframes available. A window can never need to be bigger
        than this -- a single window covering everything needs no merge
        step at all.
    image_size:
        The backbone's actual input resolution (``GeometryConfig
        .max_image_size``) -- ``estimate_memory``'s per-view/per-pair cost
        scales with this.
    vram_budget_gb:
        Memory ceiling to plan within; defaults to
        ``_DEFAULT_VRAM_BUDGET_GB`` (this project's reference GPU class).
    requested:
        A caller-preferred window size (e.g. ``GeometryConfig
        .window_size``) to use as the starting point instead of
        ``_DEFAULT_TARGET_WINDOW_SIZE``, still capped by both the memory
        budget and ``n_keyframes``.

    Returns
    -------
    A window size that never exceeds ``n_keyframes`` or a caller-given
    ``requested``. ``_MIN_PLANNED_WINDOW_SIZE`` only comes into play when
    the *memory budget* (not an explicit small ``requested``, and not a
    genuinely small ``n_keyframes``) would otherwise push the window below
    it -- a caller that deliberately asks for a small window (e.g. a test
    fixture, or a short flight) gets exactly that, not a second-guessed
    floor it never asked to be capped away from.
    """
    if n_keyframes <= 0:
        return max(requested if requested is not None else _DEFAULT_TARGET_WINDOW_SIZE, 1)

    desired = requested if requested is not None else _DEFAULT_TARGET_WINDOW_SIZE
    desired = max(desired, 1)
    # Never plan a window bigger than the whole flight: covering every
    # keyframe in one window needs no merge step (and therefore no
    # junction-alignment degeneracy) at all. This also caps what the
    # memory-budget floor below is allowed to grow back up to.
    desired = min(desired, n_keyframes)

    # max_window_for_budget returning 0 means the budget can't even fit a
    # single view at this image_size -- still a real cap (tighter than any
    # positive one), not "no cap" (a prior version of this function treated
    # 0 as falsy and skipped the cap, silently ignoring an exhausted
    # budget).
    budget_cap = max_window_for_budget(vram_budget_gb, image_size)
    if budget_cap < desired:
        # The memory budget, not the caller's own request or the keyframe
        # count, is what's limiting here -- don't let it push the window
        # down to a barely-useful sliver; floor it back up to whichever is
        # smaller of _MIN_PLANNED_WINDOW_SIZE and what was actually wanted.
        desired = max(budget_cap, min(_MIN_PLANNED_WINDOW_SIZE, desired))

    return desired
