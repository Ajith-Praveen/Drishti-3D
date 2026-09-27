"""Video-native keyframe triage: the piece that differentiates this pipeline.

Fixed-stride sampling (every Nth frame, or every K seconds) is the obvious
thing to do and the wrong thing to do: how much a drone shot changes per
frame depends on ground speed, altitude, and gimbal motion, none of which
are constant across a single-pass flight. A slow, low-altitude pass over
one area needs far more temporal density than a fast, high-altitude
transit to get the same photogrammetric baseline between frames, and a
fixed stride either wastes reconstruction budget on near-duplicate frames
in the slow section or starves the fast section of overlap entirely.

Instead we accumulate a "how much real baseline has built up since the
last accepted keyframe" signal and accept a new keyframe exactly when
enough has built up. Frame spacing falls out of scene motion (or actual
flight distance) instead of clock time.

That signal is hybrid, in priority order:

1. **GPS-driven metric baseline** (``TriageConfig.use_gps_baseline``, on by
   default): when telemetry with position fixes is available, we track
   real distance travelled in a local ENU metre frame
   (``ingest.telemetry.telemetry_to_enu``) and trigger once it clears an
   altitude-aware target (``TriageConfig.min_baseline_m`` /
   ``baseline_to_altitude_ratio`` -- see ``TriageConfig``'s docstring for
   the photogrammetric reasoning). This is metric, camera-model-agnostic,
   and does not depend on scene depth structure at all, which matters
   because...
2. **Vision-only fallback** (no telemetry): a homography-residual-only
   parallax estimate has a fundamental blind spot -- pure camera rotation
   *and* translation over a planar/near-planar scene (the dominant
   nadir-drone-over-terrain case) are both exactly described by a single
   homography, so the residual collapses toward zero for both, starving
   the accumulator on flat terrain, precisely the common case. See
   ``triage.metrics.estimate_parallax_detailed`` /
   ``ParallaxEstimate.useful_baseline_px`` for the hybrid raw-displacement
   +residual+inlier-ratio heuristic used here instead, and its docstring
   for why vision alone can't perfectly resolve that ambiguity (which is
   exactly why GPS is preferred whenever it's available).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from drishti3d.config import TriageConfig
from drishti3d.ingest.telemetry import (
    resample_telemetry,
    telemetry_to_enu,
    trajectory_length,
)
from drishti3d.ingest.video import VideoSource, downscale_image
from drishti3d.triage.metrics import (
    blur_score,
    estimate_parallax_detailed,
    exposure_score,
    mean_luma,
)
from drishti3d.types import (
    CameraIntrinsics,
    Frame,
    FrameMetrics,
    Keyframe,
    TelemetrySample,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

# Working resolution for every triage metric (scan pass and window re-check
# alike). Keeping this one constant for the whole module is what makes
# config.min_blur_score / config.min_parallax_px meaningful thresholds:
# both are calibrated against this resolution, not the source video's.
_SCAN_DOWNSCALE_PX = 640

# Local window (in native frame indices, each side) searched for the
# sharpest candidate once the parallax/baseline trigger fires.
_MIN_WINDOW_RADIUS = 3

# Ring buffer sizing (DEFECT 3 fix): must comfortably outlive a full
# +/-radius window plus the checkpoint-to-checkpoint gap between triggers,
# with slack for the lookahead extension in _pick_sharpest_in_window. Each
# entry is one small downscaled grayscale frame, so even a generous buffer
# is cheap.
_RING_BUFFER_SLACK = 32
_RING_BUFFER_MULTIPLIER = 8

# Minimum number of buffered blur samples before trusting a running-median
# relative floor (DEFECT 4 fix); below this we fall back to the absolute
# floor only, since a median of 1-2 samples isn't a meaningful "typical
# sharpness for this video" estimate yet.
_MIN_BUFFER_FOR_RELATIVE_BLUR = 5

# Adaptive-threshold behavior (see _maybe_adapt_threshold): don't react to
# noise in the first handful of keyframes, and grow geometrically rather
# than jumping straight to a "correct" value we can't actually compute in
# one shot from a streaming pass.
_ADAPT_CHECK_MIN_KEYFRAMES = 5
_ADAPT_OVERSHOOT_FACTOR = 1.5
_ADAPT_GROWTH = 1.25

# A single buffered/decoded frame: (native frame index, timestamp, small
# grayscale image, blur score). Shared shape between the main scan loop and
# the window search so the two can hand entries back and forth freely.
_BufferEntry = tuple[int, float, np.ndarray, float]


def _frame_gray(frame: Frame) -> np.ndarray:
    assert frame.image is not None
    return cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)


class _LazyEntry:
    """A buffer entry that converts and blur-scores its frame only when read.

    Behaves as the ``(index, timestamp, gray, blur)`` tuple every other
    function here expects -- indexing and unpacking both work -- so the
    window search and blur floor are unchanged. In GPS-baseline mode the
    trigger needs only timestamps, so most frames are never converted.
    """

    __slots__ = ("_lazy", "_gray", "_blur")

    def __init__(self, lazy) -> None:
        self._lazy = lazy
        self._gray: np.ndarray | None = None
        self._blur: float | None = None

    def _ensure(self) -> None:
        if self._gray is None:
            self._gray = _frame_gray(self._lazy.small(_SCAN_DOWNSCALE_PX))
            self._blur = blur_score(self._gray)

    def __getitem__(self, i: int):
        if i == 0:
            return self._lazy.index
        if i == 1:
            return self._lazy.timestamp
        self._ensure()
        return self._gray if i == 2 else self._blur

    def __iter__(self):
        return iter((self[0], self[1], self[2], self[3]))

    def __len__(self) -> int:
        return 4

    def bgr24(self) -> np.ndarray:
        return self._lazy.bgr24()


def _eager_entry(frame: Frame) -> _BufferEntry:
    gray = _frame_gray(frame)
    return (frame.index, frame.timestamp, gray, blur_score(gray))


def _coerce_samples(
    telemetry: list[TelemetrySample] | tuple[list[TelemetrySample], dict] | None,
) -> list[TelemetrySample] | None:
    """Accept either a bare sample list or the ``(samples, stats)`` tuple ``load_telemetry`` returns."""
    if telemetry is None:
        return None
    if isinstance(telemetry, tuple) and len(telemetry) == 2 and isinstance(telemetry[1], dict):
        return telemetry[0]
    return telemetry


# ---------------------------------------------------------------------------
# GPS-driven baseline (DEFECT 2a)
# ---------------------------------------------------------------------------


def _prepare_enu_lookup(
    samples: list[TelemetrySample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """One-time prep for cheap per-checkpoint ENU position interpolation.

    Returns ``(sorted_timestamps, enu_xyz, alt_rel_or_None)`` restricted to
    samples that actually carry a geo fix, or ``None`` if fewer than two
    such samples exist (not enough to interpolate a trajectory from).

    This deliberately duplicates a slice of what
    ``ingest.telemetry.resample_telemetry`` does rather than calling it
    per-checkpoint: that function re-sorts and re-projects the *entire*
    telemetry list on every call, which would be wasteful when we just need
    a cheap position lookup once per checkpoint frame during the scan.
    """
    geo_samples = [s for s in samples if s.geo is not None]
    if len(geo_samples) < 2:
        return None
    ordered = sorted(geo_samples, key=lambda s: s.timestamp)
    enu, _origin = telemetry_to_enu(ordered)
    t_arr = np.array([s.timestamp for s in ordered], dtype=np.float64)
    alt_rel = np.array(
        [s.geo.alt_rel if s.geo is not None and s.geo.alt_rel is not None else np.nan for s in ordered],
        dtype=np.float64,
    )
    alt_rel_out = None if np.all(np.isnan(alt_rel)) else alt_rel
    return t_arr, enu, alt_rel_out


def _interp_enu_position(
    lookup: tuple[np.ndarray, np.ndarray, np.ndarray | None], timestamp: float
) -> tuple[np.ndarray | None, float | None]:
    """Interpolate ENU xyz (metres) and relative altitude (metres) at ``timestamp``.

    Returns ``(None, None)`` outside the telemetry's covered time range --
    no extrapolation, matching ``resample_telemetry``'s policy (an
    extrapolated position is worse than none for a distance-accumulation
    signal, since it can be confidently wrong).
    """
    t_arr, enu, alt_rel = lookup
    if timestamp < t_arr[0] or timestamp > t_arr[-1]:
        return None, None
    xyz = np.array([np.interp(timestamp, t_arr, enu[:, k]) for k in range(3)])
    alt = float(np.interp(timestamp, t_arr, alt_rel)) if alt_rel is not None else None
    return xyz, alt


def _turn_rate_deg_s(lookup, timestamp: float, half_window_s: float = 1.0) -> float | None:
    """GPS course change rate (deg/s) around ``timestamp``; ``None`` when not measurable.

    Course from the track over ``[t - h, t]`` versus ``[t, t + h]``. Slow
    or hovering stretches (under 2 m moved per half-window) have no
    meaningful course and report ``None`` rather than noise.
    """
    a, _ = _interp_enu_position(lookup, timestamp - half_window_s)
    b, _ = _interp_enu_position(lookup, timestamp)
    c, _ = _interp_enu_position(lookup, timestamp + half_window_s)
    if a is None or b is None or c is None:
        return None
    d1, d2 = (b - a)[:2], (c - b)[:2]
    if np.linalg.norm(d1) < 2.0 or np.linalg.norm(d2) < 2.0:
        return None
    ang = np.degrees(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], float(d1 @ d2)))
    return abs(float(ang)) / half_window_s


# ---------------------------------------------------------------------------
# Ring-buffered sharpest-frame window search (DEFECT 3)
# ---------------------------------------------------------------------------


def _blur_floor(buffer: deque[_BufferEntry], absolute_min: float, relative_threshold: float) -> float:
    """Combine the absolute hard floor with a running-median relative floor.

    DEFECT 4 fix: ``blur_score`` mixes two differently-scaled terms, so an
    absolute default threshold does not transfer across cameras,
    resolutions, or scenes. The real gate is relative -- reject candidates
    below ``relative_threshold`` times the running median blur score
    observed so far in *this* video (approximated here via the current
    ring buffer, per the brief's suggestion, rather than an unbounded
    running statistic). ``absolute_min`` stays as a low-default hard floor
    only for pathological cases (e.g. a corrupt/black frame).
    """
    if len(buffer) < _MIN_BUFFER_FOR_RELATIVE_BLUR:
        return absolute_min
    running_median = float(np.median([entry[3] for entry in buffer]))
    return max(absolute_min, relative_threshold * running_median)


def _extend_buffer_to(
    frame_iter: Iterator[Frame], buffer: deque[_BufferEntry], target_index: int, make_entry=_eager_entry
) -> None:
    """Pull more frames from the sequential decode iterator until the buffer reaches ``target_index``.

    The window search looks ahead of the trigger point
    (``center_index + radius``), but a streaming forward decode has, by
    construction, not necessarily decoded that far yet. Pulling more
    frames here is still a plain sequential ``next()`` on the same
    iterator the main scan loop is consuming -- never a seek -- which is
    the whole point of the DEFECT 3 fix. Stops once the buffer's newest
    entry reaches ``target_index`` or the stream ends.
    """
    last_index = buffer[-1][0] if buffer else None
    while last_index is None or last_index < target_index:
        frame = next(frame_iter, None)
        if frame is None:
            return
        buffer.append(make_entry(frame))
        last_index = frame.index


def _pick_sharpest_in_window(
    video: VideoSource,
    frame_iter: Iterator[Frame],
    buffer: deque[_BufferEntry],
    center_index: int,
    radius: int,
    absolute_min_blur: float,
    relative_blur_threshold: float,
    min_index_exclusive: int = -1,
    make_entry=_eager_entry,
) -> _BufferEntry | None:
    """Among frames near ``center_index``, return the sharpest one that clears the blur floor.

    The parallax/baseline trigger fires on a checkpoint frame, which is not
    necessarily the best frame to actually keep -- motion blur is common
    right around a fast pan or gimbal correction, exactly when baseline is
    also accumulating quickly. Historically this was resolved by
    random-access re-decoding a native-resolution window on every trigger
    (thousands of seek+decode operations over a full video -- see DEFECT 3
    in the triage brief). Instead, we search the bounded ring ``buffer`` of
    already-decoded, already-downscaled frames the streaming scan has been
    filling in, extending it forward (still sequential decode, never a
    seek) if the window reaches past what's buffered so far.

    ``min_index_exclusive`` (normally the caller's ``last_keyframe_index``)
    excludes any candidate at or before an already-accepted frame from
    consideration. This matters beyond just the checkpoint-recency check
    ``select_keyframes`` already does: this window search covers
    ``[center_index - radius, center_index + radius]``, and the *chosen*
    frame from one trigger can land ahead of its own checkpoint (anywhere
    in that window) -- so the very next checkpoint, only a few native
    frames later, is easily non-stale by index while its own search window
    still overlaps the previous one and can re-select the *same* already
    -accepted frame as "sharpest" again. Filtering the candidate pool here,
    not just gating whether to search at all, is what actually prevents
    that duplicate (see ``select_keyframes``' bug report: a GPS-driven
    baseline in particular can clear its trigger threshold within a single
    frame's motion, so consecutive checkpoints' windows overlap constantly).

    Falls back to ``video.read_frames`` -- the old random-access path --
    only if nothing buffered clears the floor, which should be rare (the
    buffer normally has dense enough coverage of the window already).
    Returns ``None`` if nothing anywhere clears the floor, so the caller
    can defer acceptance.
    """
    lo = max(0, center_index - radius, min_index_exclusive + 1)
    hi = center_index + radius
    _extend_buffer_to(frame_iter, buffer, hi, make_entry)

    floor = _blur_floor(buffer, absolute_min_blur, relative_blur_threshold)

    best: _BufferEntry | None = None
    for entry in buffer:
        index = entry[0]
        if index < lo or index > hi:
            continue
        blur = entry[3]
        if blur < floor:
            continue
        if best is None or blur > best[3]:
            best = entry

    if best is not None:
        return best

    if lo > hi:
        # Every candidate index in the nominal window is already covered
        # by a previously-accepted keyframe -- nothing legitimately new to
        # search for (not even via the read_frames fallback below).
        return None

    logger.debug("triage: ring buffer had no candidate above blur floor near frame %d; falling back to read_frames", center_index)
    frames = video.read_frames(list(range(lo, hi + 1)))
    fallback_best: _BufferEntry | None = None
    for frame in frames:
        if frame.image is None:
            continue
        small = downscale_image(frame.image, _SCAN_DOWNSCALE_PX)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = blur_score(gray)
        if blur < floor:
            continue
        if fallback_best is None or blur > fallback_best[3]:
            fallback_best = (frame.index, frame.timestamp, gray, blur)

    return fallback_best


def _maybe_adapt_threshold(
    effective_threshold: float,
    config: TriageConfig,
    n_keyframes: int,
    total_frames: int,
    current_frame_index: int,
    progress_cb: ProgressCallback | None,
) -> float:
    """Raise the spacing trigger if the current acceptance rate implies wildly overshooting the target.

    Unit-agnostic: ``effective_threshold`` is either ``min_parallax_px``
    (vision mode) or ``min_baseline_m`` (GPS mode) depending on which
    signal ``select_keyframes`` is currently driving off; the scaling logic
    is identical either way.

    A fixed-stride sampler can't overshoot its target count -- the count is
    fixed by construction. Ours can: a slow, hovering, or close-to-ground
    stretch of the flight trips the trigger far more often per second of
    footage than a fast high-altitude pass would, and we don't know in
    advance how much of the video looks like which. We extrapolate the
    current acceptance rate (keyframes accepted / fraction of video
    scanned so far) to the end of the video; if that projects well past
    ``target_keyframes``, we scale the threshold up so subsequent triggers
    require more actual baseline, self-correcting the rate rather than
    truncating the video early once ``target_keyframes`` is hit.
    """
    if n_keyframes < _ADAPT_CHECK_MIN_KEYFRAMES or total_frames <= 0 or current_frame_index <= 0:
        return effective_threshold

    progress_fraction = current_frame_index / total_frames
    if progress_fraction <= 0:
        return effective_threshold

    projected_total = n_keyframes / progress_fraction
    if projected_total <= config.target_keyframes * _ADAPT_OVERSHOOT_FACTOR:
        return effective_threshold

    new_threshold = effective_threshold * _ADAPT_GROWTH
    message = (
        f"triage: adapting spacing threshold {effective_threshold:.3f} -> {new_threshold:.3f} "
        f"(projected ~{projected_total:.0f} keyframes vs target {config.target_keyframes})"
    )
    logger.info(message)
    if progress_cb is not None:
        progress_cb(current_frame_index, total_frames, message)
    return new_threshold


def select_keyframes(
    video: VideoSource,
    config: TriageConfig,
    telemetry: list[TelemetrySample] | tuple[list[TelemetrySample], dict] | None = None,
    progress_cb: ProgressCallback | None = None,
    intrinsics: CameraIntrinsics | None = None,
    full_res_sink: dict[int, np.ndarray] | None = None,
    diag_out: dict | None = None,
) -> tuple[list[Keyframe], list[FrameMetrics]]:
    """Select a baseline-driven subset of frames worth reconstructing from.

    Single streaming pass, sequential decode throughout (see DEFECT 3 in
    the triage brief this implements -- ``max_frames_scanned`` now bounds
    the number of *checkpoint* evaluations, decoupled from decode, which is
    cheap enough to run dense):

    1. Decode frames one at a time (downscaled), pushing each into a
       bounded ring buffer alongside its blur score.
    2. Every ``scan_step``-th decoded frame is a "checkpoint": accumulate
       either real GPS-derived metric baseline (preferred, when telemetry
       is available) or a vision-only "useful baseline" signal since the
       last accepted keyframe (see module docstring).
    3. Once the accumulated signal clears the (possibly adapted, possibly
       altitude-scaled) target, search the ring buffer -- extending it
       forward with more sequential decode if needed, never a seek -- for
       the sharpest frame in a small window around the checkpoint that
       clears the relative blur floor; defer (don't reset the accumulator)
       if nothing qualifies.
    4. Stop at ``config.target_keyframes``, or when the video runs out.
    5. Attach interpolated telemetry to each keyframe, if telemetry was given.

    ``telemetry`` accepts either a bare ``list[TelemetrySample]`` or the
    ``(samples, stats)`` tuple ``ingest.telemetry.load_telemetry`` returns.
    ``progress_cb(current_frame_index, total_frames, message)`` is called
    on every checkpoint and on every acceptance/deferral/adaptation event,
    so a GUI can show live progress and explain *why* the keyframe count is
    what it is. ``intrinsics``, when given, is threaded through to the
    vision-only fallback's ``estimate_parallax_detailed`` call so it can run
    the principled rotation-vs-plane-translation test (see
    ``triage.metrics.ParallaxEstimate.useful_baseline_px``) instead of the
    cruder raw/residual/inlier heuristic -- unused in GPS-baseline mode,
    which doesn't need a camera model at all.
    """
    samples = _coerce_samples(telemetry)

    enu_lookup: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None
    use_gps = False
    if samples and config.use_gps_baseline:
        enu_lookup = _prepare_enu_lookup(samples)
        use_gps = enu_lookup is not None

    # Overlap-driven spacing (triage.footprint): measure how much ground one
    # image covers along the track and space keyframes for
    # `forward_overlap` of it. Falls back to the altitude rule below when
    # the footprint cannot be measured.
    footprint = None
    overlap = float(getattr(config, "forward_overlap", 0.0) or 0.0)
    if use_gps and enu_lookup is not None and 0.0 < overlap < 1.0 and hasattr(video, "read_frames"):
        from drishti3d.triage.footprint import measure_footprint

        try:
            footprint = measure_footprint(video, enu_lookup)
        except Exception:
            logger.warning("triage: footprint measurement failed; using the altitude rule", exc_info=True)
            footprint = None
    if diag_out is not None and footprint is not None:
        diag_out["footprint"] = footprint.summary()
        diag_out["forward_overlap"] = overlap
        diag_out["spacing_m_median"] = round((1.0 - overlap) * footprint.median_along_m, 2)

    def _spacing_at(t: float) -> float | None:
        return None if footprint is None else (1.0 - overlap) * footprint.at(t)

    if use_gps and enu_lookup is not None:
        # The keyframe cap must never truncate the flight: covering the whole
        # track needs about length / spacing keyframes. DJI_1001 (4.1 km)
        # once stopped at 150 after 3.4 km, leaving the last 150 s of video
        # unreconstructed.
        t_arr, enu_xyz, _alt = enu_lookup
        # Only the stretch the video covers: a flight log usually runs well
        # past both ends of the clip (flight01's log is 15.4 km, a 6-minute
        # clip of it ~6 km).
        fps = float(getattr(video, "fps", 0.0) or 0.0)
        frames = int(getattr(video, "frame_count", 0) or 0)
        if fps > 0 and frames > 0:
            in_video = (t_arr >= 0.0) & (t_arr <= frames / fps)
            enu_xyz = enu_xyz[in_video]
        track_m = float(np.linalg.norm(np.diff(enu_xyz[:, :2], axis=0), axis=1).sum()) if len(enu_xyz) > 1 else 0.0
        spacing = (1.0 - overlap) * footprint.median_along_m if footprint is not None else config.max_baseline_m
        needed = int(math.ceil(track_m / max(spacing, 1e-6) * 1.15)) + 10
        if needed > config.target_keyframes:
            logger.info(
                "triage: raising keyframe target %d -> %d to cover the %.0f m GPS track at ~%.0f m spacing",
                config.target_keyframes,
                needed,
                track_m,
                spacing,
            )
            import dataclasses

            config = dataclasses.replace(config, target_keyframes=needed)
    total_frames = video.frame_count or 0
    scan_step = 1
    if total_frames > 0 and config.max_frames_scanned > 0:
        scan_step = max(1, math.ceil(total_frames / config.max_frames_scanned))

    window_radius = max(_MIN_WINDOW_RADIUS, scan_step)
    buffer: deque[_BufferEntry] = deque(maxlen=_RING_BUFFER_MULTIPLIER * window_radius + _RING_BUFFER_SLACK)

    keyframes: list[Keyframe] = []
    all_metrics: list[FrameMetrics] = []

    accumulated_vision_px = 0.0
    accumulated_baseline_m = 0.0
    prev_gray: np.ndarray | None = None
    prev_pos: np.ndarray | None = None
    effective_min_parallax = float(config.min_parallax_px)
    effective_min_baseline_m = float(config.min_baseline_m)
    checkpoint_count = 0
    # The sharpest-frame window search can land ahead of the checkpoint
    # cursor (it searches +/- radius around the trigger frame). Track the
    # last accepted keyframe's index so we can skip re-triggering on
    # checkpoints the accepted keyframe already "covers" -- otherwise the
    # loop keeps re-finding the same window and appending duplicate
    # keyframes until the cursor finally advances past it.
    last_keyframe_index = -1

    # GPS mode never needs pixels to decide WHEN to take a keyframe, so
    # frames are decoded but only converted and blur-scored when a window
    # search reads them (see _LazyEntry). Vision mode needs every
    # checkpoint's pixels for parallax, so it stays eager.
    lazy = use_gps and getattr(config, "lazy_decode", True) and hasattr(video, "iter_frames_lazy")
    if lazy:
        frame_iter = video.iter_frames_lazy()
        make_entry = _LazyEntry
    else:
        frame_iter = video.iter_frames(step=1, downscale=_SCAN_DOWNSCALE_PX)
        make_entry = _eager_entry
    last_checkpoint_entry = None
    turn_deferrals = 0

    for frame in frame_iter:
        entry = make_entry(frame)
        buffer.append(entry)

        if frame.index % scan_step != 0:
            # Decoded (and buffered) for the ring buffer's sake, but not a
            # checkpoint: no metric evaluation this frame.
            continue

        checkpoint_count += 1
        if lazy:
            # A lazy checkpoint's pixel metrics are measured only if a
            # window search reads it; the first and last are always
            # recorded so the report's scanned timeline spans the video.
            last_checkpoint_entry = entry
            gray = entry[2] if checkpoint_count == 1 else None
            blur = entry[3] if checkpoint_count == 1 else None
        else:
            gray, blur = entry[2], entry[3]
        stale = frame.index <= last_keyframe_index

        step_signal = 0.0
        triggered = False
        if not stale:
            if use_gps:
                assert enu_lookup is not None
                pos, alt = _interp_enu_position(enu_lookup, frame.timestamp)
                if pos is not None:
                    if prev_pos is not None:
                        step_signal = float(np.linalg.norm(pos - prev_pos))
                        accumulated_baseline_m += step_signal
                    prev_pos = pos
                target_baseline_m = effective_min_baseline_m
                measured = _spacing_at(frame.timestamp)
                if measured is not None:
                    # Overlap-driven: (1 - forward_overlap) x measured footprint.
                    target_baseline_m = max(effective_min_baseline_m, measured)
                elif alt is not None:
                    target_baseline_m = max(effective_min_baseline_m, config.baseline_to_altitude_ratio * alt)
                triggered = accumulated_baseline_m >= target_baseline_m
                # Hold the keyframe through a hard turn: the aircraft banks,
                # the image blurs, and on flight01 (17.6 m/s lawnmower) the
                # windows spanning turns landed tens of metres off. Baseline
                # keeps accumulating, so the first steady frame after the
                # turn is taken instead.
                max_turn = getattr(config, "max_turn_rate_deg_s", None)
                if triggered and max_turn:
                    rate = _turn_rate_deg_s(enu_lookup, frame.timestamp)
                    if rate is not None and rate > max_turn:
                        triggered = False
                        turn_deferrals += 1
            else:
                if prev_gray is not None:
                    step_signal = estimate_parallax_detailed(prev_gray, gray, intrinsics=intrinsics).useful_baseline_px
                    accumulated_vision_px += step_signal
                prev_gray = gray
                triggered = accumulated_vision_px >= effective_min_parallax

        if gray is not None:
            all_metrics.append(
                FrameMetrics(
                    index=frame.index,
                    timestamp=frame.timestamp,
                    blur_score=blur,
                    exposure_score=exposure_score(gray),
                    mean_luma=mean_luma(gray),
                    estimated_parallax=step_signal,
                )
            )

        if progress_cb is not None:
            progress_cb(
                frame.index, total_frames, f"scanned {checkpoint_count} frames, {len(keyframes)} keyframes accepted"
            )

        if triggered:
            chosen = _pick_sharpest_in_window(
                video,
                frame_iter,
                buffer,
                frame.index,
                window_radius,
                config.min_blur_score,
                config.relative_blur_threshold,
                min_index_exclusive=last_keyframe_index,
                make_entry=make_entry,
            )
            if chosen is None:
                logger.debug("triage: deferred keyframe near frame %d (no sharp candidate in window)", frame.index)
                if progress_cb is not None:
                    progress_cb(
                        frame.index, total_frames, f"deferred keyframe near frame {frame.index}: too blurry"
                    )
                continue

            chosen_index, chosen_timestamp, chosen_gray, chosen_blur = chosen
            if full_res_sink is not None and isinstance(chosen, _LazyEntry):
                # Decoded right now: keep its full-resolution pixels so the
                # keyframe cache needs no second pass over the video.
                full_res_sink[chosen_index] = chosen.bgr24()
            chosen_metrics = FrameMetrics(
                index=chosen_index,
                timestamp=chosen_timestamp,
                blur_score=chosen_blur,
                exposure_score=exposure_score(chosen_gray),
                mean_luma=mean_luma(chosen_gray),
                estimated_parallax=accumulated_baseline_m if use_gps else accumulated_vision_px,
            )
            keyframes.append(Keyframe(frame_index=chosen_index, timestamp=chosen_timestamp, metrics=chosen_metrics))

            # Restart accumulation from the *accepted* keyframe's own
            # position/visual content, not the checkpoint frame that
            # happened to trigger it -- the window search can land on a
            # different index.
            accumulated_vision_px = 0.0
            accumulated_baseline_m = 0.0
            prev_gray = chosen_gray
            if use_gps:
                assert enu_lookup is not None
                pos_chosen, _alt_chosen = _interp_enu_position(enu_lookup, chosen_timestamp)
                if pos_chosen is not None:
                    prev_pos = pos_chosen
            last_keyframe_index = chosen_index

            if progress_cb is not None:
                progress_cb(
                    frame.index, total_frames, f"accepted keyframe {len(keyframes)} at frame {chosen_index}"
                )

            if len(keyframes) >= config.target_keyframes:
                break

            if use_gps:
                effective_min_baseline_m = _maybe_adapt_threshold(
                    effective_min_baseline_m, config, len(keyframes), total_frames, frame.index, progress_cb
                )
                # Never past MapAnything's usable footprint: beyond it consecutive
                # keyframes stop overlapping (see TriageConfig.max_baseline_m).
                if footprint is None:
                    effective_min_baseline_m = min(
                        effective_min_baseline_m, max(config.max_baseline_m, config.min_baseline_m)
                    )
                else:
                    # Never let the count-driven adaptation push spacing past
                    # the overlap target: overlap, not count, sets spacing.
                    effective_min_baseline_m = min(
                        effective_min_baseline_m, max(config.min_baseline_m, (1.0 - overlap) * footprint.median_along_m)
                    )
            else:
                effective_min_parallax = _maybe_adapt_threshold(
                    effective_min_parallax, config, len(keyframes), total_frames, frame.index, progress_cb
                )

        if checkpoint_count >= config.max_frames_scanned:
            logger.info("triage: hit max_frames_scanned (%d) safety cap", config.max_frames_scanned)
            break

    if lazy and last_checkpoint_entry is not None and (not all_metrics or all_metrics[-1].index != last_checkpoint_entry[0]):
        gray = last_checkpoint_entry[2]
        all_metrics.append(
            FrameMetrics(
                index=last_checkpoint_entry[0],
                timestamp=last_checkpoint_entry[1],
                blur_score=last_checkpoint_entry[3],
                exposure_score=exposure_score(gray),
                mean_luma=mean_luma(gray),
                estimated_parallax=0.0,
            )
        )

    if samples:
        telemetry_results = resample_telemetry(samples, [kf.timestamp for kf in keyframes])
        for kf, tsample in zip(keyframes, telemetry_results, strict=True):
            kf.telemetry = tsample

    # Cheap defensive dedup: the window-search filtering above
    # (`_pick_sharpest_in_window`'s `min_index_exclusive`) is what actually
    # prevents re-selecting an already-accepted frame, but a belt-and-
    # suspenders pass here means a future spacing-mode addition (a third
    # signal path, say) can't reintroduce this class of bug by forgetting
    # to thread the guard through -- duplicate `frame_index` values would
    # corrupt the bundle-adjustment observation model (two "different"
    # cameras claiming the same image). Keeps the first occurrence, stable
    # order.
    seen_indices: set[int] = set()
    deduped_keyframes: list[Keyframe] = []
    for kf in keyframes:
        if kf.frame_index in seen_indices:
            logger.warning("triage: dropping duplicate keyframe at frame %d (defensive dedup)", kf.frame_index)
            continue
        seen_indices.add(kf.frame_index)
        deduped_keyframes.append(kf)
    keyframes = deduped_keyframes

    if turn_deferrals:
        logger.info("triage: held %d keyframe trigger(s) through hard turns", turn_deferrals)
    return keyframes, all_metrics


def triage_report(
    keyframes: list[Keyframe], all_metrics: list[FrameMetrics], spacing_mode: str | None = None
) -> dict:
    """Summarize a triage run for a UI panel / logs.

    Reports selected vs. rejected blur stats (to sanity-check the blur
    floor is doing something), the median parallax between consecutive
    keyframes (the actual achieved baseline spacing), how much of the
    video's timeline the selected keyframes span, and -- when telemetry
    was attached -- the trajectory length covered and the GPS-metric
    inter-keyframe spacing actually achieved.

    ``spacing_mode`` reports which signal ``select_keyframes`` used to
    drive spacing decisions ("gps_baseline" or "vision_parallax" -- see its
    docstring). If not given explicitly, it's inferred from whether the
    keyframes carry GPS-tagged telemetry, which is a reasonable proxy since
    the same telemetry availability gates both.
    """
    selected_idx = {kf.frame_index for kf in keyframes}

    # DEFECT 1 fix: selected-frame blur stats must come from each
    # keyframe's own `metrics`, not from a lookup into `all_metrics` by
    # index. `all_metrics` only ever contains checkpoint-cadence frames;
    # the actual frame `_pick_sharpest_in_window` keeps is chosen from a
    # *different* native index (checkpoint +/- window radius), which --
    # once the checkpoint stride is greater than 1 -- is almost never a
    # member of `all_metrics`. Looking it up there by index silently
    # produced an empty list and made blur_selected_* report 0.0 on any
    # realistically long video. `Keyframe.metrics` already carries that
    # keyframe's own real, freshly-computed metrics, so use those directly.
    selected_metrics = [kf.metrics for kf in keyframes]
    rejected_metrics = [m for m in all_metrics if m.index not in selected_idx]

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def _min(xs: list[float]) -> float:
        return float(np.min(xs)) if xs else 0.0

    # estimated_parallax on a keyframe's own metrics is the accumulated
    # spacing signal that triggered *it* (px in vision mode, metres in GPS
    # mode -- see `spacing_mode`); skip the first keyframe, which has
    # nothing before it to be spaced from.
    inter_keyframe_signal = [kf.metrics.estimated_parallax for kf in keyframes[1:]]
    median_parallax = float(np.median(inter_keyframe_signal)) if inter_keyframe_signal else 0.0

    coverage = 0.0
    if all_metrics:
        scanned_timestamps = [m.timestamp for m in all_metrics]
        span = max(scanned_timestamps) - min(scanned_timestamps)
        if span > 0 and len(keyframes) > 1:
            kf_span = keyframes[-1].timestamp - keyframes[0].timestamp
            coverage = float(kf_span / span)

    telemetry_samples = [kf.telemetry for kf in keyframes if kf.telemetry is not None and kf.telemetry.geo is not None]
    trajectory_m: float | None = None
    median_baseline_m: float | None = None
    if telemetry_samples:
        trajectory_m = trajectory_length(telemetry_samples)
        if len(telemetry_samples) >= 2:
            ordered = sorted(telemetry_samples, key=lambda s: s.timestamp)
            enu, _origin = telemetry_to_enu(ordered)
            dists = np.linalg.norm(np.diff(enu, axis=0), axis=1)
            if len(dists):
                median_baseline_m = float(np.median(dists))

    if spacing_mode is None:
        spacing_mode = "gps_baseline" if telemetry_samples else "vision_parallax"

    return {
        "keyframes_selected": len(keyframes),
        "frames_scanned": len(all_metrics),
        "blur_selected_mean": _mean([m.blur_score for m in selected_metrics]),
        "blur_selected_min": _min([m.blur_score for m in selected_metrics]),
        "blur_rejected_mean": _mean([m.blur_score for m in rejected_metrics]),
        "blur_rejected_min": _min([m.blur_score for m in rejected_metrics]),
        "median_interkeyframe_parallax_px": median_parallax,
        "timeline_coverage_fraction": coverage,
        "trajectory_length_m": trajectory_m,
        "spacing_mode": spacing_mode,
        "median_interkeyframe_baseline_m": median_baseline_m,
    }
