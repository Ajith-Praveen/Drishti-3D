"""Auto-detect the drone's capture configuration and recommend a submap-merge strategy.

Why this module exists
-----------------------
``geometry.submap.merge_submaps`` has to stitch a sequence of independent,
per-window reconstructions into one global frame. The right way to do that
depends entirely on *how the flight was flown*:

- A single, arrow-straight nadir pass gives near-collinear camera centres
  -- rotation cannot be recovered from those positions alone (see
  ``geometry.submap``'s module docstring), but the gimbal telemetry is
  rock-solid (locked at one pitch for the whole flight), so the right move
  is to trust telemetry for orientation and only fit scale/translation.
- A flight with real heading changes (a grid/lawnmower survey, an orbit, a
  curved pass) gives camera centres that DO constrain rotation well, so an
  independent per-submap GPS anchor (solving the full Sim(3), rotation
  included) is both safe and stronger than chaining.
- A flight with no usable telemetry at all (no GPS, no gimbal attitude) has
  nothing external to anchor to and has to fall back to the original
  chained-Sim(3) scheme, degraded but not broken.

Before this module existed, ``GeometryStage`` picked a merge strategy (in
practice, whatever ``geometry.submap`` happened to default to) without ever
looking at the data to check whether that choice made sense for *this*
flight. ``analyze_flight_profile`` looks at the actual telemetry + keyframe
poses triage/geometry already have on hand and classifies the capture, so
the merge-strategy choice (``FlightProfile.recommended_merge_strategy``,
consumed by ``geometry.submap.merge_submaps``'s ``strategy`` parameter) is
a decision made from evidence, not a hardcoded assumption -- and so an
operator looking at the log can see *why* the pipeline picked what it
picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from drishti3d.types import Keyframe

# ---------------------------------------------------------------------------
# Gimbal-mode classification
# ---------------------------------------------------------------------------

_NADIR_PITCH_DEG = -90.0
_FORWARD_PITCH_DEG = 0.0
# How close the median pitch has to be to nadir/forward to call it that,
# rather than "oblique" -- DJI gimbals rarely sit at *exactly* -90/0, and a
# few degrees of mechanical/telemetry noise is normal for a "locked" mode.
_GIMBAL_MODE_TOLERANCE_DEG = 20.0
# Spread (std) above this means the gimbal was actively being flown through
# a range of angles during the capture, not held in one mode -- reported as
# "varying" regardless of where the median happens to land.
_GIMBAL_VARYING_STD_DEG = 15.0
_MIN_GIMBAL_SAMPLES = 3

# ---------------------------------------------------------------------------
# Trajectory-shape classification
# ---------------------------------------------------------------------------

_MIN_TRACK_POINTS = 3
# Collinearity (see _classify_trajectory) at/above this is called "linear" --
# deliberately generous (triggers early) because a track that is *mostly*
# straight still has the rotation-from-camera-centres degeneracy this
# module exists to route around.
_COLLINEARITY_LINEAR_THRESHOLD = 0.85
# A lawnmower/grid survey is characterised by sharp heading reversals at the
# end of each pass; a heading change bigger than this between consecutive
# track segments counts as a "turn".
_TURN_HEADING_CHANGE_DEG = 60.0
_MIN_GRID_TURNS = 2
# Orbit detection: roughly-constant radius from the track's own centroid
# (coefficient of variation below this) plus a wide angular sweep.
_ORBIT_RADIUS_CV_MAX = 0.35
_ORBIT_MIN_ANGULAR_SPAN_DEG = 270.0

# Vertical-ascent/descent detection: a track that is mostly a straight line
# in 3D (high SVD-based collinearity, see _classify_trajectory) can be a
# horizontal flight-strip (this module's existing "linear" semantics: a
# locked-nadir gimbal on a single survey pass) OR a climb/descent in place
# (altitude changes a lot, horizontal position barely moves at all) -- the
# generic 3D collinearity score cannot tell these apart (it is deliberately
# axis-agnostic), but they need different handling downstream (see
# analyze_flight_profile's strategy-selection notes), so this is checked
# explicitly, directly on horizontal span vs altitude range, rather than
# inferred from the axis-agnostic collinearity number alone. Thresholds are
# deliberately loose (triggers early) on altitude range so a real climb
# (real example: ~1m -> ~119m AGL) is always caught, and tight on
# horizontal span so a real horizontal survey pass (tens to hundreds of
# metres of track) is never misclassified as "vertical" just for having a
# little GPS jitter.
_VERTICAL_MIN_ALTITUDE_RANGE_M = 10.0
_VERTICAL_MAX_HORIZONTAL_SPAN_M = 10.0
_VERTICAL_ALTITUDE_TO_HORIZONTAL_RATIO_MIN = 3.0

# ---------------------------------------------------------------------------
# Reliability / strategy thresholds
# ---------------------------------------------------------------------------

# Fraction of keyframes that need a usable gimbal pitch *and* yaw reading
# (the same pair ``pipeline.stages._poses_from_telemetry`` requires to
# condition a pose) before telemetry orientation is trusted enough to
# anchor a submap's rotation on.
_ORIENTATION_RELIABLE_COVERAGE = 0.5
# Even with good coverage, a gimbal that's all over the place (already
# "varying" territory) is not a reliable single orientation reference.
_ORIENTATION_RELIABLE_MAX_STD_DEG = 45.0


@dataclass
class FlightProfile:
    """What ``analyze_flight_profile`` concluded about one flight's capture geometry.

    ``recommended_merge_strategy`` is one of ``"telemetry_rotation"``,
    ``"gps_anchored"``, ``"chained_sim3"`` -- see ``geometry.submap``'s
    module docstring for what each one actually does; this field is what
    ``pipeline.stages.GeometryStage`` passes straight through to
    ``geometry.submap.merge_submaps``'s ``strategy`` parameter.
    """

    gimbal_mode: str  # "nadir" / "oblique" / "forward" / "varying"
    gimbal_pitch_median_deg: float | None
    gimbal_pitch_std_deg: float | None

    trajectory_shape: str  # "linear" / "curved" / "grid" / "orbit" / "vertical"
    collinearity: float | None  # 0 (spread out) .. 1 (perfectly collinear)

    altitude_agl_median_m: float | None
    altitude_variation_m: float | None

    orientation_is_reliable: bool
    recommended_merge_strategy: str
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line, log-friendly summary -- see ``GeometryStage.run``'s prominent log line."""
        pitch = (
            f"{self.gimbal_pitch_median_deg:.1f}+/-{self.gimbal_pitch_std_deg:.1f} deg"
            if self.gimbal_pitch_median_deg is not None and self.gimbal_pitch_std_deg is not None
            else "unknown"
        )
        alt = (
            f"{self.altitude_agl_median_m:.1f}m (+/-{self.altitude_variation_m:.1f}m)"
            if self.altitude_agl_median_m is not None and self.altitude_variation_m is not None
            else "unknown"
        )
        collin = f"{self.collinearity:.2f}" if self.collinearity is not None else "n/a"
        return (
            f"gimbal={self.gimbal_mode} (pitch {pitch}); trajectory={self.trajectory_shape} "
            f"(collinearity={collin}); altitude AGL={alt}; "
            f"orientation_reliable={self.orientation_is_reliable} -> "
            f"recommended_merge_strategy={self.recommended_merge_strategy!r}"
        )


def _classify_gimbal_mode(median_deg: float | None, std_deg: float | None) -> str:
    if median_deg is None:
        return "varying"
    if std_deg is not None and std_deg > _GIMBAL_VARYING_STD_DEG:
        return "varying"
    if abs(median_deg - _NADIR_PITCH_DEG) <= _GIMBAL_MODE_TOLERANCE_DEG:
        return "nadir"
    if abs(median_deg - _FORWARD_PITCH_DEG) <= _GIMBAL_MODE_TOLERANCE_DEG:
        return "forward"
    return "oblique"


def _wrapped_heading_diff_deg(a: float, b: float) -> float:
    """Smallest-magnitude angular difference ``b - a``, wrapped to [-180, 180] degrees."""
    return float((np.degrees(b - a) + 180.0) % 360.0 - 180.0)


def _classify_trajectory(enu: np.ndarray) -> tuple[str, float]:
    """Classify a ``(N, 3)`` centred-or-not ENU track. Returns ``(shape, collinearity)``.

    ``collinearity`` comes from the SVD of the *centred* track: the ratio of
    the second-largest to largest singular value is ~0 for a perfectly
    straight track (all variance along one axis) and close to 1 for a track
    that spreads out roughly evenly in a second direction -- so
    ``1 - s1/s0`` is a clean 0 (spread out) .. 1 (collinear) score. This is
    the same "collinear vs merely coplanar" distinction
    ``geometry.submap.umeyama_alignment`` makes for the shared-camera
    degeneracy check, applied here to the whole flight track instead of one
    junction's 3 shared cameras.
    """
    n = enu.shape[0]
    if n < _MIN_TRACK_POINTS:
        return "linear", 1.0

    centred = enu - enu.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    s0 = float(singular_values[0])
    s1 = float(singular_values[1]) if singular_values.shape[0] > 1 else 0.0
    collinearity = float(np.clip(1.0 - (s1 / s0 if s0 > 1e-9 else 0.0), 0.0, 1.0))

    # Vertical ascent/descent check (see module-level threshold constants'
    # docstring): checked before the generic "linear" collinearity return
    # below, since a vertical climb IS collinear in 3D (all variance
    # concentrated along one axis, here the vertical one) and would
    # otherwise silently fall through to the horizontal-flight-strip
    # "linear" label -- misleading for a human reading the flight-profile
    # summary, and (see analyze_flight_profile) needing a different
    # strategy-selection note since telemetry_rotation's whole design
    # assumes a horizontal survey pass.
    horizontal_span = float(np.linalg.norm(centred[:, :2].max(axis=0) - centred[:, :2].min(axis=0)))
    altitude_range = float(enu[:, 2].max() - enu[:, 2].min())
    if (
        altitude_range >= _VERTICAL_MIN_ALTITUDE_RANGE_M
        and horizontal_span <= _VERTICAL_MAX_HORIZONTAL_SPAN_M
        and altitude_range >= _VERTICAL_ALTITUDE_TO_HORIZONTAL_RATIO_MIN * max(horizontal_span, 1e-6)
    ):
        return "vertical", collinearity

    if collinearity >= _COLLINEARITY_LINEAR_THRESHOLD:
        return "linear", collinearity

    # Orbit check: does the track sweep around its own centroid at a
    # roughly constant radius through a wide angular span?
    xy = centred[:, :2]
    radii = np.linalg.norm(xy, axis=1)
    mean_r = float(radii.mean())
    if mean_r > 1e-6:
        radius_cv = float(radii.std() / mean_r)
        angles = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        angular_span_deg = float(np.degrees(abs(angles[-1] - angles[0])))
        if radius_cv < _ORBIT_RADIUS_CV_MAX and angular_span_deg > _ORBIT_MIN_ANGULAR_SPAN_DEG:
            return "orbit", collinearity

    # Grid/lawnmower check: count sharp heading reversals along the path.
    deltas = np.diff(enu[:, :2], axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    valid = seg_lengths > 1e-6
    if int(valid.sum()) >= _MIN_GRID_TURNS + 1:
        headings = np.arctan2(deltas[valid, 1], deltas[valid, 0])
        turn_count = sum(
            1
            for a, b in pairwise(headings)
            if abs(_wrapped_heading_diff_deg(a, b)) > _TURN_HEADING_CHANGE_DEG
        )
        if turn_count >= _MIN_GRID_TURNS:
            return "grid", collinearity

    return "curved", collinearity


def analyze_flight_profile(keyframes: list[Keyframe]) -> FlightProfile:
    """Classify the capture configuration from ``keyframes``' own telemetry + poses.

    Every figure here comes from the same per-keyframe ``TelemetrySample``
    (``Keyframe.telemetry``) the rest of the pipeline already uses (e.g.
    ``pipeline.stages._poses_from_telemetry``, ``_gps_enu_by_keyframe``) --
    this function adds no new data source, only a classification pass over
    data already on hand.
    """
    notes: list[str] = []

    pitches = np.array(
        [
            kf.telemetry.gimbal_pitch
            for kf in keyframes
            if kf.telemetry is not None and kf.telemetry.gimbal_pitch is not None
        ],
        dtype=np.float64,
    )
    if pitches.size >= _MIN_GIMBAL_SAMPLES:
        pitch_median = float(np.median(pitches))
        pitch_std = float(np.std(pitches))
    elif pitches.size > 0:
        pitch_median = float(np.median(pitches))
        pitch_std = float(np.std(pitches)) if pitches.size > 1 else 0.0
        notes.append(f"only {pitches.size} gimbal-pitch sample(s); gimbal_mode classification is low-confidence")
    else:
        pitch_median = None
        pitch_std = None
        notes.append("no gimbal-pitch telemetry available; assuming gimbal_mode='varying'")

    gimbal_mode = _classify_gimbal_mode(pitch_median, pitch_std)

    geo_samples = [kf.telemetry for kf in keyframes if kf.telemetry is not None and kf.telemetry.geo is not None]
    enu = np.zeros((0, 3), dtype=np.float64)
    if len(geo_samples) >= _MIN_TRACK_POINTS:
        from drishti3d.ingest.telemetry import telemetry_to_enu

        enu_all, _origin = telemetry_to_enu(geo_samples)
        finite = np.all(np.isfinite(enu_all), axis=1)
        enu = enu_all[finite]

    if enu.shape[0] >= _MIN_TRACK_POINTS:
        trajectory_shape, collinearity = _classify_trajectory(enu)
    else:
        trajectory_shape = "linear"
        collinearity = None
        notes.append("too few geo-tagged keyframes to assess trajectory shape")

    alt_rel = np.array(
        [
            kf.telemetry.geo.alt_rel
            for kf in keyframes
            if kf.telemetry is not None and kf.telemetry.geo is not None and kf.telemetry.geo.alt_rel is not None
        ],
        dtype=np.float64,
    )
    if alt_rel.size == 0:
        alt_rel = np.array(
            [
                kf.telemetry.geo.alt_msl
                for kf in keyframes
                if kf.telemetry is not None and kf.telemetry.geo is not None
            ],
            dtype=np.float64,
        )
        if alt_rel.size:
            notes.append("no relative/AGL altitude in telemetry; using MSL altitude as a proxy for AGL")

    if alt_rel.size:
        altitude_agl_median_m = float(np.median(alt_rel))
        altitude_variation_m = float(alt_rel.max() - alt_rel.min())
    else:
        altitude_agl_median_m = None
        altitude_variation_m = None

    n = len(keyframes)
    orientation_coverage = (
        sum(
            1
            for kf in keyframes
            if kf.telemetry is not None
            and kf.telemetry.gimbal_pitch is not None
            and kf.telemetry.gimbal_yaw is not None
        )
        / n
        if n
        else 0.0
    )
    orientation_is_reliable = (
        orientation_coverage >= _ORIENTATION_RELIABLE_COVERAGE
        and pitch_std is not None
        and pitch_std <= _ORIENTATION_RELIABLE_MAX_STD_DEG
    )
    if not orientation_is_reliable:
        notes.append(
            f"gimbal-attitude coverage {orientation_coverage * 100.0:.0f}% "
            f"(need >= {_ORIENTATION_RELIABLE_COVERAGE * 100.0:.0f}%) or spread too large; "
            "telemetry orientation is not trusted for submap rotation"
        )

    has_gps_track = enu.shape[0] >= _MIN_TRACK_POINTS

    if collinearity is not None and collinearity >= _COLLINEARITY_LINEAR_THRESHOLD and orientation_is_reliable:
        recommended = "telemetry_rotation"
    elif has_gps_track and collinearity is not None and collinearity < _COLLINEARITY_LINEAR_THRESHOLD:
        recommended = "gps_anchored"
    else:
        recommended = "chained_sim3"
        if trajectory_shape == "vertical":
            # A vertical climb/descent is collinear too (just along the
            # altitude axis instead of a horizontal flight direction), so
            # it has exactly the same rotation-from-camera-centres
            # degeneracy a horizontal flight strip has (see
            # geometry.submap's degenerate-alignment discussion, which is
            # axis-agnostic) -- telemetry_rotation would sidestep that
            # (as it does for a horizontal nadir pass) if trustworthy
            # gimbal attitude were available, but it isn't here
            # (orientation_is_reliable is False whenever this branch is
            # reached for a vertical track), and gps_anchored's own full
            # Sim(3) fit needs a non-collinear camera track to be
            # well-posed at all -- neither alternative applies, so this is
            # a genuine, not merely thresholding-related, fallback.
            notes.append(
                "trajectory_shape='vertical' (altitude range "
                f"{altitude_variation_m:.1f}m vs. essentially no horizontal displacement -- a "
                "climb/descent in place, not a horizontal survey pass) with no reliable telemetry "
                "orientation available (no/insufficient gimbal attitude): rotation is exactly as "
                "unrecoverable from a vertically-collinear camera track as from a horizontally-"
                "collinear one, and a non-collinear GPS track (which gps_anchored would need) isn't "
                "available either -- falling back to chained Sim(3) merging."
                if altitude_variation_m is not None
                else "trajectory_shape='vertical' with no reliable telemetry orientation and no "
                "non-collinear GPS track available -- falling back to chained Sim(3) merging."
            )
        else:
            notes.append(
                "neither a trustworthy telemetry orientation on a collinear track, nor a "
                "non-collinear GPS track, is available; falling back to chained Sim(3) merging"
            )

    return FlightProfile(
        gimbal_mode=gimbal_mode,
        gimbal_pitch_median_deg=pitch_median,
        gimbal_pitch_std_deg=pitch_std,
        trajectory_shape=trajectory_shape,
        collinearity=collinearity,
        altitude_agl_median_m=altitude_agl_median_m,
        altitude_variation_m=altitude_variation_m,
        orientation_is_reliable=orientation_is_reliable,
        recommended_merge_strategy=recommended,
        notes=notes,
    )
