"""Tests for drishti3d.geometry.flight_profile: capture-configuration auto-detection.

Pure numpy -- no torch, no CUDA, no model weights needed.
"""

from __future__ import annotations

import numpy as np

from drishti3d.geometry.flight_profile import analyze_flight_profile
from drishti3d.types import FrameMetrics, GeoPoint, Keyframe, TelemetrySample

_LAT0, _LON0 = 12.9, 77.6


def _make_keyframe(i: int, lat: float, lon: float, alt_rel: float, pitch: float, yaw: float, roll: float = 0.0) -> Keyframe:
    metrics = FrameMetrics(index=i, timestamp=float(i), blur_score=1.0, exposure_score=1.0, mean_luma=100.0, estimated_parallax=1.0)
    geo = GeoPoint(lat=lat, lon=lon, alt_msl=100.0 + alt_rel, alt_rel=alt_rel)
    telemetry = TelemetrySample(timestamp=float(i), geo=geo, gimbal_pitch=pitch, gimbal_roll=roll, gimbal_yaw=yaw)
    return Keyframe(frame_index=i, timestamp=float(i), metrics=metrics, telemetry=telemetry)


def _straight_line_keyframes(n: int, pitch_deg: float, pitch_noise_std: float, seed: int) -> list[Keyframe]:
    rng = np.random.default_rng(seed)
    keyframes = []
    for i in range(n):
        dlat = (5.0 * i) / 111320.0  # ~5m spacing due north
        pitch = pitch_deg + float(rng.normal(0.0, pitch_noise_std))
        keyframes.append(_make_keyframe(i, _LAT0 + dlat, _LON0, alt_rel=100.0, pitch=pitch, yaw=90.0))
    return keyframes


def _lawnmower_keyframes(n_legs: int, points_per_leg: int, leg_length_m: float, leg_spacing_m: float, seed: int) -> list[Keyframe]:
    rng = np.random.default_rng(seed)
    xs_forward = np.linspace(0.0, leg_length_m, points_per_leg)
    points: list[tuple[float, float]] = []
    for leg in range(n_legs):
        xs = xs_forward if leg % 2 == 0 else xs_forward[::-1]
        y = leg * leg_spacing_m
        points.extend((x, y) for x in xs)

    keyframes = []
    for i, (x, y) in enumerate(points):
        dlat = y / 111320.0
        dlon = x / (111320.0 * np.cos(np.radians(_LAT0)))
        pitch = -90.0 + float(rng.normal(0.0, 0.5))
        keyframes.append(_make_keyframe(i, _LAT0 + dlat, _LON0 + dlon, alt_rel=100.0, pitch=pitch, yaw=90.0))
    return keyframes


# ---------------------------------------------------------------------------
# gimbal_mode / trajectory_shape / recommended_merge_strategy
# ---------------------------------------------------------------------------


def test_nadir_linear_flight_recommends_telemetry_rotation():
    keyframes = _straight_line_keyframes(n=20, pitch_deg=-90.0, pitch_noise_std=0.5, seed=0)

    profile = analyze_flight_profile(keyframes)

    assert profile.gimbal_mode == "nadir"
    assert profile.trajectory_shape == "linear"
    assert profile.collinearity is not None and profile.collinearity > 0.95
    assert profile.orientation_is_reliable is True
    assert profile.recommended_merge_strategy == "telemetry_rotation"
    assert profile.altitude_agl_median_m == 100.0


def test_oblique_linear_flight_recommends_telemetry_rotation():
    keyframes = _straight_line_keyframes(n=20, pitch_deg=-45.0, pitch_noise_std=0.5, seed=1)

    profile = analyze_flight_profile(keyframes)

    assert profile.gimbal_mode == "oblique"
    assert profile.trajectory_shape == "linear"
    assert profile.orientation_is_reliable is True
    # Gimbal mode doesn't affect strategy choice -- only collinearity +
    # orientation reliability do (see module docstring).
    assert profile.recommended_merge_strategy == "telemetry_rotation"


def test_nadir_grid_flight_recommends_gps_anchored():
    keyframes = _lawnmower_keyframes(n_legs=3, points_per_leg=5, leg_length_m=20.0, leg_spacing_m=15.0, seed=2)

    profile = analyze_flight_profile(keyframes)

    assert profile.gimbal_mode == "nadir"
    assert profile.trajectory_shape == "grid"
    assert profile.collinearity is not None and profile.collinearity < 0.85
    assert profile.orientation_is_reliable is True
    # A grid survey's camera centres genuinely constrain rotation -- the
    # per-submap independent GPS anchor (full Sim(3)) is the right choice,
    # not fixing rotation from telemetry alone.
    assert profile.recommended_merge_strategy == "gps_anchored"


def test_forward_gimbal_classified_as_forward():
    keyframes = _straight_line_keyframes(n=10, pitch_deg=0.0, pitch_noise_std=1.0, seed=3)

    profile = analyze_flight_profile(keyframes)

    assert profile.gimbal_mode == "forward"


def test_varying_gimbal_when_pitch_spread_is_large():
    rng = np.random.default_rng(4)
    keyframes = []
    for i in range(12):
        dlat = (5.0 * i) / 111320.0
        # Sweep pitch from -90 to 0 across the flight -- a large spread,
        # not one held mode.
        pitch = -90.0 + i * 8.0 + float(rng.normal(0, 1))
        keyframes.append(_make_keyframe(i, _LAT0 + dlat, _LON0, alt_rel=100.0, pitch=pitch, yaw=90.0))

    profile = analyze_flight_profile(keyframes)

    assert profile.gimbal_mode == "varying"


# ---------------------------------------------------------------------------
# No/sparse telemetry -- must degrade, never crash
# ---------------------------------------------------------------------------


def test_no_telemetry_falls_back_to_chained_sim3():
    metrics = [
        FrameMetrics(index=i, timestamp=float(i), blur_score=1.0, exposure_score=1.0, mean_luma=100.0, estimated_parallax=1.0)
        for i in range(5)
    ]
    keyframes = [Keyframe(frame_index=i, timestamp=float(i), metrics=metrics[i], telemetry=None) for i in range(5)]

    profile = analyze_flight_profile(keyframes)

    assert profile.orientation_is_reliable is False
    assert profile.recommended_merge_strategy == "chained_sim3"
    assert profile.notes  # should explain why


def test_empty_keyframes_does_not_crash():
    profile = analyze_flight_profile([])

    assert profile.recommended_merge_strategy == "chained_sim3"
    assert profile.orientation_is_reliable is False


def _vertical_ascent_keyframes(n: int, seed: int) -> list[Keyframe]:
    """A near-stationary vertical climb: tiny horizontal drift, large altitude change, no gimbal telemetry."""
    rng = np.random.default_rng(seed)
    keyframes = []
    for i in range(n):
        # ~2m E / 1m N total drift over the whole climb, dominated by GPS
        # jitter rather than deliberate horizontal motion.
        dlat = (1.0 * i / max(n - 1, 1) + float(rng.normal(0, 0.05))) / 111320.0
        dlon = (2.0 * i / max(n - 1, 1) + float(rng.normal(0, 0.05))) / (111320.0 * np.cos(np.radians(_LAT0)))
        alt_rel = 1.1 + (119.4 - 1.1) * i / max(n - 1, 1)
        metrics = FrameMetrics(index=i, timestamp=float(i), blur_score=1.0, exposure_score=1.0, mean_luma=100.0, estimated_parallax=1.0)
        geo = GeoPoint(lat=_LAT0 + dlat, lon=_LON0 + dlon, alt_msl=100.0 + alt_rel, alt_rel=alt_rel)
        # No gimbal attitude keys at all -- mirrors the real modern-DJI-SRT
        # clip this category was added for.
        telemetry = TelemetrySample(timestamp=float(i), geo=geo, gimbal_pitch=None, gimbal_roll=None, gimbal_yaw=None)
        keyframes.append(Keyframe(frame_index=i, timestamp=float(i), metrics=metrics, telemetry=telemetry))
    return keyframes


def test_vertical_ascent_classified_as_vertical_and_falls_back_to_chained_sim3():
    keyframes = _vertical_ascent_keyframes(n=16, seed=6)

    profile = analyze_flight_profile(keyframes)

    assert profile.trajectory_shape == "vertical"
    # No gimbal attitude at all -- telemetry orientation must not be trusted.
    assert profile.orientation_is_reliable is False
    # Neither telemetry_rotation (no orientation) nor gps_anchored (the
    # track is collinear, just vertically) applies -- chained_sim3 is the
    # only honest fallback left.
    assert profile.recommended_merge_strategy == "chained_sim3"
    assert any("vertical" in note for note in profile.notes)


def test_horizontal_flight_with_altitude_change_not_misclassified_as_vertical():
    # A real horizontal survey pass can also gain/lose altitude (terrain
    # following); large horizontal span must keep this "linear", not
    # "vertical", even with a real altitude change alongside it.
    keyframes = _straight_line_keyframes(n=10, pitch_deg=-90.0, pitch_noise_std=0.1, seed=7)
    for kf in keyframes[:5]:
        kf.telemetry.geo.alt_rel = 90.0
    for kf in keyframes[5:]:
        kf.telemetry.geo.alt_rel = 110.0

    profile = analyze_flight_profile(keyframes)

    assert profile.trajectory_shape == "linear"


def test_altitude_agl_uses_relative_altitude():
    keyframes = _straight_line_keyframes(n=10, pitch_deg=-90.0, pitch_noise_std=0.1, seed=5)
    for kf in keyframes[:5]:
        kf.telemetry.geo.alt_rel = 90.0
    for kf in keyframes[5:]:
        kf.telemetry.geo.alt_rel = 110.0

    profile = analyze_flight_profile(keyframes)

    assert profile.altitude_agl_median_m == 100.0
    assert profile.altitude_variation_m == 20.0
