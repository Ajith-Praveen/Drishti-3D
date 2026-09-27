"""Video <-> telemetry clock measurement from keyframe image rotation (ingest.timesync, TimeSyncStage)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from drishti3d.config import Config
from drishti3d.export.report import build_report, render_report_html, render_report_text
from drishti3d.ingest import timesync
from drishti3d.pipeline.stages import PipelineState, StageUnavailable, TimeSyncStage
from drishti3d.types import FrameMetrics, GeoPoint, Keyframe, TelemetrySample

_RNG = np.random.default_rng(7)
_TEXTURE = cv2.GaussianBlur((_RNG.random((1400, 1400)) * 255).astype(np.uint8), (0, 0), 2.0)
_TEXTURE = cv2.cvtColor(_TEXTURE, cv2.COLOR_GRAY2BGR)


def _heading(t: np.ndarray | float) -> np.ndarray:
    """Two turns (at 12 s and 30 s) on an otherwise straight flight, degrees clockwise from North."""
    t = np.asarray(t, dtype=np.float64)
    turn = lambda centre, deg, width: deg / (1.0 + np.exp(-(t - centre) / width))
    return 40.0 + turn(12.0, 70.0, 0.8) + turn(30.0, -110.0, 1.0)


def _frame(heading_deg: float) -> np.ndarray:
    """A nadir view of the textured ground, rotated with the camera heading."""
    h, w = _TEXTURE.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), heading_deg, 1.0)
    rotated = cv2.warpAffine(_TEXTURE, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated[h // 2 - 240 : h // 2 + 240, w // 2 - 320 : w // 2 + 320].copy()


def _keyframes(times: list[float]) -> tuple[list[np.ndarray], list[Keyframe]]:
    images = [_frame(float(_heading(t))) for t in times]
    kfs = [
        Keyframe(frame_index=i, timestamp=t, metrics=FrameMetrics(i, t, 100.0, 1.0, 120.0, 0.0))
        for i, t in enumerate(times)
    ]
    return images, kfs


def _telemetry(lag_s: float, t_end: float = 45.0) -> list[TelemetrySample]:
    """Samples stamped ``lag_s`` AHEAD of the video: stamp t describes video time t - lag_s."""
    out = []
    for t in np.arange(0.0, t_end, 0.2):
        video_t = t - lag_s
        north = 15.0 * video_t
        out.append(
            TelemetrySample(
                timestamp=float(t),
                geo=GeoPoint(lat=41.77 + north / 111_320.0, lon=-0.74, alt_msl=433.0, alt_rel=108.0),
                gimbal_pitch=-90.0,
                gimbal_roll=0.0,
                gimbal_yaw=float((_heading(video_t) + 180.0) % 360.0 - 180.0),
            )
        )
    return out


_TIMES = [float(t) for t in np.arange(2.0, 40.0, 1.5)]


def test_image_rotation_recovers_a_known_rotation():
    a, b = _frame(10.0), _frame(27.5)
    angle = timesync.image_rotation_deg(a, b)
    assert angle is not None
    assert abs(abs(angle) - 17.5) < 0.3


def test_lag_is_recovered_from_two_turns():
    images, kfs = _keyframes(_TIMES)
    est = timesync.estimate_from_keyframes(images, [k.timestamp for k in kfs], _telemetry(1.3), (-10.0, 10.0))
    assert est is not None and est.confident, est
    assert est.lag_s == pytest.approx(1.3, abs=0.1)
    assert est.heading_source == "gimbal_yaw"
    assert est.turning_pairs >= 3


def test_straight_flight_is_not_confident():
    times = _TIMES
    images = [_frame(40.0) for _ in times]
    samples = _telemetry(0.0)
    for s in samples:
        s.gimbal_yaw = 40.0
    est = timesync.estimate_from_keyframes(images, times, samples, (-10.0, 10.0))
    assert est is not None
    assert not est.confident
    assert "turned" in est.reason


def test_gps_course_is_used_without_gimbal_yaw():
    samples = _telemetry(0.0)
    for s in samples:
        s.gimbal_yaw = None
    track = timesync.heading_track(samples)
    assert track is not None and track[2] == "gps_course"


def _state(offset_source: str, lag_s: float, mode: str = "auto") -> PipelineState:
    images, kfs = _keyframes(_TIMES)
    cfg = Config()
    cfg.ingest.auto_sync = mode
    state = PipelineState(video_path="v.mp4", telemetry_path="t.csv", config=cfg, backbone_name="null")
    state.telemetry_samples = _telemetry(lag_s)
    state.telemetry_stats = {"time_offset_s": 0.0, "offset_source": offset_source, "format": "csv"}
    state.keyframes = kfs
    state.keyframe_cache = type("Cache", (), {"get": lambda self, i: images[i], "__len__": lambda self: len(images)})()
    return state


def test_unmeasured_offset_is_replaced_and_keyframes_rebuilt():
    state = _state("assumed_zero", 2.0)
    before = [s.timestamp for s in state.telemetry_samples[:3]]
    _artifacts, message = TimeSyncStage().run(state, None, None)
    sync = state.telemetry_stats["time_sync"]
    assert sync["applied"] and sync["confident"]
    assert state.telemetry_stats["offset_source"] == "image_motion"
    assert state.telemetry_stats["time_offset_s"] == pytest.approx(2.0, abs=0.1)
    after = [s.timestamp for s in state.telemetry_samples[:3]]
    assert after[0] == pytest.approx(before[0] - sync["lag_s"])
    # Keyframe telemetry now describes the keyframe's own instant.
    kf = state.keyframes[5]
    assert kf.telemetry is not None
    expected = (_heading(kf.timestamp) + 180.0) % 360.0 - 180.0
    assert math.isclose(kf.telemetry.gimbal_yaw, float(expected), abs_tol=1.0)
    assert kf.pose is not None
    assert "corrected" in message


def test_explicit_offset_is_kept_but_disagreement_reported():
    state = _state("explicit", 1.5)
    TimeSyncStage().run(state, None, None)
    sync = state.telemetry_stats["time_sync"]
    assert not sync["applied"]
    assert sync["disagrees"]
    assert state.telemetry_stats["offset_source"] == "explicit"
    assert state.telemetry_stats["time_offset_s"] == 0.0


def test_correct_mode_applies_over_an_explicit_offset():
    state = _state("explicit", 1.5, mode="correct")
    TimeSyncStage().run(state, None, None)
    assert state.telemetry_stats["time_sync"]["applied"]
    assert state.telemetry_stats["time_offset_s"] == pytest.approx(1.5, abs=0.1)


@pytest.mark.parametrize(("source", "mode"), [("video_clock", "auto"), ("explicit", "off")])
def test_skipped_when_there_is_nothing_to_measure(source, mode):
    with pytest.raises(StageUnavailable):
        TimeSyncStage().run(_state(source, 0.0, mode=mode), None, None)


def test_report_card_shows_the_measurement():
    state = _state("explicit", 1.5)
    TimeSyncStage().run(state, None, None)
    text = render_report_text(
        build_report(
            {
                "telemetry_offset_s": 0.0,
                "telemetry_offset_source": "explicit",
                "telemetry_format": "csv",
                "time_sync": state.telemetry_stats["time_sync"],
            }
        )
    )
    assert "Image-motion check: WARNING" in text
    html = render_report_html(build_report({"time_sync": state.telemetry_stats["time_sync"]}))
    assert "Image-motion check" in html
