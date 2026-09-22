"""Tests for drishti3d.ingest: video decoding, telemetry parsing, and intrinsics priors.

All fixtures are synthesized in-test (no external data files), per project convention.
"""

from __future__ import annotations

import csv as csv_module
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest

from drishti3d.ingest.intrinsics import intrinsics_from_video
from drishti3d.ingest.telemetry import (
    VideoSegment,
    detect_video_segments,
    load_telemetry,
    parse_srt_string,
    resample_telemetry,
)
from drishti3d.ingest.video import VideoSource
from drishti3d.types import GeoPoint, TelemetrySample

# ---------------------------------------------------------------------------
# Synthetic video fixture
# ---------------------------------------------------------------------------

_VIDEO_WIDTH = 96
_VIDEO_HEIGHT = 64
_VIDEO_FPS = 15
_VIDEO_N_FRAMES = 60


def _synthetic_video_frame(i: int, width: int, height: int) -> np.ndarray:
    """A textured, time-varying BGR frame: checkerboard base + moving markers."""
    tile = 8
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    checker = (((xx // tile) + (yy // tile)) % 2 * 200).astype(np.uint8)
    img = np.stack([checker, checker, checker], axis=-1).copy()

    cx = (i * 3) % max(width - 20, 1)
    cy = (i * 2) % max(height - 20, 1)
    img[cy : cy + 10, cx : cx + 10] = (0, 0, 255)
    return img


def _make_synthetic_video(path: Path, n_frames: int = _VIDEO_N_FRAMES) -> None:
    """Encode a short synthetic mp4 with PyAV: CFR, no B-frames (so decode order == presentation order)."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=_VIDEO_FPS)
    stream.width = _VIDEO_WIDTH
    stream.height = _VIDEO_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.codec_context.max_b_frames = 0

    for i in range(n_frames):
        arr = _synthetic_video_frame(i, _VIDEO_WIDTH, _VIDEO_HEIGHT)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.mp4"
    _make_synthetic_video(path)
    return path


# ---------------------------------------------------------------------------
# VideoSource
# ---------------------------------------------------------------------------


def test_video_source_dimensions_fps_duration(synthetic_video_path: Path) -> None:
    with VideoSource(synthetic_video_path) as video:
        assert video.width == _VIDEO_WIDTH
        assert video.height == _VIDEO_HEIGHT
        assert video.fps == pytest.approx(_VIDEO_FPS, rel=0.05)
        assert video.rotation == 0
        # frame_count may be estimated, but should be close to the true count.
        assert abs(video.frame_count - _VIDEO_N_FRAMES) <= 1
        assert video.duration == pytest.approx(_VIDEO_N_FRAMES / _VIDEO_FPS, rel=0.1)


def test_video_source_timestamps_monotonically_increasing(synthetic_video_path: Path) -> None:
    with VideoSource(synthetic_video_path) as video:
        frames = list(video.iter_frames())

    assert len(frames) == _VIDEO_N_FRAMES
    timestamps = [f.timestamp for f in frames]
    assert all(b > a for a, b in pairwise(timestamps))
    # CFR source: timestamps should track i / fps closely.
    for i, t in enumerate(timestamps):
        assert t == pytest.approx(i / _VIDEO_FPS, abs=1e-3)


def test_video_source_read_frames_matches_sequential_iteration(synthetic_video_path: Path) -> None:
    with VideoSource(synthetic_video_path) as video:
        sequential = {f.index: f.image for f in video.iter_frames()}
        target_indices = [0, 5, 13, 40, 59]
        random_access = video.read_frames(target_indices)

    assert [f.index for f in random_access] == target_indices
    for f in random_access:
        assert f.image is not None
        np.testing.assert_array_equal(f.image, sequential[f.index])


def test_video_source_iter_frames_downscale(synthetic_video_path: Path) -> None:
    with VideoSource(synthetic_video_path) as video:
        frames = list(video.iter_frames(downscale=32))
    for f in frames:
        assert f.image is not None
        assert max(f.image.shape[:2]) <= 32


# ---------------------------------------------------------------------------
# DJI SRT telemetry
# ---------------------------------------------------------------------------

_SRT_TEXT = """1
00:00:00,000 --> 00:00:00,033
<font size="36">SrtCnt : 1, DiffTime : 33ms
2024-01-01 10:00:00,000
[iso : 100] [shutter : 1/500.0] [fnum : 280] [ev : 0] [focal_len : 24.00] [latitude: 22.543200] [longitude: 114.056800] [rel_alt: 10.500 abs_alt: 50.200] </font>

2
00:00:00,033 --> 00:00:00,066
[lat : 22.601500] [long : 114.101200] [altitude : 60.750] [focal_len : 4.50]

3
this cue has a deliberately corrupt timecode line and no arrow
[latitude: 10.0] [longitude: 20.0]
"""


def test_parse_srt_modern_and_legacy_layouts() -> None:
    samples, stats = parse_srt_string(_SRT_TEXT)

    assert stats["format"] == "dji_srt"
    assert stats["records_total"] == 3
    assert stats["records_parsed"] == 2
    assert stats["records_skipped"] == 1  # the corrupt cue, skipped rather than raising

    assert len(samples) == 2

    modern, legacy = samples
    assert modern.timestamp == pytest.approx(0.0, abs=1e-3)
    assert modern.geo is not None
    assert modern.geo.lat == pytest.approx(22.5432)
    assert modern.geo.lon == pytest.approx(114.0568)
    assert modern.geo.alt_msl == pytest.approx(50.2)
    assert modern.geo.alt_rel == pytest.approx(10.5)

    assert legacy.timestamp == pytest.approx(0.033, abs=1e-3)
    assert legacy.geo is not None
    assert legacy.geo.lat == pytest.approx(22.6015)
    assert legacy.geo.lon == pytest.approx(114.1012)
    assert legacy.geo.alt_msl == pytest.approx(60.75)

    # focal_len from both cues (24.00, 4.50) surfaced via stats for intrinsics.
    assert stats["focal_len_mm"] == pytest.approx(np.median([24.0, 4.5]))


def test_parse_srt_corrupt_line_does_not_raise() -> None:
    text = "1\nnot a timecode at all\ngarbage\n"
    samples, stats = parse_srt_string(text)  # must not raise
    assert samples == []
    assert stats["records_skipped"] == 1


# ---------------------------------------------------------------------------
# Airdata-style CSV telemetry: video-start offset auto-detection
# ---------------------------------------------------------------------------

_AIRDATA_HEADER = [
    "time(millisecond)",
    "latitude",
    "longitude",
    "altitude_above_seaLevel(feet)",
    "height_above_takeoff(feet)",
    "isVideo",
    "isPhoto",
    "pitch(degrees)",
    "roll(degrees)",
    "compass_heading(degrees)",
    "gimbal_pitch(degrees)",
    "gimbal_heading(degrees)",
]


def _airdata_row(
    t_s: float,
    is_video: bool,
    lat: float = 12.0,
    lon: float = 77.0,
    alt_msl_ft: float = 100.0,
    alt_rel_ft: float = 10.0,
    pitch: float = 1.0,
    roll: float = 2.0,
    compass: float = 3.0,
    gimbal_pitch: float = -4.0,
    gimbal_heading: float = 5.0,
) -> list[str]:
    return [
        str(round(t_s * 1000)),
        f"{lat:.7f}",
        f"{lon:.7f}",
        f"{alt_msl_ft:.3f}",
        f"{alt_rel_ft:.3f}",
        "1" if is_video else "0",
        "0",
        f"{pitch:.2f}",
        f"{roll:.2f}",
        f"{compass:.2f}",
        f"{gimbal_pitch:.2f}",
        f"{gimbal_heading:.2f}",
    ]


def _write_airdata_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_csv_isvideo_offset_autodetected_and_timestamps_rebased(tmp_path: Path) -> None:
    """A single isVideo window whose duration matches the video is auto-detected and used to re-base."""
    path = tmp_path / "airdata.csv"
    rows = []
    # Ground/pre-roll telemetry (not recording): t = 0..4s.
    for t in range(5):
        rows.append(_airdata_row(float(t), is_video=False))
    # Recording window: t = 5..15s inclusive (duration 10s).
    for t in range(5, 16):
        rows.append(_airdata_row(float(t), is_video=True))
    # Post-roll (landing, not recording): t = 16..20s.
    for t in range(16, 21):
        rows.append(_airdata_row(float(t), is_video=False))
    _write_airdata_csv(path, _AIRDATA_HEADER, rows)

    samples, stats = load_telemetry(path, video_duration_s=10.0)

    assert stats["format"] == "csv"
    assert stats["offset_source"] == "isVideo_autodetect"
    assert stats["time_offset_s"] == pytest.approx(5.0)

    # Every sample's timestamp has been re-based so the recording window
    # now starts at (approximately) t=0.
    video_window_samples = [s for s in samples if 0.0 <= s.timestamp <= 10.0 + 1e-6]
    assert len(video_window_samples) == 11
    assert min(s.timestamp for s in samples) == pytest.approx(-5.0)
    assert max(s.timestamp for s in samples) == pytest.approx(15.0)

    assert stats["telemetry_video_coverage_fraction"] == pytest.approx(1.0)


def test_detect_video_segments_returns_all_contiguous_runs() -> None:
    """A flight log with several separate recordings reports every one, not just the first/longest."""
    rows = (
        [{"timestamp": float(t), "is_video": False} for t in range(3)]
        + [{"timestamp": float(t), "is_video": True} for t in range(3, 8)]  # 3..7 -> duration 4
        + [{"timestamp": float(t), "is_video": False} for t in range(8, 12)]
        + [{"timestamp": float(t), "is_video": True} for t in range(12, 20)]  # 12..19 -> duration 7
        + [{"timestamp": float(t), "is_video": False} for t in range(20, 22)]
    )

    segments = detect_video_segments(rows)

    assert len(segments) == 2
    assert all(isinstance(s, VideoSegment) for s in segments)
    assert segments[0].start_s == pytest.approx(3.0)
    assert segments[0].end_s == pytest.approx(7.0)
    assert segments[0].duration_s == pytest.approx(4.0)
    assert segments[1].start_s == pytest.approx(12.0)
    assert segments[1].end_s == pytest.approx(19.0)
    assert segments[1].duration_s == pytest.approx(7.0)


def test_csv_multiple_segments_all_surfaced_and_correct_one_autodetected(tmp_path: Path) -> None:
    """When several recordings exist, only the one matching the video's duration is auto-selected."""
    path = tmp_path / "airdata_multi.csv"
    rows = []
    for t in range(3):
        rows.append(_airdata_row(float(t), is_video=False))
    # First recording: t = 3..7 (duration 4s) -- does NOT match the video.
    for t in range(3, 8):
        rows.append(_airdata_row(float(t), is_video=True))
    for t in range(8, 12):
        rows.append(_airdata_row(float(t), is_video=False))
    # Second recording: t = 12..19 (duration 7s) -- matches the 7s video.
    for t in range(12, 20):
        rows.append(_airdata_row(float(t), is_video=True))
    for t in range(20, 22):
        rows.append(_airdata_row(float(t), is_video=False))
    _write_airdata_csv(path, _AIRDATA_HEADER, rows)

    samples, stats = load_telemetry(path, video_duration_s=7.0)

    assert len(stats["video_segments"]) == 2
    assert stats["offset_source"] == "isVideo_autodetect"
    assert stats["time_offset_s"] == pytest.approx(12.0)
    assert samples[0].timestamp == pytest.approx(0.0 - 12.0)


def test_explicit_telemetry_offset_overrides_autodetection(tmp_path: Path) -> None:
    path = tmp_path / "airdata.csv"
    rows = [_airdata_row(float(t), is_video=(5 <= t <= 15)) for t in range(21)]
    _write_airdata_csv(path, _AIRDATA_HEADER, rows)

    # Auto-detection would pick offset=5.0 (see the first test above); an
    # explicit override must win regardless.
    samples, stats = load_telemetry(path, video_duration_s=10.0, time_offset_s=1.5)

    assert stats["offset_source"] == "explicit"
    assert stats["time_offset_s"] == pytest.approx(1.5)
    assert samples[0].timestamp == pytest.approx(0.0 - 1.5)


def test_csv_without_isvideo_column_assumes_zero_offset(tmp_path: Path) -> None:
    path = tmp_path / "no_isvideo.csv"
    header = ["time(millisecond)", "latitude", "longitude"]
    rows = [[str(t * 1000), "12.0", "77.0"] for t in range(5)]
    _write_airdata_csv(path, header, rows)

    samples, stats = load_telemetry(path, video_duration_s=4.0)

    assert stats["offset_source"] == "assumed_zero"
    assert stats["time_offset_s"] == pytest.approx(0.0)
    assert "video_segments" not in stats
    assert [s.timestamp for s in samples] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_csv_feet_to_metres_conversion(tmp_path: Path) -> None:
    path = tmp_path / "altitude.csv"
    header = [
        "time(millisecond)",
        "latitude",
        "longitude",
        "altitude_above_seaLevel(feet)",
        "height_above_takeoff(feet)",
    ]
    # 1000 ft MSL, 500 ft AGL/relative.
    rows = [["0", "12.0", "77.0", "1000.0", "500.0"]]
    _write_airdata_csv(path, header, rows)

    samples, _stats = load_telemetry(path)

    assert len(samples) == 1
    geo = samples[0].geo
    assert geo is not None
    assert geo.alt_msl == pytest.approx(1000.0 * 0.3048)
    assert geo.alt_rel == pytest.approx(500.0 * 0.3048)


def test_csv_prefers_gimbal_columns_over_aircraft_body_columns(tmp_path: Path) -> None:
    """When both an aircraft-body and a gimbal/camera reading are present, the gimbal one wins.

    Real Airdata exports report both ``pitch(degrees)``/``compass_heading(degrees)``
    (aircraft body) and ``gimbal_pitch(degrees)``/``gimbal_heading(degrees)``
    (camera gimbal) -- ``TelemetrySample.gimbal_pitch``/``gimbal_yaw`` must
    reflect the camera's own attitude, not the aircraft's, regardless of
    which column happens to come first in the CSV.
    """
    path = tmp_path / "gimbal_priority.csv"
    rows = [_airdata_row(0.0, is_video=False, pitch=17.0, compass=170.0, gimbal_pitch=-30.0, gimbal_heading=270.0)]
    _write_airdata_csv(path, _AIRDATA_HEADER, rows)

    samples, _stats = load_telemetry(path)

    assert samples[0].gimbal_pitch == pytest.approx(-30.0)
    assert samples[0].gimbal_yaw == pytest.approx(270.0)


# ---------------------------------------------------------------------------
# Real drone data (optional): sih_data_samples/
# ---------------------------------------------------------------------------

_REAL_DATA_DIR = Path("/Users/ajith/Desktop/sih_data_samples")
_REAL_CSV_PATH = _REAL_DATA_DIR / "Aug-30th-2022-12-59PM-Flight-Airdata.csv"
_REAL_VIDEO_DURATION_S = 242.08  # DJI_0753.MP4: 7255 frames @ 29.97fps


@pytest.mark.skipif(not _REAL_CSV_PATH.exists(), reason="real drone data sample not present on this machine")
def test_real_airdata_csv_offset_matches_known_video_start() -> None:
    samples, stats = load_telemetry(_REAL_CSV_PATH, video_duration_s=_REAL_VIDEO_DURATION_S)

    assert stats["format"] == "csv"
    assert stats["offset_source"] == "isVideo_autodetect"
    assert stats["time_offset_s"] == pytest.approx(42.9, abs=1.0)

    segments = stats["video_segments"]
    assert len(segments) == 1
    assert segments[0]["duration_s"] == pytest.approx(242.4, abs=1.0)

    assert samples  # sanity: real file actually parsed rows


# ---------------------------------------------------------------------------
# resample_telemetry
# ---------------------------------------------------------------------------


def test_resample_telemetry_midpoint_and_out_of_span() -> None:
    samples = [
        TelemetrySample(timestamp=0.0, geo=GeoPoint(lat=10.0, lon=20.0, alt_msl=100.0)),
        TelemetrySample(timestamp=10.0, geo=GeoPoint(lat=10.001, lon=20.001, alt_msl=110.0)),
    ]

    results = resample_telemetry(samples, [-5.0, 5.0, 15.0])

    assert results[0] is None  # before span
    assert results[2] is None  # after span

    mid = results[1]
    assert mid is not None
    assert mid.geo is not None
    # Interpolation happens in a local metric (ENU) frame, but for a small
    # displacement like this the recovered midpoint should still land very
    # close to the arithmetic midpoint in lat/lon/alt.
    assert mid.geo.lat == pytest.approx(0.5 * (10.0 + 10.001), abs=1e-5)
    assert mid.geo.lon == pytest.approx(0.5 * (20.0 + 20.001), abs=1e-5)
    assert mid.geo.alt_msl == pytest.approx(105.0, abs=1e-2)


def test_resample_telemetry_empty_inputs() -> None:
    assert resample_telemetry([], [1.0, 2.0]) == [None, None]
    samples = [TelemetrySample(timestamp=0.0, geo=GeoPoint(lat=0.0, lon=0.0, alt_msl=0.0))]
    assert resample_telemetry(samples, []) == []


# ---------------------------------------------------------------------------
# intrinsics
# ---------------------------------------------------------------------------


def test_intrinsics_from_video_default_guess(synthetic_video_path: Path) -> None:
    with VideoSource(synthetic_video_path) as video:
        intrinsics, provenance = intrinsics_from_video(video)

    assert provenance == "default_guess"
    assert intrinsics.width == _VIDEO_WIDTH
    assert intrinsics.height == _VIDEO_HEIGHT
    assert intrinsics.fx > 0
    assert intrinsics.fx == pytest.approx(intrinsics.fy)


def test_intrinsics_from_video_camera_db(synthetic_video_path: Path) -> None:
    with VideoSource(synthetic_video_path) as video:
        intrinsics, provenance = intrinsics_from_video(video, camera_model="DJI Mavic 3")

    assert provenance == "camera_db"
    expected_fx = 12.29 / 17.3 * _VIDEO_WIDTH
    assert intrinsics.fx == pytest.approx(expected_fx)


def test_intrinsics_from_video_srt_focal_len_preferred_over_db(synthetic_video_path: Path) -> None:
    telemetry = ([], {"focal_len_mm": 10.0})
    with VideoSource(synthetic_video_path) as video:
        intrinsics, provenance = intrinsics_from_video(video, telemetry=telemetry, camera_model="DJI Mavic 3")

    assert provenance == "srt_focal_len"
    expected_fx = 10.0 / 17.3 * _VIDEO_WIDTH
    assert intrinsics.fx == pytest.approx(expected_fx)
