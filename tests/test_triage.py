"""Tests for drishti3d.triage: per-frame quality metrics and keyframe selection.

All fixtures are synthesized in-test (no external data files), per project convention.
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from drishti3d.config import TriageConfig
from drishti3d.ingest.video import VideoSource
from drishti3d.triage.metrics import (
    blur_score,
    estimate_parallax_detailed,
    estimate_rotation_compensated_parallax,
)
from drishti3d.triage.selector import select_keyframes, triage_report
from drishti3d.types import (
    CameraIntrinsics,
    FrameMetrics,
    GeoPoint,
    Keyframe,
    TelemetrySample,
)

# ---------------------------------------------------------------------------
# blur_score
# ---------------------------------------------------------------------------


def _random_textured_gray(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return cv2.GaussianBlur(base, (3, 3), 0)


def test_blur_score_ranks_sharp_above_blurred() -> None:
    sharp = _random_textured_gray(240, 160, seed=3)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)

    assert blur_score(sharp) > blur_score(blurred)


# ---------------------------------------------------------------------------
# estimate_rotation_compensated_parallax
# ---------------------------------------------------------------------------


def test_rotation_compensated_parallax_near_zero_for_pure_homography() -> None:
    width, height = 240, 160
    prev = _random_textured_gray(width, height, seed=2)

    # A pure rotation + translation *is* a homography, so fitting one and
    # measuring the residual should leave almost nothing.
    m = cv2.getRotationMatrix2D((width / 2, height / 2), 3.0, 1.0)
    m[:, 2] += [5, -3]
    curr = cv2.warpAffine(prev, m, (width, height), borderMode=cv2.BORDER_REFLECT)

    residual = estimate_rotation_compensated_parallax(prev, curr)
    assert residual < 0.5


def test_rotation_compensated_parallax_positive_for_genuine_depth_disparity() -> None:
    width, height = 240, 160
    prev = _random_textured_gray(width, height, seed=2)

    # Two interleaved depth layers (checkerboard of tiles) shifted by
    # different, opposite-signed amounts. No single global homography (a
    # smooth function of image position) can explain a spatially
    # high-frequency, sign-flipping displacement field, so this is genuine
    # depth-induced parallax -- unlike the pure-homography case above.
    grid = 8
    dx_near, dx_far = 8.0, -8.0
    tile_w, tile_h = width // grid, height // grid
    curr = prev.copy()
    for r in range(grid):
        for c in range(grid):
            dx = dx_near if (r + c) % 2 == 0 else dx_far
            y0, y1 = r * tile_h, height if r == grid - 1 else (r + 1) * tile_h
            x0, x1 = c * tile_w, width if c == grid - 1 else (c + 1) * tile_w
            m = np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32)
            curr[y0:y1, x0:x1] = cv2.warpAffine(
                prev[y0:y1, x0:x1], m, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REFLECT
            )

    residual_pure_baseline = estimate_rotation_compensated_parallax(
        prev, cv2.warpAffine(prev, np.array([[1, 0, 5], [0, 1, -3]], dtype=np.float32), (width, height))
    )
    residual_depth = estimate_rotation_compensated_parallax(prev, curr)

    assert residual_depth > 0.2
    assert residual_depth > 2 * residual_pure_baseline


# ---------------------------------------------------------------------------
# estimate_parallax_detailed / ParallaxEstimate.useful_baseline_px (DEFECT 2)
# ---------------------------------------------------------------------------

# Perspective homography used for the "planar scene, translating" tests
# below: a real camera translation over a (near-)plane is *exactly* a
# homography, same as pure rotation, so this is chosen to have enough
# perspective distortion that the residual after fitting is a small but
# clearly-nonzero, comfortably-above-the-noise-floor value (~0.1-0.15px)
# rather than sitting right at the edge of tracking noise.
_PLANE_HOMOGRAPHY = np.array(
    [
        [1.0, 0.03, 10.0],
        [0.015, 1.0, 4.0],
        [0.0004, 0.0002, 1.0],
    ]
)


def test_parallax_detailed_planar_translation_yields_nonzero_useful_baseline() -> None:
    """DEFECT 2's core bug: translation over a planar scene collapses the
    homography residual toward zero, exactly like pure rotation does --
    that's the design flaw. A structured estimate must still recognize
    real motion as a usable baseline via raw displacement + a high inlier
    ratio, even though the old residual-only signal alone would call this
    "no parallax".
    """
    width, height = 240, 160
    prev = _random_textured_gray(width, height, seed=11)
    curr = cv2.warpPerspective(prev, _PLANE_HOMOGRAPHY, (width, height), borderMode=cv2.BORDER_REFLECT)

    result = estimate_parallax_detailed(prev, curr)

    assert result.residual_px < 0.5  # the old, sole signal collapses toward (near-)zero
    assert result.raw_px > 3.0  # but real, substantial motion happened
    assert result.homography_inlier_ratio > 0.9  # the scene is well explained by one homography (it's planar)

    # The fix: useful_baseline_px must still be non-zero here.
    assert result.useful_baseline_px > 0.0
    assert result.useful_baseline_px == pytest.approx(result.raw_px)


def test_parallax_detailed_pure_rotation_yields_near_zero_useful_baseline() -> None:
    """A small in-place camera rotation (no translation) carries no
    reconstruction-useful baseline; useful_baseline_px must stay at ~0
    even though the raw/residual *signature* looks identical to the
    planar-translation case above (both fit a homography cleanly) -- the
    discriminator here is that the rotation is small enough that the raw
    displacement itself isn't significant.
    """
    width, height = 240, 160
    prev = _random_textured_gray(width, height, seed=11)
    m = cv2.getRotationMatrix2D((width / 2, height / 2), 0.5, 1.0)
    curr = cv2.warpAffine(prev, m, (width, height), borderMode=cv2.BORDER_REFLECT)

    result = estimate_parallax_detailed(prev, curr)

    assert result.useful_baseline_px == pytest.approx(0.0)


def test_parallax_detailed_depth_disparity_has_higher_residual_and_lower_inlier_ratio() -> None:
    """Genuine depth disparity (unlike the planar/rotation cases above)
    should not fit a single global homography nearly as cleanly: expect a
    higher residual and a lower inlier ratio than the planar-translation
    case, and a non-zero useful baseline.
    """
    width, height = 240, 160
    prev = _random_textured_gray(width, height, seed=11)

    grid = 8
    dx_near, dx_far = 8.0, -8.0
    tile_w, tile_h = width // grid, height // grid
    curr = prev.copy()
    for r in range(grid):
        for c in range(grid):
            dx = dx_near if (r + c) % 2 == 0 else dx_far
            y0, y1 = r * tile_h, height if r == grid - 1 else (r + 1) * tile_h
            x0, x1 = c * tile_w, width if c == grid - 1 else (c + 1) * tile_w
            mm = np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32)
            curr[y0:y1, x0:x1] = cv2.warpAffine(
                prev[y0:y1, x0:x1], mm, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REFLECT
            )

    depth_result = estimate_parallax_detailed(prev, curr)
    plane_curr = cv2.warpPerspective(prev, _PLANE_HOMOGRAPHY, (width, height), borderMode=cv2.BORDER_REFLECT)
    plane_result = estimate_parallax_detailed(prev, plane_curr)

    assert depth_result.residual_px > plane_result.residual_px
    assert depth_result.homography_inlier_ratio < plane_result.homography_inlier_ratio
    assert depth_result.useful_baseline_px > 0.0


# ---------------------------------------------------------------------------
# select_keyframes
# ---------------------------------------------------------------------------

_SEL_WIDTH, _SEL_HEIGHT = 240, 160
_SEL_GRID = 8
_SEL_DX_NEAR_PER_FRAME = 3.0
_SEL_DX_FAR_PER_FRAME = 0.4
_SEL_N_FRAMES = 90
_SEL_FPS = 15
_SEL_PAD = 60


def _base_texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(_SEL_HEIGHT + _SEL_PAD, _SEL_WIDTH + _SEL_PAD), dtype=np.uint8)
    return cv2.GaussianBlur(base, (3, 3), 0)


def _synthetic_parallax_frame(base: np.ndarray, i: int) -> np.ndarray:
    """Frame ``i`` of a synthetic dolly shot with genuine (checkerboard-tiled) depth parallax.

    Each frame is re-derived directly from ``base`` (not chained from the
    previous frame) so warp artifacts don't compound over many frames.
    Two interleaved "depth" layers drift apart at different per-frame
    rates, giving select_keyframes real accumulating rotation-compensated
    parallax to trigger on -- a flat, single-depth pan would be fully
    explained by a homography and would never trigger a keyframe.
    """
    tile_w, tile_h = _SEL_WIDTH // _SEL_GRID, _SEL_HEIGHT // _SEL_GRID
    out = np.zeros((_SEL_HEIGHT, _SEL_WIDTH), dtype=np.uint8)
    pad = _SEL_PAD // 2
    for r in range(_SEL_GRID):
        for c in range(_SEL_GRID):
            near = (r + c) % 2 == 0
            dx = (_SEL_DX_NEAR_PER_FRAME if near else _SEL_DX_FAR_PER_FRAME) * i
            y0, y1 = r * tile_h, _SEL_HEIGHT if r == _SEL_GRID - 1 else (r + 1) * tile_h
            x0, x1 = c * tile_w, _SEL_WIDTH if c == _SEL_GRID - 1 else (c + 1) * tile_w
            src_x0, src_x1 = x0 + pad, x1 + pad
            src_y0, src_y1 = y0 + pad, y1 + pad
            left = max(0, src_x0 - pad)
            padded = base[src_y0:src_y1, left : src_x1 + pad]
            m = np.array([[1, 0, -dx], [0, 1, 0]], dtype=np.float32)
            warped = cv2.warpAffine(padded, m, (padded.shape[1], padded.shape[0]), borderMode=cv2.BORDER_REFLECT)
            off = src_x0 - left
            out[y0:y1, x0:x1] = warped[:, off : off + (x1 - x0)]
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _make_parallax_video(path: Path, n_frames: int = _SEL_N_FRAMES) -> None:
    base = _base_texture()
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=_SEL_FPS)
    stream.width = _SEL_WIDTH
    stream.height = _SEL_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.codec_context.max_b_frames = 0

    for i in range(n_frames):
        arr = _synthetic_parallax_frame(base, i)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def parallax_video_path(tmp_path: Path) -> Path:
    path = tmp_path / "parallax.mp4"
    _make_parallax_video(path)
    return path


def test_select_keyframes_returns_fewer_unique_increasing(parallax_video_path: Path) -> None:
    config = TriageConfig(target_keyframes=8, min_blur_score=50.0, min_parallax_px=8.0, max_frames_scanned=200)

    with VideoSource(parallax_video_path) as video:
        keyframes, all_metrics = select_keyframes(video, config)

        assert 0 < len(keyframes) < video.frame_count

        indices = [kf.frame_index for kf in keyframes]
        assert len(set(indices)) == len(indices)  # no duplicates
        assert indices == sorted(indices)  # strictly increasing order
        assert all(b > a for a, b in pairwise(indices))

        report = triage_report(keyframes, all_metrics)
        assert report["keyframes_selected"] == len(keyframes)
        assert report["frames_scanned"] == len(all_metrics)
        assert report["blur_selected_mean"] > 0


def test_select_keyframes_respects_target_keyframes_cap(parallax_video_path: Path) -> None:
    config = TriageConfig(target_keyframes=3, min_blur_score=50.0, min_parallax_px=5.0, max_frames_scanned=200)

    with VideoSource(parallax_video_path) as video:
        keyframes, _ = select_keyframes(video, config)

    assert len(keyframes) <= 3


def test_select_keyframes_progress_callback_invoked(parallax_video_path: Path) -> None:
    calls: list[tuple[int, int, str]] = []

    def progress_cb(current: int, total: int, message: str) -> None:
        calls.append((current, total, message))

    config = TriageConfig(target_keyframes=4, min_blur_score=50.0, min_parallax_px=8.0, max_frames_scanned=200)
    with VideoSource(parallax_video_path) as video:
        select_keyframes(video, config, progress_cb=progress_cb)

    assert len(calls) > 0
    assert any("accepted keyframe" in c[2] for c in calls)


# ---------------------------------------------------------------------------
# triage_report (DEFECT 1)
# ---------------------------------------------------------------------------


def test_triage_report_blur_selected_uses_keyframe_own_metrics() -> None:
    """Regression for the confirmed DEFECT 1 bug (exact repro from the brief).

    `triage_report` used to look selected-frame blur stats up in
    `all_metrics` by index. `all_metrics` only ever holds checkpoint-cadence
    frames, while `_pick_sharpest_in_window` accepts a keyframe at a
    *different* native index (checkpoint +/- window radius) -- once the
    checkpoint stride is > 1 (i.e. on any realistically long video), that
    index is almost never a member of `all_metrics`, so the lookup silently
    returned an empty list and `blur_selected_mean`/`blur_selected_min`
    reported 0.0 even though the keyframes carry real, much sharper blur
    scores on their own `metrics`.
    """
    scanned = [
        FrameMetrics(
            index=i, timestamp=i / 30, blur_score=500.0, exposure_score=1.0, mean_luma=128.0, estimated_parallax=3.0
        )
        for i in range(0, 100, 5)  # a scan_step > 1 -style coarse cadence
    ]
    keyframes = [
        Keyframe(
            frame_index=i,
            timestamp=i / 30,
            metrics=FrameMetrics(
                index=i, timestamp=i / 30, blur_score=900.0, exposure_score=1.0, mean_luma=128.0, estimated_parallax=25.0
            ),
        )
        for i in (12, 27, 43)  # none of these are multiples of 5 -> never in `scanned`
    ]

    report = triage_report(keyframes, scanned)

    assert report["blur_selected_mean"] == pytest.approx(900.0)
    assert report["blur_selected_min"] == pytest.approx(900.0)
    assert report["blur_rejected_mean"] == pytest.approx(500.0)
    assert report["blur_selected_mean"] > 0.0
    assert report["blur_selected_mean"] != report["blur_rejected_mean"]


# ---------------------------------------------------------------------------
# GPS-driven baseline vs. vision-only starvation (DEFECT 2, selector-level)
# ---------------------------------------------------------------------------

_METERS_PER_DEG_LAT = 111320.0


def _flat_pan_frame(base: np.ndarray, i: int, px_per_frame: float) -> np.ndarray:
    """A flat, single-depth pan: the whole frame translates uniformly.

    Fully explained by one homography every step -- exactly DEFECT 2's
    headline scenario (nadir flight over flat terrain). ``px_per_frame`` is
    kept small enough that even the raw (uncompensated) per-checkpoint
    pixel displacement stays below the "significant motion" floor, so a
    vision-only accumulator has nothing to accumulate at all, not even the
    hybrid raw-displacement fallback -- only a real metric (GPS) signal can
    resolve this.
    """
    pad = _SEL_PAD // 2
    cropped = base[pad : pad + _SEL_HEIGHT, pad : pad + _SEL_WIDTH]
    dx = px_per_frame * i
    m = np.array([[1, 0, -dx], [0, 1, 0]], dtype=np.float32)
    warped = cv2.warpAffine(cropped, m, (_SEL_WIDTH, _SEL_HEIGHT), borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def _make_flat_pan_video(path: Path, n_frames: int, fps: int, px_per_frame: float) -> None:
    base = _base_texture()
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = _SEL_WIDTH
    stream.height = _SEL_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.codec_context.max_b_frames = 0
    for i in range(n_frames):
        arr = _flat_pan_frame(base, i, px_per_frame)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _make_telemetry_for_flat_pan(n_frames: int, fps: float, meters_per_frame: float) -> list[TelemetrySample]:
    """Synthetic telemetry advancing at a constant real metric rate, independent of pixel motion."""
    return [
        TelemetrySample(
            timestamp=i / fps,
            geo=GeoPoint(lat=12.0 + (meters_per_frame * i) / _METERS_PER_DEG_LAT, lon=77.0, alt_msl=550.0, alt_rel=50.0),
        )
        for i in range(n_frames)
    ]


def test_select_keyframes_gps_baseline_prevents_starvation_on_flat_scene(tmp_path: Path) -> None:
    """DEFECT 2's headline design flaw, demonstrated end-to-end: a flat,
    single-depth pan starves the vision-only accumulator completely (it
    never even clears the raw-displacement significance floor, since real
    per-frame motion here is sub-pixel), which would mean *zero* keyframes
    ever get selected over flat terrain -- the common nadir-drone case.
    Feeding the same footage real (if pixel-invisible) GPS motion must
    fix this, since a metric baseline doesn't depend on scene depth at all.
    """
    n_frames, fps = 60, 15
    path = tmp_path / "flat_pan.mp4"
    _make_flat_pan_video(path, n_frames=n_frames, fps=fps, px_per_frame=0.3)

    vision_config = TriageConfig(
        target_keyframes=10, min_blur_score=1.0, min_parallax_px=8.0, max_frames_scanned=200, use_gps_baseline=False
    )
    with VideoSource(path) as video:
        vision_keyframes, _ = select_keyframes(video, vision_config)
    assert vision_keyframes == []  # confirms the flaw this fix addresses

    telemetry = _make_telemetry_for_flat_pan(n_frames, fps, meters_per_frame=0.5)
    gps_config = TriageConfig(
        target_keyframes=10, min_blur_score=1.0, min_parallax_px=8.0, max_frames_scanned=200, use_gps_baseline=True
    )
    with VideoSource(path) as video:
        gps_keyframes, gps_all_metrics = select_keyframes(video, gps_config, telemetry=telemetry)

    assert len(gps_keyframes) > 0
    report = triage_report(gps_keyframes, gps_all_metrics)
    assert report["spacing_mode"] == "gps_baseline"


# ---------------------------------------------------------------------------
# Ring-buffered window search avoids random-access re-decode (DEFECT 3)
# ---------------------------------------------------------------------------


def test_select_keyframes_ring_buffer_avoids_random_access_redecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEFECT 3 regression: the old implementation re-decoded a native-
    resolution window via `VideoSource.read_frames` (seek + GOP replay) on
    every single keyframe trigger -- on a longer video with many triggers,
    thousands of seek+decode operations dominating runtime. The
    ring-buffer fix should need zero calls to `read_frames` on a clean,
    well-textured synthetic video: nothing should ever miss the blur floor
    and need the random-access fallback.
    """
    path = tmp_path / "long_parallax.mp4"
    _make_parallax_video(path, n_frames=320)

    call_count = 0
    original_read_frames = VideoSource.read_frames

    def counting_read_frames(self: VideoSource, indices: list[int]) -> list:
        nonlocal call_count
        call_count += 1
        return original_read_frames(self, indices)

    monkeypatch.setattr(VideoSource, "read_frames", counting_read_frames)

    config = TriageConfig(target_keyframes=20, min_blur_score=1.0, min_parallax_px=6.0, max_frames_scanned=400)
    with VideoSource(path) as video:
        keyframes, _ = select_keyframes(video, config)

    assert len(keyframes) > 5
    assert call_count == 0


# ---------------------------------------------------------------------------
# Relative (not absolute) blur gate (DEFECT 4)
# ---------------------------------------------------------------------------


def test_relative_blur_threshold_is_the_real_gate_not_absolute_min_blur_score() -> None:
    """DEFECT 4: `blur_score` mixes two differently-scaled terms (variance-
    of-Laplacian and Tenengrad), so a fixed absolute threshold doesn't
    transfer across cameras, resolutions, or scenes. The running-median
    relative floor must dominate the window-pick decision even when the
    absolute floor is set arbitrarily low, adapting to whatever "typical
    sharpness" looks like in this particular video/buffer rather than
    using one fixed number everywhere.
    """
    from drishti3d.triage.selector import _blur_floor

    high_sharpness_buffer = deque((i, i / 30, np.zeros((2, 2), dtype=np.uint8), 1000.0) for i in range(10))
    low_sharpness_buffer = deque((i, i / 30, np.zeros((2, 2), dtype=np.uint8), 10.0) for i in range(10))

    absolute_floor = 1.0  # low hard floor -- must not be the binding constraint here
    relative_threshold = 0.6

    high_floor = _blur_floor(high_sharpness_buffer, absolute_floor, relative_threshold)
    low_floor = _blur_floor(low_sharpness_buffer, absolute_floor, relative_threshold)

    assert high_floor == pytest.approx(600.0)
    assert low_floor == pytest.approx(6.0)

    # The same absolute blur score is judged differently depending on what's
    # "typical" for the video it's in -- proof the gate is relative.
    candidate_blur = 50.0
    assert candidate_blur >= absolute_floor
    assert candidate_blur < high_floor  # relatively blurry in a sharp video
    assert candidate_blur > low_floor  # relatively sharp in a blurry video


def test_triage_config_blur_defaults_favor_relative_over_absolute() -> None:
    config = TriageConfig()
    assert config.min_blur_score <= 5.0  # low hard floor only, not a meaningful absolute gate on its own
    assert config.relative_blur_threshold == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Principled rotation-vs-plane-translation discriminator (BUG 4)
#
# The heuristic fallback (raw displacement + residual/inlier signature,
# tested above) can't tell a pure camera rotation apart from a real
# translation over a planar/near-planar scene when the raw motion itself is
# small -- both fit a clean homography. With camera intrinsics, they
# mathematically can be told apart: ``K^-1 H K`` is (up to scale) a
# rotation matrix iff H is pure-rotation-induced (see
# ``triage.metrics._homography_rotation_deviation`` and the module-level
# comment above ``_ROTATION_DEVIATION_THRESHOLD``). These homographies are
# synthesized directly from a real ``K``/rotation/translation-over-a-plane
# model (``cv2.warpPerspective``), not the ad hoc 2D affine warps used by
# the fallback-heuristic tests above, so the fitted H genuinely has the
# claimed physical origin.
# ---------------------------------------------------------------------------

_ROT_WIDTH, _ROT_HEIGHT = 240, 160
_ROT_INTRINSICS = CameraIntrinsics(fx=200.0, fy=200.0, cx=_ROT_WIDTH / 2, cy=_ROT_HEIGHT / 2, width=_ROT_WIDTH, height=_ROT_HEIGHT)


def _rotation_y_matrix(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _pure_rotation_homography(deg: float) -> np.ndarray:
    """``H = K R K^-1`` for a real camera rotation about the camera centre (no translation)."""
    K = _ROT_INTRINSICS.K()
    H = K @ _rotation_y_matrix(deg) @ np.linalg.inv(K)
    return H / H[2, 2]


def _plane_translation_homography(tx: float, depth_m: float = 20.0) -> np.ndarray:
    """``H = K (I + t n^T / d) K^-1`` for a real sideways translation over a fronto plane at ``depth_m``.

    This is exactly the nadir-drone-over-flat-terrain case (module docstring):
    the camera's own optical axis is the plane normal, and it translates
    perpendicular to that axis -- the configuration a homography-residual
    -only signal is blind to (translation over a plane is *exactly* a
    homography, same as rotation), which is the whole reason this
    intrinsics-based test exists.
    """
    K = _ROT_INTRINSICS.K()
    n = np.array([0.0, 0.0, 1.0])
    t = np.array([tx, 0.0, 0.0])
    H = K @ (np.eye(3) + np.outer(t, n) / depth_m) @ np.linalg.inv(K)
    return H / H[2, 2]


def test_rotation_test_pure_rotation_about_camera_centre_yields_near_zero_useful_baseline() -> None:
    """(a) A genuine 3D camera rotation (K R K^-1, not just a 2D in-plane spin) must read as ~0 useful baseline."""
    prev = _random_textured_gray(_ROT_WIDTH, _ROT_HEIGHT, seed=11)
    curr = cv2.warpPerspective(prev, _pure_rotation_homography(1.5), (_ROT_WIDTH, _ROT_HEIGHT), borderMode=cv2.BORDER_REFLECT)

    result = estimate_parallax_detailed(prev, curr, intrinsics=_ROT_INTRINSICS)

    assert result.used_intrinsics_test is True
    assert result.raw_px > 2.0  # real, substantial pixel motion happened
    assert result.rotation_deviation is not None
    assert result.rotation_deviation < 0.005  # clean rotation signature
    assert result.useful_baseline_px == pytest.approx(0.0)


def test_rotation_test_translation_over_plane_yields_clearly_nonzero_useful_baseline() -> None:
    """(b) The case that currently fails without this fix: a real translation over a
    plane, with modest enough raw motion that the old heuristic's significance
    floor and "clean fit" rejection would both zero it out, must still be
    recognized as a genuine (if small) baseline.
    """
    prev = _random_textured_gray(_ROT_WIDTH, _ROT_HEIGHT, seed=11)
    curr = cv2.warpPerspective(prev, _plane_translation_homography(tx=0.3), (_ROT_WIDTH, _ROT_HEIGHT), borderMode=cv2.BORDER_REFLECT)

    result = estimate_parallax_detailed(prev, curr, intrinsics=_ROT_INTRINSICS)

    assert result.used_intrinsics_test is True
    assert result.rotation_deviation is not None
    assert result.rotation_deviation >= 0.005  # clearly not a rotation signature
    assert result.useful_baseline_px > 0.0
    assert result.useful_baseline_px == pytest.approx(result.raw_px)


def test_rotation_test_genuine_depth_parallax_stays_nonzero() -> None:
    """(c) A pair with genuine (non-planar) depth disparity must still read as useful
    with the intrinsics-based test active, same as it did under the old heuristic.
    """
    width, height = _ROT_WIDTH, _ROT_HEIGHT
    prev = _random_textured_gray(width, height, seed=11)

    grid = 8
    dx_near, dx_far = 8.0, -8.0
    tile_w, tile_h = width // grid, height // grid
    curr = prev.copy()
    for r in range(grid):
        for c in range(grid):
            dx = dx_near if (r + c) % 2 == 0 else dx_far
            y0, y1 = r * tile_h, height if r == grid - 1 else (r + 1) * tile_h
            x0, x1 = c * tile_w, width if c == grid - 1 else (c + 1) * tile_w
            m = np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32)
            curr[y0:y1, x0:x1] = cv2.warpAffine(
                prev[y0:y1, x0:x1], m, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REFLECT
            )

    result = estimate_parallax_detailed(prev, curr, intrinsics=_ROT_INTRINSICS)

    assert result.used_intrinsics_test is True
    assert result.useful_baseline_px > 0.0
    assert result.useful_baseline_px == pytest.approx(result.raw_px)


def test_estimate_parallax_detailed_without_intrinsics_falls_back_and_says_so() -> None:
    """No intrinsics given -> the struct must report that it used the fallback heuristic, not the rotation test."""
    prev = _random_textured_gray(_ROT_WIDTH, _ROT_HEIGHT, seed=11)
    curr = cv2.warpPerspective(prev, _pure_rotation_homography(1.5), (_ROT_WIDTH, _ROT_HEIGHT), borderMode=cv2.BORDER_REFLECT)

    result = estimate_parallax_detailed(prev, curr)

    assert result.used_intrinsics_test is False
    assert result.rotation_deviation is None


def test_select_keyframes_vision_only_selects_substantially_more_than_one_keyframe(parallax_video_path: Path) -> None:
    """(d) End-to-end: without telemetry, feeding real camera intrinsics through to
    the vision-only fallback must turn the old "1 keyframe from a whole video"
    starvation failure into a real, multi-keyframe selection -- using the
    default-shaped ``TriageConfig`` (not a hand-tuned one), matching the
    original bug report.
    """
    config = TriageConfig()
    intrinsics = CameraIntrinsics(fx=133.27, fy=133.27, cx=120.0, cy=80.0, width=240, height=160)

    with VideoSource(parallax_video_path) as video:
        keyframes, all_metrics = select_keyframes(video, config, telemetry=None, intrinsics=intrinsics)

    assert len(all_metrics) > 10
    assert len(keyframes) > 3  # substantially more than the pre-fix 1

    indices = [kf.frame_index for kf in keyframes]
    assert len(set(indices)) == len(indices)
    assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# No duplicate keyframes in either spacing mode (BUG 3)
# ---------------------------------------------------------------------------


def test_select_keyframes_never_returns_duplicate_frame_indices_gps_mode(tmp_path: Path) -> None:
    """Regression for the GPS-baseline path not honoring the same last-accepted-frame
    guard the vision path already had: a fast metric baseline that clears its
    trigger threshold within a single frame's motion causes consecutive
    checkpoints' sharpest-frame search windows to overlap, and the window
    search could re-select the same already-accepted frame as "sharpest" a
    second time.
    """
    n_frames, fps = 90, 15
    path = tmp_path / "gps_dup_repro.mp4"
    _make_parallax_video(path, n_frames=n_frames)

    # Telemetry advancing fast enough (~5 m/frame) that a single frame's
    # motion alone clears a 2 m baseline target -- the exact condition that
    # triggered the duplicate in the original bug report.
    meters_per_frame = 5.0
    telemetry = [
        TelemetrySample(
            timestamp=i / fps,
            geo=GeoPoint(lat=12.9716 + (meters_per_frame * i) / _METERS_PER_DEG_LAT, lon=77.5946, alt_msl=900.0, alt_rel=80.0),
        )
        for i in range(n_frames)
    ]

    config = TriageConfig(target_keyframes=30, min_blur_score=1.0, max_frames_scanned=200, use_gps_baseline=True, min_baseline_m=2.0)
    with VideoSource(path) as video:
        keyframes, _ = select_keyframes(video, config, telemetry=telemetry)

    indices = [kf.frame_index for kf in keyframes]
    assert len(indices) > 1
    assert len(set(indices)) == len(indices), f"duplicate frame_index values in GPS-baseline mode: {indices}"


def test_select_keyframes_never_returns_duplicate_frame_indices_vision_mode(parallax_video_path: Path) -> None:
    config = TriageConfig(target_keyframes=15, min_blur_score=1.0, min_parallax_px=3.0, max_frames_scanned=300, use_gps_baseline=False)
    with VideoSource(parallax_video_path) as video:
        keyframes, _ = select_keyframes(video, config)

    indices = [kf.frame_index for kf in keyframes]
    assert len(indices) > 1
    assert len(set(indices)) == len(indices), f"duplicate frame_index values in vision-only mode: {indices}"
