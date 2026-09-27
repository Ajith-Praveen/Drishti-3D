"""Tests for drishti3d.pipeline: the runner that wires every stage together.

Must pass on macOS with no torch, no CUDA, and no model weights -- every
run here uses ``backbone="null"`` (see ``geometry.backbone.NullBackbone``).
Deliberately does NOT import Qt/PySide6/VTK anywhere: VTK segfaults under
``QT_QPA_PLATFORM=offscreen`` on macOS, so pipeline-only tests must stay
Qt-free (``tests/test_app.py`` is the (separately-run) place for anything
that touches the GUI).

All fixtures are synthesized in-test (no external data files), per project
convention -- see ``tests/test_triage.py`` for the same
genuine-multi-depth-parallax synthetic video technique reused here.
"""

from __future__ import annotations

import sys
import types
from itertools import pairwise
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from drishti3d.config import Config
from drishti3d.pipeline.result import PipelineResult, StageResult
from drishti3d.pipeline.runner import _build_stages, run_pipeline
from drishti3d.pipeline.stages import CancelToken
from drishti3d.types import GeoPoint, TelemetrySample

# ---------------------------------------------------------------------------
# Synthetic video fixture: a small multi-depth "dolly" shot with genuine
# parallax (a flat/single-depth pan is fully explained by a homography and
# would never trigger a keyframe -- see triage.selector's module docstring
# and tests/test_triage.py's identically-shaped fixture).
# ---------------------------------------------------------------------------

_WIDTH, _HEIGHT = 240, 160
_GRID = 8
_DX_NEAR_PER_FRAME = 3.0
_DX_FAR_PER_FRAME = 0.4
_N_FRAMES = 90
_FPS = 15
_PAD = 60

_METERS_PER_DEG_LAT = 111320.0


@pytest.fixture(autouse=True)
def _skip_ba_for_tiled_video(monkeypatch):
    # This fixture pans independent tiles and assigns arbitrary GPS: it has no
    # single physical camera model. Exercise pipeline wiring here; physical BA
    # recovery/rejection has its own projected-scene tests.
    from drishti3d.pipeline.stages import MatchingStage, StageUnavailable

    def unavailable(*args, **kwargs):
        raise StageUnavailable("tiled pipeline fixture has no physical camera model")

    monkeypatch.setattr(MatchingStage, "run", unavailable)


def _base_texture(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(_HEIGHT + _PAD, _WIDTH + _PAD), dtype=np.uint8)
    return cv2.GaussianBlur(base, (3, 3), 0)


def _parallax_frame(base: np.ndarray, i: int) -> np.ndarray:
    tile_w, tile_h = _WIDTH // _GRID, _HEIGHT // _GRID
    out = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    pad = _PAD // 2
    for r in range(_GRID):
        for c in range(_GRID):
            near = (r + c) % 2 == 0
            dx = (_DX_NEAR_PER_FRAME if near else _DX_FAR_PER_FRAME) * i
            y0, y1 = r * tile_h, _HEIGHT if r == _GRID - 1 else (r + 1) * tile_h
            x0, x1 = c * tile_w, _WIDTH if c == _GRID - 1 else (c + 1) * tile_w
            src_x0, src_x1 = x0 + pad, x1 + pad
            src_y0, src_y1 = y0 + pad, y1 + pad
            left = max(0, src_x0 - pad)
            padded = base[src_y0:src_y1, left : src_x1 + pad]
            m = np.array([[1, 0, -dx], [0, 1, 0]], dtype=np.float32)
            warped = cv2.warpAffine(padded, m, (padded.shape[1], padded.shape[0]), borderMode=cv2.BORDER_REFLECT)
            off = src_x0 - left
            out[y0:y1, x0:x1] = warped[:, off : off + (x1 - x0)]
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _make_parallax_video(path: Path, n_frames: int = _N_FRAMES) -> None:
    base = _base_texture()
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=_FPS)
    stream.width = _WIDTH
    stream.height = _HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.codec_context.max_b_frames = 0

    for i in range(n_frames):
        arr = _parallax_frame(base, i)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _make_telemetry_samples(n_frames: int, fps: float, meters_per_frame: float = 0.5) -> list[TelemetrySample]:
    """Synthetic GPS telemetry advancing at a constant real metric rate."""
    return [
        TelemetrySample(
            timestamp=i / fps,
            geo=GeoPoint(lat=12.0 + (meters_per_frame * i) / _METERS_PER_DEG_LAT, lon=77.0, alt_msl=550.0, alt_rel=50.0),
        )
        for i in range(n_frames)
    ]


def _write_srt(path: Path, samples: list[TelemetrySample]) -> None:
    """Write a minimal DJI-style SRT sidecar matching ``ingest.telemetry.parse_srt_string``."""

    def _fmt(t: float) -> str:
        h, rem = divmod(t, 3600.0)
        m, s = divmod(rem, 60.0)
        ms = round((s - int(s)) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

    blocks = []
    for i, sample in enumerate(samples):
        t0, t1 = sample.timestamp, sample.timestamp + (1.0 / _FPS)
        assert sample.geo is not None
        blocks.append(
            f"{i + 1}\n{_fmt(t0)} --> {_fmt(t1)}\n"
            f"[latitude: {sample.geo.lat:.7f}] [longitude: {sample.geo.lon:.7f}] "
            f"[rel_alt: {sample.geo.alt_rel:.1f} abs_alt: {sample.geo.alt_msl:.1f}]"
        )
    path.write_text("\n\n".join(blocks) + "\n")


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "parallax.mp4"
    _make_parallax_video(path)
    return path


@pytest.fixture
def telemetry_path(tmp_path: Path) -> Path:
    path = tmp_path / "telemetry.srt"
    _write_srt(path, _make_telemetry_samples(_N_FRAMES, _FPS))
    return path


def _fast_config() -> Config:
    """A Config tuned so a full pipeline run over the tiny fixture video finishes quickly.

    ``baseline_to_altitude_ratio`` is pinned back to (approximately) its
    pre-Fix-1 value: the production default was raised from 0.02 to 0.25
    (see ``TriageConfig``'s docstring -- real photogrammetric B/H practice
    targets 0.2-0.3, not 0.02), but this fixture's synthetic telemetry
    moves only ``meters_per_frame=0.5`` at a fixed 50 m "altitude" (see
    ``_make_telemetry_samples``), so the *production* default would demand
    ``0.25 * 50 = 12.5`` m of accumulated baseline between keyframes --
    25 frames' worth of synthetic motion, which this tiny fixture video
    doesn't have enough of to produce more than one or two keyframes,
    starving bundle adjustment of tracks/cameras. This test exercises
    pipeline wiring, not real-world baseline tuning, so it keeps the old,
    densely-spaced keyframe behaviour explicitly, the same way it already
    pins ``min_baseline_m`` below.
    """
    cfg = Config()
    cfg.triage.target_keyframes = 6
    cfg.triage.min_blur_score = 50.0
    cfg.triage.min_parallax_px = 8.0
    cfg.triage.max_frames_scanned = 200
    cfg.triage.min_baseline_m = 1.0
    cfg.triage.baseline_to_altitude_ratio = 0.02
    cfg.triage.forward_overlap = 0  # fixture has no physical ground footprint
    cfg.geometry.window_size = 3
    # Fix 2 raised the production default from 518 to 924px, which would
    # *upscale* this fixture's tiny native 240x160 frames (see _WIDTH/
    # _HEIGHT above) by ~3.85x per axis before NullBackbone ray-casts one
    # point per pixel -- ~14.8x more points per view than the native
    # resolution needs, and hence a correspondingly slower fusion/TSDF
    # step across every test using this fixture. Pin it back to the
    # fixture's native size (no upscale, no wasted point density) for the
    # same "tiny fixture needs its own tuned knobs" reason every other
    # override in this function exists.
    cfg.geometry.max_image_size = _WIDTH
    return cfg


# ---------------------------------------------------------------------------
# End-to-end run with the null backbone
# ---------------------------------------------------------------------------


def test_unavailable_backbone_never_exports_synthetic_geometry(video_path, telemetry_path, monkeypatch):
    from types import SimpleNamespace

    requested = []

    def unavailable(name, **kwargs):
        requested.append(name)
        return SimpleNamespace(is_available=lambda: False)

    monkeypatch.setattr("drishti3d.pipeline.stages.get_backbone", unavailable)
    cfg = _fast_config()
    cfg.geometry.dense_method = "full3d"
    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=cfg, backbone="mapanything")
    assert requested == ["mapanything"]
    assert result.stage("geometry").status == "failed"
    assert result.stage("export").status == "skipped"
    assert result.point_cloud is None or not len(result.point_cloud.xyz)


def test_run_pipeline_completes_end_to_end_with_null_backbone(video_path: Path, telemetry_path: Path) -> None:
    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=_fast_config(), backbone="null")

    assert isinstance(result, PipelineResult)
    assert result.point_cloud is not None
    assert result.point_cloud.xyz.shape[0] > 0
    assert result.point_cloud.xyz.shape[1] == 3
    assert len(result.keyframes) > 0
    assert len(result.poses) > 0

    statuses = {sr.name: sr.status for sr in result.stage_results}
    # Asserted against the runner's OWN stage list rather than a copy of
    # it. The copy that used to live here named nine stages and went
    # stale the moment drone_path and coverage were added, so this test
    # failed for a reason unrelated to what it tests.
    assert set(statuses) == {stage.name for stage in _build_stages()}
    # every stage reaches a terminal status -- never "pending"/"running"
    for status in statuses.values():
        assert status in ("ok", "skipped", "failed")

    assert statuses["ingest"] == "ok"
    assert statuses["triage"] == "ok"
    assert statuses["geometry"] == "ok"
    # Failed placement in this nonphysical fixture must now mark fusion as
    # failed, while preserving diagnostic geometry and allowing its export.
    for name in ("bundle_adjustment", "export"):
        assert statuses[name] in ("ok", "skipped")
    assert statuses["fusion"] == "failed"
    assert result.outcome == "failed"

    assert result.report["cancelled"] is False
    assert result.report["backbone"] == "null"


def test_run_pipeline_dry_run_only_runs_ingest_and_triage(video_path: Path, telemetry_path: Path) -> None:
    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=_fast_config(), backbone="null", dry_run=True)

    statuses = {sr.name: sr.status for sr in result.stage_results}
    assert statuses["ingest"] == "ok"
    assert statuses["triage"] == "ok"
    # geometry/bundle/fusion/export never ran in a dry run.
    for name in ("geometry", "bundle_adjustment", "fusion", "export"):
        assert statuses[name] == "skipped"
    assert result.point_cloud is None
    assert len(result.keyframes) > 0


# ---------------------------------------------------------------------------
# save/load round trip -- the cached-demo safety net
# ---------------------------------------------------------------------------


def test_pipeline_result_save_load_round_trip(video_path: Path, telemetry_path: Path, tmp_path: Path) -> None:
    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=_fast_config(), backbone="null")
    assert result.point_cloud is not None and result.point_cloud.xyz.shape[0] > 0
    assert len(result.poses) > 0

    out_dir = tmp_path / "saved_result"
    result.save(out_dir)
    loaded = PipelineResult.load(out_dir)

    assert loaded.point_cloud is not None
    np.testing.assert_array_equal(loaded.point_cloud.xyz, result.point_cloud.xyz)
    if result.point_cloud.confidence is not None:
        assert loaded.point_cloud.confidence is not None
        np.testing.assert_array_equal(loaded.point_cloud.confidence, result.point_cloud.confidence)

    assert len(loaded.poses) == len(result.poses)
    for loaded_pose, original_pose in zip(loaded.poses, result.poses, strict=True):
        np.testing.assert_allclose(loaded_pose.R, original_pose.R)
        np.testing.assert_allclose(loaded_pose.t, original_pose.t)

    assert [sr.to_dict() for sr in loaded.stage_results] == [sr.to_dict() for sr in result.stage_results]
    assert loaded.report == result.report
    assert loaded.video_path == result.video_path
    assert loaded.telemetry_path == result.telemetry_path
    assert len(loaded.keyframes) == len(result.keyframes)
    for loaded_kf, original_kf in zip(loaded.keyframes, result.keyframes, strict=True):
        assert loaded_kf.frame_index == original_kf.frame_index
        assert loaded_kf.timestamp == pytest.approx(original_kf.timestamp)


def test_pipeline_result_save_load_round_trip_empty_result(tmp_path: Path) -> None:
    """A never-run (or fully-failed) result -- no point cloud, no poses -- still round-trips."""
    result = PipelineResult(stage_results=[StageResult(name="ingest", status="failed", elapsed_s=0.1, message="boom")])

    out_dir = tmp_path / "empty_result"
    result.save(out_dir)
    loaded = PipelineResult.load(out_dir)

    assert loaded.point_cloud is None
    assert loaded.poses == []
    assert [sr.to_dict() for sr in loaded.stage_results] == [sr.to_dict() for sr in result.stage_results]


# ---------------------------------------------------------------------------
# Accuracy report card: must actually be populated with real numbers, not
# discard values other stages already computed (see export.report's "never
# fabricate, but never discard a real measurement either" rule).
# ---------------------------------------------------------------------------


def test_full_run_with_telemetry_populates_real_report_values(
    video_path: Path, telemetry_path: Path, tmp_path: Path, monkeypatch
) -> None:
    """report.txt must carry real numbers for reprojection error, coverage, and stage timings.

    Regression test: these were all previously discarded even though the
    pipeline already measured them -- bundle adjustment reports its own
    converged reprojection RMSE, triage reports timeline coverage, and
    every ``StageResult`` carries ``elapsed_s`` -- none of it ever reached
    ``ExportStage``'s ``report_artifacts`` (see ``pipeline.stages
    .ExportStage.run`` and ``_georeference_for_report``).
    """
    from drishti3d.export.report import NOT_COMPUTED
    from drishti3d.pipeline.stages import PosePriorStage

    def measured_prior(self, state, *args, **kwargs):
        # Known measurement tests report propagation independently of the
        # intentionally nonphysical video fixture's feature matches.
        state.mean_reprojection_error_px = 1.25
        return {"rmse_after_px": 1.25}, "fixture measurement"

    monkeypatch.setattr(PosePriorStage, "run", measured_prior)

    config = _fast_config()
    export_dir = tmp_path / "export_out"
    config.export.output_dir = str(export_dir)

    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=config, backbone="null")

    statuses = {sr.name: sr.status for sr in result.stage_results}
    assert statuses["pose_prior"] == "ok"
    assert statuses["bundle_adjustment"] == "skipped"
    assert statuses["export"] == "ok"

    export_sr = next(sr for sr in result.stage_results if sr.name == "export")
    report_txt_path = Path(export_sr.artifacts["files"]["report_txt"])
    assert report_txt_path.is_file()
    report_text = report_txt_path.read_text()

    # Mean reprojection error: a real px figure, not "not computed".
    assert "Mean reprojection error:" in report_text
    reproj_line = next(line for line in report_text.splitlines() if line.startswith("Mean reprojection error:"))
    assert NOT_COMPUTED not in reproj_line
    assert "px" in reproj_line
    assert "1.25" in reproj_line

    # Coverage: a real percentage from triage.
    coverage_line = next(line for line in report_text.splitlines() if line.startswith("Coverage:"))
    assert NOT_COMPUTED not in coverage_line
    assert "%" in coverage_line

    # Stage timings: at least the stages that ran before export.
    assert "Stage timings:" in report_text
    timings_block = report_text.split("Stage timings:\n", 1)[1].split("\nReference alignment", 1)[0]
    assert NOT_COMPUTED not in timings_block
    for stage_name in ("ingest", "triage", "geometry", "bundle_adjustment"):
        assert f"{stage_name}:" in timings_block


# ---------------------------------------------------------------------------
# CLI: `drishti3d-run --out` must anchor every deliverable under it
# ---------------------------------------------------------------------------


def test_cli_export_writes_under_out_directory_not_cwd(
    video_path: Path, telemetry_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``drishti3d-run --out <dir>`` must put every deliverable under ``<dir>``, never under the cwd.

    Regression test: ``ExportStage`` resolves ``config.export.output_dir``
    (a plain relative ``"output"`` string by default) against the
    *process's* cwd, not against the run's requested ``--out`` root, so
    without ``main`` anchoring it first, every exported deliverable
    (PLY/LAS/GLB/rasters/report) silently landed in ``./output`` under
    wherever the CLI happened to be invoked from -- only the cached
    ``PipelineResult`` (``result.save``) actually respected ``--out``.
    """
    from drishti3d.config import save_config
    from drishti3d.pipeline import runner

    config_path = tmp_path / "fast_config.yaml"
    save_config(_fast_config(), config_path)

    workdir = tmp_path / "cwd"
    workdir.mkdir()
    out_dir = tmp_path / "result_out"

    monkeypatch.chdir(workdir)
    argv = [
        str(video_path),
        "--telemetry",
        str(telemetry_path),
        "--backbone",
        "null",
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
        "--log-level",
        "ERROR",
    ]
    rc = runner.main(argv)
    assert rc == 1  # failed placement is diagnostic output, not a successful reconstruction

    exported_dir = out_dir / "output" / "diagnostic"
    assert exported_dir.is_dir(), f"expected deliverables under {exported_dir}, found nothing there"
    assert (exported_dir / "report.txt").exists()
    assert (exported_dir / "report.html").exists()
    assert any(exported_dir.glob("model.*")), "expected at least one model.* deliverable under --out"

    # The cached PipelineResult (arrays.npz/meta.json) also lands under --out.
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "arrays.npz").exists()

    # And, critically, nothing was written into the cwd's own "output" dir.
    assert not (workdir / "output").exists()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_run_pipeline_cancellation_stops_early_without_raising(video_path: Path, telemetry_path: Path) -> None:
    cancel_token = CancelToken()
    call_count = {"n": 0}

    def progress_cb(stage: str, current: int, total: int, message: str) -> None:
        call_count["n"] += 1
        if call_count["n"] > 3:
            cancel_token.cancel()

    # Must not raise -- a cancelled run returns a normal, partial result.
    result = run_pipeline(
        video_path,
        telemetry_path=telemetry_path,
        config=_fast_config(),
        progress_cb=progress_cb,
        cancel_token=cancel_token,
        backbone="null",
    )

    assert isinstance(result, PipelineResult)
    assert result.report["cancelled"] is True

    statuses = {sr.name: sr.status for sr in result.stage_results}
    # Asserted against the runner's OWN stage list rather than a copy of
    # it. The copy that used to live here named nine stages and went
    # stale the moment drone_path and coverage were added, so this test
    # failed for a reason unrelated to what it tests.
    assert set(statuses) == {stage.name for stage in _build_stages()}
    for status in statuses.values():
        assert status in ("ok", "skipped", "failed")
    # cancellation fired early enough that at least one stage never completed.
    assert any(status == "skipped" for status in statuses.values())


# ---------------------------------------------------------------------------
# Missing telemetry
# ---------------------------------------------------------------------------


def test_run_pipeline_without_telemetry_completes_with_vision_only_spacing(video_path: Path) -> None:
    result = run_pipeline(video_path, telemetry_path=None, config=_fast_config(), backbone="null")

    statuses = {sr.name: sr.status for sr in result.stage_results}
    assert statuses["ingest"] == "ok"
    assert statuses["triage"] == "ok"
    assert result.point_cloud is not None
    assert len(result.keyframes) > 0
    assert result.report["triage_spacing_mode"] == "vision_parallax"

    indices = [kf.frame_index for kf in result.keyframes]
    assert indices == sorted(set(indices))
    assert all(b > a for a, b in pairwise(indices))


def test_run_pipeline_with_telemetry_uses_gps_baseline_spacing(video_path: Path, telemetry_path: Path) -> None:
    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=_fast_config(), backbone="null")

    assert result.report["triage_spacing_mode"] == "gps_baseline"
    assert any(kf.telemetry is not None and kf.telemetry.geo is not None for kf in result.keyframes)


# ---------------------------------------------------------------------------
# An optional late stage raising is recorded "failed", not fatal
# ---------------------------------------------------------------------------


def test_optional_stage_exception_is_recorded_failed_and_does_not_abort_run(
    video_path: Path, telemetry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tsdf = types.ModuleType("drishti3d.fusion.tsdf")

    def _boom(submaps, config, stats=None, **kwargs):
        raise RuntimeError("synthetic fusion failure for test")

    fake_tsdf.fuse_submaps = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "drishti3d.fusion.tsdf", fake_tsdf)

    config = _fast_config()
    # This test is about the failure path, not the placement gate in front of it.
    config.fusion.allow_failed_placement = True
    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=config, backbone="null")

    statuses = {sr.name: sr.status for sr in result.stage_results}
    assert statuses["fusion"] == "failed"
    fusion_result = next(sr for sr in result.stage_results if sr.name == "fusion")
    assert "synthetic fusion failure for test" in fusion_result.message

    # A failure in an optional late stage must not abort the run: geometry's
    # output survives, and export still gets a chance to run afterward.
    assert statuses["geometry"] == "ok"
    assert statuses["export"] in ("ok", "skipped", "failed")
    assert result.point_cloud is not None
    assert result.point_cloud.xyz.shape[0] > 0


# ---------------------------------------------------------------------------
# Backbone pose conditioning: TriageStage must populate Keyframe.pose from
# GPS + gimbal telemetry -- previously nothing in the pipeline ever set it,
# so geometry.mapanything.MapAnythingBackbone.predict always ran with zero
# pose conditioning (see pipeline.stages._poses_from_telemetry's docstring
# for the full story; this was the root cause of the ~14x metric-scale bug
# on real footage).
# ---------------------------------------------------------------------------


def test_poses_from_telemetry_builds_pose_from_gps_and_gimbal() -> None:
    from drishti3d.pipeline.stages import _poses_from_telemetry
    from drishti3d.types import FrameMetrics, GeoPoint, Keyframe, TelemetrySample

    metrics = FrameMetrics(
        index=0, timestamp=0.0, blur_score=1.0, exposure_score=1.0, mean_luma=1.0, estimated_parallax=0.0
    )
    telemetry = [
        TelemetrySample(
            timestamp=float(i),
            geo=GeoPoint(lat=12.0 + i * 1e-5, lon=77.0, alt_msl=550.0, alt_rel=50.0),
            gimbal_pitch=-90.0,
            gimbal_roll=0.0,
            gimbal_yaw=45.0,
        )
        for i in range(3)
    ]
    keyframes = [
        Keyframe(frame_index=i, timestamp=float(i), metrics=metrics, telemetry=telemetry[i]) for i in range(3)
    ]

    poses = _poses_from_telemetry(keyframes)

    assert all(p is not None for p in poses)
    positions = np.array([p.t for p in poses])
    # Consecutive keyframes are ~1.1 m apart in latitude (1e-5 deg); poses
    # must actually reflect that, not collapse to a shared/zero position.
    assert np.linalg.norm(positions[1] - positions[0]) > 0.1
    for pose in poses:
        # A real rotation matrix (orthonormal, det +1), not a placeholder.
        np.testing.assert_allclose(pose.R.T @ pose.R, np.eye(3), atol=1e-6)
        assert np.linalg.det(pose.R) == pytest.approx(1.0, abs=1e-6)


def test_poses_from_telemetry_skips_keyframes_missing_gimbal_or_geo() -> None:
    """No pose is fabricated when geo or gimbal attitude is missing -- an assumed
    orientation would actively mislead the backbone (see the function's docstring)."""
    from drishti3d.pipeline.stages import _poses_from_telemetry
    from drishti3d.types import FrameMetrics, GeoPoint, Keyframe, TelemetrySample

    metrics = FrameMetrics(
        index=0, timestamp=0.0, blur_score=1.0, exposure_score=1.0, mean_luma=1.0, estimated_parallax=0.0
    )
    kf_no_telemetry = Keyframe(frame_index=0, timestamp=0.0, metrics=metrics, telemetry=None)
    kf_no_gimbal = Keyframe(
        frame_index=1,
        timestamp=1.0,
        metrics=metrics,
        telemetry=TelemetrySample(timestamp=1.0, geo=GeoPoint(lat=12.0, lon=77.0, alt_msl=550.0)),
    )

    poses = _poses_from_telemetry([kf_no_telemetry, kf_no_gimbal])
    assert poses == [None, None]


def test_triage_stage_populates_keyframe_pose_from_telemetry(
    video_path: Path, telemetry_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the wiring bug itself: TriageStage must actually call
    ``_poses_from_telemetry`` and assign its output onto every ``Keyframe.pose``."""
    from drishti3d.pipeline import stages as stages_mod
    from drishti3d.types import Pose

    sentinel_pose = Pose(R=np.eye(3), t=np.array([1.0, 2.0, 3.0]))
    monkeypatch.setattr(stages_mod, "_poses_from_telemetry", lambda keyframes, **_kw: [sentinel_pose] * len(keyframes))

    result = run_pipeline(video_path, telemetry_path=telemetry_path, config=_fast_config(), backbone="null")

    assert len(result.keyframes) > 0
    assert all(kf.pose is sentinel_pose for kf in result.keyframes)


# ---------------------------------------------------------------------------
# GeometryConfig.max_image_size must bound every backbone's input, not just
# MapAnight's own internal resize -- see GeometryStage's _resize_for_backbone.
# ---------------------------------------------------------------------------


def test_geometry_stage_resizes_images_and_scales_intrinsics_for_every_backbone(tmp_path: Path) -> None:
    from drishti3d.geometry.backbone import Backbone, BackboneResult, register_backbone
    from drishti3d.pipeline.stages import GeometryStage, PipelineState
    from drishti3d.types import CameraIntrinsics, Frame, FrameMetrics, Keyframe, Pose

    captured: dict = {}

    class _SpyBackbone(Backbone):
        name = "spy-resize"

        def is_available(self) -> bool:
            return True

        def load(self, device: str = "cpu", dtype=None) -> None:
            pass

        def unload(self) -> None:
            pass

        def predict(self, images, intrinsics=None, poses=None) -> BackboneResult:
            captured["images"] = images
            captured["intrinsics"] = intrinsics
            n = len(images)
            h, w = images[0].shape[:2]
            return BackboneResult(
                poses=[Pose(R=np.eye(3), t=np.array([float(i), 0.0, 0.0])) for i in range(n)],
                points=np.zeros((n, h, w, 3)),
                depth=np.zeros((n, h, w)),
                confidence=np.zeros((n, h, w), dtype=np.float32),
                intrinsics=list(intrinsics) if intrinsics is not None else [],
                is_metric=True,
            )

    register_backbone("spy-resize", _SpyBackbone)

    class _FakeVideo:
        def __init__(self, images: list[np.ndarray]) -> None:
            self._images = images

        def read_frames(self, indices: list[int]) -> list[Frame]:
            return [Frame(index=i, timestamp=float(i), image=self._images[i]) for i in indices]

    orig_w, orig_h = 1200, 800
    orig_intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=600.0, cy=400.0, width=orig_w, height=orig_h)
    image = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    metrics = FrameMetrics(
        index=0, timestamp=0.0, blur_score=1.0, exposure_score=1.0, mean_luma=1.0, estimated_parallax=0.0
    )
    kf = Keyframe(frame_index=0, timestamp=0.0, metrics=metrics, intrinsics=orig_intr)

    cfg = Config()
    cfg.geometry.window_size = 1
    cfg.geometry.dense_method = "full3d"
    cfg.geometry.max_image_size = 400

    state = PipelineState(
        video_path=tmp_path / "x.mp4", telemetry_path=None, config=cfg, backbone_name="spy-resize"
    )
    state.keyframes = [kf]
    state.intrinsics = orig_intr
    state.video = _FakeVideo([image])

    stage = GeometryStage()
    stage.run(state, cancel_token=None, progress_cb=None)

    max_side = cfg.geometry.max_image_size
    scale = max_side / max(orig_h, orig_w)

    resized_img = captured["images"][0]
    assert max(resized_img.shape[:2]) == max_side

    scaled_intr = captured["intrinsics"][0]
    assert scaled_intr.fx == pytest.approx(orig_intr.fx * scale)
    assert scaled_intr.fy == pytest.approx(orig_intr.fy * scale)
    assert scaled_intr.cx == pytest.approx(orig_intr.cx * scale)
    assert scaled_intr.cy == pytest.approx(orig_intr.cy * scale)


def test_matching_config_reaches_the_stage(tmp_path):
    """quality_profile settings for matching/BA must not be inert.

    MatchingConfig's fields were written to mirror pipeline.stages' module
    constants so the stage could be switched over mechanically -- but until
    this was wired, the "fast" profile asked for half-resolution detection,
    2,000 BA points and 50 iterations while the stage silently ran at full
    resolution, unbounded points and 100 iterations.
    """
    from types import SimpleNamespace

    from drishti3d.config import load_config
    from drishti3d.pipeline.stages import _match_cfg

    def cfg_for(profile: str):
        path = tmp_path / f"{profile}.yaml"
        path.write_text(f"quality_profile: {profile}\n")
        return SimpleNamespace(config=load_config(path))

    fast = cfg_for("fast")
    assert _match_cfg(fast, "detect_scale", 1.0) == 0.5
    assert _match_cfg(fast, "max_points_in_ba", None) == 6000
    assert _match_cfg(fast, "ba_max_iterations", 100) == 50
    assert _match_cfg(fast, "max_features", 4000) == 2000

    # "accurate" deliberately sets max_points_in_ba to None -- keep every
    # track. That None must survive as None, not fall back to a cap.
    assert _match_cfg(cfg_for("accurate"), "max_points_in_ba", 999) is None
    assert _match_cfg(cfg_for("accurate"), "ba_max_iterations", 100) == 150

    # A config object without the field at all falls back to the constant.
    assert _match_cfg(SimpleNamespace(config=SimpleNamespace()), "detect_scale", 1.0) == 1.0


def test_focal_refinement_is_applied_as_a_ratio_at_native_resolution():
    """BA sees downscaled intrinsics; the correction must not inherit that scale.

    MatchingStage scales intrinsics to whatever resolution it ran at --
    1920 px with the keyframe cache, half of the 3840 px native frame. So
    BAResult.intrinsics come back at 1920 scale. Writing them straight
    onto native-resolution keyframes halves fx. Only the RATIO is
    scale-invariant, and only the ratio is the real finding.
    """
    from drishti3d.geometry.mapanything import scale_intrinsics
    from drishti3d.pipeline.stages import _rescale_focal
    from drishti3d.types import CameraIntrinsics

    native = CameraIntrinsics(fx=2132.4, fy=2132.4, cx=1920.0, cy=1080.0, width=3840, height=2160)
    at_match_res = scale_intrinsics(native, 0.5)
    assert at_match_res.fx == pytest.approx(1066.2)

    # BA refines by 1.57x at ITS resolution. It changes only fx/fy and
    # carries width/height through unchanged -- see bundle._unpack_x.
    refined_at_match_res = _rescale_focal(at_match_res, 1.57)
    factor = refined_at_match_res.fx / at_match_res.fx
    assert factor == pytest.approx(1.57)

    # Applying the ratio to native gives the right native focal length,
    # and must NOT touch the frame size or principal point -- the sensor
    # did not change, only our estimate of the lens.
    corrected = _rescale_focal(native, factor)
    assert corrected.fx == pytest.approx(2132.4 * 1.57, rel=1e-6)
    assert corrected.width == 3840 and corrected.height == 2160
    assert corrected.cx == 1920.0 and corrected.cy == 1080.0

    # The resize helper would have been wrong here: it rescales the frame.
    assert scale_intrinsics(native, factor).width != 3840

    # ...whereas writing BA's own intrinsics back onto a native keyframe
    # would have halved fx and shrunk the recorded frame to 1920 px.
    assert refined_at_match_res.fx == pytest.approx(2132.4 * 1.57 / 2, rel=1e-6)
    assert refined_at_match_res.width == 1920


def test_focal_refinement_bounds_reject_an_unphysical_solve():
    from drishti3d.pipeline.stages import _MAX_FOCAL_REFINE_FACTOR, _MIN_FOCAL_REFINE_FACTOR

    # COLMAP's measured 1.57x correction on this footage must be accepted.
    assert _MIN_FOCAL_REFINE_FACTOR < 1.57 < _MAX_FOCAL_REFINE_FACTOR
    # A focal/depth trade-off runaway must not be.
    assert not (_MIN_FOCAL_REFINE_FACTOR <= 12.0 <= _MAX_FOCAL_REFINE_FACTOR)
    assert not (_MIN_FOCAL_REFINE_FACTOR <= 0.05 <= _MAX_FOCAL_REFINE_FACTOR)
