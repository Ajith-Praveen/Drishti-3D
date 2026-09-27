"""Operator-supplied camera calibration: config/CLI -> ingest intrinsics with provenance "user"."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drishti3d.config import Config, IngestConfig, load_config
from drishti3d.ingest.intrinsics import intrinsics_from_config
from drishti3d.pipeline import runner
from drishti3d.pipeline.stages import IngestStage, PipelineState
from tests.test_ingest import _make_synthetic_video

W, H = 1280, 720


def test_no_calibration_returns_none():
    assert intrinsics_from_config(IngestConfig(), W, H) is None


def test_focal_only_defaults_to_square_pixels_and_centre():
    intr, provenance = intrinsics_from_config(IngestConfig(camera_fx=877.0), W, H)
    assert provenance == "user"
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (877.0, 877.0, W / 2, H / 2)
    assert (intr.width, intr.height) == (W, H)
    assert intr.dist_coeffs is None


def test_hfov_sets_focal_from_width():
    intr, provenance = intrinsics_from_config(IngestConfig(camera_hfov_deg=72.3), W, H)
    assert provenance == "user"
    assert intr.fx == pytest.approx((W / 2) / np.tan(np.radians(72.3 / 2)))
    assert intr.fy == pytest.approx(intr.fx)


def test_fx_wins_over_hfov():
    intr, _ = intrinsics_from_config(IngestConfig(camera_fx=900.0, camera_hfov_deg=90.0), W, H)
    assert intr.fx == 900.0


def test_calibration_width_rescales_to_video_resolution():
    cfg = IngestConfig(camera_fx=2400.0, camera_fy=2410.0, camera_cx=1900.0, camera_cy=1090.0, camera_calibration_width=3840)
    intr, _ = intrinsics_from_config(cfg, W, H)
    factor = W / 3840
    assert intr.fx == pytest.approx(2400.0 * factor)
    assert intr.fy == pytest.approx(2410.0 * factor)
    assert intr.cx == pytest.approx(1900.0 * factor)
    assert intr.cy == pytest.approx(1090.0 * factor)


def test_distortion_coefficients_are_carried_in_opencv_order():
    intr, _ = intrinsics_from_config(IngestConfig(camera_fx=810.0, camera_dist_coeffs=[-0.2, 0.036, 0.0, 0.0, 0.0]), W, H)
    np.testing.assert_allclose(intr.dist_coeffs, [-0.2, 0.036, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "cfg",
    [
        IngestConfig(camera_fx=0.0),
        IngestConfig(camera_fx=-5.0),
        IngestConfig(camera_hfov_deg=0.5),
        IngestConfig(camera_hfov_deg=180.0),
        IngestConfig(camera_fx=800.0, camera_cx=5000.0),
        IngestConfig(camera_fx=800.0, camera_dist_coeffs=[-0.2, 0.03, 0.0]),
        IngestConfig(camera_fx=800.0, camera_dist_coeffs=[float("nan"), 0.0, 0.0, 0.0]),
        IngestConfig(camera_fx=800.0, camera_calibration_width=0),
        IngestConfig(camera_dist_coeffs=[-0.2, 0.036, 0.0, 0.0, 0.0]),
    ],
)
def test_impossible_calibrations_are_rejected(cfg):
    with pytest.raises(ValueError):
        intrinsics_from_config(cfg, W, H)


def test_yaml_ingest_section_reaches_the_config(tmp_path: Path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "ingest:\n"
        "  camera_fx: 876.7\n"
        "  camera_dist_coeffs: [-0.2, 0.036, 0.0, 0.0, 0.0]\n"
        "  camera_model: dji mavic 3\n"
    )
    cfg = load_config(path)
    assert cfg.ingest.camera_fx == 876.7
    assert cfg.ingest.camera_dist_coeffs == [-0.2, 0.036, 0.0, 0.0, 0.0]
    assert cfg.ingest.camera_model == "dji mavic 3"


def test_cli_flags_layer_over_the_config():
    args = runner.parse_args(["v.mp4", "--fx", "876.7", "--cy", "355", "--dist", "-0.2, 0.036,0,0,0", "--camera-model", "dji air 2s"])
    cfg = Config()
    runner._apply_camera_overrides(cfg, args)
    assert cfg.ingest.camera_fx == 876.7
    assert cfg.ingest.camera_cy == 355.0
    assert cfg.ingest.camera_fy is None
    assert cfg.ingest.camera_dist_coeffs == [-0.2, 0.036, 0.0, 0.0, 0.0]
    assert cfg.ingest.camera_model == "dji air 2s"


def test_cli_rejects_non_numeric_distortion():
    args = runner.parse_args(["v.mp4", "--dist", "k1,k2"])
    with pytest.raises(SystemExit):
        runner._apply_camera_overrides(Config(), args)


def _ingest(tmp_path: Path, cfg: Config) -> PipelineState:
    video = tmp_path / "v.mp4"
    _make_synthetic_video(video)
    state = PipelineState(video_path=video, telemetry_path=None, config=cfg, backbone_name="null")
    IngestStage().run(state, None, None)
    return state


def test_ingest_stage_uses_the_operator_calibration(tmp_path: Path):
    cfg = Config()
    cfg.ingest.camera_hfov_deg = 60.0
    cfg.ingest.camera_dist_coeffs = [-0.1, 0.01, 0.0, 0.0]
    state = _ingest(tmp_path, cfg)
    assert state.intrinsics_provenance == "user"
    assert state.intrinsics.fx == pytest.approx((state.video.width / 2) / np.tan(np.radians(30.0)))
    np.testing.assert_allclose(state.intrinsics.dist_coeffs, [-0.1, 0.01, 0.0, 0.0])


def test_ingest_stage_camera_model_selects_the_database_entry(tmp_path: Path):
    cfg = Config()
    cfg.ingest.camera_model = "dji mavic 3"
    state = _ingest(tmp_path, cfg)
    assert state.intrinsics_provenance == "camera_db"
    assert state.intrinsics.fx == pytest.approx(12.29 / 17.3 * state.video.width)


def test_ingest_stage_without_calibration_keeps_the_guess(tmp_path: Path):
    state = _ingest(tmp_path, Config())
    assert state.intrinsics_provenance == "default_guess"


def test_settings_panel_round_trips_the_calibration(qtbot):
    from drishti3d.app.panels.settings_panel import SettingsPanel

    panel = SettingsPanel()
    qtbot.addWidget(panel)
    assert panel.build_config().ingest.camera_fx is None

    panel.dist_edit.setText("-0.2, 0.036, 0, 0, 0")
    assert panel.build_config().ingest.camera_dist_coeffs is None  # no focal yet -> not passed on
    assert panel.dist_note.isVisibleTo(panel)

    panel.focal_spin.setValue(876.7)
    cfg = panel.build_config()
    assert cfg.ingest.camera_fx == pytest.approx(876.7)
    assert cfg.ingest.camera_dist_coeffs == [-0.2, 0.036, 0.0, 0.0, 0.0]
    assert not panel.dist_note.isVisibleTo(panel)

    panel.dist_edit.setText("-0.2, 0.036, 0")
    assert panel.build_config().ingest.camera_dist_coeffs is None
    assert panel.dist_note.isVisibleTo(panel)

    other = SettingsPanel()
    qtbot.addWidget(other)
    other.apply_config(cfg)
    assert other.camera_fx == pytest.approx(876.7)
    assert other.dist_coeffs == [-0.2, 0.036, 0.0, 0.0, 0.0]

