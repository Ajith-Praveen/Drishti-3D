from types import SimpleNamespace

import numpy as np
import pytest

from drishti3d.pipeline.result import PipelineResult, StageResult
from drishti3d.types import PointCloud


def verified_result():
    return PipelineResult(
        point_cloud=PointCloud(xyz=np.zeros((3, 3))),
        stage_results=[
            StageResult(name="pose_prior", status="ok", artifacts={"camera_validation": {"median_position_shift_m": 1.}}),
            StageResult(name="geometry", status="ok", artifacts={"placement": {"verdict": "PASS", "passed": True}}),
            StageResult(name="fusion", status="ok"),
        ],
    )


def test_valid_requires_completed_checks():
    result = verified_result()
    assert result.outcome == "valid"
    assert not result.quality["diagnostic_output"]
    result.stage_results[0].artifacts = {}
    assert result.outcome == "unverified"


@pytest.mark.parametrize("verdict,expected", [("PASS", "valid"), ("FAIL", "failed"), ("UNMEASURED", "unverified"), (None, "unverified")])
def test_placement_outcomes(verdict, expected):
    result = verified_result()
    result.stage_results[1].artifacts["placement"] = {"verdict": verdict, "passed": verdict == "PASS"} if verdict else {}
    assert result.outcome == expected


@pytest.mark.parametrize("name", ["geometry", "fusion", "export", "pose_prior"])
def test_failed_stage_cannot_be_promoted_by_successful_checks(name):
    result = verified_result()
    result.stage_results.append(StageResult(name=name, status="failed"))
    result.report["outcome"] = "valid"
    assert result.outcome == "failed"
    assert "diagnostic" in result.outcome_label


def test_cancelled_is_distinct_from_failed():
    result = verified_result()
    result.report["cancelled"] = True
    assert result.outcome == "cancelled"
    assert result.report_card()["diagnostic_output"]


def test_empty_legacy_and_synthetic_runs_are_unverified():
    assert PipelineResult().outcome == "unverified"
    result = verified_result()
    result.report["backbone"] = "null"
    assert result.outcome == "unverified"
    result = verified_result()
    result.point_cloud = None
    assert result.outcome == "unverified"


@pytest.mark.parametrize("outcome", ["valid", "unverified", "failed", "cancelled"])
def test_outcomes_survive_save_reload(tmp_path, outcome):
    result = verified_result()
    if outcome == "unverified":
        result.stage_results[0].artifacts = {}
    elif outcome != "valid":
        result.report["outcome"] = outcome
    result.save(tmp_path)
    loaded = PipelineResult.load(tmp_path)
    assert loaded.outcome == outcome
    assert loaded.report_card()["outcome"] == outcome
    assert loaded.quality["diagnostic_output"] == (outcome != "valid")


@pytest.mark.parametrize("heightmap", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_failed_placement_blocks_both_fusion_paths(monkeypatch, heightmap, override):
    from drishti3d.config import Config
    from drishti3d.fusion import tsdf
    from drishti3d.pipeline.stages import FusionStage

    class ReachedFusion(Exception):
        pass

    calls = []

    def fuse(*args, **kwargs):
        calls.append(True)
        raise ReachedFusion

    config = Config()
    config.fusion.allow_failed_placement = override
    state = SimpleNamespace(config=config, report={}, submaps=[object()], keyframes=[], geometry_heightmap=heightmap,
                            incremental_fusion=SimpleNamespace(mesh=lambda: None),
                            placement_report=SimpleNamespace(verdict="FAIL", summary=lambda: "bad scale",
                                                             as_dict=lambda: {"verdict": "FAIL"}))
    stage = FusionStage()
    monkeypatch.setattr(stage, "_from_heightmap", fuse)
    monkeypatch.setattr(stage, "_pre_mesh_point_filter", lambda state: None)
    monkeypatch.setattr(tsdf, "fuse_submaps", fuse)
    with pytest.raises(ReachedFusion if override else RuntimeError):
        stage.run(state, None, None)
    assert bool(calls) == override
    assert state.report["outcome"] == "failed"
    assert state.report["diagnostic_output"]


def test_reports_mark_diagnostic_output():
    from drishti3d.export.report import (
        build_report,
        render_report_html,
        render_report_text,
    )

    report = build_report({"outcome": "failed", "diagnostic_output": True, "quality_reasons": ["Placement failed."]})
    assert "DIAGNOSTIC OUTPUT" in render_report_text(report)
    assert "DIAGNOSTIC OUTPUT" in render_report_html(report)


@pytest.mark.parametrize("outcome", ["failed", "unverified", "cancelled", "valid"])
def test_completion_handler_never_says_finished(outcome):
    # Invoke the handler without constructing VTK or an OpenGL window.
    from drishti3d.app.main_window import MainWindow

    result = verified_result()
    if outcome == "unverified":
        result.stage_results[0].artifacts = {}
    elif outcome != "valid":
        result.report["outcome"] = outcome
    labels = []
    state = SimpleNamespace(_tick_timer=SimpleNamespace(stop=lambda: None),
                            diagnostics=SimpleNamespace(stop=lambda: None), _apply_result=lambda r: None,
                            progress_line=SimpleNamespace(set_fraction=lambda v: None), _finish_ui=labels.append,
                            _update_counts=lambda: None, viewport=None,
                            layers_panel=SimpleNamespace(sync_from_viewport=lambda v: None))
    MainWindow._on_pipeline_finished(state, result)
    assert labels == [result.outcome_label]
    assert "Finished" not in labels[0]


def test_worker_cancellation_returns_cancelled_result():
    from drishti3d.app.workers import PipelineCancelled, PipelineWorker

    def cancel(*args):
        raise PipelineCancelled

    worker = PipelineWorker(cancel)
    results, errors = [], []
    worker.finished.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    assert results[0].outcome == "cancelled"


def test_ending_unsuccessful_run_preserves_failed_rows():
    from drishti3d.app.panels.stage_rail import FAILED, RUNNING, SKIPPED, StageRail

    changes = []
    state = SimpleNamespace(
        _rows={"failed": SimpleNamespace(state=FAILED, set_state=lambda s: changes.append(("failed", s)), set_pulse=lambda p: None),
               "running": SimpleNamespace(state=RUNNING, set_state=lambda s: changes.append(("running", s)), set_pulse=lambda p: None)},
        _pulse_timer=SimpleNamespace(stop=lambda: None), summary=SimpleNamespace(setText=lambda text: None),
    )
    StageRail.finish(state, "Cancelled", successful=False)
    assert changes == [("running", SKIPPED)]


@pytest.mark.parametrize("outcome", ["failed", "unverified", "valid"])
def test_manual_export_marks_diagnostic_filename_and_sidecar(tmp_path, monkeypatch, outcome):
    import json

    from drishti3d.app import main_window
    from drishti3d.export import formats

    result = verified_result()
    if outcome == "unverified":
        result.stage_results[0].artifacts = {}
    elif outcome == "failed":
        result.report["outcome"] = "failed"
    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "chosen.ply"), ""))
    monkeypatch.setattr(formats, "export_ply", lambda path, geometry: path.write_text("fixture"))
    state = SimpleNamespace(_last_result=result, diagnostics=SimpleNamespace(append_log=lambda text: None),
                            statusBar=lambda: SimpleNamespace(showMessage=lambda *a: None))
    main_window.MainWindow._export_model(state)
    path = tmp_path / ("chosen.ply" if outcome == "valid" else "chosen.diagnostic.ply")
    assert path.exists()
    assert json.loads(path.with_suffix(".ply.quality.json").read_text())["outcome"] == outcome


def test_export_stage_routes_failed_mesh_to_diagnostic_directory(tmp_path, monkeypatch):
    import json

    import drishti3d.export
    from drishti3d.config import Config
    from drishti3d.pipeline import stages

    config = Config()
    config.export.output_dir = str(tmp_path)
    state = stages.PipelineState(video_path=tmp_path / "unused.mp4", telemetry_path=None,
                                 config=config, backbone_name="mapanything")
    state.point_cloud = PointCloud(xyz=np.zeros((3, 3)))
    state.report["outcome"] = "failed"
    state.stage_results = [StageResult(name="fusion", status="failed")]
    monkeypatch.setattr(stages, "_apply_georeferencing", lambda state: {})
    monkeypatch.setattr(stages, "_apply_reference_alignment", lambda state: None)

    def export(payload, directory, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        assert kwargs["report_artifacts"]["outcome"] == "failed"
        path = directory / "model.ply"
        path.write_text("fixture")
        return {"ply": path}

    monkeypatch.setattr(drishti3d.export, "export_all", export)
    artifacts, _ = stages.ExportStage().run(state, None, None)
    assert artifacts["output_dir"] == str(tmp_path / "diagnostic")
    assert not (tmp_path / "model.ply").exists()
    assert json.loads((tmp_path / "diagnostic" / "quality.json").read_text())["outcome"] == "failed"
