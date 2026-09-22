"""Headless tests for the DRISHTI-3D desktop app shell.

Run with QT_QPA_PLATFORM=offscreen so no real display is required.
"""

from __future__ import annotations

import numpy as np

from drishti3d.app.main_window import MainWindow
from drishti3d.app.viewport import Viewport
from drishti3d.app.workers import DemoPipeline, PipelineWorker
from drishti3d.types import Confidence, PointCloud


def _make_point_cloud(n: int = 200) -> PointCloud:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-10, 10, size=(n, 3))
    rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    confidence = rng.integers(0, 3, size=n).astype(np.uint8)
    # Force at least two distinct confidence values.
    confidence[0] = int(Confidence.MEASURED)
    confidence[1] = int(Confidence.INFERRED)
    return PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)


def test_main_window_constructs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "DRISHTI-3D"


def test_viewport_renders_point_cloud_and_actor_count_increases(qtbot):
    viewport = Viewport()
    qtbot.addWidget(viewport)

    before = viewport.actor_count()
    pc = _make_point_cloud()
    viewport.set_point_cloud(pc)
    after = viewport.actor_count()

    assert after > before
    assert viewport.point_count() == pc.xyz.shape[0]


def test_color_mode_confidence_changes_active_scalar_array(qtbot):
    viewport = Viewport()
    qtbot.addWidget(viewport)

    pc = _make_point_cloud()
    viewport.set_point_cloud(pc)

    viewport.set_color_mode("rgb")
    assert viewport.active_scalar_name() == "rgb"

    viewport.set_color_mode("confidence")
    assert viewport.active_scalar_name() == "confidence"


def test_demo_pipeline_emits_monotonic_progress_and_varied_confidence():
    pipeline = DemoPipeline(steps_per_stage=2, tick_seconds=0.0)

    progress_values: list[int] = []

    def report_progress(current, total, message):
        progress_values.append(current)

    def report_stage(stage):
        pass

    def report_partial(result):
        pass

    from drishti3d.app.workers import CancelToken

    result = pipeline(report_progress, report_stage, report_partial, CancelToken())

    assert progress_values == sorted(progress_values)
    assert len(progress_values) > 1

    point_cloud = result["point_cloud"]
    assert isinstance(point_cloud, PointCloud)
    assert point_cloud.confidence is not None
    assert len(np.unique(point_cloud.confidence)) > 1


def test_pipeline_worker_emits_failed_not_raise(qtbot):
    def bad_pipeline(report_progress, report_stage, report_partial, cancel_token):
        raise RuntimeError("boom")

    worker = PipelineWorker(bad_pipeline)

    failures: list[str] = []
    worker.failed.connect(failures.append)

    # Must not raise even though the callable throws.
    worker.run()

    assert len(failures) == 1
    assert "boom" in failures[0]
    assert "RuntimeError" in failures[0]


def test_pipeline_worker_emits_finished_on_success(qtbot):
    def good_pipeline(report_progress, report_stage, report_partial, cancel_token):
        report_progress(1, 1, "done")
        return {"ok": True}

    worker = PipelineWorker(good_pipeline)

    results: list[object] = []
    worker.finished.connect(results.append)

    worker.run()

    assert results == [{"ok": True}]


# ---------------------------------------------------------------------------
# mesh detail control
# ---------------------------------------------------------------------------


def test_mesh_detail_sets_both_budgets_together(qtbot):
    """Either budget alone is silently undone by the other.

    Measured: raising voxel_count_budget alone moved the allowed voxel
    from 2.32 m to 0.46 m, and max_mesh_faces then coarsened it straight
    back to 1.55 m -- the run looked identical. So the panel must set
    both from one control.
    """
    from drishti3d.app.panels.settings_panel import SettingsPanel

    panel = SettingsPanel()
    qtbot.addWidget(panel)

    seen = []
    for i in range(panel.mesh_combo.count()):
        panel.mesh_combo.setCurrentIndex(i)
        cfg = panel.build_config()
        seen.append((cfg.fusion.max_mesh_faces, cfg.fusion.voxel_count_budget))
        # The voxel budget must always exceed the face budget: a 0.5 m
        # voxel needs ~15.9M voxels to carry ~5.3M faces.
        assert cfg.fusion.voxel_count_budget > cfg.fusion.max_mesh_faces

    # Every preset is distinct and monotonically finer.
    assert len(set(seen)) == len(seen)
    assert seen == sorted(seen)


def test_mesh_detail_reaches_the_pipeline_not_just_the_widget(qtbot):
    """The class of bug that left MatchingConfig dead for the whole project."""
    from drishti3d.app.panels.settings_panel import SettingsPanel

    panel = SettingsPanel()
    qtbot.addWidget(panel)

    panel.mesh_combo.setCurrentIndex(0)
    draft = panel.build_config().fusion
    panel.mesh_combo.setCurrentIndex(panel.mesh_combo.count() - 1)
    maximum = panel.build_config().fusion

    assert maximum.max_mesh_faces > draft.max_mesh_faces
    assert maximum.voxel_count_budget > draft.voxel_count_budget


def test_mesh_detail_note_reports_the_chosen_budgets(qtbot):
    from drishti3d.app.panels.settings_panel import SettingsPanel

    panel = SettingsPanel()
    qtbot.addWidget(panel)
    panel.mesh_combo.setCurrentIndex(1)
    note = panel.mesh_note.text()
    assert "8M faces" in note
    assert "16M voxels" in note
