"""Tests for the DRISHTI-3D desktop app shell.

On Linux CI, run with QT_QPA_PLATFORM=offscreen so no real display is
required. On macOS use the default (cocoa) platform: VTK's
QVTKRenderWindowInteractor segfaults while creating its OpenGL context
under the offscreen platform (VTK 9.7 / PySide6 6.11), which kills the
whole pytest process instead of failing one test, so this module skips
itself in that combination.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

if sys.platform == "darwin" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    pytest.skip(
        "VTK's Qt render window segfaults under QT_QPA_PLATFORM=offscreen on macOS; "
        "run these tests with the default cocoa platform",
        allow_module_level=True,
    )

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
    def bad_pipeline(report_progress, report_stage, report_partial, cancel_token, report_preview=None):
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
    def good_pipeline(report_progress, report_stage, report_partial, cancel_token, report_preview=None):
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


def _grid_mesh(n: int):
    X, Y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    verts = np.c_[X.ravel(), Y.ravel(), 0.1 * X.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return verts, np.concatenate([np.c_[a, c, b], np.c_[b, c, d]])


def test_mesh_uncertainty_heatmap_and_motion_proxy(qtbot, monkeypatch):
    import drishti3d.app.viewport as vp

    monkeypatch.setattr(vp, "MESH_PROXY_FACES", 100)
    monkeypatch.setattr(vp, "MESH_PROXY_VERTICES", 60)
    viewport = Viewport()
    qtbot.addWidget(viewport)
    verts, faces = _grid_mesh(40)
    sigma = np.linspace(0.0, 3.0, len(verts)).astype(np.float32)
    sigma[:10] = np.nan  # inferred
    conf = np.full(len(verts), int(Confidence.MEASURED), dtype=np.uint8)
    viewport.set_mesh(verts, faces, None, confidence=conf, uncertainty_m=sigma)

    viewport.set_color_mode("uncertainty")
    data = viewport._mesh_polydata.GetPointData()
    assert data.GetScalars().GetName() == "uncertainty"
    assert np.isnan(data.GetArray("uncertainty").GetValue(0))
    assert "(m)" in viewport._scalar_bar.GetTitle()  # metres, not the tier legend
    viewport.set_color_mode("confidence")
    assert data.GetScalars().GetName() == "confidence"

    proxy = viewport._proxy_polydata
    assert proxy is not None and 0 < proxy.GetNumberOfPolys() < len(faces)
    assert proxy.GetPointData().GetArray("uncertainty") is not None
    viewport._proxy_begin()
    assert viewport._proxy_actor.GetVisibility() and not viewport._mesh_actor.GetVisibility()
    viewport._proxy_end()
    assert viewport._mesh_actor.GetVisibility() and not viewport._proxy_actor.GetVisibility()


def test_navigation_redraws_and_keeps_the_horizon_level(qtbot):
    """Camera moves must reach the screen (the RenderEvent used to go nowhere) and orbit about +Z."""
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTerrain

    viewport = Viewport()
    qtbot.addWidget(viewport)
    assert isinstance(viewport.interactor.GetInteractorStyle(), vtkInteractorStyleTerrain)
    assert viewport.interactor.HasObserver("RenderEvent")
    viewport.set_point_cloud(_make_point_cloud())
    for name in ("top", "iso", "front"):
        viewport.set_view(name)
        assert viewport.renderer.GetActiveCamera().GetViewUp()[2] > 0.99
    before = viewport.renderer.GetActiveCamera().GetDistance()
    viewport._wheel_zoom(1.1)
    assert viewport.renderer.GetActiveCamera().GetDistance() < before


def test_cluster_mesh_keeps_vertex_identity():
    from drishti3d.app.viewport import _cluster_mesh

    verts, faces = _grid_mesh(60)
    keep, proxy = _cluster_mesh(verts, faces, 200)
    assert len(keep) < len(verts) and proxy.max() < len(keep)
    assert len(np.unique(keep)) == len(keep) and keep.max() < len(verts)


def test_export_model_writes_every_offered_format(qtbot, tmp_path, monkeypatch):
    """File > Export Model must work for each format in its dialog (OBJ and GLB used to raise)."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from drishti3d.pipeline.result import PipelineResult

    window = MainWindow()
    qtbot.addWidget(window)
    verts, faces = _grid_mesh(10)
    pc = PointCloud(xyz=verts, rgb=np.full((len(verts), 3), 120, np.uint8),
                    confidence=np.full(len(verts), int(Confidence.MEASURED), np.uint8))
    window._last_result = PipelineResult(point_cloud=pc, mesh_faces=faces)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a[2] if len(a) > 2 else a))
    for ext in (".ply", ".las", ".obj", ".glb", ".xyz"):
        target = tmp_path / f"model{ext}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, t=target, **k: (str(t), ""))
        window._export_model()
        written = list(tmp_path.glob(f"model*{ext}"))
        assert written and written[0].stat().st_size > 0, (ext, errors)
    assert not errors, errors


def test_viewport_measures_volume_profile_and_point_on_the_surface(qtbot):
    """Volume / profile / point tools read the loaded surface; heights are reported as elevations."""
    from drishti3d.app.viewport import Viewport

    v = Viewport()
    qtbot.addWidget(v)
    x, y = np.meshgrid(np.arange(0.0, 40.0, 0.5), np.arange(0.0, 40.0, 0.5))
    z = np.where((np.abs(x - 20) < 5) & (np.abs(y - 20) < 5), 3.0, 0.0)  # a 3 m block on flat ground
    verts = np.c_[x.ravel(), y.ravel(), z.ravel() - 288.0]
    nx = x.shape[1]
    faces = []
    for r in range(x.shape[0] - 1):
        for c in range(nx - 1):
            a, b, cc, d = r * nx + c, r * nx + c + 1, (r + 1) * nx + c, (r + 1) * nx + c + 1
            faces += [[a, b, cc], [b, d, cc]]
    v.set_height_datum(445.6)
    v.set_mesh(verts, np.array(faces))
    done = []
    v.measurementCompleted.connect(done.append)

    v.start_measurement("volume")
    for corner in ((12, 12), (28, 12), (28, 28), (12, 28)):
        v.add_measurement_point([corner[0], corner[1], -288.0])
    v.finish_measurement()
    vol = done[-1]
    assert vol.kind == "volume" and vol.extra["above_m3"] == pytest.approx(3.0 * 9.5**2, rel=0.12)

    v.start_measurement("profile")
    v.add_measurement_point([2.0, 20.0, -288.0])
    v.add_measurement_point([38.0, 20.0, -288.0])
    v.finish_measurement()
    prof = done[-1]
    assert prof.extra["max_elevation_m"] == pytest.approx(445.6 - 288.0 + 3.0, abs=0.05)
    assert prof.extra["climb_m"] == pytest.approx(3.0, abs=0.3)

    v.start_measurement("point")
    v.add_measurement_point([5.0, 5.0, -288.0])  # finishes by itself
    assert done[-1].value == pytest.approx(157.6)
    assert v.measurement_count() == 3
    v.clear_measurements()
    assert v.measurement_count() == 0 and v.measuring() is None
