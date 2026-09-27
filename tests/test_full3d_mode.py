"""Full-3D selection must never silently turn into a terrain height field."""
from pathlib import Path

import numpy as np
import pytest

from drishti3d.config import Config, load_config, save_config
from drishti3d.pipeline import stages
from drishti3d.types import CameraIntrinsics, FrameMetrics, Keyframe, Pose


@pytest.mark.parametrize("method", ["full3d", "mapanything", "auto", "mvs3d"])
def test_full3d_bypasses_both_heightfield_routes_and_ground_view_culling(tmp_path, monkeypatch, method):
    cfg = Config()
    cfg.geometry.dense_method = method
    # Even a legacy config explicitly asking for nadir collapse cannot
    # override the user's full-3D selection.
    cfg.fusion.heightmap_for_nadir = True
    state = stages.PipelineState(tmp_path / "video.mp4", None, cfg, "mapanything")
    intr = CameraIntrinsics.from_hfov(60, 64, 48)
    poses = [Pose(R=np.diag([1., -1., -1.]), t=np.array([float(i), 0., 10.])) for i in range(4)]
    state.keyframes = [Keyframe(i, float(i), FrameMetrics(i, float(i), 100., 1., 120., 0.),
                               intrinsics=intr, pose=pose) for i, pose in enumerate(poses)]
    state.poses = poses
    state.ba_points = np.array([[0., 0., 0.]])
    state.pose_prior_refined = True
    state.unrefined_keyframes = [3]
    state.yaw_refinement = {}
    monkeypatch.setattr(stages, "_heightfield_ready", lambda *a: True)

    def forbidden(*a, **kw):
        pytest.fail("Full 3D invoked a terrain-only route")

    monkeypatch.setattr(stages.GeometryStage, "_run_heightfield", forbidden)
    monkeypatch.setattr(stages, "_select_dense_views", forbidden)
    class Backbone:
        def is_available(self): return True
        def load(self, **kw): pass
        def unload(self): pass
    monkeypatch.setattr(stages, "get_backbone", lambda *a, **kw: Backbone())
    monkeypatch.setattr(stages, "available_devices", lambda **kw: ["cpu"])
    class ReachedInference(Exception): pass
    def measured(*a, **kw):
        assert method in ("auto", "mvs3d")
        raise ReachedInference
    monkeypatch.setattr(stages.GeometryStage, "_run_mvs3d", measured)
    def inspect(*a, **kw):
        from drishti3d.fusion.incremental import IncrementalVoxelFusion
        assert not state.geometry_heightmap
        assert state.dense_views == {0, 1, 2}
        assert isinstance(state.incremental_fusion, IncrementalVoxelFusion)
        raise ReachedInference
    monkeypatch.setattr(stages, "_run_windows", inspect)
    monkeypatch.setattr(stages, "_run_windows_sequential", inspect)
    with pytest.raises(ReachedInference):
        stages.GeometryStage().run(state, None, None)


@pytest.mark.parametrize("method", ["auto", "mvs3d"])
def test_unsolved_cameras_fall_back_to_learned_depth_only_in_auto(tmp_path, monkeypatch, method):
    """No solved cameras: Automatic still produces a (learned, volumetric) model; explicit Measured 3D stops."""
    cfg = Config()
    cfg.geometry.dense_method = method
    state = stages.PipelineState(tmp_path / "video.mp4", None, cfg, "mapanything")
    intr = CameraIntrinsics.from_hfov(60, 64, 48)
    poses = [Pose(R=np.diag([1., -1., -1.]), t=np.array([float(i), 0., 10.])) for i in range(4)]
    state.keyframes = [Keyframe(i, float(i), FrameMetrics(i, float(i), 100., 1., 120., 0.),
                               intrinsics=intr, pose=pose) for i, pose in enumerate(poses)]
    state.poses = poses
    state.pose_prior_refined = False  # the camera solve did not converge
    state.yaw_refinement = {}

    def forbidden(*a, **kw):
        pytest.fail("measured stereo ran without solved cameras")

    monkeypatch.setattr(stages.GeometryStage, "_run_mvs3d", forbidden)

    class Backbone:
        def is_available(self): return True
        def load(self, **kw): pass
        def unload(self): pass
    monkeypatch.setattr(stages, "get_backbone", lambda *a, **kw: Backbone())
    monkeypatch.setattr(stages, "available_devices", lambda **kw: ["cpu"])

    class ReachedInference(Exception): pass

    def inspect(*a, **kw):
        assert not state.geometry_heightmap
        raise ReachedInference
    monkeypatch.setattr(stages, "_run_windows", inspect)
    monkeypatch.setattr(stages, "_run_windows_sequential", inspect)
    expected = ReachedInference if method == "auto" else RuntimeError
    with pytest.raises(expected):
        stages.GeometryStage().run(state, None, None)


def test_defaults_and_yaml_roundtrip(tmp_path):
    """Auto is measured volumetric stereo for every direction; legacy full3d round-trips."""
    cfg = Config()
    assert cfg.geometry.dense_method == "auto"
    cfg.geometry.dense_method = "full3d"
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    assert load_config(path).geometry.dense_method == "full3d"
    preset = load_config(Path(__file__).parents[1] / "full_3d.yaml")
    assert preset.geometry.dense_method == "full3d"
    assert not preset.geometry.single_inference


def test_incremental_fusion_preserves_stacked_surfaces():
    from drishti3d.fusion.incremental import IncrementalVoxelFusion
    fusion = IncrementalVoxelFusion(voxel_size_m=0.1)
    points = np.array([[0., 0., 0.], [0., 0., 3.], [0., 0., 6.]])
    for view in range(3):
        fusion.add(points, np.full(3, view))
    np.testing.assert_allclose(np.sort(fusion.cloud().xyz[:, 2]), [0., 3., 6.])


def test_settings_mode_survives_config_loading(qtbot):
    from drishti3d.app.panels.settings_panel import SettingsPanel
    panel = SettingsPanel()
    qtbot.addWidget(panel)
    assert panel.build_config().geometry.dense_method == "auto"
    cfg = Config()
    cfg.geometry.dense_method = "heightfield"
    panel.apply_config(cfg)
    assert panel.build_config().geometry.dense_method == "heightfield"
    cfg.geometry.dense_method = "full3d"
    panel.apply_config(cfg)
    assert panel.build_config().geometry.dense_method == "full3d"


def test_finder_launch_uses_user_output_directory(monkeypatch):
    from drishti3d.app.panels.settings_panel import _default_output_root

    monkeypatch.chdir("/")
    output = _default_output_root()
    assert output.is_relative_to(Path.home())
    assert output != Path("/output")


def test_saved_stereo_run_reports_actual_engine_and_representation():
    from drishti3d.pipeline.result import PipelineResult, StageResult

    result = PipelineResult(
        mesh_faces=np.array([[0, 1, 2]]),
        report={"backbone": "mapanything", "reconstruction_representation": "2.5D height-field surface"},
        stage_results=[StageResult(name="geometry", status="ok", elapsed_s=1., artifacts={"backbone": "mvs3d"})],
    )
    assert result.report_card()["backbone"] == "mvs3d"
    assert result.report_card()["reconstruction_representation"] == "3D volumetric mesh"
