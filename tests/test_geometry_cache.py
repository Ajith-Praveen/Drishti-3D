"""Tests for pipeline.geometry_cache: round trip and key checking."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from drishti3d.config import Config
from drishti3d.pipeline.geometry_cache import load_geometry_cache, save_geometry_cache
from drishti3d.types import PointCloud, Pose


def _state(n_kf=3, backbone="null"):
    cfg = Config()
    kfs = [SimpleNamespace(frame_index=10 * i, pose=Pose(R=np.eye(3), t=np.array([i, 0.0, 0.0]))) for i in range(n_kf)]
    return SimpleNamespace(
        config=cfg,
        backbone_name=backbone,
        keyframes=kfs,
        submaps=["submap-a", "submap-b"],
        windows=["w0", "w1"],
        point_cloud=PointCloud(xyz=np.arange(9.0).reshape(3, 3)),
        poses=[kf.pose for kf in kfs],
        depth_anchor_summary={"windows_anchored": 2},
    )


def test_round_trip_restores_state_and_keyframe_poses(tmp_path):
    src = _state()
    src.keyframes[1].pose = Pose(R=np.eye(3), t=np.array([9.0, 9.0, 9.0]))  # what yaw-from-flow does
    save_geometry_cache(tmp_path, src, {"windows": 2}, "two windows")

    dst = _state()
    dst.keyframes[1].pose = src.keyframes[1].pose
    loaded = load_geometry_cache(tmp_path, dst)

    assert loaded is not None
    artifacts, message = loaded
    assert artifacts == {"windows": 2} and "two windows" in message
    assert dst.submaps == ["submap-a", "submap-b"]
    np.testing.assert_array_equal(dst.point_cloud.xyz, np.arange(9.0).reshape(3, 3))
    assert dst.depth_anchor_summary == {"windows_anchored": 2}
    # The rewritten conditioning pose came back with the cache.
    assert dst.keyframes[1].pose.t[0] == 9.0


def test_changed_conditioning_refuses_stale_geometry(tmp_path):
    src = _state()
    save_geometry_cache(tmp_path, src, {}, "")
    dst = _state()
    dst.keyframes[1].pose = Pose(R=np.eye(3), t=np.array([9.0, 9.0, 9.0]))
    assert load_geometry_cache(tmp_path, dst) is None


def test_placement_failure_survives_cache_roundtrip(tmp_path):
    from types import SimpleNamespace

    src = _state()
    src.placement_report = SimpleNamespace(verdict="FAIL")
    save_geometry_cache(tmp_path, src, {}, "")
    dst = _state()
    assert load_geometry_cache(tmp_path, dst) is not None
    assert dst.placement_report.verdict == "FAIL"


def test_missing_cache_is_none(tmp_path):
    assert load_geometry_cache(tmp_path, _state()) is None


def test_different_keyframes_refuse_to_load(tmp_path):
    save_geometry_cache(tmp_path, _state(n_kf=3), {}, "")
    dst = _state(n_kf=4)
    assert load_geometry_cache(tmp_path, dst) is None
    assert dst.submaps == ["submap-a", "submap-b"]  # untouched


def test_changed_geometry_setting_refuses_to_load(tmp_path):
    """Any geometry knob is part of the key: stale output must not masquerade as a setting's effect."""
    save_geometry_cache(tmp_path, _state(), {}, "")
    dst = _state()
    dst.config.geometry.depth_anchor = not dst.config.geometry.depth_anchor
    assert load_geometry_cache(tmp_path, dst) is None


def test_different_backbone_refuses_to_load(tmp_path):
    save_geometry_cache(tmp_path, _state(backbone="null"), {}, "")
    assert load_geometry_cache(tmp_path, _state(backbone="mapanything")) is None


def test_terrain_cache_cannot_be_reused_as_full3d(tmp_path):
    source = _state()
    source.config.geometry.dense_method = "heightfield"
    save_geometry_cache(tmp_path, source, {}, "terrain")
    assert load_geometry_cache(tmp_path, _state()) is None


def test_measured_mesh_survives_cache_without_heightmap_conversion(tmp_path):
    from drishti3d.geometry.mvs3d import Mvs3dSurface
    from drishti3d.pipeline.stages import FusionStage

    source = _state()
    source.geometry_heightmap = False
    source.geometry_premeshed = True
    # A vertical triangle cannot be represented by one height per XY.
    source.incremental_fusion = Mvs3dSurface(
        vertices=np.array([[0., 0., 0.], [0., 0., 3.], [1., 0., 3.]]),
        faces=np.array([[0, 1, 2]]), colors=np.full((3, 3), 128, dtype=np.uint8),
        confidence=np.ones(3, dtype=np.uint8), cell_m=0.1,
    )
    source.mvs3d_diagnostics = {"method": "mvs3d"}
    save_geometry_cache(tmp_path, source, {}, "3D mesh")
    restored = _state()
    assert load_geometry_cache(tmp_path, restored) is not None
    artifacts, message = FusionStage().run(restored, None, None)
    assert artifacts["method"] == "mvs3d_tsdf"
    assert "volumetric stereo mesh" in message
    assert not restored.geometry_heightmap
    assert restored.fusion_confidence_source == "multiview_depth_consistency"
    np.testing.assert_array_equal(restored.point_cloud.xyz, source.incremental_fusion.vertices)
    np.testing.assert_array_equal(restored.mesh_faces, [[0, 1, 2]])
