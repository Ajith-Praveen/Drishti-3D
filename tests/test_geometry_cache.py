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
    assert dst.keyframes[1].pose.t[0] == 1.0
    loaded = load_geometry_cache(tmp_path, dst)

    assert loaded is not None
    artifacts, message = loaded
    assert artifacts == {"windows": 2} and "two windows" in message
    assert dst.submaps == ["submap-a", "submap-b"]
    np.testing.assert_array_equal(dst.point_cloud.xyz, np.arange(9.0).reshape(3, 3))
    assert dst.depth_anchor_summary == {"windows_anchored": 2}
    # The rewritten conditioning pose came back with the cache.
    assert dst.keyframes[1].pose.t[0] == 9.0


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
