"""Tests for drishti3d.fusion.incremental -- running voxel fusion of world-frame windows."""

from __future__ import annotations

import numpy as np

from drishti3d.fusion.incremental import IncrementalVoxelFusion
from drishti3d.geometry.submap import _chain_align
from drishti3d.types import Confidence


def test_windows_fuse_into_one_model_with_view_tiers() -> None:
    fusion = IncrementalVoxelFusion(voxel_size_m=1.0)
    p = np.array([[0.2, 0.2, 0.2]])
    fusion.add(p, np.array([0]), rgb=np.array([[100, 0, 0]]))
    fusion.add(np.array([[0.4, 0.4, 0.4], [5.5, 0.5, 0.5]]), np.array([1, 1]), rgb=np.array([[200, 0, 0], [0, 0, 0]]))
    fusion.add(np.array([[0.6, 0.6, 0.6]]), np.array([2]))

    cloud = fusion.cloud()
    assert len(fusion) == 2
    assert fusion.windows_fused == 3
    shared = np.argmin(np.linalg.norm(cloud.xyz - 0.4, axis=1))
    np.testing.assert_allclose(cloud.xyz[shared], [0.4, 0.4, 0.4])
    assert cloud.confidence[shared] == Confidence.MEASURED
    assert cloud.confidence[1 - shared] == Confidence.INFERRED


def test_confidence_weights_the_mean_and_zero_weight_is_dropped() -> None:
    fusion = IncrementalVoxelFusion(voxel_size_m=10.0)
    fusion.add(np.array([[0.0, 0, 0], [3.0, 0, 0], [9.0, 0, 0]]), np.array([0, 1, 2]), weight=np.array([1.0, 3.0, 0.0]))

    cloud = fusion.cloud()
    np.testing.assert_allclose(cloud.xyz[0], [2.25, 0, 0])
    assert cloud.confidence[0] == Confidence.LOW_CONFIDENCE


def test_world_frame_strategy_is_identity() -> None:
    class _Sm:
        pass

    transforms, diags, _, anchored = _chain_align([_Sm(), _Sm()], strategy="world_frame")
    assert all(t.scale == 1.0 and np.allclose(t.R, np.eye(3)) and np.allclose(t.t, 0) for t in transforms)
    assert all(anchored) and diags[0]["method"] == "world_frame"


def test_views_agreeing_within_tier_cell_are_measured_despite_fine_voxels() -> None:
    fusion = IncrementalVoxelFusion(voxel_size_m=0.3, tier_voxel_m=1.0)
    # Three keyframes see the same spot 0.4 m apart: three fine voxels, one 1 m cell.
    fusion.add(np.array([[0.1, 0.1, 0.1], [0.5, 0.1, 0.1], [0.9, 0.1, 0.1]]), np.array([0, 1, 2]))

    cloud = fusion.cloud()
    assert len(fusion) == 3
    assert (cloud.confidence == Confidence.MEASURED).all()


def test_each_keyframe_has_exactly_one_owning_window() -> None:
    from drishti3d.pipeline.stages import _view_owners

    class _W:
        def __init__(self, kfs):
            self._k = kfs

        def keyframe_indices(self):
            return self._k

    windows = [_W(list(range(0, 8))), _W(list(range(5, 13))), _W(list(range(10, 18)))]
    owners = _view_owners(windows)

    assert set(owners) == set(range(18))
    # Keyframe 6 is 1 from the end of window 0 but 1 from the start of window 1: tie -> earlier.
    assert owners[6] == 0
    # Keyframe 7 is at window 0's edge but 2 deep in window 1.
    assert owners[7] == 1
    assert owners[0] == 0 and owners[17] == 2


def test_keyframes_owned_by_a_failed_window_are_recovered_from_a_neighbour() -> None:
    import types

    from drishti3d.pipeline.stages import _recover_orphaned_views
    from drishti3d.types import Pose

    pose = Pose(R=np.eye(3), t=np.zeros(3))
    state = types.SimpleNamespace(
        view_owner={5: 0, 6: 0, 7: 1},
        spare_views={6: ("w1", np.ones((10, 3)), np.ones(10), None, pose), 7: ("w0", np.ones((4, 3)), np.ones(4), None, pose)},
        dense_views=None,
    )
    submaps, fused = [], []
    n = _recover_orphaned_views(state, [], {0: "excluded"}, submaps, fused.append)

    # Window 0 failed: keyframe 6 comes back from window 1's spare; 5 had no other copy;
    # 7's owner (window 1) succeeded, so its spare is NOT re-emitted.
    assert n == 1
    assert [sm.keyframe_indices for sm in submaps] == [[6]]
    assert fused == submaps
