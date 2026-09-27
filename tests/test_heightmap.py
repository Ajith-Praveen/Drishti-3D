"""Tests for drishti3d.fusion.heightmap -- median-vote 2.5D fusion."""

from __future__ import annotations

import numpy as np

from drishti3d.fusion.heightmap import HeightmapFusion, cell_size_for
from drishti3d.types import Confidence


def _patch(z: float, kf: int, n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed + kf)
    xy = rng.uniform(0.0, 5.0, (n, 2))
    return np.c_[xy, np.full(n, z) + rng.normal(0, 0.05, n)], np.full(n, kf)


def test_median_vote_ignores_one_bad_view_and_tiers_by_agreement() -> None:
    hm = HeightmapFusion(cell_m=1.0, agree_m=1.0)
    for kf, z in [(0, 10.0), (1, 10.2), (2, 9.9), (3, 40.0)]:  # view 3 is 30 m off
        xyz, ids = _patch(z, kf)
        hm.add(xyz, ids)
    pc = hm.cloud()

    assert np.all(np.abs(pc.xyz[:, 2] - 10.05) < 0.3)  # median of 4 votes, not the 17.5 m mean
    assert np.all(pc.confidence == Confidence.MEASURED)  # three views agree within 1 m


def test_two_views_that_disagree_are_inferred() -> None:
    hm = HeightmapFusion(cell_m=1.0, agree_m=1.0)
    for kf, z in [(0, 10.0), (1, 14.0)]:
        xyz, ids = _patch(z, kf)
        hm.add(xyz, ids)
    assert np.all(hm.cloud().confidence == Confidence.INFERRED)


def test_mesh_joins_the_grid_but_not_across_a_cliff() -> None:
    hm = HeightmapFusion(cell_m=1.0, max_step_m=2.0)
    xs, ys = np.meshgrid(np.arange(0.5, 10), np.arange(0.5, 4))
    z = np.where(xs < 5, 0.0, 20.0)  # a 20 m cliff at x = 5
    hm.add(np.c_[xs.ravel(), ys.ravel(), z.ravel()], np.zeros(xs.size, int))
    verts, faces = hm.mesh()

    assert faces.shape[1] == 3 and len(faces) > 0
    span = verts.xyz[faces][:, :, 2].max(1) - verts.xyz[faces][:, :, 2].min(1)
    assert span.max() <= 2.0  # nothing bridges the cliff
    # 10 x 4 cells, minus the 3 blocks straddling the cliff: (9 - 1) x 3 blocks x 2 triangles.
    assert len(faces) == 48


def test_cell_size_tracks_point_spacing() -> None:
    g = np.arange(0, 50, 0.8)
    X, Y = np.meshgrid(g, g)
    assert 1.0 < cell_size_for(np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]) < 1.4


def test_dense_view_selection_drops_redundant_legs_but_keeps_area() -> None:
    import types

    from drishti3d.geometry.bundle import gimbal_to_R
    from drishti3d.pipeline.stages import _select_dense_views
    from drishti3d.types import CameraIntrinsics, Pose

    K = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    poses, kfs = [], []
    # Four legs flown over the SAME strip (x 0..400, y 0), 10 m apart along track, 100 m up.
    for leg in range(4):
        for x in range(0, 400, 10):
            poses.append(Pose(R=gimbal_to_R(0.0, -90.0, 0.0), t=np.array([float(x), 2.0 * leg, 100.0])))
            kfs.append(types.SimpleNamespace(intrinsics=K))
    state = types.SimpleNamespace(fit_points=np.zeros((100, 3)), intrinsics=K)

    keep = _select_dense_views(state, kfs, poses, target_views=3)

    first_leg = set(range(40))
    assert len(keep) < len(poses) / 2  # the re-flown legs add nothing new
    assert len(keep & first_leg) >= 10  # the first pass still provides the coverage


def test_even_disagreeing_votes_never_invent_a_height_or_go_black() -> None:
    hm = HeightmapFusion(cell_m=1.0, agree_m=1.0)
    for kf, z in [(0, 0.0), (1, 0.1), (2, 5.0), (3, 5.1)]:
        xyz, ids = _patch(z, kf, n=200)
        hm.add(xyz, ids, rgb=np.full((200, 3), 120.0))
    pc = hm.cloud()

    # Heights come from an actual cluster (~0.05 or ~5.05), never the 2.55 midpoint.
    assert np.all((np.abs(pc.xyz[:, 2] - 0.05) < 0.3) | (np.abs(pc.xyz[:, 2] - 5.05) < 0.3))
    assert np.all(pc.rgb > 0)  # coloured from the agreeing views
    assert np.all(pc.confidence == Confidence.LOW_CONFIDENCE)  # two views agree


def test_heightmap_stage_removes_video_inconsistent_vertices_and_faces(monkeypatch):
    from types import SimpleNamespace

    from drishti3d.pipeline.stages import FusionStage

    hm = HeightmapFusion(cell_m=1)
    x, y = np.meshgrid(np.arange(0.5, 3), np.arange(0.5, 3))
    hm.add(np.c_[x.ravel(), y.ravel(), np.zeros(x.size)], np.zeros(x.size, int))
    before, faces = hm.mesh()
    keep = np.ones(len(before.xyz), bool)
    keep[4] = False
    stage = FusionStage()
    monkeypatch.setattr(stage, "_photometric_point_filter", lambda state: lambda cloud: keep)
    state = SimpleNamespace()
    artifacts, _ = stage._from_heightmap(state, hm)
    assert artifacts["photometric_points_rejected"] == 1
    np.testing.assert_array_equal(state.point_cloud.xyz, before.xyz[keep])
    assert artifacts["faces"] == int(keep[faces].all(axis=1).sum())
    assert state.mesh_faces.max() < len(state.point_cloud.xyz)
