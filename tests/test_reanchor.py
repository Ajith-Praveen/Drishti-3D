"""Tests for fusion.reanchor (per-view BA re-anchoring) and fusion.mesh.decimate_to_cap."""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.fusion.mesh import decimate_to_cap
from drishti3d.fusion.reanchor import reanchor_submaps_to_ba
from drishti3d.geometry.windows import Window
from drishti3d.types import PointCloud, Pose, Submap

_NADIR_R = np.diag([1.0, -1.0, -1.0])


def _submap_with_depth_error(depth_factor: float, n_views: int = 3, altitude: float = 120.0, seed: int = 0):
    """Nadir views of a flat ground at z=0, reconstructed at ``depth_factor`` x true depth.

    Returns ``(submap, ba_points, obs_cam, obs_pt, world_poses)``. The BA
    points are true ground points seen by every camera; the submap's dense
    points are the same rays at the wrong depth.
    """
    rng = np.random.default_rng(seed)
    poses = [Pose(R=_NADIR_R, t=np.array([25.0 * i, 0.0, altitude])) for i in range(n_views)]
    xyz, vidx = [], []
    for v, p in enumerate(poses):
        # rays through random pixels: normalised coords in [-0.5, 0.5]
        n = rng.uniform(-0.5, 0.5, (4000, 2))
        z_true = altitude
        cam = np.column_stack([n * z_true, np.full(4000, z_true)])
        wrong = cam * depth_factor
        xyz.append(wrong @ p.R.T + p.t)
        vidx.append(np.full(4000, v, dtype=np.int32))
    xyz = np.concatenate(xyz)
    vidx = np.concatenate(vidx)
    sm = Submap(
        window=Window(index=0, start=0, end=n_views),
        poses=poses,
        points=PointCloud(xyz=xyz, confidence=np.full(len(xyz), 2, np.uint8)),
        confidence=np.ones(len(xyz)),
        keyframe_indices=list(range(n_views)),
        local_origin=poses[0],
        view_index=vidx,
    )
    # BA points: true ground under the strip, each observed by every camera
    # whose frustum contains it (normalised |x/z|,|y/z| <= 0.5).
    ba = np.column_stack([rng.uniform(-40, 25 * n_views + 40, 600), rng.uniform(-50, 50, 600), np.zeros(600)])
    obs_cam, obs_pt = [], []
    for k, p in enumerate(poses):
        q = (ba - p.t) @ p.R
        inside = (np.abs(q[:, 0] / q[:, 2]) <= 0.5) & (np.abs(q[:, 1] / q[:, 2]) <= 0.5) & (q[:, 2] > 0)
        for j in np.flatnonzero(inside):
            obs_cam.append(k)
            obs_pt.append(j)
    return sm, ba, np.array(obs_cam), np.array(obs_pt), poses


@pytest.mark.parametrize("depth_factor", [0.7, 1.4, 1.0])
def test_reanchor_corrects_each_view_to_ba_depth(depth_factor):
    sm, ba, oc, op, poses = _submap_with_depth_error(depth_factor)
    out, diag = reanchor_submaps_to_ba([sm], ba, oc, op, poses, match_radius=0.02)
    assert diag["views_reanchored"] == 3, diag
    assert diag["ratio_median"] == pytest.approx(1.0 / depth_factor, rel=0.03)
    # Ground is back at z ~ 0 for every view.
    z = out[0].points.xyz[:, 2]
    assert abs(np.median(z)) < 0.05 * 120.0


def test_reanchor_refuses_absurd_correction():
    sm, ba, oc, op, poses = _submap_with_depth_error(0.1)
    out, diag = reanchor_submaps_to_ba([sm], ba, oc, op, poses, match_radius=0.02)
    assert diag["views_reanchored"] == 0
    assert all("refused" in e for e in diag["per_view"] if "ratio" in e)
    np.testing.assert_array_equal(out[0].points.xyz, sm.points.xyz)


def test_reanchor_skips_submaps_without_view_index():
    sm, ba, oc, op, poses = _submap_with_depth_error(0.7)
    sm.view_index = None
    out, diag = reanchor_submaps_to_ba([sm], ba, oc, op, poses)
    assert out[0] is sm
    assert diag["views_reanchored"] == 0


def test_decimate_to_cap_bounds_faces_and_returns_channel_map():
    rng = np.random.default_rng(0)
    # A 200x200 height-field grid: 79,202 faces.
    n = 200
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    z = np.sin(xs / 10) + np.cos(ys / 13)
    v = np.column_stack([xs.ravel(), ys.ravel(), z.ravel()])
    idx = np.arange(n * n).reshape(n, n)
    f = np.concatenate(
        [
            np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()]),
            np.column_stack([idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]),
        ]
    )
    colours = rng.integers(0, 255, (len(v), 3), dtype=np.uint8)

    new_v, new_f, nn = decimate_to_cap(v, f, max_faces=10_000)
    assert new_f.shape[0] <= 10_000
    assert new_f.shape[0] > 2_000
    assert nn.shape == (new_v.shape[0],)
    remapped = colours[nn]
    assert remapped.shape == (new_v.shape[0], 3)
    # The surface is preserved: decimated vertices lie on the height field.
    assert np.abs(new_v[:, 2] - (np.sin(new_v[:, 0] / 10) + np.cos(new_v[:, 1] / 13))).max() < 0.35


def test_decimate_to_cap_is_identity_under_cap():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    f = np.array([[0, 1, 2]])
    nv, nf, nn = decimate_to_cap(v, f, max_faces=10)
    assert nv is v and nf is f
    np.testing.assert_array_equal(nn, [0, 1, 2])


def test_unmeasurable_view_inherits_submap_median_not_left_behind():
    """A view the BA points cannot measure must move with its neighbours.

    Leaving it at the old scale while the others are rescaled is what
    split the reconstructed ground over 368 m on a real run: 45 of 149
    views went unmeasured because 2,000 BA points over 77 cameras left
    ~26 observations each, under the 20-observation floor.
    """
    sm, ba, oc, op, poses = _submap_with_depth_error(0.5, n_views=3)
    # Starve the last camera of observations so it cannot be measured.
    keep = oc != 2
    oc2, op2 = oc[keep], op[keep]

    out, diag = reanchor_submaps_to_ba([sm], ba, oc2, op2, poses, match_radius=0.02)

    assert diag["views_reanchored"] == 2
    assert diag["views_inherited_submap_ratio"] == 1
    assert diag["views_left_unscaled"] == 0

    # Every view's ground now sits at the same corrected elevation, the
    # starved one included -- no slab left at the old depth.
    xyz, vidx = out[0].points.xyz, sm.view_index
    grounds = [np.median(xyz[vidx == v][:, 2]) for v in range(3)]
    assert max(grounds) - min(grounds) < 1.0, f"views split across {max(grounds)-min(grounds):.1f} m"
    assert abs(np.median(grounds)) < 6.0  # ~0 = true ground


def test_nothing_measurable_leaves_submap_untouched():
    """With no measurable view there is no median to inherit; report it."""
    sm, ba, oc, op, poses = _submap_with_depth_error(0.5, n_views=3)
    out, diag = reanchor_submaps_to_ba([sm], ba, np.array([]), np.array([]), poses)
    assert diag["views_reanchored"] == 0
    assert diag["views_inherited_submap_ratio"] == 0
    assert diag["views_left_unscaled"] == 3
    np.testing.assert_array_equal(out[0].points.xyz, sm.points.xyz)


def test_decimate_index_maps_into_original_vertices_after_prepass(monkeypatch):
    """The returned index must address the INPUT vertices, not an intermediate.

    Above the quadric size limit a cheap clustering pre-pass runs first.
    Computing the index against that pre-pass output makes it index the
    wrong array -- and the caller gathers original-length colour,
    confidence and semantic channels through it, pairing every one with
    the wrong geometry.

    The limit is monkeypatched rather than met with a real 8.8M-face mesh:
    the branch is what matters, and building the giant fixture made this
    test take minutes for no extra coverage.
    """
    from drishti3d.fusion import mesh as mesh_mod

    n = 120
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    v = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    idx = np.arange(n * n).reshape(n, n)
    f = np.concatenate([
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()]),
        np.column_stack([idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()]),
    ])
    # Force the pre-pass branch on a small mesh.
    monkeypatch.setattr(mesh_mod, "_QUADRIC_FACE_LIMIT", 1000)
    assert f.shape[0] > 1000

    new_v, new_f, nn = mesh_mod.decimate_to_cap(v, f, max_faces=2_000)

    assert new_f.shape[0] <= 2_000
    assert nn.min() >= 0 and nn.max() < len(v), f"index out of range for {len(v)} original vertices"
    # And it points at genuinely nearby original vertices, not an identity
    # map into the smaller pre-pass array.
    d = np.linalg.norm(v[nn][:, :2] - new_v[:, :2], axis=1)
    assert np.median(d) < 4.0, f"index does not address nearby original vertices (median {np.median(d):.1f})"
