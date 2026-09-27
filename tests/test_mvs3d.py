"""Measured full-3D reconstruction (geometry.mvs3d) on a rendered forward-looking scene with known surfaces."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drishti3d.geometry.mvs3d import Mvs3dConfig, reconstruct_mvs3d
from drishti3d.types import CameraIntrinsics, Confidence, Pose

_W, _H, _F = 320, 240, 250.0
_WALL_X = 12.0
_rng = np.random.default_rng(5)
_TEX = cv2.GaussianBlur((_rng.random((1200, 1200)) * 255).astype(np.float32), (0, 0), 1.2)


def _R(pitch_deg: float) -> np.ndarray:
    """World-from-camera: looking along +x, tilted down by pitch, z up."""
    a = np.radians(pitch_deg)
    fwd = np.array([np.cos(a), 0.0, -np.sin(a)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)


def _render(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    u, v = np.meshgrid(np.arange(_W) + 0.0, np.arange(_H) + 0.0)
    d = np.stack([(u - _W / 2) / _F, (v - _H / 2) / _F, np.ones_like(u)], -1) @ R.T
    t_g = np.where(d[..., 2] < -1e-6, -C[2] / np.where(d[..., 2] < -1e-6, d[..., 2], -1.0), np.inf)
    t_w = np.where(d[..., 0] > 1e-6, (_WALL_X - C[0]) / np.where(d[..., 0] > 1e-6, d[..., 0], 1.0), np.inf)
    pw = C + t_w[..., None] * d
    t_w = np.where((np.abs(pw[..., 1]) < 6) & (pw[..., 2] > 0) & (pw[..., 2] < 5), t_w, np.inf)
    wall = t_w < t_g
    t = np.where(wall, t_w, t_g)
    P = C + t[..., None] * d
    a = np.where(wall, P[..., 1] + 10, P[..., 0] + 5)  # texture coordinates (m)
    b = np.where(wall, P[..., 2] + 20, P[..., 1] + 10)
    grey = cv2.remap(_TEX, (a * 40).astype(np.float32), (b * 40).astype(np.float32), cv2.INTER_LINEAR)
    grey[~np.isfinite(t)] = 128
    return cv2.cvtColor(np.clip(grey, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("scale,reverse", [(1.0, False), (30.0, True)])
def test_measured_3d_recovers_ground_and_wall(scale, reverse):
    poses = [Pose(R=_R(15.0), t=np.array([0.6 * k, 0.3 * (k % 2), 1.5])) for k in range(7)]
    images = [_render(p.t, p.R) for p in poses]
    K = CameraIntrinsics(fx=_F, fy=_F, cx=_W / 2, cy=_H / 2, width=_W, height=_H)
    ground = np.c_[_rng.uniform(3, 11, 200), _rng.uniform(-3, 3, 200), np.zeros(200)]
    wall = np.c_[np.full(100, _WALL_X), _rng.uniform(-3, 3, 100), _rng.uniform(0, 4, 100)]
    poses = [Pose(R=p.R, t=p.t * scale) for p in poses]
    if reverse:
        images, poses = images[::-1], poses[::-1]
    cfg = Mvs3dConfig(max_side=320, hypotheses=64, voxel_m=0.08 * scale, min_fragment_triangles=50)
    surf, diag = reconstruct_mvs3d(images, [K] * len(poses), poses, points=np.r_[ground, wall] * scale, config=cfg, device="cpu")
    V = surf.vertices / scale
    assert diag["faces"] > 500 and len(V) > 500, diag
    # Every vertex lies on the ground (z = 0) or the wall (x = 12), within a few centimetres.
    dist = np.minimum(np.abs(V[:, 2]), np.abs(V[:, 0] - _WALL_X))
    assert np.median(dist) < 0.06, np.median(dist)
    assert (np.abs(V[:, 0] - _WALL_X) < 0.15).sum() > 50  # the wall is there, not just the ground
    assert np.mean(surf.confidence == int(Confidence.MEASURED)) > 0.5


def test_topk_mean_matches_torch_topk_with_ties():
    import torch

    from drishti3d.geometry.mvs3d import _topk_mean

    gen = torch.Generator().manual_seed(0)
    for n in (1, 2, 3, 4, 6):
        for k in (1, 2, 3):
            s = torch.randint(0, 4, (n, 6, 7), generator=gen).float()  # many ties
            kk = min(k, n)
            assert torch.allclose(_topk_mean(s, kk), torch.topk(s, kk, dim=0).values.mean(0))


def test_sgm_aggregation_removes_speckle_from_a_smooth_surface():
    import torch

    from drishti3d.geometry.mvs3d import _sgm_aggregate

    D, H, W = 32, 40, 50
    true = (8 + 0.2 * np.arange(W))[None, :].repeat(H, 0)  # a gentle slope, in hypothesis units
    d = np.arange(D)[:, None, None]
    cost = np.minimum(np.abs(d - true[None]) * 0.1, 1.0)
    noise = np.random.default_rng(1).random((D, H, W))
    cost = cost + 0.6 * (noise > 0.97)  # sparse spikes...
    junk = np.random.default_rng(2).random((H, W)) < 0.15  # ...and textureless pixels: flat, weakly noisy costs
    cost[:, junk] = 0.5 + 0.1 * np.random.default_rng(3).random((D, int(junk.sum())))
    c = torch.tensor(cost, dtype=torch.float32)
    wta = c.argmin(0).numpy()
    sgm = _sgm_aggregate(c, 0.05, 0.3, 4).argmin(0).numpy()
    err_wta = np.abs(wta - np.rint(true)).mean()
    err_sgm = np.abs(sgm - np.rint(true)).mean()
    assert err_sgm < 0.25 * err_wta and err_sgm < 0.2, (err_wta, err_sgm)


def test_speckle_filter_drops_small_islands_only():
    from drishti3d.geometry.mvs3d import _speckle_keep

    index = np.full((60, 60), 10.0)
    index[5:8, 5:8] = 40.0  # a 9 px island far off the surrounding surface
    index[30:50, 30:50] = 25.0  # a 400 px plateau: real structure
    keep = _speckle_keep(index, np.ones_like(index, dtype=bool), max_px=60, steps=2.0)
    assert not keep[5:8, 5:8].any()
    assert keep[30:50, 30:50].all() and keep[0, 0]


def _nadir_cameras(spacing: float, nx: int, ny: int, alt: float):
    R = np.diag([1.0, -1.0, -1.0])  # camera x = east, y = south, z = down
    return [Pose(R=R, t=np.array([i * spacing, j * spacing, alt])) for j in range(ny) for i in range(nx)]


def test_reference_subset_still_covers_every_footprint_cell():
    from drishti3d.geometry.mvs3d import Mvs3dConfig, _select_references

    poses = _nadir_cameras(4.0, 8, 8, 50.0)  # dense: each point seen by dozens of views
    K = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])
    R = np.stack([p.R for p in poses])
    C = np.stack([p.t for p in poses])
    pts = np.c_[np.random.default_rng(0).uniform(-10, 40, (500, 2)), np.zeros(500)]
    cfg = Mvs3dConfig(ref_coverage=3, ref_min_new=0.5)
    refs = _select_references(R, C, [K] * len(poses), 240, 320, pts, cfg)
    assert 3 <= len(refs) < 0.5 * len(poses)
    # Every ground point any view sees is inside at least one reference footprint.
    half = np.array([160.0, 120.0]) / 200.0 * 50.0
    seen = lambda idx: np.any([np.all(np.abs(pts[:, :2] - C[i, :2]) < half, axis=1) for i in idx], axis=0)  # noqa: E731
    assert np.array_equal(seen(refs), seen(range(len(poses))))
    # Forward-looking footage keeps every view.
    tilted = _R(15.0)
    Rf = np.stack([tilted] * 10)
    Cf = np.c_[np.arange(10.0), np.zeros(10), np.full(10, 1.5)]
    assert _select_references(Rf, Cf, [K] * 10, 240, 320, pts, cfg) == list(range(10))


def test_fill_holes_appends_faces_and_keeps_vertices():
    from drishti3d.geometry.mvs3d import _fill_holes

    x, y = np.meshgrid(np.arange(11.0), np.arange(11.0))
    V = np.c_[x.ravel(), y.ravel(), np.zeros(121)]
    faces = []
    for i in range(10):
        for j in range(10):
            if 4 <= i < 6 and 4 <= j < 6:
                continue  # a 2 x 2 m hole
            a, b, c, d = i * 11 + j, i * 11 + j + 1, (i + 1) * 11 + j, (i + 1) * 11 + j + 1
            faces += [[a, b, c], [b, d, c]]
    F = np.array(faces)
    F2, added, area = _fill_holes(V, F, 5.0)
    assert added > 0 and abs(area - 4.0) < 1e-6
    np.testing.assert_array_equal(F2[: len(F)], F)
    assert _fill_holes(V, F, 0.5)[1] == 0  # radius limit respected


def test_planar_uvs_span_the_texture_grid():
    from drishti3d.geometry.mvs3d import Mvs3dSurface

    V = np.array([[10.0, 20.0, 0.0], [30.0, 0.0, 5.0], [20.0, 10.0, 1.0]])
    s = Mvs3dSurface(vertices=V, faces=np.array([[0, 1, 2]]), colors=None, confidence=np.ones(3, np.uint8),
                     cell_m=0.5, texture_rgb=np.zeros((40, 40, 3), np.uint8), tex_xmin=10.0, tex_ymax=20.0,
                     tex_cell=0.5, tex_grid=(40, 40))
    np.testing.assert_allclose(s.mesh_vertex_uv(), [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])


_ROOF = (6.0, 4.0, 14.0, 12.0)  # x0, y0, x1, y1 of a box building
_ROOF_Z = 5.0


def _render_nadir(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Textured ground (z = 0) with a textured box building, seen by a pinhole camera."""
    u, v = np.meshgrid(np.arange(_W) + 0.0, np.arange(_H) + 0.0)
    d = np.stack([(u - _W / 2) / _F, (v - _H / 2) / _F, np.ones_like(u)], -1) @ R.T
    x0, y0, x1, y1 = _ROOF
    best = np.full(u.shape, np.inf)
    ta = np.zeros(u.shape)
    tb = np.zeros(u.shape)
    t = -C[2] / d[..., 2]  # ground
    P = C + t[..., None] * d
    best, ta, tb = t, P[..., 0] + 20, P[..., 1] + 20
    t = (_ROOF_Z - C[2]) / d[..., 2]  # roof
    P = C + t[..., None] * d
    hit = (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1) & (t < best)
    best, ta, tb = np.where(hit, t, best), np.where(hit, P[..., 0] + 60, ta), np.where(hit, P[..., 1] + 60, tb)
    for axis, val in ((0, x0), (0, x1), (1, y0), (1, y1)):  # walls
        t = (val - C[axis]) / np.where(np.abs(d[..., axis]) > 1e-9, d[..., axis], 1e-9)
        P = C + t[..., None] * d
        o = 1 - axis
        lo, hi = (y0, y1) if axis == 0 else (x0, x1)
        hit = (t > 0) & (P[..., o] > lo) & (P[..., o] < hi) & (P[..., 2] > 0) & (P[..., 2] < _ROOF_Z) & (t < best)
        best = np.where(hit, t, best)
        ta, tb = np.where(hit, P[..., o] + 100, ta), np.where(hit, P[..., 2] + 100, tb)
    grey = cv2.remap(_TEX, (ta * 8).astype(np.float32) % 1199, (tb * 8).astype(np.float32) % 1199, cv2.INTER_LINEAR)
    return cv2.cvtColor(np.clip(grey, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_downward_flight_measures_roof_and_ground_with_an_ortho_texture():
    poses = _nadir_cameras(3.0, 4, 3, 30.0)
    poses = [Pose(R=p.R, t=p.t + np.array([2.0, 2.0, 0.0])) for p in poses]
    images = [_render_nadir(p.t, p.R) for p in poses]
    K = CameraIntrinsics(fx=_F, fy=_F, cx=_W / 2, cy=_H / 2, width=_W, height=_H)
    rng = np.random.default_rng(7)
    ground = np.c_[rng.uniform(-5, 25, 300), rng.uniform(-5, 20, 300), np.zeros(300)]
    roof = np.c_[rng.uniform(6.5, 13.5, 60), rng.uniform(4.5, 11.5, 60), np.full(60, _ROOF_Z)]
    cfg = Mvs3dConfig(max_side=320, voxel_m=0.1, min_fragment_triangles=50, texture_upsample=2)
    surf, diag = reconstruct_mvs3d(images, [K] * len(poses), poses, points=np.r_[ground, roof], config=cfg, device="cpu")
    V = surf.vertices
    on_roof = (V[:, 0] > 7) & (V[:, 0] < 13) & (V[:, 1] > 5) & (V[:, 1] < 11)
    away = (V[:, 0] < 3) | (V[:, 0] > 17) | (V[:, 1] < 1) | (V[:, 1] > 15)
    assert on_roof.sum() > 200 and away.sum() > 200, diag
    assert np.median(np.abs(V[on_roof, 2] - _ROOF_Z)) < 0.05
    assert np.median(np.abs(V[away, 2])) < 0.05
    assert surf.texture_rgb is not None and surf.texture_rgb.max() > 0
    uv = surf.mesh_vertex_uv()
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    assert diag["references"] <= len(poses)


def test_excluded_pixels_never_become_geometry():
    """A semantics mask (a parked truck on the roof, say) removes those pixels from every depth map."""
    poses = _nadir_cameras(3.0, 4, 3, 30.0)
    poses = [Pose(R=p.R, t=p.t + np.array([2.0, 2.0, 0.0])) for p in poses]
    images = [_render_nadir(p.t, p.R) for p in poses]
    K = CameraIntrinsics(fx=_F, fy=_F, cx=_W / 2, cy=_H / 2, width=_W, height=_H)
    rng = np.random.default_rng(7)
    ground = np.c_[rng.uniform(-5, 25, 300), rng.uniform(-5, 20, 300), np.zeros(300)]
    roof = np.c_[rng.uniform(6.5, 13.5, 60), rng.uniform(4.5, 11.5, 60), np.full(60, _ROOF_Z)]
    # Mask every pixel whose ray hits the roof (the roof plane, inside its outline).
    masks = []
    u, v = np.meshgrid(np.arange(_W) + 0.0, np.arange(_H) + 0.0)
    for p in poses:
        d = np.stack([(u - _W / 2) / _F, (v - _H / 2) / _F, np.ones_like(u)], -1) @ p.R.T
        P = p.t + ((_ROOF_Z - p.t[2]) / d[..., 2])[..., None] * d
        x0, y0, x1, y1 = _ROOF
        masks.append((P[..., 0] > x0 - 0.5) & (P[..., 0] < x1 + 0.5) & (P[..., 1] > y0 - 0.5) & (P[..., 1] < y1 + 0.5))
    cfg = Mvs3dConfig(max_side=320, voxel_m=0.1, min_fragment_triangles=50, texture_upsample=0, fill_holes_m=0.0)
    surf, diag = reconstruct_mvs3d(images, [K] * len(poses), poses, points=np.r_[ground, roof], config=cfg,
                                   device="cpu", exclude_masks=masks)
    V = surf.vertices
    on_roof = (V[:, 0] > 7) & (V[:, 0] < 13) & (V[:, 1] > 5) & (V[:, 1] < 11) & (V[:, 2] > 2.5)
    assert on_roof.sum() == 0, on_roof.sum()
    assert diag["depth_px_masked"] > 0
    assert (np.abs(V[:, 2]) < 0.1).sum() > 200  # the ground is still there
