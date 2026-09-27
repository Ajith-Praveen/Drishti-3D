"""Measured full-3D reconstruction: plane-sweep depth per camera, cross-view consistency, TSDF fusion.

Works for every viewing direction -- a camera 1 m above a vineyard looking
along the rows, and a mapping flight 280 m up looking down -- because every
depth is measured per camera and fused in 3D, so walls, overhangs and
stacked surfaces stay 3D instead of collapsing to one height per ground
position (``geometry.heightfield``). Nothing unobserved is filled in.

1. **References.** A depth map per keyframe, or -- on dense mapping flights
   where a ground point is seen by a dozen frames (PinPoint flight01: 14) --
   a subset chosen so every part of the scene is still covered by
   ``ref_coverage`` depth maps. All keyframes remain available as sources.
2. **Sweep.** For each reference, hypotheses uniform in inverse depth over
   the range the bundle-adjusted points give that view; each pixel is
   projected into its best source views at every hypothesis and scored by
   zero-mean NCC over a window (box filters on the GPU). The mean of the
   best ``top_k`` sources is kept per pixel, so a view that is occluded
   there cannot veto it.
3. **Semi-global matching.** The cost volume (1 - NCC) is aggregated along
   four scanlines with small/large jump penalties before the winner is taken:
   winner-take-all on raw NCC is speckled at aerial range. Sub-step
   parabolic refinement; rejection at low NCC, at the range's edge, and for
   isolated islands (speckle filter).
4. **Consistency.** A depth survives only if at least ``min_consistent``
   neighbouring references' depth maps agree with it where it reprojects,
   in depth (``consistency_rel``) and in the image after the round trip
   (``consistency_px``) -- the fusion test COLMAP and MVSNet use.
5. **Fusion.** Open3D's block-sparse TSDF integrates the surviving depth maps
   with their colours along each camera's rays (so free space is carved and
   a surface comes out once, not as a slab); marching cubes extracts the
   mesh; small disconnected fragments are dropped.

Vertices within 1.5 voxels of a consistency-checked depth sample are tiered
MEASURED; the rest (surface TSDF interpolated between samples) LOW.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from drishti3d.types import CameraIntrinsics, Confidence, PointCloud, Pose

logger = logging.getLogger(__name__)


@dataclass
class Mvs3dConfig:
    max_side: int = 640
    #: Depth hypotheses per view; 0 = automatic: enough that one step is
    #: ``hypothesis_step_rel`` of the median depth, within [48, 128]
    #: (aerial views need 48, a camera 1 m above a vineyard the maximum).
    hypotheses: int = 0
    hypothesis_step_rel: float = 0.005
    sources: int = 4
    top_k: int = 2
    #: NCC window, pixels. 7 against 5 on flight01's fields: depth fill 78 -> 82%,
    #: two-partner agreement 87 -> 92%.
    window: int = 7
    min_ncc: float = 0.4
    min_baseline_m: float = 0.25
    max_axis_angle_deg: float = 50.0
    consistency_rel: float = 0.01
    consistency_px: float = 1.0
    min_consistent: int = 2
    voxel_m: float | None = None
    min_fragment_triangles: int = 500
    #: Depth hypotheses evaluated per GPU batch.
    batch: int = 8
    #: Also sweep horizontal planes (heights) for rays below the horizon: seen
    #: from ~1 m up, ground is steeply slanted and fronto-parallel windows
    #: do not line up on it, so it went unmeasured (the near field of
    #: Front_View_Light). 0 disables.
    ground_heights: int = 0  # measured on Front_View_Light: +20 s, no gain (158k vs 165k vertices)
    #: Voxel = this x the median depth sample's pixel footprint (when voxel_m is unset).
    voxel_footprints: float = 1.5
    #: Semi-global matching over the plane-sweep cost volume (Hirschmueller
    #: 2008, 4 scanline paths). Winner-take-all on raw 5 x 5 NCC is speckled
    #: at aerial range (DJI_1001, 280 m: local depth roughness p90 0.75 m);
    #: SGM's smoothness penalties halve that at +0.5 s per 924 px view.
    #: Costs are 1 - NCC (0..2); ``sgm_p1`` penalises a one-hypothesis step
    #: between neighbouring pixels (slopes), ``sgm_p2`` any larger jump
    #: (depth edges). 0 paths disables it.
    sgm_paths: int = 4
    sgm_p1: float = 0.05
    sgm_p2: float = 0.3
    #: Depth range from the solved points the view sees: these percentiles,
    #: widened by ``range_pad`` of their span (at least 2 m) each way.
    range_percentiles: tuple[float, float] = (1.0, 99.0)
    range_pad: float = 0.25
    #: Isolated depth islands smaller than this many pixels (differing from
    #: their surroundings by more than ``speckle_steps`` hypotheses) are dropped.
    speckle_px: int = 60
    speckle_steps: float = 2.0
    #: TSDF truncation, in voxels.
    trunc_voxels: float = 4.0
    #: Photometric refinement samples across +-1 hypothesis step, used when
    #: a step exceeds a quarter of ``consistency_rel`` of the depth.
    refine_samples: int = 9
    #: Reference views: enough that every footprint cell is covered by this
    #: many depth maps (0 = every view is a reference). Only for flights
    #: looking mostly down, where footprints on the ground are well defined.
    ref_coverage: int = 5
    #: A candidate reference is taken when at least this share of its
    #: footprint is still covered by fewer than ``ref_coverage`` references.
    ref_min_new: float = 0.5
    #: Neighbouring references each depth map is checked against.
    partners: int = 6
    #: Downward flights also get a true-ortho texture of the top surface at
    #: voxel / ``texture_upsample`` (orthomosaic.tif, textured OBJ/glTF).
    #: 0 disables it.
    texture_upsample: int = 3
    texture_max_px: int = 8192
    #: Downward flights: close enclosed gaps up to this radius (m) in the
    #: mesh -- the unseen ground under a canopy's rim, the strip between a
    #: roof edge and the street -- by triangulating their boundary loops
    #: (no new vertices, so every vertex stays measured; the added area is
    #: reported). Seen from above a nadir flight cannot observe them; from
    #: the side they were see-through holes. 0 disables. Not for forward
    #: views: there a loop can span open air between vine rows.
    fill_holes_m: float = 8.0


@dataclass
class Mvs3dSurface:
    """The fused mesh, with the read-out interface FusionStage uses for measured surfaces."""

    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    cell_m: float
    photo_consistent: bool = True
    #: (ty, tx, 3) uint8 true-ortho texture of the top surface (downward
    #: flights), over the grid whose top-left corner is (tex_xmin, tex_ymax)
    #: with ``tex_cell`` metre cells (texels are finer: shape / grid size).
    texture_rgb: np.ndarray | None = None
    tex_xmin: float = 0.0
    tex_ymax: float = 0.0
    tex_cell: float = 1.0
    tex_grid: tuple[int, int] = (1, 1)  # (ny, nx) cells

    def mesh(self) -> tuple[PointCloud, np.ndarray]:
        return PointCloud(xyz=self.vertices, rgb=self.colors, confidence=self.confidence), self.faces

    def vertex_uv(self) -> np.ndarray:
        """Planar (top-down) UVs of every vertex into ``texture_rgb`` (OBJ/glTF: v up)."""
        ny, nx = self.tex_grid
        u = (self.vertices[:, 0] - self.tex_xmin) / (nx * self.tex_cell)
        v = 1.0 - (self.tex_ymax - self.vertices[:, 1]) / (ny * self.tex_cell)
        return np.stack([u, v], axis=-1)

    def mesh_vertex_uv(self) -> np.ndarray:
        return self.vertex_uv()


def _box(F, x, k):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def _topk_mean(stack, k: int):
    """Mean of the ``k`` largest along dim 0, by pairwise ranking.

    ``torch.topk``/``sort`` along a short leading dim run ~130x slower on
    MPS (488 ms vs 3.7 ms for 4 x 8 x 520 x 924) -- they were 75% of the
    sweep. Ranking with elementwise comparisons is exact, ties included.
    """
    import torch

    n = stack.shape[0]
    if k >= n:
        return stack.mean(0)
    if k == 1:
        return stack.amax(0)
    total = torch.zeros_like(stack[0])
    for i in range(n):
        rank = torch.zeros_like(stack[0])
        for j in range(n):
            if j != i:
                rank += (stack[j] > stack[i]) if j > i else (stack[j] >= stack[i])
        total += torch.where(rank < k, stack[i], torch.zeros_like(stack[i]))
    return total / k


def _sgm_aggregate(cost, p1: float, p2: float, paths: int = 4):
    """Sum of semi-global path costs over a (D, H, W) cost volume, lower is better.

    L_r(p, d) = C(p, d) + min(L_r(p-r, d), L_r(p-r, d+-1) + P1, min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k),
    along left/right/down/up scanlines (``paths`` 2 keeps the horizontal pair only).
    """
    import torch

    total = torch.zeros_like(cost)
    passes = (((2, 0, 1), False), ((2, 0, 1), True), ((1, 0, 2), False), ((1, 0, 2), True))[: max(1, paths)]
    for perm, reverse in passes:
        scan = cost.permute(*perm).contiguous()  # (n, D, m): scanned dim first
        out = torch.empty_like(scan)
        wall = torch.full((1, scan.shape[2]), 1e4, device=cost.device, dtype=cost.dtype)
        order = range(scan.shape[0] - 1, -1, -1) if reverse else range(scan.shape[0])
        prev = None
        for x in order:
            c = scan[x]
            if prev is None:
                cur = c
            else:
                m = prev.amin(0, keepdim=True)
                step = torch.minimum(torch.cat([prev[1:], wall], 0), torch.cat([wall, prev[:-1]], 0)) + p1
                cur = c + torch.minimum(torch.minimum(prev, step), m + p2) - m
            out[x] = cur
            prev = cur
        inverse = [0, 0, 0]
        for k, p in enumerate(perm):
            inverse[p] = k
        total += out.permute(*inverse)
    return total


def _speckle_keep(index: np.ndarray, valid: np.ndarray, max_px: int, steps: float) -> np.ndarray:
    """False on connected islands under ``max_px`` pixels (OpenCV filterSpeckles on the hypothesis index).

    ``index`` is each pixel's (fractional) hypothesis index, so "neighbours
    join when they differ by at most ``steps`` hypotheses" is one integer
    threshold over the whole inverse-depth range.
    """
    import cv2

    if max_px <= 0 or not valid.any():
        return valid
    q = np.zeros(index.shape, np.int16)
    q[valid] = np.clip(np.rint(index[valid]) + 1, 1, 32000).astype(np.int16)  # 0 = invalid, never joined
    cv2.filterSpeckles(q, 0, int(max_px), int(np.ceil(steps)))
    return valid & (q != 0)


def _tsdf_mesh(depth_maps, rgb, Ks, R, C, voxel: float, trunc_voxels: float, depth_max: float, progress=None):
    """TSDF-fuse per-view depth maps (metres, 0 = none) with colours into a legacy Open3D triangle mesh.

    Open3D's tensor ``VoxelBlockGrid`` integrates ~17x faster than the
    legacy ``ScalableTSDFVolume`` (1.0 s vs 19.2 s for 7 DJI_1001 views at
    924 px) and grows its block table as needed. Extraction keeps voxels
    with weight >= 0.5, which reproduces the legacy volume's output.
    """
    import open3d as o3d
    import open3d.core as o3c

    dev = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=float(voxel),
        block_resolution=8,
        block_count=50_000,
        device=dev,
    )
    n = len(depth_maps)
    for r in range(n):
        if int((depth_maps[r] > 0).sum()) < 100:
            continue
        if progress is not None:
            progress(r, n, f"volumetric fusion {r + 1}/{n}")
        depth = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth_maps[r], dtype=np.float32))).to(dev)
        color = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb[r], dtype=np.float32) / 255.0)).to(dev)
        K = o3c.Tensor(np.asarray(Ks[r], dtype=np.float64))
        ext = np.eye(4)
        ext[:3, :3] = R[r].T
        ext[:3, 3] = -R[r].T @ C[r]
        E = o3c.Tensor(ext)
        try:
            blocks = vbg.compute_unique_block_coordinates(
                depth, K, E, 1.0, float(depth_max), trunc_voxel_multiplier=float(trunc_voxels)
            )
        except RuntimeError:  # "No block is touched": nothing of this view survived to fuse
            logger.debug("mvs3d: view %d touches no TSDF block; skipped", r)
            continue
        vbg.integrate(blocks, depth, color, K, K, E, 1.0, float(depth_max), trunc_voxel_multiplier=float(trunc_voxels))
    return vbg.extract_triangle_mesh(weight_threshold=0.5).to_legacy()


def _ortho_texture(V, images, intrinsics, poses, cell: float, device, up: int, max_px: int, fill_radius_m: float = 0.0):
    """True-ortho texture of the mesh's top surface: (rgb, xmin, ymax, (ny, nx)).

    The top surface is the highest vertex per ``cell``; texels ``up`` times
    finer sample the median colour of the nearest full-resolution views at
    that height (``geometry.heightfield``'s colouriser). Texels more than two
    cells from any vertex stay black (no data) -- except enclosed gaps the
    mesh's hole filling closes (``fill_radius_m``): those are coloured from
    the views like any surface, so the road where a moving car's depth was
    rejected shows the road the other frames saw, not a black blob.
    """
    import cv2
    from scipy import ndimage

    from drishti3d.geometry.heightfield import HeightfieldConfig, _colorize, _Views

    xmin, ymin = V[:, :2].min(axis=0) - cell
    xmax, ymax = V[:, :2].max(axis=0) + cell
    nx, ny = int(np.ceil((xmax - xmin) / cell)), int(np.ceil((ymax - ymin) / cell))
    col = np.clip(((V[:, 0] - xmin) / cell).astype(np.int64), 0, nx - 1)
    row = np.clip(((ymax - V[:, 1]) / cell).astype(np.int64), 0, ny - 1)
    Z = np.full((ny, nx), -np.inf)
    np.maximum.at(Z, (row, col), V[:, 2])
    known = np.isfinite(Z)
    far, (ir, ic) = ndimage.distance_transform_edt(~known, return_indices=True)
    # Texels no finer than the frames' own ground sample distance: finer is
    # only resampling (flight01: 0.09 m texels from 0.11 m pixels, 119 MB).
    heights = np.array([p.t[2] for p in poses], dtype=np.float64) - float(np.median(V[:, 2]))
    gsd = float(np.median(heights / np.array([k.fx for k in intrinsics], dtype=np.float64)))
    if np.isfinite(gsd) and gsd > 0:
        up = min(int(up), max(1, round(cell / gsd)))
    up = max(1, min(int(up), int(max_px) // max(nx, ny)))
    ztex = cv2.resize(Z[ir, ic].astype(np.float32), (nx * up, ny * up), interpolation=cv2.INTER_LINEAR)
    views = _Views(images, intrinsics, poses, device)
    tex = _colorize(views, ztex, float(xmin), float(ymax), cell / up, HeightfieldConfig())
    empty = far > 2
    if fill_radius_m > 0 and empty.any():
        lab, n_lab = ndimage.label(empty)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        sizes = ndimage.sum(empty, lab, index=np.arange(1, n_lab + 1)) * cell * cell
        closed = np.zeros(n_lab + 1, dtype=bool)
        closed[1:] = sizes <= np.pi * fill_radius_m**2
        closed[border] = False
        empty &= ~closed[lab]
    blank = cv2.resize(empty.astype(np.uint8), (nx * up, ny * up), interpolation=cv2.INTER_NEAREST).astype(bool)
    tex[blank] = 0
    return tex, float(xmin), float(ymax), (ny, nx)


def _fill_holes(V: np.ndarray, F: np.ndarray, radius_m: float) -> tuple[np.ndarray, int, float]:
    """Triangulate boundary loops whose bounding sphere is under ``radius_m``: (faces, faces added, area added m^2).

    Open3D's ``fill_holes`` (VTK's hole filler) adds faces between existing
    boundary vertices only, so the vertex arrays -- colours, tiers -- are
    unchanged and the new faces are simply appended.
    """
    import open3d as o3d

    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(np.ascontiguousarray(V, dtype=np.float32))
    m.triangle.indices = o3d.core.Tensor(np.ascontiguousarray(F, dtype=np.int32))
    out = m.fill_holes(hole_size=float(radius_m))
    if out.vertex.positions.shape[0] != len(V):
        raise RuntimeError("hole filling changed the vertices")
    Fn = out.triangle.indices.numpy().astype(np.int64)
    # Keep the original faces as they were; append only the new ones.
    if len(Fn) >= len(F) and np.array_equal(Fn[: len(F)], F):
        new = Fn[len(F):]
    else:
        def rows(f):
            return np.ascontiguousarray(np.sort(f, axis=1)).view(np.dtype((np.void, 24))).ravel()

        new = Fn[~np.isin(rows(Fn), rows(F))]
    if not len(new):
        return F, 0, 0.0
    a, b, c = V[new[:, 0]], V[new[:, 1]], V[new[:, 2]]
    area = float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    return np.concatenate([F, new]), len(new), area


def _choose_sources(centres: np.ndarray, axes: np.ndarray, r: int, depth: float, cfg: Mvs3dConfig) -> list[int]:
    base = np.linalg.norm(centres - centres[r], axis=1)
    ang = np.degrees(np.arccos(np.clip(axes @ axes[r], -1.0, 1.0)))
    ok = (np.arange(len(centres)) != r) & (base >= cfg.min_baseline_m) & (ang <= cfg.max_axis_angle_deg)
    cand = np.flatnonzero(ok)
    if not cand.size:
        return []
    # Prefer a baseline near a tenth of the scene depth: enough parallax, still overlapping.
    target = max(0.1 * depth, cfg.min_baseline_m)
    order = np.argsort(np.abs(np.log(base[cand] / target)))
    return [int(j) for j in cand[order[: cfg.sources]]]


def _choose_partners(centres: np.ndarray, axes: np.ndarray, r: int, refs: list[int], cfg: Mvs3dConfig) -> list[int]:
    """The nearest other references looking the same way: the depth maps ``r`` is checked against."""
    cand = np.array([j for j in refs if j != r], dtype=int)
    if not cand.size:
        return []
    ang = np.degrees(np.arccos(np.clip(axes[cand] @ axes[r], -1.0, 1.0)))
    base = np.linalg.norm(centres[cand] - centres[r], axis=1)
    keep = (ang <= cfg.max_axis_angle_deg) & (base >= cfg.min_baseline_m)
    cand, base = cand[keep], base[keep]
    return [int(j) for j in cand[np.argsort(base)[: cfg.partners]]]


def _depth_range(pts, Rr, Cr, Kr, H: int, W: int, cfg: Mvs3dConfig) -> tuple[float, float]:
    """Near/far camera depth for one view from the solved points it sees."""
    near, far = 0.5, 80.0
    if pts is not None and len(pts) >= 20:
        Xc = (pts - Cr) @ Rr
        z = Xc[:, 2]
        ok = z > 0.1
        uv = (Xc[ok, :2] / z[ok, None]) @ Kr[:2, :2].T + Kr[:2, 2]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        zz = z[ok][inside]
        if zz.size >= 10:
            lo, hi = (float(v) for v in np.percentile(zz, list(cfg.range_percentiles)))
            padding = max(2.0, cfg.range_pad * (hi - lo))
            # Hypotheses are uniform in inverse depth, so padding the near end
            # additively at close range would spend most of them in front of
            # the camera: never go below 0.8x the nearest solved point.
            near = max(0.3, lo - padding, 0.8 * lo)
            far = hi + padding
    return near, max(far, near + 1.0)


def _select_references(R, C, Ks, H: int, W: int, pts, cfg: Mvs3dConfig) -> list[int]:
    """Views that get a depth map: all, or on downward flights a subset covering each cell ``ref_coverage`` times."""
    import cv2

    n = len(R)
    everything = list(range(n))
    if cfg.ref_coverage <= 0 or n <= cfg.ref_coverage + 2 or pts is None or len(pts) < 20:
        return everything
    axes = R[:, :, 2]
    if np.median(-axes[:, 2]) < 0.8:  # median tilt beyond ~37 deg: ground footprints are not meaningful
        return everything
    z_ground = float(np.median(pts[:, 2]))
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    polys = []
    for r in range(n):
        rays = (np.c_[corners, np.ones(4)] @ np.linalg.inv(Ks[r]).T) @ R[r].T
        if np.any(rays[:, 2] > -1e-3) or C[r, 2] <= z_ground:
            polys.append(None)
            continue
        t = (z_ground - C[r, 2]) / rays[:, 2]
        polys.append(C[r, :2] + t[:, None] * rays[:, :2])
    good = [p for p in polys if p is not None]
    if len(good) < 0.5 * n:
        return everything
    allp = np.concatenate(good)
    lo = allp.min(0)
    span = float(np.median([np.ptp(p, axis=0).max() for p in good]))
    cell = max(span / 40.0, 1e-3)
    shape = (int(np.ceil((allp[:, 1].max() - lo[1]) / cell)) + 2, int(np.ceil((allp[:, 0].max() - lo[0]) / cell)) + 2)
    if shape[0] * shape[1] > 16_000_000:
        return everything
    count = np.zeros(shape, dtype=np.int32)
    refs = []
    for r in range(n):
        if polys[r] is None:
            refs.append(r)  # cannot judge its footprint: keep it
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round((polys[r] - lo) / cell).astype(np.int32)], 1)
        inside = mask.astype(bool)
        if not inside.any():
            continue
        if not refs or float(np.mean(count[inside] < cfg.ref_coverage)) >= cfg.ref_min_new:
            refs.append(r)
            count[inside] += 1
    return refs if len(refs) >= 3 else everything


def _sweep_view(r, src, G, Ks, R, C, near, far, pts, cfg: Mvs3dConfig, dev, progress=None, label=""):
    """Depth map (metres, 0 = rejected) for reference ``r`` against sources ``src``."""
    import torch
    import torch.nn.functional as F

    H, W = G[r].shape[-2:]
    win = cfg.window
    vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    Kinv = torch.tensor(np.linalg.inv(Ks[r]), device=dev, dtype=torch.float32)
    rays = Kinv @ torch.stack([uu, vv, torch.ones_like(uu)], 0).reshape(3, -1)  # camera rays at unit depth
    Cw = torch.tensor(C[r], device=dev, dtype=torch.float32)[:, None]
    dirs = torch.tensor(R[r], device=dev, dtype=torch.float32) @ rays  # world directions per unit camera depth
    ref = G[r]
    mu_r = _box(F, ref, win)
    var_r = (_box(F, ref * ref, win) - mu_r * mu_r).clamp(min=1e-6)
    n_hyp = int(cfg.hypotheses) or int(np.clip(
        np.ceil((1.0 / near - 1.0 / far) * np.sqrt(near * far) / cfg.hypothesis_step_rel), 48, 128))
    inv = torch.linspace(1.0 / far, 1.0 / near, n_hyp, device=dev)
    Rs = [torch.tensor(R[j].T, device=dev, dtype=torch.float32) for j in src]
    Cs = [torch.tensor(C[j], device=dev, dtype=torch.float32)[:, None] for j in src]
    Kj = [torch.tensor(Ks[j], device=dev, dtype=torch.float32) for j in src]
    B = max(1, int(cfg.batch))

    def score_depths(Dpix):
        """(b, N) per-pixel camera depths -> (b, H, W) top-k mean NCC (-1 where unusable)."""
        b = Dpix.shape[0]
        X = Cw[None] + dirs[None] * Dpix[:, None, :]  # (b, 3, N)
        per = []
        for Rj, Cj, Kjj, j in zip(Rs, Cs, Kj, src, strict=True):
            Xc = Rj[None] @ (X - Cj[None])
            z = Xc[:, 2].clamp(min=1e-6)
            uvj = (Kjj[None] @ Xc)[:, :2] / z[:, None]
            gx = uvj[:, 0] / (W - 1) * 2 - 1
            gy = uvj[:, 1] / (H - 1) * 2 - 1
            grid = torch.stack([gx, gy], -1).reshape(b, H, W, 2)
            samp = F.grid_sample(G[j].expand(b, -1, -1, -1), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            ok = (Xc[:, 2] > 0.05) & (gx.abs() <= 1) & (gy.abs() <= 1) & (Dpix > 0)
            valid = ok.reshape(b, 1, H, W).float()
            mu_s = _box(F, samp, win)
            cov = _box(F, samp * ref, win) - mu_s * mu_r
            var_s = (_box(F, samp * samp, win) - mu_s * mu_s).clamp(min=1e-6)
            ncc = cov / torch.sqrt(var_s * var_r)
            full = _box(F, valid, win) > 0.999
            per.append(torch.where(full, ncc, torch.full_like(ncc, -1.0))[:, 0])
        return _topk_mean(torch.stack(per), min(cfg.top_k, len(per)))

    scores = torch.empty((n_hyp, H, W), device=dev)
    Npx = dirs.shape[1]
    for h0 in range(0, n_hyp, B):
        if progress is not None:
            progress(f"{label}, hypotheses {h0}/{n_hyp}")
        hs = inv[h0 : h0 + B]
        scores[h0 : h0 + hs.shape[0]] = score_depths((1.0 / hs)[:, None].expand(-1, Npx))

    # Selection volume, higher is better: SGM-aggregated (negated) costs, or
    # the raw NCC. The photometric score kept per pixel is always the raw
    # NCC at the chosen hypothesis, so min_ncc means the same thing with and
    # without aggregation.
    sel = -_sgm_aggregate((1.0 - scores).clamp(0.0, 2.0), cfg.sgm_p1, cfg.sgm_p2, cfg.sgm_paths) \
        if cfg.sgm_paths else scores
    top, idx = sel.max(0)
    i0 = (idx - 1).clamp(min=0)
    i2 = (idx + 1).clamp(max=n_hyp - 1)
    s0 = sel.gather(0, i0[None])[0]
    s2 = sel.gather(0, i2[None])[0]
    den = s0 - 2 * top + s2
    interior = (idx > 0) & (idx < n_hyp - 1) & (den < 0)
    frac = torch.where(interior, 0.5 * (s0 - s2) / torch.where(interior, den, torch.ones_like(den)), torch.zeros_like(den))
    best = scores.gather(0, idx[None])[0]
    step = inv[1] - inv[0]
    inv_best = inv[idx] + frac.clamp(-0.5, 0.5) * step
    del sel
    # Local photometric refinement within +-1 hypothesis, when one step is
    # coarse against the consistency test (a wide inverse-depth range: close
    # forward views; 280 m aerial ranges are ~0.1% per step and skip it).
    # A peak on the refinement window's edge keeps the swept estimate.
    rel_step = float(step) / float(torch.median(inv_best))
    if cfg.refine_samples >= 3 and rel_step > 0.25 * cfg.consistency_rel:
        n_ref = int(cfg.refine_samples)
        offsets = torch.linspace(-1.0, 1.0, n_ref, device=dev) * step
        fine = torch.empty((n_ref, H, W), device=dev)
        for h0 in range(0, n_ref, B):
            cand = inv_best.reshape(1, -1) + offsets[h0 : h0 + B, None]
            fine[h0 : h0 + cand.shape[0]] = score_depths(1.0 / cand.clamp(min=1e-8))
        f_best, f_idx = fine.max(0)
        left = fine.gather(0, (f_idx - 1).clamp(min=0)[None])[0]
        right = fine.gather(0, (f_idx + 1).clamp(max=n_ref - 1)[None])[0]
        curv = left - 2 * f_best + right
        f_in = (f_idx > 0) & (f_idx < n_ref - 1) & (curv < 0) & (f_best >= best)
        delta = torch.where(f_in, 0.5 * (left - right) / torch.where(f_in, curv, torch.ones_like(curv)),
                            torch.zeros_like(curv)).clamp(-0.5, 0.5)
        inv_best = torch.where(f_in, inv_best + offsets[f_idx] + delta * (offsets[1] - offsets[0]), inv_best)
        best = torch.where(f_in, f_best, best)
        del fine
    depth_t = 1.0 / inv_best
    best_t = torch.where(interior, best, torch.full_like(best, -2.0))
    if cfg.ground_heights and pts is not None and len(pts) >= 20:
        # Ground heights: the lower part of the solved points (near this camera).
        dz = dirs[2]  # world z per unit camera depth
        near_pts = pts[np.linalg.norm(pts[:, :2] - C[r, :2], axis=1) < 3.0 * float(np.sqrt(near * far))]
        zs = near_pts[:, 2] if len(near_pts) >= 20 else pts[:, 2]
        lo, hi = float(np.percentile(zs, 1)) - 1.0, float(np.percentile(zs, 40)) + 0.5
        heights = torch.linspace(lo, hi, cfg.ground_heights, device=dev)
        g_scores = torch.empty((cfg.ground_heights, H, W), device=dev)
        below = dz < -1e-3
        for h0 in range(0, cfg.ground_heights, B):
            hh = heights[h0 : h0 + B]
            Dpix = (hh[:, None] - Cw[2]) / torch.where(below, dz, torch.full_like(dz, -1.0))[None]
            Dpix = torch.where(below[None] & (Dpix > 0.1) & (Dpix < far), Dpix, torch.zeros_like(Dpix))
            g_scores[h0 : h0 + hh.shape[0]] = score_depths(Dpix)
        g_best, g_idx = g_scores.max(0)
        g_in = (g_idx > 0) & (g_idx < cfg.ground_heights - 1)
        g0 = g_scores.gather(0, (g_idx - 1).clamp(min=0)[None])[0]
        g2 = g_scores.gather(0, (g_idx + 1).clamp(max=cfg.ground_heights - 1)[None])[0]
        gden = g0 - 2 * g_best + g2
        g_in = g_in & (gden < 0)
        gfrac = torch.where(g_in, 0.5 * (g0 - g2) / torch.where(g_in, gden, torch.ones_like(gden)), torch.zeros_like(gden))
        h_best = heights[g_idx] + gfrac.clamp(-0.5, 0.5) * (heights[1] - heights[0])
        dzr = dz.reshape(H, W)
        g_depth = (h_best - Cw[2, 0]) / torch.where(dzr < -1e-3, dzr, torch.full_like(dzr, -1.0))
        use_g = g_in & (g_best > best_t) & (dzr < -1e-3) & (g_depth > 0.1)
        depth_t = torch.where(use_g, g_depth, depth_t)
        best_t = torch.where(use_g, g_best, best_t)
        interior = interior | use_g
    depth = depth_t.cpu().numpy().astype(np.float32)
    index_map = ((1.0 / depth_t.clamp(min=1e-6) - inv[0]) / step).cpu().numpy()
    keep = ((best_t >= cfg.min_ncc) & interior).cpu().numpy()
    keep = _speckle_keep(index_map, keep, cfg.speckle_px, cfg.speckle_steps)
    depth[~keep] = 0.0
    return depth, n_hyp


#: Points the live preview keeps across all views.
_PREVIEW_POINTS = 1_500_000


def _emit_preview(partial, preview, depth, rgb, K, Rr, Cr, n_refs: int, last: float, every_s: float, final: bool):
    """Add one view's measured depth (strided) to the preview; hand the whole preview over when due."""
    h, w = depth.shape
    st = max(1, int(np.ceil(np.sqrt(h * w * n_refs / _PREVIEW_POINTS))))
    d = depth[::st, ::st]
    v, u = np.nonzero(d > 0)
    if v.size:
        z = d[v, u].astype(np.float64)
        X = np.linalg.inv(K) @ np.stack([u * st, v * st, np.ones_like(u)], 0).astype(np.float64) * z
        preview.append(((Rr @ X).T + Cr, rgb[::st, ::st][v, u]))
    now = time.monotonic()
    if preview and (final or now - last >= every_s):
        try:
            partial(PointCloud(
                xyz=np.concatenate([p for p, _ in preview]),
                rgb=np.concatenate([c for _, c in preview]),
                confidence=np.full(sum(len(p) for p, _ in preview), int(Confidence.LOW_CONFIDENCE), dtype=np.uint8),
            ))
        except Exception:
            logger.debug("mvs3d: preview callback raised; ignoring", exc_info=True)
        return now
    return last


def _consistent(r, partners, maps_t, Ks, R, C, cfg: Mvs3dConfig, dev):
    """``maps_t[r]`` with unconfirmed samples zeroed (numpy), and the kept samples' world points.

    A sample is kept when ``min(min_consistent, n)`` partners agree with it,
    ``n`` being how many partners have a depth where it reprojects (at least
    one must): the edge of the survey, seen by one other reference, is
    checked against that one; the middle, seen by several, needs two.
    On the compute device -- the NumPy version was 0.5 s per reference.
    """
    import torch

    D = maps_t[r]
    H, W = D.shape
    v, u = torch.nonzero(D > 0, as_tuple=True)
    if not v.numel():
        return np.zeros((H, W), np.float32), np.zeros((0, 3))
    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=dev)
    Kr, Rr, Cr = f32(Ks[r]), f32(R[r]), f32(C[r])[:, None]
    d = D[v, u]
    pix = torch.stack([u.float(), v.float(), torch.ones_like(d)])
    Xw = Rr @ ((torch.linalg.inv(Kr) @ pix) * d) + Cr  # (3, N)
    agree = torch.zeros_like(d, dtype=torch.int32)
    observed = torch.zeros_like(agree)
    for j in partners:
        Kj, Rj, Cj = f32(Ks[j]), f32(R[j]), f32(C[j])[:, None]
        Xj = Rj.T @ (Xw - Cj)
        zj = Xj[2]
        uvj = Kj[:2, :2] @ (Xj[:2] / zj.clamp(min=1e-6)) + Kj[:2, 2:3]
        ui, vi = torch.round(uvj[0]).long(), torch.round(uvj[1]).long()
        inside = (zj > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        dj = torch.zeros_like(d)
        dj[inside] = maps_t[j][vi[inside], ui[inside]]
        seen = inside & (dj > 0)
        # Round trip: the partner's own sample, back into the reference image.
        back = Rj @ ((torch.linalg.inv(Kj) @ torch.stack([ui.float(), vi.float(), torch.ones_like(d)])) * dj) + Cj
        bc = Rr.T @ (back - Cr)
        buv = Kr[:2, :2] @ (bc[:2] / bc[2].clamp(min=1e-8)) + Kr[:2, 2:3]
        reprojection = torch.linalg.norm(buv - pix[:2], dim=0)
        agree += (seen & (bc[2] > 0) & (reprojection <= cfg.consistency_px)
                  & ((dj - zj).abs() <= cfg.consistency_rel * zj)).int()
        observed += seen.int()
    need = observed.clamp(min=1, max=max(1, int(cfg.min_consistent)))
    good = agree >= need
    out = torch.zeros_like(D)
    out[v[good], u[good]] = d[good]
    return out.cpu().numpy(), Xw[:, good].T.cpu().numpy().astype(np.float64)


def reconstruct_mvs3d(
    images: list[np.ndarray],
    intrinsics: list[CameraIntrinsics],
    poses: list[Pose],
    *,
    points: np.ndarray | None = None,
    config: Mvs3dConfig | None = None,
    device: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    partial: Callable[[PointCloud], None] | None = None,
    partial_every_s: float = 1.0,
    exclude_masks: list[np.ndarray | None] | None = None,
) -> tuple[Mvs3dSurface, dict]:
    """Mesh the scene the calibrated ``images`` (BGR uint8, pixels matching ``intrinsics``) see from ``poses``.

    ``partial`` (optional) receives the depth measured so far -- a strided,
    LOW-tier coloured point cloud, before the cross-view check -- at most
    every ``partial_every_s`` seconds, for a live view.

    ``exclude_masks`` (optional, one per image, True = exclude) are pixels
    that must never become geometry -- vehicles, people and sky from the
    semantics stage: they get no depth, so a moving car cannot leave a
    ghost in the mesh or pull the fused surface.
    """
    import cv2
    import torch

    from drishti3d.device import get_device

    cfg = config or Mvs3dConfig()
    n = len(images)
    if n < 3:
        raise ValueError(f"mvs3d needs at least 3 posed views, got {n}")
    dev = torch.device(device or get_device())
    t0 = time.monotonic()

    # Working resolution, undistorted, grey for matching and RGB for colour.
    grey, rgb, Ks, excl = [], [], [], []
    masks_in = list(exclude_masks) if exclude_masks is not None else [None] * n
    if len(masks_in) != n:
        raise ValueError(f"exclude_masks has {len(masks_in)} entries for {n} images")
    for img, K, mask in zip(images, intrinsics, masks_in, strict=True):
        h, w = img.shape[:2]
        s = min(1.0, cfg.max_side / float(max(h, w)))
        Km = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1.0]])
        if K.dist_coeffs is not None and np.any(np.asarray(K.dist_coeffs)):
            img = cv2.undistort(img, Km, np.asarray(K.dist_coeffs, dtype=np.float64))
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        rgb.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB) if small.ndim == 3 else cv2.cvtColor(small, cv2.COLOR_GRAY2RGB))
        grey.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if small.ndim == 3
                    else small.astype(np.float32) / 255.0)
        Ks.append(np.array([[K.fx * s, 0, K.cx * s], [0, K.fy * s, K.cy * s], [0, 0, 1.0]]))
        excl.append(None if mask is None else cv2.resize(
            np.asarray(mask, dtype=np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool))
    H, W = grey[0].shape
    R = np.stack([np.asarray(p.R, dtype=np.float64) for p in poses])  # world_from_cam
    C = np.stack([np.asarray(p.t, dtype=np.float64).reshape(3) for p in poses])
    axes = R[:, :, 2]
    pts = None if points is None else np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = None if pts is None else pts[np.isfinite(pts).all(axis=1)]
    ranges = [_depth_range(pts, R[r], C[r], Ks[r], H, W, cfg) for r in range(n)]
    refs = _select_references(R, C, Ks, H, W, pts, cfg)

    # 1) Depth per reference view.
    G = [torch.from_numpy(g).to(dev)[None, None] for g in grey]
    depth_maps: dict[int, np.ndarray] = {}
    preview: list[tuple[np.ndarray, np.ndarray]] = []
    last_emit = 0.0
    hyp_counts: list[int] = []
    masked_px = 0
    for k, r in enumerate(refs):
        label = f"measured depth {k + 1}/{len(refs)}"
        if progress is not None:
            progress(k, len(refs), label)
        near, far = ranges[r]
        src = _choose_sources(C, axes, r, float(np.sqrt(near * far)), cfg)
        if not src:
            depth_maps[r] = np.zeros((H, W), np.float32)
            continue
        depth_maps[r], n_hyp = _sweep_view(
            r, src, G, Ks, R, C, near, far, pts, cfg, dev,
            progress=None if progress is None else (lambda msg, k=k: progress(k, len(refs), msg)), label=label,
        )
        hyp_counts.append(n_hyp)
        if excl[r] is not None:
            masked_px += int((depth_maps[r] > 0)[excl[r]].sum())
            depth_maps[r][excl[r]] = 0.0
        if partial is not None:
            last_emit = _emit_preview(partial, preview, depth_maps[r], rgb[r], Ks[r], R[r], C[r],
                                      len(refs), last_emit, partial_every_s, final=k == len(refs) - 1)
    del G
    t1 = time.monotonic()

    # 2) Cross-view consistency, against the untouched maps of neighbouring references.
    maps_t = {r: torch.from_numpy(depth_maps[r]).to(dev) for r in refs}
    filtered: list[np.ndarray] = []
    consistent_pts = []
    kept_px = total_px = 0
    for k, r in enumerate(refs):
        if progress is not None:
            progress(k, len(refs), f"cross-view consistency {k + 1}/{len(refs)}")
        total_px += int((depth_maps[r] > 0).sum())
        good, Xw = _consistent(r, _choose_partners(C, axes, r, refs, cfg), maps_t, Ks, R, C, cfg, dev)
        kept_px += len(Xw)
        filtered.append(good)
        consistent_pts.append(Xw)
    del maps_t
    cons = np.concatenate(consistent_pts) if consistent_pts else np.zeros((0, 3))
    t2 = time.monotonic()
    if len(cons) < 1000:
        raise RuntimeError(f"mvs3d: only {len(cons)} depth samples survived the cross-view check")

    # 3) TSDF fusion of the measured depth maps.
    med_depth = float(np.median(np.concatenate([d[d > 0] for d in filtered if (d > 0).any()])))
    fx_w = float(np.median([Ks[r][0, 0] for r in refs]))
    voxel = float(cfg.voxel_m) if cfg.voxel_m else float(np.clip(cfg.voxel_footprints * med_depth / fx_w, 0.02, 0.5))
    far_max = max(ranges[r][1] for r in refs)
    mesh = _tsdf_mesh(filtered, [rgb[r] for r in refs], [Ks[r] for r in refs], R[refs], C[refs], voxel,
                      cfg.trunc_voxels, far_max, progress)
    if len(mesh.triangles) and cfg.min_fragment_triangles:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters, counts = np.asarray(clusters), np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[clusters] < cfg.min_fragment_triangles)
        mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices, dtype=np.float64)
    Fc = np.asarray(mesh.triangles, dtype=np.int64)
    if len(V) == 0 or len(Fc) == 0:
        raise RuntimeError("mvs3d: no connected surface survived volumetric fusion; more overlapping views are needed")
    col = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None
    downward = bool(np.median(-axes[:, 2]) >= 0.8)
    filled_faces, filled_m2 = 0, 0.0
    if downward and cfg.fill_holes_m > 0:
        try:
            Fc, filled_faces, filled_m2 = _fill_holes(V, Fc, cfg.fill_holes_m)
        except Exception:
            logger.warning("mvs3d: hole filling failed; the mesh keeps its gaps", exc_info=True)
    tier = np.full(len(V), int(Confidence.LOW_CONFIDENCE), dtype=np.uint8)
    from scipy.spatial import cKDTree

    sample = cons if len(cons) <= 3_000_000 else cons[:: int(np.ceil(len(cons) / 3_000_000))]
    dist, _ = cKDTree(sample).query(V, distance_upper_bound=1.5 * voxel)
    tier[np.isfinite(dist)] = int(Confidence.MEASURED)
    surface = Mvs3dSurface(vertices=V, faces=Fc, colors=col, confidence=tier, cell_m=voxel)
    t_tier = t4 = time.monotonic()
    if cfg.texture_upsample and downward:
        if progress is not None:
            progress(len(refs), len(refs), "true-ortho texture")
        try:
            tex, xmin, ymax, grid = _ortho_texture(
                V, images, intrinsics, poses, voxel, dev, cfg.texture_upsample, cfg.texture_max_px,
                fill_radius_m=cfg.fill_holes_m,
            )
            surface.texture_rgb, surface.tex_xmin, surface.tex_ymax = tex, xmin, ymax
            surface.tex_cell, surface.tex_grid = voxel, grid
        except Exception:
            logger.warning("mvs3d: true-ortho texture failed; the mesh keeps its vertex colours", exc_info=True)
        t4 = time.monotonic()
    diag = {
        "method": "mvs3d",
        "views": n,
        "references": len(refs),
        "working_px": [int(H), int(W)],
        "hypotheses": int(np.median(hyp_counts)) if hyp_counts else cfg.hypotheses,
        "sgm_paths": cfg.sgm_paths,
        "depth_px_measured": total_px,
        "depth_px_consistent": kept_px,
        "depth_px_masked": masked_px,
        "consistent_fraction": round(kept_px / max(total_px, 1), 4),
        "voxel_m": round(voxel, 4),
        "vertices": len(V),
        "faces": len(Fc),
        "measured_vertex_fraction": round(float(np.mean(tier == int(Confidence.MEASURED))), 4),
        "holes_filled_faces": int(filled_faces),
        "holes_filled_m2": round(filled_m2, 1),
        "texture_px": None if surface.texture_rgb is None else list(surface.texture_rgb.shape[:2]),
        "seconds": {"sweep": round(t1 - t0, 2), "consistency": round(t2 - t1, 2),
                    "fusion": round(t_tier - t2, 2), "texture": round(t4 - t_tier, 2)},
        "device": str(dev),
    }
    logger.info(
        "mvs3d: %d views (%d references), %d/%d depth samples consistent (%.0f%%), %.3f m voxels -> "
        "%d vertices / %d faces (%.0f%% MEASURED) in %.1f s",
        n, len(refs), kept_px, total_px, 100 * diag["consistent_fraction"], voxel, len(V), len(Fc),
        100 * diag["measured_vertex_fraction"], t4 - t0,
    )
    return surface, diag
