"""Height-field multi-view stereo for nadir flights: the DSM straight from the calibrated keyframes.

Why
---
The learned backbone regresses one depth map per view, and every view's map
is then fitted to the bundle-adjusted points with metres of residual error.
Fusing those disagreeing surfaces is what produced "thick", stacked ground
(16 m all-cell thickness on PinPoint flight01 with correct cameras). A
straight-down survey is a 2.5-D surface -- one height per ground position --
and once bundle adjustment has solved the cameras and the lens, that height
can be MEASURED rather than regressed: it is the height at which every view
that sees the spot agrees on what it looks like.

How
---
For a grid of ground cells and a set of candidate heights (a plane sweep in
world Z), every candidate point is projected into the views that see it --
with the lens's OpenCV distortion, so raw frames need no undistortion -- and
sampled. Photo-consistency is the mean normalised cross-correlation of each
view's local window against the cross-view mean (box filters, so a cell's
score uses its neighbourhood's texture). The best-scoring height wins, with
a parabolic sub-step refinement.

Coarse to fine: a wide sweep around a prior surface on a coarse grid (the
bundle-adjusted points rasterised, or the telemetry height above ground when
there are none), then a narrow sweep on the fine grid around the filtered
coarse surface. Colour is the per-cell median across the nearest views at
the chosen height: a true orthophoto, not a mosaic of perspective frames.

What it will not do: represent overhangs (a height field cannot), or invent
height where the views do not agree -- such cells are left empty, and
cells supported by few views or weak agreement are tiered LOW_CONFIDENCE /
INFERRED exactly like the rest of the pipeline's trust layer.

Measured on PinPoint flight01 (208 bundle-adjusted keyframes, 1280x720):
DSM vs the reference DEM MAD 1.5 m against 7.5 m for backbone depth fusion
on the same cameras, in 42 s on Apple MPS instead of ~20 minutes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from drishti3d.types import CameraIntrinsics, Confidence, PointCloud, Pose

logger = logging.getLogger(__name__)

__all__ = ["HeightfieldConfig", "HeightfieldSurface", "reconstruct_heightfield"]


@dataclass
class HeightfieldConfig:
    """Tuning for ``reconstruct_heightfield``. Defaults are the values measured on flight01."""

    coarse_cell_m: float = 2.0
    fine_cell_m: float = 0.5
    #: Coarse sweep around the prior surface: further up than down, because
    #: roofs and canopy rise above the sparse-point ground while the prior
    #: rarely sits far above the true surface.
    coarse_below_m: float = 25.0
    coarse_above_m: float = 35.0
    coarse_step_m: float = 0.5
    fine_range_m: float = 3.0
    fine_step_m: float = 0.1
    coarse_window: int = 5
    fine_window: int = 7
    views_per_tile: int = 10
    min_views: int = 3
    #: When > 0, per cell only the ``score_top_k`` best-agreeing views count,
    #: so an occluded or grazing view cannot drag a well-seen cell down. Off
    #: by default: on flight01 it doubled the fine level's time without
    #: moving the DSM-vs-DEM spread (1.74 vs 1.73 m).
    score_top_k: int = 0
    min_score: float = 0.35
    measured_score: float = 0.6
    #: Height uncertainty: the farthest hypothesis whose score is within this
    #: many standard deviations of the best one's (sampling noise of a
    #: correlation over the window and views, see ``_sweep_tile``).
    ambiguity_sigmas: float = 1.0
    #: A cell more than this far from its 5x5 median is a spike, not relief.
    spike_m: float = 2.0
    #: Enclosed holes of at most this many cells (rejected pinholes, removed
    #: spikes) are filled from their neighbours and tiered INFERRED, so the
    #: mesh is not a sieve; larger gaps stay open. 400 cells = 100 m^2 at 0.5 m
    #: (a low-texture patch of field): filled, but labelled as not measured.
    fill_max_cells: int = 400
    #: Complete the visible scene: every cell some camera saw but stereo did
    #: not measure (bare fields, water, shadow, ground occluded behind a
    #: building, rejected spikes) is filled from its surroundings and tiered
    #: INFERRED with no measured uncertainty. Cells no camera saw stay empty.
    fill_visible: bool = True
    #: A hole whose rim is mostly this far above the local ground (roof,
    #: canopy) is filled from the rim; any other hole from the ground only,
    #: so a roof beside a bare field does not ramp down into it.
    fill_elevated_m: float = 2.5
    #: The fill is the coarse-level surface plus the rim's difference from
    #: it, fading with distance from the nearest measured cell at this rate
    #: (metres): gaps next to measurements follow them, large gaps and the
    #: survey's edge follow the robust 2 m surface instead of extrapolating
    #: whatever the outermost (fewest-view) cells happened to measure.
    fill_decay_m: float = 10.0
    #: Also fill holes the model encloses completely, whatever their size:
    #: ground no posed camera saw (flight01: triangles between strips seen
    #: only from the banked turns whose cameras could not be posed). INFERRED,
    #: coloured by inpainting from the surroundings.
    fill_enclosed_unseen: bool = True
    #: Mesh: a step between neighbouring cells above this is a cliff or a
    #: roof edge, closed with an INFERRED wall (see ``HeightfieldSurface``).
    max_step_m: float = 3.0
    #: No structure stands this far above its 30 m surroundings' ground: a
    #: cell that does is a match between wrong hypotheses, removed (and, if
    #: seen, re-filled from the ground).
    max_structure_m: float = 60.0
    tile_m: float = 100.0
    coarse_image_scale: float = 0.5
    color_views: int = 8
    #: Texture texels per mesh cell along each axis: the photographic
    #: texture is sampled this much finer than the geometry (0.5 m cells ->
    #: ~0.17 m texels, about the source GSD at 110 m). 0 disables it.
    texture_upsample: int = 3
    texture_max_px: int = 8192


@dataclass
class HeightfieldSurface:
    """The reconstructed DSM grid. Row 0 is north (``ymax``), column 0 is west (``xmin``).

    Duck-types the ``fusion.heightmap.HeightmapFusion`` read-out interface
    (``cell_m``, ``mesh()``) so ``FusionStage`` treats it like the median
    height map it replaces.
    """

    z: np.ndarray  # (ny, nx) float32 world Z, NaN where not reconstructed
    score: np.ndarray  # (ny, nx) float32 mean NCC at the chosen height
    views: np.ndarray  # (ny, nx) float32 views agreeing over the window
    rgb: np.ndarray  # (ny, nx, 3) uint8 true-ortho colour
    xmin: float
    ymax: float
    cell_m: float
    measured_score: float = 0.6
    max_step_m: float = 3.0
    #: (ty, tx, 3) uint8 true-ortho texture over the whole grid extent, or None.
    texture_rgb: np.ndarray | None = None
    #: (ny, nx) bool: cells filled from their neighbours rather than measured.
    inferred: np.ndarray | None = None
    #: Every cell was chosen by multi-view photo-consistency, so FusionStage
    #: does not run its post-hoc photometric rejection over this surface.
    photo_consistent: bool = True
    #: (ny, nx) float32 height uncertainty in metres (the sweep's ambiguity
    #: half-width, see ``_sweep_tile``); NaN where nothing was measured.
    sigma_m: np.ndarray | None = None
    #: Close height steps above ``max_step_m`` with walls. A 2.5D surface
    #: left open there had gaps an oblique ray passed through (flight01: 6
    #: of 9 surveyed points that missed the mesh), and building sides were
    #: missing. Wall faces use duplicated vertices tiered INFERRED, so the
    #: roof and ground keep their own measured tiers.
    close_walls: bool = True

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.z)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.z.shape
        x = self.xmin + (np.arange(nx) + 0.5) * self.cell_m
        y = self.ymax - (np.arange(ny) + 0.5) * self.cell_m
        return np.meshgrid(x, y)

    def confidence(self) -> np.ndarray:
        """Per-cell ``Confidence`` tier from agreement strength and view count."""
        tier = np.full(self.z.shape, int(Confidence.INFERRED), dtype=np.uint8)
        low = self.valid & (self.views >= 2)
        measured = self.valid & (self.score >= self.measured_score) & (self.views >= 4)
        tier[low] = int(Confidence.LOW_CONFIDENCE)
        tier[measured] = int(Confidence.MEASURED)
        if self.inferred is not None:
            tier[self.inferred] = int(Confidence.INFERRED)
        return tier

    def point_cloud(self) -> PointCloud:
        X, Y = self.cell_centres()
        v = self.valid
        sigma = None
        if self.sigma_m is not None:
            sigma = self.sigma_m[v].astype(np.float32)
            if self.inferred is not None:
                sigma[self.inferred[v]] = np.nan  # filled, not measured
        return PointCloud(
            xyz=np.stack([X[v], Y[v], self.z[v]], axis=-1).astype(np.float64),
            rgb=self.rgb[v].astype(np.uint8),
            confidence=self.confidence()[v],
            uncertainty_m=sigma,
        )

    def vertex_uv(self) -> np.ndarray:
        """Planar UVs for ``point_cloud()``/``mesh()`` vertices into ``texture_rgb`` (OBJ/glTF: v up)."""
        ny, nx = self.z.shape
        rows, cols = np.nonzero(self.valid)
        return np.stack([(cols + 0.5) / nx, 1.0 - (rows + 0.5) / ny], axis=-1)

    def _triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """Candidate faces (two per 2x2 block of reconstructed cells) and whether each is flat enough."""
        v = self.valid
        idx = np.full(self.z.shape, -1, dtype=np.int64)
        idx[v] = np.arange(int(v.sum()))
        a, b = idx[:-1, :-1], idx[:-1, 1:]
        c, d = idx[1:, :-1], idx[1:, 1:]
        za, zb, zc, zd = self.z[:-1, :-1], self.z[:-1, 1:], self.z[1:, :-1], self.z[1:, 1:]
        faces, spans = [], []
        # Counter-clockwise seen from above (+Z up): north-west, south-west, north-east.
        for tri, zs in (((a, c, b), (za, zc, zb)), ((b, c, d), (zb, zc, zd))):
            f = np.stack([t.reshape(-1) for t in tri], axis=-1)
            h = np.stack([z.reshape(-1) for z in zs])
            faces.append(f)
            spans.append(h.max(axis=0) - h.min(axis=0))
        faces = np.concatenate(faces)
        spans = np.concatenate(spans)
        complete = (faces >= 0).all(axis=1)
        with np.errstate(invalid="ignore"):
            flat = spans <= self.max_step_m
        return faces[complete], flat[complete]

    def mesh(self) -> tuple[PointCloud, np.ndarray]:
        """The grid as a triangle mesh; steps above ``max_step_m`` become INFERRED walls when ``close_walls``."""
        pc = self.point_cloud()
        faces, flat = self._triangles()
        self._wall_src = np.zeros(0, dtype=np.int64)
        if not self.close_walls or flat.all():
            return pc, faces[flat]
        wall = faces[~flat]
        src = np.unique(wall)
        n = pc.xyz.shape[0]
        dup = np.full(n, -1, dtype=np.int64)
        dup[src] = n + np.arange(src.size)
        self._wall_src = src
        tier = np.full(src.size, int(Confidence.INFERRED), dtype=np.uint8)
        ext = PointCloud(
            xyz=np.concatenate([pc.xyz, pc.xyz[src]]),
            rgb=None if pc.rgb is None else np.concatenate([pc.rgb, pc.rgb[src]]),
            confidence=None if pc.confidence is None else np.concatenate([pc.confidence, tier]),
            uncertainty_m=None if pc.uncertainty_m is None
            else np.concatenate([pc.uncertainty_m, np.full(src.size, np.nan, dtype=np.float32)]),
        )
        return ext, np.concatenate([faces[flat], dup[wall]])

    def mesh_vertex_uv(self) -> np.ndarray:
        """``vertex_uv`` for ``mesh()``'s vertices (walls' duplicated vertices included)."""
        uv = self.vertex_uv()
        src = getattr(self, "_wall_src", None)
        if src is None:
            self.mesh()
            src = self._wall_src
        return uv if not src.size else np.concatenate([uv, uv[src]])


def _torch():
    import torch
    import torch.nn.functional as F

    return torch, F


class _Views:
    """All keyframes on the compute device, with per-view pinhole + OpenCV distortion parameters."""

    def __init__(self, images: list[np.ndarray], intrinsics: list[CameraIntrinsics], poses: list[Pose], device):
        torch, F = _torch()
        self.torch, self.F, self.device = torch, F, device
        h, w = images[0].shape[:2]
        gray, rgb = [], []
        scales = []
        for im, intr in zip(images, intrinsics, strict=True):
            if im.shape[:2] != (h, w):
                import cv2

                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
            scales.append(w / float(intr.width))
            if im.ndim == 2:
                g = im
                c = np.repeat(im[..., None], 3, axis=-1)
            else:
                import cv2

                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                c = im[..., ::-1]  # BGR -> RGB
            gray.append(g)
            rgb.append(c)
        self.h, self.w = h, w
        self.gray = torch.from_numpy(np.stack(gray)).to(device=device, dtype=torch.float16)[:, None] / 255.0
        self.gray_small = None
        self.rgb = np.stack(rgb)  # host memory; uploaded per tile
        dist = np.zeros((len(intrinsics), 5), dtype=np.float64)
        for i, intr in enumerate(intrinsics):
            if intr.dist_coeffs is not None:
                d = np.asarray(intr.dist_coeffs, dtype=np.float64).reshape(-1)[:5]
                dist[i, : d.size] = d
        s = np.asarray(scales)
        f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=device)
        self.fx = f32(np.array([i.fx for i in intrinsics]) * s)
        self.fy = f32(np.array([i.fy for i in intrinsics]) * s)
        self.cx = f32(np.array([i.cx for i in intrinsics]) * s)
        self.cy = f32(np.array([i.cy for i in intrinsics]) * s)
        self.k1, self.k2, self.p1, self.p2, self.k3 = (f32(dist[:, j]) for j in range(5))
        self.R = f32(np.stack([p.R for p in poses]))  # world_from_cam
        self.C = f32(np.stack([p.t for p in poses]))

    def small(self, scale: float):
        if self.gray_small is None:
            self.gray_small = self.F.interpolate(self.gray.float(), scale_factor=scale, mode="area").half()
        return self.gray_small

    def project(self, Xw, sel):
        """``Xw`` (..., 3) world points -> pixel coords (K, ..., 2) and validity (K, ...) in views ``sel``."""
        torch = self.torch
        shp = Xw.shape[:-1]
        X = Xw.reshape(-1, 3)
        R, C = self.R[sel], self.C[sel]
        Xc = torch.einsum("kji,knj->kni", R, X[None] - C[:, None, :])  # R^T (X - C)
        z = Xc[..., 2]
        x = Xc[..., 0] / z.clamp(min=1e-3)
        y = Xc[..., 1] / z.clamp(min=1e-3)
        k1, k2, p1, p2, k3 = (p[sel][:, None] for p in (self.k1, self.k2, self.p1, self.p2, self.k3))
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        u = self.fx[sel][:, None] * xd + self.cx[sel][:, None]
        v = self.fy[sel][:, None] * yd + self.cy[sel][:, None]
        # r2 bound: past the lens's calibrated field the polynomial folds back.
        valid = (z > 1.0) & (u >= 1) & (u <= self.w - 2) & (v >= 1) & (v <= self.h - 2) & (r2 < 1.2)
        k = len(sel)
        return torch.stack([u, v], -1).reshape(k, *shp, 2), valid.reshape(k, *shp)

    def views_covering(self, sample_xyz: np.ndarray, limit: int):
        """Up to ``limit`` views covering the most of ``sample_xyz`` (M, 3), most nadir first on ties.

        Ranking by coverage of the whole tile, not by whether one centre
        point is visible: a tile at the survey's edge is still mostly seen
        by the views whose footprints reach into it.
        """
        torch = self.torch
        p = torch.tensor(np.asarray(sample_xyz, dtype=np.float32).reshape(-1, 3), device=self.device)
        allv = torch.arange(self.R.shape[0], device=self.device)
        _, ok = self.project(p, allv)
        coverage = ok.float().sum(1)
        cand = torch.nonzero(coverage > 0).flatten()
        if cand.numel() == 0:
            return cand
        d = torch.linalg.norm(self.C[cand, :2] - p[:, :2].mean(0), dim=1)
        # Coverage first; distance (most nadir) breaks ties.
        key = coverage[cand] * 1e6 - d
        return cand[torch.argsort(key, descending=True)][:limit]


def _tile_samples(x0: float, x1: float, y0: float, y1: float, z: float, n: int = 5) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=-1)


def _box(F, x, k):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


def _sweep_tile(
    views: _Views, images, gx, gy, zc, deltas, sel, window: int, min_views: int, top_k: int = 0, ambiguity: float = 1.0
):
    """Best height, score, view count and height uncertainty (m) for one tile over ``zc + deltas``."""
    torch, F = views.torch, views.F
    ny, nx = gx.shape
    imgs = images[sel].float()
    best = torch.full((ny, nx), -2.0, device=views.device)
    best_z = torch.zeros((ny, nx), device=views.device)
    best_k = torch.zeros((ny, nx), device=views.device)
    scores = []
    for d in deltas:
        Z = zc + float(d)
        uv, valid = views.project(torch.stack([gx, gy, Z], -1), sel)
        grid = torch.stack([uv[..., 0] / (views.w - 1) * 2 - 1, uv[..., 1] / (views.h - 1) * 2 - 1], -1)
        samp = F.grid_sample(imgs, grid, mode="bilinear", align_corners=True)  # (K,1,ny,nx)
        vm = valid.float()[:, None]
        mean = (samp * vm).sum(0, keepdim=True) / vm.sum(0, keepdim=True).clamp(min=1)
        m = mean.expand_as(samp)
        mu_s, mu_m = _box(F, samp, window), _box(F, m, window)
        cov = _box(F, samp * m, window) - mu_s * mu_m
        var_s = (_box(F, samp * samp, window) - mu_s * mu_s).clamp(min=1e-6)
        var_m = (_box(F, m * m, window) - mu_m * mu_m).clamp(min=1e-6)
        ncc = (cov / torch.sqrt(var_s * var_m))[:, 0]
        full = _box(F, vm, window)[:, 0] > 0.999  # view valid over the whole window
        k = full.float().sum(0)
        if top_k and top_k < ncc.shape[0]:
            masked = torch.where(full, ncc, torch.full_like(ncc, -4.0))
            top = torch.topk(masked, top_k, dim=0).values
            n_top = torch.clamp(k, max=float(top_k))
            top_valid = top > -3.0
            score_all = (top * top_valid).sum(0) / n_top.clamp(min=1)
        else:
            score_all = (ncc * full).sum(0) / k.clamp(min=1)
        score = torch.where(k >= min_views, score_all, torch.full_like(k, -2.0))
        scores.append(score)
        better = score > best
        best_z = torch.where(better, Z, best_z)
        best_k = torch.where(better, k, best_k)
        best = torch.where(better, score, best)
    stack = torch.stack(scores)
    idx = stack.argmax(0)
    # A best height on the first or last hypothesis is not a measurement:
    # the surface lies outside the swept range (or there is no texture).
    at_edge = (idx == 0) | (idx == len(deltas) - 1)
    best = torch.where(at_edge, torch.full_like(best, -2.0), best)
    i0 = (idx - 1).clamp(min=0)
    i2 = (idx + 1).clamp(max=len(deltas) - 1)
    s0 = stack.gather(0, i0[None])[0]
    s1 = stack.gather(0, idx[None])[0]
    s2 = stack.gather(0, i2[None])[0]
    den = s0 - 2 * s1 + s2
    interior = (idx > 0) & (idx < len(deltas) - 1) & (den < 0)
    frac = torch.where(interior, 0.5 * (s0 - s2) / torch.where(interior, den, torch.ones_like(den)), torch.zeros_like(den))
    step = float(deltas[1] - deltas[0]) if len(deltas) > 1 else 0.0
    frac = frac.clamp(-0.5, 0.5)
    best_z = best_z + frac * step
    # Height uncertainty: the farthest hypothesis from the refined height
    # whose score the best one does not beat by more than the scores' own
    # noise. A sample correlation over n pixels has standard deviation
    # ~(1 - r^2) / sqrt(n - 1) (Fisher); the score averages it over the
    # agreeing views. A broad peak (weak parallax, little texture) and a
    # second peak (repetitive texture) both widen it; the sweep's own
    # resolution (half a step) is the floor.
    d_t = torch.as_tensor(np.asarray(deltas, dtype=np.float32), device=views.device)
    z_hat = d_t[idx] + frac * step
    r = s1.clamp(-1.0, 1.0)
    noise = (1.0 - r * r) / torch.sqrt(float(window * window - 1) * best_k.clamp(min=1.0))
    near = stack >= (s1 - ambiguity * noise)[None]
    spread = torch.where(near, (d_t[:, None, None] - z_hat[None]).abs(), torch.zeros_like(stack)).amax(0)
    return best_z, best, best_k, spread.clamp(min=0.5 * step)


def _fill_small_holes(z: np.ndarray, max_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill enclosed NaN holes of at most ``max_cells`` from their neighbours; return (z, filled mask).

    A hole touching the grid border is outside the survey, never filled.
    Filling is normalised convolution (the mean of the valid 3x3
    neighbours), repeated inward until the hole closes.
    """
    from scipy import ndimage

    holes = ~np.isfinite(z)
    filled = np.zeros(z.shape, dtype=bool)
    if max_cells <= 0 or not holes.any():
        return z, filled
    lab, n = ndimage.label(holes)
    if n == 0:
        return z, filled
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small = sizes <= max_cells
    small[0] = False
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small[border] = False
    target = small[lab]
    out = z.copy()
    remaining = target.copy()
    for _ in range(int(np.ceil(np.sqrt(max_cells))) + 2):
        if not remaining.any():
            break
        known = np.isfinite(out)
        num = ndimage.uniform_filter(np.where(known, out, 0.0), size=3, mode="constant")
        den = ndimage.uniform_filter(known.astype(np.float64), size=3, mode="constant")
        # At least one real neighbour means den >= 1/9. uniform_filter's
        # running sums leave ~1e-17 residue where there is none, and
        # "den > 0" then divided noise by noise: DJI_1001 got 2663 cells at
        # exactly z = 0 (the camera plane) and flight01 540 floating ones.
        can = remaining & (den > 0.5 / 9.0)
        out[can] = (num[can] / den[can]).astype(out.dtype)
        remaining &= ~can
    filled = target & np.isfinite(out)
    return out, filled


def _smooth_fill(z: np.ndarray, smooth_iterations: int = 4) -> np.ndarray:
    """Every NaN cell from the finite ones: a coarse-to-fine pyramid of NaN-aware means, then diffusion.

    Finite cells are returned unchanged. Each level halves the grid with
    the mean of the finite cells in each 2x2 block until nothing is
    missing; going back up, a missing cell takes the bilinear value of the
    level above. A few Gaussian passes restricted to the filled cells then
    remove the block pattern (a membrane between the rims).
    """
    import cv2
    from scipy import ndimage

    known = np.isfinite(z)
    if known.all() or not known.any():
        return z.copy() if known.any() else np.zeros_like(z)
    levels = [z.astype(np.float64)]
    while np.isnan(levels[-1]).any() and min(levels[-1].shape) > 1:
        a = levels[-1]
        h, w = a.shape
        H, W = (h + 1) // 2, (w + 1) // 2
        pad = np.full((2 * H, 2 * W), np.nan)
        pad[:h, :w] = a
        blk = pad.reshape(H, 2, W, 2)
        cnt = np.isfinite(blk).sum(axis=(1, 3))
        tot = np.nansum(blk, axis=(1, 3))
        levels.append(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan))
    top = levels[-1]
    top = np.where(np.isfinite(top), top, float(np.nanmean(z)))
    for a in reversed(levels[:-1]):
        upv = cv2.resize(top.astype(np.float32), (2 * top.shape[1], 2 * top.shape[0]), interpolation=cv2.INTER_LINEAR)
        top = np.where(np.isfinite(a), a, upv[: a.shape[0], : a.shape[1]].astype(np.float64))
    for _ in range(smooth_iterations):
        top = np.where(known, top, ndimage.gaussian_filter(top, 2.0))
    return top.astype(z.dtype)


def _fill_visible(z: np.ndarray, visible: np.ndarray, cell_m: float, elevated_m: float,
                  guide: np.ndarray | None = None, decay_m: float = 10.0):
    """Fill every visible empty cell; returns (z, newly filled mask, roof-like hole count).

    The fill is ``guide`` (the coarse-level surface; a flat plane when
    absent) plus the measured rim's difference from it, interpolated and
    faded with distance from the nearest measured cell. Holes are filled
    from the ground's differences (cells within ``elevated_m`` of their
    15 m local minimum) unless most of the hole's rim is elevated -- an
    untextured flat roof or a canopy gap -- which follows its rim.
    """
    from scipy import ndimage

    known = np.isfinite(z)
    target = visible & ~known
    if not target.any() or not known.any():
        return z, np.zeros(z.shape, dtype=bool), 0
    if guide is None:
        guide = np.full(z.shape, float(np.nanmedian(z)), dtype=np.float64)
    size = max(3, int(round(15.0 / cell_m)) | 1)
    low = ndimage.minimum_filter(np.where(known, z, np.inf), size=size, mode="nearest")
    elevated = known & np.isfinite(low) & (z - low > elevated_m)
    diff = np.where(known, z - guide, np.nan)
    fade = np.exp(-ndimage.distance_transform_edt(~known) * cell_m / max(decay_m, 1e-6))
    from_all = guide + fade * _smooth_fill(diff)
    ground_diff = np.where(elevated, np.nan, diff)
    from_ground = guide + fade * (_smooth_fill(ground_diff) if np.isfinite(ground_diff).any() else 0.0)
    lab, n = ndimage.label(target)
    rim = known & ndimage.binary_dilation(target)
    near = ndimage.maximum_filter(lab, size=3)[rim]
    elev_rim = np.bincount(near, weights=elevated[rim].astype(float), minlength=n + 1)
    all_rim = np.bincount(near, minlength=n + 1)
    roof_like = (all_rim > 0) & (elev_rim > 0.5 * np.maximum(all_rim, 1))
    roof_like[0] = False
    use_rim = roof_like[lab]
    out = z.copy()
    out[target] = np.where(use_rim, from_all, from_ground)[target].astype(z.dtype)
    return out, target, int(roof_like.sum())


def _inpaint_rgb(img: np.ndarray, mask: np.ndarray, max_side: int = 2048) -> np.ndarray:
    """Colour ``mask`` pixels from their surroundings (Telea), at reduced size for large images."""
    import cv2

    if not mask.any():
        return img
    h, w = mask.shape
    f = max(1, int(np.ceil(max(h, w) / max_side)))
    small = cv2.resize(img, (w // f or 1, h // f or 1), interpolation=cv2.INTER_AREA) if f > 1 else img
    m = cv2.resize(mask.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    fixed = cv2.inpaint(np.ascontiguousarray(small), m * 255, 5, cv2.INPAINT_TELEA)
    if f > 1:
        fixed = cv2.resize(fixed, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    out[mask] = fixed[mask]
    return out


def _prior_surface(points: np.ndarray | None, fallback_z: float, xmin: float, ymax: float, nx: int, ny: int, cell: float):
    """Coarse prior height per cell: rasterised BA points (median, holes filled by nearest), else a plane."""
    import cv2
    from scipy import ndimage

    if points is None:
        return np.full((ny, nx), fallback_z, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 50:
        return np.full((ny, nx), fallback_z, dtype=np.float32)
    rc = max(8.0, 4.0 * cell)
    rnx, rny = max(1, int(np.ceil(nx * cell / rc))), max(1, int(np.ceil(ny * cell / rc)))
    col = np.floor((pts[:, 0] - xmin) / rc).astype(np.int64)
    row = np.floor((ymax - pts[:, 1]) / rc).astype(np.int64)
    ok = (col >= 0) & (col < rnx) & (row >= 0) & (row < rny)
    key = row[ok] * rnx + col[ok]
    z = pts[ok, 2]
    grid = np.full(rny * rnx, np.nan)
    if key.size:
        order = np.argsort(key, kind="stable")
        key, z = key[order], z[order]
        uniq, start, count = np.unique(key, return_index=True, return_counts=True)
        for k_, s_, c_ in zip(uniq, start, count, strict=True):
            if c_ >= 3:
                grid[k_] = np.median(z[s_ : s_ + c_])
    grid = grid.reshape(rny, rnx)
    known = np.isfinite(grid)
    if not known.any():
        return np.full((ny, nx), fallback_z, dtype=np.float32)
    _, (ir, ic) = ndimage.distance_transform_edt(~known, return_indices=True)
    grid = grid[ir, ic]
    grid = ndimage.median_filter(grid, size=3)
    return cv2.resize(grid.astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR)


def _run_level(views, images, xmin, ymax, nx, ny, cell, prior, deltas, cfg, window, progress, label, on_tile=None):
    """Sweep every tile of one grid level; returns (z, score, views, sigma_m) arrays.

    ``on_tile(Z, S, K, U)`` (optional) sees the partly filled arrays after
    every tile -- the live preview.
    """
    torch = views.torch
    Z = np.full((ny, nx), np.nan, dtype=np.float32)
    S = np.full((ny, nx), -2.0, dtype=np.float32)
    K = np.zeros((ny, nx), dtype=np.float32)
    U = np.full((ny, nx), np.nan, dtype=np.float32)
    tn = max(8, round(cfg.tile_m / cell))
    halo = window
    tiles = [(ty, tx) for ty in range(0, ny, tn) for tx in range(0, nx, tn)]
    for n, (ty, tx) in enumerate(tiles):
        if progress is not None:
            progress(n, len(tiles), f"{label} tile {n + 1}/{len(tiles)}")
        y0, y1 = max(ty - halo, 0), min(ty + tn + halo, ny)
        x0, x1 = max(tx - halo, 0), min(tx + tn + halo, nx)
        zc_np = prior[y0:y1, x0:x1]
        samples = _tile_samples(
            xmin + x0 * cell, xmin + x1 * cell, ymax - y1 * cell, ymax - y0 * cell, float(np.nanmedian(zc_np))
        )
        sel = views.views_covering(samples, cfg.views_per_tile)
        if sel.numel() < cfg.min_views:
            continue
        gy, gx = torch.meshgrid(
            torch.arange(y0, y1, device=views.device, dtype=torch.float32),
            torch.arange(x0, x1, device=views.device, dtype=torch.float32),
            indexing="ij",
        )
        gx = xmin + (gx + 0.5) * cell
        gy = ymax - (gy + 0.5) * cell
        zc = torch.tensor(zc_np, device=views.device, dtype=torch.float32)
        bz, bs, bk, bu = _sweep_tile(
            views, images, gx, gy, zc, deltas, sel, window, cfg.min_views, cfg.score_top_k, cfg.ambiguity_sigmas
        )
        iy0, ix0 = ty - y0, tx - x0
        hh, ww = min(tn, ny - ty), min(tn, nx - tx)
        Z[ty : ty + hh, tx : tx + ww] = bz.cpu().numpy()[iy0 : iy0 + hh, ix0 : ix0 + ww]
        S[ty : ty + hh, tx : tx + ww] = bs.cpu().numpy()[iy0 : iy0 + hh, ix0 : ix0 + ww]
        K[ty : ty + hh, tx : tx + ww] = bk.cpu().numpy()[iy0 : iy0 + hh, ix0 : ix0 + ww]
        U[ty : ty + hh, tx : tx + ww] = bu.cpu().numpy()[iy0 : iy0 + hh, ix0 : ix0 + ww]
        if on_tile is not None:
            on_tile(Z, S, K, U)
    return Z, S, K, U


#: The live preview draws at most about this many cells (strided).
_PREVIEW_CELLS = 1_500_000


def _preview_cloud(Z, S, K, U, xmin: float, ymax: float, cell: float, cfg: HeightfieldConfig) -> PointCloud:
    """The cells measured so far as a tiered point cloud, strided to a display budget."""
    ny, nx = Z.shape
    st = max(1, int(np.ceil(np.sqrt(ny * nx / _PREVIEW_CELLS))))
    z, s, k, u = Z[::st, ::st], S[::st, ::st], K[::st, ::st], U[::st, ::st]
    ok = np.isfinite(z) & (s >= cfg.min_score) & (k >= cfg.min_views)
    rows, cols = np.nonzero(ok)
    xyz = np.stack(
        [xmin + (cols * st + 0.5) * cell, ymax - (rows * st + 0.5) * cell, z[ok]], axis=-1
    ).astype(np.float64)
    measured = (s[ok] >= cfg.measured_score) & (k[ok] >= 4)
    tier = np.where(measured, int(Confidence.MEASURED), int(Confidence.LOW_CONFIDENCE)).astype(np.uint8)
    return PointCloud(xyz=xyz, confidence=tier, uncertainty_m=u[ok].astype(np.float32))


def _colorize(
    views: _Views, z: np.ndarray, xmin: float, ymax: float, cell: float, cfg: HeightfieldConfig, seen_out=None
) -> np.ndarray:
    """Per-cell median colour across the nearest views at the chosen height (true-ortho).

    ``seen_out``, when given, is a (ny, nx) bool array set True for every
    cell at least one view actually sampled.
    """
    torch, F = views.torch, views.F
    ny, nx = z.shape
    out = np.zeros((ny, nx, 3), dtype=np.uint8)
    tn = max(8, round(cfg.tile_m / cell))
    fill = float(np.nanmedian(z)) if np.isfinite(z).any() else 0.0
    for ty in range(0, ny, tn):
        for tx in range(0, nx, tn):
            sl = (slice(ty, min(ty + tn, ny)), slice(tx, min(tx + tn, nx)))
            zt = z[sl]
            if not np.isfinite(zt).any():
                continue
            h, w = zt.shape
            X = xmin + (np.arange(tx, tx + w) + 0.5) * cell
            Y = ymax - (np.arange(ty, ty + h) + 0.5) * cell
            XX, YY = np.meshgrid(X, Y)
            P = np.stack([XX, YY, np.where(np.isfinite(zt), zt, fill)], -1)
            flat = P.reshape(-1, 3)
            sel = views.views_covering(flat[:: max(1, flat.shape[0] // 25)], cfg.color_views)
            if sel.numel() == 0:
                continue
            uv, ok = views.project(torch.tensor(P, dtype=torch.float32, device=views.device), sel)
            grid = torch.stack([uv[..., 0] / (views.w - 1) * 2 - 1, uv[..., 1] / (views.h - 1) * 2 - 1], -1)
            im = torch.from_numpy(views.rgb[sel.cpu().numpy()]).to(views.device, torch.float32).permute(0, 3, 1, 2)
            samp = F.grid_sample(im, grid, mode="bilinear", align_corners=True)
            samp = torch.where(ok[:, None], samp, torch.full_like(samp, float("nan")))
            med = torch.nanmedian(samp, dim=0).values.permute(1, 2, 0).cpu().numpy()
            out[sl] = np.nan_to_num(med, nan=0.0).clip(0, 255).astype(np.uint8)
            if seen_out is not None:
                seen_out[sl] = ok.any(dim=0).cpu().numpy()
    return out


def _sigma_summary(sigma: np.ndarray) -> dict | None:
    """Median / p90 height uncertainty and the share of cells within 0.5 m and 1 m."""
    sigma = sigma[np.isfinite(sigma)]
    if not sigma.size:
        return None
    return {
        "median": round(float(np.median(sigma)), 3),
        "p90": round(float(np.percentile(sigma, 90)), 3),
        "within_0_5m": round(float(np.mean(sigma <= 0.5)), 4),
        "within_1m": round(float(np.mean(sigma <= 1.0)), 4),
    }


def reconstruct_heightfield(
    images: list[np.ndarray],
    intrinsics: list[CameraIntrinsics],
    poses: list[Pose],
    *,
    prior_points: np.ndarray | None = None,
    agl_m: float | None = None,
    config: HeightfieldConfig | None = None,
    device: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    partial: Callable[[PointCloud], None] | None = None,
    partial_every_s: float = 1.0,
) -> tuple[HeightfieldSurface, dict]:
    """DSM of the ground the views share, from calibrated views and world-from-camera poses.

    ``images`` are BGR (or grey) uint8 keyframes; ``intrinsics[i]`` must
    describe ``images[i]``'s pixels (scaled to their resolution, with the
    lens's ``dist_coeffs`` when the pixels are raw). ``prior_points`` are
    world points on or near the surface (the bundle-adjusted points);
    without them the prior is a plane ``agl_m`` below the median camera.
    Returns the surface and a diagnostics dict. ``partial`` (optional)
    receives the fine level's cells as they are measured, at most every
    ``partial_every_s`` seconds, for a live view.
    """
    from scipy import ndimage

    from drishti3d.device import get_device

    torch, _ = _torch()
    cfg = config or HeightfieldConfig()
    if len(images) < cfg.min_views:
        raise ValueError(f"height-field MVS needs at least {cfg.min_views} views, got {len(images)}")
    dev = torch.device(device or get_device())
    t0 = time.monotonic()
    views = _Views(images, intrinsics, poses, dev)

    cams = np.stack([p.t for p in poses])
    cam_z = float(np.median(cams[:, 2]))
    if prior_points is not None and np.asarray(prior_points).size >= 150:
        pts = np.asarray(prior_points, dtype=np.float64).reshape(-1, 3)
        below = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] < cam_z - 5.0)]
        agl_est = float(np.median(cam_z - below[:, 2])) if below.size else (agl_m or 100.0)
        # Sparse points triangulated with little parallax land anywhere along
        # their rays; a handful between the cameras and the ground made a
        # coherent false surface under part of DJI_1001. Keep the band a real
        # surface can occupy: within 35% of the flight height of the median.
        if below.size:
            ground = float(np.median(below[:, 2]))
            below = below[np.abs(below[:, 2] - ground) <= 0.35 * agl_est]
    else:
        below = None
        agl_est = float(agl_m) if agl_m else 100.0
    # Ground extent: the camera track grown by a nadir footprint's half-width.
    half = 0.8 * agl_est
    xmin, ymin = cams[:, :2].min(axis=0) - half
    xmax, ymax = cams[:, :2].max(axis=0) + half

    # Level 1: coarse grid, wide sweep around the prior surface.
    c1 = cfg.coarse_cell_m
    nx1, ny1 = int(np.ceil((xmax - xmin) / c1)), int(np.ceil((ymax - ymin) / c1))
    prior = _prior_surface(below, cam_z - agl_est, xmin, ymax, nx1, ny1, c1)
    deltas1 = np.arange(-cfg.coarse_below_m, cfg.coarse_above_m + 1e-6, cfg.coarse_step_m)
    small = views.small(cfg.coarse_image_scale)
    Zc, Sc, _Kc, _Uc = _run_level(
        views, small, xmin, ymax, nx1, ny1, c1, prior, deltas1, cfg, cfg.coarse_window, progress, "coarse"
    )
    t1 = time.monotonic()

    good = np.isfinite(Zc) & (Sc >= cfg.min_score)
    guide = np.where(good, Zc, prior)
    guide = ndimage.median_filter(guide, size=5)

    # Level 2: fine grid, narrow sweep around the filtered coarse surface.
    import cv2

    c2 = cfg.fine_cell_m
    nx2, ny2 = int(np.ceil((xmax - xmin) / c2)), int(np.ceil((ymax - ymin) / c2))
    guide2 = cv2.resize(guide.astype(np.float32), (nx2, ny2), interpolation=cv2.INTER_LINEAR)
    deltas2 = np.arange(-cfg.fine_range_m, cfg.fine_range_m + 1e-6, cfg.fine_step_m)
    on_tile = None
    if partial is not None:
        last = [0.0]

        def on_tile(Z_, S_, K_, U_):
            now = time.monotonic()
            if now - last[0] >= partial_every_s:
                last[0] = now
                partial(_preview_cloud(Z_, S_, K_, U_, float(xmin), float(ymax), c2, cfg))

    Zf, Sf, Kf, Uf = _run_level(
        views, views.gray, xmin, ymax, nx2, ny2, c2, guide2, deltas2, cfg, cfg.fine_window, progress, "fine", on_tile
    )
    t2 = time.monotonic()

    # Keep agreement, drop spikes.
    valid = np.isfinite(Zf) & (Sf >= cfg.min_score) & (Kf >= cfg.min_views)
    Z = np.where(valid, Zf, np.nan).astype(np.float32)
    filled = np.where(valid, Z, np.nanmedian(Z) if valid.any() else 0.0)
    med = ndimage.median_filter(filled, size=5)
    spikes = valid & (np.abs(Z - med) > cfg.spike_m)
    Z[spikes] = np.nan
    # No surface within half the flight height of the camera plane: a nadir
    # survey's tallest structures are far below that, and a "surface" there
    # is a match between two wrong hypotheses, not a roof.
    too_high = np.isfinite(Z) & (Z > cam_z - 0.5 * agl_est)
    Z[too_high] = np.nan
    floating = np.zeros(Z.shape, dtype=bool)
    if cfg.max_structure_m and np.isfinite(Z).any():
        size = max(3, int(round(30.0 / c2)) | 1)
        ground_low = ndimage.minimum_filter(np.where(np.isfinite(Z), Z, np.inf), size=size, mode="nearest")
        floating = np.isfinite(Z) & np.isfinite(ground_low) & (Z - ground_low > cfg.max_structure_m)
        Z[floating] = np.nan
    Z, filled = _fill_small_holes(Z, cfg.fill_max_cells)

    seen = np.zeros(Z.shape, dtype=bool)
    # Colour (and find what the cameras saw) over the whole grid: empty cells
    # at a surface interpolated from the measured ones, so "seen" covers the
    # visible scene, not only where stereo succeeded.
    z_probe = _smooth_fill(Z) if cfg.fill_visible else Z
    rgb = _colorize(views, z_probe, xmin, ymax, c2, cfg, seen_out=seen)
    # A measured height no view can actually see is not a measurement.
    unseen = np.isfinite(Z) & ~seen
    Z[unseen] = np.nan
    filled &= np.isfinite(Z)
    measured_cells = int((np.isfinite(Z) & ~filled).sum())
    visible_fill = np.zeros(Z.shape, dtype=bool)
    roof_like_holes = 0
    if cfg.fill_visible and np.isfinite(Z).any():
        Z, visible_fill, roof_like_holes = _fill_visible(
            Z, seen, c2, cfg.fill_elevated_m, guide=guide2.astype(np.float64), decay_m=cfg.fill_decay_m
        )
        filled |= visible_fill
    if visible_fill.any():
        # A fill must not invent structure. Where measured cells lie within
        # 30 m it may not rise above them (+2 m); anywhere, it may not stand
        # max_structure_m above the local ground. The coarse guide it follows
        # far from measurements had bogus patches on DJI_1001 (44k filled
        # cells 60-280 m above the ground before this).
        size = max(3, int(round(30.0 / c2)) | 1)
        meas = np.isfinite(Z) & ~filled
        local_max = ndimage.maximum_filter(np.where(meas, Z, -np.inf), size=size, mode="nearest")
        local_min = ndimage.minimum_filter(np.where(meas, Z, np.inf), size=size, mode="nearest")
        # Within 30 m of measurements a fill stays inside their height range
        # (+-2 m): not above them (invented structure), not below them (the
        # skirts it hung off the survey's edge where the guide sat lower).
        capped = visible_fill & np.isfinite(local_max) & (Z > local_max + 2.0)
        Z[capped] = (local_max[capped] + 2.0).astype(Z.dtype)
        lifted = visible_fill & np.isfinite(local_min) & (Z < local_min - 2.0)
        Z[lifted] = (local_min[lifted] - 2.0).astype(Z.dtype)
        capped |= lifted
        ground_low = ndimage.minimum_filter(np.where(np.isfinite(Z), Z, np.inf), size=size, mode="nearest")
        tall = visible_fill & np.isfinite(ground_low) & (Z - ground_low > cfg.max_structure_m)
        Z[tall] = np.nan
        filled &= np.isfinite(Z)
        visible_fill &= np.isfinite(Z)
        fill_capped, fill_dropped = int(capped.sum()), int(tall.sum())
    else:
        fill_capped = fill_dropped = 0
    # The same plausibility rule for every cell, filled ones included.
    late_high = np.isfinite(Z) & (Z > cam_z - 0.5 * agl_est)
    if late_high.any():
        logger.info("height-field: %d cells above half the flight height removed after filling", int(late_high.sum()))
        Z[late_high] = np.nan
        filled &= np.isfinite(Z)
        visible_fill &= np.isfinite(Z)
    if cfg.fill_visible:
        # Small holes enclosed by the completed surface are inside the scene
        # even where the 8-view visibility test missed them (flight01: a 2-4 m
        # gap an oblique survey ray fell through).
        Z, enclosed = _fill_small_holes(Z, cfg.fill_max_cells)
        filled |= enclosed
        visible_fill |= enclosed
    unseen_fill = np.zeros(Z.shape, dtype=bool)
    if cfg.fill_visible and cfg.fill_enclosed_unseen and np.isfinite(Z).any():
        holes = ~np.isfinite(Z)
        lab, n = ndimage.label(holes)
        if n:
            outside = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
            inside = np.ones(n + 1, dtype=bool)
            inside[outside] = False
            inside[0] = False
            unseen_fill = inside[lab]
            if unseen_fill.any():
                Z, _, _ = _fill_visible(
                    Z, unseen_fill, c2, cfg.fill_elevated_m, guide=guide2.astype(np.float64), decay_m=cfg.fill_decay_m
                )
                unseen_fill &= np.isfinite(Z)
                filled |= unseen_fill
                visible_fill |= unseen_fill
    texture = None
    up = int(cfg.texture_upsample or 0)
    if up > 1:
        up = max(1, min(up, cfg.texture_max_px // max(nx2, ny2)))
    if up >= 1 and cfg.texture_upsample:
        # Heights for every texel: the surface interpolated, holes filled so
        # the texture has no black cracks where a cell was rejected.
        from scipy import ndimage as _nd

        known = np.isfinite(Z)
        if known.any():
            _, (ir, ic) = _nd.distance_transform_edt(~known, return_indices=True)
            zfill = Z[ir, ic]
            ztex = cv2.resize(zfill, (nx2 * up, ny2 * up), interpolation=cv2.INTER_LINEAR)
            texture = _colorize(views, ztex, xmin, ymax, c2 / up, cfg)
    if unseen_fill.any():
        # No camera saw these cells: their colour is inpainted from the
        # surroundings (they are INFERRED, the tier says so), not left black.
        rgb = _inpaint_rgb(rgb, unseen_fill)
        if texture is not None:
            tmask = cv2.resize(unseen_fill.astype(np.uint8), (texture.shape[1], texture.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            texture = _inpaint_rgb(texture, tmask & (texture.max(axis=2) == 0))
    t3 = time.monotonic()
    surface = HeightfieldSurface(
        z=Z,
        score=Sf.astype(np.float32),
        views=Kf.astype(np.float32),
        rgb=rgb,
        xmin=float(xmin),
        ymax=float(ymax),
        cell_m=float(c2),
        measured_score=cfg.measured_score,
        max_step_m=cfg.max_step_m,
        texture_rgb=texture,
        inferred=filled,
        sigma_m=np.where(np.isfinite(Z), Uf, np.nan).astype(np.float32),
    )
    reconstructed = int(np.isfinite(Z).sum())
    diag = {
        "method": "heightfield_mvs",
        "views": len(images),
        "grid": [int(ny2), int(nx2)],
        "cell_m": c2,
        "agl_estimate_m": round(agl_est, 2),
        "prior": "bundle_adjusted_points" if below is not None and below.size else "telemetry_plane",
        "cells_reconstructed": reconstructed,
        "cells_fraction": round(reconstructed / float(Z.size), 4),
        "spikes_removed": int(spikes.sum()),
        "implausible_removed": int(too_high.sum() + unseen.sum() + floating.sum() + late_high.sum()),
        "floating_removed": int(floating.sum()),
        "fill_capped_cells": fill_capped,
        "fill_dropped_cells": fill_dropped,
        "holes_filled_cells": int(filled.sum()),
        # Coverage of what the cameras saw: measured by stereo, or filled
        # from the surroundings (INFERRED). Cells no camera saw are not counted.
        "visible_cells": int(seen.sum()),
        "visible_measured_fraction": round(measured_cells / max(int(seen.sum()), 1), 4),
        "visible_inferred_fraction": round(int((filled & seen).sum()) / max(int(seen.sum()), 1), 4),
        "visible_filled_cells": int(visible_fill.sum()),
        "unseen_enclosed_filled_cells": int(unseen_fill.sum()),
        "roof_like_holes": roof_like_holes,
        "texture_px": None if texture is None else [int(texture.shape[0]), int(texture.shape[1])],
        "score_median": round(float(np.median(Sf[valid])), 4) if valid.any() else None,
        "height_uncertainty_m": _sigma_summary(Uf[np.isfinite(Z) & ~filled]),
        "seconds": {
            "coarse": round(t1 - t0, 2),
            "fine": round(t2 - t1, 2),
            "colour": round(t3 - t2, 2),
        },
        "device": str(dev),
    }
    logger.info(
        "height-field MVS: %d views -> %dx%d grid at %.2f m, %d cells (%.0f%%), median NCC %s, %.1f s",
        len(images), ny2, nx2, c2, reconstructed, 100.0 * diag["cells_fraction"], diag["score_median"], t3 - t0,
    )
    return surface, diag
