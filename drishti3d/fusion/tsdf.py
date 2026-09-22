"""Confidence-weighted volumetric fusion (TSDF).

A truncated signed distance function (TSDF) volume is the classic way to
merge many noisy depth observations of the same surface into one clean
mesh: each depth map "votes" on the signed distance to the surface for
every voxel it can see, and votes are averaged, weighted by how much each
observation should be trusted.

That last clause is the point of this module. A plain TSDF just averages
every observation with equal weight, which means one confidently wrong
observation counts exactly as much as ten solid ones. ``TSDFVolume
.integrate`` accepts an optional per-pixel ``confidence_map`` and folds it
straight into the integration weight -- a low-confidence pixel (e.g. an
inferred/hallucinated region of the geometry backbone's output, or a
grazing-incidence pixel near a depth discontinuity) simply moves the
running average less than a high-confidence one would. This is what makes
the fused output *trust-aware* rather than a plain multi-view average that
happens to look smoother: two voxels with the same final SDF value can
have arrived there via very different evidence, and this module keeps
track of exactly how much.

That same bookkeeping (accumulated weight + number of distinct
integrations that touched each voxel) is what lets mesh extraction assign
a genuine per-vertex confidence tier instead of a single global "trust
me": a voxel hit hard, from many viewpoints, is ``MEASURED``; a voxel that
one grazing frame barely brushed is ``LOW_CONFIDENCE``; a voxel nobody
ever observed, whose value is still whatever the volume's neutral "empty"
default is, produces ``INFERRED`` geometry if a triangle happens to pass
through it anyway (see the module-level note on "why unobserved voxels
still show up" below).

Why unobserved voxels can still produce a triangle
---------------------------------------------------
Every voxel starts at a neutral default (empty space, full positive
truncated distance, zero weight). Marching a zero-crossing through the
volume does not know or care *why* a voxel holds the value it does -- if
one frame carves out a thin negative shell just behind a real surface, and
the far side of an object was simply never observed, the boundary between
"genuinely measured, just behind the surface" and "never touched, still
at the default empty value" can itself look like another zero crossing.
Real TSDF fusion (this includes Open3D's) has exactly this behaviour: it
will happily "close" the unseen back side of an object with a surface
nobody ever measured. Rather than trying to suppress that (which would
just turn holes into missing triangles instead of mislabeled ones -- still
not the truth), this module leans into it: every generated vertex is
classified from the accumulated weight/view-count of the two grid samples
its position was interpolated between, so a cap like that comes out
labelled ``INFERRED`` instead of silently looking exactly as solid as the
part of the model that was actually measured. That labelling, not
suppression, is the actual point of a "trust-aware" reconstruction.

Backends
--------
When ``open3d`` is importable, ``TSDFVolume`` delegates the actual SDF/
colour integration to ``open3d.pipelines.integration.ScalableTSDFVolume``
(much faster, block-sparse, battle-tested). Confidence bookkeeping is kept
in a small parallel numpy structure either way (computing "which voxels
did this frame touch, with how much weight" is cheap relative to the
SDF integration itself, so there is no real cost to always doing it the
same way, and it keeps the two backends' confidence semantics identical).
When ``open3d`` is not importable, a pure numpy/scipy dense-grid fallback
implements the same integration math directly -- correct, and fine for
the "modest volumes" (up to a few hundred thousand voxels) this project's
per-window fusion actually needs; it is not meant to compete with a
block-sparse GPU/production implementation at city scale.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from drishti3d.types import CameraIntrinsics, Confidence, PointCloud, Pose, Submap

logger = logging.getLogger(__name__)

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:  # pragma: no cover - exercised only where open3d is installed
    o3d = None
    _HAS_OPEN3D = False

__all__ = [
    "HAS_OPEN3D",
    "TSDFVolume",
    "fuse_submaps",
]

# Re-exported so callers/tests can assert on it without reaching into a
# private module attribute.
HAS_OPEN3D = _HAS_OPEN3D

# A voxel needs at least this much accumulated weight, from at least this
# many distinct observations, to be called MEASURED. Below that but still
# >0 weight is LOW_CONFIDENCE; exactly 0 weight is INFERRED (see module
# docstring for why a 0-weight voxel can still end up on the extracted
# surface at all).
#
# These are ``TSDFVolume``'s own standalone defaults, used whenever a
# caller (e.g. this module's own tests, or anyone constructing a
# ``TSDFVolume`` directly over raw depth maps via ``integrate()``) doesn't
# override them. ``fuse_submaps`` -- the point-cloud-splatting entry point
# most callers actually go through -- overrides both from
# ``config.FusionConfig.measured_min_weight``/``measured_min_views``
# instead (see that dataclass's docstring for the calibration reasoning,
# and this module's docstring for why the point-cloud path needs different
# values than these raw-depth-map defaults).
_MEASURED_MIN_WEIGHT = 1.5
_MEASURED_MIN_VIEWS = 2

# Default value a freshly-allocated (or grid-grown) voxel holds: "far in
# front of / away from any surface", i.e. empty space, in truncated-SDF
# units. Using +1 (rather than 0) means a single observation carving out a
# thin negative shell is what creates a zero crossing, not the raw
# initial grid state.
_EMPTY_SDF = 1.0
_EMPTY_COLOR = 128.0  # neutral grey for the "never observed" default

# Growing the grid pads this many extra voxels beyond whatever a frame's
# backprojected points strictly need, so a point sitting exactly on the
# current boundary doesn't force a re-grow on the very next frame too.
_GROW_MARGIN_VOXELS = 2

# See TSDFVolume.integrate_point_cloud's "Chunked, not all at once" comment:
# the max number of (voxel, contributing-point) pairs materialized into a
# single batch of numpy arrays at once. Sized so a chunk's temporary arrays
# (a handful of float64 (N,) / (N, 3) arrays) stay on the order of a few
# hundred MB even in the worst case, regardless of how many voxels or how
# dense the input cloud is -- an *unchunked* version of this same
# vectorization was measured to thrash/OOM on a realistic-density synthetic
# benchmark (~24k contributing points per voxel at this project's actual
# sdf_trunc-radius/pre-TSDF-downsample-spacing ratio), which is worse than
# the original per-voxel Python loop it was meant to replace.
_INTEGRATE_CHUNK_MAX_PAIRS = 4_000_000


def _tier_from_weight(
    weight: np.ndarray,
    views: np.ndarray,
    measured_min_weight: float = _MEASURED_MIN_WEIGHT,
    measured_min_views: int = _MEASURED_MIN_VIEWS,
) -> np.ndarray:
    """Vectorized MEASURED/LOW_CONFIDENCE/INFERRED classification from accumulated stats.

    ``measured_min_weight``/``measured_min_views`` default to this module's
    own standalone constants; ``fuse_submaps`` passes
    ``config.FusionConfig``-sourced values instead (see that dataclass's
    docstring for why the point-cloud-splatting path needs a different
    calibration than these raw-depth-map defaults).
    """
    tier = np.full(weight.shape, Confidence.LOW_CONFIDENCE, dtype=np.uint8)
    tier[weight <= 0] = Confidence.INFERRED
    measured = (weight >= measured_min_weight) & (views >= measured_min_views)
    tier[measured] = Confidence.MEASURED
    return tier


# ---------------------------------------------------------------------------
# Marching tetrahedra: the numpy-fallback isosurface extractor.
# ---------------------------------------------------------------------------
#
# We deliberately use marching *tetrahedra* rather than marching cubes.
# Marching cubes' 256-configuration/triangle-ambiguity table is fiddly to
# hand-transcribe correctly; marching tetrahedra only has 16
# configurations (each of a tetrahedron's 4 corners is either "inside" or
# "outside"), the case logic collapses to three symmetric shapes (1-in/
# 3-out, 3-in/1-out, 2-in/2-out), and splitting each cube into 6
# tetrahedra sharing the cube's main diagonal (the standard Kuhn/Freudenthal
# triangulation) tiles space with no cracks between neighbouring cubes.
# The trade-off is a somewhat higher triangle count for the same grid --
# an acceptable price for an implementation that is actually correct.

# Cube corner offsets, in (dx, dy, dz), indexed 0..7.
_CUBE_CORNERS = np.array(
    [
        (0, 0, 0),  # 0
        (1, 0, 0),  # 1
        (1, 1, 0),  # 2
        (0, 1, 0),  # 3
        (0, 0, 1),  # 4
        (1, 0, 1),  # 5
        (1, 1, 1),  # 6
        (0, 1, 1),  # 7
    ],
    dtype=np.int64,
)

# Six tetrahedra, each a list of 4 corner indices (into _CUBE_CORNERS),
# all sharing the 0-6 body diagonal (Kuhn triangulation).
_TETRAHEDRA = (
    (0, 1, 2, 6),
    (0, 1, 5, 6),
    (0, 3, 2, 6),
    (0, 3, 7, 6),
    (0, 4, 5, 6),
    (0, 4, 7, 6),
)


def _interp_edge(
    val_a: np.ndarray,
    val_b: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    color_a: np.ndarray,
    color_b: np.ndarray,
    tier_a: np.ndarray,
    tier_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate the zero crossing between two grid samples.

    Confidence is the *minimum* of the two endpoints' tiers -- consistent
    with this whole project's rule of thumb that a boundary is only as
    trustworthy as its least-trustworthy side.
    """
    denom = val_a - val_b
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t = (val_a / denom)[:, None]
    pos = pos_a + t * (pos_b - pos_a)
    color = color_a + t * (color_b - color_a)
    conf = np.minimum(tier_a, tier_b)
    return pos, color, conf


def _marching_tetrahedra(
    values: np.ndarray,
    weights: np.ndarray,
    views: np.ndarray,
    colors: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    measured_min_weight: float = _MEASURED_MIN_WEIGHT,
    measured_min_views: int = _MEASURED_MIN_VIEWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a zero-isosurface triangle mesh from a dense TSDF grid.

    ``values``/``weights``/``views`` are ``(nx, ny, nz)``; ``colors`` is
    ``(nx, ny, nz, 3)``. Returns welded ``(vertices, faces, vertex_colors,
    vertex_confidence)``.
    """
    nx, ny, nz = values.shape
    if nx < 2 or ny < 2 or nz < 2:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
        )

    tier = _tier_from_weight(weights, views, measured_min_weight, measured_min_views)

    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    base_idx = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)  # (M, 3), M = (nx-1)(ny-1)(nz-1)
    base_pos = origin + (base_idx + 0.5) * voxel_size  # world position of local corner 0

    def corner_slice(arr: np.ndarray, dx: int, dy: int, dz: int) -> np.ndarray:
        sliced = arr[dx : nx - 1 + dx, dy : ny - 1 + dy, dz : nz - 1 + dz]
        return sliced.reshape(-1, *arr.shape[3:])

    corner_val = [corner_slice(values, *_CUBE_CORNERS[c]) for c in range(8)]
    corner_tier = [corner_slice(tier, *_CUBE_CORNERS[c]) for c in range(8)]
    corner_color = [corner_slice(colors, *_CUBE_CORNERS[c]) for c in range(8)]
    corner_pos = [base_pos + _CUBE_CORNERS[c] * voxel_size for c in range(8)]

    out_pos: list[np.ndarray] = []
    out_color: list[np.ndarray] = []
    out_conf: list[np.ndarray] = []
    out_faces: list[np.ndarray] = []

    for tet in _TETRAHEDRA:
        vals = [corner_val[c] for c in tet]
        poss = [corner_pos[c] for c in tet]
        cols = [corner_color[c] for c in tet]
        tiers = [corner_tier[c] for c in tet]

        inside = [v < 0 for v in vals]
        case = np.zeros(vals[0].shape[0], dtype=np.int64)
        for bit, ins in enumerate(inside):
            case += ins.astype(np.int64) << bit

        for case_val in range(1, 15):
            mask = case == case_val
            if not np.any(mask):
                continue
            inside_idx = [j for j in range(4) if case_val & (1 << j)]
            outside_idx = [j for j in range(4) if not (case_val & (1 << j))]
            n_inside = len(inside_idx)

            def edge(
                a: int,
                b: int,
                mask: np.ndarray = mask,
                vals: list[np.ndarray] = vals,
                poss: list[np.ndarray] = poss,
                cols: list[np.ndarray] = cols,
                tiers: list[np.ndarray] = tiers,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                return _interp_edge(
                    vals[a][mask],
                    vals[b][mask],
                    poss[a][mask],
                    poss[b][mask],
                    cols[a][mask],
                    cols[b][mask],
                    tiers[a][mask],
                    tiers[b][mask],
                )

            if n_inside in (1, 3):
                lone = inside_idx[0] if n_inside == 1 else outside_idx[0]
                others = outside_idx if n_inside == 1 else inside_idx
                p0, c0, f0 = edge(lone, others[0])
                p1, c1, f1 = edge(lone, others[1])
                p2, c2, f2 = edge(lone, others[2])
                n_tri = p0.shape[0]
                start = sum(a.shape[0] for a in out_pos)
                if n_inside == 1:
                    tri_pos = np.stack([p0, p1, p2], axis=1)
                    tri_col = np.stack([c0, c1, c2], axis=1)
                    tri_conf = np.stack([f0, f1, f2], axis=1)
                else:
                    # 3-inside/1-outside is the mirror case; reverse winding
                    # so both cases produce outward-consistent triangles
                    # relative to the "inside" bulk each cuts off.
                    tri_pos = np.stack([p0, p2, p1], axis=1)
                    tri_col = np.stack([c0, c2, c1], axis=1)
                    tri_conf = np.stack([f0, f2, f1], axis=1)
                out_pos.append(tri_pos.reshape(-1, 3))
                out_color.append(tri_col.reshape(-1, 3))
                out_conf.append(tri_conf.reshape(-1))
                idx = np.arange(start, start + n_tri * 3, dtype=np.int64).reshape(-1, 3)
                out_faces.append(idx)
            else:  # n_inside == 2: quad -> 2 triangles
                i0, i1 = inside_idx
                o0, o1 = outside_idx
                p00, c00, f00 = edge(i0, o0)
                p01, c01, f01 = edge(i0, o1)
                p11, c11, f11 = edge(i1, o1)
                p10, c10, f10 = edge(i1, o0)
                n_quad = p00.shape[0]
                start = sum(a.shape[0] for a in out_pos)
                tri_pos = np.concatenate(
                    [
                        np.stack([p00, p01, p11], axis=1),
                        np.stack([p00, p11, p10], axis=1),
                    ],
                    axis=0,
                )
                tri_col = np.concatenate(
                    [
                        np.stack([c00, c01, c11], axis=1),
                        np.stack([c00, c11, c10], axis=1),
                    ],
                    axis=0,
                )
                tri_conf = np.concatenate(
                    [
                        np.stack([f00, f01, f11], axis=1),
                        np.stack([f00, f11, f10], axis=1),
                    ],
                    axis=0,
                )
                out_pos.append(tri_pos.reshape(-1, 3))
                out_color.append(tri_col.reshape(-1, 3))
                out_conf.append(tri_conf.reshape(-1))
                idx1 = np.arange(start, start + n_quad * 3, dtype=np.int64).reshape(-1, 3)
                idx2 = np.arange(start + n_quad * 3, start + 2 * n_quad * 3, dtype=np.int64).reshape(-1, 3)
                out_faces.append(np.concatenate([idx1, idx2], axis=0))

    if not out_pos:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint8),
        )

    raw_vertices = np.concatenate(out_pos, axis=0)
    raw_colors = np.clip(np.concatenate(out_color, axis=0), 0, 255).astype(np.uint8)
    raw_conf = np.concatenate(out_conf, axis=0).astype(np.uint8)
    raw_faces = np.concatenate(out_faces, axis=0)

    # Weld coincident vertices (every tetrahedron computed its own edge
    # intersections independently, so adjacent cubes currently duplicate
    # every shared-edge crossing point).
    quant = np.round(raw_vertices / (voxel_size * 1e-4)).astype(np.int64)
    _uniq, inverse, counts = np.unique(quant, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n_verts = counts.shape[0]

    pos_sums = np.zeros((n_verts, 3), dtype=np.float64)
    np.add.at(pos_sums, inverse, raw_vertices)
    vertices = pos_sums / counts[:, None]

    color_sums = np.zeros((n_verts, 3), dtype=np.float64)
    np.add.at(color_sums, inverse, raw_colors.astype(np.float64))
    vertex_colors = np.round(color_sums / counts[:, None]).astype(np.uint8)

    conf_out = np.full(n_verts, 255, dtype=np.uint8)
    np.minimum.at(conf_out, inverse, raw_conf)

    faces = inverse[raw_faces]

    return vertices, faces, vertex_colors, conf_out


# ---------------------------------------------------------------------------
# TSDFVolume
# ---------------------------------------------------------------------------


class TSDFVolume:
    """A confidence-weighted TSDF volume.

    Grid bounds can be given explicitly (``origin`` + ``dims``), which is
    recommended whenever the scene extent is already known (e.g. from a
    merged point cloud's bounding box) -- otherwise the volume lazily
    allocates itself from the first ``integrate()`` call's backprojected
    depth extent and grows (by re-padding, not reallocating from scratch)
    whenever a later frame needs more room.
    """

    def __init__(
        self,
        voxel_size: float,
        sdf_trunc: float,
        origin: np.ndarray | None = None,
        dims: tuple[int, int, int] | None = None,
        use_open3d: bool | None = None,
        measured_min_weight: float = _MEASURED_MIN_WEIGHT,
        measured_min_views: int = _MEASURED_MIN_VIEWS,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.sdf_trunc = float(sdf_trunc)
        self._use_open3d = _HAS_OPEN3D if use_open3d is None else (use_open3d and _HAS_OPEN3D)
        # See _tier_from_weight's docstring / config.FusionConfig for what
        # these mean and why fuse_submaps overrides the module defaults.
        self._measured_min_weight = float(measured_min_weight)
        self._measured_min_views = int(measured_min_views)

        self._values: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._views: np.ndarray | None = None
        self._colors: np.ndarray | None = None
        self._origin: np.ndarray | None = None if origin is None else np.asarray(origin, dtype=np.float64)

        self._o3d_volume = None
        if self._use_open3d:
            self._o3d_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_size,
                sdf_trunc=self.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

        if origin is not None and dims is not None:
            self._alloc(np.asarray(origin, dtype=np.float64), tuple(int(d) for d in dims))

    # -- grid management -----------------------------------------------

    def _alloc(self, origin: np.ndarray, dims: tuple[int, int, int]) -> None:
        self._origin = origin
        self._values = np.full(dims, _EMPTY_SDF, dtype=np.float64)
        self._weights = np.zeros(dims, dtype=np.float64)
        self._views = np.zeros(dims, dtype=np.int32)
        self._colors = np.full((*dims, 3), _EMPTY_COLOR, dtype=np.float64)

    def _grow_to_include(self, min_pt: np.ndarray, max_pt: np.ndarray) -> None:
        margin = _GROW_MARGIN_VOXELS * self.voxel_size
        min_pt = min_pt - margin
        max_pt = max_pt + margin

        if self._values is None:
            origin = np.floor(min_pt / self.voxel_size) * self.voxel_size
            dims = np.ceil((max_pt - origin) / self.voxel_size).astype(np.int64)
            dims = np.maximum(dims, 1) + 1
            self._alloc(origin, tuple(int(d) for d in dims))
            return

        cur_dims = np.array(self._values.shape[:3])
        cur_max = self._origin + cur_dims * self.voxel_size

        lo_pad = np.maximum(np.ceil((self._origin - min_pt) / self.voxel_size), 0).astype(np.int64)
        hi_pad = np.maximum(np.ceil((max_pt - cur_max) / self.voxel_size), 0).astype(np.int64)

        if not np.any(lo_pad > 0) and not np.any(hi_pad > 0):
            return

        pad_width = [(int(lo_pad[a]), int(hi_pad[a])) for a in range(3)]
        self._values = np.pad(self._values, pad_width, mode="constant", constant_values=_EMPTY_SDF)
        self._weights = np.pad(self._weights, pad_width, mode="constant", constant_values=0.0)
        self._views = np.pad(self._views, pad_width, mode="constant", constant_values=0)
        self._colors = np.pad(
            self._colors, [*pad_width, (0, 0)], mode="constant", constant_values=_EMPTY_COLOR
        )
        self._origin = self._origin - lo_pad * self.voxel_size

    def _voxel_centers(self) -> np.ndarray:
        nx, ny, nz = self._values.shape
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        idx = np.stack([ii, jj, kk], axis=-1)
        return self._origin + (idx + 0.5) * self.voxel_size

    # -- integration ------------------------------------------------------

    def integrate(
        self,
        depth: np.ndarray,
        color: np.ndarray | None,
        intrinsics: CameraIntrinsics,
        pose: Pose,
        confidence_map: np.ndarray | None = None,
    ) -> None:
        """Integrate one RGB-D(-confidence) observation into the volume.

        ``depth`` is camera-frame z-depth in metres, ``(H, W)``, with
        ``<= 0`` meaning "no depth measurement at this pixel". ``pose`` is
        world-from-camera (see ``types.Pose``). ``confidence_map``, when
        given, is ``(H, W)`` float in ``[0, 1]`` and directly scales this
        frame's contribution to every voxel it touches -- this is the
        entire trust-aware mechanism this module exists for.
        """
        depth = np.asarray(depth, dtype=np.float64)
        h, w = depth.shape
        valid_px = depth > 0
        if not np.any(valid_px):
            return

        K_inv = np.linalg.inv(intrinsics.K())
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
        pix = np.stack([us, vs, np.ones_like(us)], axis=-1)
        dirs_cam = pix @ K_inv.T
        pts_cam = dirs_cam * depth[..., None]
        pts_world = pts_cam.reshape(-1, 3) @ pose.R.T + pose.t
        valid_flat = valid_px.reshape(-1)
        valid_pts = pts_world[valid_flat]
        if valid_pts.shape[0] == 0:
            return

        margin = self.sdf_trunc + self.voxel_size
        self._grow_to_include(valid_pts.min(axis=0) - margin, valid_pts.max(axis=0) + margin)

        if self._use_open3d:
            self._integrate_open3d(depth, color, intrinsics, pose)

        self._integrate_numpy(depth, color, intrinsics, pose, confidence_map)

    def _integrate_open3d(
        self, depth: np.ndarray, color: np.ndarray | None, intrinsics: CameraIntrinsics, pose: Pose
    ) -> None:  # pragma: no cover - requires the optional open3d dependency
        h, w = depth.shape
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        if color is not None:
            color_o3d = o3d.geometry.Image(np.ascontiguousarray(color.astype(np.uint8)))
        else:
            color_o3d = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1e6, convert_rgb_to_intensity=False
        )
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        extrinsic = np.linalg.inv(pose.matrix())  # open3d wants world-to-camera
        self._o3d_volume.integrate(rgbd, intr, extrinsic)

    def _integrate_numpy(
        self,
        depth: np.ndarray,
        color: np.ndarray | None,
        intrinsics: CameraIntrinsics,
        pose: Pose,
        confidence_map: np.ndarray | None,
    ) -> None:
        h, w = depth.shape
        centers = self._voxel_centers()
        grid_shape = centers.shape[:3]
        centers_flat = centers.reshape(-1, 3)

        p_cam = (centers_flat - pose.t) @ pose.R
        z = p_cam[:, 2]
        in_front = z > 1e-9
        safe_z = np.where(in_front, z, 1.0)
        u = intrinsics.fx * p_cam[:, 0] / safe_z + intrinsics.cx
        v = intrinsics.fy * p_cam[:, 1] / safe_z + intrinsics.cy
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        in_bounds = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

        m = centers_flat.shape[0]
        depth_measured = np.zeros(m, dtype=np.float64)
        idx_bounds = np.nonzero(in_bounds)[0]
        depth_measured[idx_bounds] = depth[vi[idx_bounds], ui[idx_bounds]]
        has_depth = in_bounds & (depth_measured > 0)

        sdf = depth_measured - z
        within_trunc = has_depth & (np.abs(sdf) <= self.sdf_trunc)
        if not np.any(within_trunc):
            return

        weight_val = np.ones(m, dtype=np.float64)
        if confidence_map is not None:
            confidence_map = np.asarray(confidence_map, dtype=np.float64)
            conf_sampled = np.zeros(m, dtype=np.float64)
            conf_sampled[idx_bounds] = confidence_map[vi[idx_bounds], ui[idx_bounds]]
            weight_val = np.where(within_trunc, conf_sampled, weight_val)

        color_sample = None
        if color is not None:
            color = np.asarray(color, dtype=np.float64)
            color_sample = np.zeros((m, 3), dtype=np.float64)
            color_sample[idx_bounds] = color[vi[idx_bounds], ui[idx_bounds]]

        values_flat = self._values.reshape(-1)
        weights_flat = self._weights.reshape(-1)
        views_flat = self._views.reshape(-1)
        colors_flat = self._colors.reshape(-1, 3)

        old_val = values_flat[within_trunc]
        old_w = weights_flat[within_trunc]
        new_w = weight_val[within_trunc]
        new_sdf = sdf[within_trunc]
        combined_w = old_w + new_w
        combined_w_safe = np.where(combined_w > 0, combined_w, 1.0)

        new_val = (old_val * old_w + new_sdf * new_w) / combined_w_safe
        values_flat[within_trunc] = np.clip(new_val, -self.sdf_trunc, self.sdf_trunc)
        weights_flat[within_trunc] = combined_w
        views_flat[within_trunc] += 1

        if color_sample is not None:
            old_color = colors_flat[within_trunc]
            new_color = color_sample[within_trunc]
            colors_flat[within_trunc] = (old_color * old_w[:, None] + new_color * new_w[:, None]) / combined_w_safe[
                :, None
            ]

        del grid_shape  # only needed conceptually; arrays already flat-viewed in place

    def integrate_point_cloud(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
    ) -> None:
        """Splat an oriented point cloud into the volume as local signed-distance samples.

        Used when the input is an already-merged point cloud with per-point
        confidence (the actual shape of data ``fuse_submaps`` gets -- see
        that function's docstring) rather than raw per-view depth maps. For
        a voxel near a point ``p`` with unit normal ``n``, the local surface
        is approximated as the tangent plane through ``p``, so the signed
        distance from a nearby voxel centre ``x`` is ``dot(x - p, n)``. Each
        voxel averages every point within ``sdf_trunc`` of it, weighted by
        that point's confidence and inverse distance -- this is a coarse,
        purely local stand-in for a real implicit-surface fit (true
        Poisson reconstruction solves a global Poisson equation), but it is
        cheap, needs no extra dependency, and reuses exactly the same grid/
        marching-tetrahedra machinery as depth-map integration.

        What counts as a "view" here (read before changing this)
        -------------------------------------------------------------
        Unlike ``integrate()`` -- where each call is genuinely one camera
        frame, so "one call = one view" is the right way to bump
        ``self._views`` -- this method is typically called *once* by
        ``fuse_submaps`` over an already-merged, multi-submap point cloud
        (see that function's docstring: it re-merges every submap up
        front, then integrates the whole cleaned cloud in a single call).
        Counting "one call = one view" here would make ``self._views`` cap
        out at 1 for literally every voxel regardless of how many
        independent backbone pixel-observations actually support it --
        which is exactly the bug that made MEASURED unreachable before
        this was fixed (see ``fusion.tsdf`` module docstring). Each row of
        ``points`` already *is* one such observation (``Submap.points``,
        and therefore the merged cloud, is one row per backbone
        pixel/keyframe hit -- see ``types.Submap``'s docstring), so the
        right unit of "a view" here is "one distinct point that actually
        contributed nonzero weight to this voxel," counted per voxel
        below, not "one call to this method."
        """
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        n = points.shape[0]
        if n == 0:
            return
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norm_len > 1e-12, norm_len, 1.0)

        if confidence is None:
            confidence = np.ones(n, dtype=np.float64)
        else:
            confidence = np.asarray(confidence, dtype=np.float64)

        margin = self.sdf_trunc + self.voxel_size
        self._grow_to_include(points.min(axis=0) - margin, points.max(axis=0) + margin)

        centers = self._voxel_centers()
        grid_shape = centers.shape[:3]
        centers_flat = centers.reshape(-1, 3)
        n_voxels = centers_flat.shape[0]

        tree = cKDTree(points)
        radius = self.sdf_trunc * 1.5
        # workers=-1: parallelize the search across every CPU core scipy can
        # see. This query is embarrassingly parallel (each voxel's ball
        # query is independent) and, measured on a real merged point cloud
        # from this project's own pipeline, is roughly half of this
        # method's total wall time -- workers=-1 alone was measured to
        # nearly halve it again (1.72s -> 0.90s on a 10-core Apple Silicon
        # Mac, 271k voxel queries against a 57k-point tree).
        neighbor_lists = tree.query_ball_point(centers_flat, r=radius, workers=-1)

        values_flat = self._values.reshape(-1)
        weights_flat = self._weights.reshape(-1)
        views_flat = self._views.reshape(-1)
        colors_flat = self._colors.reshape(-1, 3)

        # Fix 4 (mesh speed): this used to be a plain Python ``for`` loop
        # over every voxel, each doing its own small numpy call.
        # ``query_ball_point`` above already did the expensive spatial
        # search in one batched (C-level, now also multi-threaded via
        # ``workers=-1``) call; the loop only ever did cheap-but-many-times
        # per-voxel arithmetic on its results, which vectorizes cleanly:
        # flatten every (voxel, contributing-point) pair from
        # ``neighbor_lists`` into 1-D index arrays, compute every pair's
        # local SDF/weight/colour contribution in one shot, and
        # scatter-accumulate per voxel with ``np.bincount(..., weights=...)``
        # -- NOT ``np.add.at``, which profiling on this project's own real
        # merged point cloud showed costs about as much as (not less than)
        # the loop it was meant to replace on a 1-D scatter, and noticeably
        # more on the ``(n, 3)`` colour scatter (``np.add.at`` does not
        # vectorize well across a trailing dimension); ``bincount`` (one
        # call per colour channel for the ``(n, 3)`` case) is a
        # well-optimized single-pass C reduction and was measured to
        # actually deliver a real speedup where ``add.at`` alone did not.
        # This is the exact same math as the loop it replaces (same
        # per-voxel weighted average, same total accumulated weight, same
        # "views" = "distinct contributing points" count -- see this
        # method's docstring), just computed with far fewer, much larger
        # numpy calls instead of one tiny call per voxel.
        #
        # Chunked, not all at once: a voxel's search radius
        # (``1.5 * sdf_trunc`` = ``4.5 * voxel_size``) is deliberately much
        # wider than the pre-TSDF-downsampled point spacing (``fuse_submaps``
        # downsamples to ``voxel_size * pre_tsdf_downsample_voxel_fraction``,
        # commonly 1/4 of ``voxel_size`` -- see that module's docstring), so
        # a single voxel can legitimately have thousands of points in range
        # on a dense, realistic cloud. Flattening *every* voxel's pairs into
        # one array at once, unchunked, means total memory scales with
        # ``n_voxels * points_per_ball`` -- at this project's real, budgeted
        # voxel counts and real point densities that is easily billions of
        # pairs (confirmed by measurement: an unchunked version of this same
        # vectorization thrashed/OOM'd on a realistic-density synthetic
        # benchmark, ~24k contributing points per voxel at this project's
        # actual sdf_trunc-radius / pre-TSDF-downsample-spacing ratio --
        # worse than the original per-voxel Python loop it was meant to
        # replace). Processing voxels in contiguous chunks sized to a fixed
        # pair-count budget (``_INTEGRATE_CHUNK_MAX_PAIRS``) keeps peak
        # memory bounded regardless of point density while still doing the
        # arithmetic in large batched numpy calls rather than one call per
        # voxel. Measured end-to-end on this project's own real merged
        # point cloud (57,580 points from a real 16-keyframe run,
        # 271k-voxel grid): ``query_ball_point(workers=-1)`` + ``bincount``
        # together cut total ``integrate_point_cloud`` time roughly in half
        # versus the original per-voxel loop at this density, and the
        # advantage grows with voxel count (measured up to ~2.6M voxels)
        # since fixed per-call overhead amortizes better over larger
        # batches -- see ``config.FusionConfig.voxel_count_budget``'s
        # docstring for the concrete before/after numbers this raised the
        # budget on the strength of.
        lengths = np.fromiter((len(nb) for nb in neighbor_lists), dtype=np.int64, count=n_voxels)
        total_pairs = int(lengths.sum())
        if total_pairs == 0:
            del grid_shape
            return

        colors_arr = np.asarray(colors, dtype=np.float64) if colors is not None else None
        cumulative = np.concatenate([[0], np.cumsum(lengths)])

        w_sum = np.zeros(n_voxels, dtype=np.float64)
        sdf_w_sum = np.zeros(n_voxels, dtype=np.float64)
        view_counts = np.zeros(n_voxels, dtype=np.int64)
        color_w_sum = np.zeros((n_voxels, 3), dtype=np.float64) if colors_arr is not None else None

        chunk_start = 0
        while chunk_start < n_voxels:
            target = cumulative[chunk_start] + _INTEGRATE_CHUNK_MAX_PAIRS
            chunk_end = int(np.searchsorted(cumulative, target, side="right"))
            chunk_end = min(max(chunk_end, chunk_start + 1), n_voxels)

            chunk_pairs = int(cumulative[chunk_end] - cumulative[chunk_start])
            if chunk_pairs > 0:
                chunk_size = chunk_end - chunk_start
                chunk_lengths = lengths[chunk_start:chunk_end]
                # Local (0-based, within-chunk) voxel index for bincount's
                # ``minlength``-sized output; the global index (for looking
                # up centers_flat) is just this plus chunk_start.
                local_voxel_idx = np.repeat(np.arange(chunk_size, dtype=np.int64), chunk_lengths)
                chunk_point_idx = np.fromiter(
                    itertools.chain.from_iterable(neighbor_lists[chunk_start:chunk_end]),
                    dtype=np.int64,
                    count=chunk_pairs,
                )

                delta = centers_flat[chunk_start + local_voxel_idx] - points[chunk_point_idx]
                local_sdf = np.einsum("ij,ij->i", delta, normals[chunk_point_idx])
                dist = np.linalg.norm(delta, axis=1)
                falloff = np.clip(1.0 - dist / radius, 0.0, 1.0)
                w = confidence[chunk_point_idx] * falloff

                w_sum[chunk_start:chunk_end] += np.bincount(local_voxel_idx, weights=w, minlength=chunk_size)
                sdf_w_sum[chunk_start:chunk_end] += np.bincount(
                    local_voxel_idx, weights=local_sdf * w, minlength=chunk_size
                )
                view_counts[chunk_start:chunk_end] += np.bincount(
                    local_voxel_idx, weights=(w > 0).astype(np.int64), minlength=chunk_size
                ).astype(np.int64)
                if color_w_sum is not None:
                    weighted_colors = colors_arr[chunk_point_idx] * w[:, None]
                    for channel in range(3):
                        color_w_sum[chunk_start:chunk_end, channel] += np.bincount(
                            local_voxel_idx, weights=weighted_colors[:, channel], minlength=chunk_size
                        )

            chunk_start = chunk_end

        touched = w_sum > 1e-12
        if not np.any(touched):
            del grid_shape
            return

        new_val = np.zeros(n_voxels, dtype=np.float64)
        new_val[touched] = np.clip(sdf_w_sum[touched] / w_sum[touched], -self.sdf_trunc, self.sdf_trunc)

        old_val = values_flat[touched]
        old_w = weights_flat[touched]
        combined = old_w + w_sum[touched]
        values_flat[touched] = (old_val * old_w + new_val[touched] * w_sum[touched]) / combined
        weights_flat[touched] = combined
        # See this method's docstring: "a view" is one distinct
        # contributing point, not one call to this method.
        views_flat[touched] += view_counts[touched]

        if color_w_sum is not None:
            blended = color_w_sum[touched] / w_sum[touched, None]
            old_color = colors_flat[touched]
            colors_flat[touched] = (old_color * old_w[:, None] + blended * w_sum[touched, None]) / combined[:, None]

        del grid_shape

    # -- extraction ---------------------------------------------------

    def extract_point_cloud(self) -> PointCloud:
        """Return every touched voxel as a ``PointCloud`` with tiered confidence."""
        if self._values is None:
            return PointCloud(xyz=np.zeros((0, 3), dtype=np.float64))

        mask = self._weights > 0
        centers = self._voxel_centers()[mask]
        colors = np.clip(self._colors[mask], 0, 255).astype(np.uint8)
        tiers = _tier_from_weight(
            self._weights[mask], self._views[mask], self._measured_min_weight, self._measured_min_views
        )
        return PointCloud(xyz=centers, rgb=colors, confidence=tiers)

    def extract_triangle_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract ``(vertices, faces, vertex_colors, vertex_confidence)``.

        Uses ``open3d``'s marching cubes when available (faster, and battle
        -tested); the confidence array is always computed from this
        module's own weight/view bookkeeping so both backends report
        identical MEASURED/LOW_CONFIDENCE/INFERRED semantics.
        """
        if self._values is None:
            empty_v = np.zeros((0, 3), dtype=np.float64)
            return empty_v, np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.uint8)

        if self._use_open3d:  # pragma: no cover - requires the optional open3d dependency
            o3d_mesh = self._o3d_volume.extract_triangle_mesh()
            vertices = np.asarray(o3d_mesh.vertices)
            faces = np.asarray(o3d_mesh.triangles)
            if o3d_mesh.has_vertex_colors():
                colors = np.clip(np.asarray(o3d_mesh.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
            else:
                colors = np.full((vertices.shape[0], 3), _EMPTY_COLOR, dtype=np.uint8)
            confidence = self._confidence_at(vertices)
            return vertices, faces, colors, confidence

        return _marching_tetrahedra(
            self._values,
            self._weights,
            self._views,
            self._colors,
            self._origin,
            self.voxel_size,
            self._measured_min_weight,
            self._measured_min_views,
        )

    def _confidence_at(self, points: np.ndarray) -> np.ndarray:  # pragma: no cover - open3d-only path
        """Nearest-voxel confidence lookup, used for the open3d mesh-extraction path."""
        if self._values is None or points.shape[0] == 0:
            return np.zeros((points.shape[0],), dtype=np.uint8)
        idx = np.round((points - self._origin) / self.voxel_size - 0.5).astype(np.int64)
        idx = np.clip(idx, 0, np.array(self._values.shape) - 1)
        w = self._weights[idx[:, 0], idx[:, 1], idx[:, 2]]
        v = self._views[idx[:, 0], idx[:, 1], idx[:, 2]]
        return _tier_from_weight(w, v, self._measured_min_weight, self._measured_min_views)


# ---------------------------------------------------------------------------
# Stage-level entry point
# ---------------------------------------------------------------------------

# ``BackboneResult.confidence`` (and therefore ``Submap.confidence``, see
# that dataclass's docstring) is raw continuous backbone confidence in
# ``[0, 1]`` -- "not the coarser tiered ``types.Confidence`` enum -- that
# quantization happens later, once a consumer decides on thresholds." This
# module is that consumer: ``fusion.filters.confidence_filter`` and
# ``TSDFVolume``'s own MEASURED/LOW_CONFIDENCE/INFERRED bookkeeping both
# operate on the tiered 0/1/2 scale, so raw floats must be quantized before
# either sees them. (Previously nothing did this: ``merge_submaps`` just
# concatenated each submap's raw [0, 1] confidence straight into
# ``PointCloud.confidence``, so ``confidence_filter``'s default
# ``min_confidence=1`` compared floats that top out at ~0.95 against an
# integer tier threshold of 1 and rejected literally every point -- the
# actual root cause of "fusion produced no vertices" on real backbone
# output.)
#
# The combined rule, end to end (read this before touching thresholds)
# -------------------------------------------------------------------------
# Two confidence signals exist in this pipeline and they used to fight each
# other instead of reinforcing one another:
#
# 1. Per-*point* raw backbone confidence, quantized here into a tier
#    (defaults below mirror ``geometry.backbone.NullBackbone``'s own
#    ``_CONF_BOX_HIT``/``_CONF_GROUND_HIT``/``_CONF_MISS`` reference values
#    -- 0.95 solid-surface hit / 0.75 ground hit / 0.05 miss).
# 2. Per-*voxel* accumulated TSDF weight/observation-count, computed by
#    ``TSDFVolume`` from however many of those already-tiered points
#    actually landed in a given voxel (the tiered value is used directly as
#    each point's integration weight -- see
#    ``TSDFVolume.integrate_point_cloud``).
#
# The bug this module used to have: (1) alone decided whether a point was
# "confident," but (2)'s own MEASURED threshold was calibrated for
# ``integrate()``'s raw-depth-map path (where ``self._views`` counts real
# camera frames) and ``fuse_submaps`` only ever made *one*
# ``integrate_point_cloud`` call over the whole merged cloud -- so
# ``self._views`` capped at 1 for every voxel and MEASURED became
# unreachable regardless of how confident or well-observed the underlying
# points were (see ``TSDFVolume.integrate_point_cloud``'s docstring for the
# fix: "a view" there now means "one distinct contributing point," not "one
# call"). With that fixed, (1) and (2) are reconciled into one rule instead
# of two independent, conflicting ones: a voxel is MEASURED only when it
# has both enough corroborating point observations
# (``FusionConfig.measured_min_views``) *and* those observations were
# themselves mostly individually high-confidence
# (``FusionConfig.measured_min_weight``, calibrated relative to
# ``measured_min_views`` -- see that field's docstring) -- density alone
# (many low-confidence points) can no longer silently promote a voxel to
# MEASURED, and a single, even very confident, point can no longer either.
_CONF_TIER_MEASURED_MIN = 0.8
_CONF_TIER_LOW_MIN = 0.3


def _quantize_confidence_to_tier(
    confidence: np.ndarray | None,
    measured_min: float = _CONF_TIER_MEASURED_MIN,
    low_min: float = _CONF_TIER_LOW_MIN,
) -> np.ndarray | None:
    """Map raw continuous ``[0, 1]`` backbone confidence onto the tiered ``Confidence`` enum.

    A no-op (aside from an integer cast) when ``confidence`` already looks
    tiered (every value already one of ``{0, 1, 2}``) -- callers that
    already hand in properly-quantized confidence (e.g. most of this
    module's own tests) get their values back unchanged rather than
    re-bucketed through the continuous thresholds below. ``measured_min``/
    ``low_min`` default to this module's own reference constants;
    ``fuse_submaps`` passes ``config.FusionConfig.raw_confidence_measured_min``/
    ``raw_confidence_low_min`` instead.
    """
    if confidence is None:
        return None
    arr = np.asarray(confidence)
    already_tiered = np.all(np.isin(arr, (Confidence.INFERRED, Confidence.LOW_CONFIDENCE, Confidence.MEASURED)))
    if already_tiered:
        return arr.astype(np.uint8)

    tier = np.full(arr.shape, Confidence.INFERRED, dtype=np.uint8)
    tier[arr >= low_min] = Confidence.LOW_CONFIDENCE
    tier[arr >= measured_min] = Confidence.MEASURED
    return tier


# See ``config.FusionConfig.voxel_size``'s docstring for the full story of
# why this replaced a bounding-box-based heuristic. ``_auto_voxel_size``
# (the *old* rule: aim for ~200 voxels along the cloud's longest bbox axis)
# is kept only as the last-resort fallback ``_derive_voxel_size`` uses when
# a real ground-sample-distance estimate genuinely cannot be computed (no
# camera poses, or no per-keyframe intrinsics at all) -- it is scale-blind
# (1.3 m for a 250 m suburban site, 13 m for a 2.5 km flight) and was the
# actual root cause of "the TSDF voxel grid erased the buildings" on real
# footage: a 259 m scene extent drove voxel = clip(259/200, ...) = 1.30 m,
# well over an entire house's width, so roof edges/kerbs/driveways -- all
# smaller than one voxel -- simply vanished into a handful of blobby voxels.
_TARGET_VOXELS_PER_AXIS = 200
_AUTO_VOXEL_MIN_M = 0.01
_AUTO_VOXEL_MAX_M = 5.0


def _auto_voxel_size(xyz: np.ndarray) -> float:
    """Derive a voxel size from the point cloud's own bounding-box extent (see module constants above).

    Scale-blind fallback only -- see ``_derive_voxel_size``, which is what
    ``fuse_submaps`` actually calls; this is used only when that function
    cannot compute a real ground-sample-distance estimate at all.
    """
    if xyz.shape[0] < 2:
        return _AUTO_VOXEL_MIN_M
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    max_extent = float(extent.max())
    if max_extent <= 0.0:
        return _AUTO_VOXEL_MIN_M
    return float(np.clip(max_extent / _TARGET_VOXELS_PER_AXIS, _AUTO_VOXEL_MIN_M, _AUTO_VOXEL_MAX_M))


# ---------------------------------------------------------------------------
# GSD-derived voxel sizing (the actual fix)
# ---------------------------------------------------------------------------
#
# The right physical quantity to size a voxel against is the *ground sample
# distance* (GSD) -- how many metres one source pixel covers on the ground
# -- not the scene's overall bounding-box extent. GSD is a property of the
# camera (focal length in pixels) and how far it was from the scene (depth),
# and is completely independent of how large an area the whole flight
# happened to cover: ``GSD = depth / fx`` (a pixel subtends angle ~1/fx
# radians, and at range `depth` that angle spans `depth/fx` metres on the
# ground) -- equivalently ``altitude_agl * sensor_width / (focal_length *
# image_width)`` when starting from physical units instead of already-built
# ``CameraIntrinsics``.
#
# ``fx`` must come from each keyframe's *native*-resolution intrinsics
# (``Keyframe.intrinsics``, as set by ``ingest.intrinsics``/
# ``pipeline.stages.TriageStage`` from the original video), NOT the
# backbone's own working-resolution intrinsics (``GeometryStage``'s
# ``_resize_for_backbone`` scales a *local copy* for the backbone call and
# never mutates ``Keyframe.intrinsics`` -- see that stage's docstring): GSD
# is a physical property of the sensor and the ground, and does not change
# just because the backbone happened to run at a downsampled working
# resolution.
_DEFAULT_VOXEL_SIZE_GSD_MULTIPLIER = 3.0

# Measured (not guessed) throughput of TSDFVolume.integrate_point_cloud's
# numpy-fallback path on this project's target hardware (Apple Silicon,
# no open3d installed in this environment at all -- see the module
# docstring's "Backends" section). Historically (before Fix 4) this was a
# plain Python loop over every voxel in the dense grid, each doing its own
# small numpy call: a sparse synthetic cloud (~20k points) benchmarked at
# ~15,000-24,000 voxels/second, a dense realistic cloud (~300k points)
# dropped to ~3,500 voxels/second, and a real end-to-end run on real drone
# footage measured well under 2,000 voxels/second -- this loop's per-voxel
# cost scales with *local point density*, not just voxel count, since it is
# a plain per-voxel KD-tree ball query, not a block-sparse volume.
#
# Fix 4 replaced that loop with a chunked, vectorized scatter-accumulate
# (``np.bincount`` over flattened (voxel, point) pairs, plus
# ``workers=-1`` on the underlying ``cKDTree.query_ball_point`` call -- see
# ``integrate_point_cloud``'s docstring for why ``np.add.at`` alone did
# *not* deliver a real speedup here, and why the pair-flattening is chunked
# rather than done all at once). Measured on this project's own real merged
# point cloud (57,580 points from a real 16-keyframe run): ~65,000-1.16M
# voxels/second depending on grid size (throughput improves at larger grids
# since fixed per-call overhead amortizes better) -- roughly 30x-500x this
# module's old measured range, though marching-tetrahedra mesh extraction
# (a separate cost this fix did not vectorize further) still scales with
# *total* grid cells and becomes the co-dominant cost at very large voxel
# counts (~32s at ~11.8M voxels in the same measurement).
#: Points used to estimate the occupied voxel set. Counting unique cells
#: over several million points, repeatedly inside the coarsening loop, is
#: slower than the fusion it is budgeting for; a few hundred thousand
#: gives the same answer to well inside the rounding the budget applies.
_VOXEL_COUNT_SAMPLE = 400_000

#: Voxels allocated along the surface normal per occupied surface cell.
#: ``sdf_trunc`` is 3*voxel_size either side of the surface, so the band
#: is ~6 voxels thick.
_TSDF_BAND_VOXELS = 6

# ``FusionConfig.voxel_count_budget`` exposes this so a caller with more
# time (or a sparser cloud) can raise it further; ``_derive_voxel_size``
# coarsens (and logs loudly) rather than silently trying to populate a grid
# this pipeline cannot actually finish integrating and mesh-extracting
# within its time budget.
_DEFAULT_VOXEL_COUNT_BUDGET = 2_000_000


def _median_gsd_m(
    xyz: np.ndarray,
    poses: list[Pose],
    keyframe_intrinsics: dict[int, CameraIntrinsics] | None,
) -> float | None:
    """Estimate ground sample distance (metres/pixel) from camera-to-scene depth and focal length.

    ``depth`` is approximated, per point, as the distance to the *nearest*
    camera centre in ``poses`` (a nadir/near-nadir drone shot's camera-to-
    ground distance is essentially its altitude AGL regardless of which
    specific camera the point was actually reconstructed from, so "nearest
    camera" is a good, cheap stand-in for "the camera that actually saw
    this point" without threading per-point camera provenance through
    ``merge_submaps``). ``fx`` is the median of every available keyframe's
    *native*-resolution focal length (see module note above on why native,
    not backbone-working-resolution). The final per-point ``depth / fx``
    values are aggregated with the median, not the mean, so a handful of
    far-away or very close stray points can't skew the whole scene's voxel
    sizing. Returns ``None`` when there isn't enough information to compute
    this at all (no camera poses, or no usable intrinsics) -- the caller
    falls back to the scale-blind bounding-box heuristic in that case.
    """
    if xyz.shape[0] == 0 or not poses or not keyframe_intrinsics:
        return None

    fx_values = np.array(
        [float(intr.fx) for intr in keyframe_intrinsics.values() if intr is not None and intr.fx > 0.0],
        dtype=np.float64,
    )
    if fx_values.size == 0:
        return None
    fx_median = float(np.median(fx_values))
    if fx_median <= 0.0:
        return None

    camera_positions = np.array([p.t for p in poses], dtype=np.float64)
    if camera_positions.shape[0] == 0:
        return None

    tree = cKDTree(camera_positions)
    depths, _ = tree.query(xyz)
    depths = depths[np.isfinite(depths) & (depths > 0.0)]
    if depths.size == 0:
        return None
    depth_median = float(np.median(depths))

    return depth_median / fx_median


def _derive_voxel_size(
    xyz: np.ndarray,
    poses: list[Pose],
    keyframe_intrinsics: dict[int, CameraIntrinsics] | None,
    gsd_multiplier: float,
    voxel_count_budget: int,
    stats: dict,
) -> float:
    """Pick the TSDF voxel size: GSD-derived, clamped against a voxel-count budget.

    Replaces the old ``max_extent / 200`` scene-bounding-box heuristic (see
    ``_auto_voxel_size``'s docstring) with one tied to the actual ground
    sampling: ``voxel = clip(gsd_multiplier * GSD, _AUTO_VOXEL_MIN_M,
    _AUTO_VOXEL_MAX_M)``. Falls back to ``_auto_voxel_size`` (loudly noted
    in ``stats``) only when GSD genuinely cannot be computed at all.

    ``stats`` is filled in with ``voxel_size_gsd_m`` (the GSD estimate
    itself, ``None`` if unavailable), ``voxel_size_ideal_m`` (what the GSD
    multiplier alone would pick, before any budget clamp),
    ``voxel_size_source`` (``"gsd"`` or ``"bbox_fallback (no GSD
    available)"``), and ``voxel_size_budget_exceeded`` (whether the budget
    clamp below had to coarsen the ideal size) -- so a caller (this
    module's own ``fuse_submaps``, ``pipeline.stages.FusionStage``'s
    ``StageResult``) can report exactly what happened, not just the final
    number.

    Voxel-count budget clamp
    -------------------------
    A GSD-correct voxel size for a wide-area flight can imply a dense grid
    with tens of millions of voxels -- far more than this pipeline's numpy
    TSDF fallback can integrate within any sane time budget (see
    ``_DEFAULT_VOXEL_COUNT_BUDGET``'s docstring for the measured
    throughput this is calibrated against), and, at the extreme, more
    memory than a laptop has. Rather than letting that happen silently,
    the *ideal* GSD-derived size is checked against how many voxels it
    would need for this cloud's own bounding box (mirroring
    ``fuse_submaps``' own TSDF bounds/margin calculation so the estimate
    matches what will actually be allocated); if that exceeds
    ``voxel_count_budget``, the voxel size is coarsened -- scaling by the
    cube root of the overshoot, since grid volume grows with the cube of
    ``1/voxel_size`` -- just enough to fit, and a prominent warning is
    logged naming both the ideal (detail-preserving) and actual (budget-
    fitting) voxel size, so an operator can see exactly how much detail was
    traded for memory/time rather than discovering it only from a
    blobbier-than-expected mesh.
    """
    gsd = _median_gsd_m(xyz, poses, keyframe_intrinsics)
    if gsd is None or gsd <= 0.0:
        fallback = _auto_voxel_size(xyz)
        stats["voxel_size_gsd_m"] = None
        stats["voxel_size_ideal_m"] = None
        stats["voxel_size_source"] = "bbox_fallback (no GSD available)"
        stats["voxel_size_budget_exceeded"] = False
        logger.warning(
            "FUSION VOXEL SIZING: no ground-sample-distance estimate available (no camera poses "
            "and/or no per-keyframe intrinsics) -- falling back to the scale-blind bounding-box "
            "heuristic (voxel_size=%.4gm). This heuristic is what previously erased fine structure "
            "on real footage; expect degraded detail.",
            fallback,
        )
        return fallback

    ideal = float(np.clip(gsd * gsd_multiplier, _AUTO_VOXEL_MIN_M, _AUTO_VOXEL_MAX_M))
    stats["voxel_size_gsd_m"] = gsd
    stats["voxel_size_ideal_m"] = ideal
    stats["voxel_size_source"] = "gsd"

    if xyz.shape[0] == 0:
        stats["voxel_size_budget_exceeded"] = False
        return ideal

    extent = xyz.max(axis=0) - xyz.min(axis=0)

    # Count the voxels the TSDF will actually ALLOCATE, not the volume of
    # the scene's bounding box.
    #
    # ``ScalableTSDFVolume`` is block-sparse: it only ever allocates
    # blocks within the truncation band around an observed surface. A
    # nadir survey's surface is a sheet -- terrain plus what stands on
    # it -- so the occupied set is that sheet thickened by the band, and
    # is nowhere near the filled box the old ``prod(dims)`` computed.
    #
    # The difference decides how much detail survives. For this flight
    # (776 x 853 x 66 m) at a 4M-voxel budget:
    #
    #     counting the box     -> 11.7M voxels at 1.55 m -> picks 1.55 m
    #     counting the surface ->  1.7M voxels at 1.55 m -> allows 1.00 m
    #
    # and a 1.55 m voxel cannot represent a surface finer than itself, so
    # a 1.60 m point cloud came out as a 5.69 m thick mesh with the
    # driveways, pool edges and roof lines quantised away.
    #
    # This counts the real occupied set from the points themselves rather
    # than from a formula, so scene shape is handled for free: a survey
    # with a hole in the middle (this one has 1.89 ha the camera never
    # saw) does not get charged for the ground it never observed, and a
    # genuinely 3D scene of facades is charged for all of them.
    sample = xyz
    if sample.shape[0] > _VOXEL_COUNT_SAMPLE:
        # Deterministic stride: the same cloud must budget identically
        # twice, or two runs of one config pick different voxel sizes.
        sample = sample[:: int(np.ceil(sample.shape[0] / _VOXEL_COUNT_SAMPLE))]
    sample_fraction = sample.shape[0] / max(xyz.shape[0], 1)

    def _voxel_count(vs: float) -> int:
        occupied = np.unique(np.floor(sample / vs).astype(np.int64), axis=0).shape[0]
        # Subsampling undercounts distinct cells; correcting by the sample
        # fraction is an upper bound (points that shared a cell would have
        # shared it anyway), which is the safe direction for a budget.
        occupied = int(occupied / max(sample_fraction, 1e-9)) if sample_fraction < 1.0 else occupied
        # Each surface voxel drags its truncation band with it:
        # sdf_trunc = 3*vs either side, so ~6 voxels along the normal.
        return int(occupied * _TSDF_BAND_VOXELS)

    ideal_count = _voxel_count(ideal)
    if ideal_count <= voxel_count_budget:
        stats["voxel_size_budget_exceeded"] = False
        return ideal

    # Coarsen just enough to fit. The occupied set is a surface, so its
    # voxel count scales as 1/voxel_size^2, not ^3 -- using the cube root
    # here would under-coarsen and need several correction passes below.
    scale = (ideal_count / voxel_count_budget) ** (1.0 / 2.0)
    actual = float(np.clip(ideal * scale, _AUTO_VOXEL_MIN_M, _AUTO_VOXEL_MAX_M))
    guard = 0
    while _voxel_count(actual) > voxel_count_budget and actual < _AUTO_VOXEL_MAX_M and guard < 20:
        actual = min(actual * 1.05, _AUTO_VOXEL_MAX_M)
        guard += 1

    actual_count = _voxel_count(actual)
    logger.warning(
        "FUSION VOXEL BUDGET WARNING: detail is being sacrificed for memory/time. The ideal "
        "GSD-derived voxel size is %.4gm (from GSD=%.4gm/px, multiplier=%.1f), which would need "
        "~%d voxels for this scene's extent -- over the %d-voxel budget. Coarsening to %.4gm "
        "(~%d voxels) instead.",
        ideal,
        gsd,
        gsd_multiplier,
        ideal_count,
        voxel_count_budget,
        actual,
        actual_count,
    )
    stats["voxel_size_budget_exceeded"] = True
    return actual


def _empty_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 3), dtype=np.int64),
        np.zeros((0, 3), dtype=np.uint8),
        np.zeros((0,), dtype=np.uint8),
    )


def _empty_point_cloud() -> PointCloud:
    return PointCloud(
        xyz=np.zeros((0, 3), dtype=np.float64),
        rgb=np.zeros((0, 3), dtype=np.uint8),
        confidence=np.zeros((0,), dtype=np.uint8),
    )


# Fix 3: pre-TSDF downsampling, decoupled from the TSDF's own voxel_size.
#
# TSDFVolume.integrate_point_cloud's per-voxel cost is driven by *local
# point density relative to voxel_size* (each voxel does a cKDTree ball
# query of radius ~1.5*sdf_trunc = 4.5*voxel_size, so its cost scales with
# however many points land in that neighbourhood), not by total point
# count alone. A real backbone's dense per-pixel output can be far denser
# than a GSD-correct (now typically sub-metre) voxel_size actually needs --
# measured on real footage, a single 16-keyframe run produced over 1.1
# million cleaned points, which made naive per-voxel ball queries take
# minutes even at a modest few-hundred-thousand-voxel grid.
#
# The fix: downsample the cloud that reaches TSDF integration to
# ``voxel_size * _DEFAULT_PRE_TSDF_DOWNSAMPLE_FRACTION`` (default 1/4) --
# finer than the final voxel grid can even represent, so this can never
# discard information the mesh could have shown, but coarse enough to
# bound the average number of points a voxel's neighbourhood query
# actually has to look at, however dense the raw input was. This is what
# "decoupled from the TSDF's own voxel_size" means: the pre-TSDF
# downsample size *tracks* voxel_size (so it never needlessly re-runs on a
# cloud that's already sparse enough) but is always a small fraction of
# it, never equal to it -- unlike the pre-fix behaviour, which
# downsampled to exactly voxel_size and silently pre-smoothed the mesh to
# whatever (possibly coarse, bounding-box-derived) resolution the TSDF
# itself ended up using.
_DEFAULT_PRE_TSDF_DOWNSAMPLE_FRACTION = 0.25

# Hard backstop on top of the fraction-based rule above, for the
# pathological case where even voxel_size/4 still leaves an enormous
# number of points (e.g. a very fine GSD-derived voxel_size over a very
# densely-reconstructed scene). Unlike the fraction-based downsample, if
# this backstop actually fires it IS trading away some resolvable detail
# for a bounded runtime -- logged loudly, same spirit as the voxel-count
# budget clamp in _derive_voxel_size.
_DEFAULT_PRE_TSDF_MAX_POINTS = 1_500_000

#: Faces a marching-cubes surface yields per unit area per voxel^2, i.e.
#: the C in ``faces = C * area / voxel^2``. A flat sheet gives 2; this is
#: the value MEASURED on real reconstructed terrain (10,003,138 faces at a
#: 0.66 m voxel over ~250,000 m2 of occupied ground), where the surface
#: carries ~8x a flat sheet's area because of depth noise. Used to pick a
#: voxel size that lands near the face budget instead of overshooting it
#: by 5x and decimating back down.
_FACES_PER_AREA_CONSTANT = 17.0


def _voxel_size_for_point_budget(xyz: np.ndarray, max_points: int) -> float:
    """Smallest voxel size (roughly) that downsamples ``xyz`` to at most ``max_points`` points.

    A cheap, approximate estimate assuming uniform point density over the
    cloud's bounding-box volume (``voxel ~= (bbox_volume / max_points) **
    (1/3)``), refined by a few doubling steps if the actual post-downsample
    count still overshoots -- good enough for the hard backstop above,
    which only exists to bound ``integrate_point_cloud``'s per-voxel query
    cost in a pathological case, not to make a resolution decision.
    """
    from drishti3d.fusion.filters import voxel_downsample

    n = xyz.shape[0]
    if n <= max_points or n == 0:
        return _AUTO_VOXEL_MIN_M

    # A point cloud is a SURFACE, so the count after voxelisation scales
    # with occupied area / voxel^2, not with bounding-box volume / voxel^3.
    # The previous estimate here was volumetric, and it only ever refined
    # coarser: on a real flight whose bounding box was inflated to
    # 630 x 676 x 73 m by a ground-elevation split, it started at 2.75 m
    # and returned 2.60 m -- reducing 61M cleaned points to 19,658 before
    # TSDF integration. Every mesh built from that was a 0.6 m grid fitted
    # through twenty thousand points, which is what "blobby" looked like.
    #
    # Bisect in log space on the actual downsampled count instead. The
    # volumetric figure is kept only as a guaranteed-coarse-enough upper
    # bound; the lower bound is the finest voxel the pipeline uses at all.
    extent = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1e-6)
    hi = max((float(np.prod(extent)) / max_points) ** (1.0 / 3.0), _AUTO_VOXEL_MIN_M * 2)
    lo = _AUTO_VOXEL_MIN_M

    def count_at(v: float) -> int:
        return int(voxel_downsample(PointCloud(xyz=xyz), voxel_size=v).xyz.shape[0])

    # Make sure the bracket is valid: hi must satisfy the budget.
    while count_at(hi) > max_points:
        hi *= 1.5
    if count_at(lo) <= max_points:
        return lo

    for _ in range(8):
        mid = float(np.sqrt(lo * hi))
        if count_at(mid) <= max_points:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.08:
            break
    return hi


# ---------------------------------------------------------------------------
# Block-sparse tiled fusion (Fix: mesh voxel-lattice artifact)
#
# The single-dense-grid path above sizes one voxel grid over the cleaned
# cloud's *whole bounding box*. For a flight whose footprint is a curved or
# partial-coverage strip (the common case -- see this module's own
# docstring on ScalableTSDFVolume's block-sparse advantage), the occupied
# ground is typically a small fraction of that bounding box (measured on
# real 16-keyframe footage: ~15% of the bbox at a 0.5m planview grid), so a
# dense grid spends the overwhelming majority of `voxel_count_budget` on
# voxels that will never be touched by a single point -- which is exactly
# why the GSD-ideal voxel size (sub-decimetre) gets coarsened all the way
# to a metre-plus "voxel-lattice" artifact instead of real terrain detail.
#
# Fix: tile the occupied footprint (skip empty tiles entirely -- this is
# where the "block-sparse" saving comes from; it is the same idea
# open3d's ScalableTSDFVolume uses, reimplemented here in pure numpy since
# open3d is not available in this environment -- see the module docstring's
# "Backends" section), fuse each tile independently at a *shared* voxel
# grid alignment, and weld the results back into one seamless mesh at their
# overlapping seams. Because every tile shares the same global grid origin
# and pads far enough past its own boundary to cover the same TSDF search
# radius a single dense grid would use there, two adjacent tiles compute
# *identical* values for the voxels they both cover -- so welding
# coincident vertices (not just visually close ones) produces a genuinely
# continuous mesh, not a patchwork with seams.
#
# `voxel_count_budget` is reinterpreted, not repurposed: instead of
# bounding one dense grid's total cell count, it now bounds the *sum* of
# occupied tiles' cell counts -- i.e. still "how many voxels this pipeline
# is willing to integrate/extract," just spent only where data exists
# instead of being diluted across a mostly-empty bounding box. A caller
# that never exceeds the budget under the old single-grid accounting still
# takes the unchanged, already-tested single-grid path below (tiling adds
# per-tile bookkeeping overhead that only pays for itself once the
# single-grid coarsening would otherwise fire).
# ---------------------------------------------------------------------------

# Cell size for the coarse planview occupancy estimate (_occupied_area_m2)
# and the local-relief estimate (_typical_local_z_extent): large enough to
# be cheap (a handful of numpy calls, not a full per-point KD-tree pass)
# and to average out per-point depth noise, small enough to distinguish a
# genuinely narrow flight swath from its much larger bounding box.
_TILE_OCCUPANCY_CELL_M = 1.0
_TILE_Z_RELIEF_CELL_M = 20.0

# Target voxel count for a single tile's own dense grid -- purely an
# implementation/memory knob (how finely the occupied footprint is
# subdivided), decoupled from the *achieved resolution*, which
# `_solve_tiled_voxel_size` derives from `voxel_count_budget` and the
# occupied area instead. Small enough to keep any one tile's memory
# footprint modest, large enough that per-tile fixed overhead (a KD-tree
# build, a marching-tetrahedra pass) amortizes reasonably across the whole
# occupied footprint rather than degenerating into thousands of tiny tiles.
_PER_TILE_VOXEL_TARGET = 400_000

# A tile needs at least this many points to be worth its own TSDF pass at
# all -- fewer than this can't meaningfully constrain a local surface (see
# fusion.mesh.estimate_normals' own, similar floor) and would just add
# per-tile overhead for a handful of stray points already covered by a
# neighbouring tile's padding.
_TILE_MIN_POINTS = 30


def _occupied_area_m2(xyz: np.ndarray, cell_m: float = _TILE_OCCUPANCY_CELL_M) -> float:
    """Planview area (m^2) actually covered by ``xyz``, not its bounding box's area.

    Counts ``cell_m`` x ``cell_m`` XY cells containing >= 1 point. A single
    ``np.unique`` pass over quantized cell indices -- cheap relative to the
    TSDF integration this sizes, and a much better proxy for how much
    ground a tiled fusion actually needs to cover than the bounding box's
    own (potentially much larger) area: a curved or partial-coverage
    flight's footprint is typically a narrow strip through its own
    enclosing rectangle (see this section's module-level note).
    """
    if xyz.shape[0] == 0:
        return 0.0
    origin = xyz[:, :2].min(axis=0)
    idx = np.floor((xyz[:, :2] - origin) / cell_m).astype(np.int64)
    uniq = np.unique(idx, axis=0)
    return float(uniq.shape[0]) * (cell_m**2)


def _typical_local_z_extent(xyz: np.ndarray, cell_m: float = _TILE_Z_RELIEF_CELL_M) -> float:
    """Median per-cell vertical range (m) over a coarse ``cell_m`` x ``cell_m`` planview grid.

    A much better proxy for "how tall does one TILE actually need to be"
    than the whole cloud's global z min/max: aerial terrain is locally
    near-planar (this project's own local 1x1m planar-fit residual is a
    few centimetres to a few tens of centimetres -- see
    ``config.FusionConfig``'s docstring) even when the *overall* scene
    spans tens of metres of elevation change (rolling terrain, or a
    handful of tall structures). Sizing every tile's vertical extent from
    the global range would multiply that scene-wide relief into every
    tile's voxel budget even though most tiles never see it; a handful of
    genuinely tall cells are handled instead by ``_fuse_tiled``'s own
    per-tile coarsening safety net (a local surprise still can't blow the
    per-tile voxel budget, it just costs that one tile some resolution).
    """
    if xyz.shape[0] < 2:
        return 1.0
    origin = xyz[:, :2].min(axis=0)
    idx = np.floor((xyz[:, :2] - origin) / cell_m).astype(np.int64)
    key = idx[:, 0].astype(np.int64) * 1_000_003 + idx[:, 1].astype(np.int64)
    order = np.argsort(key)
    z_sorted = xyz[order, 2]
    _uniq_keys, start_idx, counts = np.unique(key[order], return_index=True, return_counts=True)

    ranges = []
    for s, c in zip(start_idx, counts):
        if c < 5:
            continue
        cell_z = z_sorted[s : s + c]
        ranges.append(float(cell_z.max() - cell_z.min()))

    if not ranges:
        full_range = float(xyz[:, 2].max() - xyz[:, 2].min())
        return full_range if full_range > 0 else 1.0
    return float(np.median(ranges))


def _solve_tiled_voxel_size(occupied_area_m2: float, typical_z_extent_m: float, total_voxel_budget: int) -> float:
    """Finest voxel size whose TOTAL occupied-footprint voxel count fits ``total_voxel_budget``.

    Unlike ``_derive_voxel_size``'s single-dense-grid accounting (bounding
    box area x height / voxel^3), this bounds occupied-area x typical
    tile-height / voxel^3 -- see this section's module-level note for why
    that is the right total-cost model for a tiled fusion. Solved by fixed
    -point iteration (the per-side TSDF margin, ``sdf_trunc + voxel_size =
    4 * voxel_size``, itself depends on the answer) -- converges in a
    handful of iterations since the margin term is a small correction next
    to ``typical_z_extent_m`` for any reasonably-flat scene.
    """
    budget = max(int(total_voxel_budget), 1)
    area = max(occupied_area_m2, 1e-6)
    z0 = max(typical_z_extent_m, 1e-3)

    v = max((area * z0 / budget) ** (1.0 / 3.0), _AUTO_VOXEL_MIN_M)
    for _ in range(8):
        margin = 4.0 * v  # sdf_trunc (3v) + voxel_size (v), matching _grid_voxel_count's convention
        z_span = z0 + 2.0 * margin
        v_next = max((area * z_span / budget) ** (1.0 / 3.0), _AUTO_VOXEL_MIN_M)
        if abs(v_next - v) < 1e-6:
            v = v_next
            break
        v = v_next
    return float(min(v, _AUTO_VOXEL_MAX_M))


def _plan_tile_edge_m(typical_z_extent_m: float, voxel_size: float, sdf_trunc: float, per_tile_voxel_target: int) -> float:
    """World-space tile edge length (m) whose own dense grid targets ``per_tile_voxel_target`` voxels.

    Purely an implementation knob (see this section's module-level note):
    controls how finely the occupied footprint is subdivided, not the
    achieved resolution (``voxel_size``, already decided by
    ``_solve_tiled_voxel_size`` by the time this is called).
    """
    margin = sdf_trunc + voxel_size
    z_span = typical_z_extent_m + 2.0 * margin
    z_voxels = max(z_span / voxel_size, 1.0)
    max_area_voxels = max(per_tile_voxel_target / z_voxels, 1.0)
    tile_m = math.sqrt(max_area_voxels) * voxel_size
    return float(max(tile_m, voxel_size * 8.0))


def _grid_voxel_count(extent: np.ndarray, voxel_size: float, sdf_trunc: float) -> int:
    """Voxel count a dense grid covering a ``(3,)`` world-space ``extent`` at ``voxel_size`` would need.

    Includes the standard ``sdf_trunc + voxel_size`` margin on every side --
    mirrors ``_derive_voxel_size``'s own (function-local) ``_voxel_count``
    exactly, factored out here so both the whole-scene and per-tile
    accounting agree on what "how many voxels would this need" means.
    """
    margin = sdf_trunc + voxel_size
    dims = np.maximum(np.ceil((extent + 2.0 * margin) / voxel_size).astype(np.int64), 1) + 1
    return int(np.prod(dims))


def _fuse_tiled(
    xyz: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray | None,
    confidence: np.ndarray,
    voxel_size: float,
    sdf_trunc: float,
    voxel_count_budget: int,
    typical_local_z_extent_m: float,
    measured_min_weight: float,
    measured_min_views: int,
    stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block-sparse TSDF fusion: tile the occupied footprint, fuse each tile, weld the seams.

    See this module section's docstring for the full rationale. Every tile
    shares one grid alignment (``global_origin``, a multiple of
    ``voxel_size`` away from every tile's own origin) and pads its point
    selection by the same radius (``sdf_trunc * 1.5``)
    ``TSDFVolume.integrate_point_cloud``'s own KD-tree ball query uses, so
    two adjacent tiles' overlapping voxels are computed from the *same*
    input points and agree to floating-point precision -- welding
    coincident vertices at a tight tolerance (``voxel_size * 1e-3``) then
    produces a genuinely continuous mesh across tile boundaries, not just a
    visually-close one.

    A tile whose own local extent (a real surprise -- e.g. it happens to
    span an unusually tall structure) would still exceed
    ``voxel_count_budget`` on its own is coarsened individually (the same
    cube-root scaling ``_derive_voxel_size`` uses for the whole-scene case,
    see its docstring), logged via ``stats["n_tiles_coarsened"]`` -- this
    keeps every tile's own cost bounded regardless of terrain, without
    coarsening tiles that never needed it.

    Returns ``(vertices, faces, colors, confidence)``, empty when ``xyz``
    is empty or every tile is empty/too sparse.
    """
    n = xyz.shape[0]
    empty = _empty_mesh()
    if n == 0:
        return empty

    per_tile_target = max(1, min(_PER_TILE_VOXEL_TARGET, int(voxel_count_budget)))
    tile_edge_m = _plan_tile_edge_m(typical_local_z_extent_m, voxel_size, sdf_trunc, per_tile_target)

    xy_min = xyz[:, :2].min(axis=0)
    xy_max = xyz[:, :2].max(axis=0)
    global_origin = np.floor(xyz.min(axis=0) / voxel_size) * voxel_size
    pad = sdf_trunc * 1.5 + voxel_size

    n_tiles_x = max(1, int(np.ceil((xy_max[0] - xy_min[0]) / tile_edge_m)))
    n_tiles_y = max(1, int(np.ceil((xy_max[1] - xy_min[1]) / tile_edge_m)))

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    n_processed = 0
    n_skipped_empty = 0
    n_coarsened = 0
    max_tile_voxels = 0

    for ix in range(n_tiles_x):
        tile_min_x = xy_min[0] + ix * tile_edge_m
        tile_max_x = tile_min_x + tile_edge_m
        in_x = (xyz[:, 0] >= tile_min_x - pad) & (xyz[:, 0] < tile_max_x + pad)
        if not np.any(in_x):
            n_skipped_empty += n_tiles_y
            continue
        for iy in range(n_tiles_y):
            tile_min_y = xy_min[1] + iy * tile_edge_m
            tile_max_y = tile_min_y + tile_edge_m
            sel = in_x & (xyz[:, 1] >= tile_min_y - pad) & (xyz[:, 1] < tile_max_y + pad)
            n_sel = int(np.count_nonzero(sel))
            if n_sel < _TILE_MIN_POINTS:
                n_skipped_empty += 1
                continue

            tile_xyz = xyz[sel]
            tile_normals = normals[sel]
            tile_colors = colors[sel] if colors is not None else None
            tile_conf = confidence[sel]

            padded_min = np.array(
                [tile_min_x - pad, tile_min_y - pad, float(tile_xyz[:, 2].min()) - pad]
            )
            padded_max = np.array(
                [tile_max_x + pad, tile_max_y + pad, float(tile_xyz[:, 2].max()) + pad]
            )
            extent = padded_max - padded_min

            tile_voxel_size = voxel_size
            tile_sdf_trunc = sdf_trunc
            voxel_count = _grid_voxel_count(extent, tile_voxel_size, tile_sdf_trunc)
            if voxel_count > voxel_count_budget:
                scale = (voxel_count / voxel_count_budget) ** (1.0 / 3.0)
                tile_voxel_size = voxel_size * scale
                tile_sdf_trunc = 3.0 * tile_voxel_size
                n_coarsened += 1

            origin = global_origin + np.round((padded_min - global_origin) / tile_voxel_size) * tile_voxel_size
            dims = np.maximum(np.ceil((padded_max - origin) / tile_voxel_size).astype(np.int64), 1) + 1
            max_tile_voxels = max(max_tile_voxels, int(np.prod(dims)))

            volume = TSDFVolume(
                voxel_size=tile_voxel_size,
                sdf_trunc=tile_sdf_trunc,
                origin=origin,
                dims=tuple(int(d) for d in dims),
                use_open3d=False,
                measured_min_weight=measured_min_weight,
                measured_min_views=measured_min_views,
            )
            volume.integrate_point_cloud(tile_xyz, tile_normals, colors=tile_colors, confidence=tile_conf)
            v, f, c, conf = volume.extract_triangle_mesh()
            n_processed += 1
            if v.shape[0] == 0:
                continue
            offset = sum(a.shape[0] for a in all_vertices)
            all_vertices.append(v)
            all_faces.append(f + offset)
            all_colors.append(c)
            all_conf.append(conf)

    stats["tile_edge_m"] = tile_edge_m
    stats["n_tiles_total"] = n_tiles_x * n_tiles_y
    stats["n_tiles_processed"] = n_processed
    stats["n_tiles_skipped_empty"] = n_skipped_empty
    stats["n_tiles_coarsened"] = n_coarsened
    stats["max_tile_voxel_count"] = max_tile_voxels

    if not all_vertices:
        return empty

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    colors_out = np.concatenate(all_colors, axis=0)
    conf_out = np.concatenate(all_conf, axis=0)

    # Weld coincident vertices from overlapping tile halos (see this
    # function's docstring: adjacent tiles share a grid alignment and pad
    # past each other's boundary by the same TSDF search radius, so
    # overlapping voxels were computed from the same input points and
    # coincide to floating-point precision, not just approximately).
    tol = voxel_size * 1e-3
    quant = np.round(vertices / tol).astype(np.int64)
    _uniq, inverse, counts = np.unique(quant, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n_out = counts.shape[0]

    pos_sums = np.zeros((n_out, 3), dtype=np.float64)
    np.add.at(pos_sums, inverse, vertices)
    vertices_welded = pos_sums / counts[:, None]

    color_sums = np.zeros((n_out, 3), dtype=np.float64)
    np.add.at(color_sums, inverse, colors_out.astype(np.float64))
    colors_welded = np.round(color_sums / counts[:, None]).astype(np.uint8)

    conf_welded = np.full(n_out, 255, dtype=np.uint8)
    np.minimum.at(conf_welded, inverse, conf_out)

    faces_remapped = inverse[faces]
    degenerate = (
        (faces_remapped[:, 0] == faces_remapped[:, 1])
        | (faces_remapped[:, 1] == faces_remapped[:, 2])
        | (faces_remapped[:, 0] == faces_remapped[:, 2])
    )
    faces_remapped = faces_remapped[~degenerate]
    if faces_remapped.shape[0] > 0:
        # Overlapping tiles independently extract the same seam-region
        # surface twice; the vertex weld above merges their *vertices*, but
        # each tile still contributes its own (now vertex-welded) copy of
        # every seam triangle -- drop exact duplicate faces (same 3 welded
        # vertex indices, any winding order), same pattern
        # ``fusion.mesh._cluster_decimate`` already uses for its own
        # post-clustering duplicate-face cleanup.
        sorted_faces = np.sort(faces_remapped, axis=1)
        _uniq_faces, first_seen = np.unique(sorted_faces, axis=0, return_index=True)
        faces_remapped = faces_remapped[np.sort(first_seen)]

    return vertices_welded, faces_remapped, colors_welded, conf_welded


def fuse_submaps(
    submaps: list[Submap],
    config: object,
    stats: dict | None = None,
    camera_gps_enu: dict[int, np.ndarray] | None = None,
    conditioned_R: dict[int, np.ndarray] | None = None,
    strategy: str = "chained_sim3",
    keyframe_intrinsics: dict[int, CameraIntrinsics] | None = None,
    point_filter: Callable[[PointCloud], np.ndarray] | None = None,
    keyframe_altitude_m: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PointCloud]:
    """Fuse a sequence of globally-aligned submaps into one confidence-aware mesh.

    Unlike ``TSDFVolume.integrate``, which expects raw per-view depth maps,
    ``Submap`` (see ``drishti3d.types``) carries an already-dense, already
    -merged local point cloud plus per-point raw confidence -- by the time
    geometry reconstruction reaches the fusion stage, the organized
    per-pixel depth/intrinsics that produced that cloud are gone (they were
    only ever geometry-stage-internal). So this entry point takes the
    point-cloud path: submaps are stitched into one global frame with
    ``geometry.submap.merge_submaps`` (reusing that module's alignment
    machinery rather than duplicating it), lightly cleaned with
    ``fusion.filters``, given oriented normals (``fusion.mesh
    .estimate_normals``, oriented toward the submaps' own camera centres),
    and splatted into a ``TSDFVolume`` via ``integrate_point_cloud`` --
    which is exactly the same confidence-weighted volumetric machinery
    ``integrate`` uses for depth maps, just fed local point-based signed-
    distance samples instead of ray-cast ones.

    ``camera_gps_enu``/``conditioned_R``/``strategy`` are passed straight
    through to ``merge_submaps`` -- **this matters, not just for API
    symmetry**: ``pipeline.stages.GeometryStage`` already merged these same
    ``submaps`` once, using whichever strategy
    ``geometry.flight_profile.analyze_flight_profile`` recommended (e.g.
    ``"telemetry_rotation"`` to route around the near-collinear-flight
    -strip rotation degeneracy -- see ``geometry.submap``'s module
    docstring). If this function re-merged with its old default
    (``strategy="chained_sim3"``, no GPS/telemetry context) instead of
    reusing that same strategy, the mesh this stage produces -- which is
    what actually reaches ``ExportStage`` whenever fusion succeeds -- would
    silently discard that fix and come out tilted again, even though
    ``GeometryStage``'s own (unused, in that scenario) merged cloud was
    correct. ``pipeline.stages.FusionStage`` passes ``state
    .geometry_camera_gps_enu``/``state.geometry_conditioned_R``/``state
    .geometry_merge_strategy`` (stashed by ``GeometryStage.run``) here for
    exactly that reason.

    ``config`` is a ``drishti3d.config.FusionConfig`` (typed loosely here
    to avoid a hard import-order dependency); ``voxel_size`` (``None`` ->
    GSD-derived, see ``_derive_voxel_size``), ``voxel_size_gsd_multiplier``,
    ``voxel_count_budget``, ``outlier_std_ratio``, ``min_confidence`` and
    the confidence-tier thresholds (``raw_confidence_measured_min``/
    ``raw_confidence_low_min``, ``measured_min_weight``/
    ``measured_min_views``, ``covariance_measured_max_m``/
    ``covariance_low_confidence_max_m`` -- see that dataclass's docstring
    for what each means and how they're reconciled) are read off it.

    ``keyframe_intrinsics`` (global keyframe index -> native-resolution
    ``CameraIntrinsics``) is what ``_derive_voxel_size`` needs to compute a
    real ground-sample-distance estimate; pass ``state.keyframes``'
    ``.intrinsics`` (``pipeline.stages.FusionStage`` does exactly this).
    Without it, voxel sizing falls back to the old scale-blind
    bounding-box heuristic -- loudly logged, since that heuristic is what
    originally erased fine structure on real footage (see module docstring
    and ``_derive_voxel_size``'s own docstring).

    Pre-TSDF downsampling is decoupled from the TSDF's own voxel_size
    ------------------------------------------------------------------------
    Before this fix, the cleaned cloud was voxel-downsampled to *exactly*
    ``voxel_size`` right before TSDF integration -- harmless when
    ``voxel_size`` was already coarse (the old bounding-box heuristic), but
    with GSD-derived sizing now typically much finer, doing that
    unconditionally would mean every point cloud gets downsampled to
    whatever TSDF resolution was chosen, silently reintroducing a
    detail-destroying pre-fusion voxelisation step independent of the fix
    above. Instead, the cloud that reaches TSDF integration is always
    downsampled to ``voxel_size * pre_tsdf_downsample_voxel_fraction``
    (default 1/4 of ``voxel_size``) -- decoupled in the sense that it is
    always a small *fraction* of the final voxel size, never equal to it,
    so nothing the mesh could actually represent is ever discarded; this
    exists purely to bound ``TSDFVolume.integrate_point_cloud``'s per-voxel
    KD-tree cost, which scales with *local point density*, not just voxel
    count (measured on real footage: a single 16-keyframe run produced
    over 1.1 million cleaned points, which made per-voxel ball queries
    take minutes without this step). A hard backstop
    (``pre_tsdf_max_points``, see ``_DEFAULT_PRE_TSDF_MAX_POINTS``) only
    fires in the pathological case where even that fraction-based
    downsample still leaves too many points -- and unlike the
    fraction-based step, coarsening there does trade away real detail, so
    it is logged loudly when it happens.

    Returns
    -------
    ``(vertices, faces, colors, confidence, raw_point_cloud)``.
    ``raw_point_cloud`` is the merged, cleaned (outlier-removed +
    confidence-filtered), but never coarsely voxel-downsampled dense point
    cloud, in the same global frame as ``vertices`` -- written by
    ``export.export_all`` as ``point_cloud.ply``/``point_cloud.las``
    *independently* of whether TSDF meshing itself produced any vertices
    at all (meshing is lossy; this is the full-detail fallback that
    matches what a plain "dump the dense point cloud" pipeline would
    produce). Non-empty whenever ``merge_submaps`` + cleaning produced any
    surviving points, even on every other early-return path below.

    Confidence source: covariance first, backbone confidence as fallback
    ------------------------------------------------------------------------
    Per-point covariance (``PointCloud.covariance``, propagated through
    ``merge_submaps`` when a submap's points carry it -- e.g. once a future
    caller attaches BA-derived uncertainty to ``Submap.points``) is the
    principled confidence source when available:
    ``geometry.covariance.confidence_from_covariance`` maps a point's actual
    propagated 3D uncertainty ellipsoid straight to a tier, with no
    dependence on how "confident" a backbone felt about its own guess.
    Whenever it's missing (the common case today -- no backbone in this
    codebase attaches per-point covariance yet), this falls back to
    quantized raw backbone confidence + the TSDF's own accumulated
    weight/view-count corroboration instead (see this module's docstring
    for the combined rule). ``stats["confidence_source"]`` records which
    one actually ran, so a report card built from ``stats`` can say so
    rather than leaving a reader to guess.

    ``stats``, when given, is filled in with the point count surviving each
    cleaning step (``"merged"``, ``"after_outlier_removal"``,
    ``"after_confidence_filter"``, ``"after_voxel_downsample"``) plus
    ``"voxel_size"`` actually used and, when the cloud empties out before
    a mesh can be extracted, ``"empty_at_step"`` naming exactly which step
    did it -- see ``pipeline.stages.FusionStage``, which surfaces this in
    its ``StageResult`` message instead of a bare "insufficient points".
    """
    from drishti3d.fusion.filters import (
        _select,
        confidence_filter,
        statistical_outlier_removal,
        voxel_downsample,
    )
    from drishti3d.fusion.mesh import estimate_normals
    from drishti3d.geometry.covariance import (
        ConfidenceThresholds,
        confidence_from_covariance,
    )
    from drishti3d.geometry.submap import merge_submaps

    if stats is None:
        stats = {}

    configured_voxel_size = getattr(config, "voxel_size", None)
    voxel_size_gsd_multiplier = float(getattr(config, "voxel_size_gsd_multiplier", _DEFAULT_VOXEL_SIZE_GSD_MULTIPLIER))
    voxel_count_budget = int(getattr(config, "voxel_count_budget", _DEFAULT_VOXEL_COUNT_BUDGET))
    pre_tsdf_downsample_fraction = float(
        getattr(config, "pre_tsdf_downsample_voxel_fraction", _DEFAULT_PRE_TSDF_DOWNSAMPLE_FRACTION)
    )
    pre_tsdf_max_points = int(getattr(config, "pre_tsdf_max_points", _DEFAULT_PRE_TSDF_MAX_POINTS))
    outlier_std_ratio = float(getattr(config, "outlier_std_ratio", 2.0))
    outlier_k = int(getattr(config, "outlier_k", 20))
    min_confidence = getattr(config, "min_confidence", None)
    raw_confidence_measured_min = float(getattr(config, "raw_confidence_measured_min", _CONF_TIER_MEASURED_MIN))
    raw_confidence_low_min = float(getattr(config, "raw_confidence_low_min", _CONF_TIER_LOW_MIN))
    measured_min_views = int(getattr(config, "measured_min_views", _MEASURED_MIN_VIEWS))
    measured_min_weight = float(getattr(config, "measured_min_weight", _MEASURED_MIN_WEIGHT))
    covariance_measured_max_m = float(getattr(config, "covariance_measured_max_m", 0.05))
    covariance_low_confidence_max_m = float(getattr(config, "covariance_low_confidence_max_m", 0.5))

    if not submaps:
        stats["empty_at_step"] = "input"
        stats["merged"] = 0
        return (*_empty_mesh(), _empty_point_cloud())

    # lock_scale: see geometry.submap.umeyama_fixed_rotation's `scale`
    # parameter. When the backbone emits metres, a per-submap scale has
    # nothing legitimate to correct and everything to get wrong.
    lock_scale = 1.0 if getattr(config, "lock_metric_scale", False) else None
    stats["merge_scale_locked"] = lock_scale is not None
    stats["altitude_anchored"] = bool(keyframe_altitude_m)
    merged, poses = merge_submaps(
        submaps,
        camera_gps_enu=camera_gps_enu,
        conditioned_R=conditioned_R,
        strategy=strategy,
        lock_scale=lock_scale,
        keyframe_altitude_m=keyframe_altitude_m,
        overlap_align=bool(getattr(config, "overlap_align", True)),
    )
    stats["merged"] = int(merged.xyz.shape[0])
    stats["raw_submap_points"] = int(sum(sm.points.xyz.reshape(-1, 3).shape[0] for sm in submaps))
    stats["deduplicated"] = max(0, stats["raw_submap_points"] - stats["merged"])

    if merged.xyz.shape[0] == 0:
        stats["empty_at_step"] = "merge_submaps (alignment produced zero points)"
        return (*_empty_mesh(), _empty_point_cloud())

    # Confidence source: BA covariance when the merged cloud actually has
    # it, quantized raw backbone confidence otherwise -- see this
    # function's docstring. Either way this must happen *before* any
    # confidence-aware filtering below (quantizing too late, or never, is
    # what previously made confidence_filter reject every point).
    if merged.covariance is not None:
        tier = confidence_from_covariance(
            merged.covariance,
            ConfidenceThresholds(
                measured_max_m=covariance_measured_max_m, low_confidence_max_m=covariance_low_confidence_max_m
            ),
        ).astype(np.uint8)
        stats["confidence_source"] = "ba_covariance"
    else:
        tier = _quantize_confidence_to_tier(merged.confidence, raw_confidence_measured_min, raw_confidence_low_min)
        stats["confidence_source"] = "backbone_confidence_and_view_count"
    merged = PointCloud(xyz=merged.xyz, rgb=merged.rgb, covariance=merged.covariance, confidence=tier)

    cleaned = statistical_outlier_removal(merged, k=outlier_k, std_ratio=outlier_std_ratio)
    stats["after_outlier_removal"] = int(cleaned.xyz.shape[0])
    if cleaned.xyz.shape[0] == 0:
        stats["empty_at_step"] = f"statistical_outlier_removal (k={outlier_k}, std_ratio={outlier_std_ratio})"
        return (*_empty_mesh(), _empty_point_cloud())

    # Evidence-based rejection, BEFORE anything is meshed.
    #
    # This hook exists because of an ordering defect that made the whole
    # photometric check decorative. Verification used to run only *after*
    # fusion had already produced a mesh, so a point the source frames
    # demonstrably disagreed about was welded into the surface and then
    # labelled LOW_CONFIDENCE afterwards. On a 956 px run that was 59.2% of
    # all points: the pipeline measured its own errors correctly and then
    # meshed them anyway. The visible result was a surface whose median
    # vertex sat 5.5 m above its own local ground in a scene that is mostly
    # flat -- buildings were present but buried in noise of the same
    # amplitude, which is what "there are no buildings in the model" looks
    # like from the outside.
    #
    # The filter is injected rather than implemented here on purpose:
    # fusion has no access to the video, the keyframes or the poses, and
    # giving it those just to run a projection pass would invert the
    # dependency between this module and the pipeline stage that owns them.
    # The caller supplies a callable taking the merged world-frame cloud
    # and returning a bool keep-mask; fusion only applies it and records
    # what it cost.
    if point_filter is not None:
        keep = np.asarray(point_filter(cleaned), dtype=bool)
        if keep.shape != (cleaned.xyz.shape[0],):
            raise ValueError(
                f"point_filter returned mask of shape {keep.shape}, expected ({cleaned.xyz.shape[0]},)"
            )
        stats["point_filter_rejected"] = int((~keep).sum())
        stats["point_filter_kept_pct"] = round(100.0 * float(keep.mean()), 2)
        cleaned = _select(cleaned, keep)
        stats["after_point_filter"] = int(cleaned.xyz.shape[0])
        if cleaned.xyz.shape[0] == 0:
            stats["empty_at_step"] = "point_filter (rejected every point)"
            return (*_empty_mesh(), _empty_point_cloud())

    if min_confidence is not None and cleaned.confidence is not None:
        cleaned = confidence_filter(cleaned, min_confidence=int(min_confidence))
    stats["after_confidence_filter"] = int(cleaned.xyz.shape[0])
    if cleaned.xyz.shape[0] == 0:
        stats["empty_at_step"] = f"confidence_filter (min_confidence={min_confidence})"
        return (*_empty_mesh(), _empty_point_cloud())

    # Fix 1: voxel size is now derived from ground sample distance, not the
    # scene's bounding-box extent -- see _derive_voxel_size's docstring for
    # why, and the module docstring for the root-cause failure this fixes.
    voxel_size = (
        float(configured_voxel_size)
        if configured_voxel_size
        else _derive_voxel_size(cleaned.xyz, poses, keyframe_intrinsics, voxel_size_gsd_multiplier, voxel_count_budget, stats)
    )
    if configured_voxel_size:
        stats["voxel_size_source"] = "config_override"

    # Fix (mesh voxel-lattice artifact): when the single-dense-grid budget
    # clamp above fired, try block-sparse tiling instead of accepting the
    # coarsened whole-bounding-box size -- see the "Block-sparse tiled
    # fusion" section above _fuse_tiled for the full rationale (in short:
    # a curved/partial-coverage flight's occupied footprint is typically a
    # small fraction of its own bounding box, so a dense grid wastes most
    # of the budget on voxels nothing ever touches).
    use_tiled_fusion = False
    if not configured_voxel_size and stats.get("voxel_size_budget_exceeded", False) and stats.get("voxel_size_ideal_m"):
        ideal = float(stats["voxel_size_ideal_m"])
        occupied_area_m2 = _occupied_area_m2(cleaned.xyz)
        typical_local_z_extent_m = _typical_local_z_extent(cleaned.xyz)
        tiled_voxel_size = max(
            _solve_tiled_voxel_size(occupied_area_m2, typical_local_z_extent_m, voxel_count_budget), ideal
        )
        # Only worth the per-tile bookkeeping overhead if it actually beats
        # the single-dense-grid coarsened size by a meaningful margin --
        # for a scene whose occupied footprint nearly fills its own
        # bounding box, tiling buys little (most tiles would be "occupied"
        # anyway).
        if tiled_voxel_size < voxel_size * 0.9:
            use_tiled_fusion = True
            voxel_size = tiled_voxel_size
            stats["voxel_size_source"] = f"{stats['voxel_size_source']}+tiled"
            stats["voxel_size_occupied_area_m2"] = occupied_area_m2
            stats["voxel_size_typical_local_z_extent_m"] = typical_local_z_extent_m
    # Size the voxel for the mesh we actually intend to keep.
    #
    # Measured on a real run: a 0.66 m voxel produced 10,003,138 faces,
    # which were then decimated to 1,999,999 -- so 80% of the TSDF
    # integration and marching-cubes work (345 s) plus the decimation
    # itself (97 s) was spent on geometry that was immediately discarded.
    #
    # Face count for a surface goes as C * area / voxel^2. C is 2 for a
    # flat sheet; on that run it measured ~17, because the surface carries
    # roughly 8x the area of the terrain it represents -- that excess IS
    # the depth noise. Coarsening to hit the face budget therefore removes
    # mostly noise, not detail, and it is strictly cheaper than building
    # the noise and decimating it away.
    target_faces = int(getattr(config, "max_mesh_faces", 0) or 0)
    if target_faces > 0 and not configured_voxel_size:
        area = _occupied_area_m2(cleaned.xyz)
        if area > 0:
            # C = 17 from the measurement above: assuming the flat-sheet 2
            # would under-estimate the voxel and rebuild the same problem.
            face_voxel = float(np.sqrt(_FACES_PER_AREA_CONSTANT * area / target_faces))
            if face_voxel > voxel_size:
                logger.info(
                    "fusion: coarsening voxel %.3g m -> %.3g m to target %d faces over %.0f m2 of "
                    "occupied ground (building ~%d faces and decimating them away costs more than "
                    "not building them)",
                    voxel_size,
                    face_voxel,
                    target_faces,
                    area,
                    target_faces,
                )
                voxel_size = face_voxel
                stats["voxel_size_source"] = f"{stats.get('voxel_size_source', 'gsd')}+face_budget"
                stats["voxel_size_face_budget_m"] = face_voxel

    stats["voxel_size"] = voxel_size
    stats["tiled_fusion"] = use_tiled_fusion

    # raw_point_cloud (Fix 2): the merged + cleaned dense cloud, captured
    # *before* any TSDF-only coarsening below -- see fuse_submaps' own
    # docstring section "Pre-TSDF downsampling is decoupled from the TSDF's
    # own voxel_size". This is what gets exported as point_cloud.ply/.las
    # independently of whatever the mesh below does or doesn't produce.
    raw_point_cloud = cleaned
    stats["pre_tsdf_points"] = int(cleaned.xyz.shape[0])

    # Fix 3: pre-TSDF downsampling, decoupled from the TSDF's own
    # voxel_size -- see _DEFAULT_PRE_TSDF_DOWNSAMPLE_FRACTION's docstring
    # for why this is safe (never coarser than a small fraction of what
    # the mesh can represent anyway) and necessary (integrate_point_cloud's
    # per-voxel cost scales with local point density, and real backbone
    # output can be *far* denser than a GSD-correct voxel_size needs --
    # measured on real footage, over a million cleaned points without this
    # step, which made per-voxel KD-tree queries take minutes).
    cleaned_for_tsdf = cleaned
    pre_tsdf_voxel_size = voxel_size * pre_tsdf_downsample_fraction
    if pre_tsdf_voxel_size > 0:
        candidate = voxel_downsample(cleaned, voxel_size=pre_tsdf_voxel_size)
        if candidate.xyz.shape[0] < cleaned.xyz.shape[0]:
            cleaned_for_tsdf = candidate
            stats["pre_tsdf_downsample_voxel_size"] = pre_tsdf_voxel_size
            logger.info(
                "fusion: pre-TSDF downsample to %.4gm voxels (1/%.3g of TSDF voxel_size=%.4gm, so nothing "
                "the final mesh could represent is lost) -- %d -> %d points, purely to bound "
                "TSDF integration cost; point_cloud.ply/.las still export the full %d-point cloud.",
                pre_tsdf_voxel_size,
                1.0 / pre_tsdf_downsample_fraction if pre_tsdf_downsample_fraction > 0 else float("inf"),
                voxel_size,
                cleaned.xyz.shape[0],
                cleaned_for_tsdf.xyz.shape[0],
                cleaned.xyz.shape[0],
            )

    # Hard backstop (see _DEFAULT_PRE_TSDF_MAX_POINTS's docstring): only
    # fires in the pathological case where even the fraction-based
    # downsample above still leaves more points than this pipeline can
    # integrate in reasonable time -- unlike the step above, this one DOES
    # trade away some resolvable detail, so it is logged loudly.
    if cleaned_for_tsdf.xyz.shape[0] > pre_tsdf_max_points:
        backstop_voxel_size = _voxel_size_for_point_budget(cleaned_for_tsdf.xyz, pre_tsdf_max_points)
        backstop_voxel_size = max(backstop_voxel_size, pre_tsdf_voxel_size)
        coarsened = voxel_downsample(cleaned_for_tsdf, voxel_size=backstop_voxel_size)
        logger.warning(
            "FUSION PRE-TSDF POINT BUDGET WARNING: even after fraction-based downsampling, %d points "
            "still exceed the %d-point hard backstop -- coarsening further to %.4gm voxels (%d points) "
            "purely to bound TSDF integration time. point_cloud.ply/.las are unaffected (still the "
            "full %d-point cleaned cloud).",
            cleaned_for_tsdf.xyz.shape[0],
            pre_tsdf_max_points,
            backstop_voxel_size,
            coarsened.xyz.shape[0],
            cleaned.xyz.shape[0],
        )
        cleaned_for_tsdf = coarsened
        stats["pre_tsdf_backstop_voxel_size"] = backstop_voxel_size

    stats["after_voxel_downsample"] = int(cleaned_for_tsdf.xyz.shape[0])

    if cleaned_for_tsdf.xyz.shape[0] < 4:
        stats["empty_at_step"] = (
            f"voxel_downsample (voxel_size={voxel_size:.4g}m; need >= 4 points, got {cleaned_for_tsdf.xyz.shape[0]})"
        )
        return (*_empty_mesh(), raw_point_cloud)

    normals = estimate_normals(
        cleaned_for_tsdf,
        k=min(30, cleaned_for_tsdf.xyz.shape[0] - 1),
        camera_positions=np.array([p.t for p in poses]) if poses else None,
    )

    sdf_trunc = 3.0 * voxel_size
    confidence_arr = (
        cleaned_for_tsdf.confidence
        if cleaned_for_tsdf.confidence is not None
        else np.full(cleaned_for_tsdf.xyz.shape[0], Confidence.LOW_CONFIDENCE)
    )

    if use_tiled_fusion:
        vertices, faces, colors, confidence = _fuse_tiled(
            cleaned_for_tsdf.xyz,
            normals,
            cleaned_for_tsdf.rgb,
            confidence_arr.astype(np.float64),
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            voxel_count_budget=voxel_count_budget,
            typical_local_z_extent_m=stats.get(
                "voxel_size_typical_local_z_extent_m", _typical_local_z_extent(cleaned_for_tsdf.xyz)
            ),
            measured_min_weight=measured_min_weight,
            measured_min_views=measured_min_views,
            stats=stats,
        )
    else:
        bounds_min = cleaned_for_tsdf.xyz.min(axis=0) - sdf_trunc - voxel_size
        bounds_max = cleaned_for_tsdf.xyz.max(axis=0) + sdf_trunc + voxel_size
        dims = np.maximum(np.ceil((bounds_max - bounds_min) / voxel_size).astype(np.int64), 1) + 1
        # A list, not a tuple: PipelineResult.save/load round-trips
        # artifacts through JSON, which has no tuple type -- a tuple here
        # would compare unequal to its own post-round-trip list, breaking
        # test_pipeline_result_save_load_round_trip's exact stage_results
        # equality check for a reason that has nothing to do with any
        # actual data loss.
        stats["grid_dims"] = [int(d) for d in dims]
        stats["grid_voxel_count"] = int(np.prod(dims))

        volume = TSDFVolume(
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            origin=bounds_min,
            dims=tuple(int(d) for d in dims),
            use_open3d=False,
            measured_min_weight=measured_min_weight,
            measured_min_views=measured_min_views,
        )
        volume.integrate_point_cloud(
            cleaned_for_tsdf.xyz, normals, colors=cleaned_for_tsdf.rgb, confidence=confidence_arr.astype(np.float64)
        )
        vertices, faces, colors, confidence = volume.extract_triangle_mesh()

    stats["vertices"] = int(vertices.shape[0])
    if vertices.shape[0] == 0:
        stats["empty_at_step"] = "TSDF mesh extraction (no zero-crossing found in the volume)"
    return vertices, faces, colors, confidence, raw_point_cloud
