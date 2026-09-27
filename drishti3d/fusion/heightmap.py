"""2.5D height-map fusion for nadir flights: one surface per ground column, by agreement vote.

Why
---
Every keyframe's depth is regressed and then fitted to triangulated points
with metres of residual error. Averaging all of them in 3-D voxels
(``fusion.incremental``) keeps every disagreeing copy that falls in a
different voxel: the "thick", stacked ground the placement check flags.
A straight-down survey sees a 2.5-D surface -- one height per ground
position -- so the right fusion is per COLUMN, and the right statistic
over disagreeing views is the median, not the mean.

How
---
Ground is gridded at ``cell_m``. For every (cell, keyframe) pair the mean
height of that keyframe's points in the cell is accumulated incrementally
(sorted parallel arrays, one ``np.unique`` per added window). When the
model is read out, each view is one vote and the cell takes the largest
cluster of votes that agree within ``agree_m`` (a mode, robust like a
median but always at a height some view actually saw); height and colour
are that cluster's mean.

Confidence tiers come from agreement, which is what a measured point is:
the number of keyframes within ``agree_m`` of the cell's chosen cluster.

    >= 3 agreeing views  MEASURED
       2                 LOW_CONFIDENCE
       1                 INFERRED

The mesh is the grid itself: two triangles per 2x2 block of occupied
cells whose heights span less than ``max_step_m`` (a cliff or roof edge is
left open rather than bridged with a false wall).
"""

from __future__ import annotations

import numpy as np

from drishti3d.types import Confidence, PointCloud

__all__ = ["HeightmapFusion", "cell_size_for"]

_OFF = 1 << 20
_M21 = (1 << 21) - 1


def cell_size_for(xyz: np.ndarray, floor_m: float = 0.3, sample: int = 4000, seed: int = 0) -> float:
    """A cell about the size of the depth maps' ground spacing (1.5x median neighbour distance, >= floor)."""
    from scipy.spatial import cKDTree

    xy = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)[:, :2]
    if xy.shape[0] < 10:
        return floor_m
    rng = np.random.default_rng(seed)
    sub = xy[rng.choice(xy.shape[0], size=min(sample, xy.shape[0]), replace=False)]
    d, _ = cKDTree(xy).query(sub, k=2)
    return float(max(floor_m, 1.5 * np.median(d[:, 1])))


class HeightmapFusion:
    """Running median-vote height map. Same add/cloud interface as ``IncrementalVoxelFusion``."""

    def __init__(self, cell_m: float = 0.5, agree_m: float = 1.0, max_step_m: float | None = None) -> None:
        if cell_m <= 0:
            raise ValueError("cell_m must be positive")
        self.cell_m = float(cell_m)
        self.agree_m = float(agree_m)
        self.max_step_m = float(max_step_m) if max_step_m is not None else max(2.0, 4.0 * self.cell_m)
        self._pair = np.zeros(0, dtype=np.int64)  # (cell << 21) | keyframe, sorted
        self._sum_z = np.zeros(0)
        self._sum_rgb = np.zeros((0, 3))
        self._w = np.zeros(0)
        self.windows_fused = 0

    def __len__(self) -> int:
        return int(np.unique(self._pair >> 21).size) if self._pair.size else 0

    def _cell(self, xy: np.ndarray) -> np.ndarray:
        ij = np.floor(xy / self.cell_m).astype(np.int64) + _OFF
        ij &= _M21
        return (ij[:, 0] << 21) | ij[:, 1]

    def add(self, xyz, keyframe_ids, rgb=None, weight=None) -> None:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = xyz.shape[0]
        w = np.ones(n) if weight is None else np.asarray(weight, dtype=np.float64).reshape(-1)
        kf = np.asarray(keyframe_ids, dtype=np.int64).reshape(-1) & _M21
        col = np.zeros((n, 3)) if rgb is None else np.asarray(rgb, dtype=np.float64).reshape(n, -1)[:, :3]
        ok = np.isfinite(xyz).all(axis=1) & np.isfinite(w) & (w > 0)
        if not ok.any():
            return
        xyz, w, kf, col = xyz[ok], w[ok], kf[ok], col[ok]
        keys = np.concatenate([self._pair, (self._cell(xyz[:, :2]) << 21) | kf])
        uniq, inv = np.unique(keys, return_inverse=True)
        m = uniq.size
        self._sum_z = np.bincount(inv, weights=np.concatenate([self._sum_z, xyz[:, 2] * w]), minlength=m)
        rgb_new = np.zeros((m, 3))
        for a in range(3):
            rgb_new[:, a] = np.bincount(inv, weights=np.concatenate([self._sum_rgb[:, a], col[:, a] * w]), minlength=m)
        self._sum_rgb = rgb_new
        self._w = np.bincount(inv, weights=np.concatenate([self._w, w]), minlength=m)
        self._pair = uniq
        self.windows_fused += 1

    def add_submap(self, submap) -> None:
        xyz = submap.points.xyz.reshape(-1, 3)
        vidx = getattr(submap, "view_index", None)
        kfs = np.asarray(submap.keyframe_indices, dtype=np.int64)
        if vidx is not None and len(vidx) == xyz.shape[0]:
            kf_ids = kfs[np.asarray(vidx, dtype=np.int64)]
        else:
            kf_ids = np.full(xyz.shape[0], kfs[0] if kfs.size else 0)
        conf = np.asarray(submap.confidence, dtype=np.float64).reshape(-1)
        rgb = submap.points.rgb.reshape(xyz.shape[0], -1) if submap.points.rgb is not None else None
        self.add(xyz, kf_ids, rgb=rgb, weight=conf if conf.size == xyz.shape[0] else None)

    def _columns(self):
        """Per cell: key, height, agreeing-view count, colour.

        Each view is one vote (its mean height in the cell). The cell takes
        the vote with the most other votes within ``agree_m`` -- the largest
        agreeing cluster -- and its height and colour are that cluster's
        mean. A plain median was used first, but with an even number of
        disagreeing votes it lands between them: a height no view saw,
        with no agreeing view to colour it.
        """
        cell = self._pair >> 21
        z = self._sum_z / self._w
        rgb = self._sum_rgb / self._w[:, None]
        order = np.lexsort((z, cell))
        cs, zs, rs = cell[order], z[order], rgb[order]
        starts = np.r_[0, np.nonzero(np.diff(cs))[0] + 1]
        counts = np.diff(np.r_[starts, cs.size])
        group = np.repeat(np.arange(starts.size), counts)
        # Votes sorted by (cell, z): neighbours within agree_m are a
        # contiguous run, found with two searchsorted calls on one key.
        span = float(zs.max() - zs.min()) + 4.0 * self.agree_m + 1.0
        key = group * span + (zs - zs.min())
        near = np.searchsorted(key, key + self.agree_m, "right") - np.searchsorted(key, key - self.agree_m, "left")
        # Best vote per cell: most neighbours; ties -> the lower-median-most one (first in z order).
        best = np.lexsort((np.arange(zs.size), -near, group))
        first = np.r_[True, group[best][1:] != group[best][:-1]]
        centre = np.empty(starts.size)
        centre[group[best][first]] = zs[best][first]
        agree = np.abs(zs - centre[group]) <= self.agree_m
        n_agree = np.bincount(group, weights=agree.astype(np.float64), minlength=starts.size).astype(np.int64)
        height = np.bincount(group, weights=zs * agree, minlength=starts.size) / np.maximum(n_agree, 1)
        col = np.stack([np.bincount(group, weights=rs[:, a] * agree, minlength=starts.size) for a in range(3)], axis=1)
        col /= np.maximum(n_agree, 1)[:, None]
        return cs[starts], height, n_agree, col

    def cloud(self) -> PointCloud:
        if not self._pair.size:
            return PointCloud(xyz=np.zeros((0, 3)))
        keys, med, n_agree, col = self._columns()
        i = ((keys >> 21) & _M21) - _OFF
        j = (keys & _M21) - _OFF
        xyz = np.c_[(i + 0.5) * self.cell_m, (j + 0.5) * self.cell_m, med]
        tiers = np.full(keys.size, int(Confidence.INFERRED), dtype=np.uint8)
        tiers[n_agree == 2] = int(Confidence.LOW_CONFIDENCE)
        tiers[n_agree >= 3] = int(Confidence.MEASURED)
        return PointCloud(xyz=xyz, rgb=np.clip(np.rint(col), 0, 255).astype(np.uint8), confidence=tiers)

    def mesh(self) -> tuple[PointCloud, np.ndarray]:
        """``(vertices, faces)`` -- the cloud's cells joined into a grid surface."""
        pc = self.cloud()
        n = pc.xyz.shape[0]
        if n < 4:
            return pc, np.zeros((0, 3), dtype=np.int64)
        i = np.floor(pc.xyz[:, 0] / self.cell_m).astype(np.int64)
        j = np.floor(pc.xyz[:, 1] / self.cell_m).astype(np.int64)
        key = (i + _OFF) * (1 << 22) + (j + _OFF)
        order = np.argsort(key)
        sk = key[order]

        def find(di, dj):
            q = (i + di + _OFF) * (1 << 22) + (j + dj + _OFF)
            pos = np.clip(np.searchsorted(sk, q), 0, n - 1)
            hit = sk[pos] == q
            return np.where(hit, order[pos], -1)

        a, b, c, d = np.arange(n), find(1, 0), find(0, 1), find(1, 1)
        z = pc.xyz[:, 2]
        ok = (b >= 0) & (c >= 0) & (d >= 0)
        zz = np.stack([z[a[ok]], z[b[ok]], z[c[ok]], z[d[ok]]], axis=1)
        flat = (zz.max(1) - zz.min(1)) <= self.max_step_m
        a, b, c, d = a[ok][flat], b[ok][flat], c[ok][flat], d[ok][flat]
        faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)]).astype(np.int64)
        return pc, faces
