"""Incremental voxel fusion: fold each new window into one running model as it finishes.

Why
---
``FusionStage`` meshes everything at the end, fourteen-plus minutes into a
run. When windows are born in the world frame (``GeometryConfig
.ba_world_frame``) there is no merge left to wait for, so each window can
be fused into the model the moment the backbone returns it. The operator
watches the model extend along the flight path instead of watching a
progress bar.

How
---
A sparse voxel grid, stored as sorted parallel arrays keyed by a packed
``(i, j, k)``. Each voxel keeps a confidence-weighted sum of position and
colour -- so the voxel's point is the weighted mean of every observation
that fell in it -- and a 64-bit mask of which keyframes observed it
(bit = keyframe index mod 64). Adding a window is one ``np.unique`` over
old and new keys; nothing is ever re-fused.

Confidence tiers
----------------
A voxel's tier comes from how many distinct keyframes agree on it, which
is what makes a point measured rather than guessed: one view cannot
triangulate anything, it can only regress it. Agreement is counted on a
coarser ``tier_voxel_m`` grid (default 1 m, the accuracy requirement), not
the fine geometry voxel: per-view depth is only good to metres, so three
views of one spot rarely land in the same 0.3 m cell, and counting there
marked almost everything inferred.

    >= 3 views  MEASURED
       2 views  LOW_CONFIDENCE
       1 view   INFERRED

The mask aliases keyframes 64 apart; windows span far fewer keyframes
than that, so neighbouring voxels are never seen by two aliased views.
"""

from __future__ import annotations

import numpy as np

from drishti3d.types import Confidence, PointCloud

__all__ = ["IncrementalVoxelFusion"]

_OFFSET = 1 << 20  # packs signed voxel indices into 21 unsigned bits each
_MASK21 = (1 << 21) - 1


def _popcount64(mask: np.ndarray) -> np.ndarray:
    as_bytes = mask.astype("<u8").view(np.uint8).reshape(-1, 8)
    return np.unpackbits(as_bytes, axis=1).sum(axis=1)


class IncrementalVoxelFusion:
    """Running confidence-weighted voxel model. See the module docstring."""

    def __init__(self, voxel_size_m: float = 0.3, tier_voxel_m: float = 1.0) -> None:
        if voxel_size_m <= 0 or tier_voxel_m <= 0:
            raise ValueError("voxel sizes must be positive")
        self.voxel_size_m = float(voxel_size_m)
        # Agreement is judged on a cell at least as big as the geometry voxel.
        self.tier_voxel_m = max(float(tier_voxel_m), self.voxel_size_m)
        self._tier_keys = np.zeros(0, dtype=np.int64)
        self._tier_views = np.zeros(0, dtype=np.uint64)
        self._keys = np.zeros(0, dtype=np.int64)
        self._sum_xyz = np.zeros((0, 3))
        self._sum_rgb = np.zeros((0, 3))
        self._weight = np.zeros(0)
        self._views = np.zeros(0, dtype=np.uint64)
        self.windows_fused = 0

    def __len__(self) -> int:
        return int(self._keys.size)

    def _pack(self, xyz: np.ndarray, size: float | None = None) -> np.ndarray:
        ijk = np.floor(xyz / (size or self.voxel_size_m)).astype(np.int64) + _OFFSET
        ijk &= _MASK21
        return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]

    def add(
        self,
        xyz: np.ndarray,
        keyframe_ids: np.ndarray,
        rgb: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> None:
        """Fuse one window's world-frame points. ``keyframe_ids`` is each point's observing keyframe."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = xyz.shape[0]
        w = np.ones(n) if weight is None else np.asarray(weight, dtype=np.float64).reshape(-1)
        kf = np.asarray(keyframe_ids, dtype=np.int64).reshape(-1)
        col = np.zeros((n, 3)) if rgb is None else np.asarray(rgb, dtype=np.float64).reshape(n, -1)[:, :3]

        ok = np.isfinite(xyz).all(axis=1) & np.isfinite(w) & (w > 0)
        if not ok.any():
            return
        xyz, w, kf, col = xyz[ok], w[ok], kf[ok], col[ok]

        keys = np.concatenate([self._keys, self._pack(xyz)])
        uniq, inv = np.unique(keys, return_inverse=True)
        m = uniq.size

        weights = np.concatenate([self._weight, w])
        sum_xyz = np.zeros((m, 3))
        sum_rgb = np.zeros((m, 3))
        for axis in range(3):
            sum_xyz[:, axis] = np.bincount(
                inv, weights=np.concatenate([self._sum_xyz[:, axis], xyz[:, axis] * w]), minlength=m
            )
            sum_rgb[:, axis] = np.bincount(
                inv, weights=np.concatenate([self._sum_rgb[:, axis], col[:, axis] * w]), minlength=m
            )
        views = np.zeros(m, dtype=np.uint64)
        bits = np.left_shift(np.uint64(1), (kf % 64).astype(np.uint64))
        np.bitwise_or.at(views, inv, np.concatenate([self._views, bits]))

        tier_keys = np.concatenate([self._tier_keys, self._pack(xyz, self.tier_voxel_m)])
        tuniq, tinv = np.unique(tier_keys, return_inverse=True)
        tviews = np.zeros(tuniq.size, dtype=np.uint64)
        np.bitwise_or.at(tviews, tinv, np.concatenate([self._tier_views, bits]))
        self._tier_keys, self._tier_views = tuniq, tviews

        self._keys = uniq
        self._sum_xyz = sum_xyz
        self._sum_rgb = sum_rgb
        self._weight = np.bincount(inv, weights=weights, minlength=m)
        self._views = views
        self.windows_fused += 1

    def view_counts(self) -> np.ndarray:
        """Distinct keyframes agreeing on each fine voxel's ``tier_voxel_m`` cell."""
        if not len(self):
            return np.zeros(0, dtype=np.int64)
        centres = self._sum_xyz / self._weight[:, None]
        pos = np.searchsorted(self._tier_keys, self._pack(centres, self.tier_voxel_m))
        pos = np.clip(pos, 0, max(self._tier_keys.size - 1, 0))
        return _popcount64(self._tier_views[pos])

    def cloud(self) -> PointCloud:
        """The fused model so far: one point per voxel, tiered by view count."""
        if not len(self):
            return PointCloud(xyz=np.zeros((0, 3)))
        w = self._weight[:, None]
        counts = self.view_counts()
        tiers = np.full(counts.shape, int(Confidence.INFERRED), dtype=np.uint8)
        tiers[counts == 2] = int(Confidence.LOW_CONFIDENCE)
        tiers[counts >= 3] = int(Confidence.MEASURED)
        return PointCloud(
            xyz=self._sum_xyz / w,
            rgb=np.clip(np.rint(self._sum_rgb / w), 0, 255).astype(np.uint8),
            confidence=tiers,
        )

    def add_submap(self, submap) -> None:
        """Fuse a world-frame ``Submap`` (its ``view_index`` maps points to keyframes)."""
        xyz = submap.points.xyz.reshape(-1, 3)
        vidx = getattr(submap, "view_index", None)
        kfs = np.asarray(submap.keyframe_indices, dtype=np.int64)
        if vidx is not None and len(vidx) == xyz.shape[0]:
            kf_ids = kfs[np.asarray(vidx, dtype=np.int64)]
        else:
            kf_ids = np.full(xyz.shape[0], kfs[0] if kfs.size else 0)
        conf = np.asarray(submap.confidence, dtype=np.float64).reshape(-1)
        weight = conf if conf.size == xyz.shape[0] else None
        rgb = submap.points.rgb.reshape(xyz.shape[0], -1) if submap.points.rgb is not None else None
        self.add(xyz, kf_ids, rgb=rgb, weight=weight)
