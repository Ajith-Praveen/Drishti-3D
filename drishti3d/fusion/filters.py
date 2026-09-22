"""Point-cloud cleanup filters that run before meshing.

Pure numpy/scipy -- no open3d import anywhere in this module. These run on
the *merged* global point cloud (post ``geometry.submap.merge_submaps``)
to remove noise before it gets baked into a TSDF volume or a Poisson mesh:
garbage in a volumetric fusion step is much harder to undo than garbage in
a point cloud, so cleanup happens here, first.

Every filter below returns a **new** ``PointCloud`` and is careful to slice
``rgb``, ``covariance`` and ``confidence`` alongside ``xyz`` using the exact
same index/mask. Silently dropping the ``confidence`` array while filtering
points would quietly degrade this whole project's core promise (a
measurable, trust-aware model) back into "just another point cloud" -- so
every function here is built around one shared helper (``_select``) that
makes it structurally impossible to slice ``xyz`` without also slicing its
side-car arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from drishti3d.types import PointCloud

__all__ = [
    "confidence_filter",
    "crop_to_bounds",
    "radius_outlier_removal",
    "statistical_outlier_removal",
    "voxel_downsample",
]


def _select(pc: PointCloud, mask: np.ndarray) -> PointCloud:
    """Index every array on ``pc`` by the same boolean mask / index array.

    The single choke point every filter in this module routes through, so
    a new ``PointCloud`` field never needs each filter to remember to carry
    it along -- add it here once.
    """
    return PointCloud(
        xyz=pc.xyz[mask],
        rgb=pc.rgb[mask] if pc.rgb is not None else None,
        covariance=pc.covariance[mask] if pc.covariance is not None else None,
        confidence=pc.confidence[mask] if pc.confidence is not None else None,
    )


def statistical_outlier_removal(pc: PointCloud, k: int = 20, std_ratio: float = 2.0) -> PointCloud:
    """Remove points whose mean distance to their ``k`` nearest neighbours is an outlier.

    For each point, computes the mean distance to its ``k`` nearest
    neighbours (excluding itself). Points whose mean distance exceeds
    ``global_mean + std_ratio * global_std`` (over the *whole* cloud) are
    dropped -- the classic PCL/Open3D statistical outlier removal test,
    reimplemented here in pure numpy/scipy so it works with no open3d.
    """
    n = pc.xyz.shape[0]
    if n == 0:
        return _select(pc, np.zeros(0, dtype=bool))

    k_eff = min(k, n - 1)
    if k_eff < 1:
        # Too few points to evaluate a neighbourhood statistic at all;
        # nothing can be confidently called an outlier, so keep everything.
        return _select(pc, np.ones(n, dtype=bool))

    tree = cKDTree(pc.xyz)
    # k_eff + 1 because the point itself is always its own nearest
    # neighbour at distance 0 and must be excluded from the mean.
    #
    # Queried in chunks: a single query over the whole cloud materialises
    # an (n, k+1) float64 distance array -- 11.8 GB at 70M points -- and
    # only the per-point mean is ever needed. 2M points per chunk keeps
    # the transient under ~350 MB regardless of cloud size.
    chunk = 2_000_000
    mean_dists = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dists, _ = tree.query(pc.xyz[start:stop], k=k_eff + 1, workers=-1)
        if dists.ndim == 1:
            dists = dists[:, None]
        mean_dists[start:stop] = dists[:, 1:].mean(axis=1)

    mu = float(mean_dists.mean())
    sigma = float(mean_dists.std())
    threshold = mu + std_ratio * sigma

    mask = mean_dists <= threshold
    return _select(pc, mask)


def radius_outlier_removal(pc: PointCloud, radius: float, min_neighbors: int) -> PointCloud:
    """Remove points with fewer than ``min_neighbors`` other points within ``radius``.

    Complementary to ``statistical_outlier_removal``: that one catches
    points that are locally far from a *globally typical* density, this
    one catches points that are simply isolated in absolute terms
    (sparse flyers a long way from any surface), which matters most right
    after triangulation noise has scattered a few points far from the
    true scene.
    """
    n = pc.xyz.shape[0]
    if n == 0:
        return _select(pc, np.zeros(0, dtype=bool))

    tree = cKDTree(pc.xyz)
    neighbor_lists = tree.query_ball_point(pc.xyz, r=radius)
    counts = np.fromiter((len(neighbors) - 1 for neighbors in neighbor_lists), dtype=np.int64, count=n)

    mask = counts >= min_neighbors
    return _select(pc, mask)


def voxel_downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    """Grid-average downsample: one output point per occupied voxel.

    xyz is the mean position of the points falling in each voxel; rgb is
    likewise averaged (rounded back to uint8); covariance is averaged
    element-wise (a rough but honest summary -- properly composing several
    points' covariances would need each point's own mean shift accounted
    for, which we don't have use for downstream, so an elementwise mean is
    a reasonable, cheap proxy for "how uncertain is this neighbourhood").

    confidence takes the **minimum** over the voxel, deliberately, not the
    mean: a voxel that merges nine ``MEASURED`` points and one ``INFERRED``
    point has not become "mostly measured" -- it now contains geometry this
    pipeline cannot actually vouch for, so the conservative (lowest) tier
    is the honest one to report. Averaging confidence here is exactly the
    kind of quiet optimism this project exists to prevent.
    """
    n = pc.xyz.shape[0]
    if n == 0:
        return PointCloud(
            xyz=pc.xyz.copy(),
            rgb=pc.rgb.copy() if pc.rgb is not None else None,
            covariance=pc.covariance.copy() if pc.covariance is not None else None,
            confidence=pc.confidence.copy() if pc.confidence is not None else None,
        )

    voxel_idx = np.floor(pc.xyz / voxel_size).astype(np.int64)
    _uniq, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n_voxels = counts.shape[0]

    xyz_sums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(xyz_sums, inverse, pc.xyz)
    xyz_out = xyz_sums / counts[:, None]

    rgb_out = None
    if pc.rgb is not None:
        rgb_sums = np.zeros((n_voxels, pc.rgb.shape[-1]), dtype=np.float64)
        np.add.at(rgb_sums, inverse, pc.rgb.astype(np.float64))
        rgb_out = np.round(rgb_sums / counts[:, None]).astype(pc.rgb.dtype)

    covariance_out = None
    if pc.covariance is not None:
        cov_sums = np.zeros((n_voxels, 3, 3), dtype=np.float64)
        np.add.at(cov_sums, inverse, pc.covariance)
        covariance_out = cov_sums / counts[:, None, None]

    confidence_out = None
    if pc.confidence is not None:
        if np.issubdtype(pc.confidence.dtype, np.integer):
            fill = np.iinfo(pc.confidence.dtype).max
        else:
            fill = np.inf
        confidence_out = np.full(n_voxels, fill, dtype=pc.confidence.dtype)
        np.minimum.at(confidence_out, inverse, pc.confidence)

    return PointCloud(xyz=xyz_out, rgb=rgb_out, covariance=covariance_out, confidence=confidence_out)


def confidence_filter(pc: PointCloud, min_confidence: int) -> PointCloud:
    """Drop every point whose confidence tier is below ``min_confidence``.

    Requires ``pc.confidence`` to be set -- there is nothing sensible to
    filter on otherwise, and silently passing everything through would
    hide the fact that no confidence information was available.
    """
    if pc.confidence is None:
        raise ValueError("confidence_filter requires pc.confidence to be set")
    mask = pc.confidence >= min_confidence
    return _select(pc, mask)


BoundsSpec = tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]


def crop_to_bounds(pc: PointCloud, bounds: BoundsSpec) -> PointCloud:
    """Crop to an axis-aligned box ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

    Any entry may be ``None`` to leave that side unbounded (e.g.
    ``(None, None, None, None, 0.0, None)`` keeps only points at or above
    z=0).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    mask = np.ones(pc.xyz.shape[0], dtype=bool)
    if xmin is not None:
        mask &= pc.xyz[:, 0] >= xmin
    if xmax is not None:
        mask &= pc.xyz[:, 0] <= xmax
    if ymin is not None:
        mask &= pc.xyz[:, 1] >= ymin
    if ymax is not None:
        mask &= pc.xyz[:, 1] <= ymax
    if zmin is not None:
        mask &= pc.xyz[:, 2] >= zmin
    if zmax is not None:
        mask &= pc.xyz[:, 2] <= zmax
    return _select(pc, mask)
