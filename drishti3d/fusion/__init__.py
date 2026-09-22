"""Fusion stage: merged point clouds -> confidence-aware meshes.

``fusion.filters`` cleans a point cloud (outlier removal, downsampling,
confidence/bounds filtering) before it is baked into geometry.
``fusion.tsdf`` implements confidence-weighted volumetric fusion (a TSDF
volume whose integration weight is scaled by per-observation confidence,
so untrustworthy observations move the fused surface less than solid
ones) and the ``fuse_submaps`` stage entry point. ``fusion.mesh`` covers
mesh-level operations that don't need a volume: normal estimation,
Poisson reconstruction (with fabricated-geometry vertices flagged
``INFERRED``), decimation, and summary statistics.

Every function in this package returns plain numpy arrays and/or
``drishti3d.types`` dataclasses -- never an open3d object -- so the rest
of the pipeline (and every test) can use fusion results with no open3d
installed. open3d, when importable, is used internally as a faster
backend; the pure numpy/scipy fallback is always correct, just slower.
"""

from drishti3d.fusion.filters import (
    confidence_filter,
    crop_to_bounds,
    radius_outlier_removal,
    statistical_outlier_removal,
    voxel_downsample,
)
from drishti3d.fusion.mesh import (
    compute_mesh_stats,
    decimate_mesh,
    estimate_normals,
    poisson_reconstruct,
)
from drishti3d.fusion.tsdf import HAS_OPEN3D, TSDFVolume, fuse_submaps

__all__ = [
    "HAS_OPEN3D",
    "TSDFVolume",
    "compute_mesh_stats",
    "confidence_filter",
    "crop_to_bounds",
    "decimate_mesh",
    "estimate_normals",
    "fuse_submaps",
    "poisson_reconstruct",
    "radius_outlier_removal",
    "statistical_outlier_removal",
    "voxel_downsample",
]
