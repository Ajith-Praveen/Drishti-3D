"""Raster deliverables: Digital Surface Model (DSM), orthomosaic, GeoTIFF I/O.

Rasterizing a point cloud always has to answer "what goes in an empty
cell" -- most DSM tools just interpolate silently, which is exactly the
kind of quiet invention this whole project exists to flag instead of
hide. ``point_cloud_to_dsm`` still fills holes (an all-``NaN`` raster is
useless to an analyst), but it returns a second array recording *which*
cells were filled rather than genuinely sampled, so a renderer/report can
show that distinction instead of erasing it.

``write_geotiff`` prefers ``rasterio`` (the standard, spec-complete way to
write a georeferenced TIFF), falls back to hand-written minimal GeoTIFF
tags via ``tifffile`` when rasterio isn't installed, and falls back again
to a plain ``.npy`` array plus a ``.tfw`` world file and a ``.prj`` WKT
file when neither is installed -- always returning which of the three
paths it actually took, so a caller (and a human reading the report card)
knows whether they got a real GeoTIFF or a "good enough to reload, not a
real GeoTIFF" substitute. Neither ``rasterio`` nor ``tifffile`` become a
hard dependency of this project because of this fallback chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from drishti3d.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "GeoTransform",
    "point_cloud_to_confidence_raster",
    "point_cloud_to_dsm",
    "point_cloud_to_orthomosaic",
    "write_geotiff",
]


@dataclass
class GeoTransform:
    """An affine pixel-to-world transform, stored in world-file (.tfw) field order.

    ``x = pixel_size_x * col + rotation_x * row + x_origin``
    ``y = rotation_y * col + pixel_size_y * row + y_origin``

    ``x_origin``/``y_origin`` are the world coordinates of the *centre* of
    the top-left pixel (row 0, col 0); ``pixel_size_y`` is conventionally
    negative (world y decreases as raster row increases, i.e. north-up).
    """

    pixel_size_x: float
    rotation_y: float
    rotation_x: float
    pixel_size_y: float
    x_origin: float
    y_origin: float

    def pixel_to_world(self, col: np.ndarray | float, row: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        x = self.pixel_size_x * col + self.rotation_x * row + self.x_origin
        y = self.rotation_y * col + self.pixel_size_y * row + self.y_origin
        return x, y

    def to_gdal(self) -> tuple[float, float, float, float, float, float]:
        """GDAL's ``GetGeoTransform``/``SetGeoTransform`` field order."""
        return (self.x_origin, self.pixel_size_x, self.rotation_x, self.y_origin, self.rotation_y, self.pixel_size_y)

    def to_world_file_lines(self) -> str:
        return "\n".join(
            f"{v:.12f}"
            for v in (self.pixel_size_x, self.rotation_y, self.rotation_x, self.pixel_size_y, self.x_origin, self.y_origin)
        )


def _grid_shape_and_indices(
    xy: np.ndarray, resolution_m: float, bounds: tuple[float, float, float, float] | None
) -> tuple[int, int, np.ndarray, np.ndarray, GeoTransform]:
    """Shared cell-index computation for the DSM/orthomosaic/confidence rasterizers."""
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
    else:
        xmin, ymin = float(xy[:, 0].min()), float(xy[:, 1].min())
        xmax, ymax = float(xy[:, 0].max()), float(xy[:, 1].max())

    ncols = max(1, int(np.ceil((xmax - xmin) / resolution_m)))
    nrows = max(1, int(np.ceil((ymax - ymin) / resolution_m)))

    col = np.clip(((xy[:, 0] - xmin) / resolution_m).astype(np.int64), 0, ncols - 1)
    # Row 0 is the north (max-y) edge, matching the north-up raster convention.
    row = np.clip(((ymax - xy[:, 1]) / resolution_m).astype(np.int64), 0, nrows - 1)

    transform = GeoTransform(
        pixel_size_x=resolution_m,
        rotation_y=0.0,
        rotation_x=0.0,
        pixel_size_y=-resolution_m,
        x_origin=xmin + resolution_m / 2.0,
        y_origin=ymax - resolution_m / 2.0,
    )
    return nrows, ncols, row, col, transform


def _fill_holes_nearest(grid: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill ``grid`` cells outside ``valid_mask`` with their nearest valid neighbour's value.

    Returns ``(filled_grid, filled_mask)`` where ``filled_mask`` is True
    exactly at the cells that were *not* directly observed and got a
    nearest-neighbour fill -- the honesty mechanism this module exists to
    provide (see module docstring).
    """
    holes = ~valid_mask
    if not np.any(holes):
        return grid.copy(), np.zeros_like(holes)
    if not np.any(valid_mask):
        # Nothing observed at all; nothing to fill from.
        return grid.copy(), holes.copy()

    _, indices = distance_transform_edt(holes, return_distances=True, return_indices=True)
    filled = grid[tuple(indices)]
    result = np.where(holes, filled, grid)
    return result, holes.copy()


def point_cloud_to_dsm(
    pc: PointCloud, resolution_m: float, bounds: tuple[float, float, float, float] | None = None
) -> tuple[np.ndarray, GeoTransform, np.ndarray]:
    """Rasterize to a Digital Surface Model: max-Z per cell, holes nearest-neighbour filled.

    Returns ``(dsm, transform, filled_mask)``. ``filled_mask[i, j]`` is
    True iff cell ``(i, j)`` had no point fall in it and its value was
    filled in from a nearby cell rather than genuinely measured --
    ``bounds``, when given, is ``(xmin, xmax, ymin, ymax)``.
    """
    xyz = np.asarray(pc.xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise ValueError("point_cloud_to_dsm requires a non-empty point cloud")

    nrows, ncols, row, col, transform = _grid_shape_and_indices(xyz[:, :2], resolution_m, bounds)
    flat_idx = row * ncols + col

    flat_dsm = np.full(nrows * ncols, -np.inf, dtype=np.float64)
    np.maximum.at(flat_dsm, flat_idx, xyz[:, 2])
    observed = np.zeros(nrows * ncols, dtype=bool)
    observed[flat_idx] = True

    dsm = flat_dsm.reshape(nrows, ncols)
    valid_mask = observed.reshape(nrows, ncols)
    dsm = np.where(valid_mask, dsm, np.nan)

    dsm_filled, filled_mask = _fill_holes_nearest(dsm, valid_mask)
    return dsm_filled, transform, filled_mask


def point_cloud_to_orthomosaic(
    pc: PointCloud, resolution_m: float, bounds: tuple[float, float, float, float] | None = None
) -> tuple[np.ndarray, GeoTransform, np.ndarray]:
    """RGB raster: colour of the highest-Z point per cell.

    Returns ``(rgb, transform, filled_mask)`` -- ``filled_mask`` marks
    cells with no point at all (colour there is the nearest observed
    cell's, same honesty convention as ``point_cloud_to_dsm``).
    """
    if pc.rgb is None:
        raise ValueError("point_cloud_to_orthomosaic requires pc.rgb")

    xyz = np.asarray(pc.xyz, dtype=np.float64)
    rgb = np.asarray(pc.rgb, dtype=np.uint8)
    if xyz.shape[0] == 0:
        raise ValueError("point_cloud_to_orthomosaic requires a non-empty point cloud")

    nrows, ncols, row, col, transform = _grid_shape_and_indices(xyz[:, :2], resolution_m, bounds)
    flat_idx = row * ncols + col

    # Highest-Z point per cell: sort by descending Z, then the first
    # occurrence of each cell id (via np.unique on the sorted order) is
    # that cell's highest point.
    order = np.argsort(-xyz[:, 2])
    flat_sorted = flat_idx[order]
    cell_ids, first_pos = np.unique(flat_sorted, return_index=True)
    best_point_idx = order[first_pos]

    ortho_flat = np.zeros((nrows * ncols, 3), dtype=np.uint8)
    observed = np.zeros(nrows * ncols, dtype=bool)
    ortho_flat[cell_ids] = rgb[best_point_idx]
    observed[cell_ids] = True

    ortho = ortho_flat.reshape(nrows, ncols, 3)
    valid_mask = observed.reshape(nrows, ncols)

    for band in range(3):
        filled_band, _ = _fill_holes_nearest(ortho[:, :, band].astype(np.float64), valid_mask)
        ortho[:, :, band] = np.clip(filled_band, 0, 255).astype(np.uint8)
    filled = ~valid_mask

    return ortho, transform, filled


def point_cloud_to_confidence_raster(
    pc: PointCloud, resolution_m: float, bounds: tuple[float, float, float, float] | None = None
) -> tuple[np.ndarray, GeoTransform, np.ndarray]:
    """The DSM's companion confidence raster: MIN confidence tier per cell.

    Meant to be written alongside ``point_cloud_to_dsm``'s output (same
    ``resolution_m``/``bounds`` -> identical grid) as a companion GeoTIFF,
    so an analyst can toggle a "how much of this surface model do I
    actually trust" overlay. MIN, not mean, for the same conservative
    reason ``fusion.filters.voxel_downsample`` uses MIN: a cell touched by
    even one inferred point is not a fully-measured cell.
    """
    if pc.confidence is None:
        raise ValueError("point_cloud_to_confidence_raster requires pc.confidence")

    xyz = np.asarray(pc.xyz, dtype=np.float64)
    confidence = np.asarray(pc.confidence)
    if xyz.shape[0] == 0:
        raise ValueError("point_cloud_to_confidence_raster requires a non-empty point cloud")

    nrows, ncols, row, col, transform = _grid_shape_and_indices(xyz[:, :2], resolution_m, bounds)
    flat_idx = row * ncols + col

    fill = np.iinfo(confidence.dtype).max if np.issubdtype(confidence.dtype, np.integer) else np.inf
    flat_conf = np.full(nrows * ncols, fill, dtype=np.float64)
    np.minimum.at(flat_conf, flat_idx, confidence.astype(np.float64))
    observed = np.zeros(nrows * ncols, dtype=bool)
    observed[flat_idx] = True

    conf_grid = flat_conf.reshape(nrows, ncols)
    valid_mask = observed.reshape(nrows, ncols)
    conf_grid = np.where(valid_mask, conf_grid, np.nan)

    conf_filled, filled_mask = _fill_holes_nearest(conf_grid, valid_mask)
    return conf_filled, transform, filled_mask


def _write_with_rasterio(path: Path, array: np.ndarray, transform: GeoTransform, crs: str | None, nodata: float | None) -> None:
    import rasterio
    from rasterio.transform import Affine

    # Affine(a, b, c, d, e, f): x = a*col + b*row + c ; y = d*col + e*row + f
    affine = Affine(
        transform.pixel_size_x, transform.rotation_x, transform.x_origin, transform.rotation_y, transform.pixel_size_y, transform.y_origin
    )

    if array.ndim == 2:
        height, width = array.shape
        count = 1
    else:
        height, width, count = array.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=affine,
        nodata=nodata,
    ) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            for band in range(count):
                dst.write(array[:, :, band], band + 1)


def _write_with_tifffile(path: Path, array: np.ndarray, transform: GeoTransform, crs: str | None, nodata: float | None) -> None:
    import tifffile

    del nodata  # tifffile's raw-tag path doesn't carry a nodata concept here
    # Minimal GeoTIFF georeferencing tags (assumes no rotation, i.e.
    # rotation_x == rotation_y == 0, the only case this project's
    # rasterizers ever produce): ModelPixelScaleTag (33550) +
    # ModelTiepointTag (33922) is the simplest valid way to place a raster
    # in world space without a full GeoKeyDirectoryTag. This does not
    # encode the CRS itself (that needs the full GeoKey machinery); ``crs``
    # is recorded as a plain ASCII ImageDescription tag instead so it is
    # at least discoverable, not silently dropped.
    if transform.rotation_x != 0.0 or transform.rotation_y != 0.0:
        logger.warning("tifffile GeoTIFF fallback does not support rotated transforms; georeferencing will be approximate")

    pixel_scale = (abs(transform.pixel_size_x), abs(transform.pixel_size_y), 0.0)
    tie_point = (0.0, 0.0, 0.0, transform.x_origin, transform.y_origin, 0.0)
    extratags = [
        (33550, "d", 3, pixel_scale, True),
        (33922, "d", 6, tie_point, True),
    ]
    description = f"drishti3d geotiff fallback (tifffile); crs={crs!r}"
    tifffile.imwrite(path, array, description=description, extratags=extratags)


def _write_npy_fallback(path: Path, array: np.ndarray, transform: GeoTransform, crs: str | None) -> None:
    stem = path.with_suffix("")
    np.save(stem.with_suffix(".npy"), array)
    stem.with_suffix(".tfw").write_text(transform.to_world_file_lines() + "\n")
    if crs is not None:
        import pyproj

        try:
            wkt = pyproj.CRS.from_user_input(crs).to_wkt()
        except pyproj.exceptions.CRSError:  # pragma: no cover - defensive, crs strings vary
            wkt = str(crs)
        stem.with_suffix(".prj").write_text(wkt)


def write_geotiff(
    path: str | Path, array: np.ndarray, transform: GeoTransform, crs: str | None = None, nodata: float | None = None
) -> str:
    """Write ``array`` as a georeferenced raster. Returns which backend actually wrote it.

    Tries, in order: ``"rasterio"`` (a real, spec-complete GeoTIFF),
    ``"tifffile"`` (a minimal-but-valid GeoTIFF via hand-written tags),
    ``"npy_fallback"`` (a plain ``.npy`` array plus a ``.tfw`` world file
    and, when ``crs`` is given, a ``.prj`` WKT file). Never raises just
    because neither optional geo dependency is installed -- the caller
    (and, transitively, the accuracy report) is expected to surface the
    returned backend name rather than assume "GeoTIFF" always means a
    real one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rasterio  # noqa: F401

        _write_with_rasterio(path, array, transform, crs, nodata)
        return "rasterio"
    except ImportError:
        pass

    try:
        import tifffile  # noqa: F401

        _write_with_tifffile(path, array, transform, crs, nodata)
        return "tifffile"
    except ImportError:
        pass

    logger.warning(
        "neither rasterio nor tifffile is installed; writing %s as .npy + .tfw (+ .prj) instead of a real GeoTIFF",
        path,
    )
    _write_npy_fallback(path, array, transform, crs)
    return "npy_fallback"
