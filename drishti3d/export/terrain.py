"""Bare-earth terrain: DTM, and the canopy/building height model above it.

DSM vs DTM
----------
The DSM (``export.geotiff.point_cloud_to_dsm``) is the *surface*: the top
of whatever is there, roof or treetop or road. The DTM is the *ground*:
what the terrain would look like with everything on it removed. Almost
every question an operator actually asks needs the second one -- how deep
is that cutting, will this route flood, how tall is that building -- and
none of them can be answered from a DSM alone.

Why this project's DTM is better founded than the usual one
-----------------------------------------------------------
The classical approach (progressive morphological filtering, cloth
simulation) infers ground purely from shape: it assumes ground is locally
low and smooth, and iteratively rejects points that rise too steeply over
too short a distance. It works, but it has no idea what anything *is*, so
it fails in the predictable places -- a large flat rooftop looks exactly
like ground, a gentle embankment looks exactly like a building edge.

This pipeline already knows the answer. ``SemanticsStage`` has labelled
every point, so ``TERRAIN``/``ROAD``/``WATER`` points *are* the ground and
``BUILDING``/``VEGETATION`` points are not -- no shape heuristic required.
``dtm_from_point_cloud`` uses those labels when they exist and falls back
to morphological filtering when they do not, and it records which of the
two it used rather than presenting them as interchangeable.

Honesty about interpolation
---------------------------
Under a building there is no ground observation. There cannot be: the
building is in the way. The DTM there is interpolated from the ground
around it, and ``dtm_from_point_cloud`` returns an ``interpolated_mask``
marking every such cell. That mask is not a debugging aid -- it is the
difference between a measured elevation and a guess, and it is exported as
its own raster so nobody measures a cutting depth off a number that was
invented by a nearest-neighbour fill.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from drishti3d.export.geotiff import (
    GeoTransform,
    _fill_holes_nearest,
    _grid_shape_and_indices,
)
from drishti3d.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "DtmResult",
    "dtm_from_point_cloud",
    "height_above_ground",
    "morphological_ground_mask",
]

#: Classes whose points lie on the ground surface itself. WATER is here
#: because a water surface *is* the terrain surface at that location --
#: excluding it would punch interpolated holes through every river.
_GROUND_CLASSES = (1, 3, 6)  # TERRAIN, ROAD, WATER

#: Largest structure (metres) the morphological fallback should be able to
#: remove. Set from the scene: too small and big roofs survive as terrain,
#: too large and real hills get flattened. 30 m clears a typical urban
#: building footprint while preserving landforms.
_MAX_STRUCTURE_M = 30.0

#: Slope (m/m) the fallback tolerates between neighbouring ground cells.
#: 0.3 (~17 deg) passes road embankments and natural hillsides while
#: rejecting building walls, which are effectively vertical.
_MAX_GROUND_SLOPE = 0.3

#: Minimum share of points that must carry a ground class before the
#: semantic path is trusted over the morphological one. 2% is low enough
#: that any genuinely ground-bearing scene clears it, and high enough to
#: reject the "segmenter labelled 40 points TERRAIN out of 10 million"
#: case that otherwise yields a DTM built almost entirely from
#: interpolation.
_MIN_SEMANTIC_GROUND_FRACTION = 0.02


class DtmResult:
    """A DTM plus the provenance needed to trust any number read off it."""

    def __init__(
        self,
        dtm: np.ndarray,
        transform: GeoTransform,
        interpolated_mask: np.ndarray,
        method: str,
        stats: dict,
    ) -> None:
        self.dtm = dtm
        self.transform = transform
        #: True where no ground point fell in the cell and the elevation
        #: was interpolated -- i.e. under buildings and dense canopy.
        self.interpolated_mask = interpolated_mask
        #: ``"semantic"`` or ``"morphological"``. Never blend the two
        #: silently; a reader must be able to tell which produced a value.
        self.method = method
        self.stats = stats


def morphological_ground_mask(
    xyz: np.ndarray,
    resolution_m: float,
    *,
    max_structure_m: float = _MAX_STRUCTURE_M,
    max_slope: float = _MAX_GROUND_SLOPE,
) -> np.ndarray:
    """Shape-only ground classification, for when no semantic labels exist.

    A progressive morphological opening on the minimum-Z surface: a
    grayscale opening with a structuring element of radius ``r`` removes
    any raised feature narrower than ``2r``, so running it at increasing
    radii and keeping points close to the opened surface strips buildings
    and trees while leaving landforms. The slope term then rejects points
    that sit far above the local opened surface relative to the cell size,
    which is what separates a wall from a hillside.

    Returns ``(N,)`` bool over the input points. This is the *fallback*;
    prefer semantic labels when the pipeline has them (see module
    docstring for why).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)

    nrows, ncols, row, col, _transform = _grid_shape_and_indices(xyz[:, :2], resolution_m, None)
    flat_idx = row * ncols + col

    # Minimum-Z surface: the lowest thing seen in each cell is the best
    # single ground candidate available without labels.
    min_surface = np.full(nrows * ncols, np.inf, dtype=np.float64)
    np.minimum.at(min_surface, flat_idx, xyz[:, 2])
    observed = np.isfinite(min_surface)
    if not observed.any():
        return np.zeros(len(xyz), dtype=bool)

    grid = min_surface.reshape(nrows, ncols)
    filled, _ = _fill_holes_nearest(
        np.where(observed.reshape(nrows, ncols), grid, np.nan),
        observed.reshape(nrows, ncols),
    )

    # Progressive opening: double the window until it exceeds the largest
    # structure we intend to remove. Each opening strips features narrower
    # than the window, so by the end `opened` approximates bare earth.
    opened = filled
    window = 3
    while window * resolution_m < max_structure_m:
        opened = ndimage.grey_opening(opened, size=(window, window))
        window = window * 2 + 1

    # Classify PER POINT, not per cell. A cell containing both ground and
    # a roof must not mark its roof points as ground just because some
    # ground also landed there -- that is precisely the case a mixed cell
    # presents, and a per-cell verdict gets it wrong every time.
    #
    # The threshold is a local elevation tolerance, deliberately NOT scaled
    # by the final (large) window size: by the end of the loop the window
    # spans the largest structure being removed, and multiplying a slope by
    # that distance yields a tolerance of tens of metres, which admits
    # entire buildings as "ground".
    threshold = max(max_slope * resolution_m, 0.0) + 0.5
    ground_surface = opened.reshape(-1)[flat_idx]
    return (xyz[:, 2] - ground_surface) <= threshold


def dtm_from_point_cloud(
    pc: PointCloud,
    resolution_m: float,
    bounds: tuple[float, float, float, float] | None = None,
    *,
    prefer_semantic: bool = True,
) -> DtmResult:
    """Rasterise a bare-earth DTM: minimum-Z of ground points per cell.

    Uses ``pc.semantic_class`` when present (ground == terrain/road/water),
    otherwise falls back to ``morphological_ground_mask``. Minimum-Z rather
    than mean: any ground point that is actually a low bush or a kerb
    biases a mean upward, and the lowest return in a cell is the best
    available estimate of the surface beneath.
    """
    xyz = np.asarray(pc.xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise ValueError("dtm_from_point_cloud requires a non-empty point cloud")

    semantic = pc.semantic_class

    # Requiring *some* ground-class points is not enough -- it has to be
    # enough to define a surface. On a nadir aerial scene an out-of-domain
    # segmenter can label a handful of points TERRAIN while classifying the
    # actual ground as unlabelled, which passed an `any()` test and then
    # produced a DTM that was 99.97% nearest-neighbour interpolation: a
    # surface invented from a few dozen samples, reported as measured.
    # Below this fraction the morphological filter -- which uses every
    # point's shape rather than a label -- is strictly more trustworthy.
    ground_fraction = 0.0
    if semantic is not None:
        ground_fraction = float(np.isin(np.asarray(semantic), _GROUND_CLASSES).mean())
    use_semantic = prefer_semantic and semantic is not None and ground_fraction >= _MIN_SEMANTIC_GROUND_FRACTION

    if use_semantic:
        ground = np.isin(np.asarray(semantic), _GROUND_CLASSES)
        method = "semantic"
    else:
        ground = morphological_ground_mask(xyz, resolution_m)
        method = "morphological"
        if semantic is not None:
            logger.info(
                "terrain: only %.2f%% of points carry a ground class (need %.0f%%); "
                "falling back to morphological ground filtering",
                ground_fraction * 100.0,
                _MIN_SEMANTIC_GROUND_FRACTION * 100.0,
            )

    nrows, ncols, row, col, transform = _grid_shape_and_indices(xyz[:, :2], resolution_m, bounds)

    if not ground.any():
        raise ValueError(f"no ground points identified by the {method} classifier; cannot build a DTM")

    g_idx = row[ground] * ncols + col[ground]
    flat = np.full(nrows * ncols, np.inf, dtype=np.float64)
    np.minimum.at(flat, g_idx, xyz[ground, 2])

    observed = np.isfinite(flat)
    grid = np.where(observed, flat, np.nan).reshape(nrows, ncols)
    dtm, interpolated_mask = _fill_holes_nearest(grid, observed.reshape(nrows, ncols))

    stats = {
        "dtm_method": method,
        "ground_points": int(ground.sum()),
        "ground_point_pct": round(100.0 * float(ground.mean()), 2),
        "dtm_cells": int(nrows * ncols),
        "dtm_interpolated_pct": round(100.0 * float(interpolated_mask.mean()), 2),
    }
    return DtmResult(dtm, transform, interpolated_mask, method, stats)


def height_above_ground(
    dsm: np.ndarray,
    dtm: np.ndarray,
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """``DSM - DTM``: object height above bare earth, per cell.

    This is the normalised height model -- canopy height over vegetation,
    building height over structures, ~0 over open ground. It is what makes
    "how tall is that" answerable directly from the rasters, and it is
    strictly more useful than either input alone.

    Small negative values occur where the DTM was interpolated slightly
    above a genuinely lower surface return; ``clip_negative`` floors them
    at zero, since a surface below bare earth is a artefact of
    interpolation, not a measurement.
    """
    if dsm.shape != dtm.shape:
        raise ValueError(f"dsm shape {dsm.shape} does not match dtm shape {dtm.shape}")
    height = np.asarray(dsm, dtype=np.float64) - np.asarray(dtm, dtype=np.float64)
    return np.clip(height, 0.0, None) if clip_negative else height
