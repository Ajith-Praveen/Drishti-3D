"""Complete building facades a single pass could never observe.

The problem
-----------
A nadir single-pass flight sees roofs and it sees ground. It barely sees
walls: a vertical surface is edge-on to a downward-looking camera, and the
few oblique glimpses it gets at the edges of frames are too grazing to
reconstruct. So the raw output is roofs floating over terrain with nothing
connecting them -- correct, but not a usable building model, and visibly
wrong to anyone who looks at it.

The completion
--------------
For each building footprint: take its roof outline, take the DTM elevation
underneath, and extrude the outline down to the ground. That turns a
floating roof into a closed building volume.

The rule that makes this legitimate
------------------------------------
**Every generated point is tagged ``Confidence.INFERRED`` and is never
mixed into the measured cloud.** This module returns the facade points as
their own ``PointCloud``; merging them is the caller's explicit choice, the
confidence tier travels with every point into LAS/PLY/GLB, and the report
card counts them separately.

That distinction is the whole basis on which this is defensible. A
pipeline that silently blends extruded geometry into measured geometry is
fabricating survey data: a user measuring a wall would get a number that
came from an assumption, with nothing to tell them so. A pipeline that
tags it is *modelling* -- the standard, accepted thing every GIS building
product does -- and the user can filter on confidence or ignore the layer
entirely.

Why semantics make this better founded than the usual version
--------------------------------------------------------------
The classical approach has to guess which roof outlines are buildings,
usually by thresholding height above ground and hoping trees do not
qualify. Trees usually do. This pipeline already knows: ``SemanticsStage``
labelled those points ``BUILDING``, and vegetation is labelled
``VEGETATION`` and excluded outright. Extruding a tree canopy into a solid
wall is the characteristic failure of footprint extrusion, and it simply
cannot happen here.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

from drishti3d.export.geotiff import _grid_shape_and_indices
from drishti3d.types import Confidence, PointCloud

logger = logging.getLogger(__name__)

__all__ = ["CompletionResult", "complete_facades"]

#: SemanticClass.BUILDING -- imported by value to keep this module free of
#: a circular import back through `semantics`.
_BUILDING = 2

#: Footprints smaller than this are rejected. A handful of stray
#: ``BUILDING`` points on a rooftop-coloured van should not become a
#: building; real structures occupy meaningful area.
_MIN_FOOTPRINT_M2 = 12.0

#: A footprint whose roof sits less than this above local ground is not a
#: building worth extruding -- it is a kerb, a low wall, or mislabelled
#: ground, and extruding it adds noise rather than structure.
_MIN_BUILDING_HEIGHT_M = 2.0


class CompletionResult:
    """Inferred facade geometry, kept deliberately separate from measurement."""

    def __init__(self, facade_points: PointCloud, stats: dict) -> None:
        #: Every point here has ``confidence == Confidence.INFERRED`` and
        #: ``semantic_class == BUILDING``. Never merge without preserving
        #: that tier -- see the module docstring.
        self.facade_points = facade_points
        self.stats = stats


def complete_facades(
    pc: PointCloud,
    dtm: np.ndarray,
    dtm_transform,
    *,
    resolution_m: float = 1.0,
    point_spacing_m: float = 0.5,
    min_footprint_m2: float = _MIN_FOOTPRINT_M2,
    min_height_m: float = _MIN_BUILDING_HEIGHT_M,
    max_points: int = 2_000_000,
) -> CompletionResult | None:
    """Extrude building roof outlines down to the DTM, as INFERRED points.

    Requires ``pc.semantic_class`` -- without labels there is no honest way
    to tell a roof from a tree canopy, and this returns ``None`` rather
    than guessing. That is deliberate: a version of this function that
    thresholds on height instead would extrude every tree in the scene into
    a solid block, which is the exact failure mode that makes naive
    footprint extrusion untrustworthy.

    ``dtm``/``dtm_transform`` come from ``export.terrain.dtm_from_point_cloud``.
    """
    if pc.semantic_class is None:
        logger.info("completion: no semantic labels; skipping facade completion (cannot distinguish roof from canopy)")
        return None

    xyz = np.asarray(pc.xyz, dtype=np.float64)
    semantic = np.asarray(pc.semantic_class)
    building = semantic == _BUILDING
    if not building.any():
        return None

    b_xyz = xyz[building]
    nrows, ncols, row, col, transform = _grid_shape_and_indices(xyz[:, :2], resolution_m, None)
    b_row, b_col = row[building], col[building]

    # Rasterise building occupancy, then label connected components: each
    # component is one contiguous structure's footprint.
    occupancy = np.zeros((nrows, ncols), dtype=bool)
    occupancy[b_row, b_col] = True
    # Close one-cell gaps so a footprint broken by a missing point does not
    # split into two half-buildings with a seam down the middle.
    occupancy = ndimage.binary_closing(occupancy, structure=np.ones((3, 3), dtype=bool))

    labels, n_components = ndimage.label(occupancy)
    if n_components == 0:
        return None

    # Roof height per cell: the max building-point Z that landed in it.
    roof = np.full((nrows, ncols), np.nan, dtype=np.float64)
    flat_roof = np.full(nrows * ncols, -np.inf, dtype=np.float64)
    np.maximum.at(flat_roof, b_row * ncols + b_col, b_xyz[:, 2])
    seen = np.isfinite(flat_roof) & (flat_roof > -np.inf)
    roof.reshape(-1)[seen] = flat_roof[seen]

    cell_area = resolution_m * resolution_m
    facades: list[np.ndarray] = []
    accepted = rejected_small = rejected_low = 0

    for comp in range(1, n_components + 1):
        mask = labels == comp
        area = float(mask.sum()) * cell_area
        if area < min_footprint_m2:
            rejected_small += 1
            continue

        # The outline is the boundary ring: cells in the footprint that
        # touch something outside it. Only the boundary gets a wall --
        # extruding the interior would fill the building with points.
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        boundary = mask & ~eroded
        if not boundary.any():
            boundary = mask

        b_rows, b_cols = np.nonzero(boundary)

        roof_z = roof[mask]
        roof_z = roof_z[np.isfinite(roof_z)]
        if roof_z.size == 0:
            continue
        # Median, not max: a single spurious high point (an antenna, a
        # mis-fused outlier) should not set the height of an entire wall.
        roof_level = float(np.median(roof_z))

        ground_z = dtm[mask]
        ground_z = ground_z[np.isfinite(ground_z)]
        if ground_z.size == 0:
            continue
        ground_level = float(np.median(ground_z))

        height = roof_level - ground_level
        if height < min_height_m:
            rejected_low += 1
            continue

        n_steps = max(2, int(np.ceil(height / point_spacing_m)))
        z_column = np.linspace(ground_level, roof_level, n_steps)

        x_world, y_world = transform.pixel_to_world(b_cols.astype(np.float64), b_rows.astype(np.float64))

        # Outer product: every boundary cell gets the full vertical column.
        n_boundary = len(b_rows)
        xs = np.repeat(x_world, n_steps)
        ys = np.repeat(y_world, n_steps)
        zs = np.tile(z_column, n_boundary)
        facades.append(np.column_stack([xs, ys, zs]))
        accepted += 1

        if sum(len(f) for f in facades) > max_points:
            logger.warning(
                "completion: facade point budget (%d) reached after %d structures; stopping early",
                max_points,
                accepted,
            )
            break

    if not facades:
        return None

    facade_xyz = np.vstack(facades)
    n = len(facade_xyz)

    facade_pc = PointCloud(
        xyz=facade_xyz,
        # Grey: these are modelled surfaces with no observed colour, and
        # giving them an invented photographic colour would make them
        # indistinguishable from measured geometry in a viewer.
        rgb=np.full((n, 3), 150, dtype=np.uint8),
        # The whole point. Every generated point declares itself.
        confidence=np.full(n, int(Confidence.INFERRED), dtype=np.uint8),
        semantic_class=np.full(n, _BUILDING, dtype=np.uint8),
        # Zero agreement: no view voted on these, because no view saw them.
        semantic_confidence=np.zeros(n, dtype=np.float32),
    )

    stats = {
        "facade_structures_completed": accepted,
        "facade_points_inferred": int(n),
        "facade_rejected_small_footprint": rejected_small,
        "facade_rejected_low_height": rejected_low,
        "facade_components_found": int(n_components),
    }
    logger.info(
        "completion: extruded %d building footprints into %d INFERRED facade points "
        "(%d rejected as too small, %d as too low)",
        accepted,
        n,
        rejected_small,
        rejected_low,
    )
    return CompletionResult(facade_pc, stats)
