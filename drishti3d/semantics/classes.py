"""The canonical semantic taxonomy every reconstruction is labelled against.

Why a project-owned taxonomy at all
-----------------------------------
Segmentation checkpoints disagree about labels. ADE20K has 150 classes,
Cityscapes has 19, an aerial-specific model might have 6. None of them is
*our* answer: what this project has to report is a fixed, small set of
classes that maps one-to-one onto the reconstruction targets the system is
required to produce -- terrain, building, road, vegetation, obstacle -- plus
the classes we need in order to *exclude* geometry rather than keep it
(vehicle, person, sky).

So every segmenter backend (see ``semantics.segmenter``) is responsible for
translating its own label space into ``SemanticClass`` below. Downstream
code -- fusion, export, the report card -- only ever sees ``SemanticClass``,
and a backbone swap can never silently change the meaning of a stored
``semantic_class`` value in an exported LAS/PLY.

Dynamic vs. static is a property of the taxonomy, not of a config flag
---------------------------------------------------------------------
``DYNAMIC_CLASSES`` is what makes the "dynamic objects" problem tractable:
a moving car is not removed because it was observed to move (we mostly
cannot observe that reliably from a single pass at 5 fps), but because it
is a car, and cars are not terrain. That is a deliberately stronger filter
than motion detection -- it also removes *parked* cars, which are equally
not part of the scene a mapping product should contain, and which no
motion-based method can catch. ``fusion`` and ``geometry`` both consult
``EXCLUDED_CLASSES``; nothing else decides this.

``UNLABELLED`` is not a class, it is an admission
-------------------------------------------------
Value ``0`` means "no segmenter ran, or it ran and this pixel's source
label had no honest mapping into our taxonomy". It is never guessed into a
real class. A point cloud whose ``semantic_class`` is entirely ``0`` is a
point cloud that was never segmented, and the report card says so rather
than showing a fabricated 100%-terrain breakdown.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "ADE20K_TO_CANONICAL",
    "ASPRS_FROM_CANONICAL",
    "CLASS_COLORS",
    "CLASS_NAMES",
    "DYNAMIC_CLASSES",
    "EXCLUDED_CLASSES",
    "RECONSTRUCTION_TARGETS",
    "SemanticClass",
    "class_histogram",
    "map_label_array",
]

import numpy as np


class SemanticClass(IntEnum):
    """Stored as ``uint8`` in ``PointCloud.semantic_class`` and in LAS/PLY/GLB.

    These integer values are a **file format contract**: they are written
    into exported LAS extra dimensions and PLY scalar properties that
    outlive any single run. Append new members; never renumber existing
    ones.
    """

    UNLABELLED = 0
    TERRAIN = 1
    BUILDING = 2
    ROAD = 3
    VEGETATION = 4
    INFRASTRUCTURE = 5
    WATER = 6
    VEHICLE = 7
    PERSON = 8
    SKY = 9
    OBSTACLE = 10


CLASS_NAMES: dict[int, str] = {
    SemanticClass.UNLABELLED: "unlabelled",
    SemanticClass.TERRAIN: "terrain",
    SemanticClass.BUILDING: "building",
    SemanticClass.ROAD: "road",
    SemanticClass.VEGETATION: "vegetation",
    SemanticClass.INFRASTRUCTURE: "infrastructure",
    SemanticClass.WATER: "water",
    SemanticClass.VEHICLE: "vehicle",
    SemanticClass.PERSON: "person",
    SemanticClass.SKY: "sky",
    SemanticClass.OBSTACLE: "obstacle",
}

# Display colours (RGB, 0-255) for the "colour by class" viewport mode and
# for the classified orthomosaic. Chosen to stay distinguishable in
# greyscale print and for the two commonest colour-vision deficiencies:
# terrain/road/building are separated by lightness, not just hue.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    SemanticClass.UNLABELLED: (128, 128, 128),
    SemanticClass.TERRAIN: (160, 120, 80),
    SemanticClass.BUILDING: (220, 90, 70),
    SemanticClass.ROAD: (70, 70, 80),
    SemanticClass.VEGETATION: (60, 160, 70),
    SemanticClass.INFRASTRUCTURE: (240, 190, 60),
    SemanticClass.WATER: (60, 130, 220),
    SemanticClass.VEHICLE: (200, 60, 200),
    SemanticClass.PERSON: (255, 240, 100),
    SemanticClass.SKY: (200, 225, 255),
    SemanticClass.OBSTACLE: (150, 90, 200),
}

#: Classes whose geometry is transient scene *content*, not scene
#: *structure*. Removed before fusion -- see module docstring.
DYNAMIC_CLASSES: frozenset[int] = frozenset(
    {
        SemanticClass.VEHICLE,
        SemanticClass.PERSON,
    }
)

#: Everything dropped before a point ever reaches the TSDF. ``SKY`` is here
#: for a different reason than the dynamic classes: sky pixels have no
#: surface at all, and a depth backbone asked for their depth returns
#: whatever the far plane happens to be -- pure noise that would otherwise
#: dominate the point cloud's bounding box and wreck the DSM's vertical
#: range.
EXCLUDED_CLASSES: frozenset[int] = DYNAMIC_CLASSES | frozenset({SemanticClass.SKY})

#: The classes the problem this system solves explicitly asks to
#: reconstruct. Used by the report card to state coverage per required
#: target rather than only in aggregate.
RECONSTRUCTION_TARGETS: tuple[int, ...] = (
    SemanticClass.TERRAIN,
    SemanticClass.BUILDING,
    SemanticClass.ROAD,
    SemanticClass.VEGETATION,
    SemanticClass.INFRASTRUCTURE,
    SemanticClass.WATER,
    SemanticClass.OBSTACLE,
)


# ----------------------------------------------------------------------
# ADE20K (150 classes, `scene_parse_150`) -> SemanticClass
#
# ADE20K is the default because it is the only widely-available
# checkpoint family that covers building / road / vegetation / vehicle /
# person / water in ONE forward pass. It is, however, trained almost
# entirely on ground-level photography: a nadir drone frame is out of its
# training distribution, and its per-pixel accuracy on such frames is
# materially worse than the benchmark numbers its model card advertises.
# That is why `labelling.label_points` votes across many views and keeps a
# per-point agreement ratio rather than trusting any single mask, and why
# `SemanticsConfig.min_vote_ratio` exists. Do not quote ADE20K mIoU as if
# it were this system's classification accuracy; it is not.
#
# Indices not listed map to UNLABELLED -- indoor furniture and the like,
# which cannot legitimately appear in an aerial survey and whose presence
# in a mask is itself a signal that the model is confused.
# ----------------------------------------------------------------------
_ADE20K_GROUPS: dict[int, tuple[int, ...]] = {
    SemanticClass.TERRAIN: (3, 13, 16, 29, 46, 68, 91, 94),
    SemanticClass.BUILDING: (0, 1, 8, 14, 25, 48, 79, 86),
    SemanticClass.ROAD: (6, 11, 52, 53, 54, 59, 121),
    SemanticClass.VEGETATION: (4, 9, 17, 66, 72),
    SemanticClass.INFRASTRUCTURE: (32, 38, 42, 43, 61, 84, 87, 93, 95, 104, 136, 140, 149),
    SemanticClass.WATER: (21, 26, 60, 109, 113, 128),
    SemanticClass.VEHICLE: (20, 76, 80, 83, 90, 102, 103, 116, 122, 127),
    SemanticClass.PERSON: (12, 126),
    SemanticClass.SKY: (2,),
    SemanticClass.OBSTACLE: (34, 41, 55, 69, 111, 114, 132, 138),
}

#: Dense ``(150,) uint8`` lookup table: ``ADE20K_TO_CANONICAL[ade_index]``.
#: Built as an array rather than a dict so mapping a full-resolution mask
#: is a single vectorised gather instead of a Python loop over ~2M pixels.
ADE20K_TO_CANONICAL: np.ndarray = np.full(150, SemanticClass.UNLABELLED, dtype=np.uint8)
for _canonical, _indices in _ADE20K_GROUPS.items():
    ADE20K_TO_CANONICAL[list(_indices)] = _canonical
del _canonical, _indices


# ----------------------------------------------------------------------
# SemanticClass -> ASPRS LAS standard classification codes
#
# Written into LAS's *standard* `classification` field, in addition to the
# full-fidelity `semantic_class` extra dimension. The point is
# interoperability: a LAS whose classification field is populated with real
# ASPRS codes opens in CloudCompare, QGIS, LAStools and every GIS on the
# planet already colour-coded and filterable, with no knowledge of this
# project at all.
#
# ASPRS's standard set is narrower than ours, so this mapping is lossy on
# purpose and in one direction only. Classes with no honest ASPRS
# equivalent -- infrastructure, obstacle, vehicle, person -- map to 1
# ("unclassified") rather than to a plausible-looking near-miss: writing
# vehicles as 6/"building" would produce a file that lies to every tool
# that reads it. Nothing is lost, because `semantic_class` carries the
# exact class alongside.
# ----------------------------------------------------------------------
ASPRS_UNCLASSIFIED = 1

_ASPRS_FROM_CANONICAL: dict[int, int] = {
    SemanticClass.UNLABELLED: ASPRS_UNCLASSIFIED,
    SemanticClass.TERRAIN: 2,  # ground
    SemanticClass.BUILDING: 6,  # building
    SemanticClass.ROAD: 11,  # road surface
    SemanticClass.VEGETATION: 5,  # high vegetation
    SemanticClass.WATER: 9,  # water
    SemanticClass.INFRASTRUCTURE: ASPRS_UNCLASSIFIED,
    SemanticClass.OBSTACLE: ASPRS_UNCLASSIFIED,
    SemanticClass.VEHICLE: ASPRS_UNCLASSIFIED,
    SemanticClass.PERSON: ASPRS_UNCLASSIFIED,
    SemanticClass.SKY: ASPRS_UNCLASSIFIED,
}

#: Dense ``(len(SemanticClass),) uint8`` lookup, same vectorised-gather
#: reasoning as ``ADE20K_TO_CANONICAL``.
ASPRS_FROM_CANONICAL: np.ndarray = np.full(len(SemanticClass), ASPRS_UNCLASSIFIED, dtype=np.uint8)
for _cls, _code in _ASPRS_FROM_CANONICAL.items():
    ASPRS_FROM_CANONICAL[int(_cls)] = _code
del _cls, _code


def map_label_array(labels: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """Translate a backend's native label mask into ``SemanticClass`` values.

    ``labels`` is any integer array of native class indices; ``lookup`` is
    a dense ``uint8`` table like ``ADE20K_TO_CANONICAL``. Indices beyond
    ``lookup``'s length become ``UNLABELLED`` rather than raising -- a
    checkpoint with more classes than its documented label space is a
    checkpoint we should refuse to interpret, not one we should crash on.
    """
    labels = np.asarray(labels)
    out = np.full(labels.shape, SemanticClass.UNLABELLED, dtype=np.uint8)
    in_range = (labels >= 0) & (labels < len(lookup))
    out[in_range] = lookup[labels[in_range]]
    return out


def class_histogram(semantic_class: np.ndarray | None, *, as_percent: bool = True) -> dict[str, float]:
    """``{class_name: share}`` over ``semantic_class``, for the report card.

    Returns an empty dict for ``None`` or an empty array -- the caller
    (``export.report.build_report``) turns that into ``"not computed"``
    rather than a row of zeroes, which would read as "we looked and found
    no buildings" instead of the truth, "we never classified anything".
    Classes absent from the cloud are omitted, not reported as ``0.0``.
    """
    if semantic_class is None:
        return {}
    arr = np.asarray(semantic_class).ravel()
    if arr.size == 0:
        return {}

    counts = np.bincount(arr.astype(np.int64), minlength=len(SemanticClass))
    total = float(arr.size)
    out: dict[str, float] = {}
    for value, name in CLASS_NAMES.items():
        n = int(counts[value]) if value < len(counts) else 0
        if n == 0:
            continue
        out[name] = round(100.0 * n / total, 2) if as_percent else float(n)
    return out
