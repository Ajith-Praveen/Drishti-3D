"""Semantic understanding: what each reconstructed surface actually *is*.

Geometry alone answers "where is there a surface". This package answers
"and what is it" -- terrain, building, road, vegetation, infrastructure,
water, obstacle -- and, just as importantly, "what is it that should not be
in the model at all": vehicles, people, and sky.

Three modules, in pipeline order:

- ``classes``: the project-owned taxonomy every backend translates into.
  Integer values here are a file-format contract (they are written into
  exported LAS/PLY) -- append, never renumber.
- ``segmenter``: swappable per-frame 2D segmentation, shaped exactly like
  ``geometry.backbone`` (ABC + registry + lazy import + Null fallback), so
  the pipeline runs identically with or without a model installed.
- ``labelling``: lifts those 2D masks to per-point 3D labels by weighted
  multi-view voting with occlusion rejection, and keeps the per-point
  agreement ratio rather than hiding it.

Everything here is optional. With no ``transformers``/``torch`` present,
``NullSegmenter`` runs, every point comes back ``UNLABELLED``, and the
report card reports that honestly instead of inventing a class breakdown.
"""

from drishti3d.semantics.classes import (
    CLASS_COLORS,
    CLASS_NAMES,
    DYNAMIC_CLASSES,
    EXCLUDED_CLASSES,
    RECONSTRUCTION_TARGETS,
    SemanticClass,
    class_histogram,
)
from drishti3d.semantics.labelling import (
    LabelVoteResult,
    label_points,
    project_points,
    visible_mask,
)
from drishti3d.semantics.segmenter import (
    NullSegmenter,
    SegFormerSegmenter,
    SegmentationResult,
    Segmenter,
    create_segmenter,
    register_segmenter,
)

__all__ = [
    "CLASS_COLORS",
    "CLASS_NAMES",
    "DYNAMIC_CLASSES",
    "EXCLUDED_CLASSES",
    "RECONSTRUCTION_TARGETS",
    "LabelVoteResult",
    "NullSegmenter",
    "SegFormerSegmenter",
    "SegmentationResult",
    "Segmenter",
    "SemanticClass",
    "class_histogram",
    "create_segmenter",
    "label_points",
    "project_points",
    "register_segmenter",
    "visible_mask",
]
