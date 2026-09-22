"""Geometry stage: triage's keyframes -> camera poses + dense point maps.

Swappable backbone (``geometry.backbone``, default MapAnything -- see
``geometry.mapanything``), planned into overlapping windows to fit a
memory-starved GPU (``geometry.windows``), reconstructed per window as a
``Submap`` and stitched into one global model (``geometry.submap``).
"""

from drishti3d.geometry.backbone import (
    Backbone,
    BackboneResult,
    NullBackbone,
    get_backbone,
    register_backbone,
)
from drishti3d.geometry.flight_profile import FlightProfile, analyze_flight_profile
from drishti3d.geometry.submap import (
    Sim3,
    alignment_residuals,
    merge_submaps,
    strategy_report,
    umeyama_alignment,
    umeyama_fixed_rotation,
)
from drishti3d.geometry.windows import (
    Window,
    estimate_memory,
    max_window_for_budget,
    plan_window_size,
    plan_windows,
)

__all__ = [
    "Backbone",
    "BackboneResult",
    "FlightProfile",
    "NullBackbone",
    "Sim3",
    "Window",
    "alignment_residuals",
    "analyze_flight_profile",
    "estimate_memory",
    "get_backbone",
    "max_window_for_budget",
    "merge_submaps",
    "plan_window_size",
    "plan_windows",
    "register_backbone",
    "strategy_report",
    "umeyama_alignment",
    "umeyama_fixed_rotation",
]
