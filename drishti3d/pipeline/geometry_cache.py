"""Save and restore GeometryStage's output so later stages can be re-run alone.

Why
---
GeometryStage is the expensive stage: on the development Mac it is ~60 of
a run's ~100 minutes, all of it backbone inference whose output does not
change between runs of the same keyframes. Every fusion, bundle-adjustment
or export change has so far cost a full re-run to test. Caching the
submaps turns those iterations into ~40-minute runs, and gives the product
a real feature for free: re-fuse or re-export a flight with different
settings without re-running the model.

What is cached
--------------
Everything GeometryStage writes to ``PipelineState``, plus the per-keyframe
conditioning poses it rewrites (``geometry.yaw_from_flow`` replaces
``Keyframe.pose`` in place, and downstream stages read the rewritten
version). The cache is keyed on what the output depends on: the keyframe
frame indices, the backbone name, and the working resolution. A cache
built from different keyframes is silently wrong, so the key is checked
and a mismatch is reported and ignored rather than loaded.

Format
------
A single pickle. This is a local, trusted, machine-specific scratch
artifact -- never an interchange format and never loaded from anywhere
the user did not point at explicitly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["GEOMETRY_CACHE_FILE", "cache_key", "load_geometry_cache", "save_geometry_cache"]

GEOMETRY_CACHE_FILE = "geometry_cache.pkl"
_VERSION = 2  # 2: Submap.view_index added; older caches lack it

#: State attributes GeometryStage owns. Declared fields and the ad-hoc
#: diagnostics attributes downstream stages and the report card read.
_STATE_ATTRS = (
    "windows",
    "submaps",
    "point_cloud",
    "poses",
    "geometry_camera_gps_enu",
    "geometry_conditioned_R",
    "geometry_dynamic_points_removed",
    "geometry_flight_profile",
    "geometry_gps_rmse_m",
    "geometry_junction_residuals",
    "geometry_merge_strategy",
    "geometry_merge_strategy_reason",
    "geometry_merge_strategy_warnings",
    "depth_anchor_diags",
    "depth_anchor_summary",
    "yaw_refinement",
)


def cache_key(state) -> dict:
    """What the geometry output depends on; must match exactly to load.

    The whole geometry and semantics config, not a hand-picked subset:
    any geometry knob (anchoring, edge mask, point budget, ...) changes the
    submaps, and semantics decides which pixels are excluded before they
    become points. A subset would let a changed setting load stale output
    and present it as the effect of the change.
    """
    from dataclasses import asdict

    return {
        "version": _VERSION,
        "frame_indices": [int(kf.frame_index) for kf in state.keyframes],
        "backbone": str(state.backbone_name),
        "geometry": asdict(state.config.geometry),
        "semantics": asdict(state.config.semantics),
        # The pose prior (matching + BA before geometry) changes the poses
        # everything in geometry is conditioned on.
        "matching": asdict(state.config.matching),
    }


def save_geometry_cache(cache_dir: str | Path, state, artifacts: dict, message: str) -> Path:
    """Write GeometryStage's output. Returns the file written."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": cache_key(state),
        "state": {name: getattr(state, name, None) for name in _STATE_ATTRS},
        "keyframe_poses": [kf.pose for kf in state.keyframes],
        "artifacts": artifacts,
        "message": message,
    }
    path = cache_dir / GEOMETRY_CACHE_FILE
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    logger.info("geometry cache: saved %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_geometry_cache(cache_dir: str | Path, state) -> tuple[dict, str] | None:
    """Restore GeometryStage's output into ``state`` if a matching cache exists.

    Returns ``(artifacts, message)`` on success, ``None`` when there is no
    cache or it was built from different inputs (logged with the reason).
    """
    path = Path(cache_dir) / GEOMETRY_CACHE_FILE
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        logger.warning("geometry cache: %s unreadable; ignoring", path, exc_info=True)
        return None

    expected = cache_key(state)
    found = payload.get("key", {})
    if found != expected:
        diff = {k: (found.get(k), expected[k]) for k in expected if found.get(k) != expected[k]}
        if "frame_indices" in diff:
            a, b = diff["frame_indices"]
            diff["frame_indices"] = (f"{len(a or [])} keyframes", f"{len(b)} keyframes")
        for section in ("geometry", "semantics", "matching"):
            if section in diff:
                old, new = diff[section]
                changed = sorted(k for k in set(old or {}) | set(new or {}) if (old or {}).get(k) != (new or {}).get(k))
                diff[section] = f"changed: {changed}"
        logger.warning("geometry cache: %s does not match this run (%s); ignoring", path, diff)
        return None

    for name, value in payload["state"].items():
        setattr(state, name, value)
    for kf, pose in zip(state.keyframes, payload["keyframe_poses"], strict=True):
        kf.pose = pose
    logger.info("geometry cache: loaded %s (%d submaps)", path, len(state.submaps))
    return payload["artifacts"], f"loaded from cache {path}: {payload['message']}"
