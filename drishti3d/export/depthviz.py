"""Render per-keyframe depth, normal and class maps from the finished model.

Why render these at all
-----------------------
Depth and normal maps are the most direct visual evidence that a
reconstruction is real geometry rather than a plausible-looking point
smear. A depth map with crisp building edges and a flat ground gradient,
and a normal map where every roof is one solid colour and every wall
another, cannot be faked by a pipeline that merely triangulated some
features. They are also the fastest way to *see* a defect -- a smeared
roof edge or a noisy normal field shows up here long before it shows up in
an accuracy number.

Why re-project instead of saving the backbone's own depth
----------------------------------------------------------
``BackboneResult`` carries per-pixel depth straight from MapAnything, and
saving that would be cheaper. But it is the depth of a *single window's*
prediction, before submap merging, bundle adjustment, fusion, outlier
removal and georeferencing -- so it shows what the network guessed, not
what the system finally concluded. Rendering from the final cloud instead
means these maps depict the geometry that was actually exported, which is
the only version anyone can check against the deliverables.

The cost is that the render is a point splat, not a surface: a sparse
cloud leaves gaps. ``splat_px`` dilates each point into a small square to
close them, which is honest at the resolutions these maps are viewed at
and is why the output is documented as a visualisation, not a depth
measurement. Read depth off ``point_cloud.las``; read *impressions* off
these.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from drishti3d.semantics.labelling import project_points
from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)

__all__ = [
    "render_class_map",
    "render_depth_map",
    "render_keyframe_maps",
    "render_normal_map",
]

#: Points are splatted into a square this many pixels across. 3 closes
#: typical survey-density gaps at 1080p without visibly fattening edges.
_DEFAULT_SPLAT_PX = 3


def _zbuffer(
    xyz: np.ndarray,
    pose: Pose,
    intrinsics: CameraIntrinsics,
    *,
    splat_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-depth buffer and the index of the point that won each pixel.

    Returns ``(depth, index)``; ``index`` is ``-1`` where nothing landed.
    Keeping the winning index (not just the depth) is what lets the normal
    and class renderers reuse this single projection pass instead of each
    repeating it.
    """
    h, w = intrinsics.height, intrinsics.width
    depth = np.full(h * w, np.inf, dtype=np.float64)
    index = np.full(h * w, -1, dtype=np.int64)

    uv, z, in_view = project_points(xyz, pose, intrinsics)
    if not np.any(in_view):
        return depth.reshape(h, w), index.reshape(h, w)

    idx = np.flatnonzero(in_view)
    base_u = uv[idx, 0].astype(np.int32)
    base_v = uv[idx, 1].astype(np.int32)
    zz = z[idx]

    r = max(0, splat_px // 2)
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            u = np.clip(base_u + du, 0, w - 1)
            v = np.clip(base_v + dv, 0, h - 1)
            flat = v.astype(np.int64) * w + u.astype(np.int64)

            # Scatter-min with index tracking: np.minimum.at settles the
            # depth, then a second pass claims the index for exactly the
            # points whose depth survived. Doing it in two passes avoids a
            # Python loop over every point.
            np.minimum.at(depth, flat, zz)
            winners = zz <= depth[flat] + 1e-9
            index[flat[winners]] = idx[winners]

    return depth.reshape(h, w), index.reshape(h, w)


def render_depth_map(
    xyz: np.ndarray,
    pose: Pose,
    intrinsics: CameraIntrinsics,
    *,
    splat_px: int = _DEFAULT_SPLAT_PX,
    colormap: int = cv2.COLORMAP_TURBO,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Colourised depth image plus the ``(near, far)`` metres it spans.

    The range is returned rather than baked in because a colourised depth
    map with no scale is decorative, not informative: two maps of the same
    scene with different auto-ranges look different for no physical
    reason. Callers should print the range alongside the image.

    Percentile clipping (2-98) sets the range, so one stray far outlier
    cannot compress the entire useful depth band into two colours.
    """
    depth, _index = _zbuffer(xyz, pose, intrinsics, splat_px=splat_px)
    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8), (0.0, 0.0)

    near, far = np.percentile(depth[valid], [2, 98])
    if far <= near:
        far = near + 1e-3

    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - near) / (far - near), 0.0, 1.0)
    # Invert so near == warm: the eye reads warm-forward more naturally,
    # and it matches the convention every depth-map figure in the
    # literature uses.
    image = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), colormap)
    image[~valid] = 0
    return image, (float(near), float(far))


def render_normal_map(
    xyz: np.ndarray,
    normals: np.ndarray,
    pose: Pose,
    intrinsics: CameraIntrinsics,
    *,
    splat_px: int = _DEFAULT_SPLAT_PX,
    camera_frame: bool = True,
) -> np.ndarray:
    """Normals as RGB, the standard ``rgb = (n + 1) / 2`` encoding.

    ``camera_frame=True`` rotates world normals into the camera's frame
    first, which is what makes the map readable: in camera frame every
    surface facing the lens is the same colour regardless of where the
    drone was, so roofs across the whole flight render identically. In
    world frame the same roof changes colour as the aircraft turns, which
    looks like a defect and is not one.
    """
    _depth, index = _zbuffer(xyz, pose, intrinsics, splat_px=splat_px)
    h, w = intrinsics.height, intrinsics.width
    out = np.zeros((h, w, 3), dtype=np.uint8)

    hit = index >= 0
    if not hit.any():
        return out

    n = np.asarray(normals, dtype=np.float64)[index[hit]]
    if camera_frame:
        n = n @ pose.R  # world -> camera

    rgb = np.clip((n + 1.0) * 0.5, 0.0, 1.0) * 255.0
    # BGR for OpenCV's writer; the array is documented as an image, and
    # every other writer in this package hands cv2 BGR.
    out[hit] = rgb[:, ::-1].astype(np.uint8)
    return out


def render_class_map(
    xyz: np.ndarray,
    semantic_class: np.ndarray,
    pose: Pose,
    intrinsics: CameraIntrinsics,
    *,
    splat_px: int = _DEFAULT_SPLAT_PX,
) -> np.ndarray:
    """Per-pixel semantic class, coloured by ``semantics.classes.CLASS_COLORS``.

    Renders the *3D* labels back into the camera, not the original 2D
    mask. That difference is the point: this shows what survived
    multi-view voting, occlusion rejection and the agreement threshold, so
    comparing it against the raw segmentation of the same frame makes the
    voting's effect directly visible.
    """
    from drishti3d.semantics.classes import CLASS_COLORS

    _depth, index = _zbuffer(xyz, pose, intrinsics, splat_px=splat_px)
    h, w = intrinsics.height, intrinsics.width
    out = np.zeros((h, w, 3), dtype=np.uint8)

    hit = index >= 0
    if not hit.any():
        return out

    labels = np.asarray(semantic_class)[index[hit]]
    lut = np.zeros((max(CLASS_COLORS) + 1, 3), dtype=np.uint8)
    for value, (r, g, b) in CLASS_COLORS.items():
        lut[int(value)] = (b, g, r)  # BGR for cv2
    out[hit] = lut[np.clip(labels, 0, len(lut) - 1)]
    return out


def render_keyframe_maps(
    point_cloud,
    poses: list[Pose],
    keyframes: list,
    out_dir: str | Path,
    *,
    default_intrinsics: CameraIntrinsics | None = None,
    max_frames: int = 12,
    splat_px: int = _DEFAULT_SPLAT_PX,
    normals: np.ndarray | None = None,
) -> dict[str, Path]:
    """Write depth/normal/class maps for an evenly-spaced sample of keyframes.

    Evenly spaced, not the first ``max_frames``: the first N are the start
    of the flight and would show one end of the scene only.

    Returns ``{name: path}`` for what was actually written. Never raises
    for one frame's failure -- these are diagnostics, and losing the whole
    set because one keyframe lacked intrinsics would be a poor trade.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # A run whose geometry stage failed carries `point_cloud=None`, not an
    # empty cloud. Diagnostics must degrade to "nothing to draw" there --
    # raising an AttributeError on top of an already-failed run buries the
    # original error under a second, unrelated traceback.
    if point_cloud is None or getattr(point_cloud, "xyz", None) is None:
        logger.info("depthviz: no point cloud to render (geometry produced none)")
        return written

    xyz = np.asarray(point_cloud.xyz, dtype=np.float64)
    if len(xyz) == 0 or not poses or not keyframes:
        return written

    usable = [i for i in range(min(len(poses), len(keyframes)))]
    if not usable:
        return written
    if len(usable) > max_frames:
        step = len(usable) / float(max_frames)
        usable = [usable[int(i * step)] for i in range(max_frames)]

    semantic = getattr(point_cloud, "semantic_class", None)
    ranges: list[str] = []

    for i in usable:
        kf = keyframes[i]
        intr = getattr(kf, "intrinsics", None) or default_intrinsics
        if intr is None:
            continue
        try:
            depth_img, (near, far) = render_depth_map(xyz, poses[i], intr, splat_px=splat_px)
            path = out_dir / f"depth_{i:04d}.png"
            cv2.imwrite(str(path), depth_img)
            written[f"depth_{i:04d}"] = path
            ranges.append(f"{i:04d}: {near:.1f}-{far:.1f} m")

            if normals is not None and len(normals) == len(xyz):
                path = out_dir / f"normal_{i:04d}.png"
                cv2.imwrite(str(path), render_normal_map(xyz, normals, poses[i], intr, splat_px=splat_px))
                written[f"normal_{i:04d}"] = path

            if semantic is not None:
                path = out_dir / f"class_{i:04d}.png"
                cv2.imwrite(str(path), render_class_map(xyz, semantic, poses[i], intr, splat_px=splat_px))
                written[f"class_{i:04d}"] = path
        except Exception:
            logger.info("depthviz: failed to render maps for keyframe %d", i, exc_info=True)

    if ranges:
        # The scale bar, as text. A colourised depth map without its metre
        # range is decorative; this is what makes it readable.
        path = out_dir / "depth_ranges.txt"
        path.write_text(
            "Depth map colour ranges (2nd-98th percentile, metres from camera).\n"
            "Warm = near, cool = far. Rendered from the final exported cloud.\n\n" + "\n".join(ranges) + "\n"
        )
        written["depth_ranges"] = path

    return written
