"""Bake real photographic texture onto the fused mesh.

What "textured" has to mean
---------------------------
A mesh with per-vertex colour is not a textured mesh. At the vertex
densities a TSDF produces, per-vertex colour is a low-pass filter over the
imagery: roof markings, lane lines, kerbs and signage -- exactly the detail
an operator zooms in to read -- are averaged away between vertices. A
textured mesh keeps a full-resolution image atlas and samples it across
each face, so detail is limited by the source video, not by mesh
resolution.

This module produces the second kind.

Pipeline
--------
1. **UV unwrap** (``xatlas``): cut the mesh into charts and pack them into
   a unit square. This may duplicate vertices along chart seams -- unwrap
   returns its own vertex mapping, and every per-vertex attribute has to
   be re-indexed through it. Failing to do that is the classic way to ship
   a mesh whose colours no longer match its geometry.
2. **View selection per texel**: for every texel, find which keyframes see
   the surface point it corresponds to, and score them.
3. **Blend**: weighted average of the best views, not a hard argmax.
   Argmax produces visible seams wherever the winning view changes; a
   weighted blend across the top-``k`` views fades between them instead.

Texel scoring -- the same three terms as ``semantics.labelling``
-----------------------------------------------------------------
Obliquity, range, and per-view quality. Deliberately the same reasoning as
the semantic vote, because it is the same underlying question ("how well
does this camera actually see this surface point"). The third term differs:
here it is the keyframe's own ``blur_score`` from triage, since a motion
-blurred frame that nonetheless sees a surface face-on and up close would
otherwise win and smear the atlas.

Occlusion uses ``semantics.labelling.visible_mask`` -- shared, not
reimplemented, so the texture baker and the semantic vote can never
disagree about what is visible from where.

Everything here degrades
------------------------
Without ``xatlas``, ``bake_texture`` returns ``None`` and the caller falls
back to per-vertex colour. The mesh is still exported, still correct, just
not texture-mapped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from drishti3d.semantics.labelling import project_points, visible_mask
from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)

__all__ = ["TextureBakeResult", "bake_texture", "is_available", "unwrap_uv"]

#: How many views may contribute to one texel. 4 is enough to hide seams
#: without turning genuine detail into a soft average of half the flight.
_BLEND_VIEWS = 4

#: Texels with no view at all are filled by inpainting from their
#: neighbours. This is cosmetic, not evidentiary: it stops the atlas
#: showing hard black holes at chart edges where bilinear sampling would
#: otherwise bleed background into the surface.
_INPAINT_RADIUS_PX = 3


def is_available() -> bool:
    """Whether UV unwrapping can run (``xatlas`` present). Never raises."""
    try:
        import xatlas  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class TextureBakeResult:
    """A textured mesh: re-indexed geometry plus the atlas it samples.

    ``vertices``/``faces`` are NOT the inputs -- unwrapping duplicates
    vertices along seams, so these are the unwrapped topology and the only
    ones the ``uv`` array is valid for. Writers must use these, not the
    originals.
    """

    vertices: np.ndarray  # (V', 3) float
    faces: np.ndarray  # (F, 3) int
    uv: np.ndarray  # (V', 2) float in [0, 1]
    texture: np.ndarray  # (S, S, 3) uint8, RGB
    vertex_map: np.ndarray  # (V',) int -- index into the ORIGINAL vertices
    stats: dict


def unwrap_uv(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """UV-unwrap with xatlas. Returns ``(vertex_map, faces, uv)`` or ``None``.

    ``vertex_map`` maps each *new* vertex back to the original index it was
    split from; ``vertices[vertex_map]`` reconstructs the unwrapped
    geometry, and the same gather re-indexes any per-vertex attribute
    (colour, confidence, semantic class) so it stays attached to the right
    surface point.
    """
    try:
        import xatlas
    except ImportError:
        logger.info("texture: xatlas not installed; skipping UV unwrap (mesh keeps per-vertex colour)")
        return None

    try:
        vmap, indices, uvs = xatlas.parametrize(
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32),
        )
    except Exception:
        logger.warning("texture: xatlas.parametrize failed; falling back to per-vertex colour", exc_info=True)
        return None

    return np.asarray(vmap, dtype=np.int64), np.asarray(indices, dtype=np.int64), np.asarray(uvs, dtype=np.float64)


def bake_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: list[tuple[Pose, CameraIntrinsics, np.ndarray, float]],
    *,
    texture_size: int = 4096,
    occlusion_tolerance_m: float = 0.5,
    blend_views: int = _BLEND_VIEWS,
    exposure_compensation: bool = True,
) -> TextureBakeResult | None:
    """Unwrap ``(vertices, faces)`` and bake a photographic atlas from ``views``.

    ``views`` is ``(pose, intrinsics, image_bgr, quality)`` per keyframe,
    where ``quality`` is a positive scalar (triage's ``blur_score``) used
    as the per-view weight term. Images must be at the resolution their
    intrinsics describe.

    Returns ``None`` when unwrapping is unavailable or fails, which the
    caller must treat as "export per-vertex colour instead" rather than as
    an error.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0 or not views:
        return None

    unwrapped = unwrap_uv(vertices, faces)
    if unwrapped is None:
        return None
    vertex_map, new_faces, uv = unwrapped

    new_vertices = vertices[vertex_map]

    # Rasterise every triangle into the atlas once, recording for each
    # texel the 3D point it corresponds to. Doing this up front means the
    # expensive per-view projection below runs over a flat array of texel
    # positions instead of per-triangle-per-view.
    texel_xyz, texel_mask = _rasterize_positions(new_vertices, new_faces, uv, texture_size)
    covered = np.flatnonzero(texel_mask.reshape(-1))
    if covered.size == 0:
        logger.warning("texture: UV rasterisation covered no texels; skipping bake")
        return None

    points = texel_xyz.reshape(-1, 3)[covered]

    # Accumulate a weighted colour sum per texel, keeping only each texel's
    # best `blend_views` contributions. Implemented as a running top-k
    # rather than storing every view's contribution for every texel, which
    # at 4096^2 texels would be gigabytes.
    top_w = np.zeros((points.shape[0], blend_views), dtype=np.float32)
    top_rgb = np.zeros((points.shape[0], blend_views, 3), dtype=np.float32)

    normals = _vertex_normals(new_vertices, new_faces)
    texel_normals = _rasterize_normals(normals, new_faces, uv, texture_size).reshape(-1, 3)[covered]

    # Exposure gains are solved BEFORE any sampling, from the same texel
    # positions the bake will use, so a gain is fitted on exactly the
    # surface points that will carry it. See ingest.photometric for why
    # this is a single scalar per view and not a colour transform.
    gains = np.ones(len(views), dtype=np.float64)
    if exposure_compensation and len(views) > 1:
        gains = _solve_gains_for_views(points, views, occlusion_tolerance_m)

    views_used = 0
    for view_index, (pose, intr, image, quality) in enumerate(views):
        if image is None:
            continue
        uvp, depth, in_view = project_points(points, pose, intr)
        vis = visible_mask(
            uvp,
            depth,
            in_view,
            width=intr.width,
            height=intr.height,
            tolerance_m=occlusion_tolerance_m,
        )
        if not np.any(vis):
            continue

        idx = np.flatnonzero(vis)
        px = np.clip(uvp[idx, 0].astype(np.int32), 0, intr.width - 1)
        py = np.clip(uvp[idx, 1].astype(np.int32), 0, intr.height - 1)

        # BGR -> RGB here: every consumer of `texture` (PNG on disk, glTF
        # image) expects RGB, and converting once at the source beats
        # remembering to convert at each writer.
        sample = image[py, px][:, ::-1].astype(np.float32)
        if exposure_compensation:
            sample = np.clip(sample * np.float32(gains[view_index]), 0.0, 255.0)

        view_dir = pose.R[:, 2].reshape(1, 3)
        cos_inc = np.abs(np.sum(texel_normals[idx] * view_dir, axis=1))
        w = np.clip(cos_inc, 0.02, 1.0).astype(np.float32)
        w *= np.float32(max(quality, 1e-3))
        w /= np.maximum(depth[idx], 1e-3).astype(np.float32) ** 2

        # Insert into the running top-k: replace the weakest slot whenever
        # this view beats it.
        weakest = top_w[idx].argmin(axis=1)
        rows = np.arange(len(idx))
        current = top_w[idx][rows, weakest]
        better = w > current
        if np.any(better):
            target_rows = idx[better]
            target_cols = weakest[better]
            top_w[target_rows, target_cols] = w[better]
            top_rgb[target_rows, target_cols] = sample[better]
        views_used += 1

    total_w = top_w.sum(axis=1, keepdims=True)
    blended = np.where(
        total_w > 0,
        (top_rgb * top_w[..., None]).sum(axis=1) / np.maximum(total_w, 1e-8),
        0.0,
    )

    atlas = np.zeros((texture_size * texture_size, 3), dtype=np.uint8)
    atlas[covered] = np.clip(blended, 0, 255).astype(np.uint8)
    atlas = atlas.reshape(texture_size, texture_size, 3)

    # Texels inside a chart that no view could see leave holes; inpaint
    # them so bilinear sampling near them does not bleed black.
    filled = np.zeros(texture_size * texture_size, dtype=np.uint8)
    filled[covered] = (total_w.reshape(-1) > 0).astype(np.uint8)
    holes = (texel_mask.reshape(-1).astype(np.uint8) & (1 - filled)).reshape(texture_size, texture_size)
    n_holes = int(holes.sum())
    if n_holes:
        atlas = cv2.inpaint(atlas, holes, _INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)

    stats = {
        "texture_size": texture_size,
        "views_used": views_used,
        "exposure_gain_min": round(float(gains.min()), 3),
        "exposure_gain_max": round(float(gains.max()), 3),
        "texels_covered": int(covered.size),
        "texels_total": texture_size * texture_size,
        "texel_coverage_pct": round(100.0 * covered.size / (texture_size * texture_size), 2),
        "texels_inpainted": n_holes,
        "vertices_before_unwrap": len(vertices),
        "vertices_after_unwrap": len(new_vertices),
    }

    return TextureBakeResult(
        vertices=new_vertices,
        faces=new_faces,
        uv=uv,
        texture=atlas,
        vertex_map=vertex_map,
        stats=stats,
    )


def _solve_gains_for_views(
    points: np.ndarray,
    views: list[tuple[Pose, CameraIntrinsics, np.ndarray, float]],
    occlusion_tolerance_m: float,
    *,
    max_samples: int = 20000,
) -> np.ndarray:
    """Per-view exposure gains, fitted on surface points seen by 2+ views.

    Samples a bounded random subset of texel positions rather than all of
    them: the gain is one scalar per view, so a few thousand co-observations
    determine it as precisely as a few million, and the full set would make
    this pass cost more than the bake it is correcting.
    """
    from drishti3d.ingest.photometric import solve_exposure_gains

    rng = np.random.default_rng(0)
    if len(points) > max_samples:
        sample_idx = rng.choice(len(points), max_samples, replace=False)
    else:
        sample_idx = np.arange(len(points))
    sample_points = points[sample_idx]

    # luma[v, p] == that view's luminance at sample point p, NaN if unseen.
    luma = np.full((len(views), len(sample_points)), np.nan, dtype=np.float32)

    for v, (pose, intr, image, _quality) in enumerate(views):
        if image is None:
            continue
        uvp, depth, in_view = project_points(sample_points, pose, intr)
        vis = visible_mask(
            uvp,
            depth,
            in_view,
            width=intr.width,
            height=intr.height,
            tolerance_m=occlusion_tolerance_m,
        )
        if not np.any(vis):
            continue
        idx = np.flatnonzero(vis)
        px = np.clip(uvp[idx, 0].astype(np.int32), 0, intr.width - 1)
        py = np.clip(uvp[idx, 1].astype(np.int32), 0, intr.height - 1)
        luma[v, idx] = image[py, px].astype(np.float32).mean(axis=1)

    observations: list[tuple[int, int, float, float]] = []
    seen_counts = np.sum(~np.isnan(luma), axis=0)
    shared = np.flatnonzero(seen_counts >= 2)
    for p in shared:
        viewers = np.flatnonzero(~np.isnan(luma[:, p]))
        # Chain consecutive viewers rather than every pair: a point seen by
        # k views contributes k-1 constraints instead of k(k-1)/2, which
        # keeps a handful of heavily-observed points from dominating the
        # fit purely by combinatorics.
        for a, b in pairwise(viewers):
            observations.append((int(a), int(b), float(luma[a, p]), float(luma[b, p])))

    if not observations:
        return np.ones(len(views), dtype=np.float64)

    return solve_exposure_gains(observations, len(views))


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------


def _rasterize_attribute(
    attribute: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric-interpolate a per-vertex attribute across the UV atlas.

    Returns ``(image, mask)`` where ``image`` is ``(size, size, C)`` and
    ``mask`` marks texels any triangle covered.

    Uses OpenCV's triangle fill per channel rather than a hand-written
    scanline loop: for the triangle counts a TSDF mesh produces, a Python
    scanline rasteriser is minutes, and this is seconds. The cost is that
    each triangle is filled with flat-shaded *vertex-averaged* values. At
    4096x4096 over a survey-scale mesh, a triangle covers a handful of
    texels, so the difference from true barycentric interpolation is below
    the resolution of the source imagery -- and the alternative (a real
    per-texel barycentric pass in numpy) allocates per-triangle bounding
    boxes for hundreds of thousands of triangles.
    """
    channels = attribute.shape[1]
    image = np.zeros((size, size, channels), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    # UV origin is bottom-left in glTF/OBJ convention; image rows go top
    # -down. Flipping V here means the atlas written to disk matches what
    # a glTF/OBJ viewer expects without the writers each flipping again.
    px = np.empty_like(uv)
    px[:, 0] = uv[:, 0] * (size - 1)
    px[:, 1] = (1.0 - uv[:, 1]) * (size - 1)
    tri_px = px[faces].astype(np.int32)

    tri_values = attribute[faces].mean(axis=1)

    for tri, value in zip(tri_px, tri_values, strict=True):
        cv2.fillConvexPoly(image, tri, [float(v) for v in value], lineType=cv2.LINE_8)
        cv2.fillConvexPoly(mask, tri, 1, lineType=cv2.LINE_8)

    return image, mask.astype(bool)


def _rasterize_positions(
    vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(size, size, 3)`` world position per texel, plus a coverage mask."""
    return _rasterize_attribute(vertices.astype(np.float32), faces, uv, size)


def _rasterize_normals(normals: np.ndarray, faces: np.ndarray, uv: np.ndarray, size: int) -> np.ndarray:
    image, _ = _rasterize_attribute(normals.astype(np.float32), faces, uv, size)
    norm = np.linalg.norm(image, axis=2, keepdims=True)
    return np.divide(image, norm, out=np.zeros_like(image), where=norm > 1e-8)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals.

    Area weighting (rather than normalising each face normal first) is what
    makes the result stable on a TSDF mesh, where a vertex is often shared
    between many slivers and a few large well-conditioned triangles: the
    slivers should not outvote the triangles that actually describe the
    surface.
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # magnitude == 2 * area

    normals = np.zeros_like(vertices, dtype=np.float64)
    for col in range(3):
        np.add.at(normals, faces[:, col], face_normals)

    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)
