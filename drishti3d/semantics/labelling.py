"""Lift per-frame 2D masks into per-point 3D labels by multi-view voting.

The problem this solves
-----------------------
A segmentation mask is a 2D opinion about one view. A point cloud needs a
3D answer. Naively taking the label of whichever frame happened to see a
point last gives a cloud that is visibly striped along flight lines --
every keyframe boundary becomes a label discontinuity, because two adjacent
frames disagree about the same roof.

So every point is projected into *every* keyframe that can see it, each
visible view casts a weighted vote, and the point takes the argmax. The
per-point ``vote_ratio`` (winning weight over total weight) is kept, not
thrown away: it is the honest measure of how much the model agreed with
itself, and it is what lets ``SemanticsConfig.min_vote_ratio`` demote a
contested point back to ``UNLABELLED`` instead of committing to a coin
flip. With a ground-level-trained checkpoint on nadir aerial frames (see
``semantics.segmenter``'s domain note), that demotion is doing real work.

Vote weighting
--------------
Each vote is weighted by three multiplied terms, all in ``[0, 1]``:

1. **Segmentation confidence** -- the mask's own max-softmax at that pixel.
   Relative signal only; these heads are overconfident in absolute terms.
2. **Viewing obliquity** -- ``cos`` of the angle between the camera ray and
   the local surface normal, when normals are available. A surface seen at
   a grazing angle occupies few pixels and is segmented badly; a surface
   seen face-on is segmented well. Without normals this term is 1.
3. **Range falloff** -- ``(z_ref / z)^2``, capped. A point 80 m from the
   camera is sampled by a quarter as many pixels as one at 40 m, so its
   label is correspondingly less trustworthy.

Occlusion is handled, not ignored
---------------------------------
A point on the ground behind a building still projects *inside* the
building's silhouette in the image, and would happily collect "building"
votes it has no right to. Each view therefore gets a coarse z-buffer:
points are binned to pixel blocks, the nearest depth per block wins, and
anything more than ``occlusion_tolerance_m`` behind that winner is excluded
from voting in that view. This is the same reasoning the texture baker uses
(see ``fusion.texture``) and the two share this module's
``visible_mask`` helper so they can never drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from drishti3d.semantics.classes import SemanticClass
from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)

__all__ = [
    "LabelVoteResult",
    "label_points",
    "project_points",
    "visible_mask",
]

#: Reference range for the falloff term, in metres. Roughly a typical
#: survey altitude; the term is a *ratio* so the absolute value only sets
#: where the weight equals 1, not the shape of the curve.
_REFERENCE_RANGE_M = 40.0

#: Range weight is clamped here so a very close view cannot dominate every
#: other view combined -- which would reintroduce the single-view striping
#: this whole module exists to remove.
_MAX_RANGE_WEIGHT = 4.0


@dataclass
class LabelVoteResult:
    """Per-point outcome of the vote, plus the diagnostics to judge it."""

    semantic_class: np.ndarray  # (N,) uint8, SemanticClass
    vote_ratio: np.ndarray  # (N,) float32 in [0, 1]
    view_count: np.ndarray  # (N,) uint16, how many views voted at all
    stats: dict = field(default_factory=dict)


def project_points(
    xyz: np.ndarray,
    pose: Pose,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame points into one camera.

    Returns ``(uv, depth, in_view)``:

    - ``uv``: ``(N, 2)`` float pixel coordinates (may be out of bounds).
    - ``depth``: ``(N,)`` metres along the camera's +Z (OpenCV convention:
      +Z points out of the lens into the scene).
    - ``in_view``: ``(N,)`` bool -- in front of the camera AND inside the
      image rectangle.

    ``Pose`` is world-from-camera with ``t`` the camera centre in world
    coordinates (see ``types.Pose``), so the world-to-camera map is
    ``X_cam = R.T @ (X_world - t)``.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    cam = (xyz - pose.t.reshape(1, 3)) @ pose.R  # == (R.T @ (X - t)).T

    depth = cam[:, 2]
    # Guard the division without branching: points at or behind the
    # principal plane get a dummy positive depth and are rejected by
    # `in_front` below anyway.
    safe_depth = np.where(depth > 1e-6, depth, 1.0)

    u = intrinsics.fx * (cam[:, 0] / safe_depth) + intrinsics.cx
    v = intrinsics.fy * (cam[:, 1] / safe_depth) + intrinsics.cy
    uv = np.stack([u, v], axis=1)

    in_front = depth > 1e-6
    inside = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
    return uv, depth, in_front & inside


def visible_mask(
    uv: np.ndarray,
    depth: np.ndarray,
    in_view: np.ndarray,
    *,
    width: int,
    height: int,
    block_px: int = 4,
    tolerance_m: float = 0.5,
) -> np.ndarray:
    """Coarse z-buffer occlusion test. Returns ``(N,)`` bool: safe to sample.

    Points are binned into ``block_px``-sized image blocks; the minimum
    depth in each block is taken as that block's surface, and any point
    more than ``tolerance_m`` behind it is occluded.

    ``block_px`` trades precision for robustness. At 1 px the test is exact
    but a sparse cloud leaves most blocks empty, so genuinely visible
    points in neighbouring blocks get no reference depth and pass
    trivially. At 4 px a typical survey cloud populates blocks densely
    enough for the minimum to mean something, while still resolving a
    building edge to within a few pixels.

    ``tolerance_m`` must exceed the cloud's own depth noise or the front
    surface will occlude *itself*. 0.5 m is comfortably above MapAnything's
    residual at survey altitude; raise it for noisier backbones.
    """
    n = len(depth)
    out = np.zeros(n, dtype=bool)
    if not np.any(in_view):
        return out

    idx = np.flatnonzero(in_view)
    bu = (uv[idx, 0] / block_px).astype(np.int32)
    bv = (uv[idx, 1] / block_px).astype(np.int32)
    bw = int(np.ceil(width / block_px))
    bh = int(np.ceil(height / block_px))
    np.clip(bu, 0, bw - 1, out=bu)
    np.clip(bv, 0, bh - 1, out=bv)
    flat = bv.astype(np.int64) * bw + bu.astype(np.int64)

    d = depth[idx]
    nearest = np.full(bw * bh, np.inf, dtype=np.float64)
    np.minimum.at(nearest, flat, d)

    out[idx] = d <= nearest[flat] + tolerance_m
    return out


def label_points(
    xyz: np.ndarray,
    views: list[tuple[Pose, CameraIntrinsics, np.ndarray, np.ndarray]],
    *,
    normals: np.ndarray | None = None,
    min_vote_ratio: float = 0.5,
    min_views: int = 1,
    occlusion_tolerance_m: float = 0.5,
    use_occlusion: bool = True,
) -> LabelVoteResult:
    """Vote per-point ``SemanticClass`` from a set of segmented views.

    ``views`` is a list of ``(pose, intrinsics, labels, confidence)``, where
    ``labels``/``confidence`` are the ``(H, W)`` arrays a
    ``SegmentationResult`` carries. Views whose pose or intrinsics are
    missing must be filtered out by the caller -- an unposed view cannot
    vote, and silently skipping it here would hide that from the stats.

    A point is left ``UNLABELLED`` when it was seen by fewer than
    ``min_views`` views, when every vote it received was for
    ``UNLABELLED``, or when the winner's share of total weight falls below
    ``min_vote_ratio``. All three are recorded separately in ``stats`` so
    the report card can distinguish "never seen" from "seen and disputed" --
    they mean very different things about a reconstruction.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    n_classes = len(SemanticClass)

    scores = np.zeros((n, n_classes), dtype=np.float32)
    view_count = np.zeros(n, dtype=np.uint16)

    n_views_used = 0
    for pose, intr, labels, conf in views:
        if labels.shape[:2] != (intr.height, intr.width):
            # A mask that does not match its own camera's intrinsics
            # cannot be indexed correctly; skipping is the only safe
            # action, and it is loud because it means an upstream resize
            # was not propagated into the intrinsics.
            logger.warning(
                "skipping view: mask %s does not match intrinsics %dx%d",
                labels.shape[:2],
                intr.height,
                intr.width,
            )
            continue

        uv, depth, in_view = project_points(xyz, pose, intr)
        if use_occlusion:
            sampleable = visible_mask(
                uv,
                depth,
                in_view,
                width=intr.width,
                height=intr.height,
                tolerance_m=occlusion_tolerance_m,
            )
        else:
            sampleable = in_view
        if not np.any(sampleable):
            continue

        idx = np.flatnonzero(sampleable)
        px = np.clip(uv[idx, 0].astype(np.int32), 0, intr.width - 1)
        py = np.clip(uv[idx, 1].astype(np.int32), 0, intr.height - 1)

        vote_class = labels[py, px].astype(np.int64)
        weight = conf[py, px].astype(np.float32)

        # Range falloff -- see module docstring.
        d = np.maximum(depth[idx], 1e-3)
        weight *= np.minimum((_REFERENCE_RANGE_M / d) ** 2, _MAX_RANGE_WEIGHT).astype(np.float32)

        # Obliquity. `pose.R[:, 2]` is the camera's +Z (viewing direction)
        # expressed in world coordinates; a surface whose normal opposes it
        # is being viewed face-on.
        if normals is not None:
            view_dir = pose.R[:, 2].reshape(1, 3)
            cos_inc = np.abs(np.sum(normals[idx] * view_dir, axis=1))
            weight *= np.clip(cos_inc, 0.05, 1.0).astype(np.float32)

        np.add.at(scores, (idx, vote_class), weight)
        view_count[idx] += 1
        n_views_used += 1

    total = scores.sum(axis=1)
    winner = scores.argmax(axis=1).astype(np.uint8)
    best = scores.max(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        vote_ratio = np.where(total > 0, best / total, 0.0).astype(np.float32)

    semantic_class = winner.copy()

    unseen = view_count < max(1, min_views)
    disputed = (~unseen) & (vote_ratio < min_vote_ratio)
    no_signal = (~unseen) & (total <= 0)
    semantic_class[unseen | disputed | no_signal] = SemanticClass.UNLABELLED

    stats = {
        "views_voted": n_views_used,
        "points": int(n),
        "unseen_points": int(unseen.sum()),
        "disputed_points": int(disputed.sum()),
        "labelled_points": int((semantic_class != SemanticClass.UNLABELLED).sum()),
        "mean_views_per_point": float(view_count.mean()) if n else 0.0,
        "mean_vote_ratio": float(vote_ratio[vote_ratio > 0].mean()) if np.any(vote_ratio > 0) else 0.0,
    }

    return LabelVoteResult(
        semantic_class=semantic_class,
        vote_ratio=vote_ratio,
        view_count=view_count,
        stats=stats,
    )
