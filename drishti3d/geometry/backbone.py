"""Swappable geometry backbone abstraction.

MapAnything is the chosen backbone (see ``geometry.mapanything`` for why:
metric-scale output plus intrinsics/pose conditioning), but it is a large
external dependency with its own license terms, and the hardware reality
here (a 6 GB laptop GPU) means we may need to fall back to a smaller/faster
model (VGGT, Pi3, ...) on VRAM grounds even if MapAnything stays the
default. Every concrete backbone implements the ``Backbone`` interface so
the rest of the pipeline (window planning, submap merging, the GUI) never
has to know which one is actually running.

``NullBackbone`` is the zero-weight, zero-GPU implementation: it synthesizes
a small geometrically-consistent scene (a ground plane, a raised box, and
cameras flying a straight line over both) instead of running any model.
This is not a toy -- it is what lets the whole pipeline and the desktop GUI
be built, wired up, and demoed with no model weights, no GPU, and no
network access, and it is the fallback the demo can lean on if a real
backbone install fails on-site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from drishti3d.types import CameraIntrinsics, Pose

# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


@dataclass
class BackboneResult:
    """Output of one ``Backbone.predict`` call over a batch of views.

    poses:
        One world-from-camera ``Pose`` per input view, same order as the
        input ``images``.
    points:
        Dense per-pixel world-frame points, ``(V, H, W, 3)``.
    depth:
        Per-pixel camera-frame z-depth, ``(V, H, W)``.
    confidence:
        Per-pixel raw backbone confidence, ``(V, H, W)``, float in
        ``[0, 1]`` (not the coarser tiered ``types.Confidence`` enum --
        that quantization happens later, once a consumer decides on
        thresholds).
    intrinsics:
        One ``CameraIntrinsics`` per view, matching whatever resolution the
        backbone actually ran at (may differ from the input images' if the
        backbone resized internally).
    is_metric:
        Whether ``points``/``depth`` are in real-world metres (True for
        MapAnything when conditioned with real intrinsics; backbones like
        plain VGGT/Pi3 produce output that is only accurate up to an
        unknown global scale, and should report False here).
    images:
        Fix 3: per-view RGB colour, ``(V, H, W, 3)`` uint8, pixel-aligned
        with ``points``/``depth``/``confidence`` -- i.e. exactly the
        resized-and-cropped image the backbone actually ran inference on,
        not the original input resolution. This is what lets a caller
        (``pipeline.stages.GeometryStage``) attach a real per-point colour
        to ``Submap.points.rgb`` instead of leaving it ``None``: before
        this field existed, ``BackboneResult`` had no colour output at
        all, so the raw point cloud (``point_cloud.ply``/``.las``) had no
        ``red``/``green``/``blue`` properties whatsoever even though the
        source video obviously had colour -- the mesh had colour (sampled
        separately, downstream, during TSDF integration) while the raw
        cloud silently didn't. ``None`` for a backbone that genuinely
        cannot report this (there currently isn't one, but the field is
        optional so a future backbone isn't forced to fabricate colour).
    metadata:
        Free-form backbone-specific extras (checkpoint id, timing, raw
        model outputs kept for debugging, ...).
    """

    poses: list[Pose]
    points: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    intrinsics: list[CameraIntrinsics]
    is_metric: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    images: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Backbone interface
# ---------------------------------------------------------------------------


class Backbone(ABC):
    """Interface every geometry backbone (MapAnything, VGGT, Pi3, Null...) implements."""

    name: str = "unnamed"

    #: Whether this backbone's output is in metres on its own. Consulted by
    #: the fusion stage to decide whether a metric-scale *check* (altitude
    #: anchoring) is even meaningful: for a scale-free backbone the answer
    #: is no, and its scale comes from the GPS track instead.
    metric: bool = True


    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backbone can actually be used right now (weights + deps present).

        Must never raise -- a missing optional dependency (torch,
        mapanything, ...) is reported here, not via an exception, so
        callers (including the GUI) can check-then-fall-back cleanly.
        """

    @abstractmethod
    def load(self, device: str, dtype: Any = None) -> None:
        """Load model weights onto ``device`` (e.g. "cuda", "mps", "cpu")."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model and any device memory it holds."""

    @abstractmethod
    def predict(
        self,
        images: list[np.ndarray],
        intrinsics: list[CameraIntrinsics] | None = None,
        poses: list[Pose] | None = None,
    ) -> BackboneResult:
        """Run the backbone over a batch of views.

        ``intrinsics``/``poses``, when given, are per-view conditioning
        priors (from EXIF/camera-DB and GPS/IMU respectively) -- backbones
        that support conditioning (MapAnything) should use them to improve
        accuracy and recover true metric scale; backbones that don't
        should simply ignore them rather than erroring.
        """


# ---------------------------------------------------------------------------
# NullBackbone: synthetic ground-plane + box + straight flight line
# ---------------------------------------------------------------------------

# Synthetic scene layout (all in metres, world ENU Z-up). Deliberately
# simple and deterministic: a flat ground plane at z=0, one raised
# axis-aligned box sitting on it, and cameras flown along +X at fixed
# altitude, all pointed straight down (nadir) -- the canonical "single-pass
# drone survey" shape this whole project targets.
_FLIGHT_ALTITUDE_M = 20.0
_FLIGHT_SPACING_M = 3.0
_BOX_MIN = np.array([4.0, -3.0, 0.0])
_BOX_MAX = np.array([9.0, 3.0, 5.0])
_DEFAULT_HFOV_DEG = 70.0

# Nadir camera orientation, expressed as the constant world-from-camera
# rotation used for every synthetic pose: camera +Z (forward, into the
# scene) points along world -Z (straight down); camera +X (image right)
# stays aligned with world +X (East); camera +Y (image down) therefore
# maps to world -Y to keep the frame right-handed. This is a rotation by
# pi about the world X axis, i.e. diag(1, -1, -1) (det = +1).
_NADIR_R = np.diag([1.0, -1.0, -1.0])

# Ray/plane and ray/box intersections below skip an intersection whose
# camera-frame z ("how far along the view ray") is smaller than this --
# treated as "behind/at the camera", not a real surface hit.
_MIN_VALID_DEPTH_M = 1e-6

# Confidence values assigned to synthetic hits, matching BackboneResult's
# "float in [0, 1]" contract. A box hit is the "sharpest" synthetic
# geometry (an actual object, not a flat backdrop) so it gets the highest
# confidence; a miss (ray parallel to the ground, or somehow pointed away
# from everything) gets the lowest.
_CONF_BOX_HIT = 0.95
_CONF_GROUND_HIT = 0.75
_CONF_MISS = 0.05


def _ray_box_intersect(
    origin: np.ndarray, directions: np.ndarray, box_min: np.ndarray, box_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized slab-method ray/AABB intersection.

    ``origin`` is a single (3,) point shared by every ray; ``directions``
    is ``(H, W, 3)`` (unnormalized -- the returned parametric ``t`` is in
    the same units as ``directions``, i.e. it *is* the camera-frame z-depth
    when ``directions`` was built as ``[x, y, 1] * z`` style camera rays,
    which is exactly how ``NullBackbone`` builds them).

    Returns ``(t_enter, hit)`` -- ``t_enter`` is the near intersection
    parameter (meaningless where ``hit`` is False).

    Axis-aligned rays (a zero component in ``directions``) are nudged by a
    tiny epsilon to avoid a literal division by zero; this is an
    approximation that is irrelevant in practice here since perspective
    projection makes an exactly-axis-aligned ray a measure-zero case.
    """
    eps = 1e-12
    d = np.where(np.abs(directions) < eps, eps, directions)

    t1 = (box_min - origin) / d
    t2 = (box_max - origin) / d
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)

    t_enter = np.max(t_near, axis=-1)
    t_exit = np.min(t_far, axis=-1)

    hit = (t_exit >= t_enter) & (t_exit >= _MIN_VALID_DEPTH_M)
    # If the camera origin is already inside the box, t_enter can be
    # negative; use the exit point's sign-safe alternative (t_exit) as the
    # depth in that edge case so we don't report a negative depth.
    t_enter = np.where(t_enter >= _MIN_VALID_DEPTH_M, t_enter, t_exit)
    hit = hit & (t_enter >= _MIN_VALID_DEPTH_M)
    return t_enter, hit


class NullBackbone(Backbone):
    """Zero-weight, zero-GPU synthetic backbone (ground plane + box + straight flight line).

    Always available, never needs ``load``/``unload`` to do anything real.
    Ignores conditioning ``poses`` and always synthesizes its own
    deterministic straight-line nadir flight (that determinism is the
    point: it gives the rest of the pipeline and the GUI something stable
    to build and demo against). Respects conditioning ``intrinsics`` if
    given (so downstream code that inspects the returned intrinsics sees
    what it asked for), otherwise synthesizes a plausible default per
    image.
    """

    name = "null"

    def __init__(self) -> None:
        self._loaded = False

    def is_available(self) -> bool:
        return True

    def load(self, device: str = "cpu", dtype: Any = None) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def predict(
        self,
        images: list[np.ndarray],
        intrinsics: list[CameraIntrinsics] | None = None,
        poses: list[Pose] | None = None,
    ) -> BackboneResult:
        del poses  # NullBackbone always synthesizes its own straight-line flight; see class docstring
        n = len(images)
        if n == 0:
            raise ValueError("NullBackbone.predict requires at least one image")
        if intrinsics is not None and len(intrinsics) != n:
            raise ValueError("intrinsics list must be the same length as images")

        out_poses: list[Pose] = []
        out_intrinsics: list[CameraIntrinsics] = []
        depths: list[np.ndarray] = []
        confs: list[np.ndarray] = []
        points_list: list[np.ndarray] = []
        images_rgb: list[np.ndarray] = []

        for i, img in enumerate(images):
            h, w = img.shape[0], img.shape[1]
            intr = intrinsics[i] if intrinsics is not None else CameraIntrinsics.from_hfov(_DEFAULT_HFOV_DEG, w, h)
            out_intrinsics.append(intr)

            cam_t = np.array([_FLIGHT_SPACING_M * i, 0.0, _FLIGHT_ALTITUDE_M])
            pose = Pose(R=_NADIR_R.copy(), t=cam_t)
            out_poses.append(pose)

            pts_world, depth, conf = _render_synthetic_view(pose, intr, h, w)
            points_list.append(pts_world)
            depths.append(depth)
            confs.append(conf)

            # Fix 3: NullBackbone doesn't resize/crop internally (unlike
            # MapAnythingBackbone), so the input image is already
            # pixel-aligned 1:1 with pts_world/depth/conf above -- just
            # convert BGR (drishti3d.ingest.video's convention for every
            # decoded frame) to RGB (BackboneResult.images' documented
            # contract) and pass it through.
            images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        points = np.stack(points_list, axis=0)
        depth_arr = np.stack(depths, axis=0)
        conf_arr = np.stack(confs, axis=0)
        images_arr = np.stack(images_rgb, axis=0)

        return BackboneResult(
            poses=out_poses,
            points=points,
            depth=depth_arr,
            confidence=conf_arr,
            intrinsics=out_intrinsics,
            is_metric=True,
            metadata={"backbone": self.name, "n_views": n},
            images=images_arr,
        )


def _render_synthetic_view(
    pose: Pose, intrinsics: CameraIntrinsics, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast the synthetic ground-plane + box scene for one camera. Returns (points_world, depth_z, confidence)."""
    K_inv = np.linalg.inv(intrinsics.K())

    us = np.arange(width, dtype=np.float64) + 0.5
    vs = np.arange(height, dtype=np.float64) + 0.5
    grid_u, grid_v = np.meshgrid(us, vs)  # each (H, W)
    ones = np.ones_like(grid_u)
    pix = np.stack([grid_u, grid_v, ones], axis=-1)  # (H, W, 3)

    # Camera-frame ray directions (OpenCV convention: x-right/y-down/z-forward).
    dir_cam = pix @ K_inv.T  # (H, W, 3)
    # World-frame ray directions: world = R @ camera (Pose.R is world-from-camera).
    dir_world = dir_cam @ pose.R.T  # (H, W, 3)

    origin = pose.t

    # Ground plane (world z = 0): origin.z + t * dir_world.z = 0.
    dz = dir_world[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = np.where(np.abs(dz) > 1e-9, -origin[2] / dz, np.inf)
    ground_hit = np.isfinite(t_ground) & (t_ground >= _MIN_VALID_DEPTH_M)
    t_ground = np.where(ground_hit, t_ground, np.inf)

    t_box, box_hit = _ray_box_intersect(origin, dir_world, _BOX_MIN, _BOX_MAX)
    t_box = np.where(box_hit, t_box, np.inf)

    # Nearer surface wins (the box occludes the ground behind it).
    use_box = box_hit & (t_box <= t_ground)
    depth = np.where(use_box, t_box, t_ground)
    valid = np.isfinite(depth)
    depth_safe = np.where(valid, depth, 0.0)

    points_world = origin.reshape(1, 1, 3) + depth_safe[..., None] * dir_world

    confidence = np.where(use_box, _CONF_BOX_HIT, np.where(valid, _CONF_GROUND_HIT, _CONF_MISS)).astype(np.float32)
    depth_out = depth_safe.astype(np.float64)

    return points_world.astype(np.float64), depth_out, confidence


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Backbone]] = {"null": NullBackbone}


def register_backbone(name: str, cls: type[Backbone]) -> None:
    """Register a ``Backbone`` implementation under ``name`` for ``get_backbone``."""
    _REGISTRY[name] = cls


def get_backbone(name: str, **kwargs: Any) -> Backbone:
    """Instantiate the registered backbone named ``name``.

    ``"mapanything"`` is lazily imported (and self-registered) here on
    first request rather than at module import time, so importing
    ``drishti3d.geometry.backbone`` never requires torch or the
    ``mapanything`` package to be installed -- only actually *requesting*
    the mapanything backbone does, and even then only the import, not the
    weights (``Backbone.is_available()``/``load()`` are what check for
    those).

    ``**kwargs`` (e.g. ``max_image_size=cfg.geometry.max_image_size``) are
    forwarded to the backbone class's constructor. Fix 2 bug this closes:
    ``pipeline.stages.GeometryStage`` previously called this with no
    arguments at all, so ``MapAnythingBackbone`` was *always* constructed
    with its own hardcoded ``max_image_size=518`` default regardless of
    ``GeometryConfig.max_image_size`` -- raising that config field alone
    would have silently done nothing, since the backbone's own internal
    ``resize_preserving_aspect`` call would immediately resize the
    already-correctly-sized input image straight back down to 518px. If
    the target class's constructor doesn't accept a given kwarg (e.g.
    ``NullBackbone``, which takes none), that kwarg is silently dropped
    rather than raising -- every registered backbone must stay
    constructible with zero arguments (``Backbone`` implementations are
    meant to be interchangeable; a caller that doesn't know or care which
    backbone is configured shouldn't have to know which kwargs each one's
    constructor happens to accept).
    """
    if name not in _REGISTRY and name == "mapanything":
        from drishti3d.geometry.mapanything import MapAnythingBackbone

        register_backbone("mapanything", MapAnythingBackbone)

    if name not in _REGISTRY:
        raise ValueError(f"Unknown backbone {name!r}; available: {sorted(_REGISTRY)}")

    cls = _REGISTRY[name]
    if not kwargs:
        return cls()
    try:
        return cls(**kwargs)
    except TypeError:
        return cls()
