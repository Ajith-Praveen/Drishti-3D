"""Shared data contract for the DRISHTI-3D pipeline.

This module is the single source of truth for the data structures passed
between pipeline stages (ingest -> triage -> geometry -> fusion -> export).
All other modules should import these types rather than redefining them.

Coordinate conventions
-----------------------
World frame:
    ENU (East-North-Up), units in metres, Z-up. X points East, Y points
    North, Z points Up. This is the frame all georeferenced/reconstructed
    geometry (poses, point clouds) is expressed in unless otherwise noted.

Camera frame:
    OpenCV convention. X points right, Y points down, Z points forward
    (out of the lens, into the scene). Pixel projection follows the
    standard pinhole model with intrinsics matrix K.

Geographic coordinates (GeoPoint):
    lat/lon in decimal degrees (WGS84), altitude in metres. These are
    distinct from the local ENU world frame; a georeferencing step is
    responsible for converting between the two (e.g. via pyproj).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Only needed for the `Submap.window` type hint below; guarded so
    # `drishti3d.types` never actually imports `drishti3d.geometry` at
    # runtime (that dependency should only ever point the other way).
    from drishti3d.geometry.windows import Window


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics.

    fx, fy: focal lengths in pixels.
    cx, cy: principal point in pixels.
    width, height: image dimensions in pixels.
    dist_coeffs: OpenCV-style distortion coefficients (k1, k2, p1, p2, k3, ...)
        or None if unknown / assumed undistorted.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray | None = None

    def K(self) -> np.ndarray:
        """Return the 3x3 camera intrinsics matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_hfov(
        cls, hfov_deg: float, width: int, height: int
    ) -> CameraIntrinsics:
        """Build intrinsics from a horizontal field-of-view (degrees).

        Assumes square pixels (fx == fy) and a centred principal point.
        """
        hfov_rad = np.deg2rad(hfov_deg)
        fx = width / (2.0 * np.tan(hfov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


@dataclass
class Pose:
    """Rigid-body pose of a camera in the world (ENU) frame.

    R: 3x3 rotation matrix, world-from-camera (rotates a vector expressed
        in the camera frame into the world frame).
    t: (3,) camera centre expressed in the world frame.
    """

    R: np.ndarray
    t: np.ndarray

    def matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous camera-to-world transform."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def inverse(self) -> Pose:
        """Return the inverse pose (world-to-camera, as a Pose).

        Swaps the roles of "world" and "camera": the returned Pose's
        rotation is R^T and its translation is -R^T @ t, such that
        ``pose.inverse().matrix() == np.linalg.inv(pose.matrix())``.
        """
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Pose(R=R_inv, t=t_inv)


@dataclass
class GeoPoint:
    """A single georeferenced point (WGS84 lat/lon, altitude in metres)."""

    lat: float
    lon: float
    alt_msl: float
    alt_rel: float | None = None
    accuracy_h: float | None = None
    accuracy_v: float | None = None


@dataclass
class TelemetrySample:
    """A single telemetry sample synchronized to the video timeline."""

    timestamp: float  # seconds from video start
    geo: GeoPoint | None = None
    gimbal_pitch: float | None = None
    gimbal_roll: float | None = None
    gimbal_yaw: float | None = None
    imu_accel: np.ndarray | None = None
    imu_gyro: np.ndarray | None = None
    baro_alt: float | None = None


@dataclass
class Frame:
    """A single decoded video frame."""

    index: int
    timestamp: float
    image: np.ndarray | None = None  # BGR uint8, HxWx3; may be None if lazy


@dataclass
class FrameMetrics:
    """Quality/geometry metrics computed for a frame during triage."""

    index: int
    timestamp: float
    blur_score: float
    exposure_score: float
    mean_luma: float
    estimated_parallax: float


@dataclass
class Keyframe:
    """A frame selected by triage as useful for reconstruction."""

    frame_index: int
    timestamp: float
    metrics: FrameMetrics
    telemetry: TelemetrySample | None = None
    intrinsics: CameraIntrinsics | None = None
    pose: Pose | None = None


class Confidence(IntEnum):
    """Confidence tier for a reconstructed quantity (e.g. a 3D point)."""

    INFERRED = 0
    LOW_CONFIDENCE = 1
    MEASURED = 2


@dataclass
class PointCloud:
    """A (possibly colored, possibly uncertain) 3D point cloud in world (ENU) frame."""

    xyz: np.ndarray  # (N, 3) float
    rgb: np.ndarray | None = None  # (N, 3) uint8
    covariance: np.ndarray | None = None  # (N, 3, 3) float
    confidence: np.ndarray | None = None  # (N,) uint8

    # Per-point semantic class -- ``semantics.classes.SemanticClass``
    # values, produced by ``SemanticsStage`` and carried through fusion
    # into LAS/PLY/GLB. ``None`` means no segmenter ran; a stored ``0``
    # (``UNLABELLED``) means one ran and declined to commit for that
    # point. Those are different facts and the report card reports them
    # differently, so do not collapse one into the other by defaulting
    # this to a zero array.
    semantic_class: np.ndarray | None = None  # (N,) uint8

    # Per-point share of total vote weight held by the winning class (see
    # ``semantics.labelling``). Kept alongside the label because a label
    # without its agreement ratio is exactly the kind of unqualified
    # number this project refuses to ship -- a 0.95 "building" and a 0.34
    # "building" are not the same claim.
    semantic_confidence: np.ndarray | None = None  # (N,) float32 in [0, 1]


@dataclass
class Submap:
    """One geometry-stage reconstruction window's output, ready for merging.

    Produced by running a backbone (see ``geometry.backbone.Backbone``) over
    one ``geometry.windows.Window`` of keyframes. Everything here is still
    expressed in that window's own *local* reconstruction frame -- an
    independent backbone inference call has no reason to land in the same
    scale/orientation/origin as any other window's -- ``geometry.submap
    .merge_submaps`` is what aligns a sequence of these into one shared
    global frame, using the cameras each pair of neighbouring submaps has
    in common.

    window:
        The ``geometry.windows.Window`` this submap was reconstructed from;
        carries the keyframe-list index range and which keyframe indices
        are shared with the previous window (the anchor points
        ``merge_submaps`` aligns on).
    poses:
        One ``Pose`` per keyframe this submap covers, in local-frame
        coordinates, in the same order as ``keyframe_indices``.
    points:
        The submap's local-frame dense point cloud.
    confidence:
        Per-point confidence, parallel to ``points.xyz``'s first dimension
        (raw continuous backbone confidence, e.g. ``BackboneResult
        .confidence`` flattened -- distinct from ``PointCloud.confidence``,
        which holds the coarser tiered ``Confidence`` enum once a value is
        chosen for export).
    keyframe_indices:
        The (global, into the original ``list[Keyframe]``) indices this
        submap covers, in the same order as ``poses``.
    local_origin:
        The pose that anchors this submap's local coordinate frame --
        conventionally its first covered keyframe's local pose -- so a
        human (or a debugger) can reason about where "local frame origin"
        sits without hunting through ``poses[0]``.
    """

    window: Window
    poses: list[Pose]
    points: PointCloud
    confidence: np.ndarray
    keyframe_indices: list[int]
    local_origin: Pose
    #: Per-point index into ``poses``/``keyframe_indices`` saying which
    #: view each point was reconstructed from. Lets a later stage correct
    #: a single view's depth about its own camera (``fusion.reanchor``)
    #: after the per-pixel grid structure has been flattened away.
    #: ``None`` for submaps built before this field existed.
    view_index: np.ndarray | None = None
