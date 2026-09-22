"""Observability field and corrective-flight synthesis -- the project's actual novelty.

``covariance.py`` propagates bundle-adjustment covariance to a per-point
anisotropic uncertainty ellipsoid. That is already more than most
photogrammetry tools report, but on its own it is diagnostic, not
actionable: it tells a human "this region is badly constrained in this
direction" and leaves them to work out what to do about it. This module
closes the loop: it **inverts** the uncertainty field into the flight line
that would most reduce it, so the system's output is not just a trust
layer but a recommendation the drone operator can fly.

Existing work computes per-point covariance for well-conditioned
multi-pass blocks (structure-from-motion covariance estimation is not new)
and, separately, does next-best-view / next-best-trajectory planning for
active exploration. What is new here is doing this specifically for the
**single-pass degenerate regime** -- where the failure mode is not "some
areas are less covered" but "the whole strip has a systematically
under-constrained axis" (see ``geometry.bundle`` and ``geometry.covariance``
module docstrings) -- and closing the loop between diagnosis
(covariance/anisotropy) and prescription (one corrective flight line) in
one pipeline stage, rather than treating them as unrelated capabilities.

The core geometric idea (read this before touching ``required_viewing_direction``)
--------------------------------------------------------------------------------------
A 3D point's uncertainty ellipsoid has a worst-constrained axis ``w``
(``covariance.covariance_to_ellipsoid``'s largest eigenvector). The
existing observations of that point have some mean viewing ray direction
``v`` (camera-to-point, averaged over the cameras that saw it). Adding one
more observation from viewing direction ``v_new`` constrains the
components of the point's position that project differently onto
``v_new`` than they did onto the rays already collected -- in the
classic two-ray stereo triangulation picture, the achievable reduction in
positional uncertainty along any given direction scales with the sine of
the angle between the *new* ray and the existing mean ray, and is
maximised (for a fixed existing geometry) when that angle is 90 degrees
**and** the new ray's deviation from the old one happens specifically
*within the plane containing the worst-constrained axis* -- deviating out
of that plane spends "new angle" on a direction that wasn't uncertain in
the first place and buys nothing for the axis we actually care about.

Formally: project ``w`` onto the plane orthogonal to ``v`` --
``w_perp = w - (w . v) v``, normalized. ``w_perp`` is, by construction,
perpendicular to ``v`` and lies in ``span(v, w)``, which is exactly "a
direction perpendicular to the current mean viewing ray, within the plane
containing the worst-constrained axis." This is ``required_viewing_direction``.

The degenerate special case -- and it is the *common* case for a
single-pass nadir survey, not a rare edge condition -- is ``w`` already
(anti)parallel to ``v`` (empirically the dominant failure mode: see
``tests/test_covariance.py``'s collinear-flight test, where the worst axis
comes out within a few degrees of vertical/boresight-aligned for nadir
cameras). Then ``w_perp`` collapses to (near) zero: *any* horizontal
viewing direction is equally "perpendicular to the boresight," so the
projection alone doesn't pick one. We resolve the tie by picking the
horizontal direction that is also **new relative to the existing flight
line** -- perpendicular to the dominant axis of the existing camera
centres (a classic cross-strip) -- since repeating the same line adds
zero new baseline diversity (see ``plan_corrective_flight``'s docstring
and its dedicated test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from drishti3d.geometry.covariance import anisotropy_ratio as _anisotropy_ratio
from drishti3d.geometry.covariance import covariance_to_ellipsoid
from drishti3d.types import CameraIntrinsics, Pose

# If the worst-constrained axis w and the mean viewing ray v are aligned
# to within this cosine (~31.8 degrees at 0.85), we treat w as
# (anti)parallel to v and use the cross-strip fallback rather than the raw
# w_perp = w - (w.v)v projection. This is deliberately generous, not a
# near-zero epsilon: near the true single-pass depth degeneracy, w and v
# are almost exactly aligned but not *exactly* (each point's own geometry
# perturbs the worst eigenvector by a few degrees), and that small
# leftover component is dominated by per-point noise, not a robust
# directional signal -- normalizing it up to a full direction vector would
# amplify a few degrees of noise into a specific (and often nearly
# along-track, i.e. useless-repeat-pass) recommendation. See the module
# docstring's worked example.
_PARALLEL_ALIGNMENT_COS_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Observability field
# ---------------------------------------------------------------------------


@dataclass
class ObservabilityField:
    """Per-point uncertainty/geometry summary over a reconstructed point cloud.

    All arrays are ``(N, ...)``, parallel to ``points``.

    worst_direction:
        Unit vector, world frame -- the largest-uncertainty eigenvector of
        each point's covariance (``covariance.covariance_to_ellipsoid``'s
        first column).
    mean_viewing_ray:
        Unit vector, world frame -- the mean (camera -> point) direction
        over that point's observing cameras. ``[0, 0, 0]`` for points with
        no observations (should not occur for a point that came out of
        triangulation, but guarded defensively).
    required_direction:
        Unit vector, world frame -- ``required_viewing_direction``'s
        output per point: the new viewing ray that would most reduce the
        worst-constrained axis. This is a *prediction* based on current
        geometry, not a guarantee (see ``plan_corrective_flight``).
    mean_triangulation_angle_deg / max_triangulation_angle_deg:
        Pairwise angle (degrees) between observing cameras' viewing rays
        to the point, averaged / maxed over all observing pairs -- the
        single best predictor of depth uncertainty available without
        actually inverting the normal equations (a small triangulation
        angle means near-parallel rays, i.e. poor depth constraint,
        regardless of how many cameras contributed).
    """

    points: np.ndarray
    covariances: np.ndarray
    semi_axes: np.ndarray
    axes_rotation: np.ndarray
    anisotropy: np.ndarray
    n_observing_cameras: np.ndarray
    mean_triangulation_angle_deg: np.ndarray
    max_triangulation_angle_deg: np.ndarray
    worst_direction: np.ndarray
    mean_viewing_ray: np.ndarray
    required_direction: np.ndarray


def worst_constrained_direction(cov: np.ndarray) -> np.ndarray:
    """The direction (unit vector, world frame) of greatest positional uncertainty.

    Thin wrapper around ``covariance.covariance_to_ellipsoid``'s largest
    eigenvector -- kept as its own named function per this module's public
    API, since "which direction is worst" is a first-class question here,
    not just an implementation detail of the ellipsoid.
    """
    _, rotation = covariance_to_ellipsoid(cov)
    if np.asarray(cov).ndim == 2:
        return rotation[:, 0]
    return rotation[:, :, 0]


def _pairwise_ray_angles_deg(rays: np.ndarray) -> np.ndarray:
    """All pairwise angles (degrees) between a set of unit vectors, ``(k,3) -> (k*(k-1)/2,)``."""
    k = rays.shape[0]
    if k < 2:
        return np.zeros(0)
    iu = np.triu_indices(k, k=1)
    cos_theta = np.clip(np.einsum("ij,ij->i", rays[iu[0]], rays[iu[1]]), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def observability_field(
    points: np.ndarray,
    poses: list[Pose],
    intrinsics: list[CameraIntrinsics],
    covariances: np.ndarray,
    obs_camera_idx: np.ndarray,
    obs_point_idx: np.ndarray,
) -> ObservabilityField:
    """Assemble the full per-point observability summary.

    ``obs_camera_idx``/``obs_point_idx`` use the same convention as
    ``bundle.BAProblem`` (parallel arrays: observation ``k`` is "point
    ``obs_point_idx[k]`` was seen by camera ``obs_camera_idx[k]``") -- pass
    the same arrays used to build the ``BAProblem`` this reconstruction
    came from. ``intrinsics`` is accepted for interface completeness /
    future per-camera FOV-aware weighting but is not currently used in the
    triangulation-angle computation, which depends only on camera-centre
    geometry.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    covariances = np.asarray(covariances, dtype=np.float64).reshape(-1, 3, 3)
    n_points = points.shape[0]
    if covariances.shape[0] != n_points:
        raise ValueError(f"points/covariances length mismatch: {n_points} vs {covariances.shape[0]}")

    centres = np.array([p.t for p in poses], dtype=np.float64)

    obs_camera_idx = np.asarray(obs_camera_idx, dtype=np.int64)
    obs_point_idx = np.asarray(obs_point_idx, dtype=np.int64)

    # Group observing camera indices per point.
    observers: list[list[int]] = [[] for _ in range(n_points)]
    for cam_i, pt_i in zip(obs_camera_idx, obs_point_idx, strict=True):
        observers[pt_i].append(int(cam_i))

    semi_axes, axes_rotation = covariance_to_ellipsoid(covariances)
    anisotropy = _anisotropy_ratio(covariances)
    worst_dir = axes_rotation[:, :, 0]

    n_observing = np.zeros(n_points, dtype=np.int64)
    mean_tri_angle = np.zeros(n_points, dtype=np.float64)
    max_tri_angle = np.zeros(n_points, dtype=np.float64)
    mean_view_ray = np.zeros((n_points, 3), dtype=np.float64)
    required_dir = np.zeros((n_points, 3), dtype=np.float64)

    # Dominant axis of the *entire* camera trajectory (used as the
    # cross-strip fallback reference when a point's worst axis is
    # (anti)parallel to its own mean viewing ray -- see module docstring).
    flight_dir = _dominant_direction(centres)

    for i in range(n_points):
        cam_ids = observers[i]
        n_observing[i] = len(cam_ids)
        if not cam_ids:
            continue
        obs_centres = centres[cam_ids]
        rays = points[i][None, :] - obs_centres
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-9
        if not valid.any():
            continue
        unit_rays = rays[valid] / norms[valid]

        pairwise = _pairwise_ray_angles_deg(unit_rays)
        if pairwise.size:
            mean_tri_angle[i] = float(np.mean(pairwise))
            max_tri_angle[i] = float(np.max(pairwise))

        v_sum = unit_rays.sum(axis=0)
        v_norm = np.linalg.norm(v_sum)
        v_mean = v_sum / v_norm if v_norm > 1e-9 else np.array([0.0, 0.0, -1.0])
        mean_view_ray[i] = v_mean

        required_dir[i] = _resolve_direction(worst_dir[i], v_mean, flight_dir)

    return ObservabilityField(
        points=points,
        covariances=covariances,
        semi_axes=semi_axes,
        axes_rotation=axes_rotation,
        anisotropy=anisotropy,
        n_observing_cameras=n_observing,
        mean_triangulation_angle_deg=mean_tri_angle,
        max_triangulation_angle_deg=max_tri_angle,
        worst_direction=worst_dir,
        mean_viewing_ray=mean_view_ray,
        required_direction=required_dir,
    )


def _dominant_direction(centres: np.ndarray) -> np.ndarray:
    """Dominant (unit) axis of a set of 3D points, via SVD of the centred coordinates.

    Same idea ``geometry.submap.umeyama_alignment`` uses to *detect*
    collinearity; here we want the axis itself (the flight line's own
    direction), not a degeneracy flag.
    """
    if centres.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0])
    centred = centres - centres.mean(axis=0)
    if np.linalg.norm(centred) < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    _, _, Vt = np.linalg.svd(centred)
    return Vt[0] / np.linalg.norm(Vt[0])


def _resolve_direction(w: np.ndarray, v_mean: np.ndarray, flight_dir: np.ndarray) -> np.ndarray:
    """Core geometric derivation described in the module docstring, plus the degenerate fallback."""
    cos_align = abs(float(np.dot(w, v_mean)))
    if cos_align <= _PARALLEL_ALIGNMENT_COS_THRESHOLD:
        w_perp = w - np.dot(w, v_mean) * v_mean
        norm = np.linalg.norm(w_perp)
        if norm >= 1e-9:
            return w_perp / norm

    # w is (anti)parallel to v_mean (within _PARALLEL_ALIGNMENT_COS_THRESHOLD):
    # any direction orthogonal to v_mean is equally "perpendicular to the
    # current viewing ray" from the pure single-point geometry alone. Break
    # the tie with the direction that is also new relative to the existing
    # flight line (cross-strip).
    candidate = np.cross(v_mean, flight_dir)
    norm = np.linalg.norm(candidate)
    if norm < 1e-9:
        candidate = np.cross(v_mean, np.array([0.0, 0.0, 1.0]))
        norm = np.linalg.norm(candidate)
    if norm < 1e-9:
        candidate = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    return candidate / norm


def required_viewing_direction(point: np.ndarray, cov: np.ndarray, existing_poses: list[Pose]) -> np.ndarray:
    """The single new viewing ray (unit vector, world frame) that would most reduce ``cov``'s worst axis.

    See the module docstring for the full geometric derivation. Standalone
    convenience wrapper around the same per-point logic
    ``observability_field`` applies to every point at once (useful for
    ad-hoc "what should observe this one point better" queries without
    building a full field).
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    w = worst_constrained_direction(cov)

    centres = np.array([p.t for p in existing_poses], dtype=np.float64)
    if centres.shape[0] == 0:
        raise ValueError("required_viewing_direction needs >= 1 existing pose")

    rays = point[None, :] - centres
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-9
    if not valid.any():
        raise ValueError("all existing poses coincide with the query point")
    unit_rays = rays[valid] / norms[valid]
    v_sum = unit_rays.sum(axis=0)
    v_norm = np.linalg.norm(v_sum)
    v_mean = v_sum / v_norm if v_norm > 1e-9 else np.array([0.0, 0.0, -1.0])

    flight_dir = _dominant_direction(centres)
    return _resolve_direction(w, v_mean, flight_dir)


# ---------------------------------------------------------------------------
# Corrective flight-line planning
# ---------------------------------------------------------------------------


@dataclass
class PlanConfig:
    """Options for ``plan_corrective_flight``.

    uncertainty_threshold_m:
        Points with largest 1-sigma semi-axis above this are candidates
        for correction. ``None`` (default) uses the 75th percentile of
        the field's own semi-axes, so a plan can always be produced
        without the caller having to know an absolute scale up front.
    cluster_distance_m:
        Horizontal (XY) linkage distance for clustering candidate points
        (via ``scipy.cluster.hierarchy.fclusterdata``); points closer than
        this are considered part of the same problem area.
    min_obliquity_deg:
        A corrective pass cannot literally fly at the mathematically
        "ideal" convergence angle implied by ``required_direction`` when
        that angle is near-horizontal (the camera would photograph the
        horizon, not the ground) -- see ``plan_corrective_flight``'s
        docstring. This clamps the recommended gimbal pitch to be at
        least this many degrees off horizontal (i.e. pitch in
        ``[-90, -min_obliquity_deg]``).
    altitude_agl_m:
        Recommended flight altitude above the point cluster. ``None``
        (default) reuses the existing trajectory's own mean AGL over the
        cluster, i.e. "fly the new line at the same altitude as before."
    """

    uncertainty_threshold_m: float | None = None
    cluster_distance_m: float = 30.0
    min_obliquity_deg: float = 30.0
    altitude_agl_m: float | None = None


@dataclass
class FlightLinePlan:
    """A single straight corrective flight line, plus the (predicted) payoff of flying it."""

    heading_deg: float
    altitude_agl_m: float
    gimbal_pitch_deg: float
    start_enu: np.ndarray
    end_enu: np.ndarray
    predicted_resolved_fraction: float
    cluster_point_indices: np.ndarray
    recommendation: str


def plan_corrective_flight(
    field: ObservabilityField, existing_trajectory: list[Pose], config: PlanConfig | None = None
) -> FlightLinePlan:
    """Synthesize the single straight flight line that best addresses the worst-constrained cluster.

    Why one straight line, and why its heading differs from the original pass
    ---------------------------------------------------------------------------
    A parallel repeat of the original pass observes every point from
    (approximately) the same viewing direction as before -- zero new
    triangulation angle, zero information about the worst-constrained
    axis, by definition of what "worst-constrained" means after a bundle
    adjustment that already used the first pass's geometry. The heading
    returned here comes from aggregating each candidate point's
    ``required_direction`` (see module docstring) -- for the common
    single-strip vertical/depth degeneracy this converges to the
    cross-strip heading (perpendicular to the original flight line), which
    is exactly the classical photogrammetric fix for weak vertical
    accuracy in a single nadir strip.

    Practicality clamp on gimbal pitch
    -------------------------------------
    ``required_direction`` is the mathematically ideal (maximally
    informative) new viewing ray, which for the common vertical-depth
    degeneracy comes out purely horizontal (a 90 degree convergence angle
    from a nadir mean ray) -- flyable in theory, useless in practice (the
    camera would image the horizon). We clamp the derived gimbal pitch to
    ``config.min_obliquity_deg`` off horizontal and correspondingly scale
    down the predicted resolved fraction (see ``_predicted_resolved_fraction``),
    rather than pretend the achievable pass is as good as the theoretical
    ideal.
    """
    config = config or PlanConfig()

    severity = field.semi_axes[:, 0]
    threshold = (
        config.uncertainty_threshold_m
        if config.uncertainty_threshold_m is not None
        else float(np.percentile(severity, 75))
    )
    candidate_mask = severity >= threshold
    candidate_idx = np.where(candidate_mask)[0]
    if candidate_idx.size == 0:
        raise ValueError("no points exceed the uncertainty threshold -- nothing to plan a corrective pass for")

    candidate_points = field.points[candidate_idx]

    if candidate_idx.size >= 2:
        xy = candidate_points[:, :2]
        if np.linalg.norm(xy - xy.mean(axis=0)) < 1e-9:
            cluster_labels = np.ones(candidate_idx.size, dtype=np.int64)
        else:
            cluster_labels = fclusterdata(xy, t=config.cluster_distance_m, criterion="distance")
    else:
        cluster_labels = np.array([1])

    labels, counts = np.unique(cluster_labels, return_counts=True)
    best_label = labels[np.argmax(counts)]
    cluster_mask = cluster_labels == best_label
    cluster_global_idx = candidate_idx[cluster_mask]
    cluster_points = field.points[cluster_global_idx]
    cluster_severity = severity[cluster_global_idx]

    weights = cluster_severity / cluster_severity.sum()
    agg_dir = (field.required_direction[cluster_global_idx] * weights[:, None]).sum(axis=0)
    agg_dir_norm = np.linalg.norm(agg_dir)
    agg_dir = agg_dir / agg_dir_norm if agg_dir_norm > 1e-9 else np.array([0.0, -1.0, 0.0])

    agg_mean_ray = (field.mean_viewing_ray[cluster_global_idx] * weights[:, None]).sum(axis=0)
    ray_norm = np.linalg.norm(agg_mean_ray)
    agg_mean_ray = agg_mean_ray / ray_norm if ray_norm > 1e-9 else np.array([0.0, 0.0, -1.0])

    # Ideal (unclamped) pitch from the aggregated required direction.
    ideal_pitch_deg = float(np.degrees(np.arcsin(np.clip(agg_dir[2], -1.0, 1.0))))
    gimbal_pitch_deg = float(np.clip(ideal_pitch_deg, -90.0, -config.min_obliquity_deg))

    horizontal = agg_dir[:2]
    horizontal_norm = np.linalg.norm(horizontal)
    if horizontal_norm < 1e-6:
        # Purely-vertical aggregate direction (shouldn't normally happen --
        # observability_field already resolves this via the cross-strip
        # fallback -- but guarded here too): fall back to perpendicular to
        # the existing trajectory's own dominant axis.
        centres = np.array([p.t for p in existing_trajectory], dtype=np.float64)
        flight_dir = _dominant_direction(centres)[:2]
        if np.linalg.norm(flight_dir) < 1e-9:
            flight_dir = np.array([1.0, 0.0])
        horizontal = np.array([-flight_dir[1], flight_dir[0]])
        horizontal_norm = np.linalg.norm(horizontal)

    east, north = horizontal[0] / horizontal_norm, horizontal[1] / horizontal_norm
    heading_deg = float(np.degrees(np.arctan2(east, north)) % 360.0)

    centres = np.array([p.t for p in existing_trajectory], dtype=np.float64)
    cluster_ground_z = float(np.mean(cluster_points[:, 2]))
    if config.altitude_agl_m is not None:
        altitude_agl_m = config.altitude_agl_m
    else:
        altitude_agl_m = float(np.mean(centres[:, 2])) - cluster_ground_z

    cluster_centroid = cluster_points.mean(axis=0)
    cluster_extent = float(np.linalg.norm(cluster_points.max(axis=0)[:2] - cluster_points.min(axis=0)[:2]))
    half_len = max(cluster_extent, 20.0) / 2.0 + 10.0  # pad past the cluster on both ends
    heading_vec = np.array([east, north, 0.0])
    line_centre = np.array([cluster_centroid[0], cluster_centroid[1], cluster_ground_z + altitude_agl_m])
    start_enu = line_centre - half_len * heading_vec
    end_enu = line_centre + half_len * heading_vec

    # Angle actually achieved between the new (clamped, flyable) boresight
    # and the existing mean viewing ray -- both are "camera -> point"
    # direction conventions, so a direct dot product is the convergence
    # angle between old and new observations (see _predicted_resolved_fraction).
    new_boresight = _final_boresight(heading_deg, gimbal_pitch_deg)
    achieved_angle_rad = float(np.arccos(np.clip(np.dot(agg_mean_ray, new_boresight), -1.0, 1.0)))
    resolved_fraction = _predicted_resolved_fraction(achieved_angle_rad)

    recommendation = (
        f"One more pass, heading {heading_deg:03.0f}°, {altitude_agl_m:.0f} m AGL, "
        f"gimbal {gimbal_pitch_deg:.0f}° — predicted to resolve "
        f"{resolved_fraction * 100:.0f}% of remaining uncertainty."
    )

    return FlightLinePlan(
        heading_deg=heading_deg,
        altitude_agl_m=altitude_agl_m,
        gimbal_pitch_deg=gimbal_pitch_deg,
        start_enu=start_enu,
        end_enu=end_enu,
        predicted_resolved_fraction=resolved_fraction,
        cluster_point_indices=cluster_global_idx,
        recommendation=recommendation,
    )


def _final_boresight(heading_deg: float, gimbal_pitch_deg: float) -> np.ndarray:
    """World-frame camera boresight (forward) direction for a given heading/pitch.

    Uses the same (yaw, pitch) convention as ``geometry.bundle.gimbal_to_R``
    (roll assumed zero for a planned line) -- see that module's docstring
    for the derivation. Here we only need the resulting forward vector,
    not the full rotation matrix.
    """
    yaw = np.radians(heading_deg)
    pitch = np.radians(gimbal_pitch_deg)
    # Forward-world(pitch) = [0, cos(pitch), sin(pitch)] in the yaw=0 frame
    # (derived in geometry.bundle's module docstring / gimbal_to_R); apply
    # yaw as a rotation about world Z.
    fwd_yaw0 = np.array([0.0, np.cos(pitch), np.sin(pitch)])
    c, s = np.cos(yaw), np.sin(yaw)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ fwd_yaw0


def _predicted_resolved_fraction(achieved_angle_rad: float) -> float:
    """Heuristic: resolved fraction scales with sin^2(convergence angle), capped short of 1.0.

    Stereo triangulation's achievable uncertainty reduction along a given
    axis scales with the sine of the angle between the new and existing
    viewing rays (near-parallel rays add ~no depth information; a 90
    degree convergence angle is maximally informative). Squaring is a
    standard information-theoretic heuristic (variance, not standard
    deviation, is what combines additively for independent
    measurements), and the 0.9 cap is a deliberate, explicit acknowledgement
    that this is a *prediction* from geometry alone -- it does not model
    reprojection noise, scene texture, or anything else that could make a
    real flight underperform the geometric ideal.
    """
    fraction = np.sin(achieved_angle_rad) ** 2
    return float(np.clip(fraction, 0.0, 0.9))


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyHeatmap:
    """A 2D raster of max positional uncertainty, ready for display or GeoTIFF export.

    raster:
        ``(rows, cols)`` float array, largest 1-sigma semi-axis (metres)
        of any point falling in that cell; ``NaN`` where no point fell.
    origin_xy:
        World (ENU) ``(x, y)`` of the raster's ``[0, 0]`` cell's
        lower-left corner (row 0 is the southernmost row).
    resolution_m:
        Cell size, matching the caller's requested resolution.
    """

    raster: np.ndarray
    origin_xy: np.ndarray
    resolution_m: float


def uncertainty_heatmap(field: ObservabilityField, resolution_m: float) -> UncertaintyHeatmap:
    """Rasterize per-point worst-case uncertainty onto an XY grid."""
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be > 0, got {resolution_m}")

    xy = field.points[:, :2]
    severity = field.semi_axes[:, 0]

    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    cols = max(1, int(np.ceil((max_xy[0] - min_xy[0]) / resolution_m)) + 1)
    rows = max(1, int(np.ceil((max_xy[1] - min_xy[1]) / resolution_m)) + 1)

    raster = np.full((rows, cols), np.nan, dtype=np.float64)
    col_idx = np.clip(((xy[:, 0] - min_xy[0]) / resolution_m).astype(np.int64), 0, cols - 1)
    row_idx = np.clip(((xy[:, 1] - min_xy[1]) / resolution_m).astype(np.int64), 0, rows - 1)

    for r, c, s in zip(row_idx, col_idx, severity, strict=True):
        raster[r, c] = s if np.isnan(raster[r, c]) else max(raster[r, c], s)

    return UncertaintyHeatmap(raster=raster, origin_xy=min_xy, resolution_m=resolution_m)
