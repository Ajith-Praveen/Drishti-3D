"""Multi-view triangulation of feature tracks, and assembly into a ``bundle.BAProblem``.

This is the last link between ``tracks.TrackSet`` (2D correspondences) and
``bundle.bundle_adjust`` (which needs 3D point *seeds* plus the same 2D
observations, see ``bundle.BAProblem``): triangulate an initial 3D
position per track from its observing cameras' already-known poses, reject
the tracks whose geometry cannot support a trustworthy triangulation, and
pack what's left into exactly the field layout ``BAProblem`` expects.

Why triangulation angle is the rejection criterion (read this before touching ``min_angle_deg``)
--------------------------------------------------------------------------------------------------
Two observing rays that are nearly parallel (a small angle between them at
the triangulated point) barely constrain depth along their shared viewing
direction: a large change in the point's position along that direction
produces a tiny change in either ray, so the DLT solve below is
numerically trying to intersect two lines that are close to not
intersecting at all, and small pixel noise in either observation gets
amplified into large uncertainty in the recovered depth. This is *exactly*
the same geometric quantity ``geometry.observability`` uses (see its
module docstring's "mean viewing ray" discussion) and that
``geometry.bundle``'s module docstring calls out as the single-pass
strip's structural weak point: a flight strip with modest across-track
baseline produces small triangulation angles for anything not close to the
camera, so this failure mode is not a rare edge case here -- it is close
to the *typical* case for far-away or grazing-angle points, and rejecting
them here (rather than letting bundle adjustment "resolve" a
near-degenerate point with wildly overconfident depth) is what keeps a bad
triangulation from ever reaching ``BAProblem.points``.

Why the default threshold is 1.5 degrees
--------------------------------------------
This is deliberately low -- a "clearly degenerate" floor, not a "good
triangulation" bar. A single nadir flight strip's ground points routinely
triangulate at angles of a few degrees even in the *healthy* case (limited
across-track baseline is the whole reason ``geometry.observability``
exists to recommend a corrective cross-strip flight), so a stricter
default would reject the majority of otherwise-usable points from a
legitimate single-pass survey. 1.5 degrees is closer to "the rays are
*almost* collinear" than "the geometry is comfortable" -- it exists to
catch points whose depth is effectively unconstrained (headed toward
infinity), not to grade triangulation quality generally; ``covariance.py``
/ ``observability.py`` are where quality (as opposed to outright validity)
gets quantified.
"""

from __future__ import annotations

import numpy as np

from drishti3d.geometry.bundle import BAProblem
from drishti3d.geometry.tracks import TrackSet
from drishti3d.types import CameraIntrinsics, Pose

_DEFAULT_MIN_ANGLE_DEG = 1.5


def _projection_matrix(pose: Pose, intrinsics: CameraIntrinsics) -> np.ndarray:
    """World-to-pixel 3x4 projection matrix ``K [R^T | -R^T t]`` for a world-from-camera ``Pose``."""
    r_t = pose.R.T
    t_cam = -r_t @ pose.t
    return intrinsics.K() @ np.hstack([r_t, t_cam.reshape(3, 1)])


def _max_triangulation_angle_deg(point: np.ndarray, camera_centers: np.ndarray) -> float:
    """Largest angle (degrees), at ``point``, between any two observing cameras' viewing rays.

    Viewing ray convention matches ``geometry.observability``'s "camera to
    point" direction: ``ray_i = normalize(point - camera_centers[i])``.
    """
    rays = point.reshape(1, 3) - camera_centers
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    rays = rays / norms

    n = rays.shape[0]
    max_angle = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cos_angle = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_angle)))
            max_angle = max(max_angle, angle)
    return max_angle


def triangulate_track(
    observations: list[tuple[int, np.ndarray]],
    poses: list[Pose],
    intrinsics: list[CameraIntrinsics],
) -> tuple[np.ndarray, float]:
    """Linear (DLT) multi-view triangulation of one track.

    Parameters
    ----------
    observations:
        ``[(camera_idx, uv), ...]``, ``camera_idx`` indexing into ``poses``/
        ``intrinsics`` (the same convention ``bundle.BAProblem`` uses for
        ``obs_camera_idx``), ``uv`` a ``(2,)`` pixel coordinate.
    poses, intrinsics:
        Full per-camera lists; only the entries ``observations`` actually
        references are used.

    Returns
    -------
    ``(point_xyz, max_angle_deg)``. ``point_xyz`` is the standard linear
    least-squares (SVD-of-the-DLT-system) triangulation -- not
    reprojection-error-optimal, but a good-enough seed for bundle
    adjustment to refine, which is all this needs to be (see the module
    docstring: bundle adjustment, not this function, is what actually
    minimizes reprojection error). ``max_angle_deg`` is the maximum
    triangulation angle across all pairs of observing cameras (see the
    module docstring) -- always returned, even when small, so callers can
    apply their own threshold (``triangulate_tracks`` does this) or simply
    record it for diagnostics.

    A near-degenerate system (``observations`` all essentially collinear
    with the point, i.e. exactly the failure mode this angle is meant to
    catch) can produce a point at effectively infinite depth (the
    homogeneous DLT solution's 4th coordinate collapsing toward zero); that
    case is detected explicitly and reported as a zero angle (guaranteed
    below any real ``min_angle_deg`` threshold) rather than dividing by a
    near-zero number into a numerically meaningless finite point.
    """
    if len(observations) < 2:
        raise ValueError("triangulate_track needs >= 2 observations")

    rows = []
    centers = []
    for camera_idx, uv in observations:
        p_mat = _projection_matrix(poses[camera_idx], intrinsics[camera_idx])
        u, v = float(uv[0]), float(uv[1])
        rows.append(u * p_mat[2, :] - p_mat[0, :])
        rows.append(v * p_mat[2, :] - p_mat[1, :])
        centers.append(poses[camera_idx].t)

    a_mat = np.stack(rows, axis=0)
    _, _, vt = np.linalg.svd(a_mat)
    point_h = vt[-1]

    if abs(point_h[3]) < 1e-9:
        return np.full(3, np.nan), 0.0

    point = point_h[:3] / point_h[3]
    angle = _max_triangulation_angle_deg(point, np.array(centers))
    return point, angle


def triangulate_tracks(
    trackset: TrackSet,
    poses: list[Pose],
    intrinsics: list[CameraIntrinsics],
    min_angle_deg: float = _DEFAULT_MIN_ANGLE_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate every track, rejecting degenerate (small-angle) geometry.

    Returns ``(points3d, valid_mask, angles_deg)``, all length
    ``len(trackset)`` and index-aligned with ``trackset.tracks`` -- a
    rejected track (``valid_mask[i] is False``) still gets a (possibly
    ``nan``-containing) entry in ``points3d`` so callers can index
    consistently; use ``valid_mask`` to select the usable subset (see
    ``build_ba_problem``, which expects to be handed only the valid ones).
    """
    n = len(trackset.tracks)
    points = np.zeros((n, 3), dtype=np.float64)
    angles = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for i, track in enumerate(trackset.tracks):
        observations = [(cam_idx, uv) for cam_idx, _kp_idx, uv in track.observations]
        point, angle = triangulate_track(observations, poses, intrinsics)
        points[i] = point
        angles[i] = angle
        valid[i] = bool(np.all(np.isfinite(point))) and angle >= min_angle_deg

    return points, valid, angles


def filter_by_reprojection(
    points: np.ndarray,
    trackset: TrackSet,
    poses: list[Pose],
    intrinsics: list[CameraIntrinsics],
    max_px: float,
    max_points: int | None = None,
) -> tuple[np.ndarray, TrackSet]:
    """Drop tracks whose triangulated point reprojects poorly into any of its observing cameras.

    ``points`` must be index-aligned with ``trackset.tracks`` (e.g.
    ``triangulate_tracks``'s output, already restricted to its
    ``valid_mask``). Populates ``Track.reprojection_error_px`` (the maximum
    per-observation error, in pixels) on every track *before* filtering --
    including the ones this then drops -- so a caller that wants the raw
    per-track errors for diagnostics can inspect ``trackset.tracks``
    in place beforehand.

    max_points:
        If given and more than ``max_points`` tracks survive the
        ``max_px`` cut, keep only the best ``max_points`` of them --
        ranked by (track length descending, reprojection error ascending)
        so the longest, best-constrained tracks win ties. This is the
        "cap the number of 3D points entering BA" lever (see
        ``config.MatchingConfig.max_points_in_ba``): bundle adjustment's
        cost scales with point count (each point is 3 more optimization
        parameters plus its observations' residual rows), but pose
        accuracy is set by how well-distributed and well-constrained the
        observed points are, not by their raw count -- a few thousand
        long, low-error tracks give the same pose accuracy as hundreds of
        thousands of marginal (barely-3-view, near-threshold-error) ones,
        at a fraction of the optimizer's per-iteration cost. ``None``
        (default) keeps every track that passes ``max_px`` -- unchanged
        behaviour for every existing caller.
    """
    n = len(trackset.tracks)
    if points.shape[0] != n:
        raise ValueError(f"filter_by_reprojection: points ({points.shape[0]}) and trackset ({n}) length mismatch")

    errors = np.empty(n, dtype=np.float64)
    for i, track in enumerate(trackset.tracks):
        point = points[i]
        worst = 0.0
        for camera_idx, _kp_idx, uv in track.observations:
            pose = poses[camera_idx]
            k = intrinsics[camera_idx]
            xc = pose.R.T @ (point - pose.t)
            if xc[2] <= 1e-6:
                worst = float("inf")
                continue
            u_pred = k.fx * xc[0] / xc[2] + k.cx
            v_pred = k.fy * xc[1] / xc[2] + k.cy
            err = float(np.hypot(u_pred - uv[0], v_pred - uv[1]))
            worst = max(worst, err)
        track.reprojection_error_px = worst
        errors[i] = worst

    keep_mask = errors <= max_px
    keep_idx = np.where(keep_mask)[0]

    if max_points is not None and keep_idx.shape[0] > max_points:
        lengths = np.array([len(trackset.tracks[i]) for i in keep_idx])
        # Lexsort ranks by the *last* key primarily: length descending
        # (negated), then reprojection error ascending as the tiebreaker.
        order = np.lexsort((errors[keep_idx], -lengths))
        keep_idx = keep_idx[order[:max_points]]
        keep_idx = np.sort(keep_idx)

    kept_points = points[keep_idx]
    kept_tracks = TrackSet(tracks=[trackset.tracks[i] for i in keep_idx])
    return kept_points, kept_tracks


def build_ba_problem(
    trackset: TrackSet,
    poses: list[Pose],
    intrinsics: list[CameraIntrinsics],
    points3d: np.ndarray,
) -> BAProblem:
    """Assemble a ``bundle.BAProblem`` from triangulated tracks.

    Field-for-field mapping onto ``bundle.BAProblem`` (see that class's
    docstring, which this follows exactly):

    - ``cameras`` / ``intrinsics``: passed through unchanged, one per
      camera, in the same order/indexing that ``Track.observations``'
      ``camera_idx`` already uses (both this module and ``tracks.py``
      consistently index cameras by position in the keyframe list, which
      is also what ``obs_camera_idx`` must index into).
    - ``points``: ``points3d`` unchanged -- row ``k`` of ``points3d`` is
      point index ``k`` in the problem, and ``trackset.tracks[k]`` is the
      track that produced it (index-aligned, same convention as
      ``triangulate_tracks``'s return value).
    - ``obs_camera_idx`` / ``obs_point_idx`` / ``obs_uv``: built by walking
      every observation of every track once; observation ``m`` in the
      flattened arrays is "track (point) ``obs_point_idx[m]`` was seen in
      camera ``obs_camera_idx[m]`` at pixel ``obs_uv[m]``", exactly
      ``BAProblem``'s documented contract.

    ``camera_priors`` / ``gcps`` / ``fixed_camera_indices`` are left at
    their ``BAProblem`` defaults (empty) -- attaching GPS/gravity priors
    from keyframe telemetry is a pipeline-level concern (see
    ``pipeline.stages.BundleAdjustmentStage``), not something this
    geometry-only assembly step has the telemetry to do itself.
    """
    n_points = len(trackset.tracks)
    if points3d.shape[0] != n_points:
        raise ValueError(f"build_ba_problem: points3d ({points3d.shape[0]}) and trackset ({n_points}) length mismatch")

    obs_camera_idx: list[int] = []
    obs_point_idx: list[int] = []
    obs_uv: list[np.ndarray] = []
    for point_idx, track in enumerate(trackset.tracks):
        for camera_idx, _kp_idx, uv in track.observations:
            obs_camera_idx.append(camera_idx)
            obs_point_idx.append(point_idx)
            obs_uv.append(uv)

    return BAProblem(
        cameras=list(poses),
        intrinsics=list(intrinsics),
        points=points3d.copy(),
        obs_camera_idx=np.array(obs_camera_idx, dtype=np.int64),
        obs_point_idx=np.array(obs_point_idx, dtype=np.int64),
        obs_uv=np.array(obs_uv, dtype=np.float64) if obs_uv else np.zeros((0, 2), dtype=np.float64),
    )
