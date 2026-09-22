"""Submap merging: align per-window geometry-stage outputs into one global model.

Each ``Submap`` (see ``drishti3d.types.Submap``) comes out of its own,
independent backbone inference call over one ``geometry.windows.Window`` --
there is no reason for two windows' local reconstruction frames to agree in
scale, orientation, or origin. ``merge_submaps`` stitches a whole sequence
of them into one shared global frame. *How* it does that is a choice of
**merge strategy**, selected by ``pipeline.stages.GeometryStage`` from
``geometry.flight_profile.analyze_flight_profile``'s recommendation and
passed straight through as the ``strategy`` argument below:

``"telemetry_rotation"``
    For a (near-)collinear track with trustworthy gimbal telemetry (the
    common single-nadir-pass survey case -- see "The single-flight-strip
    degeneracy" below for why this case needs special handling at all).
    Each submap's world *rotation* comes directly from telemetry (gimbal
    attitude), averaged across every one of that submap's own keyframes
    that has both a backbone-local pose and a telemetry-conditioned world
    pose (``_telemetry_rotation_transform`` / ``_average_rotation``) --
    deliberately NOT estimated from the submap's (near-collinear, rotation
    -unobservable) camera centres. With rotation fixed this way, only
    scale + translation are fit from camera-centre correspondences
    (``umeyama_fixed_rotation``), which a collinear configuration
    determines perfectly well.
``"gps_anchored"``
    For a track with enough heading variation (grid, orbit, curved) that
    camera centres genuinely constrain rotation. Each submap is aligned
    **directly and independently** onto its own GPS-tagged keyframes with
    a full Sim(3) fit (rotation included, via ``_umeyama_ransac``) --
    never chained onto a neighbouring submap's already-noisy output, so
    estimation noise from one submap's fit cannot compound into any
    other's (see "Chained alignment compounds error" below for why that
    matters).
``"chained_sim3"``
    The fallback: each submap ``i`` aligned onto submap ``i - 1``'s
    already-globalized positions via the shared cameras
    ``Window.shared_with_previous`` tracks. Used when neither of the above
    is usable (no/unreliable telemetry orientation AND no non-collinear
    GPS track) -- degraded, but the only option left. This is also what
    ``merge_submaps``/``alignment_residuals`` do by default when called
    without ``camera_gps_enu``/``conditioned_R`` at all (e.g. every test in
    this module that only exercises plain geometry).

Whichever strategy is selected, a submap that individually lacks what that
strategy needs (e.g. ``"telemetry_rotation"`` but this particular submap
has too few conditioned keyframes) falls back to chaining onto its
neighbour for *that submap only* -- logged loudly (never silently), since
a silent per-submap downgrade is exactly the kind of thing that turns into
an unexplained tilted model three stages later.

The single-flight-strip degeneracy (read this before trusting a result)
--------------------------------------------------------------------------
A single nadir/near-nadir drone pass is nearly collinear: the shared
camera centres between two consecutive windows lie close to one line (the
flight track segment they cover). Estimating a full Sim(3) (rotation +
translation + isotropic scale) from a *nearly collinear* point set is
numerically ill-conditioned specifically in its rotation component: any
small rotation about the axis running through the points (roughly, roll)
barely moves them, so the data does not actually constrain that rotation
degree of freedom, even though the least-squares solver will still hand
back *some* rotation matrix as if it had. The symptom is a merged model
whose camera centres line up with GPS just fine while the point cloud
itself comes out visibly tilted -- the part of the fit that's easy to
sanity-check by eye (are the camera positions right?) looks fine, and the
part that's easy to get wrong (is the whole thing rotated?) is silently
broken. ``umeyama_alignment`` computes the condition number of the shared
source points' spread and reports a ``degenerate`` flag whenever it's
above a threshold, so a caller (the accuracy report card, or a human) is
told "do not trust this junction's rotation" instead of silently shipping
a tilted model -- this is what ``"chained_sim3"`` and ``"gps_anchored"``
fall back on when their own per-junction/per-submap fit turns out
degenerate; ``"telemetry_rotation"`` sidesteps the whole problem by never
asking camera centres for rotation in the first place.

Chained alignment compounds error -- and the fix (``camera_gps_enu``)
--------------------------------------------------------------------------
The degeneracy discussion above is about *rotation being unobservable*
from 3 near-collinear points. There is a second, distinct failure mode in
the plain chained scheme (each submap ``i`` aligned onto submap ``i-1``'s
*already-globalized* positions, per ``Window.shared_with_previous``):
every junction typically has *exactly* ``_MIN_CORRESPONDENCES`` (3) shared
cameras (``geometry.windows.plan_windows``' overlap floor, as actually
configured by ``pipeline.stages.GeometryStage``), which is not enough
correspondences for ``_umeyama_ransac`` to reject anything as an outlier
(it needs ``>= _MIN_CORRESPONDENCES + 1`` before RANSAC can do anything
but degrade to a plain full-sample fit) -- so *every* junction's Sim(3)
is an unrobustified fit of the bare minimum sample, with real, irreducible
estimation noise (a few metres of translation, mirrored in rotation/scale)
even when the 3 points are not collinear at all (a healthy condition
number). Because each submap aligns onto the *previous submap's own
already-noisy* global positions rather than a fixed external reference,
that per-junction noise **compounds multiplicatively down the chain**:
measured directly on real footage (``scripts``/session diagnostics, not a
guess), predicted camera position vs. GPS ENU position error grows from
0.07 m at keyframe 0 to 23 m at keyframe 7 (8 keyframes, 2 junctions) and
to 806 m at keyframe 15 (16 keyframes, 6 junctions) -- an explosively
growing curve, not a bounded one. This is what ``"telemetry_rotation"``
and ``"gps_anchored"`` both fix at the root: every submap's transform
depends only on its own keyframes and the one shared, fixed, external GPS
reference frame -- never on another submap's (possibly already-noisy)
output -- so estimation noise from one submap's fit can no longer compound
into any other's, regardless of how many windows the chain has.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from drishti3d.types import PointCloud, Pose, Submap

logger = logging.getLogger(__name__)

# Umeyama needs >= 3 (non-collinear) correspondences to be well-posed at
# all; RANSAC below requires strictly more than that to have any inliers
# to reject.
_MIN_CORRESPONDENCES = 3

# RANSAC tuning for the Sim(3) fit between consecutive submaps' shared
# camera centres.
_RANSAC_ITERS = 200
_RANSAC_INLIER_THRESHOLD_M = 0.5

# Condition number (largest / smallest singular value of the shared source
# points' covariance) above which we call a junction's rotation estimate
# unreliable -- see module docstring. Collinear (or near-collinear) points
# drive this towards infinity (one near-zero singular value); this
# threshold is deliberately generous (i.e. triggers early) since a "mostly
# fine, mildly suspicious" alignment is exactly the dangerous case a
# report card should flag rather than silently pass.
_DEGENERATE_CONDITION_NUMBER = 1e4

# The three merge strategies ``merge_submaps``/``alignment_residuals``
# understand -- see the module docstring for what each does and when
# ``geometry.flight_profile.analyze_flight_profile`` recommends it.
_VALID_STRATEGIES = frozenset({"chained_sim3", "telemetry_rotation", "gps_anchored"})


@dataclass
class Sim3:
    """A similarity transform: ``dst = scale * (R @ src) + t``."""

    scale: float
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply this transform to an ``(N, 3)`` array of points."""
        return self.scale * (points @ self.R.T) + self.t

    def apply_pose(self, pose: Pose) -> Pose:
        """Apply this transform to a camera ``Pose`` (transforms both centre and orientation)."""
        new_t = self.apply(pose.t[None, :])[0]
        new_R = self.R @ pose.R
        return Pose(R=new_R, t=new_t)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[Sim3, bool, float]:
    """Estimate the Sim(3) mapping ``src`` points onto ``dst`` points (Umeyama, 1991).

    Parameters
    ----------
    src, dst:
        ``(N, 3)`` arrays of corresponding points, ``N >= 3``.

    Returns
    -------
    ``(transform, degenerate, condition_number)``. ``degenerate`` is True
    when ``src``'s point spread is nearly rank-deficient (near-collinear or
    near-coplanar-through-a-line) -- see this module's docstring for why
    that makes the *rotation* component of ``transform`` numerically
    unreliable even when translation/scale are fine.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"src/dst shape mismatch: {src.shape} vs {dst.shape}")
    n = src.shape[0]
    if n < _MIN_CORRESPONDENCES:
        raise ValueError(f"Umeyama alignment needs >= {_MIN_CORRESPONDENCES} correspondences, got {n}")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = float((src_c**2).sum() / n)

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    # Degeneracy check: how collinear is the *source* point set?
    #
    # Note this deliberately is NOT "is `src_cov` full rank" -- with
    # exactly 3 correspondences (the RANSAC minimal sample size, and a
    # perfectly legitimate, well-posed input for Umeyama/Kabsch), the
    # centered covariance is *always* rank <= 2 (3 points minus their
    # centroid span at most a 2D subspace), which would make a
    # smallest-vs-largest-singular-value ratio blow up for every minimal
    # sample regardless of whether it's actually degenerate. The real
    # question for a rigid/similarity fit is whether the points are
    # COLLINEAR (rank <= 1), not merely coplanar (rank <= 2, which is
    # perfectly well-posed) -- so we compare the largest and *second*
    # largest singular value instead: for a non-collinear set (rank >= 2)
    # this stays a modest, finite number; for a collinear (or
    # near-collinear) set the second singular value collapses toward zero
    # and the ratio blows up.
    src_cov = (src_c.T @ src_c) / n
    singular_values = np.linalg.svd(src_cov, compute_uv=False)
    if singular_values[1] > 1e-12:
        condition_number = float(singular_values[0] / singular_values[1])
    else:
        condition_number = float("inf")
    degenerate = condition_number > _DEGENERATE_CONDITION_NUMBER

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - scale * (R @ mu_src)

    return Sim3(scale=scale, R=R, t=t), degenerate, condition_number


def _umeyama_ransac(src: np.ndarray, dst: np.ndarray, seed: int) -> tuple[Sim3, bool, np.ndarray, float]:
    """RANSAC wrapper around ``umeyama_alignment`` over shared-camera correspondences.

    Returns ``(transform, degenerate, inlier_mask, condition_number)``,
    where ``transform``/``degenerate``/``condition_number`` come from the
    final refit over the best inlier consensus set found (or, if RANSAC
    can't find a usable consensus set at all -- too few points, or nothing
    agrees -- from a plain full-set fit, with the ``degenerate`` flag as
    the caller's warning that the result may not be trustworthy).
    """
    n = src.shape[0]
    if n < _MIN_CORRESPONDENCES + 1:
        # Not enough points to hold any out as a consensus check; use them all.
        transform, degenerate, condition_number = umeyama_alignment(src, dst)
        return transform, degenerate, np.ones(n, dtype=bool), condition_number

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None

    for _ in range(_RANSAC_ITERS):
        sample_idx = rng.choice(n, size=_MIN_CORRESPONDENCES, replace=False)
        try:
            candidate, _, _ = umeyama_alignment(src[sample_idx], dst[sample_idx])
        except ValueError:
            continue
        residuals = np.linalg.norm(candidate.apply(src) - dst, axis=1)
        inliers = residuals < _RANSAC_INLIER_THRESHOLD_M
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < _MIN_CORRESPONDENCES:
        transform, degenerate, condition_number = umeyama_alignment(src, dst)
        return transform, degenerate, np.ones(n, dtype=bool), condition_number

    transform, degenerate, condition_number = umeyama_alignment(src[best_inliers], dst[best_inliers])
    return transform, degenerate, best_inliers, condition_number


def umeyama_fixed_rotation(src: np.ndarray, dst: np.ndarray, R: np.ndarray, scale: float | None = None) -> Sim3:
    """Umeyama alignment with rotation *fixed* at ``R`` -- solve only scale + translation.

    ``merge_submaps``' ``"telemetry_rotation"`` strategy (see module
    docstring) needs exactly this: on a near-collinear flight strip, camera
    centres alone cannot constrain rotation at all (``umeyama_alignment``'s
    degeneracy check exists precisely to catch that), but they perfectly
    well constrain scale and translation once rotation is pinned down some
    other way -- here, from telemetry gimbal attitude
    (``_telemetry_rotation_transform``), not from the points themselves.

    With ``R`` fixed, ``dst ~= scale * (R @ src) + t`` is an ordinary
    isotropic least-squares regression in the rotated frame: substituting
    ``src_rot = src @ R.T`` reduces this to exactly the ``R = I`` case
    (fit ``dst ~= scale * src_rot + t``), so the closed-form solution is the
    same mean-centred scale/translation estimator ``umeyama_alignment``
    itself uses for those two components, just with ``src`` pre-rotated by
    the given ``R`` instead of estimating a rotation from the data.

    Parameters
    ----------
    src, dst:
        ``(N, 3)`` arrays of corresponding points, ``N >= 2`` (2
        non-coincident points fully determine the 1-DOF scale + 3-DOF
        translation this solves for; 3+ points make the fit
        least-squares rather than exact).
    R:
        The ``(3, 3)`` rotation to fix ``transform.R`` at -- not
        re-estimated from ``src``/``dst`` at all.
    scale:
        When given, scale is **not** estimated either -- only translation
        is fit, with scale pinned at this value.

        This exists because estimating scale from camera centres is unsound
        whenever the backbone already produces metric geometry. The scale
        term is determined almost entirely by the *horizontal* spread of the
        camera track, while the quantity it multiplies is dominated by the
        *vertical* camera-to-ground distance. On a 120 m AGL survey those
        differ by more than an order of magnitude, so a scale error well
        inside the camera track's own fit tolerance moves the reconstructed
        ground by tens of metres.

        Measured on a 77-keyframe single-pass flight: submaps fitted with
        free scale placed the ground at two distinct elevations 54 m apart,
        splitting the mesh into 68 disconnected components even though every
        junction RMSE was 0.1-0.35 m and the camera centres tracked GPS
        exactly. The cameras were right and the ground was wrong, which is
        the signature of a scale term absorbing error it should never have
        been free to absorb.

        Pass ``1.0`` when the backbone reports ``is_metric`` -- its output
        is already in metres, so there is nothing legitimate for a
        per-submap scale to correct.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"src/dst shape mismatch: {src.shape} vs {dst.shape}")
    if src.shape[0] < 2:
        raise ValueError(f"umeyama_fixed_rotation needs >= 2 correspondences, got {src.shape[0]}")

    src_rot = src @ np.asarray(R, dtype=np.float64).T
    mu_src_rot = src_rot.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    if scale is None:
        src_rot_c = src_rot - mu_src_rot
        dst_c = dst - mu_dst
        denom = float((src_rot_c**2).sum())
        scale = float((dst_c * src_rot_c).sum() / denom) if denom > 1e-12 else 1.0
    else:
        scale = float(scale)
    t = mu_dst - scale * mu_src_rot
    return Sim3(scale=scale, R=np.array(R, dtype=np.float64).copy(), t=t)


# Minimal sample for the rotation-fixed translation+scale fit above: 2
# non-coincident points fully determine a 1-DOF isotropic scale + 3-DOF
# translation (4 unknowns, 2 points x 3 coords = 6 equations); RANSAC below
# needs strictly more than that to hold any point out as a consensus check.
_MIN_CORRESPONDENCES_TS = 2


def _translation_scale_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    seed: int,
    R: np.ndarray | None = None,
    lock_scale: float | None = None,
) -> tuple[Sim3, np.ndarray]:
    """RANSAC wrapper around ``umeyama_fixed_rotation``, mirroring ``_umeyama_ransac``'s structure.

    ``R`` defaults to identity (the "submap already output in world-frame
    orientation" case); pass a specific rotation (e.g. from
    ``_average_rotation`` over telemetry-conditioned keyframes) to fix that
    instead. ``lock_scale`` pins scale rather than fitting it -- see
    ``umeyama_fixed_rotation``'s ``scale`` parameter for why a metric
    backbone must never have its scale re-fit from camera centres.
    """
    if R is None:
        R = np.eye(3)
    n = src.shape[0]
    if n < _MIN_CORRESPONDENCES_TS + 1:
        return umeyama_fixed_rotation(src, dst, R, scale=lock_scale), np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    for _ in range(_RANSAC_ITERS):
        sample_idx = rng.choice(n, size=_MIN_CORRESPONDENCES_TS, replace=False)
        candidate = umeyama_fixed_rotation(src[sample_idx], dst[sample_idx], R, scale=lock_scale)
        residuals = np.linalg.norm(candidate.apply(src) - dst, axis=1)
        inliers = residuals < _RANSAC_INLIER_THRESHOLD_M
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < _MIN_CORRESPONDENCES_TS:
        return umeyama_fixed_rotation(src, dst, R, scale=lock_scale), np.ones(n, dtype=bool)
    return umeyama_fixed_rotation(src[best_inliers], dst[best_inliers], R, scale=lock_scale), best_inliers


def _rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Angle (degrees) of the relative rotation ``r_a^T @ r_b`` -- how far apart two rotations are."""
    relative = r_a.T @ r_b
    cos_angle = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


# Minimum independent orientation samples (keyframes with both a
# backbone-local pose and a telemetry-conditioned world pose) needed before
# _average_rotation's estimate of a submap's local->world rotation offset
# is trusted at all -- 2 is already more independent constraint than 3
# collinear camera *centres* give (each orientation sample constrains all 3
# rotational DOF on its own; collinear centres constrain none), but we
# still want more than one in case a single keyframe's conditioning was
# noisy.
_MIN_ROTATION_SAMPLES = 2


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal-L2 (projected arithmetic) mean of a list of rotation matrices.

    ``sum(rotations)`` is, in general, not itself a rotation matrix; its
    nearest rotation (in Frobenius/chordal distance) is recovered the same
    way ``umeyama_alignment`` recovers a rotation from a cross-covariance:
    SVD, then re-orthogonalize with a reflection fix if the SVD handed back
    an improper (det < 0) rotation. This is the standard closed-form
    approximation to the Karcher mean on SO(3) -- exact when the input
    rotations are close together (the case here: they should all be
    estimates of the *same* submap-local -> world rotation offset, so any
    spread between them is estimation noise, not genuinely different
    rotations), and a reasonable one otherwise.
    """
    M = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        M += np.asarray(R, dtype=np.float64)
    U, _S, Vt = np.linalg.svd(M)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    return U @ S @ Vt


def _telemetry_rotation_transform(
    submap: Submap,
    camera_gps_enu: dict[int, np.ndarray],
    conditioned_R: dict[int, np.ndarray],
    seed: int,
    lock_scale: float | None = None,
) -> tuple[Sim3 | None, dict | None]:
    """Strategy A (``"telemetry_rotation"``): fix rotation from telemetry, fit scale+translation only.

    For every one of ``submap``'s own keyframes that has both a
    backbone-local pose (``submap.poses``) and a telemetry-conditioned
    world pose (``conditioned_R``, e.g. ``Keyframe.pose.R`` from
    ``pipeline.stages._poses_from_telemetry`` -- gimbal attitude via
    ``geometry.bundle.gimbal_to_R``), the local -> world rotation offset
    ``conditioned_R[kf] @ submap_local_R[kf]^T`` is computed independently;
    these should all agree (mod estimation noise) since a Sim(3) transform
    applies one *global* rotation to the whole submap, and are averaged
    (``_average_rotation``) into a single ``R_fixed``. Crucially, this uses
    however many keyframes the submap covers (with the Task 3 window-size
    increase, typically 12-16), each an *independent* 3-DOF orientation
    constraint -- not the 3 (near-collinear, rotation-degenerate) shared
    camera *centres* a junction has to work with.

    Scale + translation are then fit (``_translation_scale_ransac`` with
    ``R_fixed``) from whichever of the submap's own keyframes have a GPS
    position (``camera_gps_enu``) -- again, however many that submap
    covers, not just the 3-camera junction overlap.

    Returns ``(None, diag)`` when there aren't enough rotation samples
    (``>= _MIN_ROTATION_SAMPLES``) or GPS-tagged keyframes
    (``>= _MIN_CORRESPONDENCES_TS``) to do this for ``submap`` -- the
    caller falls back to chaining that submap onto its neighbour, logged
    loudly (see module docstring: never a silent downgrade).
    """
    local_pos = {kf: j for j, kf in enumerate(submap.keyframe_indices)}
    rot_kf = [kf for kf in submap.keyframe_indices if kf in conditioned_R]
    if len(rot_kf) < _MIN_ROTATION_SAMPLES:
        return None, {
            "method": "telemetry_rotation",
            "failure": "too few telemetry-conditioned keyframes for rotation",
            "n_rotation_samples": len(rot_kf),
        }

    r_samples = [conditioned_R[kf] @ submap.poses[local_pos[kf]].R.T for kf in rot_kf]
    r_fixed = _average_rotation(r_samples)
    rotation_spread_deg = float(np.mean([_rotation_angle_deg(r_fixed, r) for r in r_samples]))

    gps_kf = [kf for kf in submap.keyframe_indices if kf in camera_gps_enu]
    if len(gps_kf) < _MIN_CORRESPONDENCES_TS:
        return None, {
            "method": "telemetry_rotation",
            "failure": "too few own GPS-tagged keyframes for scale/translation",
            "n_gps": len(gps_kf),
            "n_rotation_samples": len(rot_kf),
            "rotation_spread_deg": rotation_spread_deg,
        }

    src = np.array([submap.poses[local_pos[kf]].t for kf in gps_kf])
    dst = np.array([camera_gps_enu[kf] for kf in gps_kf])
    transform, inliers = _translation_scale_ransac(src, dst, seed=seed, R=r_fixed, lock_scale=lock_scale)

    residuals = np.linalg.norm(transform.apply(src) - dst, axis=1)
    rmse_source = residuals[inliers] if inliers.any() else residuals
    diag = {
        "method": "telemetry_rotation",
        "n_gps": len(gps_kf),
        "n_inliers": int(inliers.sum()),
        "n_rotation_samples": len(rot_kf),
        "rotation_spread_deg": rotation_spread_deg,
        "degenerate": False,
        "condition_number": None,
        "rmse_m": float(np.sqrt(np.mean(rmse_source**2))),
    }
    return transform, diag


#: Fraction of a submap's points taken as "the ground beneath the camera"
#: when measuring reconstructed altitude. A low percentile, not the
#: minimum: one blunder below the surface would otherwise set the
#: reference. 10% tracks the ground while ignoring the deepest outliers.
_GROUND_PERCENTILE = 10.0

#: Radius (in local reconstruction units) around a camera's ground track
#: within which points count toward that camera's altitude measurement.
#: Applied in the submap's own frame BEFORE scale is known, so it is
#: deliberately generous -- it only has to isolate "under this camera" from
#: "the far end of the window".
_ALTITUDE_FOOTPRINT_FRACTION = 0.25

#: Scale corrections outside this range are refused. A metric backbone that
#: is off by more than 10x has not mis-scaled, it has failed, and silently
#: multiplying its output by 12 would manufacture a plausible-looking model
#: out of garbage. Reported as a failure instead.
_MAX_ALTITUDE_SCALE = 10.0
_MIN_ALTITUDE_SCALE = 0.1


def altitude_anchor_scale(
    submap: Submap,
    keyframe_altitude_m: dict[int, float],
) -> tuple[float | None, dict]:
    """Scale a submap so its reconstructed flying height matches telemetry.

    Why this exists, and why camera centres cannot do this job
    ----------------------------------------------------------
    ``_gps_full_anchor_transform``'s Umeyama fit estimates scale from how
    far apart the camera centres are. On a nadir survey that is a purely
    *horizontal* measurement, while the quantity scale actually governs --
    how far the ground is from the camera -- is purely *vertical*. The two
    are only linked through the reconstruction being internally consistent,
    and a backbone whose depth is systematically compressed is internally
    consistent while being entirely wrong.

    That is not hypothetical. Measured on a 77-keyframe flight held at a
    steady 119.6-120.3 m AGL by GPS, MapAnything -- which reports
    ``is_metric=True`` -- reconstructed the camera-to-ground distance as
    ~70 m over the first three windows (ratio 0.59) and ~18 m over the rest
    (ratio **0.15**), i.e. depth roughly six times too shallow, and
    inconsistent between windows. Camera centres still tracked GPS to well
    under a metre throughout, so every horizontal check passed: junction
    RMSE 0.1-0.35 m, reprojection 2.34 px. The resulting mesh put the
    ground at two different elevations 50 m apart and broke into 56
    disconnected components.

    Telemetry already carries the missing measurement. ``alt_rel`` is a
    direct observation of exactly the distance the backbone got wrong, from
    a sensor that is independent of the reconstruction. Using it is
    strictly better founded than inferring scale from camera spacing.

    What it does
    ------------
    For each keyframe with a known altitude, measures the reconstructed
    distance from that camera down to the ground beneath it (a low
    percentile of nearby points' height below the camera), and returns the
    median ratio ``telemetry_altitude / reconstructed_altitude``.

    Returns ``(None, diag)`` rather than guessing when the submap has too
    few altitude-tagged keyframes, when no ground is visible beneath them,
    or when the implied correction is beyond ``_MAX_ALTITUDE_SCALE`` -- a
    backbone off by more than 10x has failed rather than mis-scaled, and
    rescaling it would dress up garbage as a model.
    """
    xyz = np.asarray(submap.points.xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] < 50:
        return None, {"method": "altitude_anchor", "failure": "submap has too few points"}

    local_pos = {kf: j for j, kf in enumerate(submap.keyframe_indices)}
    usable = [kf for kf in submap.keyframe_indices if kf in keyframe_altitude_m]
    if not usable:
        return None, {"method": "altitude_anchor", "failure": "no keyframe has a telemetry altitude"}

    # Footprint radius in the submap's own (unknown-scale) units.
    extent = float(np.ptp(xyz[:, :2], axis=0).max())
    radius = max(extent * _ALTITUDE_FOOTPRINT_FRACTION, 1e-6)

    ratios: list[float] = []
    for kf in usable:
        cam = submap.poses[local_pos[kf]].t
        near = np.linalg.norm(xyz[:, :2] - cam[:2], axis=1) <= radius
        if int(near.sum()) < 20:
            continue
        ground = float(np.percentile(xyz[near, 2], _GROUND_PERCENTILE))
        reconstructed = float(cam[2] - ground)
        true_alt = float(keyframe_altitude_m[kf])
        if reconstructed <= 1e-6 or true_alt <= 1e-6:
            continue
        ratios.append(true_alt / reconstructed)

    if not ratios:
        return None, {
            "method": "altitude_anchor",
            "failure": "no keyframe had enough ground points beneath it to measure altitude",
            "n_altitude_keyframes": len(usable),
        }

    scale = float(np.median(ratios))
    diag = {
        "method": "altitude_anchor",
        "n_altitude_keyframes": len(usable),
        "n_measured": len(ratios),
        "scale": scale,
        "scale_spread": float(np.ptp(ratios)) if len(ratios) > 1 else 0.0,
    }
    if not (_MIN_ALTITUDE_SCALE <= scale <= _MAX_ALTITUDE_SCALE):
        diag["failure"] = (
            f"implied scale correction {scale:.3g} is outside "
            f"[{_MIN_ALTITUDE_SCALE}, {_MAX_ALTITUDE_SCALE}]; treating the backbone as failed "
            f"rather than rescaling it into plausibility"
        )
        return None, diag
    return scale, diag


def _gps_full_anchor_transform(
    submap: Submap,
    camera_gps_enu: dict[int, np.ndarray],
    seed: int,
    conditioned_R: dict[int, np.ndarray] | None = None,
    lock_scale: float | None = None,
) -> tuple[Sim3 | None, dict | None]:
    """Strategy B (``"gps_anchored"``): full Sim(3) fit, directly and independently onto GPS.

    Aligns ``submap`` onto its own GPS-tagged keyframes (typically the
    whole window, not just the 3-camera junction overlap) with a full
    Umeyama + RANSAC fit -- rotation included. Only sound when the
    submap's own camera-centre track is *not* near-collinear (the caller,
    ``_chain_align``, only selects this strategy when
    ``geometry.flight_profile`` reported a non-collinear-enough
    trajectory); the returned ``degenerate``/``condition_number`` are
    still reported per submap as a safety net regardless.

    Real-footage bug this guards against (read before touching this)
    ------------------------------------------------------------------------
    A flight can be non-collinear *overall* (``flight_profile`` correctly
    recommends ``"gps_anchored"``) while one particular *window* still
    happens to be locally near-straight -- a curved survey is often built
    out of several fairly-straight legs joined by turns, and a window can
    land entirely on one leg. When that happens, this function's own
    Umeyama+RANSAC fit comes back ``degenerate`` for that one submap: not a
    "too few points" failure (which already routes to chaining, above), but
    a *numerically unreliable rotation returned anyway*, exactly the
    "camera centres line up with GPS while the point cloud comes out
    tilted" failure mode this module's docstring warns about. Measured on
    real 16-keyframe drone footage: 2 of 3 submaps had condition numbers in
    the 1e5 range (threshold is 1e4) despite the whole flight's
    collinearity being a comfortable 0.62 (threshold 0.85) -- and the
    resulting merged cloud showed those two submaps' dense geometry
    projected tens to over a hundred metres away from the correct,
    GPS-consistent location, as several small disconnected blobs
    completely detached from the rest of the model (camera *positions*
    still matched GPS to within under a metre; only the *dense points*,
    displaced by the bad rotation acting on their offset from the camera
    centre, were wrong).

    The fix: when this fit comes back degenerate AND ``conditioned_R``
    offers this submap's own telemetry-conditioned orientation (the same
    source ``"telemetry_rotation"`` trusts, independent of camera-centre
    collinearity -- see ``_telemetry_rotation_transform``), substitute that
    rotation and refit only scale+translation from it, instead of keeping
    the unreliable full-Sim(3) rotation. Falls through to the original
    (degenerate, but only available) fit -- still flagged, still logged by
    the caller -- when telemetry conditioning isn't available either.

    Returns ``(None, diag)`` when ``submap`` doesn't have
    ``>= _MIN_CORRESPONDENCES`` of its own GPS-tagged keyframes -- the
    caller falls back to chaining that submap onto its neighbour.
    """
    local_pos = {kf: j for j, kf in enumerate(submap.keyframe_indices)}
    gps_kf = [kf for kf in submap.keyframe_indices if kf in camera_gps_enu]
    if len(gps_kf) < _MIN_CORRESPONDENCES:
        return None, {
            "method": "gps_anchored",
            "failure": "too few own GPS-tagged keyframes for a full Sim(3) fit",
            "n_gps": len(gps_kf),
        }

    src = np.array([submap.poses[local_pos[kf]].t for kf in gps_kf])
    dst = np.array([camera_gps_enu[kf] for kf in gps_kf])
    transform, degenerate, inliers, condition_number = _umeyama_ransac(src, dst, seed=seed)
    if lock_scale is not None:
        # Keep the rotation this fit just estimated, discard its scale. The
        # rotation is constrained by the *shape* of the camera track and is
        # the reason to run the full fit at all; the scale is constrained by
        # its *extent*, which says nothing about a backbone that already
        # emits metres. See umeyama_fixed_rotation's `scale` parameter.
        transform = umeyama_fixed_rotation(src, dst, transform.R, scale=lock_scale)

    if degenerate and conditioned_R:
        corrected, corrected_diag = _telemetry_rotation_transform(
            submap, camera_gps_enu, conditioned_R, seed=seed, lock_scale=lock_scale
        )
        if corrected is not None:
            logger.warning(
                "submap merge: strategy='gps_anchored' submap's own full Sim(3) fit is degenerate "
                "(condition_number=%.4g -- this window's camera track is locally more collinear than "
                "the flight-wide classification expected, even though the whole flight is not) -- "
                "substituting telemetry-conditioned rotation instead of trusting the unreliable "
                "full-Sim(3) rotation (scale/translation are unaffected, still fit from this "
                "submap's own GPS positions). See _gps_full_anchor_transform's docstring.",
                condition_number,
            )
            corrected_residuals = np.linalg.norm(corrected.apply(src) - dst, axis=1)
            diag = {
                "method": "gps_anchored",
                "n_gps": len(gps_kf),
                "n_inliers": int(inliers.sum()),
                "degenerate": False,
                "condition_number": condition_number,
                "rmse_m": float(np.sqrt(np.mean(corrected_residuals**2))),
                "rotation_source": "telemetry_conditioned",
                "rotation_correction_reason": (
                    f"gps_anchored's own full-Sim(3) rotation was degenerate (condition_number={condition_number:.4g})"
                ),
                "telemetry_rotation_diag": corrected_diag,
            }
            return corrected, diag
        # No usable telemetry conditioning for this submap either -- fall
        # through and keep the degenerate fit, exactly as before: still
        # flagged (`degenerate=True` below), still logged loudly by the
        # caller (`_chain_align`'s per-submap warning and
        # `pipeline.stages.GeometryStage`'s own gps_anchored-degenerate
        # check), never silent.

    residuals = np.linalg.norm(transform.apply(src) - dst, axis=1)
    rmse_source = residuals[inliers] if inliers.any() else residuals
    diag = {
        "method": "gps_anchored",
        "n_gps": len(gps_kf),
        "n_inliers": int(inliers.sum()),
        "degenerate": degenerate,
        "condition_number": condition_number,
        "rmse_m": float(np.sqrt(np.mean(rmse_source**2))),
    }
    return transform, diag


def _chain_align(
    submaps: list[Submap],
    camera_gps_enu: dict[int, np.ndarray] | None = None,
    conditioned_R: dict[int, np.ndarray] | None = None,
    strategy: str = "chained_sim3",
    lock_scale: float | None = None,
    keyframe_altitude_m: dict[int, float] | None = None,
    overlap_align: bool = True,
) -> tuple[list[Sim3], list[dict], list[dict | None], list[bool]]:
    """Compute a per-submap Sim(3)-to-global transform, plus diagnostics.

    ``strategy`` selects how each submap is independently anchored before
    any chaining happens at all -- see the module docstring for what
    ``"telemetry_rotation"``/``"gps_anchored"``/``"chained_sim3"`` each do.
    For the first two, every submap is offered to the matching per-submap
    anchor function (``_telemetry_rotation_transform`` /
    ``_gps_full_anchor_transform``); a submap that function can't handle
    (logged as a warning -- never silent) falls back to the third case:
    chained onto the *already-globalized* shared cameras from submap
    ``i - 1`` (``transforms[0]`` defaults to identity when submap 0 itself
    isn't independently anchored, anchoring the global frame the old way).

    Returns ``(transforms, diagnostics, submap_diags, anchored_flags)``:

    - ``transforms[i]`` maps submap ``i``'s local frame into the shared
      global frame.
    - ``diagnostics`` has one entry per junction (``len(submaps) - 1``
      entries) comparing submap ``i``'s shared-with-previous cameras
      against submap ``i - 1``'s already-globalized positions -- this is
      purely informational (a cross-check of how well independently
      -anchored submaps agree) whenever anchoring means it no longer
      *determines* ``transforms[i]``, but it is still computed the same
      way so the accuracy report card's "rising RMSE across junctions"
      drift signal keeps meaning the same thing either way.
    - ``submap_diags[i]`` is the raw per-submap anchor attempt's
      diagnostic (success or failure reason) from
      ``_telemetry_rotation_transform``/``_gps_full_anchor_transform``,
      ``None`` when ``strategy == "chained_sim3"`` or no attempt was made
      (e.g. no ``camera_gps_enu`` given).
    - ``anchored_flags[i]`` is whether submap ``i`` actually ended up
      independently anchored (``True``) vs. falling back to chaining
      (``False``).
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"unknown merge strategy {strategy!r}; expected one of {sorted(_VALID_STRATEGIES)}")

    if not submaps:
        return [], [], [], []

    n = len(submaps)
    transforms: list[Sim3 | None] = [None] * n
    submap_diags: list[dict | None] = [None] * n
    # Per-submap altitude-anchoring diagnostics, merged into submap_diags
    # below so a reader can see which scale source each submap actually
    # used rather than having to infer it.
    altitude_diags: list[dict | None] = [None] * n

    if strategy != "chained_sim3" and camera_gps_enu:
        for i, sm in enumerate(submaps):
            # Scale source, in order of how well founded it is:
            #   1. telemetry altitude (a direct measurement of the distance
            #      the backbone gets wrong -- see altitude_anchor_scale),
            #   2. lock_scale (trust the backbone's metric claim),
            #   3. the Umeyama fit's own scale from camera-centre spread.
            # (3) is last because it measures horizontal extent to correct a
            # vertical error; it is the fallback, not the default.
            submap_scale = lock_scale
            if keyframe_altitude_m:
                measured, alt_diag = altitude_anchor_scale(sm, keyframe_altitude_m)
                altitude_diags[i] = alt_diag
                if measured is not None:
                    submap_scale = measured
                else:
                    logger.warning(
                        "submap merge: submap %d could not be altitude-anchored (%s); "
                        "falling back to %s",
                        i,
                        alt_diag.get("failure", "unknown reason"),
                        "lock_scale" if lock_scale is not None else "the camera-centre scale fit",
                    )
            if strategy == "telemetry_rotation":
                transforms[i], submap_diags[i] = _telemetry_rotation_transform(
                    sm, camera_gps_enu, conditioned_R or {}, seed=1000 + i, lock_scale=submap_scale
                )
            else:  # "gps_anchored"
                transforms[i], submap_diags[i] = _gps_full_anchor_transform(
                    sm, camera_gps_enu, seed=1000 + i, conditioned_R=conditioned_R, lock_scale=submap_scale
                )
            if transforms[i] is None:
                failure = (submap_diags[i] or {}).get("failure", "unknown reason")
                logger.warning(
                    "submap merge: strategy=%r could not independently anchor submap %d "
                    "(keyframes %s) -- %s; falling back to chaining this submap onto its "
                    "neighbour instead",
                    strategy,
                    i,
                    sm.keyframe_indices,
                    failure,
                )

    # Refine translation from the geometry consecutive windows SHARE.
    #
    # The Sim(3) above is fitted to camera centres, which on a survey
    # flight are a near-flat sheet (measured planarity ratio 0.045) -- so
    # it is well determined horizontally and nearly unconstrained
    # vertically. The overlap between windows is the opposite: a
    # horizontal ground surface pins vertical offset extremely well.
    # See geometry.overlap_align.
    if overlap_align and len(submaps) > 1 and all(t is not None for t in transforms):
        from drishti3d.geometry.overlap_align import solve_overlap_translations

        try:
            transforms, overlap_diag = solve_overlap_translations(
                submaps,
                transforms,
                camera_gps_enu=camera_gps_enu,
                keyframe_altitude_m=keyframe_altitude_m,
            )
        except Exception:
            logger.warning("submap merge: overlap alignment failed; keeping the camera-centre fit", exc_info=True)
        else:
            for i, e in enumerate(overlap_diag.get("per_junction", [])):
                if i + 1 < len(submap_diags) and submap_diags[i + 1] is not None:
                    submap_diags[i + 1] = {**submap_diags[i + 1], "overlap_align": e}

    for i, alt_diag in enumerate(altitude_diags):
        if alt_diag is not None and submap_diags[i] is not None:
            submap_diags[i] = {**submap_diags[i], "altitude_anchor": alt_diag}

    anchored_flags = [t is not None for t in transforms]

    if transforms[0] is None:
        transforms[0] = Sim3(scale=1.0, R=np.eye(3), t=np.zeros(3))

    global_pos_by_kf: dict[int, np.ndarray] = {
        kf: transforms[0].apply(submaps[0].poses[j].t[None, :])[0] for j, kf in enumerate(submaps[0].keyframe_indices)
    }

    diagnostics: list[dict] = []

    for i in range(1, n):
        curr = submaps[i]
        local_pos = {kf: j for j, kf in enumerate(curr.keyframe_indices)}
        shared_kf = [kf for kf in curr.window.shared_with_previous if kf in local_pos and kf in global_pos_by_kf]

        chained_transform: Sim3 | None = None
        diag: dict | None = None

        if len(shared_kf) >= _MIN_CORRESPONDENCES:
            src = np.array([curr.poses[local_pos[kf]].t for kf in shared_kf])
            dst = np.array([global_pos_by_kf[kf] for kf in shared_kf])

            chained_transform, degenerate, inliers, condition_number = _umeyama_ransac(src, dst, seed=i)
            if lock_scale is not None:
                # Chaining compounds: a free scale here multiplies into every
                # downstream submap, so an unconstrained scale at one junction
                # displaces the rest of the flight. Same reasoning as the
                # anchored path above.
                chained_transform = umeyama_fixed_rotation(src, dst, chained_transform.R, scale=lock_scale)
            residuals = np.linalg.norm(chained_transform.apply(src) - dst, axis=1)
            rmse_source = residuals[inliers] if inliers.any() else residuals
            diag = {
                "junction": (i - 1, i),
                "rmse_m": float(np.sqrt(np.mean(rmse_source**2))),
                "n_shared": len(shared_kf),
                "n_inliers": int(inliers.sum()),
                "degenerate": degenerate,
                "condition_number": condition_number,
            }
        elif transforms[i] is None:
            raise ValueError(
                f"junction {i - 1}->{i}: only {len(shared_kf)} shared keyframe camera centres "
                f"in common with the previous submap (need >= {_MIN_CORRESPONDENCES}); check "
                "the overlap passed to windows.plan_windows"
            )

        if transforms[i] is None:
            # No GPS anchor for this submap: the chained fit (guaranteed
            # non-None above, or we'd already have raised) IS the transform.
            assert chained_transform is not None
            transforms[i] = chained_transform
        elif diag is None:
            # GPS-anchored, but too few shared cameras with the previous
            # submap to even compute the informational cross-check fit
            # (an edge case -- geometry.windows.plan_windows' overlap floor
            # normally guarantees >= _MIN_CORRESPONDENCES shared cameras at
            # every junction by construction). Fall back to a direct
            # position comparison (no refit needed) so the report card
            # still gets one real, non-fabricated rmse_m per junction.
            common = [kf for kf in shared_kf]
            if common:
                curr_global = np.array([transforms[i].apply(curr.poses[local_pos[kf]].t[None, :])[0] for kf in common])
                prev_global = np.array([global_pos_by_kf[kf] for kf in common])
                rmse = float(np.sqrt(np.mean(np.linalg.norm(curr_global - prev_global, axis=1) ** 2)))
            else:
                rmse = 0.0
            diag = {
                "junction": (i - 1, i),
                "rmse_m": rmse,
                "n_shared": len(common),
                "n_inliers": len(common),
                "degenerate": False,
                "condition_number": 1.0,
            }

        if submap_diags[i] is not None:
            diag = {**diag, "gps_anchored": anchored_flags[i], "gps_diag": submap_diags[i]}

        diagnostics.append(diag)

        for j, kf in enumerate(curr.keyframe_indices):
            global_pos_by_kf[kf] = transforms[i].apply(curr.poses[j].t[None, :])[0]

    return transforms, diagnostics, submap_diags, anchored_flags


# Overlapping ``geometry.windows.plan_windows`` windows deliberately share
# keyframes (real alignment margin -- see that module's docstring), so the
# same keyframe's dense per-pixel points get independently reconstructed by
# every window it falls in and end up concatenated into the merged cloud
# once per window (2-3x over, for a typical ~30% overlap). Post-alignment,
# a duplicated point's copies land at (numerically) the same global
# position -- the Sim(3) fit is derived *from* those very shared-camera
# correspondences, so its residual there is at the solver's numerical
# floor, nowhere near a real few-centimetre disagreement worth keeping both
# copies of. This is an exact/near-exact duplicate collapse on that basis,
# not a general resolution-trading downsample (contrast
# ``fusion.filters.voxel_downsample``, which is a deliberately coarser,
# caller-scaled pass over genuinely-distinct nearby points) -- the
# tolerance is deliberately tiny so it only ever merges true duplicates.
_DEDUP_EPSILON_M = 1e-4  # 0.1 mm: far below any real duplicate's alignment residual, far above float64 noise


def _dedup_merged_points(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    confidence: np.ndarray | None,
    covariance: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Collapse near-exact duplicate points (see module note above) after all submaps are concatenated.

    ``labels`` (per-point submap index, for ``export.placement``) is
    min-reduced over each group: when two windows rendered the same
    keyframe's point to within ``_DEDUP_EPSILON_M`` the survivor is
    attributed to the earlier window. Any choice is arbitrary here --
    the points are the same point -- and a deterministic one keeps the
    placement report reproducible between runs.

    Duplicates are grouped by rounding to a fixed, tiny metric grid
    (``_DEDUP_EPSILON_M``) -- deliberately not a caller-configurable scale,
    since this only ever needs to catch bit-for-bit (to alignment-residual
    precision) re-renderings of the same keyframe, never genuinely close
    but distinct geometry. Grouped points are averaged (xyz, rgb,
    covariance -- covariance elementwise, the same cheap-but-honest proxy
    ``fusion.filters.voxel_downsample`` uses) or min-reduced (confidence --
    consistent with ``voxel_downsample``'s own "a cell is only as
    trustworthy as its least-trustworthy contributor" rule).
    """
    n = xyz.shape[0]
    if n == 0:
        return xyz, rgb, confidence, covariance, labels

    quant = np.round(xyz / _DEDUP_EPSILON_M).astype(np.int64)
    _uniq, inverse, counts = np.unique(quant, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n_out = counts.shape[0]
    if n_out == n:
        return xyz, rgb, confidence, covariance, labels

    xyz_sums = np.zeros((n_out, 3), dtype=np.float64)
    np.add.at(xyz_sums, inverse, xyz)
    xyz_out = xyz_sums / counts[:, None]

    rgb_out = None
    if rgb is not None:
        rgb_sums = np.zeros((n_out, rgb.shape[-1]), dtype=np.float64)
        np.add.at(rgb_sums, inverse, rgb.astype(np.float64))
        rgb_out = np.round(rgb_sums / counts[:, None]).astype(rgb.dtype)

    confidence_out = None
    if confidence is not None:
        if np.issubdtype(confidence.dtype, np.integer):
            fill = np.iinfo(confidence.dtype).max
        else:
            fill = np.inf
        confidence_out = np.full(n_out, fill, dtype=confidence.dtype)
        np.minimum.at(confidence_out, inverse, confidence)

    covariance_out = None
    if covariance is not None:
        cov_sums = np.zeros((n_out, 3, 3), dtype=np.float64)
        np.add.at(cov_sums, inverse, covariance)
        covariance_out = cov_sums / counts[:, None, None]

    labels_out = None
    if labels is not None:
        labels_out = np.full(n_out, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(labels_out, inverse, labels.astype(np.int64))

    return xyz_out, rgb_out, confidence_out, covariance_out, labels_out


def merge_submaps(
    submaps: list[Submap],
    method: str = "umeyama",
    camera_gps_enu: dict[int, np.ndarray] | None = None,
    conditioned_R: dict[int, np.ndarray] | None = None,
    strategy: str = "chained_sim3",
    lock_scale: float | None = None,
    keyframe_altitude_m: dict[int, float] | None = None,
    overlap_align: bool = True,
    return_labels: bool = False,
) -> tuple[PointCloud, list[Pose]] | tuple[PointCloud, list[Pose], np.ndarray]:
    """Align a sequence of submaps into one global point cloud and pose list.

    ``return_labels`` additionally returns a per-point submap index,
    parallel to the returned cloud. ``export.placement`` needs it to
    colour each window separately and to measure how far apart the
    windows placed the same ground -- a question the merged cloud alone
    cannot answer, because concatenation erases which window a point came
    from. Off by default so every existing caller's 2-tuple unpacking is
    unaffected.

    ``strategy`` (one of ``"telemetry_rotation"``, ``"gps_anchored"``,
    ``"chained_sim3"`` -- see this module's docstring) picks how each
    submap gets its transform; typically the caller passes through
    ``geometry.flight_profile.FlightProfile.recommended_merge_strategy``
    (``pipeline.stages.GeometryStage`` does exactly this). Defaults to
    ``"chained_sim3"`` -- the original chain-onto-the-previous-submap
    behaviour -- so calling this without ``camera_gps_enu``/``strategy``
    at all (as every synthetic test in this module does) is unaffected.
    Use ``alignment_residuals``/``strategy_report`` to inspect per
    -junction/per-submap fit quality without re-merging.

    ``camera_gps_enu`` (a ``dict[int, np.ndarray]`` mapping global keyframe
    index to a GPS-derived ENU position, see
    ``ingest.telemetry.telemetry_to_enu``) and ``conditioned_R`` (mapping
    global keyframe index to the world-from-camera rotation that keyframe
    was conditioned with, e.g. ``Keyframe.pose.R``) are what
    ``"telemetry_rotation"``/``"gps_anchored"`` anchor each submap on --
    see ``_telemetry_rotation_transform``/``_gps_full_anchor_transform``.

    The concatenated result is deduplicated (see ``_dedup_merged_points``)
    before being returned: overlapping windows reconstruct their shared
    keyframes' points independently, so without this step the merged cloud
    would count every overlap keyframe's geometry once per window it
    appears in. Callers that want the dedup count (raw concatenated total
    minus the returned cloud's size) can compute it from
    ``sum(sm.points.xyz.reshape(-1, 3).shape[0] for sm in submaps)`` vs.
    the returned ``PointCloud.xyz.shape[0]`` -- see
    ``pipeline.stages.GeometryStage``/``FusionStage``, which report it.
    """
    if method != "umeyama":
        raise ValueError(f"unsupported merge method {method!r}; only 'umeyama' is implemented")

    if not submaps:
        empty = PointCloud(xyz=np.zeros((0, 3), dtype=np.float64))
        if return_labels:
            return empty, [], np.zeros(0, dtype=np.int64)
        return empty, []

    transforms, _diagnostics, _submap_diags, _anchored = _chain_align(
        submaps,
        camera_gps_enu=camera_gps_enu,
        conditioned_R=conditioned_R,
        strategy=strategy,
        lock_scale=lock_scale,
        keyframe_altitude_m=keyframe_altitude_m,
        overlap_align=overlap_align,
    )

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    cov_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    pose_by_kf: dict[int, Pose] = {}

    for submap_index, (submap, transform) in enumerate(zip(submaps, transforms, strict=True)):
        points = submap.points.xyz.reshape(-1, 3)
        xyz_parts.append(transform.apply(points))
        label_parts.append(np.full(points.shape[0], submap_index, dtype=np.int64))
        if submap.points.rgb is not None:
            rgb_parts.append(submap.points.rgb.reshape(-1, submap.points.rgb.shape[-1]))
        if submap.points.covariance is not None:
            # Propagate each point's local-frame covariance through the
            # same Sim(3) this submap's points/poses were just transformed
            # by: Cov' = scale^2 * R @ Cov @ R^T (the standard linear/
            # similarity-transform covariance propagation rule).
            cov = submap.points.covariance.reshape(-1, 3, 3)
            cov_parts.append((transform.scale**2) * np.einsum("ij,njk,lk->nil", transform.R, cov, transform.R))
        conf_parts.append(np.asarray(submap.confidence).reshape(-1))

        for j, kf in enumerate(submap.keyframe_indices):
            # Later submaps' poses for a shared keyframe are kept (they've
            # been aligned through one more, presumably-refining, fit) --
            # overwriting here as we go through submaps in order achieves
            # exactly that.
            pose_by_kf[kf] = transform.apply_pose(submap.poses[j])

    xyz = np.concatenate(xyz_parts, axis=0) if xyz_parts else np.zeros((0, 3))
    rgb = np.concatenate(rgb_parts, axis=0) if len(rgb_parts) == len(submaps) and rgb_parts else None
    confidence = np.concatenate(conf_parts, axis=0) if conf_parts else None
    # Only carry covariance through if *every* submap had it -- a partial
    # set would silently misalign covariance rows with the wrong points
    # once concatenated (same reasoning as the rgb_parts length check
    # above).
    covariance = np.concatenate(cov_parts, axis=0) if len(cov_parts) == len(submaps) and cov_parts else None
    labels = np.concatenate(label_parts, axis=0) if label_parts else np.zeros(0, dtype=np.int64)

    xyz, rgb, confidence, covariance, labels = _dedup_merged_points(xyz, rgb, confidence, covariance, labels)

    poses = [pose_by_kf[kf] for kf in sorted(pose_by_kf)]

    cloud = PointCloud(xyz=xyz, rgb=rgb, covariance=covariance, confidence=confidence)
    if return_labels:
        return cloud, poses, labels
    return cloud, poses


def alignment_residuals(
    submaps: list[Submap],
    camera_gps_enu: dict[int, np.ndarray] | None = None,
    conditioned_R: dict[int, np.ndarray] | None = None,
    strategy: str = "chained_sim3",
) -> list[dict]:
    """Per-junction alignment diagnostics for ``merge_submaps``' input.

    Pass the same ``camera_gps_enu``/``conditioned_R``/``strategy`` given
    to ``merge_submaps`` to get diagnostics consistent with what that call
    actually did; omitting them still returns one informational entry per
    junction (see ``_chain_align``'s docstring on how that comparison is
    computed either way). ``strategy_report`` wraps this with per-submap
    (not just per-junction) detail and a top-level strategy summary.

    Returns one dict per junction (``len(submaps) - 1`` total; empty for 0
    or 1 submaps), each with:

    - ``junction``: ``(i, i + 1)``, the pair of submap indices this
      diagnostic is for.
    - ``rmse_m``: RMSE (metres), over the inlier shared cameras, between
      submap ``i + 1``'s aligned camera centres and submap ``i``'s
      already-globalized ones. Rising RMSE across junctions is the drift
      signal this exists to surface for the accuracy report card --
      still meaningful even when independent anchoring made both submaps'
      transforms independent of each other, as a cross-check of how well
      they agree.
    - ``n_shared`` / ``n_inliers``: how many shared cameras were available
      / survived RANSAC.
    - ``degenerate``: True if the shared cameras were too close to
      collinear to trust this junction's rotation (see module docstring).
    - ``condition_number``: the raw number ``degenerate`` was thresholded
      from, for anyone who wants a finer-grained signal than the boolean.
    - ``gps_anchored`` / ``gps_diag``: present only when submap ``i + 1``
      had an independent-anchor attempt (``strategy != "chained_sim3"``
      and ``camera_gps_enu`` given); ``gps_anchored`` is whether that
      attempt succeeded, ``gps_diag`` carries its diagnostics either way
      (success metrics, or a ``"failure"`` reason).
    """
    _, diagnostics, _submap_diags, _anchored = _chain_align(
        submaps, camera_gps_enu=camera_gps_enu, conditioned_R=conditioned_R, strategy=strategy
    )
    return diagnostics


def strategy_report(
    submaps: list[Submap],
    camera_gps_enu: dict[int, np.ndarray] | None = None,
    conditioned_R: dict[int, np.ndarray] | None = None,
    strategy: str = "chained_sim3",
) -> dict:
    """Full merge-strategy diagnostics: per-submap anchor attempts + per-junction residuals + a summary.

    Where ``alignment_residuals`` only reports junctions (so submap 0,
    which has no "previous" junction, is invisible), this also reports
    every submap's own independent-anchor attempt (submap 0 included) --
    what ``pipeline.stages.GeometryStage`` needs to tell whether the
    *requested* strategy (``strategy``, typically
    ``FlightProfile.recommended_merge_strategy``) was actually usable for
    this flight, or silently degraded submap-by-submap into chaining (see
    module docstring: that degradation is logged, never silent, but a
    caller assembling the accuracy report/``StageResult`` still needs the
    aggregate count, not just individual log lines).

    Returns a dict with:

    - ``strategy``: the requested strategy (echoed back).
    - ``n_submaps``: total submap count.
    - ``n_anchored``: how many submaps were independently anchored (True
      in ``anchored_flags``) -- ``0`` whenever ``strategy ==
      "chained_sim3"`` or no ``camera_gps_enu`` was given, by construction.
    - ``n_fallback_to_chaining``: ``n_submaps - n_anchored`` (excluding
      submap 0, which has no "chaining" alternative to fall back to when
      it isn't anchored -- it just anchors the global frame at identity).
    - ``fully_used_requested_strategy``: ``True`` iff every submap that
      *could* be independently anchored under ``strategy`` was (i.e. no
      submap silently fell back) -- ``True`` trivially for
      ``"chained_sim3"``.
    - ``submap_diagnostics``: list of ``(anchored: bool, diag: dict |
      None)`` pairs, one per submap, from the strategy-specific anchor
      function.
    - ``junction_residuals``: same list ``alignment_residuals`` returns.
    """
    _transforms, diagnostics, submap_diags, anchored_flags = _chain_align(
        submaps, camera_gps_enu=camera_gps_enu, conditioned_R=conditioned_R, strategy=strategy
    )
    n = len(submaps)
    n_anchored = sum(anchored_flags)
    # Every submap other than 0 that wasn't anchored had to fall back to
    # chaining; submap 0 (index 0) never "falls back" in that sense -- with
    # nothing before it, it either anchors independently or seeds the
    # global frame at identity, there is no neighbour to chain onto.
    n_fallback = sum(1 for i in range(1, n) if not anchored_flags[i]) if n else 0
    attempted = strategy != "chained_sim3" and bool(camera_gps_enu)
    fully_used = (not attempted) or (n_anchored == n)
    return {
        "strategy": strategy,
        "n_submaps": n,
        "n_anchored": n_anchored,
        "n_fallback_to_chaining": n_fallback,
        "fully_used_requested_strategy": fully_used,
        "submap_diagnostics": list(zip(anchored_flags, submap_diags, strict=True)),
        "junction_residuals": diagnostics,
    }
