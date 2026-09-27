"""Physical acceptance checks independent of reprojection convergence."""
from __future__ import annotations

import numpy as np


class ReconstructionRejected(RuntimeError):
    """A physical validation failure stops even an optional pipeline stage."""


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


#: Largest radial displacement at the image corner accepted from a solved
#: Brown lens (flight01's wide action-camera lens: ~14%). Beyond this the
#: polynomial is extrapolating, not describing a lens.
_MAX_CORNER_DISTORTION = 0.4


def _lens_failure(k) -> tuple[str | None, float | None]:
    """Why a lens's radial model is non-physical inside its own image, if it is.

    A radial polynomial that folds over (radius stops increasing) maps two
    rays to one pixel and cannot be undistorted; that happens when a
    self-calibration extrapolates k1/k2 past the data.
    """
    if k.dist_coeffs is None:
        return None, None
    d = np.zeros(5)
    given = np.asarray(k.dist_coeffs, dtype=np.float64).reshape(-1)[:5]
    d[: given.size] = given
    if not np.isfinite(d).all():
        return "non-finite distortion coefficients", None
    if not d.any():
        return None, 0.0
    corners = np.array([[0.0, 0.0], [k.width, 0.0], [0.0, k.height], [k.width, k.height]])
    r_max = float(np.max(np.hypot((corners[:, 0] - k.cx) / k.fx, (corners[:, 1] - k.cy) / k.fy)))
    r = np.linspace(0.0, 1.05 * r_max, 256)
    r2 = r * r
    rd = r * (1 + r2 * (d[0] + r2 * (d[1] + r2 * d[4])))
    if np.any(np.diff(rd) <= 0):
        return "radial distortion folds over inside the image", None
    corner = float(abs(rd[-1] / r[-1] - 1))
    if corner > _MAX_CORNER_DISTORTION:
        return f"radial distortion {corner:.0%} at the image corner exceeds {_MAX_CORNER_DISTORTION:.0%}", corner
    return None, corner


def validate_cameras(poses, reference, intrinsics, reference_intrinsics, *, gps=None, attitude_reference=None) -> dict:
    """Check metre/degree bounds against references preceding ALL BA passes.

    Long baselines check scale without aligning away errors. These conservative
    safety bounds are not an accuracy certification.

    ``attitude_reference`` is an optional second, independent attitude per
    camera (``None`` entries allowed): the raw telemetry attitude when the
    seed rotations came from another estimator. Each camera's rotation
    change is measured against whichever of the two it is closer to, so a
    camera passes when it agrees with at least one attitude measurement.
    Neither source is reliable alone: on flight01 yaw-from-flow seeded two
    cameras 29 deg away from a correct gimbal log; on the sample flight
    described in ``geometry.yaw_from_flow`` the log was stale by up to
    87 deg.
    """
    n = len(reference)
    if not n or len(poses) != n or len(intrinsics) != n or len(reference_intrinsics) != n:
        raise ReconstructionRejected("camera validation: inconsistent camera/calibration counts")
    if attitude_reference is not None and len(attitude_reference) != n:
        raise ReconstructionRejected("camera validation: inconsistent attitude reference count")
    shifts, angles, focal_ratios = [], [], []
    for i, (p, seed, k, k0) in enumerate(zip(poses, reference, intrinsics, reference_intrinsics, strict=True)):
        if p is None or seed is None:
            raise ReconstructionRejected(f"camera {i}: missing conditioning pose")
        if not all(np.isfinite(v).all() for v in (p.R, p.t, seed.R, seed.t)):
            raise ReconstructionRejected(f"camera {i}: non-finite pose")
        if not np.allclose(p.R.T @ p.R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(p.R), 1, atol=1e-5):
            raise ReconstructionRejected(f"camera {i}: invalid world-from-camera rotation")
        anchor = gps.get(i, seed.t) if gps is not None else seed.t
        if not np.isfinite(anchor).all():
            raise ReconstructionRejected(f"camera {i}: non-finite GPS anchor")
        shifts.append(float(np.linalg.norm(p.t - anchor)))
        angle = _angle_deg(p.R, seed.R)
        alt = attitude_reference[i] if attitude_reference is not None else None
        if alt is not None and np.isfinite(alt.R).all():
            angle = min(angle, _angle_deg(p.R, alt.R))
        angles.append(angle)
        vals = [k.fx, k.fy, k.cx, k.cy, k0.fx, k0.fy]
        if not np.isfinite(vals).all() or min(k.fx, k.fy, k0.fx, k0.fy, k.width, k.height) <= 0:
            raise ReconstructionRejected(f"camera {i}: invalid calibration")
        if (k.width, k.height) != (k0.width, k0.height):
            raise ReconstructionRejected(f"camera {i}: calibration resolution changed")
        ratios = np.array([k.fx / k0.fx, k.fy / k0.fy])
        if np.any((ratios < 0.5) | (ratios > 2.0)):
            raise ReconstructionRejected(f"camera {i}: focal/depth scale outside [0.5, 2.0]")
        if abs(k.cx - k0.cx) > 0.05 * k.width or abs(k.cy - k0.cy) > 0.05 * k.height:
            raise ReconstructionRejected(f"camera {i}: principal point moved over 5% of image size")
        if abs(ratios[0] / ratios[1] - 1) > 0.05:
            raise ReconstructionRejected(f"camera {i}: focal aspect ratio changed over 5%")
        lens_problem, _ = _lens_failure(k)
        if lens_problem:
            raise ReconstructionRejected(f"camera {i}: {lens_problem}")
        focal_ratios.append(float(ratios.mean()))
    diagnostics = {
        "median_position_shift_m": float(np.median(shifts)),
        "max_position_shift_m": float(np.max(shifts)),
        "median_rotation_change_deg": float(np.median(angles)),
        "max_rotation_change_deg": float(np.max(angles)),
    }
    _, corner = _lens_failure(intrinsics[0])
    if corner is not None:
        diagnostics["lens_corner_distortion"] = corner
    limits = {"median_position_shift_m": 15.0, "max_position_shift_m": 30.0,
              "median_rotation_change_deg": 15.0, "max_rotation_change_deg": 30.0}
    failures = [f"{key}={diagnostics[key]:.3f} exceeds {limit:g}" for key, limit in limits.items() if diagnostics[key] > limit]
    if max(focal_ratios) / min(focal_ratios) > 1.15:
        failures.append("per-camera focal corrections differ by over 15%")
    indices = sorted(gps) if gps is not None else list(range(n))
    ratios = []
    for ai, i in enumerate(indices):
        for j in indices[ai + 1:]:
            a = gps[i] if gps is not None else reference[i].t
            b = gps[j] if gps is not None else reference[j].t
            baseline = np.linalg.norm(a - b)
            if baseline >= 10.0:
                ratios.append(float(np.linalg.norm(poses[i].t - poses[j].t) / baseline))
    if ratios:
        scale = float(np.median(ratios))
        diagnostics["baseline_scale_ratio"] = scale
        if not 0.8 <= scale <= 1.25:
            failures.append(f"baseline scale {scale:.3f} outside [0.8, 1.25]")
    if failures:
        raise ReconstructionRejected("Camera solution rejected before dense geometry: " + "; ".join(failures)
                                     + ". Verify telemetry synchronization, heading conventions and camera calibration.")
    return diagnostics


def validate_sparse_depth(result, problem) -> None:
    """Reject projectively valid points behind their observing cameras."""
    if not np.isfinite(result.points).all() or not np.isfinite(result.rmse_after_px):
        raise ReconstructionRejected("bundle adjustment produced non-finite points/reprojection")
    ci = np.asarray(problem.obs_camera_idx, dtype=int)
    pi = np.asarray(problem.obs_point_idx, dtype=int)
    if not len(ci):
        raise ReconstructionRejected("bundle adjustment has no observed depth")
    rotations = np.asarray([p.R for p in result.poses])
    centres = np.asarray([p.t for p in result.poses])
    depths = np.einsum("ni,ni->n", rotations[ci, :, 2], result.points[pi] - centres[ci])
    if np.mean(depths > 0) < 0.99:
        raise ReconstructionRejected("bundle adjustment rejected: over 1% of observations have non-positive depth")
