"""Fit each view's dense depth to the pose-prior bundle adjustment's sparse points.

Why per view, and why here
--------------------------
``geometry.depth_anchor`` measures ONE scale per window from pair
parallax it triangulates itself. On the sample flight that measurement
came out at 6.0x and still left the ground 18 m below where GPS puts it:
one number cannot absorb a backbone whose error changes from frame to
frame, and a pair-parallax estimate is only as good as its own matches.

By the time geometry runs, ``PosePriorStage`` has already solved a
sparse bundle adjustment -- thousands of points, each seen by several
cameras, at ~1 px reprojection, with GPS position priors fixing the
scale. Those points are the best metric depth this pipeline has. This
module fits every view's predicted depth to the BA points that view
observed::

    z_ba = a * z_pred + b

and applies the fit before the window becomes a submap, so the frame
placement check, the merge and fusion all see BA-grade depth.
``fusion.reanchor`` still runs later as a residual check; after this
pass its ratios should sit near 1.0.

Frame independence
------------------
Backbone points live in the window's frame; BA points live in the world
frame. Neither is transformed into the other. Each BA point is put into
its observing camera's frame with the refined world pose -- which gives
its depth and its pixel -- and the dense depth map is simply read at that
pixel. Depth along the optical axis does not depend on where the camera
sits, so the comparison needs no merge transform.

When the shift is not fitted
----------------------------
On flat nadir ground every sample has nearly the same depth, and ``a``
and ``b`` trade off against each other freely -- any line through the
one cluster fits. When the predicted depths span less than
``min_relative_spread`` of their median, the fit falls back to scale
only (``b = 0``, ``a`` = median ratio), which is well determined.

Spatially varying correction
----------------------------
One ``(a, b)`` per view left a 6.4-7.4 m RMS residual on DJI_1001 (~2.3%
at 280 m): the backbone's error is not one number per view but a smooth
tilt/bowl across the image. With ~1,300 BA points per view there is ample
support for a quadratic scale FIELD,

    z_true = z_pred * (c0 + c1 u + c2 v + c3 u^2 + c4 u v + c5 v^2)

with ``(u, v)`` the pixel in [-1, 1]. It is fitted on the affine fit's
inliers, re-weighted twice against 3x the median absolute residual, and
only kept when it beats the affine residual; the field is clamped to the
affine scale range so it can never invert or explode at the image edge.

What it refuses
---------------
A view with fewer than ``min_samples`` inliers, a scale outside
``[min_scale, max_scale]``, or a shift larger than ``max_shift_fraction``
of the median depth is left unchanged and reported, not applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)

__all__ = ["DepthFit", "ba_points_in_camera", "fit_view_depth"]


@dataclass
class DepthFit:
    """Result of fitting one view: ``z_true = scale * z_pred + shift``."""

    applied: bool
    scale: float = 1.0
    shift: float = 0.0
    samples: int = 0
    inliers: int = 0
    residual_m: float | None = None
    mode: str = "none"
    failure: str | None = None
    field: np.ndarray | None = None  # quadratic scale-field coefficients, or None

    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Corrected copy of ``depth`` (invalid pixels stay 0)."""
        d = np.asarray(depth, dtype=np.float64)
        valid = np.isfinite(d) & (d > 1e-6)
        out = np.zeros_like(d)
        if self.field is not None:
            h, w = d.shape
            u, v = np.meshgrid(2 * (np.arange(w) + 0.5) / w - 1,
                               2 * (np.arange(h) + 0.5) / h - 1)
            scale = _field_scale(self.field, u, v)
            out[valid] = d[valid] * scale[valid]
        else:
            out[valid] = self.scale * d[valid] + self.shift
        out[out <= 1e-6] = 0.0
        return out

    def as_dict(self) -> dict:
        return {
            "applied": self.applied,
            "scale": round(self.scale, 5),
            "shift_m": round(self.shift, 3),
            "samples": self.samples,
            "inliers": self.inliers,
            "residual_m": None if self.residual_m is None else round(self.residual_m, 3),
            "mode": self.mode,
            "failure": self.failure,
        }


def _field_basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)


def _field_scale(coef: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _field_basis(u, v) @ coef


def ba_points_in_camera(
    ba_points: np.ndarray,
    obs_camera_idx: np.ndarray,
    obs_point_idx: np.ndarray,
    keyframe_index: int,
    world_pose: Pose,
) -> np.ndarray:
    """Camera-frame coordinates of the BA points ``keyframe_index`` observed, ``(N, 3)``."""
    sel = np.unique(np.asarray(obs_point_idx)[np.asarray(obs_camera_idx) == keyframe_index])
    if sel.size == 0:
        return np.zeros((0, 3))
    world = np.asarray(ba_points, dtype=np.float64)[sel]
    R = np.asarray(world_pose.R, dtype=np.float64)
    t = np.asarray(world_pose.t, dtype=np.float64).reshape(3)
    return (world - t) @ R  # R^T (X - t), row-vector form


def _sample_depth(
    depth: np.ndarray, intr: CameraIntrinsics, cam: np.ndarray, return_uv: bool = False
):
    """Predicted depth at each camera point's pixel. Returns ``(z_true, z_pred)`` for valid pairs."""
    h, w = depth.shape
    z = cam[:, 2]
    front = z > 1e-6
    cam, z = cam[front], z[front]
    # _points_from_depth casts pixel (row, col) through its centre
    # (col + 0.5, row + 0.5), so the pixel a ray lands in is floor(u).
    u = intr.fx * cam[:, 0] / z + intr.cx
    v = intr.fy * cam[:, 1] / z + intr.cy
    ix = np.floor(u).astype(np.int64)
    iy = np.floor(v).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    z_pred = np.asarray(depth, dtype=np.float64)[iy[inside], ix[inside]]
    z_true = z[inside]
    ok = np.isfinite(z_pred) & (z_pred > 1e-6)
    if return_uv:
        un = 2.0 * (ix[inside][ok] + 0.5) / w - 1.0
        vn = 2.0 * (iy[inside][ok] + 0.5) / h - 1.0
        return z_true[ok], z_pred[ok], un, vn
    return z_true[ok], z_pred[ok]


def fit_view_depth(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    cam_points: np.ndarray,
    *,
    min_samples: int = 30,
    min_scale: float = 0.05,
    # Raw MapAnything depth on DJI_1001 is ~14x short, so the fit must be
    # able to supply that whole correction on its own.
    max_scale: float = 50.0,
    max_shift_fraction: float = 0.5,
    min_relative_spread: float = 0.1,
    inlier_fraction: float = 0.05,
    iterations: int = 200,
    seed: int = 0,
    spatial: bool = True,
    min_field_samples: int = 60,
) -> DepthFit:
    """Robustly fit ``z_ba = a * z_pred + b`` for one view. See the module docstring."""
    z_true, z_pred, un, vn = _sample_depth(
        np.asarray(depth), intrinsics, np.asarray(cam_points, dtype=np.float64), return_uv=True
    )
    n = int(z_true.size)
    if n < min_samples:
        return DepthFit(applied=False, samples=n, failure=f"only {n} BA points land on valid depth (< {min_samples})")

    median_true = float(np.median(z_true))
    tol = inlier_fraction * median_true
    q25, q75 = np.percentile(z_pred, [25, 75])
    affine = (q75 - q25) / max(float(np.median(z_pred)), 1e-9) >= min_relative_spread

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    for _ in range(iterations):
        if affine:
            i, j = rng.choice(n, size=2, replace=False)
            dz = z_pred[i] - z_pred[j]
            if abs(dz) < 1e-9:
                continue
            a = (z_true[i] - z_true[j]) / dz
            b = z_true[i] - a * z_pred[i]
        else:
            i = rng.integers(n)
            a, b = z_true[i] / z_pred[i], 0.0
        if not (min_scale <= a <= max_scale):
            continue
        inliers = np.abs(a * z_pred + b - z_true) <= tol
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    k = int(best_inliers.sum())
    mode = "scale_shift" if affine else "scale"
    if k < min_samples:
        return DepthFit(applied=False, samples=n, inliers=k, mode=mode, failure=f"only {k} inliers (< {min_samples})")

    zp, zt = z_pred[best_inliers], z_true[best_inliers]
    if affine:
        A = np.stack([zp, np.ones_like(zp)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, zt, rcond=None)
    else:
        a, b = float(np.median(zt / zp)), 0.0
    a, b = float(a), float(b)
    residual = float(np.sqrt(np.mean((a * zp + b - zt) ** 2)))

    fit = DepthFit(applied=False, scale=a, shift=b, samples=n, inliers=k, residual_m=residual, mode=mode)
    if not (min_scale <= a <= max_scale):
        fit.failure = f"scale {a:.3f} outside [{min_scale}, {max_scale}]"
        return fit
    if abs(b) > max_shift_fraction * median_true:
        fit.failure = f"shift {b:.2f} m exceeds {max_shift_fraction:.0%} of median depth {median_true:.1f} m"
        return fit
    fit.applied = True
    if spatial and k >= min_field_samples:
        _fit_field(fit, z_true[best_inliers], z_pred[best_inliers], un[best_inliers], vn[best_inliers], min_scale, max_scale)
    return fit


def _fit_field(fit: DepthFit, zt, zp, u, v, min_scale: float, max_scale: float) -> None:
    """Upgrade an applied affine fit to a quadratic scale field when that fits better (see module docstring)."""
    A = _field_basis(u, v) * zp[:, None]
    w = np.ones_like(zt)
    coef = None
    for _ in range(3):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], zt * sw, rcond=None)
        r = A @ coef - zt
        mad = float(np.median(np.abs(r))) + 1e-9
        w = (np.abs(r) <= 3.0 * 1.4826 * mad).astype(np.float64)
        if w.sum() < 6:
            return
    keep = w > 0
    residual = float(np.sqrt(np.mean((A[keep] @ coef - zt[keep]) ** 2)))
    # A quadratic can have interior/edge extrema missed by corners+centre.
    candidates = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)]
    for u0 in (-1., 1.):
        if abs(coef[5]) > 1e-12:
            v0 = -(coef[2] + coef[4] * u0) / (2 * coef[5])
            if -1 <= v0 <= 1:
                candidates.append((u0, v0))
    for v0 in (-1., 1.):
        if abs(coef[3]) > 1e-12:
            u0 = -(coef[1] + coef[4] * v0) / (2 * coef[3])
            if -1 <= u0 <= 1:
                candidates.append((u0, v0))
    hessian = np.array([[2 * coef[3], coef[4]], [coef[4], 2 * coef[5]]])
    if abs(np.linalg.det(hessian)) > 1e-12:
        stationary = np.linalg.solve(hessian, -coef[1:3])
        if np.all(np.abs(stationary) <= 1):
            candidates.append(tuple(stationary))
    uv = np.asarray(candidates)
    extrema = _field_scale(coef, uv[:, 0], uv[:, 1])
    # Compare on identical support: rejecting samples must not make a more
    # flexible model win merely by grading itself on fewer points.
    baseline = float(np.sqrt(np.mean((fit.scale * zp[keep] + fit.shift - zt[keep]) ** 2)))
    if residual < 0.95 * baseline and np.all(np.isfinite(extrema) & (extrema >= min_scale) & (extrema <= max_scale)):
        fit.field = coef
        fit.residual_m = residual
        fit.mode = "scale_field"
