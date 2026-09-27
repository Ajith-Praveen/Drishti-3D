"""Camera yaw from image motion + GPS displacement, for when the gimbal log lies.

Why the gimbal heading cannot be trusted
-----------------------------------------
On the sample flight the telemetry's gimbal heading held one value for whole
legs and jumped at the corners: 10 distinct values across 77 keyframes while
the GPS course turned continuously. At one keyframe the logged yaw was 87
degrees from the direction of travel. On the straight legs it sat a constant
15 degrees from the compass heading, which itself tracked the GPS course to
within 2 degrees -- so even where the log was not stale, it disagreed with
the aircraft by a fixed offset that nothing in the export explains.

A camera rotation that wrong poisons everything conditioned on it:
MapAnything's pose conditioning (its depth was worst exactly on the loop),
the parallax depth anchor (triangulation with a wrong rotation fails its own
reprojection gate, so the loop windows produced 0-2 samples), and the
telemetry-rotation submap merge.

What the images know that the log does not
-------------------------------------------
Between two nadir keyframes the ground slides across the image in one
direction: opposite to the drone's motion, expressed in the camera's axes.
GPS gives that motion in world axes. The rotation that maps one onto the
other is the camera's yaw -- measured from the footage and the GPS track,
which are the two mandatory inputs the problem statement guarantees, and
independent of any gimbal field.

Concretely, for keyframes ``i`` and ``i+1``:

1. match features, keep the geometrically verified inliers, and take the
   circular mean of their displacement direction: ``phi_measured``;
2. for a candidate yaw, build the world-from-camera rotation the way the
   rest of the pipeline does (``bundle.gimbal_to_R``), express the GPS
   displacement in camera axes, negate it (ground moves opposite to the
   camera), and read its image-plane direction: ``phi_predicted(yaw)``;
3. pick the yaw where the two agree. It is a 1-D search over a periodic
   function with one minimum, so a coarse grid then a fine one is exact
   enough (0.1 degree) and cheap.

Each keyframe gets the circular mean of the estimates from the pair before
and the pair after it. Where a pair cannot be measured (the drone was not
moving, or matching failed) the telemetry yaw is kept, and the fallback is
recorded per keyframe rather than blended in silently.

Pitch and roll are taken from telemetry as before: a nadir gimbal's pitch
reading has been reliable on every flight seen so far, and the image-motion
cue constrains yaw only.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from drishti3d.geometry.bundle import gimbal_to_R
from drishti3d.geometry.features import (
    detect_and_describe,
    geometric_verify,
    match_features,
)

logger = logging.getLogger(__name__)

__all__ = ["estimate_yaw_from_flow", "yaw_for_pair"]

#: Below this displacement the flow direction is noise, not motion.
_MIN_MOVE_M = 2.0

#: Verified inliers needed before a pair's flow direction is believed.
_MIN_INLIERS = 30


def _wrap_deg(a: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(a) + 180.0) % 360.0) - 180.0


def _predicted_flow_angle(yaw_deg: np.ndarray, pitch_deg: float, roll_deg: float, d_world: np.ndarray) -> np.ndarray:
    """Image-plane direction (deg) the ground appears to move, per candidate yaw."""
    out = np.empty(len(yaw_deg))
    for k, y in enumerate(yaw_deg):
        R = gimbal_to_R(float(y), pitch_deg, roll_deg)
        d_cam = R.T @ d_world
        # Ground moves opposite to the camera; image x = cam x, image y = cam y.
        out[k] = np.degrees(np.arctan2(-d_cam[1], -d_cam[0]))
    return out


def yaw_for_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    d_world: np.ndarray,
    pitch_deg: float,
    roll_deg: float,
    *,
    max_features: int = 1500,
) -> tuple[float | None, dict]:
    """Yaw (deg) of camera ``a`` from the flow between ``a`` and ``b``. ``None`` if unmeasurable."""
    move = float(np.linalg.norm(d_world[:2]))
    if move < _MIN_MOVE_M:
        return None, {"failure": f"displacement {move:.1f} m below {_MIN_MOVE_M} m"}

    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
    fa = detect_and_describe(ga, method="sift", max_features=max_features)
    fb = detect_and_describe(gb, method="sift", max_features=max_features)
    raw = match_features(fa, fb, ratio=0.8)
    if len(raw) < _MIN_INLIERS:
        return None, {"failure": f"only {len(raw)} raw matches"}
    ver = geometric_verify(fa, fb, raw)
    if ver.inlier_mask is None or int(ver.inlier_mask.sum()) < _MIN_INLIERS:
        n = 0 if ver.inlier_mask is None else int(ver.inlier_mask.sum())
        return None, {"failure": f"only {n} verified inliers"}

    q, t = ver.query_idx[ver.inlier_mask], ver.train_idx[ver.inlier_mask]
    flow = fb.keypoints[t, :2] - fa.keypoints[q, :2]
    norms = np.linalg.norm(flow, axis=1)
    keep = norms > 1e-3
    if keep.sum() < _MIN_INLIERS:
        return None, {"failure": "flow too small to give a direction"}
    unit = flow[keep] / norms[keep, None]
    mean_vec = unit.mean(axis=0)
    coherence = float(np.linalg.norm(mean_vec))
    phi_measured = float(np.degrees(np.arctan2(mean_vec[1], mean_vec[0])))

    # Separate image rotation from translation before deriving a heading.
    # Raw feature flow includes camera yaw, especially around flight turns.
    # For nadir imagery a similarity maps reference pixels into source
    # pixels; its centre displacement, rotated back, is translation in A.
    affine, affine_inliers = cv2.estimateAffinePartial2D(
        fa.keypoints[q, :2], fb.keypoints[t, :2], method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99,
    )
    if affine is None or affine_inliers is None or int(affine_inliers.sum()) < _MIN_INLIERS:
        return None, {"failure": "cannot separate image rotation from translation"}
    A = affine[:, :2]
    if abs(np.linalg.det(A)) < 1e-8:
        return None, {"failure": "degenerate image motion"}
    centre = np.array([img_a.shape[1] / 2, img_a.shape[0] / 2])
    translation = np.linalg.solve(A, A @ centre + affine[:, 2] - centre)
    if np.linalg.norm(translation) < 1:
        return None, {"failure": "insufficient translation after compensating rotation"}
    phi_measured = float(np.degrees(np.arctan2(translation[1], translation[0])))
    yaw_change = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))

    # Coarse-to-fine 1-D search. The residual is periodic with a single
    # minimum for a nadir camera, so this is exact to the fine step.
    coarse = np.arange(0.0, 360.0, 1.0)
    res = np.abs(_wrap_deg(_predicted_flow_angle(coarse, pitch_deg, roll_deg, d_world) - phi_measured))
    best = coarse[int(np.argmin(res))]
    fine = np.arange(best - 1.5, best + 1.5, 0.1)
    res_f = np.abs(_wrap_deg(_predicted_flow_angle(fine, pitch_deg, roll_deg, d_world) - phi_measured))
    yaw = float(fine[int(np.argmin(res_f))] % 360.0)

    return yaw, {
        "yaw_b_deg": (yaw + yaw_change) % 360.0,
        "image_rotation_deg": round(yaw_change, 3),
        "inliers": int(keep.sum()),
        "flow_coherence": round(coherence, 3),
        "residual_deg": round(float(res_f.min()), 2),
        "move_m": round(move, 2),
    }


def _circular_mean_deg(angles: list[float]) -> float:
    a = np.radians(angles)
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0)


def estimate_yaw_from_flow(
    images: list[np.ndarray | None],
    positions: list[np.ndarray | None],
    pitch_deg: list[float | None],
    roll_deg: list[float | None],
    prior_yaw_deg: list[float | None],
    *,
    max_features: int = 1500,
) -> tuple[list[float | None], dict]:
    """Per-keyframe yaw from image flow + GPS. Falls back to ``prior_yaw_deg`` per keyframe.

    Returns ``(yaw_per_keyframe, diag)``. ``diag["per_keyframe"]`` records,
    for every keyframe, which source its yaw came from and how far it sits
    from the telemetry prior -- the offset between the two is itself a
    finding worth reading (a constant one means a fixed gimbal offset; a
    wandering one means the log is stale).
    """
    n = len(images)
    pair_yaw: list[float | None] = [None] * max(n - 1, 0)
    pair_diag: list[dict] = [{} for _ in range(max(n - 1, 0))]
    for i in range(n - 1):
        if images[i] is None or images[i + 1] is None or positions[i] is None or positions[i + 1] is None:
            pair_diag[i] = {"failure": "missing image or position"}
            continue
        pitch = pitch_deg[i] if pitch_deg[i] is not None else -90.0
        roll = roll_deg[i] if roll_deg[i] is not None else 0.0
        d_world = np.asarray(positions[i + 1], dtype=np.float64) - np.asarray(positions[i], dtype=np.float64)
        pair_yaw[i], pair_diag[i] = yaw_for_pair(
            images[i], images[i + 1], d_world, pitch, roll, max_features=max_features
        )

    yaws: list[float | None] = []
    per_kf: list[dict] = []
    n_flow = 0
    deltas: list[float] = []
    for i in range(n):
        previous_yaw = pair_diag[i - 1].get("yaw_b_deg", pair_yaw[i - 1]) if i > 0 else None
        candidates = [y for y in (previous_yaw, pair_yaw[i] if i < n - 1 else None) if y is not None]
        prior = prior_yaw_deg[i]
        if candidates:
            yaw = _circular_mean_deg(candidates)
            source = "flow"
            n_flow += 1
        else:
            yaw = float(prior) % 360.0 if prior is not None else None
            source = "telemetry" if prior is not None else "none"
        delta = float(_wrap_deg(yaw - prior)) if (yaw is not None and prior is not None) else None
        if source == "flow" and delta is not None:
            deltas.append(delta)
        yaws.append(yaw)
        per_kf.append({"keyframe": i, "yaw": yaw, "source": source, "prior_yaw": prior, "delta_deg": delta})

    diag = {
        "keyframes": n,
        "from_flow": n_flow,
        "from_telemetry": sum(1 for p in per_kf if p["source"] == "telemetry"),
        "median_delta_deg": float(np.median(deltas)) if deltas else None,
        "delta_iqr_deg": float(np.subtract(*np.percentile(deltas, [75, 25]))) if len(deltas) > 3 else None,
        "per_keyframe": per_kf,
        "pairs": pair_diag,
    }
    if deltas:
        logger.info(
            "yaw from flow: %d/%d keyframes measured; flow - telemetry yaw median %+.1f deg (IQR %s)",
            n_flow,
            n,
            diag["median_delta_deg"],
            f"{diag['delta_iqr_deg']:.1f}" if diag["delta_iqr_deg"] is not None else "n/a",
        )
    return yaws, diag
