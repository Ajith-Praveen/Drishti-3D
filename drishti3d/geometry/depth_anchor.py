"""Anchor a feed-forward backbone's depth to parallax measured from telemetry.

The problem this solves
-----------------------
MapAnything regresses depth. It is conditioned on our GPS/gimbal camera
poses and reports ``is_metric=True``, and the poses it returns do match the
ones it was given -- but a regressor's *depth* comes from what it learned,
not from the baseline it was handed. Its training data is overwhelmingly
close-range, and at survey altitude the learned range prior wins: measured
on real 120 m AGL footage, reconstructed camera-to-ground came out at
**0.15-0.59x** the GPS-confirmed height, inconsistent between windows.
Every downstream symptom -- ground split 50 m between parts of the flight,
mesh in 56-68 disconnected pieces, "no buildings" -- was that one error.

No amount of Sim(3) fitting fixes it. A similarity fit on camera centres
finds scale ~1 (the centres already match GPS); the points are simply too
close to their cameras. The correction has to move each point along its
own viewing ray, and it needs an independent measurement of how far.

The measurement
---------------
Parallax. Two cameras a known metric distance apart (from GPS) observing
the same feature give its depth by triangulation -- the one thing a
single-pass drone video always has. This module matches SIFT features
between views inside a window, triangulates them with the *telemetry*
poses (metric baseline), reads the backbone's own depth at those same
pixels, and takes the robust median of the ratio. That ratio is the
backbone's depth error for that view, measured against the footage rather
than guessed, and each point is then rescaled about its camera centre:

    p' = C + r (p - C)

which is exactly "move along the ray until the depth agrees with parallax".

This is the SRT/GPS alignment step of the intended architecture
(MapAnything -> telemetry alignment -> refinement), done at the level where
the error actually lives: per view, before anything is merged.

What it refuses to do
---------------------
- Rescale by a factor outside ``[min_ratio, max_ratio]``. A backbone that
  is off by 50x has failed, not mis-scaled; rescaling would dress the
  failure up as a model. The window is left as-is and the refusal is
  reported.
- Rescale a view on fewer than ``min_samples`` measurements. The window
  median stands in when a single view is thin; when the whole window is
  thin, nothing is touched.
- Move masked-out (zero-depth) pixels. They are not points.

Limits worth stating
--------------------
Gimbal orientations from telemetry are good to roughly a degree. A
rotation error delta over baseline B at depth Z biases triangulated depth
by about ``Z * delta / B``: ~9% at 23 m spacing, ~30% at 7 m. Pairs are
therefore taken at several strides (larger baselines are better
conditioned) and the median over all of them is used, which cancels much
of the bias. Bundle adjustment downstream refines the rotations; what this
module provides is a scale that is right to a few percent instead of wrong
by a factor of six.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import cv2
import numpy as np

from drishti3d.geometry.backbone import BackboneResult
from drishti3d.geometry.features import detect_and_describe, geometric_verify, match_features
from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)

__all__ = ["anchor_depth_fused", "anchor_depth_to_gps_altitude", "anchor_depth_to_parallax", "measure_depth_ratios"]

#: Furthest apart (in window index) two views are matched. Larger baselines
#: condition triangulation better; beyond ~4 keyframes at survey spacing the
#: footprints stop overlapping and matching returns nothing useful.
_MAX_STRIDE = 4

#: Reprojection gate for a triangulated point, as a fraction of image width.
#: Loose on purpose: telemetry rotations are ~1 deg off, which at 956 px is
#: ~9 px of reprojection error on a perfectly good match. 2% (19 px at
#: 956) keeps those and still rejects wrong matches, which land far away.
_REPROJ_GATE_FRACTION = 0.02

#: Minimum angle between the two viewing rays. Below this the DLT solution
#: is numerically meaningless (module docstring of geometry.triangulate).
_MIN_ANGLE_DEG = 0.5


def _crop_offsets(image_hw: tuple[int, int], grid_hw: tuple[int, int]) -> tuple[int, int]:
    """Offset of the backbone's (patch-aligned, centre-cropped) grid inside the input image.

    Mirrors ``geometry.mapanything.crop_to_patch_multiple`` exactly:
    ``(h - aligned_h) // 2``. Recomputed here from the two shapes rather
    than threaded through as extra state, because the shapes are the
    ground truth and cannot drift from what was actually done.
    """
    top = (image_hw[0] - grid_hw[0]) // 2
    left = (image_hw[1] - grid_hw[1]) // 2
    return max(top, 0), max(left, 0)


def _projection_matrix(pose: Pose, intr: CameraIntrinsics) -> np.ndarray:
    """``K [R^T | -R^T C]`` -- world-to-image for a world-from-camera ``Pose``."""
    r_cw = pose.R.T
    t_cw = -r_cw @ pose.t
    return intr.K() @ np.hstack([r_cw, t_cw[:, None]])


def _reprojection_error(P: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Pixel error of world points ``X`` (N,3) projected through ``P`` against ``uv`` (N,2)."""
    xh = np.hstack([X, np.ones((len(X), 1))])
    proj = xh @ P.T
    w = proj[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pix = proj[:, :2] / w[:, None]
    err = np.linalg.norm(pix - uv, axis=1)
    err[~np.isfinite(err) | (w <= 0)] = np.inf
    return err


def measure_depth_ratios(
    images: list[np.ndarray],
    result: BackboneResult,
    cond_poses: list[Pose | None],
    cond_intrinsics: list[CameraIntrinsics | None],
    *,
    max_features: int = 3000,
    max_stride: int = _MAX_STRIDE,
    min_angle_deg: float = _MIN_ANGLE_DEG,
) -> tuple[list[np.ndarray], dict]:
    """Per-view samples of ``parallax_depth / backbone_depth`` at matched features.

    Returns ``(ratios_per_view, diag)`` where ``ratios_per_view[i]`` is a
    1-D array of ratio samples for view ``i`` (possibly empty). Separated
    from the decision logic in ``anchor_depth_to_parallax`` so the raw
    evidence can be inspected and tested on its own.
    """
    n = len(images)
    ratios: list[list[float]] = [[] for _ in range(n)]
    diag: dict = {"pairs_tried": 0, "pairs_verified": 0, "samples_total": 0}

    usable = [
        i for i in range(n) if cond_poses[i] is not None and cond_intrinsics[i] is not None
    ]
    if len(usable) < 2:
        diag["failure"] = "fewer than two views carry a telemetry pose + intrinsics"
        return [np.zeros(0) for _ in range(n)], diag

    grid_h, grid_w = result.depth.shape[1], result.depth.shape[2]
    offsets = [_crop_offsets(img.shape[:2], (grid_h, grid_w)) for img in images]

    feats = {}
    for i in usable:
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY) if images[i].ndim == 3 else images[i]
        feats[i] = detect_and_describe(gray, method="sift", max_features=max_features)

    gate_px = _REPROJ_GATE_FRACTION * max(img.shape[1] for img in images)

    for a_pos, a in enumerate(usable):
        for b in usable[a_pos + 1 : a_pos + 1 + max_stride]:
            diag["pairs_tried"] += 1
            raw = match_features(feats[a], feats[b], ratio=0.8)
            if len(raw) < 12:
                continue
            ver = geometric_verify(feats[a], feats[b], raw, intrinsics=cond_intrinsics[a], method="essential")
            if ver.inlier_mask is None or int(ver.inlier_mask.sum()) < 12:
                continue
            diag["pairs_verified"] += 1

            q, t = ver.query_idx[ver.inlier_mask], ver.train_idx[ver.inlier_mask]
            uv_a = feats[a].keypoints[q, :2].astype(np.float64)
            uv_b = feats[b].keypoints[t, :2].astype(np.float64)

            P_a = _projection_matrix(cond_poses[a], cond_intrinsics[a])
            P_b = _projection_matrix(cond_poses[b], cond_intrinsics[b])
            Xh = cv2.triangulatePoints(P_a, P_b, uv_a.T, uv_b.T)
            w = Xh[3]
            ok = np.abs(w) > 1e-9
            X = np.full((len(w), 3), np.nan)
            X[ok] = (Xh[:3, ok] / w[ok]).T

            # Geometric sanity: in front of both cameras, decent angle, and
            # the point actually projects back to where it was seen.
            C_a, C_b = cond_poses[a].t, cond_poses[b].t
            ray_a, ray_b = X - C_a, X - C_b
            z_a = ray_a @ cond_poses[a].R[:, 2]
            z_b = ray_b @ cond_poses[b].R[:, 2]
            with np.errstate(invalid="ignore", divide="ignore"):
                cosang = np.sum(ray_a * ray_b, axis=1) / (
                    np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
                )
            angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            err = np.maximum(_reprojection_error(P_a, X, uv_a), _reprojection_error(P_b, X, uv_b))
            good = ok & (z_a > 0) & (z_b > 0) & (angle >= min_angle_deg) & (err <= gate_px)
            if not good.any():
                continue

            for view, uv, z_tri in ((a, uv_a, z_a), (b, uv_b, z_b)):
                top, left = offsets[view]
                u = np.round(uv[good, 0]).astype(int) - left
                v = np.round(uv[good, 1]).astype(int) - top
                inside = (u >= 0) & (v >= 0) & (u < grid_w) & (v < grid_h)
                if not inside.any():
                    continue
                z_bb = result.depth[view][v[inside], u[inside]]
                valid = z_bb > 1e-6
                r = z_tri[good][inside][valid] / z_bb[valid]
                ratios[view].extend(r[np.isfinite(r) & (r > 0)].tolist())

    out = [np.asarray(r, dtype=np.float64) for r in ratios]
    diag["samples_total"] = int(sum(len(r) for r in out))
    diag["samples_per_view"] = [int(len(r)) for r in out]
    return out, diag


def anchor_depth_to_parallax(
    images: list[np.ndarray],
    result: BackboneResult,
    cond_poses: list[Pose | None],
    cond_intrinsics: list[CameraIntrinsics | None],
    *,
    max_features: int = 3000,
    min_samples: int = 30,
    min_ratio: float = 0.05,
    max_ratio: float = 20.0,
) -> tuple[BackboneResult, dict]:
    """Rescale each view's depth so it agrees with parallax. Returns ``(result, diag)``.

    ``result`` is returned unchanged (same object) when anchoring is refused;
    ``diag["applied"]`` says which happened and why.
    """
    n = len(images)
    ratios, diag = measure_depth_ratios(
        images, result, cond_poses, cond_intrinsics, max_features=max_features
    )
    diag["applied"] = False
    if "failure" in diag:
        return result, diag

    pooled = np.concatenate([r for r in ratios if len(r)]) if any(len(r) for r in ratios) else np.zeros(0)
    if len(pooled) < min_samples:
        diag["failure"] = f"only {len(pooled)} parallax samples in the window (need >= {min_samples})"
        return result, diag

    window_ratio = float(np.median(pooled))
    diag["window_ratio"] = window_ratio
    diag["window_mad"] = float(np.median(np.abs(pooled - window_ratio)))
    if not (min_ratio <= window_ratio <= max_ratio):
        diag["failure"] = (
            f"window depth ratio {window_ratio:.3g} outside [{min_ratio}, {max_ratio}]; "
            "treating the backbone output as failed rather than rescaling it into plausibility"
        )
        return result, diag

    per_view = []
    source = []
    for i in range(n):
        if len(ratios[i]) >= min_samples:
            per_view.append(float(np.median(ratios[i])))
            source.append("view")
        else:
            per_view.append(window_ratio)
            source.append("window")
    diag["per_view_ratio"] = per_view
    diag["per_view_source"] = source

    points = np.array(result.points, dtype=np.float64, copy=True)
    depth = np.array(result.depth, dtype=np.float64, copy=True)
    for i in range(n):
        r = per_view[i]
        valid = depth[i] > 1e-6
        C = np.asarray(result.poses[i].t, dtype=np.float64)
        points[i][valid] = C + r * (points[i][valid] - C)
        depth[i][valid] *= r

    diag["applied"] = True
    logger.info(
        "depth anchor: window ratio %.3f (MAD %.3f, %d samples); per-view %s",
        window_ratio,
        diag["window_mad"],
        len(pooled),
        [round(v, 3) for v in per_view],
    )
    anchored = replace(result, points=points, depth=depth, metadata={**result.metadata, "depth_anchor": diag})
    return anchored, diag


# ---------------------------------------------------------------------------
# GPS-altitude fallback
# ---------------------------------------------------------------------------

#: Ground is taken as this percentile of depth within a view. Not the
#: maximum: a few pixels of sky or a bad depth spike would otherwise
#: define "the ground". The 60th percentile sits on the dominant surface
#: for a nadir view, where most of the frame IS ground.
_GROUND_DEPTH_PERCENTILE = 60.0

#: Corrections outside this range are refused, same reasoning as the
#: parallax anchor: a backbone off by more than this has failed, and
#: rescaling it would dress the failure up as a model.
_GPS_MIN_RATIO = 0.05
_GPS_MAX_RATIO = 20.0


def anchor_depth_to_gps_altitude(result, altitudes, *, min_views: int = 1):
    """Scale each view's depth so the camera sits its GPS height above the ground.

    Why this exists alongside the parallax anchor
    ----------------------------------------------
    ``anchor_depth_to_parallax`` is the better measurement -- it triangulates
    real features and assumes nothing about terrain shape. But it needs
    feature matching to succeed, and on real footage it often does not:
    measured on a 77-keyframe flight, **15 of 19 windows produced zero
    usable parallax samples** and kept MapAnything's raw depth, which was
    5-6x too shallow. The result was ground reconstructed across a 132.9 m
    spread of elevations and a 19 m-thick slab of duplicated surfaces.

    GPS altitude is always available. For a nadir view the camera sits
    ``alt_rel`` above the ground it is looking at, so the ratio between
    that and the reconstructed camera-to-ground distance is the scale
    correction -- no matching, no features, no failure mode that depends
    on texture.

    What it assumes, and when that breaks
    --------------------------------------
    ``alt_rel`` is height above the TAKEOFF POINT, not above the ground
    being imaged. Over sloping terrain the two differ by the terrain
    offset, and this inherits that error directly. That is a few metres on
    typical survey ground -- against the 5-6x errors it exists to correct,
    a second-order concern, but it is the reason this is the FALLBACK and
    parallax is preferred wherever it can be measured.

    Returns ``(result, diag)``; ``result`` is returned unchanged when the
    correction cannot be made or is refused.
    """
    from dataclasses import replace

    depth = np.asarray(result.depth, dtype=np.float64)
    n_views = depth.shape[0]
    diag: dict = {"applied": False, "method": "gps_altitude"}

    ratios: list[float] = []
    per_view: list[float | None] = []
    for v in range(n_views):
        alt = altitudes[v] if v < len(altitudes) else None
        d = depth[v]
        valid = d > 1e-6
        if alt is None or alt <= 1.0 or valid.sum() < 1000:
            per_view.append(None)
            continue
        # Camera-to-ground in the reconstruction, as a robust percentile
        # of this view's own depth.
        reconstructed = float(np.percentile(d[valid], _GROUND_DEPTH_PERCENTILE))
        if reconstructed <= 1e-6:
            per_view.append(None)
            continue
        r = float(alt) / reconstructed
        per_view.append(r)
        ratios.append(r)

    if len(ratios) < min_views:
        diag["failure"] = f"only {len(ratios)} views had a usable altitude and depth"
        return result, diag

    window_ratio = float(np.median(ratios))
    diag["window_ratio"] = window_ratio
    diag["views_measured"] = len(ratios)
    if not (_GPS_MIN_RATIO <= window_ratio <= _GPS_MAX_RATIO):
        diag["failure"] = (
            f"implied scale {window_ratio:.3g} outside [{_GPS_MIN_RATIO}, {_GPS_MAX_RATIO}]; "
            "treating the backbone output as failed rather than rescaling it into plausibility"
        )
        return result, diag

    points = np.array(result.points, dtype=np.float64, copy=True)
    new_depth = depth.copy()
    for v in range(n_views):
        r = per_view[v] if per_view[v] is not None else window_ratio
        valid = new_depth[v] > 1e-6
        centre = np.asarray(result.poses[v].t, dtype=np.float64)
        points[v][valid] = centre + r * (points[v][valid] - centre)
        new_depth[v][valid] *= r

    diag["applied"] = True
    logger.info(
        "gps-altitude anchor: window ratio %.3f over %d views (parallax unavailable); "
        "camera-to-ground now matches telemetry altitude",
        window_ratio,
        len(ratios),
    )
    return replace(result, points=points, depth=new_depth), diag


# ---------------------------------------------------------------------------
# Fusing the two estimates
# ---------------------------------------------------------------------------

# Floors on each estimate's relative standard error.
#
# Separate, because the two are limited by DIFFERENT physical things and
# a shared floor would make them equally trustworthy whenever both have
# plenty of samples -- which is wrong.
#
#: Parallax is limited by telemetry rotation error. A rotation error
#: `delta` over baseline `B` at depth `Z` biases triangulated depth by
#: about `Z*delta/B`: ~2% at this flight's 23 m spacing and ~1 deg of
#: gimbal accuracy. Sample count cannot reduce a systematic bias.
_MIN_SIGMA_PARALLAX = 0.02

#: The altitude anchor is limited by `alt_rel` being height above the
#: TAKEOFF POINT rather than above the ground being imaged. Terrain relief
#: of a few metres at 120 m is ~4%, and it does not average away over
#: views either -- every view in a window sees nearly the same ground.
_MIN_SIGMA_GPS = 0.04

#: Ratio disagreement beyond which the two estimates are not describing the
#: same thing. Reported rather than averaged: a 2x disagreement means one
#: of them is wrong, and the mean of a right and a wrong answer is wrong.
_MAX_DISAGREEMENT = 1.6


def _robust_sigma(samples: np.ndarray, estimate: float, floor: float) -> float:
    """Relative standard error of the median of ``samples``.

    MAD (scaled to a Gaussian sigma by 1.4826) over sqrt(n): the spread of
    the samples tells us how noisy one measurement is, and averaging n of
    them reduces that by sqrt(n). Floored, because neither estimator is
    truly better than a few percent however many samples it gathers.
    """
    if len(samples) < 2 or estimate <= 0:
        return 1.0
    mad = float(np.median(np.abs(samples - np.median(samples)))) * 1.4826
    return max(mad / max(np.sqrt(len(samples)), 1.0) / estimate, floor)


def anchor_depth_fused(
    images,
    result,
    cond_poses,
    cond_intrinsics,
    altitudes,
    *,
    max_features: int = 3000,
    min_samples: int = 30,
    min_ratio: float = 0.05,
    max_ratio: float = 20.0,
):
    """Anchor depth using parallax AND GPS altitude, weighted by their own uncertainty.

    Why fuse rather than choose
    ----------------------------
    The two estimates are INDEPENDENT. Parallax divides a pixel shift by a
    GPS *horizontal* baseline; the altitude anchor divides by a GPS
    *vertical* height. GPS is better horizontally than vertically (every
    satellite is above the receiver, so the geometry constraining height is
    the weaker one), and the two errors do not share a source. Two
    independent estimates combine to something better than either.

    A hard "parallax, else GPS" switch throws that away, and makes the
    answer discontinuous at whatever sample count the threshold sits at --
    a window with 31 samples would trust parallax completely while one with
    29 ignored it entirely.

    Inverse-variance weighting instead::

        r = (r_p/s_p^2 + r_g/s_g^2) / (1/s_p^2 + 1/s_g^2)

    which is the maximum-likelihood combination for independent Gaussian
    estimates: whichever measurement is better-determined dominates,
    automatically, with no threshold to tune.

    Disagreement is a finding, not something to average away. When the two
    differ by more than ``_MAX_DISAGREEMENT`` they are not measuring the
    same thing -- terrain sloping away from the takeoff elevation, or
    matching that locked onto the wrong surface -- so the better-determined
    one is used alone and the disagreement is reported.
    """
    from dataclasses import replace

    diag: dict = {"applied": False, "method": "fused"}

    # --- parallax ---
    r_par = s_par = None
    per_view_par: list[np.ndarray] = []
    try:
        per_view_par, par_diag = measure_depth_ratios(
            images, result, cond_poses, cond_intrinsics, max_features=max_features
        )
        pooled = np.concatenate([r for r in per_view_par if len(r)]) if any(len(r) for r in per_view_par) else np.zeros(0)
        if len(pooled) >= min_samples:
            r_par = float(np.median(pooled))
            s_par = _robust_sigma(pooled, r_par, _MIN_SIGMA_PARALLAX)
            diag["parallax"] = {"ratio": round(r_par, 4), "sigma": round(s_par, 4), "samples": int(len(pooled))}
        else:
            diag["parallax"] = {"failure": f"only {len(pooled)} samples", "samples": int(len(pooled))}
    except Exception as exc:  # noqa: BLE001 - a failed estimator is a result, not a crash
        diag["parallax"] = {"failure": f"{type(exc).__name__}: {exc}"[:120]}

    # --- gps altitude ---
    r_gps = s_gps = None
    depth = np.asarray(result.depth, dtype=np.float64)
    gps_samples = []
    for v in range(depth.shape[0]):
        alt = altitudes[v] if v < len(altitudes) else None
        d = depth[v]
        valid = d > 1e-6
        if alt is None or alt <= 1.0 or valid.sum() < 1000:
            continue
        reconstructed = float(np.percentile(d[valid], _GROUND_DEPTH_PERCENTILE))
        if reconstructed > 1e-6:
            gps_samples.append(float(alt) / reconstructed)
    if gps_samples:
        arr = np.asarray(gps_samples)
        r_gps = float(np.median(arr))
        s_gps = _robust_sigma(arr, r_gps, _MIN_SIGMA_GPS)
        diag["gps_altitude"] = {"ratio": round(r_gps, 4), "sigma": round(s_gps, 4), "views": int(len(arr))}
        # The two numbers the ratio is built from, recorded separately.
        # A ratio that differs between windows is either the altitude
        # changing (expected, benign -- the correction should track it)
        # or the backbone returning a different scale for the same scene
        # (a real defect that no downstream stage can fix). Those demand
        # opposite responses and the ratio alone cannot tell them apart,
        # so both inputs are kept.
        alts = [float(a) for a in altitudes[: depth.shape[0]] if a is not None and float(a) > 1.0]
        if alts:
            diag["altitude_m"] = round(float(np.median(alts)), 2)
            diag["backbone_ground_depth_m"] = round(float(np.median(alts)) / max(r_gps, 1e-9), 2)
    else:
        diag["gps_altitude"] = {"failure": "no view had a usable altitude and depth"}

    # --- combine ---
    if r_par is None and r_gps is None:
        diag["failure"] = "neither parallax nor GPS altitude could measure this window"
        return result, diag
    if r_par is None:
        ratio, source = r_gps, "gps_only"
    elif r_gps is None:
        ratio, source = r_par, "parallax_only"
    else:
        disagreement = max(r_par, r_gps) / max(min(r_par, r_gps), 1e-9)
        diag["disagreement"] = round(disagreement, 3)
        if disagreement > _MAX_DISAGREEMENT:
            # Not the same measurement. Trust the better-determined one and
            # say so, rather than averaging a right answer with a wrong one.
            ratio, source = (r_par, "parallax_disagreement") if s_par <= s_gps else (r_gps, "gps_disagreement")
            logger.warning(
                "depth anchor: parallax says %.3f, GPS altitude says %.3f (%.2fx apart) -- "
                "using the better-determined one (%s). Terrain sloping away from the takeoff "
                "elevation will do this, and so will matching on the wrong surface.",
                r_par,
                r_gps,
                disagreement,
                source,
            )
        else:
            w_p, w_g = 1.0 / s_par**2, 1.0 / s_gps**2
            ratio = (r_par * w_p + r_gps * w_g) / (w_p + w_g)
            source = "fused"
            diag["weight_parallax"] = round(w_p / (w_p + w_g), 3)

    diag["window_ratio"] = float(ratio)
    diag["source"] = source
    if not (min_ratio <= ratio <= max_ratio):
        diag["failure"] = f"combined ratio {ratio:.3g} outside [{min_ratio}, {max_ratio}]"
        return result, diag

    # ONE ratio for the whole window, never per-view.
    #
    # Scaling each view about its own camera centre by its own measured
    # ratio is not a similarity transform of the window -- it is a
    # non-rigid deformation of it. A ground point seen by two views whose
    # ratios differ by 10% lands at two different heights, several metres
    # apart at survey altitude, and nothing downstream can undo that: the
    # merge fits one Sim(3) per submap, so it can move a window as a whole
    # but cannot un-warp it. The result is a window that is internally
    # thick before it is merged with anything, which then looks exactly
    # like a merge failure and is not one.
    #
    # Per-view estimates are kept as a diagnostic instead. Their spread is
    # the honest uncertainty on the window ratio -- on a nadir survey at
    # near-constant altitude they are measuring one number, so a wide
    # spread means the measurement is noisy, not that the scene varies.
    per_view_ratios = [
        float(np.median(samples))
        for samples in per_view_par
        if len(samples) >= min_samples
    ]
    if len(per_view_ratios) >= 2:
        lo, hi = float(min(per_view_ratios)), float(max(per_view_ratios))
        diag["per_view_ratio_spread"] = round(hi / max(lo, 1e-9), 3)
        diag["per_view_ratio_range"] = [round(lo, 4), round(hi, 4)]

    points = np.array(result.points, dtype=np.float64, copy=True)
    new_depth = depth.copy()
    for v in range(depth.shape[0]):
        valid = new_depth[v] > 1e-6
        centre = np.asarray(result.poses[v].t, dtype=np.float64)
        points[v][valid] = centre + ratio * (points[v][valid] - centre)
        new_depth[v][valid] *= ratio

    diag["applied"] = True
    logger.info(
        "depth anchor [%s]: ratio %.3f (parallax %s, gps %s)",
        source,
        ratio,
        f"{r_par:.3f}+/-{s_par:.1%}" if r_par is not None else "n/a",
        f"{r_gps:.3f}+/-{s_gps:.1%}" if r_gps is not None else "n/a",
    )
    return replace(result, points=points, depth=new_depth), diag
