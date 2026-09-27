"""Camera poses from the images alone, for when the GPS/attitude log cannot seed them.

The pose prior normally triangulates feature tracks through cameras placed
at their GPS fixes and gimbal attitudes, then refines. That works when the
log's error is small against the camera spacing -- a nadir survey 100 m
up with 20-80 m between keyframes. It fails when it is not: the
Front_View_Light vineyard clip was flown ~1 m above the ground at ~1 m/s
with consumer GPS (1-3 m error) and ~3 m between keyframes, and all 721
tracks reprojected more than 60 px through the logged cameras.

This module solves the cameras the way image-only structure from motion
does, and uses the log only where it is still sound:

1. **Initial pair.** The keyframe pair with the most shared tracks and
   enough parallax gives a relative pose from the essential matrix; its
   inliers are triangulated.
2. **Incremental registration.** Each next keyframe is the unregistered one
   seeing the most triangulated points, placed by PnP with RANSAC; new
   tracks are triangulated as soon as two posed keyframes see them, and
   the whole set is bundle-adjusted every few additions.
3. **Placement.** The result is levelled and turned with the log's
   attitude (gimbal pitch/roll and heading, averaged over every camera),
   then scaled and shifted onto the GPS track by least squares. A single
   similarity over the whole track averages out GPS noise that no
   individual camera could.

The caller then triangulates and bundle-adjusts as usual, with the GPS as
a weak datum only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from drishti3d.types import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)


@dataclass
class ImageOnlyResult:
    """Poses in the seeds' world frame (None where a keyframe could not be registered) and diagnostics."""

    poses: list[Pose | None]
    diagnostics: dict = field(default_factory=dict)

    @property
    def registered(self) -> list[int]:
        return [i for i, p in enumerate(self.poses) if p is not None]


def _normalized(uv: np.ndarray, K: CameraIntrinsics) -> np.ndarray:
    """Pixels -> undistorted normalized image coordinates (x/z, y/z)."""
    import cv2

    Km = np.array([[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]])
    dist = None if K.dist_coeffs is None else np.asarray(K.dist_coeffs, dtype=np.float64)
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, Km, dist).reshape(-1, 2)


def _triangulate(P: list[np.ndarray], x: list[np.ndarray]) -> np.ndarray:
    """Linear triangulation from normalized observations; P are 3x4 [R|t] (camera from world)."""
    A = []
    for Pi, xi in zip(P, x, strict=True):
        A.append(xi[0] * Pi[2] - Pi[0])
        A.append(xi[1] * Pi[2] - Pi[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)


def _chordal_mean(rotations: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def image_only_poses(
    trackset,
    intrinsics: list[CameraIntrinsics],
    seeds: list[Pose | None],
    *,
    reproj_px: float = 4.0,
    min_inliers: int = 25,
    min_parallax_deg: float = 0.8,
    max_rotation_disagreement_deg: float = 5.0,
    max_direction_disagreement_deg: float = 45.0,
    ba_every: int = 3,
) -> ImageOnlyResult | None:
    """Incremental structure from motion over ``trackset``, placed onto ``seeds`` (GPS centres + log attitude).

    Returns None when no pair can initialise or fewer than three keyframes
    register -- the caller keeps its own failure then.
    """
    import cv2

    from drishti3d.geometry import bundle

    n = len(intrinsics)
    tracks = [t for t in trackset.tracks if len(t.observations) >= 2]
    if len(tracks) < 50 or n < 3:
        return None
    # Normalized observations per track: {frame: xy}.
    obs: list[dict[int, np.ndarray]] = []
    pix: list[dict[int, np.ndarray]] = []
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for ti, t in enumerate(tracks):
        o, pp = {}, {}
        for f, _k, uv in t.observations:
            pp[f] = np.asarray(uv, dtype=np.float64)
            by_frame.setdefault(f, []).append((ti, pp[f]))
        obs.append(o)
        pix.append(pp)
    for f, items in by_frame.items():
        xy = _normalized(np.array([u for _, u in items]), intrinsics[f])
        for (ti, _), x in zip(items, xy, strict=True):
            obs[ti][f] = x
    thr = reproj_px / float(np.median([k.fx for k in intrinsics]))  # normalized units

    # 1) initial pair: many shared tracks, enough parallax.
    shared: dict[tuple[int, int], list[int]] = {}
    for ti, o in enumerate(obs):
        fs = sorted(o)
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                shared.setdefault((fs[a], fs[b]), []).append(ti)
    # Every pair with enough shared tracks, not only the busiest (those are
    # neighbours, with the least parallax under forward motion).
    candidates = sorted(((k, v) for k, v in shared.items() if len(v) >= 2 * min_inliers), key=lambda kv: -len(kv[1]))[:60]
    best = None
    rejected = {"rotation": 0, "direction": 0, "parallax": 0}
    for (i, j), tis in candidates:
        xi = np.array([obs[t][i] for t in tis])
        xj = np.array([obs[t][j] for t in tis])
        E, mask = cv2.findEssentialMat(xi, xj, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=thr)
        if E is None or E.shape != (3, 3):
            continue
        _, R, tv, mask2 = cv2.recoverPose(E, xi, xj, np.eye(3), mask=mask)
        inl = mask2.ravel().astype(bool)
        if inl.sum() < min_inliers:
            continue
        # The log is still sound for how the camera turned and roughly which
        # way it went: an essential-matrix solution that disagrees is one of
        # the degenerate ones forward motion over a deep scene produces
        # (Front_View_Light: "moved straight down", rotated 8 deg).
        si, sj = seeds[i], seeds[j]
        if si is not None and sj is not None:
            R_rel = np.asarray(sj.R).T @ np.asarray(si.R)  # cam_j from cam_i, per the log
            c = np.clip((np.trace(R.T @ R_rel) - 1.0) / 2.0, -1.0, 1.0)
            if np.degrees(np.arccos(c)) > max_rotation_disagreement_deg:
                rejected["rotation"] += 1
                continue
            base = np.asarray(si.R).T @ (np.asarray(sj.t, dtype=np.float64) - np.asarray(si.t, dtype=np.float64))
            if np.linalg.norm(base) > 1.0:
                d_img = -R.T @ tv.ravel()
                cosd = float(d_img @ base / (np.linalg.norm(d_img) * np.linalg.norm(base)))
                if np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0))) > max_direction_disagreement_deg:
                    rejected["direction"] += 1
                    continue
        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = np.hstack([R, tv.reshape(3, 1)])
        X = cv2.triangulatePoints(P0, P1, xi[inl].T, xj[inl].T)
        X = (X[:3] / X[3]).T
        c1 = -R.T @ tv.ravel()
        v0 = X / np.linalg.norm(X, axis=1, keepdims=True)
        v1 = X - c1
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip((v0 * v1).sum(1), -1, 1)))
        parallax = float(np.median(ang))
        if parallax < min_parallax_deg:
            rejected["parallax"] += 1
            continue
        score = inl.sum() * min(parallax, 8.0)
        if best is None or score > best[0]:
            best = (score, i, j, R, tv.ravel(), [t for t, k in zip(tis, inl, strict=True) if k], parallax)
    if best is None:
        logger.info("image-only cameras: no keyframe pair can start the solve (rejected: %s)", rejected)
        return None
    _, i0, j0, R01, t01, init_tracks, parallax = best

    # Camera-from-world rotations/translations of registered frames (world = frame i0).
    Rcw: dict[int, np.ndarray] = {i0: np.eye(3), j0: R01}
    tcw: dict[int, np.ndarray] = {i0: np.zeros(3), j0: t01}
    points: dict[int, np.ndarray] = {}

    def P_of(f):
        return np.hstack([Rcw[f], tcw[f].reshape(3, 1)])

    def try_triangulate(ti):
        fs = [f for f in obs[ti] if f in Rcw]
        if len(fs) < 2:
            return
        X = _triangulate([P_of(f) for f in fs], [obs[ti][f] for f in fs])
        if not np.isfinite(X).all():
            return
        centres = [-Rcw[f].T @ tcw[f] for f in fs]
        rays = [(X - c) / max(np.linalg.norm(X - c), 1e-12) for c in centres]
        ang = max(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))) for k, a in enumerate(rays) for b in rays[k + 1:])
        if ang < 1.0:
            return
        for f in fs:
            Xc = Rcw[f] @ X + tcw[f]
            if Xc[2] <= 0:
                return
            if np.linalg.norm(Xc[:2] / Xc[2] - obs[ti][f]) > 2 * thr:
                return
        points[ti] = X

    for ti in init_tracks:
        try_triangulate(ti)

    def run_ba():
        regs = sorted(Rcw)
        pt_ids = sorted(points)
        if len(pt_ids) < 20:
            return
        cam_index = {f: k for k, f in enumerate(regs)}
        pt_index = {t: k for k, t in enumerate(pt_ids)}
        oc, op, ouv = [], [], []
        for t in pt_ids:
            for f, uv in pix[t].items():
                if f in cam_index:
                    oc.append(cam_index[f])
                    op.append(pt_index[t])
                    ouv.append(uv)
        cams = [Pose(R=Rcw[f].T.copy(), t=(-Rcw[f].T @ tcw[f]).copy()) for f in regs]
        problem = bundle.BAProblem(
            cameras=cams, intrinsics=[intrinsics[f] for f in regs],
            points=np.array([points[t] for t in pt_ids]), obs_camera_idx=np.array(oc), obs_point_idx=np.array(op),
            obs_uv=np.array(ouv), camera_priors=[], fixed_camera_indices={cam_index[i0], cam_index[j0]},
        )
        res = bundle.bundle_adjust(problem, bundle.BAConfig(max_iterations=30, warm_start_linear_nfev=5))
        for f, k in cam_index.items():
            Rwc = np.asarray(res.poses[k].R)
            Rcw[f] = Rwc.T
            tcw[f] = -Rwc.T @ np.asarray(res.poses[k].t)
        for t, k in pt_index.items():
            points[t] = np.asarray(res.points[k])
        # Drop points the refined cameras no longer explain.
        for t in list(points):
            for f in obs[t]:
                if f in Rcw:
                    Xc = Rcw[f] @ points[t] + tcw[f]
                    if Xc[2] <= 0 or np.linalg.norm(Xc[:2] / Xc[2] - obs[t][f]) > 2 * thr:
                        del points[t]
                        break

    added = 0
    while True:
        best_f, best_c = None, []
        for f in range(n):
            if f in Rcw:
                continue
            c = [t for t, _ in by_frame.get(f, []) if t in points]
            if len(c) > len(best_c):
                best_f, best_c = f, c
        if best_f is None or len(best_c) < min_inliers:
            break
        X = np.array([points[t] for t in best_c])
        x = np.array([obs[t][best_f] for t in best_c])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            X, x, np.eye(3), None, reprojectionError=thr, iterationsCount=2000, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inl is None or len(inl) < min_inliers:
            break
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], x[inl], np.eye(3), None, rvec, tvec)
        Rcw[best_f] = cv2.Rodrigues(rvec)[0]
        tcw[best_f] = tvec.ravel()
        for t, _ in by_frame.get(best_f, []):
            if t not in points:
                try_triangulate(t)
        added += 1
        if added % ba_every == 0:
            run_ba()
    run_ba()
    regs = sorted(Rcw)
    if len(regs) < 3:
        logger.info("image-only cameras: only %d keyframes registered", len(regs))
        return None

    # 3) Place onto the log: attitude for level and heading, GPS for scale and position.
    pairs = [(f, seeds[f]) for f in regs if seeds[f] is not None]
    if len(pairs) < 3:
        return None
    R_wsfm = _chordal_mean([np.asarray(s.R) @ Rcw[f] for f, s in pairs])  # world_from_sfm
    c_sfm = np.array([-Rcw[f].T @ tcw[f] for f, _ in pairs])
    g = np.array([np.asarray(s.t, dtype=np.float64) for _, s in pairs])

    def fit(R):
        a = (R @ c_sfm.T).T
        am, gm = a.mean(0), g.mean(0)
        s = float(((a - am) * (g - gm)).sum() / max(((a - am) ** 2).sum(), 1e-12))
        return s, gm - s * am, np.linalg.norm(s * a + (gm - s * am) - g, axis=1)

    # Heading from a compass is the least reliable part of the log: refine a
    # yaw offset about the vertical against the GPS track.
    best_fit = None
    for yaw in np.radians(np.arange(-45.0, 45.01, 1.0)):
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        s, t, r = fit(Rz @ R_wsfm)
        if s > 0 and (best_fit is None or np.median(r) < np.median(best_fit[3])):
            best_fit = (Rz @ R_wsfm, s, t, r, np.degrees(yaw))
    if best_fit is None:
        return None
    R_final, s, t, resid, yaw_deg = best_fit
    poses: list[Pose | None] = [None] * n
    for f in regs:
        Rwc = R_final @ Rcw[f].T
        c = s * (R_final @ (-Rcw[f].T @ tcw[f])) + t
        poses[f] = Pose(R=Rwc, t=c)
    diag = {
        "registered": len(regs),
        "keyframes": n,
        "initial_pair": [int(i0), int(j0)],
        "initial_parallax_deg": round(parallax, 2),
        "points": len(points),
        "gps_fit_median_m": round(float(np.median(resid)), 3),
        "gps_fit_p90_m": round(float(np.percentile(resid, 90)), 3),
        "scale": round(s, 4),
        "heading_correction_deg": round(float(yaw_deg), 1),
    }
    logger.info(
        "image-only cameras: %d/%d keyframes registered from pair %s (parallax %.1f deg), %d points; "
        "placed on GPS with median residual %.2f m, heading correction %+.0f deg",
        len(regs), n, diag["initial_pair"], parallax, len(points), diag["gps_fit_median_m"], yaw_deg,
    )
    return ImageOnlyResult(poses=poses, diagnostics=diag)
