"""Cheap per-frame quality metrics used by triage's keyframe selection.

Everything here operates on grayscale and is deliberately cheap: triage
scans every candidate frame in a coarse pass over the whole video, so any
metric with real cost (dense optical flow, learned quality models, ...)
would dominate ingest runtime. Sparse Lucas-Kanade tracking of a few
hundred corners is the most expensive thing we do, and even that only runs
once per scanned frame pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from drishti3d.types import CameraIntrinsics, Frame, FrameMetrics

# Lucas-Kanade / feature-tracking tuning. Kept module-level so both parallax
# functions (plain and rotation-compensated) track features identically,
# which matters for comparing their outputs meaningfully.
_MAX_CORNERS = 200
_QUALITY_LEVEL = 0.01
_MIN_DISTANCE = 8
_FB_ERROR_THRESHOLD_PX = 1.0
_MIN_SURVIVING_TRACKS = 8

# ---------------------------------------------------------------------------
# "Useful baseline" discriminator tuning (see ParallaxEstimate.useful_baseline_px).
#
# Homography-residual-only parallax has a fundamental blind spot: a pure
# camera rotation *and* a pure translation over a planar/near-planar scene
# (the dominant nadir-drone-over-terrain case) are BOTH exactly described by
# a single global homography, so both collapse the residual toward zero.
# Two frames alone can't be told apart *from the residual signature* --
# but with camera intrinsics, they mathematically can: a pure rotation's
# homography factors as ``H = s * K @ R @ K^-1`` for some rotation R and
# scale s (``H`` is only defined up to scale), so ``M = K^-1 @ H @ K`` is,
# up to that same scale, a rotation matrix -- orthonormal, det +1. A
# translation over a plane (``H = R + t @ n^T / d``, the general planar-
# homography decomposition) is a rank-1 perturbation of a rotation and is
# *not* orthonormal after the same normalization, whether or not the raw
# pixel motion it produces happens to be small. Normalizing ``M`` by its
# *middle* singular value (rather than, say, the largest) is what makes
# this scale-invariant: for an exact scaled rotation all three singular
# values are equal, so dividing by any one of them yields an exact
# orthonormal matrix; for a genuine planar-translation ``H`` the three
# singular values differ, and dividing by the middle one is the standard
# homography-decomposition normalization (see Faugeras & Lustman, 1988)
# that keeps the result closest to unit scale without an arbitrary choice
# of which singular value to trust.
#
# ``_is_pure_rotation`` (used by ``useful_baseline_px`` whenever intrinsics
# were supplied) is this test: ``||M_norm @ M_norm.T - I||_F`` near zero
# (orthonormal) *and* ``det(M_norm)`` near +1 (a proper rotation, not a
# reflection) together mean "this pair is explained by camera rotation
# alone -- zero reconstruction-useful baseline, no matter how large the
# raw pixel motion was". Large deviation means the homography needed real
# scene-plane structure to fit, i.e. a genuine (if possibly small) baseline.
# Calibrated empirically (see tests/test_triage.py's rotation-vs-translation
# tests): a real ``K R K^-1`` pure rotation, run through actual LK tracking
# + RANSAC homography fitting (not just the noiseless closed-form matrix),
# lands its deviation in the ~1e-4-1e-3 range across a wide span of angles
# and random textures; even a barely-perceptible plane-translation (~1px
# raw motion) lands at ~0.007 and grows from there. This threshold sits
# roughly midway between those two clusters on a log scale, comfortably
# inside the gap rather than at either edge.
_ROTATION_DEVIATION_THRESHOLD = 0.005
_ROTATION_DET_TOLERANCE = 0.15

# Absolute noise floor: below this, raw pixel "motion" is almost certainly
# tracking jitter, not real camera motion, regardless of what any
# homography test says about it -- avoids chasing sub-pixel LK noise.
_NOISE_FLOOR_PX = 0.3

# ---------------------------------------------------------------------------
# Fallback heuristic (used only when no intrinsics are available -- see
# ``estimate_parallax_detailed``'s ``intrinsics`` parameter and
# ``ParallaxEstimate.used_intrinsics_test``). Vision alone (two frames,
# sparse correspondences, no camera model) cannot resolve the rotation/
# plane-translation ambiguity the principled test above resolves, so this
# biases toward not starving the accumulator instead: any frame pair with a
# large enough raw displacement is treated as carrying a usable baseline
# unless the homography fit is a *bit-perfect*, near-zero-residual,
# near-all-inlier fit -- the signature of genuinely zero relative-depth
# information (either pure rotation, or translation over a scene so flat/
# distant that no depth cue survives).
_MIN_SIGNIFICANT_RAW_PX = 2.0
_RESIDUAL_FLOOR_PX = 0.05
_LOW_INLIER_RATIO = 0.5


def _homography_rotation_deviation(H: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """``(deviation, det)`` of ``M = K^-1 @ H @ K`` normalized by its middle singular value.

    ``deviation`` is ``||M_norm @ M_norm.T - I||_F`` (0 for an exact
    rotation, growing with how far ``H`` is from one); ``det`` is
    ``M_norm``'s determinant (+1 for a proper rotation after sign
    correction). See the module-level comment above
    ``_ROTATION_DEVIATION_THRESHOLD`` for the full derivation.
    """
    K_inv = np.linalg.inv(K)
    M = K_inv @ H @ K
    singular_values = np.linalg.svd(M, compute_uv=False)
    sigma_mid = singular_values[1]
    if sigma_mid < 1e-12:
        return float("inf"), 0.0

    M_norm = M / sigma_mid
    det = float(np.linalg.det(M_norm))
    if det < 0:
        M_norm = -M_norm
        det = -det
    deviation = float(np.linalg.norm(M_norm @ M_norm.T - np.eye(3)))
    return deviation, det


def blur_score(gray: np.ndarray) -> float:
    """Sharpness proxy: variance-of-Laplacian averaged with Tenengrad gradient energy.

    Variance-of-Laplacian (the classic "blur detection" measure) is fast
    but can be fooled by images that are sharp but low-contrast/textureless
    in a way that happens to still have some high-frequency noise. Sobel
    gradient energy (Tenengrad) is a complementary measure of the same
    underlying thing -- edge strength -- computed differently, so averaging
    the two is cheap insurance against either metric's individual blind
    spots. Both metrics grow with genuine edge content, so higher is
    sharper for the combination too.
    """
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))

    return 0.5 * (lap_var + tenengrad)


def exposure_score(gray: np.ndarray) -> float:
    """0..1 exposure quality, 1 = well exposed, penalized by clipped tonal extremes.

    A frame with a large fraction of pixels pinned at (near-)black or
    (near-)white has lost detail there irrecoverably (clipping, not just
    "dark" or "bright"), which hurts feature matching in that region. We
    measure exactly that: the fraction of pixels in the extreme histogram
    bins, subtracted from 1.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = float(gray.size)
    if total <= 0:
        return 0.0
    shadow_clip = float(hist[:2].sum()) / total
    highlight_clip = float(hist[254:].sum()) / total
    return float(np.clip(1.0 - (shadow_clip + highlight_clip), 0.0, 1.0))


def mean_luma(gray: np.ndarray) -> float:
    """Mean grayscale intensity (0..255)."""
    return float(np.mean(gray))


def _tracked_displacements(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Sparse LK tracking with a forward-backward consistency check.

    Returns ``(src_points, dst_points)`` for tracks that survive (Nx2
    each), or ``None`` if too few tracks survive to trust a displacement
    estimate. Forward-backward error (track prev->curr->prev and compare to
    the start) is a cheap way to reject tracks that locked onto the wrong
    feature or drifted off a moving/occluded region, without needing a
    second frame pair or a learned model.
    """
    pts_prev = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=_MAX_CORNERS, qualityLevel=_QUALITY_LEVEL, minDistance=_MIN_DISTANCE
    )
    if pts_prev is None or len(pts_prev) < _MIN_SURVIVING_TRACKS:
        return None

    pts_curr, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
    if pts_curr is None:
        return None
    pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, pts_curr, None)
    if pts_back is None:
        return None

    status_fwd = status_fwd.reshape(-1).astype(bool)
    status_bwd = status_bwd.reshape(-1).astype(bool)
    fb_error = np.linalg.norm((pts_prev - pts_back).reshape(-1, 2), axis=1)
    good = status_fwd & status_bwd & (fb_error < _FB_ERROR_THRESHOLD_PX)

    if good.sum() < _MIN_SURVIVING_TRACKS:
        return None

    src = pts_prev.reshape(-1, 2)[good]
    dst = pts_curr.reshape(-1, 2)[good]
    return src, dst


def estimate_parallax(
    prev_gray: np.ndarray, curr_gray: np.ndarray, intrinsics: CameraIntrinsics | None = None
) -> float:
    """Median raw pixel displacement of sparse tracked features between two frames.

    This is the *uncompensated* parallax: it mixes true depth-induced
    disparity with apparent motion from pure camera rotation (panning
    produces large pixel displacement with zero depth information). Use
    ``estimate_rotation_compensated_parallax`` when the goal is judging
    reconstruction-useful baseline; this function is kept as the simpler,
    cheaper building block (and for callers that specifically want raw
    motion magnitude, e.g. a "how much did anything move" sanity check).
    ``intrinsics`` is accepted for API symmetry with the rotation-
    compensated variant (future undistortion) but unused here.
    """
    del intrinsics
    tracked = _tracked_displacements(prev_gray, curr_gray)
    if tracked is None:
        return 0.0
    src, dst = tracked
    magnitudes = np.linalg.norm(dst - src, axis=1)
    return float(np.median(magnitudes))


@dataclass
class ParallaxEstimate:
    """Structured result of the vision-only parallax fallback (no telemetry).

    ``raw_px``: median raw tracked-point displacement (what
        ``estimate_parallax`` returns) -- mixes true depth parallax with
        apparent motion from pure rotation.
    ``residual_px``: median displacement left over after removing the
        best-fit homography (what ``estimate_rotation_compensated_parallax``
        returns) -- isolates depth-dependent motion, *except* that it also
        collapses toward zero for translation over a planar/near-planar
        scene, not just for pure rotation (see module-level comment above
        ``_MIN_SIGNIFICANT_RAW_PX``).
    ``homography_inlier_ratio``: fraction of tracked points RANSAC accepted
        as consistent with the fitted homography. Low means the scene isn't
        well explained by a single plane/rotation -- i.e. probably has real
        depth variation.
    ``n_tracks``: number of surviving point tracks the estimate is based on
        (0 if tracking failed outright), so a caller can judge confidence.
    ``rotation_deviation``: ``||M_norm @ M_norm.T - I||_F`` from the
        principled ``K^-1 H K`` test (see ``_homography_rotation_deviation``)
        -- ``None`` when that test didn't run (no intrinsics given, or the
        homography fit was degenerate).
    ``used_intrinsics_test``: whether ``useful_baseline_px`` used the
        principled intrinsics-based rotation test (``True``) or fell back
        to the raw/residual/inlier heuristic (``False``, e.g. no
        intrinsics were supplied) -- see that property's docstring.
    """

    raw_px: float
    residual_px: float
    homography_inlier_ratio: float
    n_tracks: int
    rotation_deviation: float | None = None
    used_intrinsics_test: bool = False

    @property
    def useful_baseline_px(self) -> float:
        """Best-effort "is this pair worth spacing a keyframe over" signal, in pixels.

        When intrinsics were available (``used_intrinsics_test``), this is
        the principled discriminator: a homography whose ``K^-1 H K`` is
        (up to scale) a rotation matrix carries zero reconstruction-useful
        baseline no matter how large the raw pixel motion was (pure
        rotation); anything else -- including small-raw-motion translation
        over a plane, which the old heuristic below would zero out -- keeps
        ``raw_px``. See the module-level comment above
        ``_ROTATION_DEVIATION_THRESHOLD`` for the derivation.

        Without intrinsics, falls back to the raw/residual/inlier heuristic
        (see the comment above ``_MIN_SIGNIFICANT_RAW_PX``): raw
        displacement below a significance floor is treated as noise:
        above it, the pair counts as useful unless the homography fit is a
        suspiciously clean, near-all-inlier, near-zero-residual match --
        the one signature pure rotation and flat-plane translation share,
        and that a fallback with no camera model to disambiguate them
        further can't safely call "useful".
        """
        if self.raw_px < _NOISE_FLOOR_PX:
            return 0.0

        if self.used_intrinsics_test:
            if self.rotation_deviation is None:
                # Homography fit was degenerate (e.g. near-collinear
                # points) even though intrinsics were available -- can't
                # confirm pure rotation, so don't discard real motion.
                return self.raw_px
            is_pure_rotation = (
                self.rotation_deviation < _ROTATION_DEVIATION_THRESHOLD
            )
            return 0.0 if is_pure_rotation else self.raw_px

        if self.raw_px < _MIN_SIGNIFICANT_RAW_PX:
            return 0.0
        if self.residual_px > _RESIDUAL_FLOOR_PX or self.homography_inlier_ratio < _LOW_INLIER_RATIO:
            return self.raw_px
        return 0.0


def estimate_parallax_detailed(
    prev_gray: np.ndarray, curr_gray: np.ndarray, intrinsics: CameraIntrinsics | None = None
) -> ParallaxEstimate:
    """Compute raw displacement, homography residual/inlier ratio, and (with intrinsics) the rotation test together.

    Superset of ``estimate_rotation_compensated_parallax``: that function
    only ever returns the homography residual, which is blind to the
    planar-scene-translation case described above. This returns everything
    ``ParallaxEstimate.useful_baseline_px`` needs: with ``intrinsics``, the
    principled ``K^-1 H K`` rotation-vs-plane-translation test (see
    ``_homography_rotation_deviation``); without them, the raw/residual/
    inlier-ratio signature the fallback heuristic uses instead.
    """
    tracked = _tracked_displacements(prev_gray, curr_gray)
    if tracked is None:
        return ParallaxEstimate(raw_px=0.0, residual_px=0.0, homography_inlier_ratio=0.0, n_tracks=0)

    src, dst = tracked
    n_tracks = len(src)
    raw_px = float(np.median(np.linalg.norm(dst - src, axis=1)))

    if n_tracks < 4:
        # Can't fit a homography (needs >= 4 correspondences); nothing to
        # say about depth-dependence, so report zero residual/inliers
        # rather than guessing. Unreachable in practice: _tracked_displacements
        # already requires >= _MIN_SURVIVING_TRACKS (8) survivors.
        return ParallaxEstimate(raw_px=raw_px, residual_px=0.0, homography_inlier_ratio=0.0, n_tracks=n_tracks)

    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None:
        # Degenerate configuration (e.g. all points collinear): fall back to
        # raw displacement rather than claiming zero parallax/zero inliers.
        return ParallaxEstimate(
            raw_px=raw_px,
            residual_px=raw_px,
            homography_inlier_ratio=0.0,
            n_tracks=n_tracks,
            used_intrinsics_test=intrinsics is not None,
        )

    inlier_ratio = float(inlier_mask.reshape(-1).sum()) / n_tracks if inlier_mask is not None else 0.0

    src_h = np.hstack([src, np.ones((n_tracks, 1))])
    warped = (homography @ src_h.T).T
    warped_xy = warped[:, :2] / warped[:, 2:3]
    residual_px = float(np.median(np.linalg.norm(dst - warped_xy, axis=1)))

    rotation_deviation: float | None = None
    if intrinsics is not None:
        deviation, det = _homography_rotation_deviation(homography, intrinsics.K())
        if np.isfinite(deviation) and abs(det - 1.0) < _ROTATION_DET_TOLERANCE:
            rotation_deviation = deviation
        elif np.isfinite(deviation):
            # A finite-but-improper (det far from +1) normalization is not
            # a rotation candidate at all -- report a deviation far above
            # the threshold rather than None, so useful_baseline_px still
            # (correctly) calls this "not pure rotation" instead of
            # spuriously falling back to "can't tell, keep the motion".
            rotation_deviation = max(deviation, _ROTATION_DEVIATION_THRESHOLD * 10)

    return ParallaxEstimate(
        raw_px=raw_px,
        residual_px=residual_px,
        homography_inlier_ratio=inlier_ratio,
        n_tracks=n_tracks,
        rotation_deviation=rotation_deviation,
        used_intrinsics_test=intrinsics is not None,
    )


def estimate_rotation_compensated_parallax(
    prev_gray: np.ndarray, curr_gray: np.ndarray, intrinsics: CameraIntrinsics | None = None
) -> float:
    """Median residual pixel displacement after removing the best-fit homography.

    Why: a pure camera rotation (or a translation with all scene points at
    effectively the same depth) is fully explained by a single homography
    between the two frames -- every tracked point's motion is consistent
    with one global transform, and none of it comes from parallax between
    points at different depths. That kind of motion carries zero
    information for multi-view triangulation, even though the raw pixel
    displacement (what ``estimate_parallax`` returns) can be large during a
    fast pan. By fitting a homography with RANSAC and measuring what's left
    over (the residual each point's actual match deviates from where the
    homography predicts), we isolate the depth-dependent motion -- the
    actual stereo baseline -- which is what determines whether two frames
    are useful together for reconstruction.

    Kept for API compatibility and as the cheaper building block other code
    may still want; it shares the exact same blind spot for planar-scene
    translation described on ``estimate_parallax_detailed`` /
    ``ParallaxEstimate.useful_baseline_px``, which is why
    ``triage.selector.select_keyframes`` no longer uses this function alone
    to drive keyframe spacing -- see ``estimate_parallax_detailed`` for the
    structured, hybrid-aware replacement.

    ``intrinsics`` is accepted for API symmetry / future undistortion but
    not required: the homography fit works directly in pixel space.
    """
    return estimate_parallax_detailed(prev_gray, curr_gray, intrinsics).residual_px


def compute_frame_metrics(
    frame: Frame, prev_gray: np.ndarray | None = None, intrinsics: CameraIntrinsics | None = None
) -> FrameMetrics:
    """Convenience wrapper computing all per-frame metrics from a decoded ``Frame``.

    ``prev_gray`` is the grayscale of whatever frame parallax should be
    measured against (typically the previous scanned frame, or the last
    accepted keyframe); parallax is left at 0.0 when it's not given, since
    there's nothing to compare against yet (e.g. the very first frame).
    """
    if frame.image is None:
        raise ValueError("compute_frame_metrics requires a decoded frame.image")

    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    parallax = 0.0
    if prev_gray is not None:
        parallax = estimate_rotation_compensated_parallax(prev_gray, gray, intrinsics)

    return FrameMetrics(
        index=frame.index,
        timestamp=frame.timestamp,
        blur_score=blur_score(gray),
        exposure_score=exposure_score(gray),
        mean_luma=mean_luma(gray),
        estimated_parallax=parallax,
    )
