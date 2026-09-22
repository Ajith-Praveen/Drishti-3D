"""2D feature detection, matching, and geometric verification.

This is the missing first link in the chain the module docstring of
``geometry.bundle`` assumes already exists: ``BAProblem.obs_camera_idx`` /
``obs_point_idx`` / ``obs_uv`` are *2D pixel observations of 3D points
across cameras*, and nothing upstream of this module ever produced that --
``geometry.backbone`` outputs dense per-pixel points with no explicit
cross-view correspondence, and ``geometry.submap.merge_submaps`` only ever
sees the already-flattened point cloud. This module is step one of
building that correspondence structure from scratch, classically:

    detect + describe (this file) -> match (this file) -> geometrically
    verify (this file) -> chain into multi-view tracks (``tracks.py``) ->
    triangulate (``triangulate.py``) -> ``bundle.BAProblem``.

Why classical (SIFT/AKAZE/ORB) and not a learned matcher
----------------------------------------------------------
Learned detectors/matchers (ALIKED, SuperPoint, LightGlue, ...) generally
out-match classical features on hard cases (repetitive texture, large
viewpoint change), but they need torch and (usually) a GPU and shipped
model weights -- exactly the dependency set this project's installer
cannot assume is available on-site (see ``geometry.backbone``'s module
docstring for the same constraint on the reconstruction backbone). OpenCV
ships SIFT/AKAZE/ORB with zero extra weights and runs entirely on CPU, so
that is what actually has to run here. The ``Matcher`` protocol at the
bottom of this file is the extension point: a learned matcher can be
dropped into ``MatchingStage`` (see ``pipeline.stages``) later without
touching ``tracks.py`` or anything downstream, as long as it produces a
``Matches`` object.

Why SIFT is the default detector
----------------------------------
Aerial/drone imagery is exactly the case where **scale varies a lot**
between views of the same ground point: altitude changes, oblique angles,
and a single-pass flight's own along-track motion all change how large a
given patch of ground appears from frame to frame. SIFT's whole design is
built around being scale- and rotation-invariant via an explicit
scale-space search, which is precisely the invariance this problem needs;
its (comparatively) higher per-frame cost is affordable here because
matching runs once per selected keyframe pair, not once per scanned video
frame (contrast ``triage.metrics``, which deliberately uses only sparse
LK corner tracking because it runs over *every* scanned frame).

Why ORB is the fast fallback
-------------------------------
ORB (an oriented, rotation-aware BRIEF variant with a FAST detector) is
roughly an order of magnitude cheaper to compute and match (binary
descriptors, Hamming distance, no floating-point KD-tree search) than
SIFT, at the cost of materially weaker scale invariance and lower match
quality on repetitive/low-texture aerial scenes. Use it when a flight has
many keyframes and wall-clock time matters more than match density/
robustness (e.g. interactive triage previews), not as the default for the
matching stage that actually feeds bundle adjustment.

Why AKAZE is the middle ground
----------------------------------
AKAZE builds its scale space on a nonlinear (edge-preserving) diffusion
pyramid rather than SIFT's Gaussian pyramid, so it keeps meaningful scale
invariance (unlike ORB) while producing a binary descriptor (unlike SIFT's
float descriptor), which keeps matching cheap (Hamming distance, no FLANN
KD-tree) while still coping reasonably with the altitude-driven scale
changes SIFT was chosen for. It sits between the other two on both cost
and quality axes -- a reasonable choice when SIFT's matching cost (FLANN
over float descriptors) is the actual bottleneck but ORB's scale
brittleness is unacceptable. Not every OpenCV build ships
``cv2.AKAZE_create`` (it is occasionally absent from otherwise-complete
``opencv-python`` wheels); ``detect_and_describe`` raises a clear
``RuntimeError`` naming the missing constructor rather than silently
falling back, since a silent fallback to a different method would change
match quality/behaviour without the caller asking for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Methods whose OpenCV descriptor is binary (Hamming-distance matchable via
# BFMatcher(NORM_HAMMING)) rather than float (L2-distance matchable via
# FLANN's KD-tree index). Drives both the matcher choice in
# ``_make_matcher`` and the dtype coercion in ``match_features``.
_BINARY_DESCRIPTOR_METHODS = frozenset({"orb", "akaze"})

# FLANN KD-tree index type constant (cv2.flann.FLANN_INDEX_KDTREE is not
# always exposed as a top-level cv2 attribute across opencv-python builds;
# 1 is FLANN's own stable enum value for it).
_FLANN_INDEX_KDTREE = 1


@dataclass
class Features:
    """Detected keypoints + descriptors for one image.

    keypoints:
        ``(N, 2)`` float64 pixel coordinates ``(x, y)`` -- OpenCV's
        ``cv2.KeyPoint.pt`` convention, matching ``CameraIntrinsics``'
        pixel convention used everywhere else in this codebase.
    descriptors:
        ``(N, D)`` array (float32 for SIFT, uint8 for ORB/AKAZE), or
        ``None`` if nothing was detected.
    scores:
        ``(N,)`` float64 per-keypoint detector response/strength
        (``cv2.KeyPoint.response``), for anyone who wants to rank or
        subsample keypoints downstream.
    method:
        The detector name that produced this (``"sift"``, ``"akaze"``, or
        ``"orb"``) -- ``match_features`` uses it to decide binary vs. float
        matching, and refuses to match two ``Features`` with different
        methods (their descriptors are not comparable).
    """

    keypoints: np.ndarray
    descriptors: np.ndarray | None
    scores: np.ndarray
    method: str

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64).reshape(-1, 2)
        if self.descriptors is not None:
            self.descriptors = np.asarray(self.descriptors)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)

    def __len__(self) -> int:
        return self.keypoints.shape[0]


@dataclass
class Matches:
    """Correspondences between two ``Features`` sets, by index.

    query_idx / train_idx:
        Parallel ``(M,)`` int64 arrays: match ``k`` pairs
        ``fa.keypoints[query_idx[k]]`` with ``fb.keypoints[train_idx[k]]``
        (OpenCV's own "query"/"train" naming, kept for familiarity).
    distances:
        ``(M,)`` float64 descriptor distance for each match (Hamming for
        binary descriptors, L2 for float), pre-ratio-test/verification.
    inlier_mask:
        ``(M,)`` bool, set by ``geometric_verify``; ``None`` before
        verification has run (meaning "no opinion yet", not "all inliers"
        -- callers that need a mask unconditionally should verify first).
    E, F:
        The estimated essential/fundamental matrix from the most recent
        ``geometric_verify`` call (whichever was requested), or ``None``.
    relative_pose:
        ``(R, t)`` recovered from the essential matrix via
        ``cv2.recoverPose`` when ``geometric_verify(method="essential")``
        was used -- ``R``/``t`` take camera *a*'s frame into camera *b*'s
        (OpenCV's own convention), ``t`` unit-norm (essential-matrix pose
        recovery is scale-free). ``None`` when verification used the
        fundamental matrix instead, or hasn't run.
    """

    query_idx: np.ndarray
    train_idx: np.ndarray
    distances: np.ndarray
    inlier_mask: np.ndarray | None = None
    E: np.ndarray | None = None
    F: np.ndarray | None = None
    relative_pose: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.query_idx = np.asarray(self.query_idx, dtype=np.int64).reshape(-1)
        self.train_idx = np.asarray(self.train_idx, dtype=np.int64).reshape(-1)
        self.distances = np.asarray(self.distances, dtype=np.float64).reshape(-1)
        if self.inlier_mask is not None:
            self.inlier_mask = np.asarray(self.inlier_mask, dtype=bool).reshape(-1)

    def __len__(self) -> int:
        return self.query_idx.shape[0]

    def inlier_ratio(self) -> float:
        """Fraction of matches ``geometric_verify`` accepted; 0.0 before verification or with no matches."""
        if self.inlier_mask is None or len(self) == 0:
            return 0.0
        return float(self.inlier_mask.sum()) / len(self)

    def inliers(self) -> tuple[np.ndarray, np.ndarray]:
        """``(query_idx, train_idx)`` restricted to inliers (or all matches, if not yet verified)."""
        if self.inlier_mask is None:
            return self.query_idx, self.train_idx
        return self.query_idx[self.inlier_mask], self.train_idx[self.inlier_mask]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _make_detector(method: str, max_features: int):
    method = method.lower()
    if method == "sift":
        return cv2.SIFT_create(nfeatures=max_features)
    if method == "orb":
        return cv2.ORB_create(nfeatures=max_features)
    if method == "akaze":
        if not hasattr(cv2, "AKAZE_create"):
            raise RuntimeError(
                "cv2.AKAZE_create is not available in this OpenCV build "
                "(some opencv-python wheels omit it); use method='sift' or 'orb' instead"
            )
        return cv2.AKAZE_create()
    raise ValueError(f"unknown feature method {method!r}; expected 'sift', 'akaze', or 'orb'")


def detect_and_describe(
    gray: np.ndarray, method: str = "sift", max_features: int = 4000, detect_scale: float = 1.0
) -> Features:
    """Detect keypoints and compute descriptors on a grayscale image.

    See the module docstring for why SIFT is the default, and when ORB/
    AKAZE are the better trade-off. ``max_features`` caps how many
    keypoints the detector keeps (its own internal response-based ranking
    decides which, for detectors that support a cap -- AKAZE has no such
    cap and returns however many it finds).

    detect_scale:
        Run the detector on the image downscaled by this factor (e.g.
        ``0.5`` -> detect at half resolution), then rescale the resulting
        keypoint coordinates back up to the *original* image's pixel
        frame before returning. ``1.0`` (default) runs at full
        resolution -- unchanged behaviour for every existing caller.

        Why this matters for 4K drone frames: SIFT's per-frame cost grows
        with pixel count (more scale-space octaves to build and search),
        so a 3840x2160 frame costs roughly ``1/detect_scale**2`` times
        less to detect on at a given ``detect_scale`` -- ``detect_scale
        =0.5`` is a ~4x speedup, measured at ~0.95s/frame -> ~0.25s/frame
        on real 4K footage (see ``config.MatchingConfig``'s "fast"/
        "balanced" quality profiles, which set this).

        Accuracy tradeoff (measure before trusting a non-default value):
        keypoint pixel coordinates recovered at ``detect_scale`` carry
        that scale's quantization into the upscale (a keypoint localized
        to +-0.5px at half resolution becomes +-1.0px after rescaling to
        full resolution), and small/fine texture that only resolves at
        full resolution can be missed by the detector entirely at a
        coarser scale, so both match density and per-observation pixel
        precision degrade somewhat as ``detect_scale`` drops -- this is
        exactly the "reprojection error" tradeoff bundle adjustment's
        accuracy ultimately depends on, so it must stay opt-in (not the
        default) and be reported, not silently applied. ``0.5`` is a
        reasonable "fast" floor; going much lower starts losing real
        matches on aerial imagery's already-repetitive ground texture.
    """
    if gray.ndim != 2:
        raise ValueError(f"detect_and_describe expects a grayscale (H, W) image, got shape {gray.shape}")
    if detect_scale <= 0.0 or detect_scale > 1.0:
        raise ValueError(f"detect_scale must be in (0, 1], got {detect_scale}")

    detect_gray = gray
    if detect_scale < 1.0:
        h, w = gray.shape
        new_w = max(1, round(w * detect_scale))
        new_h = max(1, round(h * detect_scale))
        detect_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    detector = _make_detector(method, max_features)
    keypoints, descriptors = detector.detectAndCompute(detect_gray, None)

    if keypoints and detect_scale < 1.0:
        # Rescale keypoint centres back into the original image's pixel
        # frame; ``size`` (the keypoint's own scale-space diameter) is
        # rescaled too so anything downstream reading it stays consistent
        # with the un-scaled image, even though this module only reads
        # ``.pt``/``.response`` itself.
        inv_scale = 1.0 / detect_scale
        for kp in keypoints:
            kp.pt = (kp.pt[0] * inv_scale, kp.pt[1] * inv_scale)
            kp.size = kp.size * inv_scale

    if not keypoints:
        return Features(
            keypoints=np.zeros((0, 2), dtype=np.float64),
            descriptors=None,
            scores=np.zeros((0,), dtype=np.float64),
            method=method.lower(),
        )

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float64)
    scores = np.array([kp.response for kp in keypoints], dtype=np.float64)
    return Features(keypoints=pts, descriptors=descriptors, scores=scores, method=method.lower())


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _make_matcher(method: str):
    if method in _BINARY_DESCRIPTOR_METHODS:
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    index_params = {"algorithm": _FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    return cv2.FlannBasedMatcher(index_params, search_params)


def _coerce_descriptors(descriptors: np.ndarray, method: str) -> np.ndarray:
    if method in _BINARY_DESCRIPTOR_METHODS:
        return np.ascontiguousarray(descriptors, dtype=np.uint8)
    return np.ascontiguousarray(descriptors, dtype=np.float32)


def _ratio_test_matches(matcher, desc_query: np.ndarray, desc_train: np.ndarray, ratio: float) -> dict[int, tuple[int, float]]:
    """Lowe's ratio test: keep query descriptor ``i``'s best train match only if it clearly beats the second-best.

    Returns ``{query_idx: (train_idx, distance)}``. ``knnMatch`` needs at
    least 2 train descriptors to return a "second best" at all, and OpenCV
    matchers are unhappy being handed a zero-row query array, so both are
    guarded here rather than left to raise from inside OpenCV.
    """
    if desc_query.shape[0] == 0 or desc_train.shape[0] < 2:
        return {}
    knn = matcher.knnMatch(desc_query, desc_train, k=2)
    best: dict[int, tuple[int, float]] = {}
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            best[m.queryIdx] = (m.trainIdx, float(m.distance))
    return best


def match_features(fa: Features, fb: Features, ratio: float = 0.8, cross_check: bool = True) -> Matches:
    """Match ``fa`` against ``fb`` via Lowe's ratio test, optionally with mutual-NN cross-check.

    ``cross_check=True`` (the default) additionally requires that
    ``fb``'s best ratio-test match for a given train descriptor points
    back at the same query descriptor -- i.e. each surviving pair is each
    other's nearest neighbour in *both* directions, not just accepted
    one-way. This is a standard, cheap way to reject one-to-many matches
    (a descriptor that looks like several things at once, e.g. from
    repetitive texture) that the ratio test alone can miss when the
    "several things" happen to also individually pass their own ratio
    tests.
    """
    if fa.method != fb.method:
        raise ValueError(f"cannot match Features from different methods ({fa.method!r} vs {fb.method!r})")
    if fa.descriptors is None or fb.descriptors is None or len(fa) == 0 or len(fb) == 0:
        return Matches(query_idx=np.zeros(0, dtype=np.int64), train_idx=np.zeros(0, dtype=np.int64), distances=np.zeros(0))

    matcher = _make_matcher(fa.method)
    desc_a = _coerce_descriptors(fa.descriptors, fa.method)
    desc_b = _coerce_descriptors(fb.descriptors, fb.method)

    forward = _ratio_test_matches(matcher, desc_a, desc_b, ratio)
    if not cross_check:
        if not forward:
            return Matches(query_idx=np.zeros(0, dtype=np.int64), train_idx=np.zeros(0, dtype=np.int64), distances=np.zeros(0))
        query_idx = np.array(list(forward.keys()), dtype=np.int64)
        train_idx = np.array([v[0] for v in forward.values()], dtype=np.int64)
        distances = np.array([v[1] for v in forward.values()], dtype=np.float64)
        return Matches(query_idx=query_idx, train_idx=train_idx, distances=distances)

    backward = _ratio_test_matches(matcher, desc_b, desc_a, ratio)
    query_idx_list: list[int] = []
    train_idx_list: list[int] = []
    dist_list: list[float] = []
    for qi, (ti, dist) in forward.items():
        back = backward.get(ti)
        if back is not None and back[0] == qi:
            query_idx_list.append(qi)
            train_idx_list.append(ti)
            dist_list.append(dist)

    return Matches(
        query_idx=np.array(query_idx_list, dtype=np.int64),
        train_idx=np.array(train_idx_list, dtype=np.int64),
        distances=np.array(dist_list, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Geometric verification
# ---------------------------------------------------------------------------

# RANSAC reprojection threshold (pixels) for the fundamental-matrix path,
# and its normalized-coordinate equivalent for the essential-matrix path
# (divided by an approximate focal length -- see ``geometric_verify``).
_RANSAC_THRESHOLD_PX = 1.5
_RANSAC_CONFIDENCE = 0.999


def _intrinsics_pair(intrinsics):
    """Accept either one shared ``CameraIntrinsics`` or an explicit ``(Ka, Kb)`` pair."""
    if isinstance(intrinsics, tuple):
        return intrinsics
    return intrinsics, intrinsics


def geometric_verify(
    fa: Features,
    fb: Features,
    matches: Matches,
    intrinsics=None,
    method: str = "fundamental",
) -> Matches:
    """RANSAC-filter ``matches`` against an epipolar geometry model.

    ``method="essential"`` (used whenever ``intrinsics`` is given and this
    is requested) is the *correct* model for a calibrated camera: it
    directly parametrizes the relative pose (rotation + translation
    direction) with the right degrees of freedom (5, via the 5-point
    algorithm), rather than the fundamental matrix's uncalibrated 7 DOF,
    and it hands back a metrically-meaningful relative pose for free
    (``Matches.relative_pose``) that ``tracks``/``triangulate`` can use as
    a sanity check or a pose-graph edge. ``method="fundamental"`` is the
    fallback when intrinsics aren't known (or weren't passed): it still
    rejects matches inconsistent with *any* rigid relative motion, just
    without recovering a metric pose.

    Both intrinsics-aware paths normalize points via ``cv2.undistortPoints``
    (which also removes lens distortion when ``CameraIntrinsics
    .dist_coeffs`` is set) before calling ``cv2.findEssentialMat`` with an
    identity camera matrix -- this handles the case of two different
    cameras/intrinsics (e.g. a lens change mid-flight) correctly, not just
    the common single-camera case, at the cost of expressing the RANSAC
    threshold in normalized-coordinate units (pixels / average focal
    length) rather than raw pixels.
    """
    n = len(matches)
    if n == 0:
        return Matches(
            query_idx=matches.query_idx, train_idx=matches.train_idx, distances=matches.distances,
            inlier_mask=np.zeros(0, dtype=bool),
        )

    pts_a = fa.keypoints[matches.query_idx]
    pts_b = fb.keypoints[matches.train_idx]

    # Both RANSAC models need strictly more correspondences than their
    # minimal solver (5-point essential / 8-point fundamental) to have any
    # outliers to actually reject.
    min_needed = 8
    if n < min_needed:
        return Matches(
            query_idx=matches.query_idx, train_idx=matches.train_idx, distances=matches.distances,
            inlier_mask=np.zeros(n, dtype=bool),
        )

    if method == "essential" and intrinsics is not None:
        ka, kb = _intrinsics_pair(intrinsics)
        norm_a = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), ka.K(), ka.dist_coeffs).reshape(-1, 2)
        norm_b = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), kb.K(), kb.dist_coeffs).reshape(-1, 2)
        avg_focal = 0.5 * (0.5 * (ka.fx + ka.fy) + 0.5 * (kb.fx + kb.fy))
        threshold_norm = _RANSAC_THRESHOLD_PX / max(avg_focal, 1e-6)
        identity_k = np.eye(3)

        E, mask = cv2.findEssentialMat(
            norm_a, norm_b, identity_k, method=cv2.RANSAC, prob=_RANSAC_CONFIDENCE, threshold=threshold_norm
        )
        if E is None or mask is None:
            return Matches(
                query_idx=matches.query_idx, train_idx=matches.train_idx, distances=matches.distances,
                inlier_mask=np.zeros(n, dtype=bool),
            )
        E = E[:3, :3]
        mask_e = mask.reshape(-1).astype(np.uint8)
        _, R, t, mask_pose = cv2.recoverPose(E, norm_a, norm_b, identity_k, mask=mask_e.copy())
        inlier_mask = mask_pose.reshape(-1).astype(bool)
        return Matches(
            query_idx=matches.query_idx,
            train_idx=matches.train_idx,
            distances=matches.distances,
            inlier_mask=inlier_mask,
            E=E,
            relative_pose=(R, t.reshape(3)),
        )

    if method not in ("fundamental", "essential"):
        raise ValueError(f"unknown geometric_verify method {method!r}; expected 'fundamental' or 'essential'")

    F, mask = cv2.findFundamentalMat(
        pts_a, pts_b, cv2.FM_RANSAC, ransacReprojThreshold=_RANSAC_THRESHOLD_PX, confidence=_RANSAC_CONFIDENCE
    )
    if F is None or mask is None:
        return Matches(
            query_idx=matches.query_idx, train_idx=matches.train_idx, distances=matches.distances,
            inlier_mask=np.zeros(n, dtype=bool),
        )
    F = F[:3, :3]
    inlier_mask = mask.reshape(-1).astype(bool)
    return Matches(
        query_idx=matches.query_idx, train_idx=matches.train_idx, distances=matches.distances,
        inlier_mask=inlier_mask, F=F,
    )


# ---------------------------------------------------------------------------
# Extension point for learned matchers
# ---------------------------------------------------------------------------


@runtime_checkable
class Matcher(Protocol):
    """Interface a learned matcher (LightGlue, SuperGlue, ...) implements to replace ``match_features``.

    See the module docstring for why classical detectors run today and how
    this protocol keeps the door open: anything with a ``match(fa, fb) ->
    Matches`` method can be swapped in wherever ``match_features`` is
    called (``tracks.build_tracks``'s callers -- see ``pipeline.stages
    .MatchingStage``) without changing ``tracks.py`` or ``triangulate.py``
    at all, since both only ever consume ``Matches``. ``geometric_verify``
    remains a separate, always-applicable step -- RANSAC epipolar
    verification is sensible regardless of which matcher produced the raw
    correspondences.
    """

    def match(self, fa: Features, fb: Features) -> Matches: ...
