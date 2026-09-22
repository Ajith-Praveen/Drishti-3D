"""Tests for drishti3d.geometry.features: detect/describe/match/verify.

Synthetic, deterministic, no external data: a textured random image and a
known homography-warped copy of it. Real photographs would work too, but a
synthetic warp gives us ground truth (the exact homography/essential
matrix the detector+matcher chain is supposed to recover) to check against.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drishti3d.geometry.features import (
    Features,
    Matches,
    detect_and_describe,
    geometric_verify,
    match_features,
)
from drishti3d.types import CameraIntrinsics, Pose

_SIZE = 480


def _textured_image(seed: int = 0, size: int = _SIZE) -> np.ndarray:
    """A random-blob textured grayscale image with plenty of corner-like structure for SIFT/ORB."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8)
    for _ in range(400):
        x, y = rng.integers(0, size, size=2)
        r = rng.integers(3, 14)
        color = int(rng.integers(40, 255))
        cv2.circle(img, (int(x), int(y)), int(r), color, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def _rotation_translation_warp(image: np.ndarray, angle_deg: float, tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``image`` by an in-plane rotation + translation; returns (warped, 3x3 homography)."""
    h, w = image.shape
    center = (w / 2.0, h / 2.0)
    m2x3 = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    m2x3[0, 2] += tx
    m2x3[1, 2] += ty
    warped = cv2.warpAffine(image, m2x3, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    homography = np.vstack([m2x3, [0.0, 0.0, 1.0]])
    return warped, homography


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_detect_and_describe_finds_features(method):
    img = _textured_image()
    feats = detect_and_describe(img, method=method, max_features=2000)
    assert len(feats) > 100
    assert feats.keypoints.shape == (len(feats), 2)
    assert feats.descriptors is not None
    assert feats.descriptors.shape[0] == len(feats)
    assert feats.method == method


def test_detect_and_describe_rejects_color_image():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_and_describe(img)


def test_detect_and_describe_unknown_method_raises():
    img = _textured_image()
    with pytest.raises(ValueError):
        detect_and_describe(img, method="bogus")


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_match_and_verify_recover_known_warp(method):
    """Detect/match/verify across a known rotation+translation warp; check inlier ratio and the fitted homography."""
    img_a = _textured_image(seed=1)
    angle_deg, tx, ty = 8.0, 15.0, -10.0
    img_b, true_homography = _rotation_translation_warp(img_a, angle_deg, tx, ty)

    feats_a = detect_and_describe(img_a, method=method, max_features=3000)
    feats_b = detect_and_describe(img_b, method=method, max_features=3000)
    assert len(feats_a) > 50 and len(feats_b) > 50

    matches = match_features(feats_a, feats_b, ratio=0.8, cross_check=True)
    assert len(matches) > 20

    verified = geometric_verify(feats_a, feats_b, matches, intrinsics=None, method="fundamental")
    assert verified.inlier_mask is not None
    inlier_ratio = verified.inlier_ratio()
    assert inlier_ratio > 0.6, f"expected a high inlier ratio for a clean rigid warp, got {inlier_ratio}"

    # Check the *fitted* homography (via inlier correspondences) is close to
    # the known warp -- a robust independent check that geometric_verify's
    # inliers are actually the geometrically-consistent points, not just
    # "some matches got in by chance."
    q_idx, t_idx = verified.inliers()
    pts_a = feats_a.keypoints[q_idx]
    pts_b = feats_b.keypoints[t_idx]
    fitted_h, _ = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 3.0)
    assert fitted_h is not None

    # Compare by transforming a handful of canonical points with both
    # homographies rather than comparing matrices directly (they only agree
    # up to an overall scale).
    fitted_h = fitted_h / fitted_h[2, 2]
    probe_pts = np.array([[0, 0, 1], [_SIZE, 0, 1], [0, _SIZE, 1], [_SIZE, _SIZE, 1]], dtype=np.float64)
    true_proj = (true_homography @ probe_pts.T).T
    true_proj = true_proj[:, :2] / true_proj[:, 2:3]
    fitted_proj = (fitted_h @ probe_pts.T).T
    fitted_proj = fitted_proj[:, :2] / fitted_proj[:, 2:3]
    residual = np.linalg.norm(true_proj - fitted_proj, axis=1)
    assert np.all(residual < 5.0), f"fitted homography disagrees with the true warp by {residual} px"


def test_geometric_verify_essential_recovers_relative_pose():
    """Essential-matrix verification on genuine (non-planar) 3D correspondences recovers the true relative rotation.

    A pure 2D image warp (as used by the homography test above) is exactly
    the classical degenerate case for essential-matrix estimation -- every
    correspondence lies on a single homography, which the 5-point algorithm
    cannot disambiguate from a translation. This test instead projects
    depth-varying synthetic 3D points through two genuinely different
    camera poses (so real parallax exists) and checks ``geometric_verify``
    recovers a valid rotation matrix close to the known one, with a
    scattering of gross outlier correspondences for RANSAC to reject.
    """
    rng = np.random.default_rng(4)
    intrinsics = CameraIntrinsics.from_hfov(70.0, _SIZE, _SIZE)
    n_pts = 80
    points = np.stack(
        [
            rng.uniform(-5.0, 5.0, n_pts),
            rng.uniform(-5.0, 5.0, n_pts),
            rng.uniform(15.0, 25.0, n_pts),
        ],
        axis=1,
    )

    pose_a = Pose(R=np.eye(3), t=np.zeros(3))
    true_rotation = Rotation.from_euler("y", 5.0, degrees=True).as_matrix()
    pose_b = Pose(R=true_rotation, t=np.array([2.0, 0.3, -0.5]))

    def _project(pose: Pose) -> np.ndarray:
        xc = (pose.R.T @ (points - pose.t).T).T
        u = intrinsics.fx * xc[:, 0] / xc[:, 2] + intrinsics.cx
        v = intrinsics.fy * xc[:, 1] / xc[:, 2] + intrinsics.cy
        return np.stack([u, v], axis=1)

    pts_a = _project(pose_a)
    pts_b = _project(pose_b)

    outlier_idx = rng.choice(n_pts, size=8, replace=False)
    pts_b[outlier_idx] += rng.uniform(-200.0, 200.0, size=(8, 2))

    feats_a = Features(keypoints=pts_a, descriptors=None, scores=np.ones(n_pts), method="synthetic")
    feats_b = Features(keypoints=pts_b, descriptors=None, scores=np.ones(n_pts), method="synthetic")
    matches = Matches(query_idx=np.arange(n_pts), train_idx=np.arange(n_pts), distances=np.zeros(n_pts))

    verified = geometric_verify(feats_a, feats_b, matches, intrinsics=intrinsics, method="essential")

    assert verified.inlier_mask is not None
    assert verified.inlier_ratio() > 0.8
    assert verified.relative_pose is not None
    rot, _t = verified.relative_pose
    # A valid rotation matrix: orthonormal with determinant +1.
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-6)

    # cv2.recoverPose's R maps camera A's frame into camera B's frame
    # (X_camB = R @ X_camA + t); since camera A's pose here is the
    # identity (world == camera A), that is `pose_b.R.T`, not `pose_b.R`
    # (which is world-from-camera-B, i.e. the inverse direction).
    expected_rotation = true_rotation.T
    angle_err_deg = np.degrees(np.arccos(np.clip((np.trace(rot.T @ expected_rotation) - 1.0) / 2.0, -1.0, 1.0)))
    assert angle_err_deg < 2.0, f"recovered rotation off by {angle_err_deg:.2f} degrees"


def test_detect_and_describe_detect_scale_speeds_up_and_keeps_keypoints_close():
    """``detect_scale`` < 1 must (a) still find features, (b) return them in
    the *original* image's pixel frame (not the downscaled one -- the whole
    point is downstream code never needs to know detection happened at a
    lower resolution), and (c) be measurably faster.
    """
    import time

    img = _textured_image(size=960)

    t0 = time.perf_counter()
    full = detect_and_describe(img, method="sift", max_features=3000, detect_scale=1.0)
    dt_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    half = detect_and_describe(img, method="sift", max_features=3000, detect_scale=0.5)
    dt_half = time.perf_counter() - t0

    assert len(full) > 50
    assert len(half) > 50
    # Keypoint coordinates must land within the original image bounds, not
    # the (half-sized) detection buffer's.
    assert half.keypoints[:, 0].max() <= img.shape[1]
    assert half.keypoints[:, 1].max() <= img.shape[0]
    assert dt_half < dt_full, f"half-resolution detection ({dt_half:.4f}s) should be faster than full ({dt_full:.4f}s)"

    # Every half-resolution keypoint should have a nearby full-resolution
    # keypoint (same underlying blob structure, just found at a coarser
    # scale) -- checks the rescale-back-to-full-image-frame math is right,
    # not just "some points came out."
    for pt in half.keypoints[: min(20, len(half))]:
        dists = np.linalg.norm(full.keypoints - pt, axis=1)
        assert dists.min() < 15.0, f"half-res keypoint {pt} has no nearby full-res match (min dist {dists.min():.1f}px)"


def test_detect_and_describe_rejects_bad_detect_scale():
    img = _textured_image()
    with pytest.raises(ValueError):
        detect_and_describe(img, detect_scale=0.0)
    with pytest.raises(ValueError):
        detect_and_describe(img, detect_scale=1.5)


def test_match_features_rejects_mismatched_methods():
    img = _textured_image()
    feats_sift = detect_and_describe(img, method="sift")
    feats_orb = detect_and_describe(img, method="orb")
    with pytest.raises(ValueError):
        match_features(feats_sift, feats_orb)


def test_match_features_empty_when_no_descriptors():
    empty = detect_and_describe(np.zeros((32, 32), dtype=np.uint8), method="orb")
    other = detect_and_describe(_textured_image(), method="orb")
    matches = match_features(empty, other)
    assert len(matches) == 0


def test_geometric_verify_too_few_matches_returns_all_outliers():
    img = _textured_image()
    feats = detect_and_describe(img, method="orb", max_features=50)
    # Fabricate a tiny match set (fewer than the minimal RANSAC sample size).
    matches = Matches(query_idx=np.array([0, 1, 2]), train_idx=np.array([0, 1, 2]), distances=np.array([1.0, 1.0, 1.0]))
    verified = geometric_verify(feats, feats, matches, method="fundamental")
    assert verified.inlier_mask is not None
    assert not verified.inlier_mask.any()
