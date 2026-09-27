"""Tests for geometry.plane_sweep.

A synthetic scene with EXACT known depth: nadir cameras over a textured
ground plane with a raised box. The backbone's depth is simulated by
corrupting the truth, and the refiner must recover it by matching -- which
is the whole claim, so it is tested against truth rather than against
self-consistency.
"""

from __future__ import annotations

import cv2
import numpy as np

from drishti3d.geometry.plane_sweep import refine_depth_by_plane_sweep
from drishti3d.types import CameraIntrinsics, Pose

_NADIR_R = np.diag([1.0, -1.0, -1.0])


def _scene(n_views=4, altitude=120.0, baseline=20.0, w=192, h=144, fx=260.0, seed=0, box=True, yaw_step=0.0):
    """Nadir views of a textured ground plane (z=0) with a raised slab.

    Returns (images, true_depth, poses, intrinsics). Depth is exact: it is
    computed from the ray/plane and ray/box intersections, not estimated.
    """
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (1500, 1500), dtype=np.uint8), (3, 3), 0)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    gsd, origin = 0.35, np.array([-140.0, -140.0])
    intr = CameraIntrinsics(fx=fx, fy=fx, cx=w / 2, cy=h / 2, width=w, height=h)
    K = intr.K()
    K[:2, 2] -= 0.5  # OpenCV array coordinates vs camera pixel centres.

    images, depths, poses = [], [], []
    ys, xs = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    for i in range(n_views):
        C = np.array([baseline * i, 0.0, altitude])
        angle = np.radians(yaw_step * i)
        yaw = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        pose = Pose(R=yaw @ _NADIR_R, t=C)
        r_cw = pose.R.T
        t_cw = -r_cw @ C
        Hw = K @ np.column_stack([r_cw[:, 0], r_cw[:, 1], t_cw])
        T = np.array([[gsd, 0, origin[0]], [0, gsd, origin[1]], [0, 0, 1.0]])
        images.append(cv2.warpPerspective(tex, Hw @ T, (w, h), flags=cv2.INTER_LINEAR))

        # Exact depth: nadir camera, ground at z=0 -> depth = altitude.
        d = np.full((h, w), altitude, dtype=np.float64)
        if box:
            # A 40x40 m slab 8 m tall, centred in the flight strip.
            X = (xs - intr.cx) / intr.fx * altitude + C[0]
            Y = -((ys - intr.cy) / intr.fy) * altitude + C[1]
            inside = (np.abs(X - baseline * (n_views - 1) / 2) < 20) & (np.abs(Y) < 20)
            d[inside] = altitude - 8.0
        depths.append(d)
        poses.append(pose)
    return images, np.stack(depths), poses, [intr] * n_views


def test_recovers_depth_corrupted_by_a_scale_error():
    """A 6% depth error -- the regression noise this exists to remove."""
    images, truth, poses, intr = _scene(box=False)
    noisy = truth * 1.06

    out = refine_depth_by_plane_sweep(np.stack(images), noisy, poses, intr, min_views=1)

    assert out.stats["refined_pct"] > 40, out.stats
    err_before = np.abs(noisy - truth)[out.refined]
    err_after = np.abs(out.depth - truth)[out.refined]
    assert np.median(err_after) < 0.4 * np.median(err_before), (
        f"median error {np.median(err_before):.2f} -> {np.median(err_after):.2f} m"
    )


def test_recovers_depth_corrupted_by_per_pixel_noise():
    """Random per-pixel noise, which no scale correction can fix.

    geometry.depth_anchor and fusion.reanchor both rescale a whole view;
    neither can touch noise like this. Matching can, and that is the
    difference this module exists for.

    The scene is flown LOWER than the others on purpose. Triangulation
    sensitivity is ``fx*B/Z^2`` pixels of disparity per metre of depth
    error, and a depth error below ~1 px of disparity is not measurable
    by any matcher. On the real footage (fx 2697, B 23 m, Z 120 m) that
    is 4.3 px/m, so the measured 0.38 m noise shows as 1.6 px -- clearly
    resolvable. A small synthetic image at Z=120 gives only 0.36 px/m,
    which is BELOW the resolution limit; a test built there would be
    asserting that the matcher does something geometrically impossible.
    Halving the altitude restores a representative 1.4 px/m.
    """
    rng = np.random.default_rng(3)
    images, truth, poses, intr = _scene(box=False, altitude=60.0)
    sensitivity = intr[0].fx * 20.0 / 60.0**2  # px of disparity per metre
    assert sensitivity > 1.0, f"test scene is below the resolution limit ({sensitivity:.2f} px/m)"
    noisy = truth + rng.normal(0, 3.0, truth.shape)

    out = refine_depth_by_plane_sweep(np.stack(images), noisy, poses, intr, min_views=1)

    err_before = np.abs(noisy - truth)[out.refined]
    err_after = np.abs(out.depth - truth)[out.refined]
    assert np.median(err_after) < np.median(err_before), (
        f"median error {np.median(err_before):.2f} -> {np.median(err_after):.2f} m"
    )


def test_leaves_pixels_alone_when_evidence_is_weak():
    """Untextured scene: every depth explains it equally, so refuse.

    A flat cost curve means the argmin is noise. Accepting it would
    replace a merely-imprecise depth with a confidently-wrong one.
    """
    images, truth, poses, intr = _scene(box=False)
    blank = [np.full_like(im, 128) for im in images]
    noisy = truth * 1.06

    out = refine_depth_by_plane_sweep(np.stack(blank), noisy, poses, intr, min_views=1)

    assert out.stats["refined_pct"] < 5.0, out.stats
    np.testing.assert_allclose(out.depth[~out.refined], noisy[~out.refined])


def test_single_view_is_refused_not_guessed():
    images, truth, poses, intr = _scene(n_views=1, box=False)
    out = refine_depth_by_plane_sweep(np.stack(images), truth, poses, intr)
    assert out.stats.get("failure")
    assert not out.refined.any()
    np.testing.assert_array_equal(out.depth, truth)


def test_unrefined_pixels_keep_the_backbone_depth():
    """Never fills a rejected pixel with a guess."""
    images, truth, poses, intr = _scene(box=False)
    noisy = truth * 1.06
    out = refine_depth_by_plane_sweep(np.stack(images), noisy, poses, intr, min_ncc=0.999, min_views=1)
    np.testing.assert_allclose(out.depth[~out.refined], noisy[~out.refined])


def test_rotating_cameras_recover_known_depth():
    images, truth, poses, intr = _scene(box=False, yaw_step=25.0, altitude=60.0)
    noisy = truth * 1.06
    out = refine_depth_by_plane_sweep(np.stack(images), noisy, poses, intr, min_views=1)
    assert out.stats["refined_pct"] > 20, out.stats
    assert np.median(np.abs(out.depth[out.refined] - truth[out.refined])) < 0.15
