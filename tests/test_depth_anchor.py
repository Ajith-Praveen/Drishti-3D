"""Tests for geometry.depth_anchor: parallax-anchored depth rescaling.

The scenario is the measured failure, reproduced synthetically: nadir
cameras at a known metric height over a textured ground plane, and a
"backbone" whose depth is a constant factor too shallow. Views are rendered
by warping one texture through the plane-induced homography, so the images
really do share features at geometrically consistent positions -- SIFT,
matching, verification and triangulation all run for real; only the depth
being corrected is synthetic.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drishti3d.geometry.backbone import BackboneResult
from drishti3d.geometry.depth_anchor import anchor_depth_to_parallax, measure_depth_ratios
from drishti3d.types import CameraIntrinsics, Pose

_NADIR_R = np.diag([1.0, -1.0, -1.0])  # camera +Z looks down world -Z


def _render_nadir_views(n_views: int, altitude: float, spacing: float, w: int = 640, h: int = 400, seed: int = 0):
    """Render ``n_views`` nadir views of a random-texture ground plane at z=0."""
    rng = np.random.default_rng(seed)
    intr = CameraIntrinsics.from_hfov(70.0, w, h)
    K = intr.K()

    # World texture: a big random image mapped onto the ground plane at
    # ``texture_gsd`` metres per texel, origin at world (-tx0, -ty0).
    tex = cv2.GaussianBlur(rng.integers(0, 255, (1400, 1400), dtype=np.uint8), (3, 3), 0)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    gsd = 0.25
    origin = np.array([-100.0, -100.0])

    def texel_from_world(xy):
        return (xy - origin) / gsd

    poses, images = [], []
    for i in range(n_views):
        C = np.array([i * spacing, 0.0, altitude])
        pose = Pose(R=_NADIR_R, t=C)
        # Homography ground(x,y,0) -> pixel: K [r1 r2 | -R^T C]
        r_cw = pose.R.T
        t_cw = -r_cw @ C
        H_world = K @ np.column_stack([r_cw[:, 0], r_cw[:, 1], t_cw])
        # Compose with texel->world: world = texel*gsd + origin
        T = np.array([[gsd, 0, origin[0]], [0, gsd, origin[1]], [0, 0, 1.0]])
        H = H_world @ T
        img = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR)
        poses.append(pose)
        images.append(img)
    return images, poses, [intr] * n_views


def _fake_backbone_result(poses, intr, images, depth_factor: float) -> BackboneResult:
    """A backbone whose depth is the true depth times ``depth_factor``."""
    n = len(images)
    h, w = images[0].shape[:2]
    depth = np.full((n, h, w), poses[0].t[2] * depth_factor, dtype=np.float64)
    K_inv = np.linalg.inv(intr.K())
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1)
    points = np.empty((n, h, w, 3))
    for i, p in enumerate(poses):
        cam = (pix @ K_inv.T) * depth[i][..., None]
        points[i] = cam @ p.R.T + p.t
    return BackboneResult(
        poses=list(poses),
        points=points,
        depth=depth,
        confidence=np.full((n, h, w), 0.9, dtype=np.float32),
        intrinsics=[intr] * n,
        is_metric=True,
        metadata={},
    )


@pytest.mark.parametrize("depth_factor", [0.15, 0.59, 1.0])
def test_anchor_recovers_true_altitude(depth_factor):
    """0.15 and 0.59 are the ratios measured on real footage; 1.0 must be a no-op."""
    altitude = 120.0
    images, poses, intrs = _render_nadir_views(4, altitude, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, depth_factor)

    anchored, diag = anchor_depth_to_parallax(images, result, poses, intrs, min_samples=20)

    assert diag["applied"], diag
    assert diag["window_ratio"] == pytest.approx(1.0 / depth_factor, rel=0.05)
    # After anchoring the ground sits at z ~ 0 beneath every camera.
    ground = np.median(anchored.points[..., 2])
    assert abs(ground) < 0.05 * altitude
    # Identity case must not disturb the geometry.
    if depth_factor == 1.0:
        np.testing.assert_allclose(anchored.points, result.points, rtol=0.05, atol=0.5)


def test_anchor_refuses_absurd_ratio():
    """A backbone off by 50x is reported as failed, not rescaled into plausibility."""
    images, poses, intrs = _render_nadir_views(4, 120.0, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, 0.02)
    anchored, diag = anchor_depth_to_parallax(images, result, poses, intrs, min_samples=20, max_ratio=20.0)
    assert not diag["applied"]
    assert "outside" in diag["failure"]
    assert anchored is result


def test_anchor_leaves_masked_pixels_alone():
    """Zero-depth (masked) pixels are not points and must not be moved."""
    images, poses, intrs = _render_nadir_views(3, 120.0, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, 0.5)
    result.depth[:, :50, :] = 0.0
    result.points[:, :50, :, :] = 0.0
    anchored, diag = anchor_depth_to_parallax(images, result, poses, intrs, min_samples=20)
    assert diag["applied"]
    assert np.all(anchored.depth[:, :50, :] == 0.0)
    assert np.all(anchored.points[:, :50, :, :] == 0.0)


def test_measure_reports_missing_telemetry():
    images, poses, intrs = _render_nadir_views(3, 120.0, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, 0.5)
    _, diag = measure_depth_ratios(images, result, [None] * 3, intrs)
    assert "failure" in diag


# ---------------------------------------------------------------------------
# GPS-altitude fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth_factor", [0.17, 0.6, 1.0])
def test_gps_altitude_anchor_recovers_true_depth(depth_factor):
    """Ground must end up its GPS altitude below the camera.

    0.17 is the ratio measured on windows that got NO parallax anchoring
    at all on real footage -- 15 of 19 windows, which is what produced a
    132.9 m spread of ground elevations and a 19 m-thick fused slab.
    """
    from drishti3d.geometry.depth_anchor import anchor_depth_to_gps_altitude

    altitude = 120.0
    images, poses, intrs = _render_nadir_views(4, altitude, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, depth_factor)

    out, diag = anchor_depth_to_gps_altitude(result, [altitude] * 4)

    assert diag["applied"], diag
    assert diag["window_ratio"] == pytest.approx(1.0 / depth_factor, rel=0.05)
    # Ground now sits ~altitude below each camera, i.e. at z ~ 0.
    assert abs(float(np.median(out.points[..., 2]))) < 0.05 * altitude


def test_gps_altitude_anchor_refuses_absurd_corrections():
    from drishti3d.geometry.depth_anchor import anchor_depth_to_gps_altitude

    images, poses, intrs = _render_nadir_views(3, 120.0, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, 0.002)  # 500x out
    out, diag = anchor_depth_to_gps_altitude(result, [120.0] * 3)
    assert not diag["applied"]
    assert "outside" in diag["failure"]
    assert out is result


def test_gps_altitude_anchor_reports_when_altitude_is_missing():
    from drishti3d.geometry.depth_anchor import anchor_depth_to_gps_altitude

    images, poses, intrs = _render_nadir_views(3, 120.0, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, 0.5)
    out, diag = anchor_depth_to_gps_altitude(result, [None, None, None])
    assert not diag["applied"]
    assert "altitude" in diag["failure"]
    assert out is result


def test_gps_altitude_anchor_closes_the_gap_between_windows():
    """Two windows anchored independently must land at the SAME ground.

    This is the 132.9 m elevation spread in miniature: unanchored windows
    with different depth errors put the same ground at different heights,
    and the fused cloud becomes a thick slab of duplicates.
    """
    from drishti3d.geometry.depth_anchor import anchor_depth_to_gps_altitude

    altitude = 120.0
    images, poses, intrs = _render_nadir_views(4, altitude, spacing=25.0)
    a, _ = anchor_depth_to_gps_altitude(
        _fake_backbone_result(poses, intrs[0], images, 0.17), [altitude] * 4
    )
    b, _ = anchor_depth_to_gps_altitude(
        _fake_backbone_result(poses, intrs[0], images, 0.60), [altitude] * 4
    )
    ground_a = float(np.median(a.points[..., 2]))
    ground_b = float(np.median(b.points[..., 2]))
    # Before anchoring these differ by 0.43 * 120 = ~52 m.
    assert abs(ground_a - ground_b) < 2.0, f"windows still {abs(ground_a-ground_b):.1f} m apart"


# ---------------------------------------------------------------------------
# Fusing parallax + GPS altitude
# ---------------------------------------------------------------------------


def test_fusion_uses_parallax_when_it_is_well_determined():
    """With thousands of matched samples, parallax should dominate the blend."""
    from drishti3d.geometry.depth_anchor import anchor_depth_fused

    alt = 120.0
    images, poses, intrs = _render_nadir_views(4, alt, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, 0.2)

    out, diag = anchor_depth_fused(images, result, poses, intrs, [alt] * 4)

    assert diag["applied"], diag
    assert diag["source"] in ("fused", "parallax_only"), diag
    assert diag["window_ratio"] == pytest.approx(5.0, rel=0.08)
    assert abs(float(np.median(out.points[..., 2]))) < 0.05 * alt
    if diag["source"] == "fused":
        # Both are sample-rich here, so both sit at their systematic floor
        # (2% for parallax, 4% for the altitude anchor). Weight goes as
        # 1/sigma^2, so parallax should carry ~4x the weight -- not because
        # it has more samples, but because it is limited by a smaller
        # systematic error.
        assert diag["weight_parallax"] == pytest.approx(0.8, abs=0.05), diag


def test_fusion_falls_back_to_gps_when_parallax_has_nothing():
    """Blank imagery kills matching; GPS altitude must still anchor the window.

    This is the measured failure: 15 of 19 real windows produced zero
    parallax samples and kept depth 5-6x too shallow.
    """
    from drishti3d.geometry.depth_anchor import anchor_depth_fused

    alt = 120.0
    images, poses, intrs = _render_nadir_views(4, alt, spacing=25.0)
    blank = [np.full_like(im, 128) for im in images]
    result = _fake_backbone_result(poses, intrs[0], images, 0.17)

    out, diag = anchor_depth_fused(blank, result, poses, intrs, [alt] * 4)

    assert diag["applied"], diag
    assert diag["source"] == "gps_only"
    assert diag["window_ratio"] == pytest.approx(1 / 0.17, rel=0.05)
    assert abs(float(np.median(out.points[..., 2]))) < 0.05 * alt


def test_fusion_reports_disagreement_instead_of_averaging_it():
    """A wrong altitude must not drag a good parallax estimate halfway to it.

    Terrain sloping away from the takeoff elevation does exactly this. The
    mean of a right and a wrong answer is wrong, so the better-determined
    estimate is used alone and the disagreement is surfaced.
    """
    from drishti3d.geometry.depth_anchor import anchor_depth_fused

    alt = 120.0
    images, poses, intrs = _render_nadir_views(4, alt, spacing=25.0)
    result = _fake_backbone_result(poses, intrs[0], images, 0.2)

    # Tell it the drone is at 40 m when the scene says 120 m.
    out, diag = anchor_depth_fused(images, result, poses, intrs, [40.0] * 4)

    assert diag["disagreement"] > 1.6
    assert "disagreement" in diag["source"]
    # Parallax has thousands of samples; it must win.
    assert diag["source"] == "parallax_disagreement"
    assert diag["window_ratio"] == pytest.approx(5.0, rel=0.1)


def test_fusion_refuses_when_neither_estimator_works():
    from drishti3d.geometry.depth_anchor import anchor_depth_fused

    images, poses, intrs = _render_nadir_views(3, 120.0, spacing=25.0)
    blank = [np.full_like(im, 128) for im in images]
    result = _fake_backbone_result(poses, intrs[0], images, 0.5)

    out, diag = anchor_depth_fused(blank, result, poses, intrs, [None, None, None])
    assert not diag["applied"]
    assert "neither" in diag["failure"]
    assert out is result
