"""Tests for geometry.focal_from_flow.

The estimator inverts d = fx*B/Z for a nadir camera over flat ground, so
a synthetic scene rendered with a KNOWN focal must return that focal.
Views are warped through the real plane-induced homography, so SIFT,
matching and verification all run for real.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drishti3d.geometry.focal_from_flow import estimate_focal_from_flow


def _render(fx: float, altitude: float, baseline: float, n: int = 8, w: int = 1280, h: int = 720, seed: int = 0):
    """Nadir views of a textured ground plane at z=0, from a camera with focal ``fx``."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (2200, 2200), dtype=np.uint8), (3, 3), 0)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    gsd, origin = 0.12, np.array([-80.0, -80.0])
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])
    images, positions = [], []
    for i in range(n):
        C = np.array([baseline * i, 0.0, altitude])
        r_cw = R.T
        t_cw = -r_cw @ C
        Hw = K @ np.column_stack([r_cw[:, 0], r_cw[:, 1], t_cw])
        T = np.array([[gsd, 0, origin[0]], [0, gsd, origin[1]], [0, 0, 1.0]])
        images.append(cv2.warpPerspective(tex, Hw @ T, (w, h), flags=cv2.INTER_LINEAR))
        positions.append(C)
    return images, positions


@pytest.mark.parametrize("fx_true", [1400.0, 2698.0])
def test_recovers_the_focal_length_it_was_rendered_with(fx_true):
    """2698 px is the value measured on the real sample flight."""
    alt, base = 120.0, 25.0
    images, positions = _render(fx_true, alt, base)
    est = estimate_focal_from_flow(
        images, positions, [alt] * len(images), [-90.0] * len(images), native_width=1280
    )
    assert est is not None
    assert est.fx == pytest.approx(fx_true, rel=0.03)
    assert est.spread_fraction < 0.05


def test_reports_native_pixels_even_when_matching_is_downscaled():
    """The focal returned must describe the FULL-resolution frame.

    Displacements are measured on a downscaled working image; forgetting
    to convert them back yields a focal that is wrong by exactly the
    downscale ratio -- and nothing downstream would notice.
    """
    alt, base, fx_native = 120.0, 25.0, 2698.0
    # Render at half size: the same camera, described in half-size pixels.
    images, positions = _render(fx_native / 2, alt, base, w=640, h=360)
    est = estimate_focal_from_flow(
        images, positions, [alt] * len(images), [-90.0] * len(images), native_width=1280
    )
    assert est is not None
    assert est.fx == pytest.approx(fx_native, rel=0.05)


def test_refuses_when_the_camera_is_not_nadir():
    alt, base = 120.0, 25.0
    images, positions = _render(2698.0, alt, base)
    est = estimate_focal_from_flow(
        images, positions, [alt] * len(images), [-40.0] * len(images), native_width=1280
    )
    assert est is None


def test_refuses_when_the_drone_is_not_moving():
    alt = 120.0
    images, positions = _render(2698.0, alt, baseline=0.5)
    est = estimate_focal_from_flow(
        images, positions, [alt] * len(images), [-90.0] * len(images), native_width=1280
    )
    assert est is None
