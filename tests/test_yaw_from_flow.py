"""Tests for geometry.yaw_from_flow: recovering camera yaw from image motion + GPS."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from drishti3d.geometry.bundle import gimbal_to_R
from drishti3d.geometry.yaw_from_flow import estimate_yaw_from_flow, yaw_for_pair
from drishti3d.types import CameraIntrinsics


def _render(yaw_deg: float, heading_deg: float, n: int = 3, altitude: float = 120.0, step: float = 20.0, w=640, h=400, seed=0):
    """Nadir camera at ``yaw_deg`` flying along compass ``heading_deg`` over a textured plane."""
    rng = np.random.default_rng(seed)
    intr = CameraIntrinsics.from_hfov(70.0, w, h)
    K = intr.K()
    tex = cv2.GaussianBlur(rng.integers(0, 255, (1600, 1600), dtype=np.uint8), (3, 3), 0)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    gsd, origin = 0.25, np.array([-200.0, -200.0])
    R = gimbal_to_R(yaw_deg, -90.0, 0.0)
    hdg = np.radians(heading_deg)
    d = np.array([np.sin(hdg), np.cos(hdg), 0.0]) * step  # ENU: x=E, y=N
    images, positions = [], []
    for i in range(n):
        C = np.array([0.0, 0.0, altitude]) + i * d
        r_cw = R.T
        t_cw = -r_cw @ C
        Hw = K @ np.column_stack([r_cw[:, 0], r_cw[:, 1], t_cw])
        T = np.array([[gsd, 0, origin[0]], [0, gsd, origin[1]], [0, 0, 1.0]])
        images.append(cv2.warpPerspective(tex, Hw @ T, (w, h), flags=cv2.INTER_LINEAR))
        positions.append(C)
    return images, positions


@pytest.mark.parametrize("yaw,heading", [(0.0, 0.0), (169.1, 184.0), (328.4, 344.0), (45.0, 300.0)])
def test_yaw_for_pair_recovers_camera_yaw(yaw, heading):
    """The camera's yaw is recovered regardless of which way the drone flies.

    169.1/184 and 328.4/344 are real (gimbal, course) pairs from the sample
    flight: the estimator must not be fooled by yaw and heading differing.
    """
    images, positions = _render(yaw, heading)
    est, diag = yaw_for_pair(images[0], images[1], positions[1] - positions[0], -90.0, 0.0)
    assert est is not None, diag
    err = abs(((est - yaw + 180) % 360) - 180)
    assert err < 2.0, f"yaw {est:.1f} vs true {yaw:.1f} ({diag})"


def test_stationary_pair_is_refused():
    images, positions = _render(30.0, 90.0, step=0.5)
    est, diag = yaw_for_pair(images[0], images[1], positions[1] - positions[0], -90.0, 0.0)
    assert est is None and "displacement" in diag["failure"]


def test_estimate_falls_back_to_telemetry_where_unmeasurable():
    """A keyframe with no measurable neighbour keeps its telemetry yaw, labelled as such."""
    images, positions = _render(97.0, 113.0, n=3)
    images[2] = None  # last pair unmeasurable
    yaws, diag = estimate_yaw_from_flow(images, positions, [-90.0] * 3, [0.0] * 3, prior_yaw_deg=[80.0, 80.0, 80.0])
    assert diag["per_keyframe"][0]["source"] == "flow"
    assert diag["per_keyframe"][1]["source"] == "flow"
    assert diag["per_keyframe"][2]["source"] == "telemetry"
    assert yaws[2] == 80.0
    assert abs(((yaws[0] - 97.0 + 180) % 360) - 180) < 2.0
    # The measured offset from the (deliberately wrong) prior is reported.
    assert diag["median_delta_deg"] == pytest.approx(17.0, abs=2.5)
