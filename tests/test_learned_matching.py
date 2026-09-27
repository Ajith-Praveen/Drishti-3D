"""Tests for drishti3d.geometry.learned_matching's gravity check (no model weights needed)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from drishti3d.geometry.bundle import gimbal_to_R
from drishti3d.geometry.learned_matching import gravity_consistent


def test_true_relative_rotation_passes_and_tilted_one_fails() -> None:
    Ra = gimbal_to_R(30.0, -88.0, 1.0)
    Rb = gimbal_to_R(35.0, -85.0, -2.0)
    true_rel = Rb.T @ Ra  # a's camera frame -> b's camera frame

    ok, err = gravity_consistent(true_rel, Ra, Rb)
    assert ok and err < 1e-6

    wrong = Rotation.from_euler("x", 8.0, degrees=True).as_matrix() @ true_rel
    ok, err = gravity_consistent(wrong, Ra, Rb, max_deg=3.0)
    assert not ok and 7.0 < err < 9.0


def test_yaw_error_is_not_vetoed() -> None:
    Ra = gimbal_to_R(0.0, -90.0, 0.0)
    Rb = gimbal_to_R(0.0, -90.0, 0.0)
    # 20 deg about the (vertical) optical axis: a heading error, not a tilt error.
    yawed = Rotation.from_euler("z", 20.0, degrees=True).as_matrix() @ (Rb.T @ Ra)
    ok, _ = gravity_consistent(yawed, Ra, Rb)
    assert ok
