"""Tests for the shared data contract in drishti3d.types."""

import numpy as np

from drishti3d.types import CameraIntrinsics, Pose


def test_camera_intrinsics_K():
    intr = CameraIntrinsics(fx=1000.0, fy=1100.0, cx=640.0, cy=360.0, width=1280, height=720)
    K = intr.K()
    assert K.shape == (3, 3)
    expected = np.array(
        [
            [1000.0, 0.0, 640.0],
            [0.0, 1100.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(K, expected)


def test_camera_intrinsics_from_hfov():
    width, height = 1920, 1080
    hfov_deg = 90.0
    intr = CameraIntrinsics.from_hfov(hfov_deg, width, height)

    assert intr.width == width
    assert intr.height == height
    assert intr.cx == width / 2.0
    assert intr.cy == height / 2.0

    # For a 90 degree HFOV, fx = width / (2 * tan(45 deg)) = width / 2.
    expected_fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg / 2.0)))
    assert intr.fx == expected_fx
    assert intr.fy == intr.fx

    # Sanity check the projected HFOV matches what we asked for.
    recovered_hfov = 2.0 * np.degrees(np.arctan(width / (2.0 * intr.fx)))
    np.testing.assert_allclose(recovered_hfov, hfov_deg, atol=1e-6)


def test_pose_matrix():
    # 90 degree rotation about Z: world-from-camera.
    theta = np.pi / 2
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t = np.array([1.0, 2.0, 3.0])
    pose = Pose(R=R, t=t)

    M = pose.matrix()
    assert M.shape == (4, 4)
    np.testing.assert_allclose(M[:3, :3], R)
    np.testing.assert_allclose(M[:3, 3], t)
    np.testing.assert_allclose(M[3, :], [0.0, 0.0, 0.0, 1.0])


def test_pose_inverse_roundtrip():
    theta = 0.3
    axis_R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t = np.array([5.0, -2.0, 0.5])
    pose = Pose(R=axis_R, t=t)

    inv = pose.inverse()

    # inv.matrix() should equal the numerical inverse of pose.matrix().
    np.testing.assert_allclose(inv.matrix(), np.linalg.inv(pose.matrix()), atol=1e-10)

    # Inverting twice should recover the original pose.
    roundtrip = inv.inverse()
    np.testing.assert_allclose(roundtrip.R, pose.R, atol=1e-10)
    np.testing.assert_allclose(roundtrip.t, pose.t, atol=1e-10)
