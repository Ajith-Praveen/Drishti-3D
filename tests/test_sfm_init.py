"""Image-only camera solve (geometry.sfm_init) on a forward-looking, low-altitude flight with bad GPS."""

from __future__ import annotations

import numpy as np

from drishti3d.geometry.sfm_init import image_only_poses
from drishti3d.geometry.tracks import Track, TrackSet
from drishti3d.types import CameraIntrinsics, Pose

_K = CameraIntrinsics(fx=900.0, fy=900.0, cx=640.0, cy=360.0, width=1280, height=720)


def _rot_forward(yaw_deg: float) -> np.ndarray:
    """World-from-camera for a level camera looking along heading ``yaw_deg`` (0 = +x), z up."""
    a = np.radians(yaw_deg)
    fwd = np.array([np.cos(a), np.sin(a), 0.0])
    right = np.array([np.sin(a), -np.cos(a), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    return np.stack([right, down, fwd], axis=1)


def _scene(rng):
    # A vineyard-like corridor: ground, two rows of vines, and far background.
    g = rng.uniform([0, -6, 0], [45, 6, 0.05], (1500, 3))
    rows = np.concatenate([rng.uniform([0, s - 0.3, 0.2], [45, s + 0.3, 2.0], (1200, 3)) for s in (-2.5, 2.5)])
    far = rng.uniform([45, -15, 0], [60, 15, 12], (800, 3))
    return np.concatenate([g, rows, far])


def test_solves_a_forward_low_flight_the_gps_cannot_place():
    rng = np.random.default_rng(3)
    pts = _scene(rng)
    true = [Pose(R=_rot_forward(0.0), t=np.array([1.5 * k, 0.0, 1.0])) for k in range(12)]
    tracks = []
    for p_i, X in enumerate(pts):
        o = []
        for f, P in enumerate(true):
            Xc = P.R.T @ (X - P.t)
            if Xc[2] < 1.0:
                continue
            uv = np.array([_K.fx * Xc[0] / Xc[2] + _K.cx, _K.fy * Xc[1] / Xc[2] + _K.cy])
            if 0 <= uv[0] < _K.width and 0 <= uv[1] < _K.height:
                o.append((f, p_i, uv + rng.normal(0, 0.4, 2)))
        if len(o) >= 2:
            tracks.append(Track(observations=o))
    # The log: GPS off by ~2 m per fix, compass heading off by 8 degrees.
    seeds = [Pose(R=_rot_forward(8.0), t=P.t + rng.normal(0.0, 2.0, 3)) for P in true]
    res = image_only_poses(TrackSet(tracks=tracks), [_K] * len(true), seeds)
    assert res is not None and len(res.registered) >= 10, res and res.diagnostics
    C = np.array([res.poses[i].t for i in res.registered])
    T = np.array([true[i].t for i in res.registered])
    # Shape from the images: after a best-fit similarity the track is within centimetres.
    A, B = C - C.mean(0), T - T.mean(0)
    U, S, Vt = np.linalg.svd(B.T @ A)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A**2).sum()
    shape = np.linalg.norm(s * (R @ A.T).T - B, axis=1)
    assert np.median(shape) < 0.05, np.median(shape)
    # Placement: one similarity over the whole GPS track beats the raw fixes it came from
    # (with 2 m noise on a 16.5 m track the scale itself is only good to ~10%).
    raw = np.median([np.linalg.norm(seeds[i].t - true[i].t) for i in res.registered])
    assert np.median(np.linalg.norm(C - T, axis=1)) < raw, (np.linalg.norm(C - T, axis=1), raw)
    assert abs(res.diagnostics["heading_correction_deg"] + 8.0) <= 4.0, res.diagnostics
    fwd = np.array([res.poses[i].R[:, 2] for i in res.registered])
    assert np.degrees(np.arccos(np.clip(fwd @ np.array([1.0, 0.0, 0.0]), -1, 1))).max() < 5.0  # heading from a 16 m noisy track


def test_returns_none_without_enough_tracks():
    assert image_only_poses(TrackSet(tracks=[]), [_K] * 5, [None] * 5) is None


def test_flow_yaw_is_not_applied_to_a_forward_camera():
    """Image rotation is heading only for a nadir camera; a level camera keeps its logged heading."""
    from types import SimpleNamespace

    from drishti3d.pipeline.stages import _refine_yaw_from_flow

    pose = Pose(R=_rot_forward(261.0), t=np.zeros(3))
    kfs = [SimpleNamespace(pose=pose, frame_index=i, telemetry=SimpleNamespace(gimbal_pitch=0.0, gimbal_roll=0.0, gimbal_yaw=261.0))
           for i in range(5)]
    state = SimpleNamespace(video=object(), keyframe_cache=None)
    diag = _refine_yaw_from_flow(state, kfs, SimpleNamespace(yaw_from_flow=True))
    assert diag["enabled"] is False and "not nadir" in diag["failure"]
    assert all(k.pose is pose for k in kfs)
