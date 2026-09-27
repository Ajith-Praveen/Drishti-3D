"""Learned features (DISK) + LightGlue matching, and a gravity check on verified pairs.

Why
---
SIFT ratio-test matching on video frames starves the bundle adjustment on
exactly the frames that matter: motion blur, low texture and the large
viewpoint change between the ends of a long track. LightGlue (Lindenberger
et al., ICCV 2023) matches learned DISK descriptors with attention over
both images and returns far more correct correspondences on such pairs
(see UAVD4L, 3DV 2024, Table 3: learned matching leads its UAV benchmark).
More tracks mean a better-conditioned bundle adjustment and more points for
the per-view depth fit (``geometry.depth_fit``).

Both run through ``kornia`` on the best available torch device. Weights
are downloaded once by kornia and cached; nothing here needs network at
run time after that.

Gravity check
-------------
UAVD4L's gravity-guided PnP RANSAC rejects pose hypotheses whose "down"
disagrees with the IMU. The same idea applies to a pair: the essential
matrix gives a relative rotation ``R_ab``, and the gimbal/IMU gives each
camera's gravity direction. ``R_ab`` must carry camera a's gravity onto
camera b's; a pair where it misses by more than a few degrees was solved
on wrong matches (repeated texture, a moving object) and is dropped
before it can poison a track. Only tilt is compared -- yaw from telemetry
is not trusted enough to veto anything.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from drishti3d.geometry.features import Features, Matches

logger = logging.getLogger(__name__)

__all__ = ["detect_disk", "LightGlueMatcher", "gravity_consistent", "is_available"]


def is_available() -> bool:
    """Whether DISK + LightGlue can run (``kornia``, and with it torch, importable).

    Ask this before choosing the DISK path. kornia is only imported inside
    ``_disk()`` / ``_lightglue()``, so importing THIS module succeeds
    without it and the missing package surfaces later, in ``detect_disk``.
    """
    try:
        import kornia.feature  # noqa: F401
    except ImportError:
        return False
    return True


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _disk():
    import kornia.feature as KF

    return KF.DISK.from_pretrained("depth").to(_device()).eval()


@lru_cache(maxsize=1)
def _lightglue():
    import kornia.feature as KF

    return KF.LightGlueMatcher("disk").to(_device()).eval()


def detect_disk(gray_or_bgr: np.ndarray, max_features: int = 2048, detect_scale: float = 1.0) -> Features:
    """DISK keypoints + 128-d descriptors, in the ORIGINAL image's pixel frame."""
    import cv2
    import torch

    img = gray_or_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if detect_scale != 1.0:
        img = cv2.resize(img, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    # DISK's U-Net needs sides divisible by 16.
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).to(_device())
    with torch.inference_mode():
        feats = _disk()(t, n=max_features, window_size=5, score_threshold=0.0, pad_if_not_divisible=False)[0]
    kp = feats.keypoints.detach().cpu().numpy().astype(np.float64)
    keep = (kp[:, 0] < w) & (kp[:, 1] < h)
    kp = kp[keep] / detect_scale
    desc = feats.descriptors.detach().cpu().numpy().astype(np.float32)[keep]
    scores = feats.detection_scores.detach().cpu().numpy().astype(np.float64)[keep]
    f = Features(keypoints=kp, descriptors=desc, scores=scores, method="disk")
    f.image_hw = (int(round(h / detect_scale)), int(round(w / detect_scale)))
    return f


class LightGlueMatcher:
    """``features.Matcher`` implementation: LightGlue over DISK features."""

    def match(self, fa: Features, fb: Features) -> Matches:
        import kornia.feature as KF
        import torch

        empty = Matches(query_idx=np.zeros(0), train_idx=np.zeros(0), distances=np.zeros(0))
        if len(fa) < 8 or len(fb) < 8 or fa.descriptors is None or fb.descriptors is None:
            return empty
        dev = _device()

        def _laf(f):
            kp = torch.from_numpy(f.keypoints).float()[None].to(dev)
            return KF.laf_from_center_scale_ori(kp, torch.ones(1, kp.shape[1], 1, 1, device=dev))

        hw_a = torch.tensor(getattr(fa, "image_hw", (2048, 2048)), device=dev)
        hw_b = torch.tensor(getattr(fb, "image_hw", (2048, 2048)), device=dev)
        with torch.inference_mode():
            dists, idxs = _lightglue()(
                torch.from_numpy(fa.descriptors).to(dev),
                torch.from_numpy(fb.descriptors).to(dev),
                _laf(fa),
                _laf(fb),
                hw1=hw_a,
                hw2=hw_b,
            )
        idxs = idxs.detach().cpu().numpy()
        if idxs.size == 0:
            return empty
        return Matches(
            query_idx=idxs[:, 0],
            train_idx=idxs[:, 1],
            distances=dists.detach().cpu().numpy().reshape(-1).astype(np.float64),
        )


def gravity_consistent(
    relative_R: np.ndarray, R_world_from_a: np.ndarray, R_world_from_b: np.ndarray, max_deg: float = 3.0
) -> tuple[bool, float]:
    """Does ``relative_R`` (a's frame -> b's frame, OpenCV) carry a's gravity onto b's?

    Returns ``(ok, error_deg)``. Up is +Z in the ENU world frame.
    """
    up = np.array([0.0, 0.0, 1.0])
    g_a = np.asarray(R_world_from_a).T @ up
    g_b = np.asarray(R_world_from_b).T @ up
    pred = np.asarray(relative_R) @ g_a
    cos = float(np.clip(pred @ g_b / (np.linalg.norm(pred) * np.linalg.norm(g_b)), -1.0, 1.0))
    err = float(np.degrees(np.arccos(cos)))
    return err <= max_deg, err
