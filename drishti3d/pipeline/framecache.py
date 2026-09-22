"""Decode each keyframe once and share it across every stage that needs it.

The cost being removed
----------------------
Reaching keyframe 7,212 in a 4K H.264 file means the decoder walks the
whole GOP chain to get there. Measured on the sample flight: **144 s to
pull 77 keyframes, 1.87 s per frame** -- and the pipeline was paying that
repeatedly:

- ``_refine_yaw_from_flow`` decodes all 77 (then throws them away at 640 px)
- ``MatchingStage`` decodes all 77 again, at full resolution
- ``GeometryStage._decode_window`` decodes them a third time, window by window
- the second bundle-adjustment pass decoded them a fourth time

That is ~290 s of a ~600 s ``pose_prior`` stage spent re-decoding bytes it
already had, before any of the actual work.

Why a resolution cap
--------------------
77 frames of 4K BGR is 1.9 GB resident. Every consumer here downscales
immediately -- geometry to 518-956 px, matching to 1920 px, yaw to 640 px
-- so the cache stores at ``max_size`` (default 1920, ~480 MB) and each
consumer scales down from there. Intrinsics are scaled by the *exact*
ratio the image was scaled by, never re-derived from rounded integer
sizes, which is the drift ``resize_preserving_aspect`` exists to avoid.

Texture baking is deliberately NOT a consumer: it needs the full 4K pixels
to bake a sharp atlas, and it reads them itself. Serving it downscaled
frames would quietly degrade the one deliverable whose whole point is
resolution.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["KeyframeImageCache", "build_keyframe_cache"]

#: Longest side the cache stores. Above every consumer's working
#: resolution (matching at 1920 is the highest), below the 1.9 GB that
#: full 4K would cost.
_DEFAULT_MAX_SIZE = 1920


class KeyframeImageCache:
    """Decoded keyframe images, keyed by position in ``state.keyframes``."""

    def __init__(self, images: dict[int, np.ndarray], max_size: int, scales: dict[int, float]) -> None:
        self._images = images
        self.max_size = max_size
        #: Per keyframe: stored_size / original_size. A consumer that needs
        #: intrinsics for a cached image must scale them by this.
        self.scales = scales

    def __len__(self) -> int:
        return len(self._images)

    def get(self, index: int) -> np.ndarray | None:
        """The cached image for keyframe ``index``, or ``None`` if absent."""
        return self._images.get(index)

    def scale_for(self, index: int) -> float:
        return self.scales.get(index, 1.0)

    def nbytes(self) -> int:
        return sum(im.nbytes for im in self._images.values())


def build_keyframe_cache(state, max_size: int = _DEFAULT_MAX_SIZE) -> KeyframeImageCache | None:
    """Decode every keyframe once, in one sequential pass, and cache it.

    One pass in decode order, not 77 random seeks: the frames are spread
    across the whole file, so a single ordered walk costs one traversal
    instead of one per frame.

    Returns ``None`` (and logs) when there is no video or no keyframe
    decodes -- callers fall back to their own ``read_frames`` path, which
    is slower but correct.
    """
    from drishti3d.geometry.mapanything import resize_preserving_aspect

    if state.video is None or not state.keyframes:
        return None

    wanted: dict[int, list[int]] = {}
    for pos, kf in enumerate(state.keyframes):
        wanted.setdefault(int(kf.frame_index), []).append(pos)

    images: dict[int, np.ndarray] = {}
    scales: dict[int, float] = {}
    try:
        import av

        container = av.open(str(state.video_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        remaining = set(wanted)
        for i, frame in enumerate(container.decode(stream)):
            if i not in remaining:
                continue
            bgr = frame.to_ndarray(format="bgr24")
            resized, scale = resize_preserving_aspect(bgr, max_size) if max(bgr.shape[:2]) > max_size else (bgr, 1.0)
            for pos in wanted[i]:
                images[pos] = resized
                scales[pos] = scale
            remaining.discard(i)
            if not remaining:
                break
        container.close()
    except Exception:
        logger.warning("keyframe cache: decode failed; stages will read frames individually", exc_info=True)
        return None

    if not images:
        return None

    cache = KeyframeImageCache(images, max_size, scales)
    logger.info(
        "keyframe cache: %d/%d keyframes decoded once at <=%dpx (%.0f MB); "
        "yaw refinement, matching and geometry now share this instead of re-decoding",
        len(images),
        len(state.keyframes),
        max_size,
        cache.nbytes() / 1e6,
    )
    return cache
