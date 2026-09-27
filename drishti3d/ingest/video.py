"""Video decoding for the ingest stage.

We use PyAV (``av``) rather than ``cv2.VideoCapture``. cv2's FFmpeg backend
reports frame counts and rotation unreliably (and sometimes not at all) on
long, variable-frame-rate 4K drone footage, and it exposes no clean way to
get an accurate per-frame presentation timestamp. Reconstruction downstream
needs both of those to be right: telemetry alignment depends on accurate
per-frame timestamps (not ``index / fps``, since drone footage commonly has
small VFR jitter), and self-calibration/triage assume upright frames.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Self

import av
import cv2
import numpy as np

from drishti3d.types import Frame

logger = logging.getLogger(__name__)

# How far back (in seconds) we seek before a random-access target so that we
# land on/before the keyframe that precedes it. Generous enough for typical
# drone-footage GOP lengths (often 1-2s at consumer bitrates) without making
# every seek decode a large chunk of the file.
_SEEK_MARGIN_SEC = 2.0

# If the next requested random-access index is within this many frames of
# where our decoder already is, keep decoding forward instead of re-seeking;
# a re-seek plus GOP replay is usually more expensive than a short skip.
_MAX_FORWARD_SKIP_FRAMES = 60

# PyAV's ``VideoFrame.rotation`` is the CCW angle (degrees, in [-180, 180])
# needed to display the coded frame upright. cv2.rotate's constants rotate
# in the same geometric senses; ROTATE_90_COUNTERCLOCKWISE for +90 CCW, etc.
_ROTATION_TO_CV2 = {
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    -270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    -90: cv2.ROTATE_90_CLOCKWISE,
    270: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    -180: cv2.ROTATE_180,
}


class LazyFrame:
    """A decoded frame whose pixels are converted only when asked for (see ``VideoSource.iter_frames_lazy``)."""

    __slots__ = ("index", "timestamp", "_frame", "_source")

    def __init__(self, index: int, timestamp: float, frame: av.VideoFrame, source: VideoSource) -> None:
        self.index = index
        self.timestamp = timestamp
        self._frame = frame
        self._source = source

    def small(self, downscale: int | None) -> Frame:
        """The same ``Frame`` ``iter_frames(downscale=...)`` would have yielded for this index."""
        image = self._source._to_upright_bgr(self._frame)
        if downscale is not None:
            image = downscale_image(image, downscale)
        return Frame(index=self.index, timestamp=self.timestamp, image=image)

    def bgr24(self) -> np.ndarray:
        """Native-resolution BGR exactly as ``pipeline.framecache`` decodes it."""
        return self._frame.to_ndarray(format="bgr24")


def downscale_image(image: np.ndarray, max_long_side: int) -> np.ndarray:
    """Shrink (never enlarge) ``image`` so its longer side is <= ``max_long_side``.

    Uses ``cv2.INTER_AREA``, which anti-aliases while shrinking. Triage
    metrics computed downstream (blur, optical-flow parallax) are sensitive
    to aliasing that a naive nearest/bilinear resize would introduce, so the
    cheap-but-correct resampling matters even though this runs on a hot path.
    """
    if max_long_side <= 0:
        return image
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class VideoSource:
    """Context-managed reader over a single video file, backed by PyAV.

    Frames are returned upright (container/stream rotation applied) and as
    BGR uint8 arrays, matching the ``Frame`` contract in ``drishti3d.types``.
    ``width``/``height`` report the *upright* (post-rotation) dimensions,
    since that's what every downstream consumer (intrinsics, triage, GUI
    preview) actually wants to reason about.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._container = av.open(str(self.path))
        video_streams = self._container.streams.video
        if not video_streams:
            raise ValueError(f"No video stream found in {self.path}")
        self._stream = video_streams[0]
        # Decode with all available cores; drone footage is often 4K and a
        # single-threaded decode would dominate the triage pass's runtime.
        self._stream.thread_type = "AUTO"
        self._time_base: Fraction = self._stream.time_base or Fraction(1, 1)

        coded_w = int(self._stream.codec_context.width)
        coded_h = int(self._stream.codec_context.height)

        self._rotation = self._detect_rotation()
        if self._rotation in (90, -90, 270, -270):
            self._width, self._height = coded_h, coded_w
        else:
            self._width, self._height = coded_w, coded_h

        rate = self._stream.average_rate
        self._fps = float(rate) if rate else 0.0

        self._duration = self._detect_duration()
        self._frame_count = self._detect_frame_count()
        self._undistort_maps: tuple[np.ndarray, np.ndarray] | None = None

        # The rotation probe above may have decoded a frame; rewind so the
        # first real call to iter_frames()/read_frames() starts from zero.
        self._container.seek(0, stream=self._stream, backward=True)

    def set_undistortion(self, K: np.ndarray | None, dist_coeffs: np.ndarray | None) -> None:
        """Remove lens distortion from every frame decoded from now on.

        ``K`` is the 3x3 camera matrix at the upright native resolution and
        stays the camera matrix of the undistorted frames (same size, same
        principal point), so downstream pinhole code keeps ``K`` and drops
        ``dist_coeffs``. ``None`` switches undistortion off.
        """
        if K is None or dist_coeffs is None or not np.any(np.asarray(dist_coeffs)):
            self._undistort_maps = None
            return
        K = np.asarray(K, dtype=np.float64)
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        self._undistort_maps = cv2.initUndistortRectifyMap(K, dist, None, K, (self._width, self._height), cv2.CV_16SC2)

    @property
    def undistorting(self) -> bool:
        return self._undistort_maps is not None

    # -- construction-time probing --------------------------------------

    def _detect_rotation(self) -> int:
        """Return the upright-correction rotation in degrees: one of 0/90/180/270.

        Some containers only surface the display-matrix rotation on decoded
        frames (not on stream-level metadata), so we decode a single frame
        to find out for certain rather than guessing from tags.
        """
        try:
            for frame in self._container.decode(self._stream):
                raw = int(frame.rotation) % 360
                if raw not in (0, 90, 180, 270):
                    # A non-axis-aligned rotation can't be fixed by a
                    # discrete cv2.rotate; leave the frame as coded rather
                    # than mangling it with a wrong guess.
                    logger.warning("Non-axis-aligned rotation (%d deg) in %s; ignoring", raw, self.path)
                    return 0
                return raw
        except Exception:
            logger.warning("Could not probe rotation for %s; assuming 0", self.path, exc_info=True)
        return 0

    def _detect_duration(self) -> float:
        if self._stream.duration is not None:
            return float(self._stream.duration * self._time_base)
        if self._container.duration is not None:
            return float(self._container.duration) / 1_000_000.0
        return 0.0

    def _detect_frame_count(self) -> int:
        if self._stream.frames:
            return int(self._stream.frames)
        if self._duration > 0 and self._fps > 0:
            # Best-effort estimate for containers that don't carry a frame
            # count (common for streamed/edited footage); callers should
            # treat this as approximate, per the class docstring.
            return max(1, round(self._duration * self._fps))
        return 0

    # -- properties -------------------------------------------------------

    @property
    def width(self) -> int:
        """Upright frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Upright frame height in pixels."""
        return self._height

    @property
    def fps(self) -> float:
        """Nominal (average) frame rate; footage may still be variable frame rate."""
        return self._fps

    @property
    def duration(self) -> float:
        """Stream duration in seconds (0.0 if unknown)."""
        return self._duration

    @property
    def frame_count(self) -> int:
        """Total frame count; may be estimated from duration * fps if the container omits it."""
        return self._frame_count

    @property
    def rotation(self) -> int:
        """Upright-correction rotation in degrees (CCW), one of 0/90/180/270."""
        return self._rotation

    @property
    def metadata(self) -> dict[str, str]:
        """Best-effort container + stream metadata tags (device model, encoder, etc).

        Not part of the minimal contract, but cheap to expose; used by
        ``ingest.intrinsics`` to opportunistically recognize the camera
        model or an embedded focal length before falling back to a generic
        HFOV guess.
        """
        merged: dict[str, str] = {}
        merged.update(dict(self._container.metadata or {}))
        merged.update(dict(self._stream.metadata or {}))
        return merged

    # -- decoding -----------------------------------------------------------

    def _to_upright_bgr(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        code = _ROTATION_TO_CV2.get(self._rotation)
        if code is not None:
            image = cv2.rotate(image, code)
        if self._undistort_maps is not None and image.shape[:2] == (self._height, self._width):
            image = cv2.remap(image, *self._undistort_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return image

    def _frame_timestamp(self, frame: av.VideoFrame, decode_index: int) -> float:
        """Derive a frame's timestamp from PTS * time_base, never from index / fps.

        Drone footage frequently has small frame-rate jitter; assuming a
        constant frame period would silently desynchronize telemetry
        alignment over a multi-minute flight.
        """
        if frame.pts is not None:
            return float(frame.pts * self._time_base)
        if frame.time is not None:
            return float(frame.time)
        if self._fps > 0:
            return decode_index / self._fps
        return float(decode_index)

    def iter_frames(
        self,
        step: int = 1,
        max_frames: int | None = None,
        downscale: int | None = None,
    ) -> Iterator[Frame]:
        """Stream-decode frames from the start of the video.

        Parameters
        ----------
        step:
            Yield every ``step``-th decoded frame (decode-order index).
            Frames are still decoded in between (PyAV has no way to skip
            decoding without losing inter-frame references), but skipping
            the expensive per-frame work (rotation copy, resize, and
            whatever the caller does with the yielded frame) is still a
            large saving for a coarse triage scan.
        max_frames:
            Stop after yielding this many frames (not this many decoded
            frames).
        downscale:
            If given, the max-long-side in pixels frames are resized to
            (never enlarged). Triage works on small grayscale images, so
            downscaling here avoids decoding-then-immediately-shrinking
            full 4K frames through Python.
        """
        if step < 1:
            raise ValueError("step must be >= 1")

        self._container.seek(0, stream=self._stream, backward=True)
        yielded = 0
        for decode_index, frame in enumerate(self._container.decode(self._stream)):
            if decode_index % step != 0:
                continue
            if max_frames is not None and yielded >= max_frames:
                break

            image = self._to_upright_bgr(frame)
            if downscale is not None:
                image = downscale_image(image, downscale)
            timestamp = self._frame_timestamp(frame, decode_index)

            yield Frame(index=decode_index, timestamp=timestamp, image=image)
            yielded += 1

    def iter_frames_lazy(self) -> Iterator[LazyFrame]:
        """Stream-decode every frame, deferring all pixel work until a caller asks for it.

        Decoding cannot be skipped (inter-frame references), but the colour
        conversion, rotation and resize that ``iter_frames`` does for every
        frame can: a GPS-driven keyframe scan only needs pixels for the few
        frames around each trigger. Each ``LazyFrame`` holds its decoded
        ``av.VideoFrame`` and converts on demand.
        """
        self._container.seek(0, stream=self._stream, backward=True)
        for decode_index, frame in enumerate(self._container.decode(self._stream)):
            yield LazyFrame(decode_index, self._frame_timestamp(frame, decode_index), frame, self)

    def _estimate_decode_index(self, frame: av.VideoFrame, fps: float, previous: int | None) -> int:
        """Estimate a decoded frame's position in the original decode-order sequence.

        Assumes an approximately constant frame rate, which is the only way
        to convert a post-seek PTS back into "the Nth frame decoded from the
        start" without having decoded from the start. For true VFR sources
        this is an approximation (documented on ``read_frames``); it is
        exact for the common CFR case.
        """
        if frame.pts is not None:
            return round(float(frame.pts * self._time_base) * fps)
        if previous is not None:
            return previous + 1
        return 0

    def read_frames(self, indices: list[int]) -> list[Frame]:
        """Random access by decode-order frame index, seeking instead of decoding from zero.

        Note: "frame index" is decode-order position (0, 1, 2, ...), matching
        ``iter_frames``'s numbering. Efficient random access to an arbitrary
        decode-order index fundamentally requires *some* assumption about
        frame timing to convert index -> PTS for seeking; we assume constant
        frame rate (accurate for CFR footage, approximate for VFR). This
        trades a small amount of precision on VFR sources for avoiding a
        full linear decode, which is the whole point of random access.
        """
        if not indices:
            return []

        fps = self._fps if self._fps > 0 else 1.0
        unique_sorted = sorted(set(indices))
        results: dict[int, Frame] = {}

        decoder: Iterator[av.VideoFrame] | None = None
        current_est: int | None = None

        for target in unique_sorted:
            need_seek = decoder is None or current_est is None or not (0 <= target - current_est <= _MAX_FORWARD_SKIP_FRAMES)
            if need_seek:
                target_time = target / fps
                seek_time = max(0.0, target_time - _SEEK_MARGIN_SEC)
                offset = int(seek_time / self._time_base)
                self._container.seek(offset, stream=self._stream, backward=True, any_frame=False)
                decoder = self._container.decode(self._stream)
                current_est = None

            found: av.VideoFrame | None = None
            assert decoder is not None
            for candidate in decoder:
                current_est = self._estimate_decode_index(candidate, fps, current_est)
                if current_est >= target:
                    found = candidate
                    break

            if found is None:
                # Ran off the end of the stream looking for this index.
                continue

            image = self._to_upright_bgr(found)
            timestamp = self._frame_timestamp(found, target)
            results[target] = Frame(index=target, timestamp=timestamp, image=image)

        # Leave the container in a predictable state for a subsequent
        # iter_frames() call.
        self._container.seek(0, stream=self._stream, backward=True)

        return [results[i] for i in indices if i in results]

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the underlying container's resources."""
        self._container.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
