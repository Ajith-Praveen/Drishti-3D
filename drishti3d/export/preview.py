"""Live progress snapshots: is the reconstruction actually growing?

The problem
-----------
Geometry is the longest stage, and on a long flight it runs for many
minutes emitting nothing but log lines. A stalled run and a working run
look identical from outside: both print a progress counter that may or may
not correspond to anything real. On a Kaggle session with a hard time
limit, discovering at the end that the last forty minutes produced garbage
is expensive.

``GeometryStage`` already merges every completed window into a partial
point cloud and hands it to ``partial_cb`` (that is how the desktop
viewport live-updates). Nothing in the headless path consumed it. This
module renders those partials to PNG so a CLI or notebook run leaves a
visible trail: ``snapshot_0001.png``, ``snapshot_0002.png``, ... each one
showing more of the scene than the last.

Why top-down
------------
An orthographic top-down view is the one projection that needs no camera,
no pose and no intrinsics -- so it cannot itself fail for the same reasons
the reconstruction might, and it stays comparable frame to frame as the
cloud grows. A perspective render would change viewpoint as the bounding
box moved, making two snapshots hard to compare at a glance, which is the
only thing these are for.

These are diagnostics, not deliverables. They are a fast point splat with
no occlusion handling and no georeferencing. Read geometry off the
exported LAS; read *liveness* off these.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ProgressSnapshotWriter", "render_topdown"]

#: Default longest-edge size for a snapshot. Big enough to see structure,
#: small enough that writing one costs milliseconds against a stage that
#: takes minutes.
_DEFAULT_SIZE = 720


def render_topdown(
    point_cloud,
    *,
    size: int = _DEFAULT_SIZE,
    colour_by: str = "auto",
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray | None:
    """Orthographic top-down raster of a point cloud. BGR uint8, or ``None``.

    ``colour_by``:
      - ``"rgb"``   -- the points' own colour, when present.
      - ``"height"`` -- turbo colormap over Z. Better for spotting a warped
        or bowl-shaped reconstruction, which true colour hides.
      - ``"auto"``  -- rgb when available, else height.

    ``bounds`` pins the extent across snapshots. Without it each frame
    auto-scales to its own cloud, so the scene appears to shrink as it
    grows -- the exact opposite of the impression these are meant to give.
    """
    if point_cloud is None or getattr(point_cloud, "xyz", None) is None:
        return None
    xyz = np.asarray(point_cloud.xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return None

    if bounds is None:
        xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
        xmax, ymax = xyz[:, 0].max(), xyz[:, 1].max()
    else:
        xmin, xmax, ymin, ymax = bounds

    span_x = max(float(xmax - xmin), 1e-6)
    span_y = max(float(ymax - ymin), 1e-6)
    scale = size / max(span_x, span_y)
    width = max(1, round(span_x * scale))
    height = max(1, round(span_y * scale))

    col = np.clip(((xyz[:, 0] - xmin) * scale).astype(np.int32), 0, width - 1)
    # Flip Y so north is up: raster rows increase downward, world +Y is north.
    row = np.clip((height - 1 - (xyz[:, 1] - ymin) * scale).astype(np.int32), 0, height - 1)

    rgb = getattr(point_cloud, "rgb", None)
    use_rgb = colour_by == "rgb" or (colour_by == "auto" and rgb is not None)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    if use_rgb and rgb is not None:
        # Painter's order: later points overwrite earlier ones. Cheap, and
        # for a liveness check the difference from a true z-buffer is
        # invisible.
        image[row, col] = np.asarray(rgb, dtype=np.uint8)[:, ::-1]  # RGB -> BGR
    else:
        z = xyz[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        if hi <= lo:
            hi = lo + 1e-3
        norm = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
        colours = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[row, col] = colours[:, 0, :]

    return image


class ProgressSnapshotWriter:
    """Writes a top-down PNG every ``every_n`` partial updates.

    Attach as ``run_pipeline(partial_cb=writer)``. Rate-limited because
    ``partial_cb`` fires once per completed window and a large flight has
    dozens -- writing every one would bury the useful frames and add I/O to
    the stage being measured.

    Never raises into the pipeline. A diagnostic that can abort a
    forty-minute reconstruction is worse than no diagnostic.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        every_n: int = 1,
        size: int = _DEFAULT_SIZE,
        colour_by: str = "auto",
        lock_bounds_after: int = 3,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.every_n = max(1, int(every_n))
        self.size = int(size)
        self.colour_by = colour_by
        #: Extent is allowed to grow for the first few snapshots, then
        #: pinned. Early partials cover a fraction of the scene, so locking
        #: immediately would frame the whole flight to the first window's
        #: bounding box.
        self.lock_bounds_after = int(lock_bounds_after)

        self.calls = 0
        self.written = 0
        self._bounds: tuple[float, float, float, float] | None = None
        self.paths: list[Path] = []

    def __call__(self, point_cloud) -> None:
        self.calls += 1
        if self.calls % self.every_n:
            return
        try:
            self._write(point_cloud)
        except Exception:
            logger.debug("preview: snapshot failed; continuing", exc_info=True)

    def _write(self, point_cloud) -> None:
        xyz = np.asarray(getattr(point_cloud, "xyz", np.zeros((0, 3))), dtype=np.float64)
        if xyz.shape[0] == 0:
            return

        if self.written < self.lock_bounds_after or self._bounds is None:
            self._bounds = (
                float(xyz[:, 0].min()),
                float(xyz[:, 0].max()),
                float(xyz[:, 1].min()),
                float(xyz[:, 1].max()),
            )

        image = render_topdown(point_cloud, size=self.size, colour_by=self.colour_by, bounds=self._bounds)
        if image is None:
            return

        self.written += 1
        label = f"update {self.calls}   {xyz.shape[0]:,} points"
        cv2.rectangle(image, (0, 0), (image.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(image, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        path = self.out_dir / f"snapshot_{self.written:04d}.png"
        cv2.imwrite(str(path), image)
        self.paths.append(path)
        # A stable filename too, so a viewer can watch one path rather than
        # hunting for the newest numbered file.
        cv2.imwrite(str(self.out_dir / "latest.png"), image)
        logger.info("preview: wrote %s (%d points)", path.name, xyz.shape[0])
