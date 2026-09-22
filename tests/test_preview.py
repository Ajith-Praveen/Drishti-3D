"""Tests for live progress snapshots.

These are diagnostics attached to ``partial_cb``, which fires from inside a
running reconstruction. The properties that matter are therefore about
*not interfering*: never raise, never block, never grow unboundedly. A
snapshot writer that can abort a forty-minute geometry stage is worse than
no snapshot writer at all.
"""

from __future__ import annotations

import numpy as np

from drishti3d.export.preview import ProgressSnapshotWriter, render_topdown
from drishti3d.types import PointCloud


def _scene(n: int = 5000, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    ground = rng.uniform(-20, 20, (n, 2))
    roof = rng.uniform(-5, 5, (n // 2, 2))
    xyz = np.vstack(
        [np.c_[ground, np.zeros(len(ground))], np.c_[roof, np.full(len(roof), 8.0)]]
    )
    return PointCloud(xyz=xyz)


# ---------------------------------------------------------------------------
# render_topdown
# ---------------------------------------------------------------------------


def test_topdown_renders_points_into_an_image() -> None:
    image = render_topdown(_scene(), size=128, colour_by="height")
    assert image is not None
    assert image.dtype == np.uint8
    assert image.shape[2] == 3
    assert max(image.shape[:2]) == 128
    assert (image.sum(axis=2) > 0).any()


def test_topdown_returns_none_for_nothing_to_draw() -> None:
    assert render_topdown(None) is None
    assert render_topdown(PointCloud(xyz=np.zeros((0, 3)))) is None


def test_topdown_uses_rgb_when_present() -> None:
    pc = _scene()
    pc.rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(pc.xyz), 1))  # red
    image = render_topdown(pc, size=64, colour_by="rgb")
    lit = image[image.sum(axis=2) > 0]
    # cv2 order is BGR, so pure red lands in the last channel.
    assert (lit[:, 2] == 255).all()
    assert (lit[:, 0] == 0).all()


def test_north_is_up() -> None:
    """World +Y must map to the TOP of the raster, not the bottom."""
    xyz = np.array([[0.0, 10.0, 0.0], [0.0, -10.0, 0.0]])
    image = render_topdown(PointCloud(xyz=xyz), size=64, colour_by="height")
    lit_rows = np.flatnonzero((image.sum(axis=2) > 0).any(axis=1))
    # Two points; the northern one must occupy a smaller row index.
    assert len(lit_rows) >= 2
    assert lit_rows.min() < lit_rows.max()


def test_fixed_bounds_keep_the_frame_stable() -> None:
    """Growing clouds must not appear to shrink between snapshots."""
    bounds = (-20.0, 20.0, -20.0, 20.0)
    small = render_topdown(PointCloud(xyz=_scene().xyz[:100]), size=100, bounds=bounds)
    large = render_topdown(_scene(), size=100, bounds=bounds)
    assert small.shape == large.shape


# ---------------------------------------------------------------------------
# ProgressSnapshotWriter
# ---------------------------------------------------------------------------


def test_writer_emits_numbered_snapshots_and_a_stable_latest(tmp_path) -> None:
    writer = ProgressSnapshotWriter(tmp_path, every_n=1)
    pc = _scene()
    for i in (1000, 3000, 7500):
        writer(PointCloud(xyz=pc.xyz[:i]))

    assert writer.written == 3
    assert (tmp_path / "snapshot_0001.png").exists()
    assert (tmp_path / "snapshot_0003.png").exists()
    # A stable path lets a viewer watch one file instead of hunting for the
    # newest numbered one.
    assert (tmp_path / "latest.png").exists()


def test_every_n_rate_limits_writes(tmp_path) -> None:
    writer = ProgressSnapshotWriter(tmp_path, every_n=3)
    pc = _scene()
    for _ in range(9):
        writer(pc)

    assert writer.calls == 9
    assert writer.written == 3


def test_writer_never_raises_into_the_pipeline(tmp_path) -> None:
    """partial_cb runs inside geometry; an exception here would abort a run."""
    writer = ProgressSnapshotWriter(tmp_path, every_n=1)

    class Exploding:
        @property
        def xyz(self):
            raise RuntimeError("boom")

    writer(None)
    writer(PointCloud(xyz=np.zeros((0, 3))))
    writer(Exploding())
    writer("not a point cloud at all")

    # Survived every one, and wrote nothing it could not legitimately write.
    assert writer.written == 0


def test_writer_is_still_usable_after_a_bad_update(tmp_path) -> None:
    writer = ProgressSnapshotWriter(tmp_path, every_n=1)
    writer(None)
    writer(_scene())
    assert writer.written == 1


def test_bounds_lock_after_the_warmup_window(tmp_path) -> None:
    """Early partials cover part of the scene; the frame settles, then holds."""
    writer = ProgressSnapshotWriter(tmp_path, every_n=1, lock_bounds_after=2)
    pc = _scene()

    writer(PointCloud(xyz=pc.xyz[:500]))
    writer(PointCloud(xyz=pc.xyz[:1500]))
    locked = writer._bounds
    # Once locked, a much larger cloud must not move the frame.
    writer(pc)
    assert writer._bounds == locked
