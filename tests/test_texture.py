"""Tests for drishti3d.fusion.texture and the textured-OBJ writer.

Split by dependency: the writer and the geometry helpers are tested
unconditionally, while the bake itself is skipped when ``xatlas`` is
absent. That split matters -- "the mesh still exports correctly without
xatlas" is itself a behaviour this project relies on, so the fallback path
is asserted rather than merely assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.export.formats import export_obj_textured
from drishti3d.fusion.texture import (
    _rasterize_attribute,
    _vertex_normals,
    bake_texture,
    is_available,
)
from drishti3d.types import CameraIntrinsics, Pose

_NADIR_R = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])

requires_xatlas = pytest.mark.skipif(not is_available(), reason="xatlas not installed")


def _grid_mesh(n: int = 12, half: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """A flat n x n ground-plane mesh at z=0, spanning +-half metres."""
    axis = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(axis, axis)
    vertices = np.stack([gx.ravel(), gy.ravel(), np.zeros(n * n)], axis=1)

    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return vertices, np.array(faces, dtype=np.int64)


def _channel_view(channel: int, x: float, *, alt: float = 25.0):
    """A nadir view whose image saturates exactly one BGR channel."""
    intr = CameraIntrinsics.from_hfov(70.0, 320, 240)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, channel] = 255
    return Pose(R=_NADIR_R.copy(), t=np.array([x, 0.0, alt])), intr, image, 100.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_vertex_normals_of_a_flat_plane_point_up() -> None:
    vertices, faces = _grid_mesh(n=5)
    normals = _vertex_normals(vertices, faces)
    assert normals.shape == vertices.shape
    assert np.allclose(np.abs(normals[:, 2]), 1.0)
    assert np.allclose(normals[:, :2], 0.0, atol=1e-9)


def test_vertex_normals_are_unit_length() -> None:
    vertices, faces = _grid_mesh(n=6)
    normals = _vertex_normals(vertices, faces)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_rasterize_attribute_covers_a_full_unit_square_uv() -> None:
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])

    image, mask = _rasterize_attribute(vertices, faces, uv, 16)

    assert image.shape == (16, 16, 3)
    assert mask.shape == (16, 16)
    assert mask.all()


def test_rasterize_attribute_leaves_uncovered_texels_masked_out() -> None:
    """Half a UV square covered means roughly half the texels covered."""
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    uv = np.array([[0.0, 0], [1, 0], [1, 1]])

    _, mask = _rasterize_attribute(vertices, faces, uv, 64)

    covered = mask.mean()
    assert 0.35 < covered < 0.65
    assert not mask.all()


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_bake_returns_none_for_empty_input() -> None:
    assert bake_texture(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), []) is None
    vertices, faces = _grid_mesh(n=4)
    # No views at all is a "cannot texture", not an error.
    assert bake_texture(vertices, faces, []) is None


# ---------------------------------------------------------------------------
# The real bake
# ---------------------------------------------------------------------------


@requires_xatlas
def test_bake_produces_a_covered_atlas_and_consistent_topology() -> None:
    vertices, faces = _grid_mesh(n=12)
    views = [_channel_view(k, x) for k, x in enumerate((-3.0, 0.0, 3.0))]

    result = bake_texture(vertices, faces, views, texture_size=256)

    assert result is not None
    # Unwrapping may split vertices; UVs must match the unwrapped count,
    # never the original one.
    assert len(result.uv) == len(result.vertices)
    assert result.faces.max() < len(result.vertices)
    assert result.vertex_map.max() < len(vertices)
    assert result.uv.min() >= 0.0 and result.uv.max() <= 1.0
    assert result.texture.shape == (256, 256, 3)
    assert result.stats["texel_coverage_pct"] > 90.0
    assert result.stats["views_used"] == 3


@requires_xatlas
def test_bake_blends_views_rather_than_picking_a_single_winner() -> None:
    """Three views, one saturated BGR channel each, must average, not argmax.

    Each view contributes 255 in a different channel, so an equal blend
    lands near 255/3 in every channel. A hard argmax would instead produce
    a saturated single channel -- which is exactly the seam-producing
    behaviour `blend_views` exists to avoid.
    """
    vertices, faces = _grid_mesh(n=12)
    views = [_channel_view(k, x) for k, x in enumerate((-3.0, 0.0, 3.0))]

    result = bake_texture(vertices, faces, views, texture_size=256, blend_views=4)

    assert result is not None
    covered = result.texture[result.texture.sum(axis=2) > 0]
    mean_rgb = covered.mean(axis=0)
    assert np.allclose(mean_rgb, 255.0 / 3.0, atol=8.0)


@requires_xatlas
def test_bake_vertex_map_reindexes_attributes_onto_unwrapped_geometry() -> None:
    """`vertices[vertex_map]` must reproduce the unwrapped positions exactly."""
    vertices, faces = _grid_mesh(n=10)
    views = [_channel_view(0, 0.0)]

    result = bake_texture(vertices, faces, views, texture_size=128)

    assert result is not None
    assert np.allclose(vertices[result.vertex_map], result.vertices)


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


def test_export_obj_textured_writes_obj_mtl_and_png(tmp_path) -> None:
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    texture = np.full((32, 32, 3), 200, dtype=np.uint8)

    paths = export_obj_textured(tmp_path / "m.obj", vertices, faces, uv, texture)

    assert set(paths) == {"obj_textured", "mtl", "texture_png"}
    for path in paths.values():
        assert path.exists()

    obj = paths["obj_textured"].read_text()
    # A real texture reference, not a flat placeholder material.
    assert "mtllib m.mtl" in obj
    assert "usemtl drishti3d_texture" in obj
    assert obj.count("\nvt ") == 4
    # Faces must carry texture indices (v/vt), not bare vertex indices.
    assert "f 1/1 2/2 3/3" in obj

    mtl = paths["mtl"].read_text()
    assert "map_Kd m.png" in mtl


def test_export_obj_textured_rejects_mismatched_uv_count(tmp_path) -> None:
    """Pairing original faces with unwrapped UVs scrambles the texture."""
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2]])
    uv = np.array([[0.0, 0], [1, 0]])  # one short

    with pytest.raises(ValueError, match="uv has 2 entries"):
        export_obj_textured(tmp_path / "m.obj", vertices, faces, uv, np.zeros((8, 8, 3), np.uint8))


def test_exported_texture_png_round_trips_as_rgb(tmp_path) -> None:
    """`texture` is RGB by contract; the PNG on disk must match it."""
    import cv2

    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2]])
    uv = np.array([[0.0, 0], [1, 0], [1, 1]])
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    texture[:, :, 0] = 250  # pure red in RGB

    paths = export_obj_textured(tmp_path / "m.obj", vertices, faces, uv, texture)
    reloaded_bgr = cv2.imread(str(paths["texture_png"]))

    # cv2 reads BGR, so pure-red RGB must come back in the *last* channel.
    assert reloaded_bgr[0, 0, 2] == 250
    assert reloaded_bgr[0, 0, 0] == 0
