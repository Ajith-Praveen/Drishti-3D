"""Tests for drishti3d.export: file formats, GeoTIFF rasterization, and the accuracy report.

Everything here must pass with no rasterio, no tifffile, no open3d, no
torch -- format round-trips use this package's own hand-rolled readers
(``read_ply``/``read_glb``) or ``laspy`` (an unconditional dependency), and
the GeoTIFF tests exercise the guaranteed-available ``.npy``/``.tfw``
fallback path.
"""

from __future__ import annotations

import laspy
import numpy as np
import pytest

from drishti3d.export.formats import (
    export_glb,
    export_las,
    export_obj,
    export_ply,
    export_xyz,
    read_glb,
    read_ply,
)
from drishti3d.export.geotiff import (
    point_cloud_to_confidence_raster,
    point_cloud_to_dsm,
    point_cloud_to_orthomosaic,
    write_geotiff,
)
from drishti3d.export.report import (
    NOT_COMPUTED,
    build_report,
    check_point_residuals,
    render_report_html,
    render_report_text,
)
from drishti3d.types import Confidence, GeoPoint, PointCloud

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_point_cloud(n: int = 64, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3))
    rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    confidence = rng.integers(0, 3, size=(n,)).astype(np.uint8)
    covariance = np.tile(np.eye(3) * 0.02, (n, 1, 1))
    return PointCloud(xyz=xyz, rgb=rgb, covariance=covariance, confidence=confidence)


def _make_mesh(n: int = 4):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]][:n]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]][:n], dtype=np.uint8)
    confidence = np.array([Confidence.MEASURED, Confidence.LOW_CONFIDENCE, Confidence.INFERRED, Confidence.MEASURED][:n], dtype=np.uint8)
    return vertices, faces, colors, confidence


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binary", [True, False])
def test_ply_point_cloud_roundtrip(tmp_path, binary):
    pc = _make_point_cloud()
    path = tmp_path / "cloud.ply"
    export_ply(path, pc, binary=binary)

    xyz, faces, rgb, confidence = read_ply(path)

    np.testing.assert_allclose(xyz, pc.xyz, atol=1e-3)
    assert faces is None
    np.testing.assert_array_equal(rgb, pc.rgb)
    np.testing.assert_array_equal(confidence, pc.confidence)


def test_ply_mesh_roundtrip(tmp_path):
    vertices, faces, colors, confidence = _make_mesh()
    path = tmp_path / "mesh.ply"
    export_ply(path, (vertices, faces, colors, confidence), binary=True)

    v2, f2, c2, conf2 = read_ply(path)
    np.testing.assert_allclose(v2, vertices, atol=1e-4)
    np.testing.assert_array_equal(f2, faces)
    np.testing.assert_array_equal(c2, colors)
    np.testing.assert_array_equal(conf2, confidence)


def test_ply_without_confidence_has_no_confidence_property(tmp_path):
    pc = PointCloud(xyz=np.random.default_rng(0).normal(size=(10, 3)))
    path = tmp_path / "noconf.ply"
    export_ply(path, pc, binary=True)
    _xyz, _faces, rgb, confidence = read_ply(path)
    assert rgb is None
    assert confidence is None


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def test_obj_export_writes_mtl_and_vertex_colors(tmp_path):
    vertices, faces, colors, _confidence = _make_mesh()
    path = tmp_path / "mesh.obj"
    export_obj(path, vertices, faces, colors=colors, mtl=True)

    assert path.exists()
    mtl_path = path.with_suffix(".mtl")
    assert mtl_path.exists()

    text = path.read_text()
    v_lines = [line for line in text.splitlines() if line.startswith("v ")]
    f_lines = [line for line in text.splitlines() if line.startswith("f ")]
    assert len(v_lines) == vertices.shape[0]
    assert len(f_lines) == faces.shape[0]
    # Vertex colour extension: "v x y z r g b".
    assert len(v_lines[0].split()) == 7
    # OBJ face indices are 1-based.
    first_face = [int(tok) for tok in f_lines[0].split()[1:]]
    assert min(first_face) >= 1


def test_obj_export_without_mtl(tmp_path):
    vertices, faces, _colors, _confidence = _make_mesh()
    path = tmp_path / "mesh_nomtl.obj"
    export_obj(path, vertices, faces, colors=None, mtl=False)
    assert path.exists()
    assert not path.with_suffix(".mtl").exists()


# ---------------------------------------------------------------------------
# GLB
# ---------------------------------------------------------------------------


def test_glb_roundtrip_with_confidence(tmp_path):
    vertices, faces, colors, confidence = _make_mesh()
    path = tmp_path / "mesh.glb"
    export_glb(path, vertices, faces, colors=colors, confidence=confidence)

    v2, f2, c2, conf2 = read_glb(path)
    np.testing.assert_allclose(v2, vertices, atol=1e-5)
    np.testing.assert_array_equal(f2, faces)
    np.testing.assert_array_equal(c2, colors)
    np.testing.assert_array_equal(conf2, confidence)


def test_glb_header_is_well_formed(tmp_path):
    import struct

    vertices, faces, colors, confidence = _make_mesh()
    path = tmp_path / "mesh2.glb"
    export_glb(path, vertices, faces, colors=colors, confidence=confidence)

    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67  # b"glTF"
    assert version == 2
    assert length == len(data)

    # First chunk must be JSON.
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == 0x4E4F534A
    assert 12 + 8 + chunk_length <= len(data)


def test_glb_point_cloud_without_faces(tmp_path):
    vertices, _faces, colors, confidence = _make_mesh()
    path = tmp_path / "points.glb"
    export_glb(path, vertices, faces=None, colors=colors, confidence=confidence)
    v2, f2, c2, conf2 = read_glb(path)
    np.testing.assert_allclose(v2, vertices, atol=1e-5)
    assert f2 is None
    np.testing.assert_array_equal(c2, colors)
    np.testing.assert_array_equal(conf2, confidence)


# ---------------------------------------------------------------------------
# LAS
# ---------------------------------------------------------------------------


def test_las_roundtrip_with_confidence_and_covariance(tmp_path):
    pc = _make_point_cloud()
    path = tmp_path / "cloud.las"
    export_las(path, pc, crs="EPSG:4326")

    las = laspy.read(str(path))
    xyz = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1)
    np.testing.assert_allclose(xyz, pc.xyz, atol=1e-3)

    assert "confidence" in las.point_format.extra_dimension_names
    np.testing.assert_array_equal(np.asarray(las.confidence), pc.confidence)

    assert "cov_trace" in las.point_format.extra_dimension_names
    expected_trace = np.trace(pc.covariance, axis1=1, axis2=2)
    np.testing.assert_allclose(np.asarray(las.cov_trace), expected_trace, atol=1e-3)

    rgb16 = np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], axis=1)
    expected_rgb16 = (pc.rgb.astype(np.float64) / 255.0 * 65535.0).astype(np.uint16)
    np.testing.assert_allclose(rgb16, expected_rgb16, atol=1)


def test_las_without_covariance_has_no_extra_dim(tmp_path):
    pc = PointCloud(xyz=np.random.default_rng(1).normal(size=(20, 3)), confidence=np.zeros(20, dtype=np.uint8))
    path = tmp_path / "cloud2.las"
    export_las(path, pc)
    las = laspy.read(str(path))
    assert "confidence" in las.point_format.extra_dimension_names
    assert "cov_trace" not in las.point_format.extra_dimension_names


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------


def test_xyz_export(tmp_path):
    pc = _make_point_cloud(n=20)
    path = tmp_path / "cloud.xyz"
    export_xyz(path, pc)

    loaded = np.loadtxt(path)
    np.testing.assert_allclose(loaded[:, :3], pc.xyz, atol=1e-4)
    np.testing.assert_array_equal(loaded[:, 3:6].astype(np.uint8), pc.rgb)


# ---------------------------------------------------------------------------
# GeoTIFF / DSM
# ---------------------------------------------------------------------------


def test_dsm_tilted_plane_gradient_and_fill_mask():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = 0.5 * x + 0.1 * y  # tilted plane, increasing with x

    # Punch an actual hole: no points at all in this region.
    hole = (x > 4) & (x < 5) & (y > 4) & (y < 5)
    xyz = np.stack([x[~hole], y[~hole], z[~hole]], axis=1)
    pc = PointCloud(xyz=xyz)

    dsm, _transform, filled_mask = point_cloud_to_dsm(pc, resolution_m=0.25)

    assert not np.isnan(dsm).any()
    assert filled_mask.any()  # the punched hole should have been filled
    assert filled_mask.sum() < filled_mask.size  # but not everything

    # Gradient direction: mean DSM value should increase from the first
    # column (low x) to the last column (high x).
    assert dsm[:, -1].mean() > dsm[:, 0].mean()


def test_orthomosaic_and_confidence_raster_shapes():
    rng = np.random.default_rng(3)
    n = 2000
    xyz = np.stack([rng.uniform(0, 5, n), rng.uniform(0, 5, n), rng.uniform(0, 1, n)], axis=1)
    rgb = rng.integers(0, 255, size=(n, 3)).astype(np.uint8)
    confidence = rng.integers(0, 3, size=(n,)).astype(np.uint8)
    pc = PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)

    ortho, _t1, filled_ortho = point_cloud_to_orthomosaic(pc, resolution_m=0.5)
    assert ortho.shape[:2] == filled_ortho.shape
    assert ortho.dtype == np.uint8

    conf_raster, _t2, filled_conf = point_cloud_to_confidence_raster(pc, resolution_m=0.5)
    assert conf_raster.shape == filled_conf.shape
    assert np.nanmin(conf_raster) >= 0


def _force_npy_fallback(monkeypatch):
    """Make ``import rasterio`` / ``import tifffile`` raise, whatever is installed.

    Setting a module to ``None`` in ``sys.modules`` is the documented way to
    make its import raise ``ImportError``. Without this the test is
    environment-dependent: it passes on a machine with no GIS stack and
    fails on one that has rasterio or tifffile, because ``write_geotiff``
    then correctly writes a real GeoTIFF instead of the fallback. That is
    the product behaving properly, so the test -- not the product -- has to
    pin which branch it is exercising.
    """
    import sys

    monkeypatch.setitem(sys.modules, "rasterio", None)
    monkeypatch.setitem(sys.modules, "tifffile", None)


def test_write_geotiff_npy_fallback(tmp_path, monkeypatch):
    _force_npy_fallback(monkeypatch)

    rng = np.random.default_rng(4)
    xyz = np.stack([rng.uniform(0, 5, 500), rng.uniform(0, 5, 500), rng.uniform(0, 1, 500)], axis=1)
    pc = PointCloud(xyz=xyz)
    dsm, transform, _filled = point_cloud_to_dsm(pc, resolution_m=0.5)

    path = tmp_path / "dsm.tif"
    backend = write_geotiff(path, dsm.astype(np.float32), transform, crs="EPSG:4326", nodata=-9999)

    assert backend == "npy_fallback"
    stem = path.with_suffix("")
    assert stem.with_suffix(".npy").exists()
    assert stem.with_suffix(".tfw").exists()
    assert stem.with_suffix(".prj").exists()

    loaded = np.load(stem.with_suffix(".npy"))
    np.testing.assert_allclose(loaded, dsm.astype(np.float32), equal_nan=True)

    tfw_lines = stem.with_suffix(".tfw").read_text().strip().splitlines()
    assert len(tfw_lines) == 6
    assert float(tfw_lines[0]) == pytest.approx(transform.pixel_size_x)


def _has_geotiff_backend() -> bool:
    for mod in ("rasterio", "tifffile"):
        try:
            __import__(mod)
        except ImportError:
            continue
        return True
    return False


@pytest.mark.skipif(not _has_geotiff_backend(), reason="neither rasterio nor tifffile installed")
def test_write_geotiff_uses_a_real_backend_when_one_is_available(tmp_path):
    """The branch a GPU/CI box actually takes, which the fallback test hides.

    On a machine with rasterio or tifffile, ``write_geotiff`` must produce
    an actual ``.tif`` and say which backend wrote it -- and must NOT leave
    the ``.npy`` sidecar behind, since a caller seeing both would not know
    which one is authoritative.
    """
    rng = np.random.default_rng(5)
    xyz = np.stack([rng.uniform(0, 5, 500), rng.uniform(0, 5, 500), rng.uniform(0, 1, 500)], axis=1)
    dsm, transform, _filled = point_cloud_to_dsm(PointCloud(xyz=xyz), resolution_m=0.5)

    path = tmp_path / "dsm.tif"
    backend = write_geotiff(path, dsm.astype(np.float32), transform, crs="EPSG:4326", nodata=-9999)

    assert backend in {"rasterio", "tifffile"}
    assert path.exists()
    assert path.stat().st_size > 0
    assert not path.with_suffix(".npy").exists()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_missing_junction_residual_does_not_break_report():
    report = build_report({"junction_residuals": [{"junction": 0, "rmse_m": None}]})
    assert "not computed" in render_report_text(report)
    assert "not computed" in render_report_html(report)


def test_build_report_missing_keys_are_not_computed():
    report = build_report({})
    for key in (
        "relative_rmse_m",
        "absolute_rmse_m",
        "scale_error_pct",
        "mean_reprojection_error_px",
        "coverage_pct",
        "keyframe_count",
        "confidence_breakdown_pct",
        "junction_residuals",
        "stage_timings_s",
    ):
        assert report[key] == NOT_COMPUTED
        assert report[key] != 0.0


def test_build_report_never_fabricates_zero():
    # A pathological "everything is falsy/zero-like" input should still
    # not be confused with "not computed" for keys that were never given.
    report = build_report({"coverage_pct": 0.0})
    assert report["coverage_pct"] == 0.0  # explicitly measured as zero is fine...
    assert report["relative_rmse_m"] == NOT_COMPUTED  # ...but never-given must stay "not computed"


def test_build_report_confidence_from_raw_array():
    confidence = np.array([Confidence.MEASURED] * 7 + [Confidence.INFERRED] * 3, dtype=np.uint8)
    report = build_report({"confidence": confidence})
    breakdown = report["confidence_breakdown_pct"]
    assert breakdown != NOT_COMPUTED
    np.testing.assert_allclose(breakdown["measured_pct"], 70.0)
    np.testing.assert_allclose(breakdown["inferred_pct"], 30.0)


def test_build_report_confidence_from_mesh_stats():
    mesh_stats = {"confidence_breakdown_pct": {"measured": 1.0, "low_confidence": 2.0, "inferred": 3.0}}
    report = build_report({"mesh_stats": mesh_stats})
    assert report["confidence_breakdown_pct"] == mesh_stats["confidence_breakdown_pct"]


def test_render_report_text_and_html_handle_not_computed():
    report = build_report({})
    text = render_report_text(report)
    html = render_report_html(report)
    assert NOT_COMPUTED in text
    assert NOT_COMPUTED in html
    assert "<html" in html


def test_check_point_residuals_measures_vertical_offset():
    origin = GeoPoint(lat=12.0, lon=78.0, alt_msl=50.0)
    rng = np.random.default_rng(5)
    xyz = rng.uniform(-20, 20, size=(3000, 3))
    xyz[:, 2] = 0.0
    pc = PointCloud(xyz=xyz)

    from drishti3d.geometry.georef import enu_to_geopoints

    check_points = enu_to_geopoints(np.array([[1.0, 1.0, 0.75]]), origin)
    result = check_point_residuals(pc, check_points, origin)

    assert result["n_check_points"] == 1
    assert result["vertical_rmse_m"] == pytest.approx(0.75, abs=0.05)


def test_check_point_residuals_empty_list():
    pc = PointCloud(xyz=np.zeros((5, 3)))
    origin = GeoPoint(lat=0.0, lon=0.0, alt_msl=0.0)
    result = check_point_residuals(pc, [], origin)
    assert result["n_check_points"] == 0
    assert result["horizontal_rmse_m"] == NOT_COMPUTED


# ---------------------------------------------------------------------------
# FBX (a required output format in the problem statement)
# ---------------------------------------------------------------------------


def test_export_fbx_writes_a_readable_ascii_mesh(tmp_path):
    from drishti3d.export.formats import export_fbx

    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    path = tmp_path / "m.fbx"
    export_fbx(path, vertices, faces)
    text = path.read_text()

    assert "FBXVersion: 7400" in text
    assert f"Vertices: *{vertices.size}" in text
    assert f"PolygonVertexIndex: *{faces.size}" in text
    assert "Connections" in text


def test_fbx_encodes_polygon_ends_by_negation(tmp_path):
    """FBX marks the last index of each polygon as ``-i - 1``.

    That is how a reader finds face boundaries in an otherwise flat index
    list. Get it wrong and the whole mesh imports as one degenerate
    polygon -- which looks like a geometry bug, not a format bug.
    """
    import re

    from drishti3d.export.formats import export_fbx

    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    path = tmp_path / "m.fbx"
    export_fbx(path, vertices, faces)
    indices = re.search(r"PolygonVertexIndex: \*6 \{\s*a: ([^}]+)\}", path.read_text()).group(1)

    # face [0,1,2] -> 0,1,-3   and   face [0,2,3] -> 0,2,-4
    assert [int(v) for v in indices.strip().split(",")] == [0, 1, -3, 0, 2, -4]


def test_fbx_carries_vertex_colour_when_given(tmp_path):
    from drishti3d.export.formats import export_fbx

    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    path = tmp_path / "m.fbx"
    export_fbx(path, vertices, faces, colors=colors)
    text = path.read_text()

    assert "LayerElementColor" in text
    assert 'MappingInformationType: "ByVertice"' in text
    # 3 vertices x RGBA
    assert "Colors: *12" in text


def test_fbx_omits_colour_layer_when_absent(tmp_path):
    from drishti3d.export.formats import export_fbx

    path = tmp_path / "m.fbx"
    export_fbx(path, np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]]), np.array([[0, 1, 2]]))

    assert "LayerElementColor" not in path.read_text()


def test_fbx_is_an_offered_bundle_format():
    """The problem statement lists .fbx among required output formats."""
    from drishti3d.export.bundle_export import _ALL_FORMATS

    assert "fbx" in _ALL_FORMATS
    # And every other required format is still there.
    for required in ("obj", "ply", "las", "glb"):
        assert required in _ALL_FORMATS


def test_map_deliverables_are_written_in_utm_not_local_enu(tmp_path) -> None:
    import json

    import numpy as np
    import pyproj
    import rasterio

    from drishti3d.export.bundle_export import export_all
    from drishti3d.types import GeoPoint, PointCloud

    origin = GeoPoint(lat=41.7716476, lon=-0.7458537, alt_msl=324.72)
    rng = np.random.default_rng(0)
    xyz = np.c_[rng.uniform(0, 200, 5000), rng.uniform(0, 200, 5000), rng.uniform(-1, 1, 5000)]
    xyz[0] = [100.0, 0.0, 0.0]  # 100 m due east of the origin
    pc = PointCloud(xyz=xyz, rgb=np.full((5000, 3), 128, np.uint8))

    written = export_all(pc, tmp_path, formats={"las", "ply"}, crs="EPSG:32630", geo_origin=origin)

    e0, n0 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True).transform(origin.lon, origin.lat)
    import laspy

    las = laspy.read(written["las"])
    p = np.c_[las.x, las.y][0]
    # 100 m east in ENU is ~100 m in UTM, rotated by the ~1.5 deg meridian convergence.
    assert abs(np.hypot(p[0] - e0, p[1] - n0) - 100.0) < 0.2
    assert abs(p[1] - n0) > 1.0  # the convergence rotation really was applied

    with rasterio.open(written["dsm"]) as d:
        assert d.crs.to_epsg() == 32630
        assert d.bounds.left > 600000 and d.bounds.bottom > 4.6e6
    assert json.loads((tmp_path / "georef.json").read_text())["map_crs"] == "EPSG:32630"


def test_exported_dsm_leaves_unobserved_ground_empty(tmp_path):
    """Nearest-neighbour fill across ground the camera never saw is invented height; only pinholes are filled."""
    pytest.importorskip("rasterio")
    import rasterio

    from drishti3d.export import export_all

    rng = np.random.default_rng(0)
    # Two observed patches 60 m apart, nothing in between; a 1-cell pinhole inside the first patch.
    a = np.c_[rng.uniform(0, 20, 4000), rng.uniform(0, 20, 4000), np.zeros(4000)]
    b = np.c_[rng.uniform(80, 100, 4000), rng.uniform(0, 20, 4000), np.full(4000, 5.0)]
    xyz = np.vstack([a, b])
    written = export_all(PointCloud(xyz=xyz, rgb=np.full((len(xyz), 3), 100, np.uint8)), tmp_path, formats={"ply"})
    with rasterio.open(written["dsm"]) as d:
        dsm = d.read(1)
    with rasterio.open(written["dsm_interpolated"]) as d:
        interp = d.read(1)
    cols = dsm.shape[1]
    assert np.isnan(dsm[:, int(0.5 * cols)]).all()  # the unobserved middle stays nodata
    assert np.isfinite(dsm[:, 2]).mean() > 0.9 and np.isfinite(dsm[:, -3]).mean() > 0.9
    assert interp.max() <= 1 and not interp[:, int(0.5 * cols)].any()


def test_true_ortho_raster_samples_the_texture_at_its_own_resolution():
    """Vertices on a 0.5 m grid with a 3x texture: the orthomosaic keeps ~0.17 m texels and lands on the right ground."""
    from drishti3d.export.bundle_export import _true_ortho_raster

    ny, nx, up = 40, 60, 3
    xs = 500000.0 + (np.arange(nx) + 0.5) * 0.5
    ys = 4600000.0 - (np.arange(ny) + 0.5) * 0.5
    X, Y = np.meshgrid(xs, ys)
    rows, cols = np.mgrid[0:ny, 0:nx]
    uv = np.c_[(cols.ravel() + 0.5) / nx, 1.0 - (rows.ravel() + 0.5) / ny]
    texture = np.zeros((ny * up, nx * up, 3), np.uint8)
    texture[:, : nx * up // 2] = (255, 0, 0)  # west half red
    texture[:, nx * up // 2 :] = (0, 0, 255)  # east half blue
    rgb, tr = _true_ortho_raster(np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)], texture, uv)
    assert abs(tr.pixel_size_x - 0.5 / up) < 1e-6 and tr.pixel_size_y < 0
    h, w = rgb.shape[:2]
    assert rgb[h // 2, w // 4, 0] == 255 and rgb[h // 2, 3 * w // 4, 2] == 255
    # The red/blue boundary sits at the grid's middle easting.
    mid_col = int(round((xs.mean() - (tr.x_origin - tr.pixel_size_x / 2)) / tr.pixel_size_x))
    assert rgb[h // 2, mid_col - 3, 0] == 255 and rgb[h // 2, mid_col + 3, 2] == 255


def test_las_carries_per_point_height_uncertainty(tmp_path):
    from drishti3d.export.formats import export_las

    n = 50
    sigma = np.linspace(0.05, 2.5, n).astype(np.float32)
    sigma[:3] = np.nan  # inferred points stay NaN, not a made-up number
    pc = PointCloud(
        xyz=np.random.default_rng(0).uniform(0, 10, (n, 3)),
        confidence=np.full(n, int(Confidence.MEASURED), np.uint8),
        uncertainty_m=sigma,
    )
    export_las(tmp_path / "m.las", pc)
    got = np.asarray(laspy.read(tmp_path / "m.las").height_uncertainty_m)
    assert np.isnan(got[:3]).all() and np.allclose(got[3:], sigma[3:])


def test_pipeline_result_round_trips_uncertainty(tmp_path):
    from drishti3d.pipeline.result import PipelineResult

    pc = PointCloud(xyz=np.zeros((4, 3)), uncertainty_m=np.array([0.1, 0.2, np.nan, 1.5], np.float32))
    PipelineResult(point_cloud=pc).save(tmp_path / "r")
    back = PipelineResult.load(tmp_path / "r").point_cloud.uncertainty_m
    assert np.allclose(back[[0, 1, 3]], [0.1, 0.2, 1.5]) and np.isnan(back[2])
