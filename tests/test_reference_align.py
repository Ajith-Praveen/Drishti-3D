"""End-to-end test for drishti3d.geometry.reference_align on a synthetic reference."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("kornia")


def _texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3.0)
    base = (base - base.min()) / (base.max() - base.min())
    return (np.stack([base, base**0.8, base**1.2], axis=-1) * 255).astype(np.uint8)


def test_recovers_a_gps_style_offset(tmp_path) -> None:
    import pyproj
    import rasterio
    from rasterio.transform import from_origin

    from drishti3d.geometry.georef import enu_to_wgs84, wgs84_to_enu
    from drishti3d.geometry.reference_align import align_to_reference, apply_to_enu
    from drishti3d.types import GeoPoint

    origin = GeoPoint(lat=41.7716, lon=-0.7458, alt_msl=320.0)
    crs = "EPSG:32630"
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    e0, n0 = to_utm.transform(origin.lon, origin.lat)
    gsd, size = 0.5, 900
    img = _texture(size, size)
    x0, y1 = e0 - 225.0, n0 + 225.0
    path = tmp_path / "ortho.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=crs,
        transform=from_origin(x0, y1, gsd, gsd),
    ) as dst:
        dst.write(img.transpose(2, 0, 1))

    # Model points on a 0.25 m grid over the central 300 m, coloured from the
    # reference at a location 4 m WEST / 3 m NORTH of where they are placed --
    # i.e. the model sits 4 m east and 3 m south of the truth.
    g = np.arange(-150.0, 150.0, 0.25)
    E, N = np.meshgrid(g, g)
    true_e, true_n = e0 + E.ravel() - 4.0, n0 + N.ravel() + 3.0
    col = ((true_e - x0) / gsd).astype(int).clip(0, size - 1)
    row = ((y1 - true_n) / gsd).astype(int).clip(0, size - 1)
    rgb = img[row, col]
    from_utm = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = from_utm.transform(e0 + E.ravel(), n0 + N.ravel())
    enu = wgs84_to_enu(np.c_[lon, lat, np.full(lon.size, 320.0)], origin)

    al = align_to_reference(enu, rgb, origin, str(path), gsd=0.5)

    assert al.applied, al.failure
    assert abs(al.t[0] - (-4.0)) < 0.5 and abs(al.t[1] - 3.0) < 0.5
    fixed = apply_to_enu(al, enu[:1])
    llh = enu_to_wgs84(fixed, origin)
    fe, fn = to_utm.transform(llh[0, 0], llh[0, 1])
    assert abs(fe - true_e[0]) < 0.5 and abs(fn - true_n[0]) < 0.5


def test_report_card_shows_an_applied_reference_alignment():
    """The alignment result used to be dropped by build_report, so the card always said 'not computed'."""
    from drishti3d.export.report import build_report, render_report_text

    applied = {
        "applied": True, "failure": None, "shift_east_m": -5.333, "shift_north_m": -2.758, "shift_up_m": -6.01,
        "rotation_deg": -0.0152, "scale": 0.9839, "ortho": "ortho.tif", "dem": "dem.tif",
        "horizontal": {"method": "lightglue", "inliers": 297, "residual_rms_m": 0.94},
    }
    text = render_report_text(build_report({"reference_alignment": applied}))
    assert "shift applied: E -5.33 m, N -2.76 m, U -6.01 m" in text
    refused = render_report_text(build_report({"reference_alignment": {"applied": False, "failure": "too few inliers"}}))
    assert "REFUSED: too few inliers" in refused
    assert "no reference configured" in render_report_text(build_report({}))


def test_translation_fit_uses_bare_ground_and_ignores_leaning_canopy(tmp_path) -> None:
    """The reference ortho is rectified on a terrain model, so its canopy is relief-displaced.

    Half the scene is 8 m 'trees' whose reference appearance sits 6 m further
    east (lean); only the bare-ground half shows the true -4 E / +3 N offset.
    With the reference DEM the fit must use the bare ground and recover it.
    """
    import pyproj
    import rasterio
    from rasterio.transform import from_origin

    from drishti3d.geometry.georef import wgs84_to_enu
    from drishti3d.geometry.reference_align import align_to_reference
    from drishti3d.types import GeoPoint

    origin = GeoPoint(lat=41.7716, lon=-0.7458, alt_msl=320.0)
    crs = "EPSG:32630"
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    e0, n0 = to_utm.transform(origin.lon, origin.lat)
    gsd, size = 0.5, 900
    img = _texture(size, size, seed=3)
    x0, y1 = e0 - 225.0, n0 + 225.0
    lean_px = 12  # 6 m
    ref = img.copy()
    half = size // 2
    ref[:, half:] = np.roll(img, lean_px, axis=1)[:, half:]  # canopy half displaced east
    ortho = tmp_path / "ortho.tif"
    with rasterio.open(ortho, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=crs,
                       transform=from_origin(x0, y1, gsd, gsd)) as dst:
        dst.write(ref.transpose(2, 0, 1))
    dem = tmp_path / "dem.tif"
    with rasterio.open(dem, "w", driver="GTiff", width=45, height=45, count=1, dtype="float32", crs=crs,
                       transform=from_origin(x0, y1, 10.0, 10.0)) as dst:
        dst.write(np.full((1, 45, 45), 320.0, dtype=np.float32))

    g = np.arange(-150.0, 150.0, 0.25)
    E, N = np.meshgrid(g, g)
    true_e, true_n = e0 + E.ravel() - 4.0, n0 + N.ravel() + 3.0
    col = ((true_e - x0) / gsd).astype(int).clip(0, size - 1)
    row = ((y1 - true_n) / gsd).astype(int).clip(0, size - 1)
    rgb = img[row, col]
    canopy = E.ravel() > 0.0
    z = np.where(canopy, 328.0, 320.0)
    from_utm = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = from_utm.transform(e0 + E.ravel(), n0 + N.ravel())
    enu = wgs84_to_enu(np.c_[lon, lat, z], origin)

    al = align_to_reference(enu, rgb, origin, str(ortho), str(dem), gsd=0.5)
    assert al.applied, al.failure
    assert al.horizontal.get("evidence") == "bare_ground", al.horizontal
    assert al.scale == 1.0 and al.rot_deg == 0.0
    assert abs(al.t[0] - (-4.0)) < 0.5 and abs(al.t[1] - 3.0) < 0.5, al.t


def test_vertical_offset_is_measured_on_the_typical_surface_not_the_treetops(tmp_path) -> None:
    """Scattered 6 m shrubs over a field must not drag the aligned ground below the reference."""
    import pyproj
    import rasterio
    from rasterio.transform import from_origin

    from drishti3d.geometry.georef import wgs84_to_enu
    from drishti3d.geometry.reference_align import align_to_reference
    from drishti3d.types import GeoPoint

    origin = GeoPoint(lat=41.7716, lon=-0.7458, alt_msl=320.0)
    crs = "EPSG:32630"
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    e0, n0 = to_utm.transform(origin.lon, origin.lat)
    gsd, size = 0.5, 900
    img = _texture(size, size, seed=5)
    x0, y1 = e0 - 225.0, n0 + 225.0
    ortho = tmp_path / "ortho.tif"
    with rasterio.open(ortho, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=crs,
                       transform=from_origin(x0, y1, gsd, gsd)) as dst:
        dst.write(img.transpose(2, 0, 1))
    dem = tmp_path / "dem.tif"
    with rasterio.open(dem, "w", driver="GTiff", width=45, height=45, count=1, dtype="float32", crs=crs,
                       transform=from_origin(x0, y1, 10.0, 10.0)) as dst:
        dst.write(np.full((1, 45, 45), 300.0, dtype=np.float32))

    g = np.arange(-150.0, 150.0, 0.25)
    E, N = np.meshgrid(g, g)
    col = ((e0 + E.ravel() - x0) / gsd).astype(int).clip(0, size - 1)
    row = ((y1 - (n0 + N.ravel())) / gsd).astype(int).clip(0, size - 1)
    rgb = img[row, col]
    rng = np.random.default_rng(2)
    shrub = rng.random(E.size) < 0.15
    z = np.where(shrub, 302.0 + 6.0, 302.0)  # model ground 2 m above the reference, 15% shrubs
    from_utm = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = from_utm.transform(e0 + E.ravel(), n0 + N.ravel())
    enu = wgs84_to_enu(np.c_[lon, lat, z], origin)
    al = align_to_reference(enu, rgb, origin, str(ortho), str(dem), gsd=0.5)
    assert al.applied, al.failure
    assert abs(al.dz - (-2.0)) < 0.3, al.vertical
