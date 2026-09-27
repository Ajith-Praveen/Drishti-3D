"""Surface measurements (drishti3d.app.measure): exact answers on synthetic surfaces, and GIS output."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from drishti3d.app.measure import (
    HeightGrid,
    Measurement,
    cut_fill_volume,
    elevation_profile,
    polygon_area_xy,
    polyline_length,
    to_geojson,
    to_kml,
    write_measurements,
)
from drishti3d.types import GeoPoint


def _surface(fn, extent=40.0, step=0.25):
    x, y = np.meshgrid(np.arange(0.0, extent, step), np.arange(0.0, extent, step))
    return np.c_[x.ravel(), y.ravel(), fn(x.ravel(), y.ravel())]


def test_height_grid_samples_a_slope_exactly():
    grid = HeightGrid.from_points(_surface(lambda x, y: 0.1 * x + 2.0), cell=0.5)
    z = grid.sample(np.array([[10.0, 10.0], [20.3, 5.7]]))
    np.testing.assert_allclose(z, [3.0, 4.03], atol=0.06)
    assert np.isnan(grid.sample(np.array([[500.0, 500.0]]))[0])


def test_volume_of_a_box_stockpile_and_a_pit():
    pile = lambda x, y: np.where((np.abs(x - 20) < 5) & (np.abs(y - 20) < 5), 3.0, 0.0)
    grid = HeightGrid.from_points(_surface(pile), cell=0.25)
    outline = np.array([[12.0, 12.0, 0.0], [28.0, 12.0, 0.0], [28.0, 28.0, 0.0], [12.0, 28.0, 0.0]])
    v = cut_fill_volume(grid, outline)
    # |x - 20| < 5 on a 0.25 m lattice: 39 samples, a 9.75 m square.
    assert v["above_m3"] == pytest.approx(3.0 * 9.75**2, rel=0.02)
    assert v["below_m3"] == pytest.approx(0.0, abs=1.0)
    assert v["area_m2"] == pytest.approx(256.0)
    assert v["max_above_m"] == pytest.approx(3.0, abs=0.01)
    pit = HeightGrid.from_points(_surface(lambda x, y: -pile(x, y)), cell=0.25)
    assert cut_fill_volume(pit, outline)["below_m3"] == pytest.approx(3.0 * 9.75**2, rel=0.02)


def test_volume_base_follows_sloping_ground():
    # A pile on a 5% slope: the base plane through the outline removes the slope.
    ground = lambda x, y: 0.05 * x
    pile = lambda x, y: ground(x, y) + np.where((np.abs(x - 20) < 4) & (np.abs(y - 20) < 4), 2.0, 0.0)
    grid = HeightGrid.from_points(_surface(pile), cell=0.25)
    outline = np.array([[x, y, ground(x, y)] for x, y in ((10, 10), (30, 10), (30, 30), (10, 30))], dtype=float)
    # |x - 20| < 4 on the 0.25 m lattice: a 7.75 m square, 2 m tall.
    assert cut_fill_volume(grid, outline)["above_m3"] == pytest.approx(2.0 * 7.75**2, rel=0.02)


def test_profile_reads_climb_descent_and_slope():
    ridge = lambda x, y: np.where(x < 20, 0.5 * x, 0.5 * (40 - x))
    grid = HeightGrid.from_points(_surface(ridge), cell=0.25)
    prof = elevation_profile(grid, np.array([[2.0, 10.0, 1.0], [38.0, 10.0, 1.0]]), step_m=0.5)
    assert prof["length_m"] == pytest.approx(36.0)
    assert prof["max_m"] == pytest.approx(10.0, abs=0.2)
    assert prof["climb_m"] == pytest.approx(9.0, abs=0.3)
    assert prof["descent_m"] == pytest.approx(9.0, abs=0.3)
    assert prof["max_slope_deg"] == pytest.approx(np.degrees(np.arctan(0.5)), abs=1.5)
    assert prof["valid_fraction"] == 1.0


def test_lengths_and_areas():
    sq = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 5.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]])
    assert polygon_area_xy(sq) == pytest.approx(100.0)  # planimetric: heights ignored
    assert polyline_length(np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])) == pytest.approx(17.0)


def test_gis_exports_are_geodetic_and_well_formed(tmp_path):
    origin = GeoPoint(lat=30.275974, lon=-97.764502, alt_msl=445.6)
    ms = [
        Measurement("point", np.array([[0.0, 0.0, -288.0]]), 157.6, "m", "Point 1"),
        Measurement("distance", np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]), 100.0, "m", "Distance 1"),
        Measurement("volume", np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0]]), 12.5, "m³",
                    "Volume 1", extra={"above_m3": 12.5, "coverage": 0.9}),
        Measurement("profile", np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]), 50.0, "m", "Profile 1",
                    extra={"_xy": np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0]]), "_z": np.array([0.0, 1.0, 2.0])}),
    ]
    gj = to_geojson(ms, origin)
    lon, lat, h = gj["features"][0]["geometry"]["coordinates"]
    assert (lon, lat) == pytest.approx((-97.764502, 30.275974), abs=1e-7)
    assert h == pytest.approx(157.6, abs=0.01)  # elevation above sea level, not the local -288 m
    east = gj["features"][1]["geometry"]["coordinates"][1]
    assert east[0] > -97.764502 and east[1] == pytest.approx(30.275974, abs=1e-5)  # 100 m east
    poly = gj["features"][2]["geometry"]
    assert poly["type"] == "Polygon" and poly["coordinates"][0][0] == poly["coordinates"][0][-1]
    assert len(gj["features"][3]["geometry"]["coordinates"]) == 3  # the sampled profile line
    assert gj["features"][2]["properties"]["above_m3"] == 12.5
    ET.fromstring(to_kml(ms, origin))  # well-formed XML
    write_measurements(tmp_path / "m.geojson", ms, origin)
    write_measurements(tmp_path / "m.kml", ms, origin)
    assert json.loads((tmp_path / "m.geojson").read_text())["type"] == "FeatureCollection"
    with pytest.raises(ValueError):
        write_measurements(tmp_path / "m.shp", ms, origin)
