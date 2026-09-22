"""Tests for drishti3d.geometry.georef.

Central theme: relative vs. absolute accuracy must never be conflated
(see georef.py's module docstring), and the honesty-path enforcement
(``_absolute_accuracy_m``) must actually refuse a sub-metre absolute
accuracy claim for standalone GPS input, in code, not just in a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from drishti3d.geometry.georef import (
    GpsGrade,
    align_to_gps,
    detect_gps_grade,
    enu_to_geopoints,
    enu_to_wgs84,
    estimate_utm_crs,
    georeference,
    scale_error_estimate,
    wgs84_to_enu,
)
from drishti3d.geometry.submap import umeyama_alignment
from drishti3d.types import GeoPoint, PointCloud, Pose, TelemetrySample

_STANDALONE_GPS_ACCURACY_M = 3.5
_RTK_GPS_ACCURACY_M = 0.02


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# ---------------------------------------------------------------------------
# align_to_gps recovers a known Sim3
# ---------------------------------------------------------------------------


def test_align_to_gps_recovers_known_similarity_transform():
    rng = np.random.default_rng(9)
    src = rng.uniform(-10.0, 10.0, size=(20, 3))
    true_R = _random_rotation(rng)
    true_scale = 1.8
    true_t = np.array([100.0, -50.0, 5.0])
    dst = true_scale * (src @ true_R.T) + true_t

    poses = [Pose(R=np.eye(3), t=s) for s in src]
    transform, residuals = align_to_gps(poses, dst)

    assert transform.scale == pytest.approx(true_scale, abs=1e-6)
    np.testing.assert_allclose(transform.R, true_R, atol=1e-6)
    np.testing.assert_allclose(transform.t, true_t, atol=1e-6)
    assert residuals.max() < 1e-6


def test_align_to_gps_rejects_too_few_cameras():
    poses = [Pose(R=np.eye(3), t=np.array([0.0, 0.0, 0.0])), Pose(R=np.eye(3), t=np.array([1.0, 0.0, 0.0]))]
    with pytest.raises(ValueError):
        align_to_gps(poses, np.zeros((2, 3)))


def test_align_to_gps_fixed_rotation_recovers_scale_translation_on_collinear_cameras():
    # The degenerate case align_to_gps's fixed_rotation parameter exists
    # for: collinear camera centres cannot constrain rotation via Umeyama
    # (see geometry.submap's module docstring). By the time georeferencing
    # runs after a strategy="telemetry_rotation" submap merge (see
    # pipeline.stages._apply_georeferencing), pose *positions* are already
    # expressed in the world/ENU frame (only scale/translation drift
    # remains -- no further rotation is needed, hence fixed_rotation=I).
    rng = np.random.default_rng(21)
    line_param = np.linspace(0.0, 40.0, 10)
    local_centres = np.stack([line_param, np.zeros(10), np.zeros(10)], axis=1)
    true_scale = 1.3
    true_t = np.array([20.0, -10.0, 5.0])
    dst = true_scale * local_centres + true_t

    # Orientation (pose.R) is independent of align_to_gps's position-only
    # fit -- arbitrary here, e.g. whatever telemetry_rotation fixed it to.
    orientation = _random_rotation(rng)
    poses = [Pose(R=orientation, t=c) for c in local_centres]

    # Sanity: the unconstrained (rotation-included) fit really is
    # degenerate on this collinear data.
    _unfixed_transform, degenerate_check, _ = umeyama_alignment(local_centres, dst)
    assert degenerate_check is True

    transform, residuals = align_to_gps(poses, dst, fixed_rotation=np.eye(3))

    assert transform.scale == pytest.approx(true_scale, abs=1e-6)
    np.testing.assert_allclose(transform.t, true_t, atol=1e-6)
    np.testing.assert_allclose(transform.R, np.eye(3), atol=1e-12)
    assert residuals.max() < 1e-6

    # Applying transform to a pose keeps its (already-correct) orientation
    # untouched -- transform.R is identity, so apply_pose's R is unchanged.
    aligned_pose = transform.apply_pose(poses[0])
    np.testing.assert_allclose(aligned_pose.R, orientation, atol=1e-12)


def test_align_to_gps_is_robust_to_outliers():
    """RANSAC should recognize a small minority of grossly wrong GPS fixes as outliers."""
    rng = np.random.default_rng(10)
    src = rng.uniform(-10.0, 10.0, size=(20, 3))
    true_R = _random_rotation(rng)
    true_scale = 1.2
    true_t = np.array([10.0, 20.0, 0.0])
    dst = true_scale * (src @ true_R.T) + true_t

    dst_outliers = dst.copy()
    dst_outliers[:3] += rng.normal(scale=50.0, size=(3, 3))  # 3 badly-wrong GPS fixes

    poses = [Pose(R=np.eye(3), t=s) for s in src]
    transform, residuals = align_to_gps(poses, dst_outliers, seed=1)

    assert transform.scale == pytest.approx(true_scale, abs=0.05)
    # The inlier (non-outlier) cameras should have tiny residuals.
    assert np.median(residuals) < 1.0


# ---------------------------------------------------------------------------
# scale_error_estimate
# ---------------------------------------------------------------------------


def test_scale_error_estimate_known_percentage():
    rng = np.random.default_rng(11)
    gps = rng.uniform(-10.0, 10.0, size=(15, 3))
    recon = gps * 1.05  # exactly 5% scale error
    poses = [Pose(R=np.eye(3), t=p) for p in recon]
    pct = scale_error_estimate(poses, gps)
    assert pct == pytest.approx(5.0, abs=1e-6)


def test_scale_error_estimate_zero_for_perfect_scale():
    rng = np.random.default_rng(12)
    gps = rng.uniform(-10.0, 10.0, size=(10, 3))
    poses = [Pose(R=np.eye(3), t=p.copy()) for p in gps]
    pct = scale_error_estimate(poses, gps)
    assert pct == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ENU <-> WGS84 round trip, UTM zone
# ---------------------------------------------------------------------------


def test_enu_wgs84_roundtrip_to_millimetre():
    rng = np.random.default_rng(13)
    origin = GeoPoint(lat=13.0, lon=77.5, alt_msl=900.0)
    pts = rng.uniform(-2000.0, 2000.0, size=(25, 3))
    lonlatalt = enu_to_wgs84(pts, origin)
    back = wgs84_to_enu(lonlatalt, origin)
    assert np.abs(back - pts).max() < 1e-3  # < 1 mm


def test_enu_to_geopoints_matches_enu_to_wgs84():
    origin = GeoPoint(lat=13.0, lon=77.5, alt_msl=900.0)
    pts = np.array([[10.0, 20.0, 5.0], [-30.0, 40.0, -2.0]])
    lonlatalt = enu_to_wgs84(pts, origin)
    geopoints = enu_to_geopoints(pts, origin)
    for gp, (lon, lat, alt) in zip(geopoints, lonlatalt, strict=True):
        assert gp.lat == pytest.approx(lat)
        assert gp.lon == pytest.approx(lon)
        assert gp.alt_msl == pytest.approx(alt)


@pytest.mark.parametrize(
    "lat,lon,expected_epsg",
    [
        (13.0, 77.5, "EPSG:32643"),  # Bangalore, India -- UTM zone 43N
        (-33.9, 151.2, "EPSG:32756"),  # Sydney, Australia -- UTM zone 56S
        (51.5, -0.1, "EPSG:32630"),  # London -- UTM zone 30N
    ],
)
def test_estimate_utm_crs_known_coordinates(lat, lon, expected_epsg):
    assert estimate_utm_crs(lat, lon) == expected_epsg


# ---------------------------------------------------------------------------
# detect_gps_grade
# ---------------------------------------------------------------------------


def _telemetry_with_accuracy(n: int, accuracy_h: float | None, origin: GeoPoint) -> list[TelemetrySample]:
    samples = []
    for i in range(n):
        geo = GeoPoint(lat=origin.lat + i * 1e-5, lon=origin.lon, alt_msl=origin.alt_msl, accuracy_h=accuracy_h)
        samples.append(TelemetrySample(timestamp=float(i), geo=geo))
    return samples


def test_detect_gps_grade_standalone():
    origin = GeoPoint(lat=13.0, lon=77.5, alt_msl=900.0)
    telemetry = _telemetry_with_accuracy(10, _STANDALONE_GPS_ACCURACY_M, origin)
    grade, acc = detect_gps_grade(telemetry)
    assert grade == GpsGrade.STANDALONE
    assert acc == pytest.approx(_STANDALONE_GPS_ACCURACY_M)


def test_detect_gps_grade_rtk():
    origin = GeoPoint(lat=13.0, lon=77.5, alt_msl=900.0)
    telemetry = _telemetry_with_accuracy(10, _RTK_GPS_ACCURACY_M, origin)
    grade, acc = detect_gps_grade(telemetry)
    assert grade == GpsGrade.RTK
    assert acc == pytest.approx(_RTK_GPS_ACCURACY_M)


def test_detect_gps_grade_unknown_when_no_accuracy_field():
    origin = GeoPoint(lat=13.0, lon=77.5, alt_msl=900.0)
    telemetry = _telemetry_with_accuracy(10, None, origin)
    grade, acc = detect_gps_grade(telemetry)
    assert grade == GpsGrade.UNKNOWN
    assert acc is None


# ---------------------------------------------------------------------------
# georeference(): end to end, and the critical honesty path
# ---------------------------------------------------------------------------


def _flight_scene(rng, n=10, altitude=30.0):
    origin = GeoPoint(lat=13.0, lon=77.5, alt_msl=900.0)
    enu_true = np.stack([np.linspace(0.0, 50.0, n), np.zeros(n), np.full(n, altitude)], axis=1)
    return origin, enu_true


def _geo_at(enu_xyz: np.ndarray, origin: GeoPoint) -> GeoPoint:
    lon, lat, alt = enu_to_wgs84(enu_xyz[None, :], origin)[0]
    return GeoPoint(lat=lat, lon=lon, alt_msl=alt)


def test_georeference_standalone_gps_never_reports_submetre_absolute_accuracy():
    """The critical honesty requirement, enforced in code: standalone GPS
    input must never yield a sub-metre absolute accuracy claim, no matter
    how good the internal (relative) reconstruction is.
    """
    rng = np.random.default_rng(14)
    origin, enu_true = _flight_scene(rng)
    poses = [Pose(R=np.eye(3), t=p) for p in enu_true]  # perfect internal reconstruction

    telemetry = []
    for i, p in enumerate(enu_true):
        noisy = p + rng.normal(scale=2.0, size=3)
        geo = _geo_at(noisy, origin)
        geo.accuracy_h = _STANDALONE_GPS_ACCURACY_M
        telemetry.append(TelemetrySample(timestamp=float(i), geo=geo))

    pc = PointCloud(xyz=np.zeros((5, 3)))
    result = georeference(pc, poses, telemetry)

    assert result.gps_grade == GpsGrade.STANDALONE
    assert result.absolute_accuracy_m >= 1.0
    assert result.absolute_accuracy_m >= result.gps_bias_estimate_m
    # Relative accuracy is allowed to be (and here, is) far tighter than absolute.
    assert result.relative_accuracy_m < result.absolute_accuracy_m


def test_georeference_unknown_gps_grade_also_conservative():
    """No accuracy field reported at all must not be treated as license to
    assume RTK-grade absolute accuracy.
    """
    rng = np.random.default_rng(15)
    origin, enu_true = _flight_scene(rng)
    poses = [Pose(R=np.eye(3), t=p) for p in enu_true]

    telemetry = []
    for i, p in enumerate(enu_true):
        noisy = p + rng.normal(scale=2.0, size=3)
        geo = _geo_at(noisy, origin)  # no accuracy_h set
        telemetry.append(TelemetrySample(timestamp=float(i), geo=geo))

    pc = PointCloud(xyz=np.zeros((5, 3)))
    result = georeference(pc, poses, telemetry)

    assert result.gps_grade == GpsGrade.UNKNOWN
    assert result.absolute_accuracy_m >= 1.0


def test_georeference_rtk_grade_allows_centimetre_absolute_accuracy():
    rng = np.random.default_rng(16)
    origin, enu_true = _flight_scene(rng)
    poses = [Pose(R=np.eye(3), t=p) for p in enu_true]

    telemetry = []
    for i, p in enumerate(enu_true):
        noisy = p + rng.normal(scale=0.02, size=3)
        geo = _geo_at(noisy, origin)
        geo.accuracy_h = _RTK_GPS_ACCURACY_M
        telemetry.append(TelemetrySample(timestamp=float(i), geo=geo))

    pc = PointCloud(xyz=np.zeros((5, 3)))
    result = georeference(pc, poses, telemetry)

    assert result.gps_grade == GpsGrade.RTK
    assert result.absolute_accuracy_m < 0.5  # centimetre-to-decimetre level allowed


def test_georeference_reports_crs_and_scale_error():
    rng = np.random.default_rng(17)
    origin, enu_true = _flight_scene(rng)
    # Introduce a deliberate 3% scale error in the "reconstruction".
    poses = [Pose(R=np.eye(3), t=p * 1.03) for p in enu_true]

    telemetry = []
    for i, p in enumerate(enu_true):
        geo = _geo_at(p, origin)
        geo.accuracy_h = _STANDALONE_GPS_ACCURACY_M
        telemetry.append(TelemetrySample(timestamp=float(i), geo=geo))

    pc = PointCloud(xyz=np.zeros((5, 3)))
    result = georeference(pc, poses, telemetry)

    assert result.crs == "EPSG:32643"
    assert result.scale_error_pct == pytest.approx(3.0, abs=0.5)


def test_georeference_gcp_can_tighten_absolute_accuracy_even_without_rtk():
    """A surveyed GCP is an independent absolute reference; if provided
    and residuals are small, it should be able to justify a tighter
    absolute-accuracy claim than the standalone-GPS floor, honestly (the
    claim is now backed by the GCP, not by the drone's own GPS).

    ``relative_accuracy_m`` is passed explicitly here, as a real caller
    with a bundle-adjusted reconstruction would (see ``georeference``'s
    docstring on why its own GPS-alignment-residual fallback is not a
    trustworthy internal-accuracy estimate when GPS itself is noisy --
    exactly the situation this test constructs, with 2 m per-camera GPS
    noise on every camera except the one GCP).
    """
    rng = np.random.default_rng(18)
    origin, enu_true = _flight_scene(rng)
    poses = [Pose(R=np.eye(3), t=p) for p in enu_true]  # perfect internal reconstruction

    telemetry = []
    for i, p in enumerate(enu_true):
        noisy = p + rng.normal(scale=2.0, size=3)
        geo = _geo_at(noisy, origin)
        geo.accuracy_h = _STANDALONE_GPS_ACCURACY_M
        telemetry.append(TelemetrySample(timestamp=float(i), geo=geo))

    gcp_local_xyz = enu_true[0]
    gcp_geo = _geo_at(gcp_local_xyz, origin)
    pc = PointCloud(xyz=np.zeros((5, 3)))
    result = georeference(pc, poses, telemetry, gcps=[(gcp_local_xyz, gcp_geo)], relative_accuracy_m=0.02)

    assert result.gcp_residuals_m is not None
    assert result.gcp_residuals_m.max() < 0.1
    assert result.absolute_accuracy_m < 1.0  # GCP-backed, allowed to beat the standalone-GPS floor


def test_georeference_requires_minimum_geo_tagged_samples():
    rng = np.random.default_rng(19)
    origin, enu_true = _flight_scene(rng, n=2)
    poses = [Pose(R=np.eye(3), t=p) for p in enu_true]
    telemetry = [TelemetrySample(timestamp=float(i), geo=_geo_at(p, origin)) for i, p in enumerate(enu_true)]
    pc = PointCloud(xyz=np.zeros((1, 3)))
    with pytest.raises(ValueError):
        georeference(pc, poses, telemetry)
