"""Georeferencing: tying the metric reconstruction back to the Earth, honestly.

Two different accuracy claims, never conflated
--------------------------------------------------
A drone reconstruction refined by bundle adjustment (``geometry.bundle``)
has two fundamentally different kinds of accuracy, and this module's
central job is to keep them separate rather than report one number that
quietly means whichever one happens to look better:

- **Relative / metric accuracy**: how well-shaped the reconstruction is
  internally -- point-to-point distances, surface flatness, the scale of
  the model. This can be sub-metre, even centimetre-level, because it
  comes from dense multi-view triangulation and bundle adjustment, and
  does not depend on absolute positioning at all.
- **Absolute accuracy**: how well the whole model's position on the Earth
  is known. With a standalone (non-RTK/PPK) GPS receiver -- the common
  case for a consumer/prosumer drone -- this is bounded by GPS's own
  standalone accuracy, typically several metres, and **no amount of
  bundle adjustment, dense reconstruction, or clever alignment removes
  that bias**. Aligning a beautifully self-consistent, sub-metre-accurate
  point cloud to a GPS track that is itself biased by 3-5 m simply moves
  that bias into the "absolute accuracy" number; it cannot average it
  away (the bias is systematic, not zero-mean noise, across one flight).

Reporting a single blended "accuracy" figure -- or worse, reporting the
relative accuracy as if it were the absolute one, because it's the
smaller/nicer-looking number -- is the single most common way drone survey
tools mislead users. ``georeference`` below always returns both, plus an
explicit estimate of the GPS bias driving the gap between them, and
``gps_grade_from_accuracy``/the RTK detection path is what allows the
absolute figure to legitimately drop to centimetre level -- only when the
input telemetry actually says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from drishti3d.geometry.submap import Sim3, umeyama_alignment, umeyama_fixed_rotation
from drishti3d.types import GeoPoint, PointCloud, Pose, TelemetrySample

# ---------------------------------------------------------------------------
# GPS grade detection
# ---------------------------------------------------------------------------

# Standalone consumer/prosumer GPS (no RTK/PPK correction) horizontal
# accuracy is commonly quoted at 1.5-5 m by manufacturers, and worse in
# practice (multipath, poor satellite geometry). We treat any reported
# accuracy_h above this as "standalone-grade" and refuse to report better
# than that as an absolute accuracy claim, regardless of how good the
# *relative* reconstruction is (see module docstring).
_STANDALONE_GPS_BIAS_M = 3.0

# RTK/PPK-corrected GPS reports centimetre-level horizontal accuracy
# (typically 1-3 cm in good conditions). A reported accuracy_h at or below
# this threshold is what lets `georeference` legitimately claim
# centimetre-level absolute accuracy.
_RTK_ACCURACY_THRESHOLD_M = 0.10


class GpsGrade(Enum):
    """Detected quality tier of the GPS fixes backing a flight's telemetry."""

    UNKNOWN = "unknown"  # no accuracy field reported at all
    STANDALONE = "standalone"  # reported accuracy indicates no RTK/PPK correction
    RTK = "rtk"  # reported accuracy indicates RTK/PPK-grade correction


def detect_gps_grade(telemetry: list[TelemetrySample]) -> tuple[GpsGrade, float | None]:
    """Classify a flight's GPS quality from reported ``GeoPoint.accuracy_h`` fields.

    Returns ``(grade, median_accuracy_h_m)``. ``median_accuracy_h_m`` is
    ``None`` when no sample reports an accuracy field at all -- in that
    case ``grade`` is ``GpsGrade.UNKNOWN`` and callers (``georeference``)
    must assume the conservative (standalone) bound rather than silently
    assuming RTK quality just because no worse information was given.
    """
    accuracies = [
        s.geo.accuracy_h
        for s in telemetry
        if s.geo is not None and s.geo.accuracy_h is not None
    ]
    if not accuracies:
        return GpsGrade.UNKNOWN, None

    median_acc = float(np.median(accuracies))
    if median_acc <= _RTK_ACCURACY_THRESHOLD_M:
        return GpsGrade.RTK, median_acc
    return GpsGrade.STANDALONE, median_acc


# ---------------------------------------------------------------------------
# ENU <-> WGS84 (thin pyproj wrappers matching ingest.telemetry's convention)
# ---------------------------------------------------------------------------

_WGS84_LLA = CRS.from_epsg(4979)
_ECEF = CRS.from_epsg(4978)
_LLA_TO_ECEF = Transformer.from_crs(_WGS84_LLA, _ECEF, always_xy=True)
_ECEF_TO_LLA = Transformer.from_crs(_ECEF, _WGS84_LLA, always_xy=True)


def _ecef_from_geo(lat: float, lon: float, alt: float) -> np.ndarray:
    x, y, z = _LLA_TO_ECEF.transform(lon, lat, alt)
    return np.array([x, y, z], dtype=np.float64)


def _enu_rotation_matrix(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )


def enu_to_wgs84(points: np.ndarray, origin: GeoPoint) -> np.ndarray:
    """Convert ``(N, 3)`` local ENU metres to ``(N, 3)`` WGS84 ``[lon, lat, alt_msl]``.

    Same ECEF-based route as ``ingest.telemetry.telemetry_to_enu`` (exact,
    not flat-earth), applied in reverse. Column order is ``[lon, lat,
    alt]`` (matching ``pyproj``'s ``always_xy=True`` convention) rather
    than ``[lat, lon, alt]`` -- callers that want ``GeoPoint``s should use
    ``enu_to_geopoints`` instead, which is unambiguous.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    origin_ecef = _ecef_from_geo(origin.lat, origin.lon, origin.alt_msl)
    rotation = _enu_rotation_matrix(origin.lat, origin.lon)

    ecef = origin_ecef[None, :] + points @ rotation  # rotation.T @ enu, batched
    lon, lat, alt = _ECEF_TO_LLA.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    return np.stack([np.asarray(lon), np.asarray(lat), np.asarray(alt)], axis=1)


def enu_to_geopoints(points: np.ndarray, origin: GeoPoint) -> list[GeoPoint]:
    """Convert ``(N, 3)`` local ENU metres to a list of ``GeoPoint``."""
    lonlatalt = enu_to_wgs84(points, origin)
    return [GeoPoint(lat=float(lat), lon=float(lon), alt_msl=float(alt)) for lon, lat, alt in lonlatalt]


def wgs84_to_enu(lonlatalt: np.ndarray, origin: GeoPoint) -> np.ndarray:
    """Convert ``(N, 3)`` WGS84 ``[lon, lat, alt_msl]`` to local ENU metres. Inverse of ``enu_to_wgs84``."""
    lonlatalt = np.asarray(lonlatalt, dtype=np.float64).reshape(-1, 3)
    origin_ecef = _ecef_from_geo(origin.lat, origin.lon, origin.alt_msl)
    rotation = _enu_rotation_matrix(origin.lat, origin.lon)

    x, y, z = _LLA_TO_ECEF.transform(lonlatalt[:, 0], lonlatalt[:, 1], lonlatalt[:, 2])
    ecef = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=1)
    return (ecef - origin_ecef[None, :]) @ rotation.T


def estimate_utm_crs(lat: float, lon: float) -> str:
    """Return the ``"EPSG:xxxxx"`` code of the UTM zone containing ``(lat, lon)``.

    Delegates to ``pyproj``'s own UTM-zone-lookup database
    (``query_utm_crs_info``) rather than hand-rolling the
    ``zone = floor((lon+180)/6)+1`` formula, since ``pyproj`` already
    correctly handles the standard exceptions to that formula (Norway,
    Svalbard) that a naive formula gets wrong.
    """
    aoi = AreaOfInterest(west_lon_degree=lon, south_lat_degree=lat, east_lon_degree=lon, north_lat_degree=lat)
    matches = query_utm_crs_info(datum_name="WGS 84", area_of_interest=aoi)
    if not matches:
        raise ValueError(f"no UTM CRS found for lat={lat}, lon={lon}")
    return f"{matches[0].auth_name}:{matches[0].code}"


# ---------------------------------------------------------------------------
# Alignment to GPS
# ---------------------------------------------------------------------------

_RANSAC_ITERS = 300
_RANSAC_INLIER_THRESHOLD_M = 5.0  # generous: standalone GPS noise floor, not a fit-quality target


def align_to_gps(
    poses: list[Pose], camera_gps_enu: np.ndarray, seed: int = 0, fixed_rotation: np.ndarray | None = None
) -> tuple[Sim3, np.ndarray]:
    """RANSAC-align reconstructed camera centres onto their GPS-derived ENU positions.

    Reuses ``geometry.submap.umeyama_alignment`` (Sim(3) via Umeyama) --
    the same degenerate-collinear-input detection that module already
    implements for submap merging applies identically here (a single
    flight strip's camera centres are exactly the near-collinear point set
    that alignment's ``degenerate`` flag exists to catch), so we do not
    duplicate that logic.

    Parameters
    ----------
    poses:
        Reconstructed (post-BA) camera poses, in the reconstruction's own
        local metric frame.
    camera_gps_enu:
        ``(len(poses), 3)`` GPS-derived camera-centre positions in the same
        local ENU frame (e.g. via ``ingest.telemetry.telemetry_to_enu``
        resampled to keyframe timestamps).
    fixed_rotation:
        When given, rotation is **not** re-estimated from ``poses``/
        ``camera_gps_enu`` at all -- only scale + translation are fit
        (``geometry.submap.umeyama_fixed_rotation``), with rotation fixed
        at this value. This matters, not just for API symmetry: on a
        near-collinear single-flight-strip track (exactly the case
        ``pipeline.stages.GeometryStage`` picks the ``"telemetry_rotation"``
        submap-merge strategy for -- see ``geometry.submap``'s module
        docstring), ``poses`` by this point already carry a *correct*
        world-frame orientation (from that same telemetry-rotation merge,
        further refined by bundle adjustment's gravity/GPS priors) --
        re-deriving rotation here via Umeyama on the same near-collinear
        camera centres would just replace that correct orientation with
        fitting noise, silently re-introducing the exact tilt the submap
        -merge fix exists to prevent. Pass ``np.eye(3)`` in that case (see
        ``pipeline.stages._apply_georeferencing``), since ``poses`` are
        already expressed in the world/ENU frame -- there is no further
        rotation to apply, only whatever residual scale/translation
        georeferencing still needs to correct.

    Returns
    -------
    ``(transform, residuals_m)``: the Sim(3) mapping reconstruction-frame
    points into the GPS/ENU frame, and the per-camera post-alignment
    residual distance (metres) -- both RANSAC inliers and outliers, so a
    caller can see which cameras disagreed with the fit.
    """
    src = np.array([p.t for p in poses], dtype=np.float64)
    dst = np.asarray(camera_gps_enu, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError(f"poses/camera_gps_enu length mismatch: {src.shape} vs {dst.shape}")

    n = src.shape[0]
    min_n = 2 if fixed_rotation is not None else 3
    if n < min_n:
        raise ValueError(f"align_to_gps needs >= {min_n} cameras with GPS, got {n}")

    if fixed_rotation is not None:
        rng = np.random.default_rng(seed)
        best_inliers: np.ndarray | None = None
        for _ in range(_RANSAC_ITERS):
            sample_idx = rng.choice(n, size=2, replace=False)
            candidate = umeyama_fixed_rotation(src[sample_idx], dst[sample_idx], fixed_rotation)
            residuals = np.linalg.norm(candidate.apply(src) - dst, axis=1)
            inliers = residuals < _RANSAC_INLIER_THRESHOLD_M
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers

        if best_inliers is None or best_inliers.sum() < 2:
            transform = umeyama_fixed_rotation(src, dst, fixed_rotation)
        else:
            transform = umeyama_fixed_rotation(src[best_inliers], dst[best_inliers], fixed_rotation)

        residuals_m = np.linalg.norm(transform.apply(src) - dst, axis=1)
        return transform, residuals_m

    rng = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(_RANSAC_ITERS):
        sample_idx = rng.choice(n, size=3, replace=False)
        try:
            candidate, _, _ = umeyama_alignment(src[sample_idx], dst[sample_idx])
        except ValueError:
            continue
        residuals = np.linalg.norm(candidate.apply(src) - dst, axis=1)
        inliers = residuals < _RANSAC_INLIER_THRESHOLD_M
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 3:
        transform, _, _ = umeyama_alignment(src, dst)
    else:
        transform, _, _ = umeyama_alignment(src[best_inliers], dst[best_inliers])

    residuals_m = np.linalg.norm(transform.apply(src) - dst, axis=1)
    return transform, residuals_m


# ---------------------------------------------------------------------------
# Scale error
# ---------------------------------------------------------------------------


def scale_error_estimate(poses: list[Pose], camera_gps_enu: np.ndarray) -> float:
    """Percentage scale error between reconstructed and GPS-derived inter-camera distances.

    This is the real metric-accuracy indicator for a single-pass
    reconstruction: unlike absolute position (bounded by GPS bias -- see
    module docstring), the *ratio* of distances between many camera pairs
    to their GPS-derived counterparts is not biased by a constant GPS
    offset (a uniform translation bias cancels out of pairwise distances
    entirely), so this isolates genuine reconstruction scale error from
    GPS's absolute-position bias.

    Computed as the median ratio of all pairwise reconstructed distances
    to all pairwise GPS distances (median, not mean, to be robust to the
    occasional GPS outlier fix), reported as a percentage deviation from
    1.0 (0% = perfect scale agreement).
    """
    recon = np.array([p.t for p in poses], dtype=np.float64)
    gps = np.asarray(camera_gps_enu, dtype=np.float64).reshape(-1, 3)
    if recon.shape != gps.shape:
        raise ValueError(f"poses/camera_gps_enu length mismatch: {recon.shape} vs {gps.shape}")
    n = recon.shape[0]
    if n < 2:
        raise ValueError("scale_error_estimate needs >= 2 cameras")

    iu = np.triu_indices(n, k=1)
    recon_dist = np.linalg.norm(recon[iu[0]] - recon[iu[1]], axis=1)
    gps_dist = np.linalg.norm(gps[iu[0]] - gps[iu[1]], axis=1)

    valid = gps_dist > 1e-6
    if not valid.any():
        return 0.0
    ratios = recon_dist[valid] / gps_dist[valid]
    median_ratio = float(np.median(ratios))
    return (median_ratio - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Georeference() result
# ---------------------------------------------------------------------------


@dataclass
class GeoreferenceResult:
    """Output of ``georeference``: the transform plus two separately-reported accuracy figures.

    relative_accuracy_m:
        Estimated internal/metric accuracy (1-sigma, metres) -- how
        well-shaped the reconstruction is, independent of its absolute
        position on Earth. Derived from BA/alignment residuals; can
        legitimately be sub-metre.
    absolute_accuracy_m:
        Estimated absolute positioning accuracy (1-sigma, metres) of the
        georeferenced output. Bounded below by ``gps_bias_estimate_m``
        whenever ``gps_grade`` is not RTK -- see module docstring. Never
        sub-metre for standalone GPS input, by construction (see
        ``georeference``'s honesty-path assertion).
    gps_bias_estimate_m:
        Estimated systematic GPS bias driving the absolute accuracy floor.
        For standalone GPS this is a fixed conservative literature value
        (``_STANDALONE_GPS_BIAS_M``) since a single flight has no
        independent way to measure its own GPS's systematic bias; for
        RTK/PPK input it is the (small) reported/derived accuracy figure.
    gps_grade:
        Detected input GPS quality tier (see ``detect_gps_grade``).
    scale_error_pct:
        See ``scale_error_estimate``.
    """

    transform: Sim3
    crs: str
    relative_accuracy_m: float
    absolute_accuracy_m: float
    gps_bias_estimate_m: float
    gps_grade: GpsGrade
    scale_error_pct: float
    alignment_residuals_m: np.ndarray
    origin: GeoPoint
    gcp_residuals_m: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def georeference(
    point_cloud: PointCloud,
    poses: list[Pose],
    telemetry: list[TelemetrySample],
    origin: GeoPoint | None = None,
    gcps: list[tuple[np.ndarray, GeoPoint]] | None = None,
    relative_accuracy_m: float | None = None,
    fixed_rotation: np.ndarray | None = None,
) -> GeoreferenceResult:
    """Align a reconstruction to GPS/WGS84 and report relative vs. absolute accuracy honestly.

    Parameters
    ----------
    point_cloud:
        Reconstructed point cloud in the reconstruction's local metric
        frame (used only for its size; the transform is what a caller
        then applies to it).
    poses:
        Reconstructed camera poses, local metric frame, one per telemetry
        sample used for alignment (same order/length as the geo-tagged
        subset of ``telemetry`` -- callers should have already resampled
        telemetry onto keyframe timestamps, e.g. via
        ``ingest.telemetry.resample_telemetry``).
    telemetry:
        Per-camera telemetry samples (same length/order as ``poses``),
        used both for the GPS alignment target and for GPS-grade / bias
        detection (see ``detect_gps_grade``).
    origin:
        ENU origin for the alignment frame; defaults to the first
        geo-tagged sample.
    fixed_rotation:
        Passed straight through to ``align_to_gps`` (see its docstring) --
        fixes the GPS-based alignment's rotation instead of re-estimating
        it from (possibly near-collinear) camera centres. Ignored once
        ``gcps`` (below) has >= 3 entries and supersedes the GPS-based fit
        entirely -- GCPs are a separate, generally non-collinear
        correspondence set with no reason to inherit this fix.
    gcps:
        Optional list of ``(local_xyz, geo_point)`` ground control point
        correspondences. If given (and there are >= 3, enough for a full
        Sim(3) refit), they supersede the GPS-based alignment entirely --
        a surveyed GCP is an independent, far more accurate absolute
        reference than the drone's own (possibly standalone-grade) GPS, so
        it should actually correct the transform, not just be reported as
        a residual against a GPS-only fit. With 1-2 GCPs (not enough for a
        full refit), the GPS-derived rotation/scale is kept and only
        translation is corrected to match the GCP(s) on average.
    relative_accuracy_m:
        Internal/metric accuracy of ``poses``' reconstruction (1-sigma,
        metres), if already known from an upstream stage -- e.g.
        ``covariance.point_covariances``' typical semi-axis, or bundle
        adjustment's converged reprojection RMSE converted to a metric
        error via triangulation geometry. This is the *right* source of
        truth for relative accuracy (BA is what buys sub-metre metric
        accuracy -- see this module's docstring), so when given, it is
        used directly. If omitted, ``georeference`` falls back to the
        median camera-position residual from GPS alignment -- but that
        fallback conflates GPS noise with reconstruction quality (it is
        literally "how far does the aligned reconstruction sit from
        GPS," which is dominated by GPS's own noise, not reconstruction
        error) and should be treated as an upper bound, not a genuine
        internal-accuracy estimate, whenever GPS is noisy.

    Critical honesty requirement (read this before trusting the output)
    -----------------------------------------------------------------------
    ``absolute_accuracy_m`` is **never** reported below
    ``_STANDALONE_GPS_BIAS_M`` unless the input telemetry's reported GPS
    accuracy indicates RTK/PPK-grade correction (``GpsGrade.RTK``, via
    ``detect_gps_grade``) or accurate GCPs were supplied. This is enforced
    in code (not just documented) by ``_absolute_accuracy_m`` below --
    see ``tests/test_georef.py``'s honesty-path tests.
    """
    from drishti3d.ingest.telemetry import telemetry_to_enu

    geo_samples = [s for s in telemetry if s.geo is not None]
    if len(geo_samples) < 3:
        raise ValueError(f"georeference needs >= 3 geo-tagged telemetry samples, got {len(geo_samples)}")
    if len(telemetry) != len(poses):
        raise ValueError("telemetry and poses must be the same length/order (one sample per pose)")

    camera_gps_enu, resolved_origin = telemetry_to_enu(telemetry, origin=origin)
    valid = ~np.isnan(camera_gps_enu).any(axis=1)
    valid_poses = [p for p, v in zip(poses, valid, strict=True) if v]
    valid_gps = camera_gps_enu[valid]

    # GPS-based alignment first (needed regardless -- it's what
    # scale_error_pct and the camera-position relative-accuracy residuals
    # are measured against), then, if GCPs were supplied, let them
    # override/correct that alignment below: a surveyed GCP is an
    # independent, far more accurate absolute reference than the drone's
    # own (possibly standalone-grade) GPS, so it should actually influence
    # the transform, not just be reported as a residual against a
    # GPS-only fit (which would silently inherit GPS's own bias/noise into
    # a number that's supposed to demonstrate GCPs *fixing* that).
    transform, _ = align_to_gps(valid_poses, valid_gps, fixed_rotation=fixed_rotation)
    scale_error_pct = scale_error_estimate(valid_poses, valid_gps)

    gps_grade, median_accuracy_h = detect_gps_grade(telemetry)

    gcp_residuals_m = None
    gcp_rmse: float | None = None
    if gcps:
        local_xyz = np.array([g[0] for g in gcps])
        geo_targets = np.array([[g[1].lon, g[1].lat, g[1].alt_msl] for g in gcps])
        geo_targets_enu = wgs84_to_enu(geo_targets, resolved_origin)

        if len(gcps) >= 3:
            # Enough correspondences to refit the *entire* Sim(3) directly
            # against the (trusted) GCPs, superseding the GPS-based fit.
            transform, _, _ = umeyama_alignment(local_xyz, geo_targets_enu)
        else:
            # Too few GCPs for a full refit (Umeyama needs >= 3
            # non-collinear points) -- keep the GPS-derived
            # rotation/scale (still the best available estimate of
            # orientation/scale) and apply a translation-only correction
            # so the transform is exact at the mean GCP position. This
            # matches standard practice for "a couple of checkpoints,
            # not a full control network."
            transformed = transform.apply(local_xyz)
            correction = np.mean(geo_targets_enu - transformed, axis=0)
            transform = Sim3(scale=transform.scale, R=transform.R, t=transform.t + correction)

        transformed = transform.apply(local_xyz)
        gcp_residuals_m = np.linalg.norm(transformed - geo_targets_enu, axis=1)
        gcp_rmse = float(np.sqrt(np.mean(gcp_residuals_m**2)))

    # Camera-position residuals against GPS, using whichever transform we
    # ended up with (GPS-only, or GCP-corrected above) -- reported
    # regardless (useful diagnostic either way), but only used as the
    # *relative accuracy estimate itself* when the caller hasn't supplied
    # a proper internal-accuracy figure (see the parameter docstring for
    # why this fallback conflates GPS noise with reconstruction quality).
    residuals_m = np.linalg.norm(transform.apply(np.array([p.t for p in valid_poses])) - valid_gps, axis=1)
    relative_accuracy_is_fallback = relative_accuracy_m is None
    if relative_accuracy_m is None:
        relative_accuracy_m = float(np.median(residuals_m)) if len(residuals_m) else 0.0
    # Relative accuracy also can't be reported as *better* than the
    # alignment noise floor actually observed; fall back to a small
    # nonzero epsilon rather than exactly 0.0 for a perfect synthetic fit,
    # since "exactly zero uncertainty" is never a defensible claim.
    relative_accuracy_m = max(relative_accuracy_m, 1e-6)

    absolute_accuracy_m, gps_bias_estimate_m, notes = _absolute_accuracy_m(
        gps_grade, median_accuracy_h, relative_accuracy_m, gcp_rmse
    )

    # The fallback residual above is computed *after* `transform` -- which
    # includes a Sim(3) scale correction -- has already been applied to
    # `valid_poses`. When that correction is substantial, the fallback
    # legitimately measures post-alignment consistency (a real, if
    # GPS-noise-limited, quantity), but it is NOT the same thing as "this
    # reconstruction's own raw metric scale was accurate" -- a model that
    # was internally, say, 14x too small before this fit would still show
    # a small residual here, because Sim(3) alignment absorbs a uniform
    # scale error exactly. Reporting a tight relative_accuracy_m next to a
    # large scale_error_pct without this note is exactly the "internally
    # inconsistent report" failure mode this module exists to prevent (see
    # module docstring) -- so when both apply, say so explicitly rather
    # than let a reader conflate "well-shaped after correction" with
    # "was accurate to begin with."
    if relative_accuracy_is_fallback and abs(scale_error_pct) > 1.0:
        notes.append(
            f"relative_accuracy_m falls back to the post-alignment camera-position residual "
            f"against GPS (no internal/BA-derived estimate was supplied). That alignment "
            f"includes a {scale_error_pct:+.1f}% scale correction (see scale_error_pct), so "
            "this figure reflects post-correction consistency, not the raw reconstruction's "
            "own metric scale error -- do not read it as evidence the uncorrected scale was fine."
        )

    crs = estimate_utm_crs(resolved_origin.lat, resolved_origin.lon)

    return GeoreferenceResult(
        transform=transform,
        crs=crs,
        relative_accuracy_m=relative_accuracy_m,
        absolute_accuracy_m=absolute_accuracy_m,
        gps_bias_estimate_m=gps_bias_estimate_m,
        gps_grade=gps_grade,
        scale_error_pct=scale_error_pct,
        alignment_residuals_m=residuals_m,
        origin=resolved_origin,
        gcp_residuals_m=gcp_residuals_m,
        notes=notes,
    )


def _absolute_accuracy_m(
    gps_grade: GpsGrade,
    median_accuracy_h: float | None,
    relative_accuracy_m: float,
    gcp_rmse: float | None,
) -> tuple[float, float, list[str]]:
    """The one place that decides how good an absolute-accuracy claim is allowed to be.

    This function is the actual enforcement of the "never report sub-metre
    absolute accuracy for standalone GPS" rule -- see module and
    ``georeference`` docstrings. It is deliberately small and isolated so
    it's easy to audit/test on its own (``tests/test_georef.py``'s honesty
    tests call ``georeference`` end-to-end, exercising this).
    """
    notes: list[str] = []

    if gcp_rmse is not None:
        # Surveyed GCPs are an independent absolute reference; if they are
        # available and residuals are small, they -- not the drone's own
        # GPS -- set the absolute accuracy bound, regardless of GPS grade.
        notes.append(
            f"absolute accuracy set by {len(np.atleast_1d(gcp_rmse))} ground control point(s) "
            f"(RMSE {gcp_rmse:.3f} m), not GPS."
        )
        return max(gcp_rmse, relative_accuracy_m), gcp_rmse, notes

    if gps_grade == GpsGrade.RTK:
        bias = median_accuracy_h if median_accuracy_h is not None else _RTK_ACCURACY_THRESHOLD_M
        notes.append(
            f"telemetry reports RTK/PPK-grade GPS accuracy ({bias:.3f} m); "
            "absolute accuracy is bounded by that, not the standalone-GPS floor -- provided the RTK "
            "base / NTRIP position is itself surveyed and the video-to-log clock alignment is exact "
            "(a residual clock offset shifts the model along the track by speed x offset)."
        )
        return max(bias, relative_accuracy_m), bias, notes

    # GpsGrade.STANDALONE or GpsGrade.UNKNOWN: assume the conservative
    # standalone bound. UNKNOWN is deliberately treated the same as
    # STANDALONE (not as "unconstrained"/optimistic) -- absence of an
    # accuracy field is not evidence of RTK quality.
    bias = median_accuracy_h if median_accuracy_h is not None else _STANDALONE_GPS_BIAS_M
    # Reported accuracy_h claims better than our RTK threshold would
    # already have been classified as RTK above; anything in between (or
    # an UNKNOWN grade, or no reading at all) is still standalone-like, so
    # clamp to the conservative literature bound rather than trust an
    # optimistic manufacturer number.
    bias = max(bias, _STANDALONE_GPS_BIAS_M)
    notes.append(
        f"standalone GPS input (no RTK/PPK correction detected): absolute accuracy is bounded "
        f"by an assumed {bias:.1f} m systematic GPS bias, which bundle adjustment and dense "
        "reconstruction cannot remove -- this figure will not go below that no matter how good "
        "the internal reconstruction is."
    )
    return max(bias, relative_accuracy_m), bias, notes
