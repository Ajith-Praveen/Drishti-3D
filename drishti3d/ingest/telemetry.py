"""Flight telemetry parsing and resampling for the ingest stage.

Supports the three sources drone footage actually shows up with in
practice: DJI SRT subtitle sidecars (the common case), generic/Airdata CSV
exports, and GPX tracks. All parsing is defensive -- a malformed line or
record is skipped and counted, never raised, because a single bad cue in an
otherwise-good multi-thousand-line SRT file should not abort ingest.

Design note on the shared contract
-----------------------------------
``TelemetrySample`` (drishti3d.types) has no fields for per-shot camera
settings (focal length, ISO, shutter, f-number) -- those are camera state,
not flight telemetry, and don't belong on a per-sample flight record. DJI
SRT cues carry them anyway, so we still parse them defensively (as
required), but surface anything useful (currently: focal length, for
``ingest.intrinsics``) via the ``stats`` dict returned alongside the sample
list rather than inventing new fields on the shared type. The same pattern
now covers an embedded SRT wall-clock line (``stats["wallclock_utc"]``) and
Airdata quality fields that don't belong in ``GeoPoint`` either (see the
CSV section below).

``TelemetrySample.timestamp`` is documented as "seconds from video start,"
and DJI SRT sidecars genuinely satisfy that (one SRT file per video, cue
time is the video's own PTS -- see ``parse_srt_string``). Airdata/generic
flight-log CSV exports do **not**: their ``time`` column is measured from
*flight* start (power-on), which can include ground time, multiple
recordings, hovering, and landing all on one shared timeline. Treating a
CSV's raw ``time`` column as video-relative silently misaligns every
telemetry sample against the video by however long the flight ran before
this particular recording started -- at survey speed that is a large,
purely-silent georeferencing error. ``load_telemetry`` handles this by
auto-detecting (or accepting an explicit) offset and re-basing CSV
timestamps so ``t=0`` really does mean video start before any sample ever
leaves this module -- see ``detect_video_segments``/``load_telemetry``
below.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import logging
import re
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from pyproj import CRS, Transformer

from drishti3d.types import GeoPoint, TelemetrySample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENU / ECEF geodesy helpers (shared by telemetry_to_enu and resample_telemetry)
# ---------------------------------------------------------------------------

_WGS84_LLA = CRS.from_epsg(4979)  # geodetic lat/lon/height
_ECEF = CRS.from_epsg(4978)  # earth-centered, earth-fixed
_LLA_TO_ECEF = Transformer.from_crs(_WGS84_LLA, _ECEF, always_xy=True)
_ECEF_TO_LLA = Transformer.from_crs(_ECEF, _WGS84_LLA, always_xy=True)


def _ecef_from_geo(lat: float, lon: float, alt: float) -> np.ndarray:
    x, y, z = _LLA_TO_ECEF.transform(lon, lat, alt)
    return np.array([x, y, z], dtype=np.float64)


def _enu_rotation_matrix(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Rotation matrix mapping an ECEF offset vector into local ENU axes."""
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )


def telemetry_to_enu(
    samples: list[TelemetrySample], origin: GeoPoint | None = None
) -> tuple[np.ndarray, GeoPoint]:
    """Convert geo-tagged samples to a local ENU (East-North-Up) metre frame.

    Returns ``(enu_xyz, origin)`` -- the origin is returned (rather than
    silently assumed) so callers can georeference reconstructed geometry
    back to WGS84 later. Samples without a ``geo`` fix produce a row of
    NaNs rather than being dropped, so the output stays index-aligned with
    ``samples``.

    Uses the standard geodesy route (WGS84 lat/lon/height -> ECEF -> local
    ENU via a rotation about the origin) rather than a flat-earth
    approximation, since pyproj gives us exact ECEF for free and drone
    flights can cover enough ground that equirectangular approximations
    would start to matter.
    """
    if origin is None:
        first_geo = next((s.geo for s in samples if s.geo is not None), None)
        if first_geo is None:
            raise ValueError("telemetry_to_enu: no georeferenced samples and no origin given")
        origin = first_geo

    origin_ecef = _ecef_from_geo(origin.lat, origin.lon, origin.alt_msl)
    rotation = _enu_rotation_matrix(origin.lat, origin.lon)

    out = np.full((len(samples), 3), np.nan, dtype=np.float64)
    for i, sample in enumerate(samples):
        if sample.geo is None:
            continue
        ecef = _ecef_from_geo(sample.geo.lat, sample.geo.lon, sample.geo.alt_msl)
        out[i] = rotation @ (ecef - origin_ecef)

    return out, origin


def _enu_to_geo(enu_xyz: np.ndarray, origin: GeoPoint) -> GeoPoint:
    origin_ecef = _ecef_from_geo(origin.lat, origin.lon, origin.alt_msl)
    rotation = _enu_rotation_matrix(origin.lat, origin.lon)
    ecef = origin_ecef + rotation.T @ enu_xyz
    lon, lat, alt = _ECEF_TO_LLA.transform(ecef[0], ecef[1], ecef[2])
    return GeoPoint(lat=lat, lon=lon, alt_msl=alt)


def estimate_ground_speed(samples: list[TelemetrySample]) -> np.ndarray:
    """Per-sample horizontal ground speed (m/s), via central differences in ENU.

    Deliberately horizontal-only (East/North components): "ground speed" is
    a 2D concept, and mixing in vertical rate here would conflate climb/dive
    with actual ground coverage, which is what matters for judging frame
    overlap during triage.
    """
    n = len(samples)
    speeds = np.zeros(n, dtype=np.float64)
    geo_idx = [i for i, s in enumerate(samples) if s.geo is not None]
    if len(geo_idx) < 2:
        return speeds

    enu, _ = telemetry_to_enu(samples)
    timestamps = np.array([s.timestamp for s in samples], dtype=np.float64)

    for k, i in enumerate(geo_idx):
        i_prev = geo_idx[k - 1] if k > 0 else None
        i_next = geo_idx[k + 1] if k < len(geo_idx) - 1 else None
        if i_prev is not None and i_next is not None:
            dt = timestamps[i_next] - timestamps[i_prev]
            d = enu[i_next, :2] - enu[i_prev, :2]
        elif i_next is not None:
            dt = timestamps[i_next] - timestamps[i]
            d = enu[i_next, :2] - enu[i, :2]
        elif i_prev is not None:
            dt = timestamps[i] - timestamps[i_prev]
            d = enu[i, :2] - enu[i_prev, :2]
        else:
            continue
        speeds[i] = float(np.linalg.norm(d) / dt) if dt > 0 else 0.0

    return speeds


def trajectory_length(samples: list[TelemetrySample]) -> float:
    """Total 3D distance flown (metres), summed over consecutive geo-tagged samples.

    Unlike ``estimate_ground_speed``, this includes the vertical component:
    a trajectory's *length* is the actual path length through space (e.g.
    an orbit that climbs matters for how much scene it covers), whereas
    "ground speed" is specifically about horizontal coverage rate.
    """
    geo_samples = [s for s in samples if s.geo is not None]
    if len(geo_samples) < 2:
        return 0.0
    ordered = sorted(geo_samples, key=lambda s: s.timestamp)
    enu, _ = telemetry_to_enu(ordered)
    diffs = np.diff(enu, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _interp_scalar(valid_t: np.ndarray, valid_v: np.ndarray, t: float) -> float | None:
    if len(valid_t) == 0:
        return None
    if t < valid_t[0] or t > valid_t[-1]:
        return None
    return float(np.interp(t, valid_t, valid_v))


def resample_telemetry(
    samples: list[TelemetrySample], timestamps: list[float]
) -> list[TelemetrySample | None]:
    """Interpolate telemetry to arbitrary query timestamps (e.g. keyframe times).

    Returns ``None`` for any query time outside ``[min(sample.timestamp),
    max(sample.timestamp)]`` rather than extrapolating -- extrapolated pose
    is worse than no pose for a reconstruction prior, since it can be
    confidently wrong in a way "no telemetry for this frame" isn't.

    Position is interpolated in a local ENU metre frame (via
    ``telemetry_to_enu``), not in raw lat/lon degrees: degrees are not a
    metric space (a degree of longitude shrinks toward the poles), so
    linearly blending lat/lon directly would bias the interpolated path,
    especially over the longer/slower flights where resampling matters
    most.
    """
    if not samples or not timestamps:
        return [None for _ in timestamps]

    ordered = sorted(samples, key=lambda s: s.timestamp)
    t_arr = np.array([s.timestamp for s in ordered], dtype=np.float64)
    t_min, t_max = float(t_arr[0]), float(t_arr[-1])

    have_geo = any(s.geo is not None for s in ordered)
    enu_arr: np.ndarray | None = None
    origin: GeoPoint | None = None
    alt_rel_arr = np.full(len(ordered), np.nan, dtype=np.float64)
    if have_geo:
        enu_arr, origin = telemetry_to_enu(ordered)
        for i, s in enumerate(ordered):
            if s.geo is not None and s.geo.alt_rel is not None:
                alt_rel_arr[i] = s.geo.alt_rel

    def _field(getter: object) -> np.ndarray:
        return np.array(
            [np.nan if getter(s) is None else float(getter(s)) for s in ordered],  # type: ignore[operator]
            dtype=np.float64,
        )

    pitch_arr = _field(lambda s: s.gimbal_pitch)
    roll_arr = _field(lambda s: s.gimbal_roll)
    yaw_arr = _field(lambda s: s.gimbal_yaw)
    baro_arr = _field(lambda s: s.baro_alt)

    results: list[TelemetrySample | None] = []
    for t in timestamps:
        if t < t_min or t > t_max:
            results.append(None)
            continue

        geo = None
        if have_geo and enu_arr is not None and origin is not None:
            mask = ~np.isnan(enu_arr).any(axis=1)
            valid_t = t_arr[mask]
            if len(valid_t) >= 1 and valid_t[0] <= t <= valid_t[-1]:
                valid_enu = enu_arr[mask]
                interp_enu = np.array(
                    [np.interp(t, valid_t, valid_enu[:, k]) for k in range(3)]
                )
                geo = _enu_to_geo(interp_enu, origin)
                alt_rel_valid_mask = mask & ~np.isnan(alt_rel_arr)
                if alt_rel_valid_mask.any():
                    geo.alt_rel = _interp_scalar(t_arr[alt_rel_valid_mask], alt_rel_arr[alt_rel_valid_mask], t)

        def _interp_masked(values: np.ndarray, query: float = t) -> float | None:
            mask = ~np.isnan(values)
            return _interp_scalar(t_arr[mask], values[mask], query)

        results.append(
            TelemetrySample(
                timestamp=t,
                geo=geo,
                gimbal_pitch=_interp_masked(pitch_arr),
                gimbal_roll=_interp_masked(roll_arr),
                gimbal_yaw=_interp_masked(yaw_arr),
                baro_alt=_interp_masked(baro_arr),
            )
        )

    return results


# ---------------------------------------------------------------------------
# DJI SRT
# ---------------------------------------------------------------------------

_CUE_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->")
_KV_RE = re.compile(r"([A-Za-z_]+)\s*:\s*(-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)")
_WALLCLOCK_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")

_LAT_KEYS = ("latitude", "lat")
_LON_KEYS = ("longitude", "long", "lon")


def _parse_numeric_token(value: str) -> float:
    value = value.strip()
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse_srt_string(text: str) -> tuple[list[TelemetrySample], dict]:
    """Parse a DJI SRT telemetry sidecar (as a string) into ``TelemetrySample``s.

    DJI firmware versions disagree wildly on cue layout (bracketed
    ``[key : value]`` tokens vs. bare ``key: value`` text, differing key
    spellings, HTML wrapper tags, optional wall-clock lines). Rather than
    modeling a fixed layout, we regex for ``key: value`` tokens anywhere in
    the cue body and map whichever recognized spellings show up. A cue that
    fails to parse (e.g. a corrupt timecode line) is skipped and counted,
    never raised, since one bad cue in thousands of lines shouldn't sink
    the whole load.

    Cue start time (from the SRT timecode line) is authoritative for
    ``timestamp``; an embedded wall-clock line, if present, is not used for
    timing (it's frequently only second-resolution and drifts from the
    video's actual PTS). It genuinely is video-relative timing though --
    unlike an Airdata CSV (see module docstring), an SRT sidecar is
    produced per-video by the aircraft alongside that exact recording, so
    cue time 00:00:00 really is this video's first frame; no offset
    detection is needed or applied here.

    Some firmware versions do embed an absolute wall-clock datetime on
    each cue (matched by ``_WALLCLOCK_RE``); some workflows (e.g.
    cross-referencing against another log, or PPK correction) need that
    absolute time even though it isn't used for in-video sync here, so it
    is captured per-sample and surfaced as ``stats["wallclock_utc"]`` --
    parallel to the returned sample list, ``None`` for any cue that didn't
    carry one -- rather than silently discarded.
    """
    stats = {"format": "dji_srt", "records_total": 0, "records_parsed": 0, "records_skipped": 0}
    focal_lengths: list[float] = []
    wallclocks: list[str | None] = []

    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    samples: list[TelemetrySample] = []

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        stats["records_total"] += 1
        try:
            timecode_line = next((ln for ln in lines if "-->" in ln), None)
            if timecode_line is None:
                raise ValueError(f"no timecode line in cue: {lines!r}")
            m = _CUE_TIME_RE.search(timecode_line)
            if m is None:
                raise ValueError(f"unparseable timecode: {timecode_line!r}")
            h, mn, sec, ms = m.groups()
            ms_val = int(ms.ljust(3, "0")[:3])
            timestamp = int(h) * 3600 + int(mn) * 60 + int(sec) + ms_val / 1000.0

            body_lines = [ln for ln in lines if ln is not timecode_line and not ln.strip().isdigit()]
            body = " ".join(body_lines)
            body = re.sub(r"<[^>]+>", " ", body)  # strip <font ...>/</font> wrapper tags

            kv: dict[str, float] = {}
            for key, value in _KV_RE.findall(body):
                key_norm = key.strip().lower()
                if key_norm in kv:
                    continue
                try:
                    kv[key_norm] = _parse_numeric_token(value)
                except ValueError:
                    continue

            lat = next((kv[k] for k in _LAT_KEYS if k in kv), None)
            lon = next((kv[k] for k in _LON_KEYS if k in kv), None)
            rel_alt = kv.get("rel_alt")
            abs_alt = kv.get("abs_alt", kv.get("altitude"))

            geo = None
            if lat is not None and lon is not None:
                geo = GeoPoint(
                    lat=lat,
                    lon=lon,
                    alt_msl=abs_alt if abs_alt is not None else (rel_alt if rel_alt is not None else 0.0),
                    alt_rel=rel_alt,
                )

            if "focal_len" in kv:
                focal_lengths.append(kv["focal_len"])

            wallclock_match = _WALLCLOCK_RE.search(body)
            wallclocks.append(wallclock_match.group(0) if wallclock_match else None)

            samples.append(TelemetrySample(timestamp=timestamp, geo=geo))
            stats["records_parsed"] += 1
        except Exception:  # noqa: BLE001 - one bad cue must never abort the whole load
            stats["records_skipped"] += 1
            continue

    if focal_lengths:
        stats["focal_len_mm"] = float(np.median(focal_lengths))
    if any(w is not None for w in wallclocks):
        stats["wallclock_utc"] = wallclocks

    return samples, stats


# ---------------------------------------------------------------------------
# CSV (Airdata-style / generic exports)
# ---------------------------------------------------------------------------

# role -> normalized header names that can supply it, in *preference*
# order. Preference order matters (and is why this isn't a flat
# name->role dict scanned in CSV column order): a real Airdata export
# reports both an aircraft-body reading and a camera/gimbal reading under
# separate headers for pitch (``pitch(degrees)`` vs
# ``gimbal_pitch(degrees)``) and yaw (``compass_heading(degrees)`` vs
# ``gimbal_heading(degrees)``), and ``TelemetrySample.gimbal_pitch``/
# ``gimbal_yaw`` are documented as the camera gimbal's own attitude, not
# the aircraft's. Scanning in CSV column order would pick whichever
# happens to come first in the file -- on the real Airdata export used to
# validate this module, that's the *aircraft* columns, which silently
# feeds the wrong angle into pose estimation. Listing the gimbal-specific
# name first here means it always wins when both are present, regardless
# of column order; the aircraft-body reading is only used as a fallback
# when no dedicated gimbal column exists (true for roll -- DJI/Airdata
# exports a gimbal heading and pitch but not a gimbal roll).
_ROLE_HEADER_PRIORITY: dict[str, tuple[str, ...]] = {
    "lat": ("latitude", "lat"),
    "lon": ("longitude", "long", "lon"),
    "alt_msl": ("altitude_above_sealevel", "altitudeabovesealevel", "altitude", "abs_alt"),
    "alt_rel": ("height_above_takeoff", "relative_altitude", "altitude_above_ground", "rel_alt", "height"),
    "timestamp": ("time", "time_stamp", "timestamp", "elapsed_time"),
    "gimbal_pitch": ("gimbal_pitch", "pitch"),
    "gimbal_roll": ("gimbal_roll", "roll"),
    "gimbal_yaw": ("gimbal_heading", "gimbal_yaw", "compass_heading"),
}

# Recording-state flag columns Airdata exports carry -- true exactly for
# the rows recorded while a given video/photo capture was active. Used
# only for ``detect_video_segments``, never folded into ``TelemetrySample``
# (there's no field for it there -- it's a video-sync signal, not flight
# state). "isvideo" is preferred when both are present: this module's
# offset-detection is specifically about aligning *video* time.
_CSV_FLAG_LOOKUP = {"isvideo": "is_video", "isphoto": "is_photo"}

# GPS quality fields Airdata reports. Deliberately NOT mapped into
# ``GeoPoint.accuracy_h``/``accuracy_v``: ``satellites`` is a raw
# constellation count and ``gpslevel`` is DJI's coarse 0-5 signal-strength
# bar, not a manufacturer-specified accuracy-in-metres figure -- there is
# no documented, trustworthy formula from either to a metre value. Since
# ``geometry.georef.py`` uses ``accuracy_h`` to decide whether a
# reconstruction may legitimately claim centimetre-level *absolute*
# accuracy (RTK-grade) vs. the conservative several-metre standalone-GPS
# bound (see that module's docstring), inventing a plausible-looking metre
# number here risks the exact dishonest-accuracy-claim failure that module
# exists to prevent. They're still surfaced -- as
# ``stats["satellites_median"]``/``stats["gpslevel_median"]`` -- so a
# caller can eyeball GPS health without this module fabricating a
# precision it cannot back up.
_CSV_QUALITY_LOOKUP = {"satellites": "satellites", "gpslevel": "gpslevel"}

_FEET_UNIT_TOKENS = ("feet", "ft")

_TRUTHY_FLAG_VALUES = {"1", "true", "yes", "y"}


def _parse_bool_flag(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY_FLAG_VALUES


def _normalize_header(raw_header: str) -> tuple[str, str | None]:
    key = raw_header.strip().lower()
    key = re.sub(r"\s+", "_", key)
    m = re.match(r"^([a-z0-9_]+)(?:\(([^)]*)\))?$", key)
    if not m:
        return key, None
    name, unit = m.groups()
    return name, (unit.strip().lower() if unit else None)


def _match_columns(fieldnames: list[str]) -> dict[str, tuple[str, str | None]]:
    """Map each recognized role to the raw header/unit that should supply it.

    Two passes: first normalize every header once (first raw header wins
    for a given normalized name, in case of an exact duplicate), then walk
    ``_ROLE_HEADER_PRIORITY`` role by role, taking the first candidate name
    present -- in *priority* order, not CSV column order (see that table's
    docstring for why that distinction matters for gimbal vs. aircraft-body
    columns).
    """
    normalized: dict[str, tuple[str, str | None]] = {}
    for raw_header in fieldnames:
        name, unit = _normalize_header(raw_header)
        if name not in normalized:
            normalized[name] = (raw_header, unit)

    columns: dict[str, tuple[str, str | None]] = {}
    for role, candidates in _ROLE_HEADER_PRIORITY.items():
        for candidate in candidates:
            if candidate in normalized:
                columns[role] = normalized[candidate]
                break
    return columns


@dataclasses.dataclass
class VideoSegment:
    """One contiguous run of an Airdata-style recording-state flag (e.g. ``isVideo``)."""

    start_s: float
    end_s: float
    duration_s: float


def detect_video_segments(rows: list[dict]) -> list[VideoSegment]:
    """Find contiguous recording-active runs in flight-log rows.

    ``rows`` (aka "samples_or_rows") is a list of ``{"timestamp": float,
    "is_video": bool}`` dicts, in chronological (file) order -- the
    normalized shape ``_parse_csv`` builds from an Airdata-style CSV's
    ``time``/``isVideo`` (or ``isPhoto``) columns. Returns one
    ``VideoSegment`` per maximal run where ``is_video`` is ``True``, in
    the order the runs occur; ``[]`` if ``rows`` is empty or never active.

    A real flight log commonly contains more than one such run (the pilot
    stopped and restarted recording, or shot several clips in one
    flight) -- this deliberately returns *all* of them rather than
    guessing which one matters; ``load_telemetry`` is what picks a run to
    align to, by matching duration against the video actually being
    processed.
    """
    segments: list[VideoSegment] = []
    run_start: float | None = None
    prev_t: float | None = None
    for row in rows:
        t = float(row["timestamp"])
        active = bool(row["is_video"])
        if active and run_start is None:
            run_start = t
        if not active and run_start is not None and prev_t is not None:
            segments.append(VideoSegment(start_s=run_start, end_s=prev_t, duration_s=prev_t - run_start))
            run_start = None
        prev_t = t
    if run_start is not None and prev_t is not None:
        segments.append(VideoSegment(start_s=run_start, end_s=prev_t, duration_s=prev_t - run_start))
    return segments


def _parse_csv(path: Path) -> tuple[list[TelemetrySample], dict]:
    """Parse an Airdata/generic CSV telemetry export.

    Headers are matched case-insensitively and tolerant of a trailing unit
    annotation (e.g. ``altitude(feet)``, ``time(millisecond)``); recognized
    units are converted (feet -> metres, milliseconds -> seconds) so the
    rest of the pipeline never has to think about source units again.

    ``timestamp`` here is still whatever the CSV's own clock is (flight
    start for a real Airdata export, per the module docstring) -- this
    function does not know the video's duration and cannot itself decide
    a video-start offset. It only detects and reports the raw material for
    that decision: ``stats["video_segments"]`` (via ``detect_video_segments``,
    when an ``isVideo``/``isPhoto`` column is present) and
    ``stats["recording_flag_column"]``. ``load_telemetry`` is what turns
    those into an actual offset and re-bases the returned samples.
    """
    stats: dict = {"format": "csv", "records_total": 0, "records_parsed": 0, "records_skipped": 0}

    try:
        raw_text = path.read_text(errors="replace")
    except OSError:
        logger.warning("Could not read telemetry CSV %s", path, exc_info=True)
        return [], stats

    try:
        dialect = csv.Sniffer().sniff(raw_text[:4096])
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(raw_text), dialect=dialect)
    if reader.fieldnames is None:
        return [], stats

    columns = _match_columns(reader.fieldnames)

    normalized_flags: dict[str, str] = {}
    normalized_quality: dict[str, str] = {}
    for raw_header in reader.fieldnames:
        name, _unit = _normalize_header(raw_header)
        if name in _CSV_FLAG_LOOKUP and name not in normalized_flags:
            normalized_flags[name] = raw_header
        if name in _CSV_QUALITY_LOOKUP and name not in normalized_quality:
            normalized_quality[name] = raw_header

    flag_header: str | None = None
    if "isvideo" in normalized_flags:
        flag_header = normalized_flags["isvideo"]
        stats["recording_flag_column"] = "isVideo"
    elif "isphoto" in normalized_flags:
        flag_header = normalized_flags["isphoto"]
        stats["recording_flag_column"] = "isPhoto"

    satellites_vals: list[float] = []
    gpslevel_vals: list[float] = []

    samples: list[TelemetrySample] = []
    flag_rows: list[dict] = []
    for row in reader:
        stats["records_total"] += 1
        try:
            values: dict[str, float] = {}
            for role, (raw_header, unit) in columns.items():
                raw_val = row.get(raw_header)
                if raw_val is None or not str(raw_val).strip():
                    continue
                v = float(raw_val)
                if role in ("alt_msl", "alt_rel") and unit and any(tok in unit for tok in _FEET_UNIT_TOKENS):
                    v *= 0.3048
                if role == "timestamp" and unit and "milli" in unit:
                    v /= 1000.0
                values[role] = v

            if "timestamp" not in values:
                raise ValueError("row missing a recognizable timestamp column")

            if flag_header is not None:
                raw_flag = row.get(flag_header)
                if raw_flag is not None and str(raw_flag).strip():
                    flag_rows.append({"timestamp": values["timestamp"], "is_video": _parse_bool_flag(str(raw_flag))})

            for norm_name, raw_header_q in normalized_quality.items():
                raw_q = row.get(raw_header_q)
                if raw_q is None or not str(raw_q).strip():
                    continue
                try:
                    q = float(raw_q)
                except ValueError:
                    continue
                if norm_name == "satellites":
                    satellites_vals.append(q)
                elif norm_name == "gpslevel":
                    gpslevel_vals.append(q)

            geo = None
            if "lat" in values and "lon" in values:
                geo = GeoPoint(
                    lat=values["lat"],
                    lon=values["lon"],
                    alt_msl=values.get("alt_msl", 0.0),
                    alt_rel=values.get("alt_rel"),
                    # accuracy_h/accuracy_v deliberately left None -- see
                    # _CSV_QUALITY_LOOKUP's docstring on why satellites/
                    # gpslevel are not converted into a metres figure here.
                )

            samples.append(
                TelemetrySample(
                    timestamp=values["timestamp"],
                    geo=geo,
                    gimbal_pitch=values.get("gimbal_pitch"),
                    gimbal_roll=values.get("gimbal_roll"),
                    gimbal_yaw=values.get("gimbal_yaw"),
                )
            )
            stats["records_parsed"] += 1
        except Exception:  # noqa: BLE001 - one bad row must never abort the whole load
            stats["records_skipped"] += 1
            continue

    if flag_header is not None:
        stats["video_segments"] = detect_video_segments(flag_rows)
    if satellites_vals:
        stats["satellites_median"] = float(np.median(satellites_vals))
    if gpslevel_vals:
        stats["gpslevel_median"] = float(np.median(gpslevel_vals))

    return samples, stats


# ---------------------------------------------------------------------------
# GPX
# ---------------------------------------------------------------------------


def _parse_iso8601(text: str) -> float:
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def _parse_gpx(path: Path) -> tuple[list[TelemetrySample], dict]:
    """Parse a standard GPX track into ``TelemetrySample``s via stdlib ``xml.etree``.

    GPX timestamps are absolute wall-clock; we rebase them to seconds from
    the first track point, matching the "seconds from video start"
    convention (the caller is responsible for time-aligning that origin
    with the video if the GPX log didn't start exactly when recording did).
    """
    stats = {"format": "gpx", "records_total": 0, "records_parsed": 0, "records_skipped": 0}

    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        logger.warning("Could not parse GPX %s", path, exc_info=True)
        return [], stats

    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""

    points = root.findall(f".//{ns}trkpt")
    if not points:
        points = root.findall(f".//{ns}wpt")

    parsed: list[tuple[float, GeoPoint]] = []
    for pt in points:
        stats["records_total"] += 1
        try:
            lat = float(pt.attrib["lat"])
            lon = float(pt.attrib["lon"])
            ele_el = pt.find(f"{ns}ele")
            alt = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
            time_el = pt.find(f"{ns}time")
            if time_el is None or not time_el.text:
                raise ValueError("trkpt missing <time>")
            t = _parse_iso8601(time_el.text)
            parsed.append((t, GeoPoint(lat=lat, lon=lon, alt_msl=alt)))
            stats["records_parsed"] += 1
        except Exception:  # noqa: BLE001 - one bad trkpt must never abort the whole load
            stats["records_skipped"] += 1
            continue

    if not parsed:
        return [], stats

    parsed.sort(key=lambda p: p[0])
    t0 = parsed[0][0]
    samples = [TelemetrySample(timestamp=t - t0, geo=geo) for t, geo in parsed]
    return samples, stats


# ---------------------------------------------------------------------------
# Auto-detecting entry point
# ---------------------------------------------------------------------------


def _resolve_time_offset(
    stats: dict, video_duration_s: float | None, time_offset_s: float | None
) -> tuple[float, str]:
    """Decide the video-start offset per ``load_telemetry``'s documented precedence.

    Returns ``(offset_s, offset_source)``, ``offset_source`` one of
    ``"explicit"``/``"isVideo_autodetect"``/``"assumed_zero"``. Logs a loud
    warning whenever it falls back to ``"assumed_zero"`` for a CSV (SRT/GPX
    genuinely default to 0.0 correctly -- see module docstring -- so those
    stay silent).
    """
    if time_offset_s is not None:
        return float(time_offset_s), "explicit"

    segments: list[VideoSegment] = stats.get("video_segments") or []
    fmt = stats.get("format")

    if segments and video_duration_s is not None and video_duration_s > 0:
        tolerance = max(2.0, 0.01 * video_duration_s)
        matches = [seg for seg in segments if abs(seg.duration_s - video_duration_s) <= tolerance]
        if len(matches) == 1:
            return matches[0].start_s, "isVideo_autodetect"
        if len(matches) > 1:
            logger.warning(
                "telemetry: %d recording segments' durations all match the video duration "
                "(%.2fs +/- %.2fs) -- cannot auto-select one; pass --telemetry-offset explicitly. "
                "Candidates (start_s, duration_s): %s",
                len(matches),
                video_duration_s,
                tolerance,
                [(round(s.start_s, 2), round(s.duration_s, 2)) for s in matches],
            )
        else:
            logger.warning(
                "telemetry: isVideo column present but no recording segment's duration matches the "
                "video duration (%.2fs +/- %.2fs) -- assuming telemetry t=0 is video start, which is "
                "very likely WRONG for a flight-log CSV. Pass --telemetry-offset explicitly. "
                "Segments found (start_s, duration_s): %s",
                video_duration_s,
                tolerance,
                [(round(s.start_s, 2), round(s.duration_s, 2)) for s in segments],
            )
        return 0.0, "assumed_zero"

    if fmt == "csv":
        if segments and video_duration_s is None:
            logger.warning(
                "telemetry: isVideo column present but no video duration was supplied to "
                "load_telemetry() -- cannot auto-detect the video-start offset. Assuming t=0 is video "
                "start, which is very likely WRONG for a flight-log CSV. Pass video_duration_s or an "
                "explicit time_offset_s."
            )
        elif not segments:
            logger.warning(
                "telemetry: no isVideo/isPhoto recording-flag column found in this CSV -- cannot "
                "auto-detect a video-start offset. Assuming t=0 is video start; if this is an "
                "Airdata/flight-log export whose clock runs from flight start (not video start), that "
                "assumption is very likely WRONG. Pass --telemetry-offset explicitly if known."
            )

    return 0.0, "assumed_zero"


def load_telemetry(
    path: str | Path,
    *,
    video_duration_s: float | None = None,
    time_offset_s: float | None = None,
) -> tuple[list[TelemetrySample], dict]:
    """Load a flight-log file, auto-detecting SRT / CSV / GPX by extension or content.

    Returns ``(samples, stats)``; ``stats`` always includes ``"format"``
    and parsed/skipped record counts, so a caller (or the GUI) can surface
    "N of M telemetry lines parsed" without re-deriving it.

    Video-start time alignment
    ---------------------------
    ``TelemetrySample.timestamp`` must mean "seconds from video start" (see
    ``types.TelemetrySample``'s docstring). SRT sidecars and (already
    self-rebased) GPX tracks satisfy that by construction; a flight-log CSV
    generally does not -- see this module's top docstring for why. This
    function is the one place that turns whatever a source file's raw
    clock is into genuine video-relative time before any sample leaves the
    module, with this precedence:

    1. ``time_offset_s`` given explicitly -> used as-is
       (``stats["offset_source"] = "explicit"``); always wins.
    2. Auto-detected from a CSV's ``isVideo``/``isPhoto`` column: if
       parsing found recording segments (``stats["video_segments"]``) and
       ``video_duration_s`` was given, and *exactly one* segment's
       duration matches ``video_duration_s`` within ``max(2.0s, 1%)``,
       that segment's start becomes the offset
       (``stats["offset_source"] = "isVideo_autodetect"``). Zero or
       multiple matching segments are both ambiguous -- neither
       auto-selects; see 3.
    3. Otherwise, ``0.0`` (``stats["offset_source"] = "assumed_zero"``).
       Correct and silent for SRT/GPX; for a CSV this is logged as a loud
       warning, since it is very likely wrong.

    Every returned sample's ``timestamp`` is re-based by the resolved
    offset (``t' = t - offset``) before returning, so downstream code
    never has to know or re-derive it -- ``stats["time_offset_s"]``/
    ``stats["offset_source"]`` record what was done, and
    ``stats["video_segments"]`` (when present) lists every detected
    recording segment (as plain dicts) so a caller can override the
    auto-pick.

    When ``video_duration_s`` is given, ``stats["telemetry_video_coverage_fraction"]``
    additionally reports what fraction of the video's ``[0,
    video_duration_s]`` range the (offset-applied) telemetry span actually
    covers -- less than 1.0 means part of the video has no telemetry at
    all, logged as a warning with both ranges.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".srt":
        samples, stats = parse_srt_string(path.read_text(errors="replace"))
    elif suffix == ".csv":
        samples, stats = _parse_csv(path)
    elif suffix == ".gpx":
        samples, stats = _parse_gpx(path)
    else:
        # Unknown/missing extension: sniff the content.
        try:
            head = path.read_text(errors="replace")[:4096]
        except OSError:
            return [], {"format": "unknown", "records_total": 0, "records_parsed": 0, "records_skipped": 0}

        if re.search(r"\d{2}:\d{2}:\d{2}[,.]\d+\s*-->", head):
            samples, stats = parse_srt_string(path.read_text(errors="replace"))
        elif head.lstrip().startswith("<?xml") or "<gpx" in head.lower():
            samples, stats = _parse_gpx(path)
        else:
            samples, stats = _parse_csv(path)

    offset_s, offset_source = _resolve_time_offset(stats, video_duration_s, time_offset_s)
    stats["time_offset_s"] = offset_s
    stats["offset_source"] = offset_source

    segments = stats.get("video_segments")
    if segments is not None:
        stats["video_segments"] = [dataclasses.asdict(seg) for seg in segments]

    if offset_s != 0.0:
        samples = [dataclasses.replace(s, timestamp=s.timestamp - offset_s) for s in samples]

    if video_duration_s is not None:
        if samples:
            t_values = [s.timestamp for s in samples]
            tele_min, tele_max = min(t_values), max(t_values)
            overlap = max(0.0, min(tele_max, video_duration_s) - max(tele_min, 0.0))
            coverage = overlap / video_duration_s if video_duration_s > 0 else 0.0
            if coverage < 0.999:
                logger.warning(
                    "telemetry: after offsetting (offset=%.2fs, source=%s), telemetry span "
                    "[%.2f, %.2f]s does not fully cover the video's [0.00, %.2f]s range "
                    "(%.1f%% covered) -- keyframes outside the covered range will have no "
                    "telemetry/pose prior.",
                    offset_s,
                    offset_source,
                    tele_min,
                    tele_max,
                    video_duration_s,
                    coverage * 100.0,
                )
        else:
            coverage = 0.0
        stats["telemetry_video_coverage_fraction"] = coverage

    return samples, stats
