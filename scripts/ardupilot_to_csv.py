"""Convert an ArduPilot DataFlash log (`.BIN`) into a telemetry CSV this
pipeline can already ingest.

Why a converter and not a new parser in `ingest.telemetry`
-----------------------------------------------------------
`ingest.telemetry` already reads three formats (SRT, CSV, GPX) and its
`_match_columns` pass is deliberately tolerant about header spelling. A
fourth parser would buy nothing except a hard dependency on `pymavlink`
-- an ArduPilot-specific library an order of magnitude larger than the
rest of the ingest layer, pulled in for a format that only appears when
someone hands us an open-autopilot log rather than a DJI sidecar.

So this script stays outside the package, like `benchmark_vram.py`:
`uv run --with pymavlink` supplies the dependency for exactly as long as
the conversion takes, and the pipeline keeps ingesting plain CSV.

The headers below are not arbitrary. Each one is chosen to land on a role
in `telemetry._ROLE_HEADER_PRIORITY` without an alias being added there:

    latitude       -> lat
    longitude      -> lon
    altitude       -> alt_msl
    rel_alt        -> alt_rel
    time           -> timestamp
    gimbal_pitch   -> gimbal_pitch
    gimbal_roll    -> gimbal_roll
    gimbal_yaw     -> gimbal_yaw

Which message supplies what
----------------------------
- `GPS` (5 Hz here): `Lat`, `Lng`, `Alt`. `Alt` is AMSL in metres.
- `ATT` (25 Hz): airframe attitude -- `Roll`, `Pitch`, `Yaw`.
- `MNT` (10 Hz): mount/gimbal attitude, same three fields.

GPS is the slowest stream, so it is the master clock and the other two
interpolate onto it. The alternative -- upsampling GPS to 25 Hz -- would
manufacture position fixes that were never measured, and every one of
them would then be weighted equally with a real fix by bundle adjustment.

Camera attitude: `MNT` if present, `ATT` otherwise
---------------------------------------------------
On a gimballed aircraft these differ by exactly the thing the gimbal is
there to remove, so `MNT` is correct and `ATT` is not. But a fixed-mount
rig logs no `MNT` at all, and there the airframe *is* the camera. The
fallback is therefore right for fixed mounts and wrong for gimballed
ones, which is why taking it prints a warning rather than passing
silently.

Height above takeoff is not height above ground
------------------------------------------------
`rel_alt` is computed as `Alt - Alt[first 3D fix]`, i.e. height above the
takeoff point. `depth_anchor` wants height above the terrain *currently
under the aircraft*. Those agree only while the ground stays level with
the launch point.

For PinPoint flight01 (flat farmland, 100-127 m AGL) that holds. For
flight02 (a river and a hill, 66-97 m AGL) it does not, and `rel_alt`
from this script will be wrong by the local relief -- use a DTM there, or
let the pipeline anchor on its own ground plane instead.

Video-relative time is deliberately NOT applied here
------------------------------------------------------
`TelemetrySample.timestamp` means "seconds from video start", and
`ingest.telemetry.load_telemetry` is the one place that converts a
source file's raw clock into it. This script emits seconds since the
first 3D fix and stops there; pass the real offset to that function:

    load_telemetry(csv_path, time_offset_s=1.2)   # PinPoint campaign.json

which records `offset_source="explicit"` instead of the `assumed_zero`
warning a bare flight-log CSV would otherwise earn.

Usage
-----
    uv run --with pymavlink python scripts/ardupilot_to_csv.py \\
        flight01/test_flight01.BIN -o flight01_telemetry.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# A 3D fix. ArduPilot's GPS.Status uses the u-blox convention, where 0/1
# are no-fix, 2 is 2D (no usable altitude) and >=3 is 3D. The PinPoint
# README specifies Status >= 3 for exactly this reason.
_FIX_3D = 3

_OUT_HEADERS = [
    "time",
    "latitude",
    "longitude",
    "altitude",
    "rel_alt",
    "gimbal_pitch",
    "gimbal_roll",
    "gimbal_yaw",
    "satellites",
]


def _read_messages(path: Path, types: tuple[str, ...]) -> dict[str, list[dict]]:
    """Collect the requested message types, each ordered by TimeUS."""
    try:
        from pymavlink import mavutil
    except ImportError:
        sys.exit(
            "pymavlink is not installed. Run this script as:\n"
            "  uv run --with pymavlink python scripts/ardupilot_to_csv.py ..."
        )

    log = mavutil.mavlink_connection(str(path))
    out: dict[str, list[dict]] = {t: [] for t in types}

    while True:
        msg = log.recv_match(type=list(types))
        if msg is None:
            break
        out[msg.get_type()].append(msg.to_dict())

    for records in out.values():
        records.sort(key=lambda r: r.get("TimeUS", 0))
    return out


def _interpolate(
    records: list[dict], field: str, query_us: list[float]
) -> list[float | None]:
    """Sample `field` at each query time, without extrapolating past the ends.

    Angles are interpolated as plain scalars, which is correct for roll and
    pitch (bounded well inside +/-180) but wraps badly for yaw crossing the
    +/-180 boundary. Yaw is therefore unwrapped by the caller before it
    reaches here.
    """
    if not records:
        return [None] * len(query_us)

    times = [r["TimeUS"] for r in records]
    values = [r.get(field) for r in records]
    out: list[float | None] = []
    i = 0

    for t in query_us:
        if t < times[0] or t > times[-1]:
            out.append(None)
            continue
        while i + 1 < len(times) and times[i + 1] < t:
            i += 1
        j = min(i + 1, len(times) - 1)
        t0, t1 = times[i], times[j]
        v0, v1 = values[i], values[j]
        if v0 is None or v1 is None:
            out.append(None)
        elif t1 == t0:
            out.append(float(v0))
        else:
            frac = (t - t0) / (t1 - t0)
            out.append(float(v0) + frac * (float(v1) - float(v0)))

    return out


def _unwrap_yaw(records: list[dict], field: str) -> None:
    """Make yaw continuous in place, so linear interpolation can't wrap.

    A heading stepping 179 -> -179 is a 2 degree turn, but interpolating
    those two numbers directly sweeps 358 degrees the wrong way around.
    Unwrapping first turns the sequence into one that is safe to blend;
    the caller re-wraps at the end.
    """
    prev = None
    offset = 0.0
    for r in records:
        v = r.get(field)
        if v is None:
            continue
        v = float(v)
        if prev is not None:
            delta = v - prev
            if delta > 180.0:
                offset -= 360.0
            elif delta < -180.0:
                offset += 360.0
        prev = v
        r[field] = v + offset


def _wrap_180(value: float | None) -> float | None:
    if value is None:
        return None
    return ((value + 180.0) % 360.0) - 180.0


def convert(bin_path: Path, out_path: Path) -> dict:
    msgs = _read_messages(bin_path, ("GPS", "ATT", "MNT"))

    gps = [r for r in msgs["GPS"] if r.get("Status", 0) >= _FIX_3D]
    if not gps:
        sys.exit(f"no GPS records with a 3D fix (Status >= {_FIX_3D}) in {bin_path}")

    # Camera attitude source. See the module docstring: MNT is the gimbal,
    # ATT is the airframe, and they are only interchangeable on a fixed mount.
    attitude = msgs["MNT"]
    attitude_source = "MNT"
    if not attitude:
        attitude = msgs["ATT"]
        attitude_source = "ATT"
        print(
            "warning: no MNT (mount) messages in this log -- falling back to ATT\n"
            "         (airframe attitude). Correct for a fixed camera mount,\n"
            "         WRONG for a gimballed one, where the gimbal is removing\n"
            "         exactly the motion ATT records.",
            file=sys.stderr,
        )

    # MNT logs yaw as `YawE` (earth frame, compass degrees) and `YawB`
    # (body frame); there is no plain `Yaw` field, so reading "Yaw" from MNT
    # silently produced an empty gimbal_yaw column. Earth-frame is the
    # camera heading this pipeline wants. Older firmware that does log
    # `Yaw` keeps working.
    if attitude_source == "MNT" and attitude and "Yaw" not in attitude[0] and "YawE" in attitude[0]:
        for rec in attitude:
            rec["Yaw"] = rec["YawE"]
    _unwrap_yaw(attitude, "Yaw")

    t0_us = gps[0]["TimeUS"]
    home_alt = float(gps[0]["Alt"])
    query = [float(r["TimeUS"]) for r in gps]

    pitch = _interpolate(attitude, "Pitch", query)
    roll = _interpolate(attitude, "Roll", query)
    yaw = _interpolate(attitude, "Yaw", query)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_OUT_HEADERS)
        for i, rec in enumerate(gps):
            alt = float(rec["Alt"])
            writer.writerow(
                [
                    f"{(float(rec['TimeUS']) - t0_us) / 1e6:.6f}",
                    f"{float(rec['Lat']):.8f}",
                    f"{float(rec['Lng']):.8f}",
                    f"{alt:.3f}",
                    f"{alt - home_alt:.3f}",
                    "" if pitch[i] is None else f"{pitch[i]:.3f}",
                    "" if roll[i] is None else f"{roll[i]:.3f}",
                    "" if yaw[i] is None else f"{_wrap_180(yaw[i]):.3f}",
                    rec.get("NSats", ""),
                ]
            )

    duration_s = (float(gps[-1]["TimeUS"]) - t0_us) / 1e6
    return {
        "rows": len(gps),
        "duration_s": duration_s,
        "rate_hz": len(gps) / duration_s if duration_s else 0.0,
        "attitude_source": attitude_source,
        "home_alt_msl_m": home_alt,
        "alt_rel_max_m": max(float(r["Alt"]) for r in gps) - home_alt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bin_path", type=Path, help="ArduPilot .BIN DataFlash log")
    parser.add_argument(
        "-o", "--out", type=Path, required=True, help="destination CSV path"
    )
    args = parser.parse_args()

    if not args.bin_path.exists():
        sys.exit(f"no such file: {args.bin_path}")

    stats = convert(args.bin_path, args.out)

    print(f"wrote {args.out}")
    print(f"  rows:             {stats['rows']}")
    print(f"  duration:         {stats['duration_s']:.1f} s")
    print(f"  GPS rate:         {stats['rate_hz']:.1f} Hz")
    print(f"  attitude source:  {stats['attitude_source']}")
    print(f"  takeoff altitude: {stats['home_alt_msl_m']:.1f} m AMSL")
    print(f"  max height above takeoff: {stats['alt_rel_max_m']:.1f} m")
    print(
        "\nremember: timestamps here are seconds since the first 3D fix, NOT\n"
        "video time. Pass the real offset when loading, e.g.\n"
        "  load_telemetry(path, time_offset_s=1.2)   # PinPoint campaign.json"
    )


if __name__ == "__main__":
    main()
