"""Survey control import and independent reconstruction validation.

Control points and check points have deliberately different types and
lifecycles here.  Controls are allowed into the georeferencing fit.  Check
points are withheld and must name an independently identified point in the
reconstruction; searching for the nearest cloud vertex is not a measurement
of positional accuracy on a continuous surface.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from drishti3d.types import GeoPoint


@dataclass(frozen=True)
class SurveyPoint:
    id: str
    local_xyz: np.ndarray
    geo: GeoPoint


@dataclass(frozen=True)
class ControlPointNetwork:
    measurement_set_id: str
    controls: tuple[SurveyPoint, ...]
    checkpoints: tuple[SurveyPoint, ...]

    def georeference_gcps(self) -> list[tuple[np.ndarray, GeoPoint]]:
        """Return controls only, in the shape accepted by ``georeference``."""
        return [(p.local_xyz.copy(), p.geo) for p in self.controls]


def _finite_xyz(value: Any, label: str) -> np.ndarray:
    xyz = np.asarray(value, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"{label}.local_xyz must contain three finite metre coordinates")
    return xyz


def _parse_points(values: Any, role: str) -> tuple[SurveyPoint, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{role} must be a JSON array")
    parsed: list[SurveyPoint] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, dict):
            raise ValueError(f"{role}[{index}] must be an object")
        point_id = raw.get("id")
        if not isinstance(point_id, str) or not point_id.strip():
            raise ValueError(f"{role}[{index}].id must be a non-empty string")
        geo = raw.get("geo")
        if not isinstance(geo, dict):
            raise ValueError(f"{role}[{index}].geo must be an object")
        try:
            lat = float(geo["lat"])
            lon = float(geo["lon"])
            alt = float(geo["alt_msl"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{role}[{index}].geo needs numeric lat, lon and alt_msl") from exc
        if not all(math.isfinite(v) for v in (lat, lon, alt)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{role}[{index}].geo is outside WGS84 bounds or is non-finite")
        parsed.append(
            SurveyPoint(
                id=point_id.strip(),
                local_xyz=_finite_xyz(raw.get("local_xyz"), f"{role}[{index}]"),
                geo=GeoPoint(lat=lat, lon=lon, alt_msl=alt),
            )
        )
    return tuple(parsed)


def _reject_degenerate_controls(controls: tuple[SurveyPoint, ...]) -> None:
    if len(controls) < 3:
        raise ValueError(f"at least 3 control points are required for a Sim(3) fit; got {len(controls)}")
    from drishti3d.geometry.georef import wgs84_to_enu

    local = np.stack([p.local_xyz for p in controls])
    origin = controls[0].geo
    geographic = np.array([[p.geo.lon, p.geo.lat, p.geo.alt_msl] for p in controls], dtype=np.float64)
    target = wgs84_to_enu(geographic, origin)
    # Three points must span a plane. Collinear/coincident networks leave a
    # 3-D similarity rotation ambiguous even if a numerical solver returns.
    if np.linalg.matrix_rank(local - local.mean(axis=0), tol=1e-8) < 2:
        raise ValueError("control local_xyz positions are collinear or coincident")
    if np.linalg.matrix_rank(target - target.mean(axis=0), tol=1e-5) < 2:
        raise ValueError("control surveyed positions are collinear or coincident")


def _reject_duplicate_positions(points: tuple[SurveyPoint, ...]) -> None:
    """Prevent one physical observation being relabelled into two roles."""
    if len(points) < 2:
        return
    from drishti3d.geometry.georef import wgs84_to_enu

    local = np.stack([p.local_xyz for p in points])
    llh = np.array([[p.geo.lon, p.geo.lat, p.geo.alt_msl] for p in points], dtype=np.float64)
    surveyed = wgs84_to_enu(llh, points[0].geo)
    for i in range(len(points)):
        for j in range(i):
            if np.linalg.norm(local[i] - local[j]) < 1e-8:
                raise ValueError(f"points {points[j].id!r} and {points[i].id!r} duplicate the same local_xyz")
            if np.linalg.norm(surveyed[i] - surveyed[j]) < 1e-4:
                raise ValueError(f"points {points[j].id!r} and {points[i].id!r} duplicate the same surveyed position")


def load_control_points(path: str | Path) -> ControlPointNetwork:
    """Load the explicit JSON control/checkpoint schema and validate roles."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read control-point JSON {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("coordinate_frame") != "reconstruction_local_m":
        raise ValueError("coordinate_frame must be 'reconstruction_local_m'")
    controls = _parse_points(raw.get("controls"), "controls")
    checkpoints = _parse_points(raw.get("checkpoints"), "checkpoints")
    measurement_set_id = raw.get("measurement_set_id")
    if not isinstance(measurement_set_id, str) or not measurement_set_id.strip():
        raise ValueError(
            "measurement_set_id must identify this run's freshly identified local checkpoint measurements"
        )
    control_ids = [p.id for p in controls]
    checkpoint_ids = [p.id for p in checkpoints]
    if len(set(control_ids)) != len(control_ids):
        raise ValueError("control point IDs must be unique")
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("checkpoint IDs must be unique")
    overlap = sorted(set(control_ids) & set(checkpoint_ids))
    if overlap:
        raise ValueError(f"control and checkpoint IDs must be disjoint; repeated: {', '.join(overlap)}")
    _reject_duplicate_positions(controls + checkpoints)
    _reject_degenerate_controls(controls)
    return ControlPointNetwork(
        measurement_set_id=measurement_set_id.strip(), controls=controls, checkpoints=checkpoints
    )


def transformed_checkpoint_xyz(network: ControlPointNetwork, transform: Any) -> np.ndarray:
    """Apply the fitted georeferencing transform to withheld local points."""
    if not network.checkpoints:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(transform.apply(np.stack([p.local_xyz for p in network.checkpoints])), dtype=np.float64)


def _target_enu(network: ControlPointNetwork, origin: GeoPoint) -> np.ndarray:
    from drishti3d.geometry.georef import wgs84_to_enu

    llh = np.array([[p.geo.lon, p.geo.lat, p.geo.alt_msl] for p in network.checkpoints], dtype=np.float64)
    return wgs84_to_enu(llh, origin)


def checkpoint_metrics(
    network: ControlPointNetwork,
    predicted_enu: np.ndarray,
    origin: GeoPoint,
    *,
    target_m: float = 1.0,
    minimum_checkpoints: int = 3,
) -> dict[str, Any]:
    """Score named withheld correspondences in the final exported ENU frame."""
    predicted = np.asarray(predicted_enu, dtype=np.float64)
    if predicted.shape != (len(network.checkpoints), 3) or not np.all(np.isfinite(predicted)):
        raise ValueError("predicted checkpoint coordinates must be finite Nx3 values in final ENU metres")
    if not math.isfinite(target_m) or target_m <= 0:
        raise ValueError("accuracy target_m must be positive and finite")
    if not isinstance(minimum_checkpoints, int) or isinstance(minimum_checkpoints, bool) or minimum_checkpoints < 1:
        raise ValueError("minimum_checkpoints must be a positive integer")
    expected = _target_enu(network, origin)
    residual = predicted - expected
    horizontal = np.linalg.norm(residual[:, :2], axis=1)
    vertical = np.abs(residual[:, 2])
    error_3d = np.linalg.norm(residual, axis=1)
    n = len(network.checkpoints)
    if n:
        h_rmse = float(np.sqrt(np.mean(horizontal**2)))
        v_rmse = float(np.sqrt(np.mean(vertical**2)))
        h_worst = float(np.max(horizontal))
        v_worst = float(np.max(vertical))
        rmse_3d = float(np.sqrt(np.mean(error_3d**2)))
        worst_3d = float(np.max(error_3d))
    else:
        h_rmse = v_rmse = h_worst = v_worst = rmse_3d = worst_3d = None

    enough = n >= minimum_checkpoints
    # The headline <= target claim uses the strongest, unambiguous test:
    # every full 3-D checkpoint error must be inside the target radius.
    meets = bool(enough and worst_3d <= target_m)  # type: ignore[operator]
    assessment = "pass" if meets else ("fail" if enough else "not established")
    one_m_meets = bool(enough and worst_3d <= 1.0)  # type: ignore[operator]
    one_m_assessment = "pass" if one_m_meets else ("fail" if enough else "not established")
    reason = (
        f"maximum 3-D error across independent checkpoints is <= {target_m:g} m"
        if meets
        else (
            f"maximum 3-D error across independent checkpoints exceeds {target_m:g} m"
            if enough
            else f"need at least {minimum_checkpoints} independent checkpoints; got {n}"
        )
    )
    rows = [
        {
            "id": point.id,
            "east_error_m": float(err[0]),
            "north_error_m": float(err[1]),
            "up_error_m": float(err[2]),
            "horizontal_error_m": float(h),
            "vertical_error_m": float(v),
            "error_3d_m": float(np.linalg.norm(err)),
        }
        for point, err, h, v in zip(network.checkpoints, residual, horizontal, vertical, strict=True)
    ]
    return {
        "method": "named withheld survey correspondences",
        "n_checkpoints": n,
        "checkpoint_ids": [p.id for p in network.checkpoints],
        "horizontal_rmse_m": h_rmse,
        "vertical_rmse_m": v_rmse,
        "horizontal_worst_m": h_worst,
        "vertical_worst_m": v_worst,
        "rmse_3d_m": rmse_3d,
        "worst_3d_m": worst_3d,
        "target_m": float(target_m),
        "target_assessment": assessment,
        "meets_target": meets if enough else None,
        "assessment_reason": reason,
        "one_m_target_assessment": one_m_assessment,
        "meets_one_m_target": one_m_meets if enough else None,
        "residuals": rows,
    }


def _history_record(
    network: ControlPointNetwork, metrics: dict[str, Any], run_id: str
) -> dict[str, Any]:
    by_id = {row["id"]: row for row in metrics["residuals"]}
    return {
        "run_id": run_id,
        "measurement_set_id": network.measurement_set_id,
        "checkpoints": [
            {
                "id": p.id,
                "target_wgs84": [p.geo.lon, p.geo.lat, p.geo.alt_msl],
                "error_enu_m": [by_id[p.id]["east_error_m"], by_id[p.id]["north_error_m"], by_id[p.id]["up_error_m"]],
            }
            for p in network.checkpoints
        ],
    }


def update_run_history(
    path: str | Path,
    network: ControlPointNetwork,
    metrics: dict[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist one run and report repeatability across exactly matching surveys."""
    path = Path(path)
    run_id = run_id or network.measurement_set_id
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    raw: dict[str, Any] = {"schema_version": 1, "runs": []}
    if path.exists():
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read accuracy history {path}: {exc}") from exc
        if raw.get("schema_version") != 1 or not isinstance(raw.get("runs"), list):
            raise ValueError("accuracy history has an unsupported schema")
    existing_ids = [run.get("run_id") for run in raw["runs"]]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("accuracy history already contains duplicate run IDs")
    if run_id in existing_ids:
        raise ValueError(f"accuracy history already contains run_id {run_id!r}")
    if any(run.get("measurement_set_id") == network.measurement_set_id for run in raw["runs"]):
        raise ValueError(
            f"measurement_set_id {network.measurement_set_id!r} was already recorded; "
            "repeatability requires freshly identified local checkpoint coordinates from a different reconstruction"
        )

    # Validate all stored errors before mutating history. A corrupt prior
    # run must not be preserved and then used as numerical evidence.
    for run in raw["runs"]:
        points = run.get("checkpoints")
        if not isinstance(points, list):
            raise ValueError("accuracy history contains a run without checkpoint measurements")
        for point in points:
            err = np.asarray(point.get("error_enu_m"), dtype=float)
            if err.shape != (3,) or not np.all(np.isfinite(err)):
                raise ValueError("accuracy history contains non-finite or malformed checkpoint errors")

    current = _history_record(network, metrics, run_id)
    raw["runs"].append(current)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2))
    temporary.replace(path)

    expected_ids = [p.id for p in network.checkpoints]
    expected_targets = {p.id: np.array([p.geo.lon, p.geo.lat, p.geo.alt_msl]) for p in network.checkpoints}
    comparable: list[dict[str, Any]] = []
    for run in raw["runs"]:
        points = run.get("checkpoints")
        if not isinstance(points, list) or [p.get("id") for p in points] != expected_ids:
            continue
        if any(
            not np.allclose(np.asarray(p.get("target_wgs84"), dtype=float), expected_targets[p["id"]], rtol=0, atol=1e-10)
            for p in points
        ):
            continue
        comparable.append(run)
    if len(comparable) < 2 or not expected_ids:
        return {
            "status": "not computed",
            "comparable_runs": len(comparable),
            "reason": "repeatability needs at least two unique runs with the same checkpoint IDs and surveyed coordinates",
        }

    errors = np.array(
        [[[float(v) for v in p["error_enu_m"]] for p in run["checkpoints"]] for run in comparable],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(errors)):
        raise ValueError("accuracy history contains non-finite checkpoint errors")
    centred = errors - errors.mean(axis=0, keepdims=True)
    horizontal = np.linalg.norm(centred[..., :2], axis=-1)
    vertical = np.abs(centred[..., 2])
    return {
        "status": "computed",
        "comparable_runs": len(comparable),
        "run_ids": [run["run_id"] for run in comparable],
        "horizontal_rmse_m": float(np.sqrt(np.mean(horizontal**2))),
        "vertical_rmse_m": float(np.sqrt(np.mean(vertical**2))),
        "horizontal_worst_m": float(np.max(horizontal)),
        "vertical_worst_m": float(np.max(vertical)),
        "definition": "RMS and worst deviation of each named checkpoint prediction from its across-run mean",
    }
