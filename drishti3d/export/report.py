"""The accuracy report card: the analyst-facing summary of how much to trust this model.

One rule dominates this whole module, stated up front because it is easy
to violate by accident while trying to make a report "look complete":
**never fabricate an accuracy number that was not measured.** A defence
or disaster analyst making a decision off this model needs to be able to
tell "we measured 4cm RMSE" apart from "we don't actually know" -- and a
report card that quietly writes ``0.0`` for a metric no stage computed
looks exactly like the first case while meaning the second. Every metric
below is either the real measured value or the literal string
``"not computed"``; there is no third option.

``build_report`` is deliberately forgiving about its input: pipeline
stages can fail, be skipped, or simply not produce a given metric (e.g. no
independent check points were surveyed for this site, or bundle
adjustment was skipped because there weren't enough overlapping views),
and none of that should crash report generation -- it should just show up
as "not computed" in the card.
"""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from drishti3d.types import Confidence, GeoPoint, PointCloud

__all__ = [
    "NOT_COMPUTED",
    "build_report",
    "check_point_residuals",
    "render_report_html",
    "render_report_text",
]

# The one and only value a metric that wasn't measured is allowed to hold.
NOT_COMPUTED = "not computed"

#: Ratio of the largest to the smallest per-window depth-anchor
#: correction beyond which the windows are no longer describing the same
#: scene at the same scale. Every window sees the same flight with the
#: same camera and backbone, so a well-behaved run sits near 1.0x; this
#: threshold is deliberately loose, because the failure it is meant to
#: catch is measured in whole multiples, not percentages.
_ANCHOR_SPREAD_WARN = 1.5

# Top-level scalar metrics build_report will look for directly on the
# input dict, each defaulting to NOT_COMPUTED when absent. Kept as a table
# (rather than repeating the same four lines five times) so adding a new
# passthrough metric is a one-line change.
_SCALAR_METRICS: tuple[tuple[str, str], ...] = (
    ("outcome", "outcome"),
    ("diagnostic_output", "diagnostic_output"),
    ("quality_reasons", "quality_reasons"),
    ("relative_rmse_m", "relative_rmse_m"),
    ("absolute_rmse_m", "absolute_rmse_m"),
    ("scale_error_pct", "scale_error_pct"),
    ("mean_reprojection_error_px", "mean_reprojection_error_px"),
    ("coverage_pct", "coverage_pct"),
    ("keyframe_count", "keyframe_count"),
    # Which source actually produced the confidence breakdown below:
    # "ba_covariance" (the principled path -- see
    # geometry.covariance.confidence_from_covariance) or
    # "backbone_confidence_and_view_count" (the fallback fusion.tsdf uses
    # when no per-point covariance is available). See fusion.tsdf.fuse_submaps'
    # docstring for the full reconciliation rule between the two.
    ("confidence_source", "confidence_source"),
    # Telemetry/video time-sync provenance (see ingest.telemetry
    # .load_telemetry's docstring): whether the video-start offset applied
    # to the telemetry feeding this reconstruction was given explicitly,
    # auto-detected from an isVideo/isPhoto column, or just assumed to be
    # zero -- the last of which is a guess, not a measurement, for any CSV
    # telemetry source, and this report card must say so plainly rather
    # than silently presenting a guessed sync as equivalent to a real one.
    ("telemetry_offset_s", "telemetry_offset_s"),
    ("telemetry_offset_source", "telemetry_offset_source"),
    ("telemetry_video_coverage_fraction", "telemetry_video_coverage_fraction"),
    ("telemetry_format", "telemetry_format"),
    # TimeSyncStage's image-motion clock measurement (ingest.timesync).
    ("time_sync", "time_sync"),
    # Height-field multi-view stereo diagnostics (geometry.heightfield), when
    # the dense surface was measured that way rather than by the backbone.
    ("dense_surface", "dense_surface"),
    ("reconstruction_representation", "reconstruction_representation"),
    # Which submap-merge strategy GeometryStage actually used (see
    # geometry.flight_profile/geometry.submap) and why -- a reader
    # comparing junction_residuals across runs needs to know this, since
    # the three strategies' diagnostics mean different things (e.g.
    # "gps_anchored" per-junction rmse_m is a cross-check, not what
    # determined the transform; "chained_sim3" it is).
    ("merge_strategy", "merge_strategy"),
    ("merge_strategy_reason", "merge_strategy_reason"),
)

# offset_source == "assumed_zero" is only an actual *guess* -- worth a loud
# "unverified" warning on the report card -- for a CSV/Airdata-style
# source, whose raw clock is not guaranteed video-relative (see
# ingest.telemetry's module docstring). For SRT (and GPX, self-rebased at
# parse time) an offset of 0.0 is correct by construction, not an
# assumption, so warning about it there would just be a false alarm that
# undercuts the honest cases this same field is meant to flag.
_OFFSET_GUESS_RISK_FORMATS = frozenset({"csv"})


def _confidence_breakdown_from_array(confidence: np.ndarray) -> dict[str, float] | str:
    confidence = np.asarray(confidence)
    total = confidence.shape[0]
    if total == 0:
        return NOT_COMPUTED
    return {
        "measured_pct": float(100.0 * np.sum(confidence == Confidence.MEASURED) / total),
        "low_confidence_pct": float(100.0 * np.sum(confidence == Confidence.LOW_CONFIDENCE) / total),
        "inferred_pct": float(100.0 * np.sum(confidence == Confidence.INFERRED) / total),
    }


def build_report(pipeline_artifacts: dict[str, Any]) -> dict[str, Any]:
    """Assemble the accuracy report card from whatever the pipeline actually produced.

    ``pipeline_artifacts`` is a loosely-typed grab-bag (mirroring
    ``pipeline.result.StageResult.artifacts``) -- every key below is
    optional, and a missing/``None`` one becomes ``"not computed"``, never
    a fabricated number:

    - ``relative_rmse_m``, ``absolute_rmse_m``, ``scale_error_pct``,
      ``mean_reprojection_error_px``, ``coverage_pct``, ``keyframe_count``,
      ``confidence_source``: passed through as-is when present.
    - Confidence breakdown, checked in this order: a precomputed
      ``confidence_breakdown_pct`` dict; failing that, a
      ``mesh_stats`` dict (as returned by
      ``fusion.mesh.compute_mesh_stats``) with its own
      ``confidence_breakdown_pct``; failing that, a raw per-vertex/point
      ``confidence`` array to compute the percentages from directly.
    - ``junction_residuals``: per-junction submap alignment diagnostics,
      typically ``geometry.submap.alignment_residuals``'s return value.
    - ``stage_timings_s``: a ``{stage_name: seconds}`` dict.
    """
    report: dict[str, Any] = {}

    for out_key, in_key in _SCALAR_METRICS:
        value = pipeline_artifacts.get(in_key)
        report[out_key] = value if value is not None else NOT_COMPUTED

    from drishti3d.pipeline.quality import OUTCOMES

    if report["outcome"] not in OUTCOMES:
        report["outcome"] = "unverified"
    report["diagnostic_output"] = report["outcome"] != "valid"
    if not isinstance(report["quality_reasons"], list):
        report["quality_reasons"] = []

    confidence_breakdown: dict[str, float] | str | None = None
    if pipeline_artifacts.get("confidence_breakdown_pct") is not None:
        confidence_breakdown = pipeline_artifacts["confidence_breakdown_pct"]
    elif pipeline_artifacts.get("mesh_stats") is not None:
        confidence_breakdown = pipeline_artifacts["mesh_stats"].get("confidence_breakdown_pct")
    elif pipeline_artifacts.get("confidence") is not None:
        confidence_breakdown = _confidence_breakdown_from_array(pipeline_artifacts["confidence"])

    report["confidence_breakdown_pct"] = confidence_breakdown if confidence_breakdown is not None else NOT_COMPUTED

    junction_residuals = pipeline_artifacts.get("junction_residuals")
    report["junction_residuals"] = junction_residuals if junction_residuals is not None else NOT_COMPUTED

    flight_profile = pipeline_artifacts.get("flight_profile")
    report["flight_profile"] = flight_profile if flight_profile is not None else NOT_COMPUTED

    depth_anchor = pipeline_artifacts.get("depth_anchor")
    report["depth_anchor"] = depth_anchor if depth_anchor else NOT_COMPUTED

    placement = pipeline_artifacts.get("placement")
    report["placement"] = placement if placement else NOT_COMPUTED

    scale_consensus = pipeline_artifacts.get("scale_consensus")
    report["scale_consensus"] = scale_consensus if scale_consensus else NOT_COMPUTED

    pose_prior = pipeline_artifacts.get("pose_prior")
    report["pose_prior"] = pose_prior if pose_prior else NOT_COMPUTED

    # Carried as None (not NOT_COMPUTED) when absent: the renderer treats a
    # falsy value as "no reference configured". Without this line the card
    # said so even after an alignment had been applied to the model.
    report["reference_alignment"] = pipeline_artifacts.get("reference_alignment") or None

    merge_strategy_warnings = pipeline_artifacts.get("merge_strategy_warnings")
    report["merge_strategy_warnings"] = merge_strategy_warnings if merge_strategy_warnings else NOT_COMPUTED

    # geometry.georef.georeference's own explanatory notes (e.g. "absolute
    # accuracy is bounded by an assumed 3.0 m standalone-GPS bias, not a
    # measurement" or a scale-deviation warning) -- surfaced verbatim so a
    # reader of relative_rmse_m/absolute_rmse_m/scale_error_pct above sees
    # *why* those figures are what they are, not just the bare numbers.
    # Passing through a suspiciously-round or suspiciously-precise figure
    # with no context is exactly the "looks like a placeholder" failure
    # mode this module exists to avoid (see module docstring).
    georef_notes = pipeline_artifacts.get("georef_notes")
    report["georef_notes"] = georef_notes if georef_notes else NOT_COMPUTED

    # Scene composition. Kept out of _SCALAR_METRICS because it is a dict
    # (a class histogram), not a flat scalar -- same reason
    # confidence_breakdown_pct is handled separately above. An absent or
    # empty histogram becomes NOT_COMPUTED rather than an all-zero
    # breakdown: "we never classified anything" and "we classified and
    # found nothing" are different claims and this card must not conflate
    # them.
    semantic_hist = pipeline_artifacts.get("semantic_class_pct")
    report["semantic_class_pct"] = semantic_hist if semantic_hist else NOT_COMPUTED

    for key in (
        "dtm_method",
        "ground_point_pct",
        "dtm_interpolated_pct",
        "facade_structures_completed",
        "facade_points_inferred",
        "semantic_labelled_pct",
        "semantic_unseen_points",
        "semantic_disputed_points",
        "semantic_mean_views_per_point",
        "semantic_views_voted",
        "dynamic_points_removed",
        "semantic_model",
    ):
        value = pipeline_artifacts.get(key)
        report[key] = value if value is not None else NOT_COMPUTED

    stage_timings = pipeline_artifacts.get("stage_timings_s") or pipeline_artifacts.get("stage_timings")
    report["stage_timings_s"] = stage_timings if stage_timings is not None else NOT_COMPUTED

    check_points_report = pipeline_artifacts.get("check_point_residuals")
    report["check_point_residuals"] = check_points_report if check_points_report is not None else NOT_COMPUTED

    return report


def check_point_residuals(
    pc_or_mesh: PointCloud | tuple,
    check_points: list[GeoPoint],
    origin: GeoPoint,
) -> dict[str, Any]:
    """Validate the model against independent survey check points.

    Check points are **not** the same thing as GCPs (ground control
    points): GCPs are fed *into* the solve (bundle adjustment/
    georeferencing) to help pin down scale/orientation/position, so
    checking a model against its own GCPs would just be re-measuring how
    well it fit data it already saw. Check points are surveyed
    independently and withheld from solving entirely, specifically so
    comparing the model against them measures genuine *absolute* accuracy
    rather than self-consistency.

    ``check_points`` are ``types.GeoPoint``s (surveyed lat/lon/alt "truth"
    locations); ``origin`` is the ``GeoPoint`` that anchors
    ``pc_or_mesh``'s local ENU frame (see
    ``geometry.georef.wgs84_to_enu``, which this function delegates to).
    For each check point, the nearest point/vertex in ``pc_or_mesh`` is
    found and its horizontal (planar) and vertical distance to the
    check point is recorded as that check point's residual.
    """
    from drishti3d.export.formats import as_geometry
    from drishti3d.geometry.georef import wgs84_to_enu

    xyz, _faces, _colors, _confidence = as_geometry(pc_or_mesh)
    xyz = np.asarray(xyz, dtype=np.float64)

    if len(check_points) == 0 or xyz.shape[0] == 0:
        return {
            "n_check_points": len(check_points),
            "horizontal_rmse_m": NOT_COMPUTED,
            "vertical_rmse_m": NOT_COMPUTED,
            "horizontal_residuals_m": [],
            "vertical_residuals_m": [],
        }

    lonlatalt = np.array([[cp.lon, cp.lat, cp.alt_msl] for cp in check_points])
    expected_enu = wgs84_to_enu(lonlatalt, origin)

    tree = cKDTree(xyz)
    _dist, idx = tree.query(expected_enu, k=1)
    nearest = xyz[idx]

    horizontal = np.linalg.norm(nearest[:, :2] - expected_enu[:, :2], axis=1)
    vertical = nearest[:, 2] - expected_enu[:, 2]

    return {
        "n_check_points": len(check_points),
        "horizontal_rmse_m": float(np.sqrt(np.mean(horizontal**2))),
        "vertical_rmse_m": float(np.sqrt(np.mean(vertical**2))),
        "horizontal_residuals_m": horizontal.tolist(),
        "vertical_residuals_m": vertical.tolist(),
    }


def _time_sync_line(sync: dict) -> str:
    """One report-card line for TimeSyncStage's measurement."""
    lag = sync.get("lag_s")
    measured = sync.get("offset_measured_s")
    if not sync.get("confident"):
        return f"inconclusive ({sync.get('reason', 'no reason recorded')})"
    if sync.get("applied"):
        return f"applied {lag:+.2f} s -> offset {measured:.2f} s ({sync.get('turning_pairs')} turning keyframe pairs)"
    if sync.get("disagrees"):
        return (
            f"WARNING: images put the telemetry {lag:+.2f} s off (measured offset {measured:.2f} s); "
            "the given offset was kept"
        )
    return f"agrees to {lag:+.2f} s"


def _fmt(value: Any, suffix: str = "") -> str:
    if value == NOT_COMPUTED or value is None:
        return NOT_COMPUTED
    if isinstance(value, float):
        return f"{value:.4g}{suffix}"
    return f"{value}{suffix}"


def render_report_text(report: dict[str, Any]) -> str:
    """Render the report card as plain text (console/log-friendly)."""
    lines = ["DRISHTI-3D Accuracy Report Card", "=" * 32, ""]
    lines.append(f"Reconstruction outcome: {report.get('outcome', 'unverified')}")
    lines.append(f"Representation: {report.get('reconstruction_representation') or 'not recorded'}")
    if report.get("outcome") != "valid":
        lines.append("DIAGNOSTIC OUTPUT — not a validated reconstruction")
    reasons = report.get("quality_reasons")
    if isinstance(reasons, list):
        lines.extend(f"  {reason}" for reason in reasons)
    lines.append("")

    lines.append(f"Relative RMSE:            {_fmt(report.get('relative_rmse_m'), ' m')}")
    lines.append(f"Absolute RMSE:            {_fmt(report.get('absolute_rmse_m'), ' m')}")
    lines.append(f"Scale error:              {_fmt(report.get('scale_error_pct'), ' %')}")
    lines.append(f"Mean reprojection error:  {_fmt(report.get('mean_reprojection_error_px'), ' px')}")
    lines.append(f"Coverage:                 {_fmt(report.get('coverage_pct'), ' %')}")
    lines.append(f"Keyframes used:           {_fmt(report.get('keyframe_count'))}")
    lines.append("")

    georef_notes = report.get("georef_notes", NOT_COMPUTED)
    lines.append("Georeferencing notes:")
    if georef_notes == NOT_COMPUTED or not georef_notes:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        for note in georef_notes:
            lines.append(f"  - {note}")
    lines.append("")

    offset_source = report.get("telemetry_offset_source")
    lines.append("Telemetry/video time sync:")
    lines.append(f"  Offset applied:  {_fmt(report.get('telemetry_offset_s'), ' s')}")
    lines.append(f"  Offset source:   {_fmt(offset_source)}")
    if offset_source == "assumed_zero" and report.get("telemetry_format") in _OFFSET_GUESS_RISK_FORMATS:
        lines.append("  WARNING: this offset was not measured -- it is an unverified assumption.")
    sync = report.get("time_sync")
    if isinstance(sync, dict):
        lines.append(f"  Image-motion check: {_time_sync_line(sync)}")
    coverage = report.get("telemetry_video_coverage_fraction")
    if coverage != NOT_COMPUTED and coverage is not None:
        lines.append(f"  Video coverage:  {coverage * 100.0:.1f} %")
        if coverage < 0.999:
            lines.append("  WARNING: telemetry does not cover the full video time range.")
    lines.append("")

    breakdown = report.get("confidence_breakdown_pct", NOT_COMPUTED)
    lines.append(f"Confidence breakdown (source: {_fmt(report.get('confidence_source'))}):")
    if breakdown == NOT_COMPUTED or breakdown is None:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        lines.append(f"  Measured:        {breakdown['measured_pct']:.1f} %")
        lines.append(f"  Low confidence:  {breakdown['low_confidence_pct']:.1f} %")
        lines.append(f"  Inferred:        {breakdown['inferred_pct']:.1f} %")
    lines.append("")

    semantics = report.get("semantic_class_pct", NOT_COMPUTED)
    lines.append("Scene composition (per reconstructed point):")
    if semantics == NOT_COMPUTED or not semantics:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        for name, pct in sorted(semantics.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name + ':':<18} {pct:5.1f} %")
        labelled = report.get("semantic_labelled_pct")
        if labelled not in (None, NOT_COMPUTED):
            lines.append(f"  {'classified:':<18} {labelled:5.1f} % of points")
        disputed = report.get("semantic_disputed_points")
        unseen = report.get("semantic_unseen_points")
        if disputed not in (None, NOT_COMPUTED):
            # Reported separately on purpose: "no view saw it" and "views
            # saw it and disagreed" are different defects with different
            # fixes (flight coverage vs. segmentation quality).
            lines.append(f"  unclassified: {unseen} never seen by a segmented view, {disputed} disputed between views")
        removed = report.get("dynamic_points_removed")
        if removed not in (None, NOT_COMPUTED):
            lines.append(f"  dynamic/sky points removed before fusion: {removed}")
    lines.append("")

    dtm_method = report.get("dtm_method", NOT_COMPUTED)
    lines.append("Terrain model (DTM) and inferred geometry:")
    if dtm_method == NOT_COMPUTED:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        lines.append(f"  {'ground classified by:':<28} {_fmt(dtm_method)}")
        lines.append(f"  {'points classed as ground:':<28} {_fmt(report.get('ground_point_pct'), ' %')}")
        # The number that stops someone measuring a cutting depth off a
        # guess: under every building the DTM is interpolated, not observed.
        lines.append(f"  {'DTM cells interpolated:':<28} {_fmt(report.get('dtm_interpolated_pct'), ' %')}")
        structures = report.get("facade_structures_completed")
        if structures not in (None, NOT_COMPUTED):
            lines.append(f"  {'facades completed:':<28} {structures} structures")
            lines.append(
                f"  {'facade points (INFERRED):':<28} {_fmt(report.get('facade_points_inferred'))}"
                "  -- modelled, not measured; written to facades_inferred.las only"
            )
    lines.append("")

    profile = report.get("flight_profile", NOT_COMPUTED)
    lines.append("Flight profile / merge strategy:")
    if profile == NOT_COMPUTED or not profile:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        lines.append(f"  Gimbal mode:       {_fmt(profile.get('gimbal_mode'))}")
        lines.append(f"  Trajectory shape:  {_fmt(profile.get('trajectory_shape'))}")
        lines.append(f"  Collinearity:      {_fmt(profile.get('collinearity'))}")
        lines.append(f"  Merge strategy:    {_fmt(report.get('merge_strategy'))}")
        lines.append(f"  Strategy reason:   {_fmt(report.get('merge_strategy_reason'))}")
    warnings = report.get("merge_strategy_warnings", NOT_COMPUTED)
    if warnings != NOT_COMPUTED and warnings:
        for warning in warnings:
            lines.append(f"  WARNING: {warning}")
    lines.append("")

    dense = report.get("dense_surface")
    if isinstance(dense, dict):
        secs = dense.get("seconds") or {}

        def _pct(key):
            return _fmt(round(100.0 * dense[key], 1) if dense.get(key) is not None else None)

        lines.append("Dense surface:")
        if dense.get("method") == "mvs3d":
            sgm = " + semi-global matching" if dense.get("sgm_paths") else ""
            refs = dense.get("references")
            refs_txt = f", depth maps from {refs}" if refs is not None and refs != dense.get("views") else ""
            lines.append(
                f"  method:            measured 3D -- plane-sweep stereo per camera{sgm} + TSDF "
                f"({_fmt(dense.get('views'))} views{refs_txt}, {_fmt(dense.get('voxel_m'))} m voxels)"
            )
            lines.append(
                f"  mesh:              {_fmt(dense.get('vertices'))} vertices / {_fmt(dense.get('faces'))} faces, "
                f"{_pct('measured_vertex_fraction')} % MEASURED; {_pct('consistent_fraction')} % of depth samples "
                "agreed across views"
            )
            if dense.get("depth_px_masked"):
                lines.append(
                    f"  masked:            {_fmt(dense.get('depth_px_masked'))} depth samples on vehicles, people or "
                    "sky (semantics stage) were not measured"
                )
            if dense.get("holes_filled_faces"):
                lines.append(
                    f"  gaps closed:       {_fmt(dense.get('holes_filled_faces'))} faces "
                    f"({int(round(float(dense.get('holes_filled_m2') or 0.0))):,} m^2) triangulated across enclosed holes a downward "
                    "camera cannot see (under canopy rims, beside walls) -- INFERRED surface, no new vertices"
                )
        else:
            lines.append(
                f"  method:            height-field multi-view stereo ({_fmt(dense.get('views'))} views, "
                f"{_fmt(dense.get('cell_m'))} m cells, prior: {_fmt(dense.get('prior'))})"
            )
            lines.append(
                f"  reconstructed:     {_fmt(dense.get('cells_reconstructed'))} cells ({_pct('cells_fraction')} % of the grid), "
                f"median NCC {_fmt(dense.get('score_median'))}, {_fmt(dense.get('spikes_removed'))} spikes removed"
            )
            if dense.get("visible_cells"):
                vm, vi = dense.get("visible_measured_fraction") or 0.0, dense.get("visible_inferred_fraction") or 0.0
                lines.append(
                    f"  visible scene:     {_fmt(dense.get('visible_cells'))} cells seen by the cameras; "
                    f"{100.0 * vm:.1f} % measured by stereo, {100.0 * vi:.1f} % filled from the surroundings "
                    f"(INFERRED, no measured uncertainty), {max(0.0, 100.0 * (1.0 - vm - vi)):.1f} % empty"
                )
            sigma = dense.get("height_uncertainty_m")
            if isinstance(sigma, dict):
                lines.append(
                    f"  height uncertainty: median {_fmt(sigma.get('median'))} m, p90 {_fmt(sigma.get('p90'))} m; "
                    f"{_fmt(round(100.0 * sigma['within_1m'], 1) if sigma.get('within_1m') is not None else None)} % of "
                    "measured cells within 1 m (per point in model.las: height_uncertainty_m)"
                )
        lines.append(f"  time:              {_fmt(round(sum(secs.values()), 1) if secs else None)} s on {_fmt(dense.get('device'))}")
        lines.append("")

    prior = report.get("pose_prior", NOT_COMPUTED)
    lines.append("Telemetry alignment (pose prior, before dense geometry):")
    if prior == NOT_COMPUTED or not prior:
        lines.append(f"  {NOT_COMPUTED} -- geometry was conditioned on raw telemetry poses")
    else:
        lines.append(
            f"  yaw from image flow:     {_fmt(prior.get('yaw_from_flow_keyframes'))} keyframes "
            f"(flow - gimbal median {_fmt(prior.get('yaw_flow_minus_telemetry_median_deg'), ' deg')})"
        )
        lines.append(
            f"  sparse BA:               {_fmt(prior.get('ba_points'))} points, "
            f"reprojection RMSE {_fmt(prior.get('rmse_after_px'), ' px')}"
        )
        lines.append(
            f"  poses moved (median):    {_fmt(prior.get('median_position_shift_m'), ' m')} / "
            f"{_fmt(prior.get('median_rotation_change_deg'), ' deg')} from telemetry"
        )
    lines.append("")

    anchor = report.get("depth_anchor", NOT_COMPUTED)
    lines.append("Depth anchoring (backbone depth vs GPS parallax):")
    if anchor == NOT_COMPUTED or not anchor:
        lines.append(f"  {NOT_COMPUTED}")
    elif not anchor.get("enabled"):
        lines.append("  disabled -- backbone depth taken at its own scale")
    else:
        lines.append(f"  windows anchored:  {anchor.get('windows_anchored')}/{anchor.get('windows_total')}")
        lines.append(
            f"  depth ratio (parallax / backbone): median {_fmt(anchor.get('ratio_median'))}, "
            f"min {_fmt(anchor.get('ratio_min'))}, max {_fmt(anchor.get('ratio_max'))}"
        )
        lines.append("  -- a ratio of 1.0 means the backbone's depth already agreed with parallax;")
        lines.append("     anything else is the correction that was applied to that window")
        # Per-window, not just the summary. Every window images the same
        # flight with the same camera and the same backbone, so these
        # should cluster tightly; a wide spread means windows were scaled
        # differently from each other and CANNOT then agree on where the
        # ground is, no matter how the merge is solved. That distinction
        # is invisible in a median/min/max line, and it is the first thing
        # to check when the placement check below fails.
        per_window = anchor.get("per_window") or []
        if per_window:
            lines.append("  per window (ratio | altitude m | backbone ground depth m | samples):")
            for w in per_window:
                ratio, samples = w.get("ratio"), w.get("samples")
                alt, backbone_d = w.get("altitude_m"), w.get("backbone_ground_depth_m")
                lines.append(
                    f"    window {w.get('window'):>3}: "
                    f"{'refused' if ratio is None else f'{ratio:7.3f}'} | "
                    f"{'-' if alt is None else f'{alt:8.1f}'} | "
                    f"{'-' if backbone_d is None else f'{backbone_d:8.1f}'} | "
                    f"{'-' if samples is None else samples:>6}"
                )
            depths = [w["backbone_ground_depth_m"] for w in per_window if w.get("backbone_ground_depth_m")]
            alts = [w["altitude_m"] for w in per_window if w.get("altitude_m")]
            if len(depths) >= 2 and len(alts) >= 2:
                alt_spread = max(alts) / max(min(alts), 1e-9)
                depth_spread = max(depths) / max(min(depths), 1e-9)
                lines.append(
                    f"  altitude varied {alt_spread:.2f}x; backbone's own ground depth varied "
                    f"{depth_spread:.2f}x"
                )
                if depth_spread > _ANCHOR_SPREAD_WARN and alt_spread < _ANCHOR_SPREAD_WARN:
                    lines.append(
                        "  WARNING: the drone held its height but the backbone returned a "
                        f"{depth_spread:.1f}x different scale for the same scene. The per-window"
                    )
                    lines.append(
                        "     corrections above are therefore covering for the backbone, not for "
                        "the flight, and they cannot fully succeed."
                    )
            ratios = [w["ratio"] for w in per_window if w.get("ratio") is not None]
            if len(ratios) >= 2:
                spread = max(ratios) / max(min(ratios), 1e-9)
                lines.append(f"  ratio spread (max/min): {spread:.2f}x")
                if spread > _ANCHOR_SPREAD_WARN:
                    lines.append(
                        f"  WARNING: windows were rescaled by factors differing {spread:.1f}x. The same "
                        "ground reconstructed at different scales cannot be merged into one surface."
                    )
        for refused in anchor.get("windows_refused", []) or []:
            lines.append(f"  window {refused.get('window')} NOT anchored: {refused.get('reason')}")
    lines.append("")

    consensus = report.get("scale_consensus", NOT_COMPUTED)
    lines.append("Flight-wide depth scale (consensus across windows):")
    if consensus == NOT_COMPUTED or not consensus:
        lines.append(f"  {NOT_COMPUTED}")
    elif not consensus.get("applied"):
        lines.append(f"  not applied -- {consensus.get('reason', 'unknown')}")
    else:
        lines.append(
            f"  consensus ratio:   {_fmt(consensus.get('consensus_ratio'))} "
            f"(from {consensus.get('source')} over {consensus.get('windows_measured')} windows)"
        )
        lines.append(f"  input spread:      {_fmt(consensus.get('input_spread'))}x before harmonising")
        lines.append(f"  windows rescaled:  {consensus.get('windows_rescaled')}")
        for c in consensus.get("corrections", []) or []:
            lines.append(
                f"    window {c.get('window'):>3}: {c.get('was')} -> {c.get('now')} "
                f"(x{c.get('correction')}, {c.get('reason')})"
            )
    lines.append("")

    placement = report.get("placement", NOT_COMPUTED)
    lines.append("Frame placement (measured before fusion, see placement/placement_check.png):")
    if placement == NOT_COMPUTED or not placement:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        verdict = placement.get("verdict") or ("PASS" if placement.get("passed") else "FAIL")
        lines.append(f"  verdict:                     {verdict}")
        lines.append(
            f"  ground spread across windows: {_fmt(placement.get('ground_spread_m'))} m "
            f"(threshold {_fmt(placement.get('spread_threshold_m'))} m)"
        )
        lines.append(f"  flat-ground thickness:        {_fmt(placement.get('flat_thickness_median_m'))} m")
        lines.append(f"  all-cell thickness:           {_fmt(placement.get('all_thickness_median_m'))} m")
        if placement.get("reference_ground_error_median_m") is not None:
            lines.append(
                f"  vs {placement.get('reference_ground_source', 'reference')} ground:  "
                f"median {_fmt(placement.get('reference_ground_error_median_m'))} m, "
                f"worst {_fmt(placement.get('reference_ground_error_max_abs_m'))} m "
                f"({_fmt(placement.get('reference_ground_cells'))} cells; decides the verdict)"
            )
        if placement.get("gps_ground_z_m") is not None:
            lines.append(
                f"  vs GPS ground height:         median {_fmt(placement.get('gps_ground_error_median_m'))} m, "
                f"worst {_fmt(placement.get('gps_ground_error_max_abs_m'))} m"
                + (" (takeoff height: wrong wherever terrain is not level with the launch point)"
                   if placement.get("reference_ground_error_median_m") is not None else "")
            )
        by_submap = placement.get("gps_ground_error_by_submap") or placement.get("ground_z_by_submap") or {}
        if by_submap:
            # Error alongside the two scale corrections that produced it.
            # A window's ground lands at anchor_ratio * merge_scale times
            # its true depth, and those two come from different evidence
            # (ground depth vs camera baselines), so the offset alone
            # never says which to fix.
            merge_scales = placement.get("merge_scale_by_submap") or {}
            product = placement.get("anchor_x_merge_by_submap") or {}
            header = "  per window (metres from where GPS says the ground is"
            lines.append(header + " | merge scale | anchor x merge):" if merge_scales else header + "):")
            for k in sorted(by_submap, key=lambda x: int(x)):
                row = f"    window {int(k):>3}: {by_submap[k]:+8.2f}"
                if merge_scales:
                    m, p = merge_scales.get(k), product.get(k)
                    row += f" | {'-' if m is None else f'{m:8.3f}'} | {'-' if p is None else f'{p:8.3f}'}"
                lines.append(row)
            if placement.get("merge_scale_spread"):
                lines.append(
                    f"  merge rescaled submaps over a {placement['merge_scale_spread']}x range -- it "
                    "takes scale from camera baselines, the anchor takes it from ground depth"
                )
        if verdict == "FAIL":
            lines.append(
                "  -- fusion cannot fix this. Windows placed metres apart mesh into several copies"
            )
            lines.append(
                "     of the same surface; cross-reference the per-window depth ratios above."
            )
    lines.append("")

    junctions = report.get("junction_residuals", NOT_COMPUTED)
    lines.append("Submap junction alignment:")
    if junctions == NOT_COMPUTED or not junctions:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        for j in junctions:
            flag = " [DEGENERATE]" if j.get("degenerate") else ""
            lines.append(f"  junction {j.get('junction')}: rmse={_fmt(j.get('rmse_m'), ' m')}{flag}")
    lines.append("")

    check = report.get("check_point_residuals", NOT_COMPUTED)
    lines.append("Independent check points:")
    if check == NOT_COMPUTED or check is None:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        lines.append(f"  n = {check.get('n_check_points', 0)}")
        lines.append(f"  Horizontal RMSE: {_fmt(check.get('horizontal_rmse_m'), ' m')}")
        lines.append(f"  Vertical RMSE:   {_fmt(check.get('vertical_rmse_m'), ' m')}")
    lines.append("")

    timings = report.get("stage_timings_s", NOT_COMPUTED)
    lines.append("Stage timings:")
    if timings == NOT_COMPUTED or not timings:
        lines.append(f"  {NOT_COMPUTED}")
    else:
        for stage, seconds in timings.items():
            lines.append(f"  {stage}: {seconds:.2f} s")
    ref = report.get("reference_alignment")
    lines.append("")
    lines.append("Reference alignment (orthophoto + elevation model):")
    if not ref:
        lines.append(f"  {NOT_COMPUTED} -- no reference configured; absolute accuracy is GPS-bounded")
    elif not ref.get("applied"):
        lines.append(f"  REFUSED: {ref.get('failure')}")
    else:
        h = ref.get("horizontal", {})
        lines.append(
            f"  shift applied: E {ref['shift_east_m']:+.2f} m, N {ref['shift_north_m']:+.2f} m, "
            f"U {ref['shift_up_m']:+.2f} m, rotation {ref['rotation_deg']:+.3f} deg"
        )
        lines.append(
            f"  match: {h.get('method')}, {h.get('inliers', 'n/a')} inliers, residual {h.get('residual_rms_m', 'n/a')} m"
        )
        lines.append(f"  reference: {ref.get('ortho')}")
    first_patch = report.get("first_fused_patch_s")
    if first_patch is not None:
        lines.append(f"  first fused patch visible at: {first_patch:.1f} s after start")

    return "\n".join(lines) + "\n"


def render_report_html(report: dict[str, Any]) -> str:
    """Render the report card as a self-contained HTML string (a printable artifact, not a UI)."""

    def row(label: str, value: str) -> str:
        return f"<tr><td class='label'>{label}</td><td class='value'>{value}</td></tr>"

    scalar_rows = "".join(
        [
            row("Reconstruction outcome", str(report.get("outcome", "unverified"))),
            row("Representation", str(report.get("reconstruction_representation") or "not recorded")),
            row("Output status", "Validated" if report.get("outcome") == "valid" else "DIAGNOSTIC OUTPUT"),
            row("Relative RMSE", _fmt(report.get("relative_rmse_m"), " m")),
            row("Absolute RMSE", _fmt(report.get("absolute_rmse_m"), " m")),
            row("Scale error", _fmt(report.get("scale_error_pct"), " %")),
            row("Mean reprojection error", _fmt(report.get("mean_reprojection_error_px"), " px")),
            row("Coverage", _fmt(report.get("coverage_pct"), " %")),
            row("Keyframes used", _fmt(report.get("keyframe_count"))),
            row("Confidence source", _fmt(report.get("confidence_source"))),
            row("Merge strategy", _fmt(report.get("merge_strategy"))),
        ]
    )

    profile = report.get("flight_profile", NOT_COMPUTED)
    if profile == NOT_COMPUTED or not profile:
        profile_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        profile_html = (
            "<table>"
            + row("Gimbal mode", _fmt(profile.get("gimbal_mode")))
            + row("Trajectory shape", _fmt(profile.get("trajectory_shape")))
            + row("Collinearity", _fmt(profile.get("collinearity")))
            + row("Strategy reason", _fmt(report.get("merge_strategy_reason")))
            + "</table>"
        )
    strategy_warnings = report.get("merge_strategy_warnings", NOT_COMPUTED)
    if strategy_warnings != NOT_COMPUTED and strategy_warnings:
        profile_html += "<ul>" + "".join(f"<li>WARNING: {w}</li>" for w in strategy_warnings) + "</ul>"

    georef_notes = report.get("georef_notes", NOT_COMPUTED)
    if georef_notes == NOT_COMPUTED or not georef_notes:
        georef_notes_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        georef_notes_html = "<ul>" + "".join(f"<li>{note}</li>" for note in georef_notes) + "</ul>"

    _offset_source = report.get("telemetry_offset_source")
    _offset_guessed = _offset_source == "assumed_zero" and report.get("telemetry_format") in _OFFSET_GUESS_RISK_FORMATS
    _offset_label = "Telemetry offset source" + (" ⚠" if _offset_guessed else "")
    _coverage = report.get("telemetry_video_coverage_fraction")
    _coverage_html = (
        row(
            "Telemetry/video coverage" + (" ⚠" if isinstance(_coverage, (int, float)) and _coverage < 0.999 else ""),
            f"{_coverage * 100.0:.1f} %",
        )
        if isinstance(_coverage, (int, float))
        else ""
    )
    _sync = report.get("time_sync")
    _sync_html = (
        row("Image-motion check" + (" ⚠" if _sync.get("disagrees") else ""), escape(_time_sync_line(_sync)))
        if isinstance(_sync, dict)
        else ""
    )
    sync_rows = (
        row("Telemetry offset applied", _fmt(report.get("telemetry_offset_s"), " s"))
        + row(_offset_label, _fmt(_offset_source))
        + _sync_html
        + _coverage_html
    )

    breakdown = report.get("confidence_breakdown_pct", NOT_COMPUTED)
    if breakdown == NOT_COMPUTED or breakdown is None:
        confidence_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        confidence_html = (
            "<table>"
            + row("Measured", f"{breakdown['measured_pct']:.1f} %")
            + row("Low confidence", f"{breakdown['low_confidence_pct']:.1f} %")
            + row("Inferred", f"{breakdown['inferred_pct']:.1f} %")
            + "</table>"
        )

    semantics = report.get("semantic_class_pct", NOT_COMPUTED)
    if semantics == NOT_COMPUTED or not semantics:
        semantics_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        # A colour swatch per class, matching CLASS_COLORS, so the table
        # reads against the classified point cloud the viewer shows rather
        # than being a separate colour language the operator has to learn.
        from drishti3d.semantics.classes import CLASS_COLORS, CLASS_NAMES

        swatch_by_name = {CLASS_NAMES[value]: rgb for value, rgb in CLASS_COLORS.items()}
        class_rows = ""
        for name, pct in sorted(semantics.items(), key=lambda kv: -kv[1]):
            r, g, b = swatch_by_name.get(name, (128, 128, 128))
            swatch = (
                f"<span style='display:inline-block;width:0.8em;height:0.8em;"
                f"background:rgb({r},{g},{b});margin-right:0.5em;vertical-align:middle'></span>"
            )
            class_rows += row(f"{swatch}{name}", f"{pct:.1f} %")

        labelled = report.get("semantic_labelled_pct")
        extra = ""
        if labelled not in (None, NOT_COMPUTED):
            extra += row("Points classified", f"{labelled:.1f} %")
        for label, key, suffix in (
            ("Never seen by a segmented view", "semantic_unseen_points", " points"),
            ("Disputed between views", "semantic_disputed_points", " points"),
            ("Mean segmented views per point", "semantic_mean_views_per_point", ""),
            ("Dynamic/sky points removed", "dynamic_points_removed", ""),
            ("Segmentation model", "semantic_model", ""),
        ):
            value = report.get(key)
            if value not in (None, NOT_COMPUTED):
                extra += row(label, f"{value}{suffix}")
        semantics_html = f"<table>{class_rows}{extra}</table>"

    junctions = report.get("junction_residuals", NOT_COMPUTED)
    if junctions == NOT_COMPUTED or not junctions:
        junctions_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        junction_rows = "".join(
            row(
                f"Junction {j.get('junction')}" + (" (DEGENERATE)" if j.get("degenerate") else ""),
                _fmt(j.get('rmse_m'), " m"),
            )
            for j in junctions
        )
        junctions_html = f"<table>{junction_rows}</table>"

    prior = report.get("pose_prior", NOT_COMPUTED)
    if prior == NOT_COMPUTED or not prior:
        prior_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        prior_rows = row("Yaw from image flow", f"{_fmt(prior.get('yaw_from_flow_keyframes'))} keyframes")
        prior_rows += row("Sparse BA", f"{_fmt(prior.get('ba_points'))} points, RMSE {_fmt(prior.get('rmse_after_px'), ' px')}")
        prior_rows += row(
            "Poses moved (median)",
            f"{_fmt(prior.get('median_position_shift_m'), ' m')} / {_fmt(prior.get('median_rotation_change_deg'), ' deg')}",
        )
        prior_html = f"<table>{prior_rows}</table>"

    anchor = report.get("depth_anchor", NOT_COMPUTED)
    if anchor == NOT_COMPUTED or not anchor:
        anchor_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    elif not anchor.get("enabled"):
        anchor_html = "<p>disabled -- backbone depth taken at its own scale</p>"
    else:
        anchor_rows = row("Windows anchored", f"{anchor.get('windows_anchored')}/{anchor.get('windows_total')}")
        anchor_rows += row(
            "Depth ratio (parallax / backbone)",
            f"median {_fmt(anchor.get('ratio_median'))}, min {_fmt(anchor.get('ratio_min'))}, "
            f"max {_fmt(anchor.get('ratio_max'))}",
        )
        for refused in anchor.get("windows_refused", []) or []:
            anchor_rows += row(f"Window {refused.get('window')} NOT anchored", str(refused.get("reason")))
        anchor_html = f"<table>{anchor_rows}</table>"

    check = report.get("check_point_residuals", NOT_COMPUTED)
    if check == NOT_COMPUTED or check is None:
        check_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        check_html = (
            "<table>"
            + row("n check points", str(check.get("n_check_points", 0)))
            + row("Horizontal RMSE", _fmt(check.get("horizontal_rmse_m"), " m"))
            + row("Vertical RMSE", _fmt(check.get("vertical_rmse_m"), " m"))
            + "</table>"
        )

    timings = report.get("stage_timings_s", NOT_COMPUTED)
    if timings == NOT_COMPUTED or not timings:
        timings_html = f"<p class='not-computed'>{NOT_COMPUTED}</p>"
    else:
        timings_html = "<table>" + "".join(row(stage, f"{seconds:.2f} s") for stage, seconds in timings.items()) + "</table>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DRISHTI-3D Accuracy Report Card</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2em; color: #1a1a1a; }}
  h1 {{ font-size: 1.4em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  h2 {{ font-size: 1.1em; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 480px; }}
  td {{ padding: 4px 10px; border-bottom: 1px solid #ddd; }}
  td.label {{ color: #555; }}
  td.value {{ font-weight: 600; text-align: right; }}
  .not-computed {{ color: #999; font-style: italic; }}
</style>
</head>
<body>
<h1>DRISHTI-3D Accuracy Report Card</h1>
<table>{scalar_rows}</table>
<h2>Flight profile / merge strategy</h2>
{profile_html}
<h2>Georeferencing notes</h2>
{georef_notes_html}
<h2>Telemetry/video time sync</h2>
<table>{sync_rows}</table>
<h2>Confidence breakdown</h2>
{confidence_html}
<h2>Scene composition</h2>
{semantics_html}
<h2>Telemetry alignment (pose prior)</h2>
{prior_html}
<h2>Depth anchoring (backbone depth vs GPS parallax)</h2>
{anchor_html}
<h2>Submap junction alignment</h2>
{junctions_html}
<h2>Independent check points</h2>
{check_html}
<h2>Stage timings</h2>
{timings_html}
</body>
</html>
"""
