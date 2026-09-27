"""The pipeline runner: wires ingest -> triage -> geometry -> bundle -> fusion -> export.

This is the one piece that turns the previously-disconnected stage modules
(``drishti3d.ingest``, ``drishti3d.triage``, ``drishti3d.geometry``, and --
once they land -- ``drishti3d.geometry.bundle``, ``drishti3d.fusion``,
``drishti3d.export``) into an actual end-to-end pipeline. See
``drishti3d.pipeline.stages`` for what each stage does; this module is
purely the glue: timing, logging, cancellation, progress/partial-result
plumbing, and deciding which stage failures are fatal.

Fail-fast vs. record-and-continue
----------------------------------
Ingest, triage, camera refinement and geometry are load-bearing: failure stops the run with
every remaining stage recorded as ``"skipped"``. Geometry never substitutes
synthetic output for a missing reconstruction model. Unavailable camera
refinement may skip, but a rejected solution stops dense geometry/export.
Fusion/export are
optional stages owned by other, concurrently-in-progress workstreams that
may not exist yet (``StageUnavailable`` -> recorded as ``"skipped"``) --
either way, a failure in any of the four never aborts the run; it's
recorded on that stage's ``StageResult`` and the pipeline moves on with
whatever it already has.

Cancellation never raises out of ``run_pipeline``
----------------------------------------------------
A cancelled run is not a failed run: ``run_pipeline`` always returns a
normal ``PipelineResult`` (see ``PipelineResult.report["cancelled"]``),
just a partial one, with the stage that was interrupted (and everything
after it) recorded as ``"skipped"``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from drishti3d.config import Config, load_config
from drishti3d.logging_setup import setup_logging
from drishti3d.pipeline.result import PipelineResult, StageResult
from drishti3d.pipeline.stages import (
    BundleAdjustmentStage,
    CancelToken,
    ExportStage,
    CoverageStage,
    DronePathStage,
    FramePlacementStage,
    FusionStage,
    GeometryStage,
    IngestStage,
    PipelineCancelled,
    PipelineStage,
    PipelineState,
    SemanticsStage,
    PosePriorStage,
    StageUnavailable,
    TimeSyncStage,
    TriageStage,
)
from drishti3d.types import PointCloud

logger = logging.getLogger(__name__)

# ``progress_cb(stage_name, current, total, message)`` -- one level up from
# each stage's own ``(current, total, message)`` callback (see
# ``stages.StageProgressCb``); the runner is what attaches the stage name.
ProgressCb = Callable[[str, int, int, str], None]
PartialCb = Callable[[PointCloud], None]

# ``preview_cb(stage_name, snapshot)`` -- see ``_preview_snapshot``.
PreviewCb = Callable[[str, dict], None]


def _preview_snapshot(state, stage_name: str, artifacts: dict | None) -> dict:
    """Everything a live viewer can draw, as of the end of ``stage_name``.

    Why this exists
    ---------------
    ``partial_cb`` only ever carries a ``PointCloud``, and on the default
    configuration (``GeometryConfig.single_inference``) the whole flight
    is one window, so it fires exactly once -- at the end of geometry,
    roughly fourteen minutes into a run. Everything before that point was
    invisible to the GUI even though the pipeline already knew it: the
    drone's track comes out of triage in seconds, refined poses and a
    sparse triangulated cloud come out of the pose prior minutes before
    the backbone starts.

    So the runner hands the caller a snapshot of the drawable state after
    every stage. It is references, not copies -- cheap to emit even when
    ``point_cloud`` holds four million points -- and the receiver decides
    what actually changed.

    Anything absent is simply not in the dict, so a consumer can use
    ``snapshot.get(...) is not None`` as "this became available".
    """
    snapshot: dict = {"stage": stage_name}

    if artifacts:
        snapshot["artifacts"] = artifacts

    # Camera track. Prefer refined poses; fall back to the telemetry
    # poses triage attaches to each keyframe, which exist far earlier.
    poses = list(state.poses) if state.poses else []
    if not poses and state.keyframes:
        poses = [kf.pose for kf in state.keyframes if kf.pose is not None]
    if poses:
        snapshot["poses"] = poses
        snapshot["pose_source"] = "refined" if state.poses else "telemetry"

    if state.intrinsics is not None:
        snapshot["intrinsics"] = state.intrinsics
    if state.keyframes:
        snapshot["keyframe_count"] = len(state.keyframes)
    if state.point_cloud is not None:
        snapshot["point_cloud"] = state.point_cloud
    if state.mesh_faces is not None:
        snapshot["mesh_faces"] = state.mesh_faces

    # The sparse bundle-adjusted cloud: a real, if thin, 3D reconstruction
    # that exists once the pose prior has run -- minutes before geometry
    # produces anything.
    sparse = getattr(state, "ba_points", None)
    if sparse is not None and len(sparse):
        snapshot["sparse_points"] = sparse

    datum = height_datum_m(state)
    if datum is not None:
        snapshot["height_datum_m"] = datum

    return snapshot


def height_datum_m(state) -> float | None:
    """Altitude of the local frame's origin: add it to z for elevations from the flight log's datum.

    The ENU origin is the first geo-tagged keyframe (``ingest.telemetry.
    telemetry_to_enu``) until georeferencing records the exact one.
    """
    origin = getattr(state, "georef_origin", None)
    if origin is not None:
        return float(origin.alt_msl)
    for kf in getattr(state, "keyframes", None) or []:
        geo = getattr(getattr(kf, "telemetry", None), "geo", None)
        if geo is not None and np.isfinite(geo.alt_msl):
            return float(geo.alt_msl)
    return None

# Without real input and geometry there is no reconstruction to export.
_FATAL_STAGE_NAMES = frozenset({"ingest", "triage", "geometry", "pose_prior", "bundle_adjustment"})


def _build_stages() -> list[PipelineStage]:
    """Fresh stage instances in execution order (stages are stateless; state lives in PipelineState)."""
    return [
        IngestStage(),
        TriageStage(),
        # Clock check against the keyframes' own image rotation, before any
        # stage uses a telemetry timestamp (TimeSyncStage). Never fatal.
        TimeSyncStage(),
        # Ordered previews in front of the expensive stages: path, then
        # coverage, then placement. Each answers one question from data
        # that already exists and renders it, so a flight that cannot
        # reconstruct is visible in seconds rather than after the
        # backbone has run. All three are diagnostic and never fatal.
        DronePathStage(),
        CoverageStage(),
        # Before geometry, deliberately: SemanticsStage's dynamic mask has
        # to exist while the depth backbone runs, so vehicle/person pixels
        # never become 3D points at all (see SemanticsStage's docstring).
        # Optional -- it raises StageUnavailable and is recorded "skipped"
        # when disabled or when transformers/torch are absent -- so it is
        # correctly absent from _FATAL_STAGE_NAMES above.
        SemanticsStage(),
        PosePriorStage(),
        # Cheap gate in front of the expensive stage: place every frame
        # on the ground from poses + intrinsics alone and render it, so a
        # coverage or pose problem is visible in seconds rather than
        # after the dense backbone has baked it into every window. Purely
        # diagnostic -- see FramePlacementStage.
        FramePlacementStage(),
        GeometryStage(),
        BundleAdjustmentStage(),
        FusionStage(),
        ExportStage(),
    ]


def _seed_everything(seed: int) -> None:
    """Seed every RNG this process can reach -- see ``run_pipeline``'s ``seed`` parameter docstring.

    Best-effort and side-effect-only: sets Python's ``random``, NumPy's
    global legacy RNG, and (when importable) torch's CPU/CUDA/MPS RNGs.
    Never raises -- torch not being installed (the plain-CPU/no-ml-extra
    dev setup) is normal, not an error, for this call.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        # torch seeds the MPS generator from the same manual_seed() call
        # above (there is no separate torch.mps.manual_seed needed as of
        # the torch version this project targets) -- nothing further to
        # do here beyond the note in run_pipeline's docstring that this
        # does not, by itself, guarantee bit-identical MPS kernel output.
        pass


def run_pipeline(
    video_path: str | Path,
    telemetry_path: str | Path | None = None,
    config: Config | None = None,
    progress_cb: ProgressCb | None = None,
    cancel_token: CancelToken | None = None,
    backbone: str | None = "null",
    partial_cb: PartialCb | None = None,
    preview_cb: PreviewCb | None = None,
    dry_run: bool = False,
    telemetry_offset_s: float | None = None,
    seed: int = 0,
    geometry_cache_dir: str | Path | None = None,
) -> PipelineResult:
    """Run the full DRISHTI-3D pipeline over a single video.

    Parameters
    ----------
    seed:
        Seeds every RNG this run can reach up front: Python's ``random``,
        NumPy's global legacy RNG (some third-party code -- OpenCV,
        scipy's ``fclusterdata`` -- still uses the global RNG rather than
        an injectable ``Generator``), and ``torch``'s CPU/CUDA/MPS RNGs
        when ``torch`` is importable. Every RNG *this project's own code*
        constructs already takes an explicit ``seed`` argument
        (``geometry.submap``'s RANSAC, ``geometry.georef.align_to_gps``,
        ...) rather than reading the global state, so those are already
        deterministic call-to-call regardless of this -- this seed exists
        for everything else (third-party libraries, and the MapAnything
        backbone's own internal randomness, if any) that isn't under this
        project's control.

        Honesty note: this does **not** guarantee bit-identical output
        across runs when ``backbone="mapanything"`` on an MPS device.
        Real, measured MPS nondeterminism exists independent of seeding
        (Apple's Metal Performance Shaders backend does not guarantee a
        fixed floating-point reduction order for attention/matmul kernels
        the way ``torch.use_deterministic_algorithms`` can request on
        CUDA/CPU -- there is no equivalent deterministic-algorithms
        guarantee documented for MPS as of this torch version). Seeding
        removes every *avoidable* source of run-to-run variance (this
        project's own RNG usage, Python/NumPy/torch's own PRNGs); it
        cannot remove a hardware/kernel-level one. Run on CPU (slower, but
        ``torch.use_deterministic_algorithms(True)`` is meaningful there)
        if bit-identical reproducibility matters more than speed for a
        given run.
    video_path:
        Path to the drone video.
    telemetry_path:
        Optional flight-telemetry sidecar (DJI SRT / CSV / GPX --
        auto-detected, see ``ingest.telemetry.load_telemetry``). When
        omitted, the pipeline still completes end to end using
        vision-only keyframe spacing (see ``triage.selector``); the
        returned report's ``triage_spacing_mode`` says which mode ran.
    telemetry_offset_s:
        Explicit video-start offset (seconds) for ``telemetry_path``,
        overriding ``load_telemetry``'s own auto-detection -- see that
        function's docstring for the full "explicit > auto-detected >
        assumed zero" precedence. Needed for an Airdata/flight-log CSV
        whose ``isVideo``/``isPhoto`` column either doesn't exist or can't
        be matched unambiguously to this video's duration (e.g. several
        same-length recording segments in one flight). Ignored (with no
        effect) when ``telemetry_path`` is ``None``.
    config:
        A ``Config``; defaults to ``Config()`` (every stage's defaults).
    progress_cb:
        ``progress_cb(stage_name, current, total, message)``, called
        throughout every stage.
    cancel_token:
        A cooperative cancellation flag (``stages.CancelToken``, or
        anything duck-type compatible with it, e.g.
        ``drishti3d.app.workers.CancelToken``). Checked between stages and
        within long inner loops (geometry's per-window loop). A cancelled
        run returns normally with a partial result -- see module docstring.
    backbone:
        Explicit geometry backbone override (e.g. ``"null"``,
        ``"mapanything"``). Defaults to ``"null"`` -- the always-available,
        zero-GPU synthetic backbone -- so calling this with no arguments
        beyond a video always works. Pass ``None`` to defer to
        ``config.geometry.backbone`` instead (what the CLI's ``--backbone``
        does when omitted); ``GeometryStage`` falls back to ``"null"``
        itself if that configured backbone turns out to be unavailable.
    partial_cb:
        Optional ``partial_cb(PointCloud)``, called with an
        incrementally-growing point cloud as ``GeometryStage`` finishes
        each reconstruction window -- lets a GUI render the model before
        the run finishes.
    preview_cb:
        Optional ``preview_cb(stage_name, snapshot)``, called once after
        EVERY stage with a reference-only snapshot of the drawable state
        so far -- camera track, intrinsics, sparse points, point cloud,
        mesh faces, and that stage's artifacts. See ``_preview_snapshot``
        for why ``partial_cb`` alone is not enough to drive a live view.
        Never raises out of the runner: a viewer that throws must not
        take down the run it is watching.
    dry_run:
        Run only ingest + triage (fast iteration on keyframe
        selection/config without paying for geometry).
    """
    _seed_everything(seed)

    config = config or Config()
    cancel_token = cancel_token or CancelToken()
    effective_backbone = backbone if backbone is not None else config.geometry.backbone

    state = PipelineState(
        video_path=Path(video_path),
        telemetry_path=Path(telemetry_path) if telemetry_path else None,
        config=config,
        backbone_name=effective_backbone,
        telemetry_offset_s=telemetry_offset_s,
    )
    # Populated below as each stage finishes, *before* the next one starts
    # -- so by the time ExportStage runs, this already has every prior
    # stage's real elapsed time (export's own isn't known until it
    # returns). Not a declared ``PipelineState`` field, same pattern as
    # ``MatchingStage``'s ``state.matching_problem`` -- see that class's
    # docstring. ``export.report.build_report`` is what actually surfaces
    # this in the accuracy report card.
    state.stage_timings_s = {}
    # Wall-clock origin for "time to first fused patch" (GeometryStage).
    state.run_t0 = time.monotonic()

    stages = _build_stages()
    # ingest + triage only, per the dry-run contract -- but every stage
    # still gets a terminal StageResult (below) so callers always see a
    # consistent, complete stage table regardless of dry_run.
    dry_run_names = (
        frozenset({"semantics", "geometry", "bundle_adjustment", "fusion", "export"}) if dry_run else frozenset()
    )

    stage_results: list[StageResult] = []
    state.stage_results = stage_results
    cancelled = False

    def _emit_partial() -> None:
        if partial_cb is not None and state.point_cloud is not None:
            try:
                partial_cb(state.point_cloud)
            except Exception:
                logger.exception("partial_cb raised; ignoring")

    def _emit_preview(stage_name: str, artifacts: dict | None = None) -> None:
        if preview_cb is None:
            return
        try:
            preview_cb(stage_name, _preview_snapshot(state, stage_name, artifacts))
        except Exception:
            logger.exception("preview_cb raised; ignoring")

    for stage in stages:
        if stage.name in dry_run_names:
            stage_results.append(
                StageResult(name=stage.name, status="skipped", elapsed_s=0.0, message="dry run: stage not executed")
            )
            continue

        if cancel_token.is_set():
            cancelled = True
            logger.info("run_pipeline: cancelled before stage %s started", stage.name)
            break

        stage_name = stage.name
        logger.info("=== stage: %s ===", stage_name)

        def _stage_progress(current: int, total: int, message: str, _name: str = stage_name) -> None:
            if progress_cb is not None:
                progress_cb(_name, current, total, message)

        def _stage_partial(pc: PointCloud) -> None:
            state.point_cloud = pc
            _emit_partial()

        t0 = time.monotonic()
        try:
            # Geometry cache (pipeline.geometry_cache): reuse the backbone's
            # output when this run's keyframes and geometry settings match a
            # previous one, so fusion/BA/export changes do not cost an hour
            # of inference to test. Loading is keyed and refused on any
            # mismatch; saving never fails the run.
            cached = None
            if stage_name == "geometry":
                from drishti3d.pipeline.stages import _validate_telemetry_clock

                _validate_telemetry_clock(state)
            if stage_name == "geometry" and geometry_cache_dir:
                from drishti3d.pipeline.geometry_cache import load_geometry_cache

                cached = load_geometry_cache(geometry_cache_dir, state)
            if cached is not None:
                artifacts, message = cached
                artifacts = {**artifacts, "loaded_from_cache": str(geometry_cache_dir)}
            else:
                artifacts, message = stage.run(state, cancel_token, _stage_progress, _stage_partial)
                if stage_name == "geometry" and geometry_cache_dir:
                    try:
                        from drishti3d.pipeline.geometry_cache import save_geometry_cache

                        save_geometry_cache(geometry_cache_dir, state, artifacts, message)
                    except Exception:
                        logger.warning("geometry cache: save failed; continuing", exc_info=True)
            elapsed = time.monotonic() - t0
            state.stage_timings_s[stage_name] = elapsed
            stage_results.append(StageResult(name=stage_name, status="ok", elapsed_s=elapsed, message=message, artifacts=artifacts))
            logger.info("stage %s ok (%.2fs): %s", stage_name, elapsed, message)
            _emit_partial()
            _emit_preview(stage_name, artifacts)
        except PipelineCancelled:
            elapsed = time.monotonic() - t0
            state.stage_timings_s[stage_name] = elapsed
            cancelled = True
            stage_results.append(
                StageResult(name=stage_name, status="skipped", elapsed_s=elapsed, message="cancelled by user")
            )
            logger.warning("stage %s cancelled after %.2fs", stage_name, elapsed)
            _emit_partial()
            _emit_preview(stage_name)
            break
        except StageUnavailable as exc:
            elapsed = time.monotonic() - t0
            state.stage_timings_s[stage_name] = elapsed
            stage_results.append(StageResult(name=stage_name, status="skipped", elapsed_s=elapsed, message=str(exc)))
            logger.info("stage %s skipped (%.2fs): %s", stage_name, elapsed, exc)
            _emit_preview(stage_name)
        except Exception:  # noqa: BLE001 - by design: the runner decides fail-fast vs. continue, never crashes
            elapsed = time.monotonic() - t0
            state.stage_timings_s[stage_name] = elapsed
            tb = traceback.format_exc()
            stage_results.append(StageResult(name=stage_name, status="failed", elapsed_s=elapsed, message=tb))
            logger.error("stage %s failed after %.2fs:\n%s", stage_name, elapsed, tb)
            _emit_preview(stage_name)
            if stage_name in _FATAL_STAGE_NAMES:
                logger.error("stage %s is fatal; stopping the pipeline", stage_name)
                break
            # Optional/late stage: keep going with whatever state already has.

    # Every stage the run never reached (fatal break, or cancellation before
    # it started) still gets a terminal StageResult, so callers never see a
    # "pending"/"running" status on a finished (even if partial) result.
    ran_names = {sr.name for sr in stage_results}
    for stage in stages:
        if stage.name not in ran_names:
            stage_results.append(StageResult(name=stage.name, status="skipped", elapsed_s=0.0, message="not reached"))

    if state.video is not None:
        state.video.close()

    report: dict = dict(state.report)
    report["cancelled"] = cancelled
    report["backbone"] = (
        "mvs3d" if getattr(state, "geometry_premeshed", False)
        else "heightfield_mvs" if getattr(state, "heightfield_diagnostics", None)
        else state.backbone_name
    )
    report["reconstruction_representation"] = (
        "2.5D height-field surface" if getattr(state, "geometry_heightmap", False)
        else "3D volumetric mesh" if state.mesh_faces is not None and len(state.mesh_faces)
        else "3D point cloud (no mesh)"
    )
    report["keyframes_selected"] = len(state.keyframes)
    origin = getattr(state, "georef_origin", None)
    if origin is not None:
        # The local ENU frame's origin, so a viewer can label heights in
        # metres above sea level instead of relative to the first GPS fix.
        report["georef_origin"] = {
            "lat": float(origin.lat), "lon": float(origin.lon), "alt_msl": float(origin.alt_msl),
        }
    if getattr(state, "first_patch_s", None) is not None:
        report["first_fused_patch_s"] = round(state.first_patch_s, 1)
    report["total_runtime_s"] = round(time.monotonic() - state.run_t0, 1)
    for key, value in state.triage_report.items():
        report[f"triage_{key}"] = value
    # Surfaced at the top level (not only inside ingest's own StageResult)
    # so a caller/CLI sees the video-start offset actually used -- and,
    # per ingest.telemetry.load_telemetry's docstring, whether it was
    # explicit/auto-detected/just assumed -- without digging into
    # stage_results, even on a dry run where ExportStage (which also
    # surfaces this, in the accuracy report card) never runs.
    if "time_offset_s" in state.telemetry_stats:
        report["telemetry_offset_s"] = state.telemetry_stats["time_offset_s"]
        report["telemetry_offset_source"] = state.telemetry_stats.get("offset_source")
    if "telemetry_video_coverage_fraction" in state.telemetry_stats:
        report["telemetry_video_coverage_fraction"] = state.telemetry_stats["telemetry_video_coverage_fraction"]

    result = PipelineResult(
        keyframes=state.keyframes,
        submaps=state.submaps,
        point_cloud=state.point_cloud,
        # Carried so the GUI can render the fused SURFACE, not just its
        # vertices, and so a saved run reloads as a mesh.
        mesh_faces=state.mesh_faces,
        poses=state.poses,
        stage_results=stage_results,
        report=report,
        config=config,
        video_path=str(state.video_path),
        telemetry_path=str(state.telemetry_path) if state.telemetry_path else None,
    )
    result.report.update(result.quality)
    return result


# ---------------------------------------------------------------------------
# CLI: `drishti3d-run` (see [project.scripts] in pyproject.toml)
# ---------------------------------------------------------------------------


def _print_stage_table(stage_results: list[StageResult]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="DRISHTI-3D Pipeline Stages")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Elapsed (s)", justify="right")
    table.add_column("Message")

    _STATUS_STYLE = {"ok": "green", "skipped": "yellow", "failed": "red"}
    for sr in stage_results:
        first_line = sr.message.splitlines()[0] if sr.message else ""
        style = _STATUS_STYLE.get(sr.status, "")
        table.add_row(sr.name, f"[{style}]{sr.status}[/{style}]" if style else sr.status, f"{sr.elapsed_s:.2f}", first_line)

    Console().print(table)


def _cli_progress(stage: str, current: int, total: int, message: str) -> None:
    pct = f"{100 * current / total:5.1f}%" if total else "  n/a"
    print(f"[{stage:>17}] {pct}  {message}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="drishti3d-run",
        description="Run the DRISHTI-3D reconstruction pipeline headlessly (no Qt/GUI required).",
    )
    parser.add_argument("video", type=str, help="Path to the drone video.")
    parser.add_argument("--telemetry", type=str, default=None, help="Path to a flight telemetry sidecar (SRT/CSV/GPX).")
    parser.add_argument(
        "--telemetry-offset",
        type=float,
        default=None,
        help=(
            "Explicit video-start offset (seconds) for --telemetry, overriding auto-detection "
            "(e.g. from an Airdata CSV's isVideo column). See ingest.telemetry.load_telemetry."
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a pipeline config YAML.")
    parser.add_argument(
        "--dense-method", choices=("full3d", "heightfield", "auto", "mapanything", "mvs3d"),
        help="auto/mvs3d: measured volumetric stereo; full3d: learned depth; heightfield: explicit terrain 2.5D.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Geometry backbone override (e.g. 'null', 'mapanything'). Defaults to the config's geometry.backbone.",
    )
    parser.add_argument("--out", type=str, default=None, help="Directory to save the PipelineResult to (PipelineResult.save).")
    parser.add_argument(
        "--geometry-cache",
        type=str,
        default=None,
        help=(
            "DEVELOPER TOOL, off by default. Directory to save GeometryStage's output to, and reload from "
            "on a later run over the SAME footage with the SAME geometry settings, skipping backbone "
            "inference. Useless for processing different flights -- it is for iterating on the stages "
            "after geometry without paying for geometry each time."
        ),
    )
    parser.add_argument("--max-keyframes", type=int, default=None, help="Cap the number of keyframes triage selects.")
    camera = parser.add_argument_group(
        "known camera calibration",
        "Optional. Replaces the metadata/camera-database/HFOV guess (provenance 'user'), which "
        "focal-from-flow and bundle adjustment then leave alone.",
    )
    camera.add_argument("--fx", type=float, default=None, help="Focal length in pixels.")
    camera.add_argument("--fy", type=float, default=None, help="Vertical focal length in pixels (default: --fx).")
    camera.add_argument("--cx", type=float, default=None, help="Principal point x in pixels (default: image centre).")
    camera.add_argument("--cy", type=float, default=None, help="Principal point y in pixels (default: image centre).")
    camera.add_argument("--hfov", type=float, default=None, help="Horizontal field of view in degrees (used when --fx is not given).")
    camera.add_argument(
        "--dist",
        type=str,
        default=None,
        help="Distortion of the raw video, OpenCV order: 'k1,k2,p1,p2[,k3]'.",
    )
    camera.add_argument(
        "--calibration-width",
        type=int,
        default=None,
        help="Image width the calibration was made at, when it differs from the video's.",
    )
    camera.add_argument("--camera-model", type=str, default=None, help="Camera database key, e.g. 'dji mavic 3'.")
    parser.add_argument("--dry-run", action="store_true", help="Run only ingest + triage; skip geometry/fusion/export.")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help=(
            "Write a top-down progress PNG every --snapshot-every window completions. "
            "Defaults to <out>/progress when --out is given; pass 'none' to disable. "
            "Purely diagnostic: it makes a long headless run visibly alive."
        ),
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=1,
        help="Write a progress snapshot every N partial updates (default: every one).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...).")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for every RNG this run touches (Python/NumPy/torch) -- see run_pipeline's "
            "seed parameter docstring for exactly what this does and does not guarantee "
            "(notably: does not remove genuine MPS kernel-level nondeterminism)."
        ),
    )
    return parser.parse_args(argv)


def _apply_camera_overrides(config: Config, args: argparse.Namespace) -> None:
    """Layer the CLI's known-calibration flags over ``config.ingest``."""
    ingest = config.ingest
    for flag, field_name in (
        ("fx", "camera_fx"),
        ("fy", "camera_fy"),
        ("cx", "camera_cx"),
        ("cy", "camera_cy"),
        ("hfov", "camera_hfov_deg"),
        ("calibration_width", "camera_calibration_width"),
        ("camera_model", "camera_model"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(ingest, field_name, value)
    if getattr(args, "dist", None) is not None:
        try:
            ingest.camera_dist_coeffs = [float(v) for v in args.dist.replace(" ", "").split(",") if v]
        except ValueError as exc:
            raise SystemExit(f"--dist: expected comma-separated numbers, got {args.dist!r}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    config = load_config(args.config) if args.config else Config()
    if args.max_keyframes is not None:
        config.triage.target_keyframes = args.max_keyframes
    _apply_camera_overrides(config, args)
    if args.dense_method is not None:
        config.geometry.dense_method = args.dense_method

    if args.out:
        # ``ExportStage`` (see pipeline.stages) resolves
        # ``config.export.output_dir`` relative to the process's current
        # working directory, not relative to ``--out`` -- so without this,
        # every deliverable (PLY/LAS/GLB/rasters/report) silently lands in
        # ``./output`` under wherever the CLI happened to be invoked from,
        # while only the cached ``PipelineResult`` (``result.save`` below)
        # actually respects ``--out``. Anchoring ``export.output_dir``
        # under ``--out`` here, *before* ``run_pipeline`` runs ExportStage,
        # makes ``--out`` the single root every deliverable from this run
        # -- cached result and exported files alike -- actually lands
        # under.
        config.export.output_dir = str(Path(args.out) / "output")

    # Live progress snapshots. Defaults ON under --out, because the failure
    # this guards against -- a long headless run that is silently producing
    # nothing -- is only discoverable at the end, when the session time is
    # already spent. Opt out with --snapshot-dir none.
    snapshot_writer = None
    snapshot_dir = args.snapshot_dir
    if snapshot_dir is None and args.out:
        snapshot_dir = str(Path(args.out) / "progress")
    if snapshot_dir and snapshot_dir.lower() != "none":
        try:
            from drishti3d.export.preview import ProgressSnapshotWriter

            snapshot_writer = ProgressSnapshotWriter(snapshot_dir, every_n=max(1, args.snapshot_every))
            print(f"progress snapshots -> {snapshot_dir}")
        except Exception:
            logger.warning("could not start progress snapshots; continuing without them", exc_info=True)

    # OFF unless explicitly asked for. The cache only ever hits when the
    # SAME footage is re-run with the SAME geometry settings -- a developer
    # iterating on fusion, not an operator processing flights. Defaulting
    # it on wrote a multi-gigabyte pickle beside every run's deliverables
    # for no benefit to anyone but me.
    geometry_cache_dir = args.geometry_cache
    if geometry_cache_dir and geometry_cache_dir.lower() == "none":
        geometry_cache_dir = None

    result = run_pipeline(
        video_path=args.video,
        telemetry_path=args.telemetry,
        config=config,
        progress_cb=_cli_progress,
        partial_cb=snapshot_writer,
        backbone=args.backbone,
        dry_run=args.dry_run,
        telemetry_offset_s=args.telemetry_offset,
        seed=args.seed,
        geometry_cache_dir=geometry_cache_dir,
    )

    if snapshot_writer is not None and snapshot_writer.written:
        print(f"wrote {snapshot_writer.written} progress snapshot(s) to {snapshot_dir}")

    _print_stage_table(result.stage_results)
    print()
    print(result.summary())

    if args.out:
        result.save(args.out)
        print(f"\nSaved result to {args.out}")

    if result.outcome == "cancelled":
        return 130
    return 1 if result.outcome == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
