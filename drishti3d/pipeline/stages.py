"""The six pipeline stages: ingest -> triage -> geometry -> bundle -> fusion -> export.

Every stage shares one signature (see ``PipelineStage.run``) so
``pipeline.runner.run_pipeline`` can time, log, cancel, and record a
``StageResult`` for each of them uniformly, without special-casing any
particular stage.

Ingest / triage / geometry are the stages this workstream owns end to end
and that must always work with zero GPU (geometry falls back to
``NullBackbone`` -- see its module docstring -- whenever the configured
backbone isn't installed/available). Bundle adjustment, fusion, and export
are owned by two other, concurrently-in-progress workstreams
(``drishti3d.geometry.bundle``, ``drishti3d.fusion``, ``drishti3d.export``);
those modules do not exist yet (or exist but are still empty) at the time
this file was written. Each of those three stages therefore imports its
backing module lazily, inside ``run()``, and raises ``StageUnavailable``
-- recorded by the runner as ``StageResult(status="skipped")``, never a
hard failure -- whenever that module is missing or doesn't yet expose a
recognized entry point. Once those modules land with their real API, only
the small ``getattr(...)`` dispatch below needs to change.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import queue
import threading
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from drishti3d.config import Config
from drishti3d.device import available_devices, device_report, get_device
from drishti3d.geometry import features as features_mod
from drishti3d.geometry import tracks as tracks_mod
from drishti3d.geometry import triangulate as triangulate_mod
from drishti3d.geometry.backbone import get_backbone
from drishti3d.geometry.flight_profile import FlightProfile, analyze_flight_profile
from drishti3d.geometry.mapanything import resize_preserving_aspect, scale_intrinsics
from drishti3d.geometry.submap import merge_submaps, strategy_report
from drishti3d.geometry.windows import Window, plan_window_size, plan_windows
from drishti3d.ingest.intrinsics import intrinsics_from_video
from drishti3d.ingest.telemetry import load_telemetry
from drishti3d.ingest.video import VideoSource
from drishti3d.triage.selector import select_keyframes, triage_report
from drishti3d.types import (
    CameraIntrinsics,
    FrameMetrics,
    Keyframe,
    PointCloud,
    Pose,
    Submap,
    TelemetrySample,
)

logger = logging.getLogger(__name__)

StageProgressCb = Callable[[int, int, str], None]
PartialCb = Callable[[PointCloud], None]

# geometry.submap.merge_submaps requires >= 3 shared cameras per junction
# (Umeyama needs >= 3 non-collinear correspondences); a window_size smaller
# than this can never supply that many while still leaving room for
# non-overlapping content, so GeometryStage floors planning at this size
# regardless of what geometry.windows.plan_window_size comes back with.
_MIN_VIABLE_WINDOW_SIZE = 4
# Default overlap floor: bigger than plan_windows' own _MIN_OVERLAP (2) and
# merge_submaps' hard minimum (3) -- lands junctions in the "4-6 shared
# cameras" range Task 3 calls for once combined with the larger
# window_size default (geometry.windows._DEFAULT_TARGET_WINDOW_SIZE).
_MERGE_MIN_OVERLAP = 4

# GeometryStage's post-merge global-consistency check (see its run()):
# merged camera positions vs. GPS RMSE above this triggers a prominent
# warning. Generous on purpose -- this runs pre-BA/pre-georeference, where
# some raw backbone/alignment error is expected; it exists to catch a
# merge that is still badly broken (e.g. chained-Sim3 drift compounding
# because GPS coverage was too sparse to anchor every submap
# independently -- see geometry.submap's module docstring), not to demand
# BA-grade accuracy this early.
_GEOMETRY_GPS_RMSE_WARN_M = 25.0

# geometry.windows.plan_window_size's default budget (6 GB) is explicitly
# sized for the CUDA-card target that module's own docstring calls out
# ("6 GB of VRAM cannot hold a whole flight..."), using an estimate_memory
# formula its own module docstring flags as an uncalibrated placeholder.
# Apple Silicon (MPS) and CPU are unified-memory devices with no isolated
# VRAM ceiling anywhere near that tight -- applying the 6 GB CUDA figure to
# them would starve window planning (e.g. cap window_size well below the
# Task 3 target of ~14) for no reason grounded in those devices' actual
# headroom. Used only when device_report() can't report a real free-VRAM
# figure (i.e. not CUDA).
_UNIFIED_MEMORY_WINDOW_BUDGET_GB = 12.0


#: Fraction of free VRAM that window planning is allowed to budget.
#: `estimate_memory` models the steady state; it cannot see transient
#: peaks, allocator fragmentation, or the caching allocator's own
#: bookkeeping. Reserving a fifth is what turns a plan that "just fits"
#: into one that actually runs.
_VRAM_BUDGET_FRACTION = 0.8


def _window_memory_budget_gb() -> float:
    """Best-effort memory budget for geometry.windows.plan_window_size, device-aware.

    CUDA reports real free VRAM (``device.device_report``); MPS/CPU fall
    back to ``_UNIFIED_MEMORY_WINDOW_BUDGET_GB`` (see its docstring for
    why the module default is wrong for them). Never raises -- a device
    query failure just falls back to the unified-memory default.
    """
    try:
        report = device_report()
    except Exception:  # noqa: BLE001 - device introspection must never break planning
        return _UNIFIED_MEMORY_WINDOW_BUDGET_GB
    if report.get("device") == "cuda" and "vram_free_gb" in report:
        # Budget a FRACTION of free VRAM, never all of it.
        #
        # Planning against 100% of free memory is planning to OOM. Measured
        # on a 16 GB T4: ~14.46 GB free, and estimate_memory(8, 924) =
        # 14.31 GB -- so window_size 8 was accepted with 0.15 GB of margin
        # and died on allocator fragmentation before the forward pass
        # finished. The estimate is a model of the steady state; it cannot
        # see transient peaks, fragmentation, or the allocator's own
        # bookkeeping.
        #
        # 0.8 leaves ~3 GB of headroom on a T4, which on that same machine
        # selects window_size 5-6 at 924 px instead of 8 -- fewer views per
        # window, more submap junctions, and a run that actually completes.
        return float(report["vram_free_gb"]) * _VRAM_BUDGET_FRACTION
    return _UNIFIED_MEMORY_WINDOW_BUDGET_GB


# geometry.submap._telemetry_rotation_transform's rotation_spread_deg
# diagnostic (how much the per-keyframe local->world rotation-offset
# samples disagreed before being averaged) above this means telemetry
# conditioning was not actually honoured consistently across the submap --
# worth a prominent warning even though _telemetry_rotation_transform still
# produced *some* answer (an average of disagreeing samples).
_ROTATION_SPREAD_WARN_DEG = 15.0

# ---------------------------------------------------------------------------
# MatchingStage / BundleAdjustmentStage tuning.
#
# These live here (module constants) rather than on ``config.Config``
# because ``Config`` and its per-stage dataclasses (``GeometryConfig`` etc.)
# are outside this workstream's file ownership -- see the module docstring
# above ``BundleAdjustmentStage``. A future pass that adds a
# ``MatchingConfig`` to ``config.py`` can lift these out wholesale.
# ---------------------------------------------------------------------------

# SIFT is the default detector (see ``geometry.features``'s module
# docstring for why: scale invariance matters for altitude-varying aerial
# imagery). ``max_features`` is a generous cap, not a target -- most
# frames will detect far fewer.
_MATCH_METHOD = "sift"
_MATCH_MAX_FEATURES = 4000
_MATCH_RATIO = 0.8

# See ``tracks.select_pairs``: match each keyframe to its next few
# neighbours, plus any keyframe within this GPS radius (loop closure for a
# flight that passes near its own earlier track).
_MATCH_WINDOW = 3
_MATCH_GPS_RADIUS_M = 15.0

# A verified pair needs at least this many inlier correspondences to be
# trusted as a track-building edge -- matches the minimal-sample floor
# ``features.geometric_verify`` itself already enforces before attempting
# RANSAC at all, applied here to the *post*-RANSAC inlier count instead.
_MIN_VERIFIED_INLIERS = 8

# geometry.tracks.filter_tracks' floor: see that function's docstring for
# why < 3 observations is too poorly constrained to bother with.
_MIN_TRACK_LENGTH = 3

# geometry.triangulate's degenerate-angle floor (see that module's
# docstring): small-angle triangulation is the single-pass failure mode,
# not a rare edge case, so this stays low rather than "strict."
_MIN_TRIANGULATION_ANGLE_DEG = 1.5

# A generous post-triangulation reprojection sanity check (not a tight
# outlier filter -- bundle_adjust's own robust loss handles the rest);
# this just keeps grossly-wrong triangulations (bad match chains that
# still passed geometric verification) out of the optimizer entirely.
_MAX_REPROJECTION_PX = 15.0
# Gate applied BEFORE the first bundle adjustment. Telemetry gimbal
# rotations are good to roughly a degree, and at 4K one degree is ~37 px of
# reprojection error on a perfectly good track. Gating at 15 px before BA
# has had a chance to fix those rotations threw away 99% of tracks on real
# footage (8718 built -> 84 kept), and BA then "converged" on 84 points.
# Pass 1 keeps everything within a rotation-error budget; pass 2
# re-triangulates with the refined poses and applies the strict gate.
_MAX_REPROJECTION_PX_PREBA = 60.0

# Below this many surviving, triangulated tracks, a BAProblem is judged too
# thin to trust: bundle adjustment would either be nearly unconstrained or
# would "succeed" numerically while meaning very little. Matching then
# degrades gracefully (``StageUnavailable`` -> ``"skipped"``, see
# ``pipeline.stages``' module docstring) rather than handing bundle_adjust
# a problem it cannot meaningfully solve.
_MIN_TRACKS_FOR_BA = 20

#: Bounds on the focal-length correction bundle adjustment may apply.
#: The intrinsics prior is a generic-HFOV guess, so a real correction of
#: tens of percent is expected (COLMAP moved this footage 1.57x). But
#: focal length and depth trade off against each other, and a solve that
#: has gone wrong expresses it as an extreme focal -- so a correction
#: outside this range is rejected as a failed solve rather than applied.
_MIN_FOCAL_REFINE_FACTOR = 0.4
_MAX_FOCAL_REFINE_FACTOR = 2.5

#: Inter-quartile spread, as a fraction of the estimate, beyond which a
#: flow-measured focal length is not trusted. The measurement assumes
#: nadir view over locally flat ground at a known height; when those hold
#: the spread is ~1% (measured on real footage). A wide spread means they
#: do not hold on this flight, and a confident-looking median would be
#: hiding that.
_MAX_FOCAL_SPREAD_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class CancelToken:
    """Minimal cooperative cancellation flag for headless pipeline runs.

    Deliberately duck-type compatible with ``drishti3d.app.workers
    .CancelToken`` (both only need ``is_set()``/``cancel()``): the Qt app
    can hand its own (thread-safe, ``threading.Event``-backed) cancel token
    straight to ``run_pipeline`` without this package ever importing Qt,
    and headless/CLI/test callers that don't care about threading can use
    this plain version instead.
    """

    def __init__(self) -> None:
        self._flag = False

    def cancel(self) -> None:
        self._flag = True

    def is_set(self) -> bool:
        return self._flag


class PipelineCancelled(Exception):
    """Raised by a stage to unwind early once ``cancel_token`` is set.

    Caught inside ``pipeline.runner.run_pipeline`` itself -- it never
    escapes to ``run_pipeline``'s caller. A cancelled run returns a normal
    (partial) ``PipelineResult``, per the pipeline brief.
    """


class StageUnavailable(Exception):
    """Raised by an optional late stage when its backing module isn't ready yet.

    Distinct from a genuine bug: the runner records this as
    ``StageResult(status="skipped")`` rather than ``"failed"``, and keeps
    running the rest of the pipeline.
    """


def _check_cancel(cancel_token: CancelToken | None) -> None:
    if cancel_token is not None and cancel_token.is_set():
        raise PipelineCancelled()


# ---------------------------------------------------------------------------
# Shared mutable state threaded through every stage
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Mutable scratchpad every stage reads from and writes into.

    Stages are small and single-purpose specifically because they all
    share this one object instead of threading a growing tuple of return
    values through ``run_pipeline`` -- a new stage (or a later workstream's
    replacement of one) only needs to know which fields it reads and which
    it writes, not the full call chain.
    """

    video_path: Path
    telemetry_path: Path | None
    config: Config
    backbone_name: str
    # Explicit video-start offset for CSV/Airdata-style telemetry (seconds,
    # ``t' = t - telemetry_offset_s``); ``None`` defers to
    # ``ingest.telemetry.load_telemetry``'s own auto-detection (see that
    # function's docstring on the "explicit > auto-detected > assumed
    # zero" precedence this threads straight through to). Set from
    # ``run_pipeline``'s ``telemetry_offset_s`` parameter /
    # ``--telemetry-offset`` CLI flag.
    telemetry_offset_s: float | None = None

    video: VideoSource | None = None
    telemetry_samples: list[TelemetrySample] = field(default_factory=list)
    telemetry_stats: dict = field(default_factory=dict)
    intrinsics: CameraIntrinsics | None = None
    intrinsics_provenance: str = ""

    keyframes: list[Keyframe] = field(default_factory=list)
    frame_metrics: list[FrameMetrics] = field(default_factory=list)
    triage_report: dict = field(default_factory=dict)

    windows: list[Window] = field(default_factory=list)
    submaps: list[Submap] = field(default_factory=list)
    point_cloud: PointCloud | None = None
    poses: list[Pose] = field(default_factory=list)
    # Triangle-face connectivity for state.point_cloud, when the current
    # point cloud came from FusionStage's TSDF mesh extraction (None for a
    # plain, unmeshed point cloud -- e.g. GeometryStage's own merged output,
    # or whenever fusion was skipped/failed and geometry's cloud carried
    # through unchanged). ExportStage uses this to write real mesh formats
    # (OBJ/faceted GLB) instead of a points-only file whenever a mesh
    # actually exists.
    mesh_faces: np.ndarray | None = None

    # ``export.placement.PlacementReport`` from the check GeometryStage
    # runs between merging the submaps and fusing them -- whether the
    # windows agree on where the ground is, before any meshing time is
    # spent on the answer. ``None`` when geometry has not run yet or the
    # check could not be computed.
    placement_report: object | None = None

    # ``geometry.scale_consensus.harmonise_submap_scales``' diagnostics:
    # the one depth scale the flight agreed on, and which windows had to
    # be pulled onto it.
    scale_consensus: dict = field(default_factory=dict)

    # ``FramePlacementStage``'s coverage/overlap metrics -- where every
    # frame lands on the ground, measured before any geometry is built.
    frame_placement: dict = field(default_factory=dict)

    # ``DronePathStage`` / ``CoverageStage`` preview metrics -- the shape
    # of the flight and the ground its camera actually saw, both known
    # before any depth is computed.
    drone_path: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)

    # ``fusion.ground_bounds`` diagnostics: how much geometry was rejected
    # as physically impossible terrain, and how far out it was.
    ground_band: dict = field(default_factory=dict)

    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage interface
# ---------------------------------------------------------------------------


class PipelineStage(ABC):
    """Common interface every pipeline stage implements."""

    name: str

    @abstractmethod
    def run(
        self,
        state: PipelineState,
        cancel_token: CancelToken | None,
        progress_cb: StageProgressCb | None,
        partial_cb: PartialCb | None = None,
    ) -> tuple[dict, str]:
        """Execute this stage, mutating ``state`` in place.

        Returns ``(artifacts, message)``, which the runner attaches to a
        ``StageResult`` on success. Implementations should:

        - Check ``cancel_token`` at entry and between any long-running
          inner-loop iterations, raising ``PipelineCancelled`` when set.
        - Raise ``StageUnavailable`` (optional late stages only) when their
          backing module isn't installed/hasn't landed yet.
        - Let any other exception propagate -- the runner decides fail-fast
          (ingest/triage) vs. record-and-continue (everything after).
        """


# ---------------------------------------------------------------------------
# IngestStage
# ---------------------------------------------------------------------------


class IngestStage(PipelineStage):
    """Open the video, load telemetry (if given), and resolve camera intrinsics."""

    name = "ingest"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)

        if progress_cb:
            progress_cb(0, 3, f"opening {state.video_path.name}")
        video = VideoSource(state.video_path)
        state.video = video

        _check_cancel(cancel_token)

        telemetry_samples: list[TelemetrySample] = []
        telemetry_stats: dict = {}
        if state.telemetry_path is not None:
            if progress_cb:
                progress_cb(1, 3, f"loading telemetry from {state.telemetry_path.name}")
            telemetry_samples, telemetry_stats = load_telemetry(
                state.telemetry_path,
                video_duration_s=video.duration,
                time_offset_s=state.telemetry_offset_s,
            )
        state.telemetry_samples = telemetry_samples
        state.telemetry_stats = telemetry_stats

        _check_cancel(cancel_token)

        if progress_cb:
            progress_cb(2, 3, "resolving camera intrinsics")
        intrinsics, provenance = intrinsics_from_video(video, telemetry=(telemetry_samples, telemetry_stats))
        state.intrinsics = intrinsics
        state.intrinsics_provenance = provenance

        telemetry_format = telemetry_stats.get("format", "none")
        # These default (rather than key-erroring) whenever no telemetry
        # was loaded at all -- ``load_telemetry`` always sets them when it
        # runs (see its docstring), but ``state.telemetry_path is None`` is
        # a normal, supported case (vision-only triage) that never calls it.
        telemetry_offset_s = telemetry_stats.get("time_offset_s", 0.0)
        telemetry_offset_source = telemetry_stats.get("offset_source", "assumed_zero")
        telemetry_video_coverage_fraction = telemetry_stats.get("telemetry_video_coverage_fraction")
        artifacts = {
            "resolution": f"{video.width}x{video.height}",
            "fps": video.fps,
            "duration_s": video.duration,
            "frame_count": video.frame_count,
            "telemetry_samples": len(telemetry_samples),
            "telemetry_format": telemetry_format,
            "telemetry_offset_s": telemetry_offset_s,
            "telemetry_offset_source": telemetry_offset_source,
            "telemetry_video_segments": telemetry_stats.get("video_segments", []),
            "telemetry_video_coverage_fraction": telemetry_video_coverage_fraction,
            "intrinsics_provenance": provenance,
        }
        message = (
            f"{video.width}x{video.height} @ {video.fps:.2f}fps, {video.duration:.1f}s; "
            f"telemetry: {len(telemetry_samples)} samples ({telemetry_format}, "
            f"offset={telemetry_offset_s:.2f}s via {telemetry_offset_source}); "
            f"intrinsics: {provenance}"
        )
        if telemetry_video_coverage_fraction is not None and telemetry_video_coverage_fraction < 0.999:
            message += (
                f"; WARNING: telemetry covers only {telemetry_video_coverage_fraction * 100.0:.1f}% "
                "of the video's time range"
            )
        if state.telemetry_path is not None and telemetry_offset_source == "assumed_zero" and telemetry_format == "csv":
            message += "; WARNING: video-start offset could not be auto-detected, assumed 0.0s -- verify with --telemetry-offset"
        if progress_cb:
            progress_cb(3, 3, message)
        return artifacts, message


# ---------------------------------------------------------------------------
# Rough GPS/gimbal poses for backbone conditioning
# ---------------------------------------------------------------------------


def _gps_enu_by_keyframe(keyframes: list[Keyframe]) -> dict[int, np.ndarray]:
    """GPS-derived ENU position per geo-tagged keyframe, keyed by its index in ``keyframes``.

    Shared helper: ``_poses_from_telemetry`` (backbone pose conditioning),
    ``_camera_priors_from_telemetry`` (BA GPS priors), and
    ``GeometryStage.run`` (GPS-anchored submap merging -- see
    ``geometry.submap``'s docstring on why chained-only Sim(3) alignment
    compounds error across junctions) all need exactly the same "global
    keyframe-list index -> ENU position" mapping, from the same
    ``ingest.telemetry.telemetry_to_enu`` conversion so every consumer
    agrees on one shared ENU origin (the first geo-tagged keyframe).
    """
    from drishti3d.ingest.telemetry import telemetry_to_enu

    geo_idx = [i for i, kf in enumerate(keyframes) if kf.telemetry is not None and kf.telemetry.geo is not None]
    enu_by_idx: dict[int, np.ndarray] = {}
    if geo_idx:
        samples = [keyframes[i].telemetry for i in geo_idx]
        enu, _origin = telemetry_to_enu(samples)
        for local_i, global_i in enumerate(geo_idx):
            if np.all(np.isfinite(enu[local_i])):
                enu_by_idx[global_i] = enu[local_i]
    return enu_by_idx


def _poses_from_telemetry(keyframes: list[Keyframe]) -> list[Pose | None]:
    """Build a rough world-from-camera ``Pose`` per keyframe from its own GPS + gimbal telemetry.

    This is THE fix for the pipeline's headline metric-scale bug: MapAnything
    (``geometry.mapanything``) regresses metric geometry, but its own module
    docstring is explicit that this requires conditioning it with real
    camera poses -- "we always condition on our own ENU-frame poses" -- and
    ``geometry.mapanything.MapAnythingBackbone.predict`` only attaches
    ``camera_poses`` conditioning when ``Keyframe.pose is not None``
    (see ``GeometryStage.run``, which reads exactly that field). Before this
    function existed, nothing in the pipeline ever set ``Keyframe.pose`` --
    it stayed at its dataclass default of ``None`` for every keyframe,
    always -- so MapAnything ran with intrinsics conditioning only, no pose
    prior at all, and had no anchor to the flight's real metric scale. On
    real footage that showed up as a reconstruction ~14x too small (-92.8%
    scale error): plausible-looking, self-consistent (low reprojection
    error), and *wrong* by over an order of magnitude, exactly the silent
    failure this project cannot ship with.

    Position comes from ``ingest.telemetry.telemetry_to_enu`` (exact,
    ECEF-routed, matching this project's ENU/metres/Z-up world-frame
    convention -- see ``types`` module docstring), converting every
    geo-tagged keyframe's ``TelemetrySample.geo`` together so they all
    share one consistent ENU origin (the first geo-tagged sample), matching
    the origin ``geometry.georef.georeference`` resolves independently
    later from the same telemetry.

    Orientation comes from ``geometry.bundle.gimbal_to_R`` -- reused rather
    than reimplemented so this pose and ``BundleAdjustmentStage``'s gravity
    prior (``_camera_priors_from_telemetry``, below) agree on what
    "gimbal_pitch/roll/yaw" mean geometrically. A keyframe only gets a pose
    when it has *both* a GPS fix and a gimbal pitch + yaw reading (roll
    defaults to 0 degrees when missing, matching
    ``_camera_priors_from_telemetry``'s own convention -- DJI/Airdata
    exports typically have no dedicated gimbal-roll column); anything less
    and this returns ``None`` for that keyframe rather than guess an
    orientation (an assumed-nadir or identity-rotation guess would actively
    mislead the backbone, unlike simply omitting conditioning).
    """
    try:
        from drishti3d.geometry.bundle import gimbal_to_R
    except ImportError:
        # geometry.bundle is owned by another, concurrently-in-progress
        # workstream (see this module's docstring) -- degrade to "no pose
        # conditioning" rather than fail triage entirely if it's ever
        # temporarily missing/broken mid-development.
        logger.info("triage: geometry.bundle not available; skipping GPS/gimbal pose conditioning for keyframes")
        return [None] * len(keyframes)

    enu_by_idx = _gps_enu_by_keyframe(keyframes)

    poses: list[Pose | None] = []
    for i, kf in enumerate(keyframes):
        position = enu_by_idx.get(i)
        telemetry = kf.telemetry
        pitch = telemetry.gimbal_pitch if telemetry is not None else None
        yaw = telemetry.gimbal_yaw if telemetry is not None else None
        roll = telemetry.gimbal_roll if telemetry is not None else None
        if position is None or pitch is None or yaw is None:
            poses.append(None)
            continue
        R = gimbal_to_R(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll if roll is not None else 0.0)
        poses.append(Pose(R=R, t=position))

    return poses


# ---------------------------------------------------------------------------
# TriageStage
# ---------------------------------------------------------------------------


class TriageStage(PipelineStage):
    """Run keyframe triage over the ingested video (+ telemetry, if any)."""

    name = "triage"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)
        if state.video is None:
            raise RuntimeError("TriageStage requires an opened VideoSource from IngestStage")

        def _inner_progress(current: int, total: int, message: str) -> None:
            _check_cancel(cancel_token)
            if progress_cb is not None:
                progress_cb(current, total, message)

        telemetry = (state.telemetry_samples, state.telemetry_stats) if state.telemetry_samples else None
        keyframes, all_metrics = select_keyframes(
            state.video, state.config.triage, telemetry=telemetry, progress_cb=_inner_progress, intrinsics=state.intrinsics
        )

        # Seed each keyframe with the ingest-stage intrinsics prior when
        # triage didn't already attach a more specific one (it never does
        # today, but this keeps GeometryStage's per-keyframe intrinsics
        # lookup simple regardless).
        for kf in keyframes:
            if kf.intrinsics is None:
                kf.intrinsics = state.intrinsics

        # Seed each keyframe's rough GPS/gimbal pose too (see
        # `_poses_from_telemetry`'s docstring): this is the conditioning
        # `GeometryStage` hands to the backbone, and nothing populated it
        # before this line existed -- every `Keyframe.pose` was `None`,
        # silently disabling pose conditioning entirely regardless of how
        # much good telemetry was available.
        for kf, pose in zip(keyframes, _poses_from_telemetry(keyframes), strict=True):
            kf.pose = pose

        state.keyframes = keyframes

        # Decode every keyframe once, here, while the decoder is already
        # walking the file. Without this, yaw refinement, matching and
        # geometry each re-decode all of them -- measured at 144 s per
        # pass on this footage. See pipeline.framecache.
        if getattr(state.config.triage, "cache_keyframe_images", True):
            from drishti3d.pipeline.framecache import build_keyframe_cache

            state.keyframe_cache = build_keyframe_cache(
                state, max_size=getattr(state.config.triage, "keyframe_cache_max_size", 1920)
            )
        state.frame_metrics = all_metrics

        has_gps = any(kf.telemetry is not None and kf.telemetry.geo is not None for kf in keyframes)
        spacing_mode = "gps_baseline" if has_gps else "vision_parallax"
        report = triage_report(keyframes, all_metrics, spacing_mode=spacing_mode)
        state.triage_report = report

        artifacts = {
            "keyframes": len(keyframes),
            "frames_scanned": len(all_metrics),
            "spacing_mode": report["spacing_mode"],
        }
        message = (
            f"{len(keyframes)} keyframes selected from {len(all_metrics)} scanned frames "
            f"({report['spacing_mode']})"
        )
        return artifacts, message


# ---------------------------------------------------------------------------
# SemanticsStage
# ---------------------------------------------------------------------------


class SemanticsStage(PipelineStage):
    """Segment every keyframe, and mask out what must never become geometry.

    Sits between triage and geometry for one structural reason: the
    dynamic-object mask has to exist *before* the depth backbone runs, so
    a car's pixels never produce 3D points at all. Filtering cars out of a
    finished cloud is strictly weaker -- by then the car has already
    contributed weight to the TSDF, pulled on submap alignment, and
    inflated the scene bounding box.

    This stage is optional in the strong sense: with ``transformers``
    absent, the model unavailable, or ``semantics.enabled`` false, it
    raises ``StageUnavailable`` and the runner records it ``"skipped"``.
    The pipeline then behaves exactly as it did before semantics existed.
    It is deliberately NOT in ``_FATAL_STAGE_NAMES``.

    What it leaves on ``state``
    ---------------------------
    - ``semantic_masks``: ``{keyframe_index: SegmentationResult}``, at full
      decoded frame resolution. Consumed by ``GeometryStage`` (masking)
      and ``FusionStage`` (per-point voting).
    - ``semantic_excluded_masks``: ``{keyframe_index: (H, W) bool}``, the
      dilated union of ``EXCLUDED_CLASSES``. Precomputed here so the
      dilation cost is paid once per keyframe rather than once per window
      (windows overlap, so keyframes are revisited).

    Both use the same ``{keyframe_index: ...}`` shape as the existing
    ``keyframe_intrinsics`` dict that ``FusionStage`` already builds, so
    there is one convention for per-keyframe side data, not two.
    """

    name = "semantics"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)

        cfg = state.config.semantics
        if not cfg.enabled:
            raise StageUnavailable("semantics disabled in config (semantics.enabled = false)")

        keyframes = state.keyframes
        if not keyframes:
            raise StageUnavailable("no keyframes to segment")
        if state.video is None:
            raise StageUnavailable("SemanticsStage requires an opened VideoSource")

        # Imported here, not at module scope: `semantics.segmenter` pulls
        # in torch/transformers only when a real segmenter is constructed,
        # and `pipeline.stages` must stay importable with neither present.
        from drishti3d.semantics.classes import EXCLUDED_CLASSES, class_histogram
        from drishti3d.semantics.segmenter import create_segmenter

        segmenter = create_segmenter(
            cfg.model,
            checkpoint=cfg.checkpoint,
            max_image_size=cfg.max_image_size,
            batch_size=cfg.batch_size,
            local_weights=cfg.local_weights,
            hf_token=getattr(cfg, "hf_token", None),
        )
        if not segmenter.is_available():
            # Degrade rather than fail, and say so at WARNING: a run that
            # silently produced no labels because transformers was missing
            # would otherwise look identical to one where the scene
            # genuinely had nothing classifiable in it.
            raise StageUnavailable(
                f"segmenter {cfg.model!r} unavailable (missing torch/transformers); "
                "install the 'semantics' extra to classify this reconstruction"
            )

        device = get_device()
        masks: dict[int, object] = {}
        excluded: dict[int, np.ndarray] = {}
        label_tally: list[np.ndarray] = []

        kernel = None
        if cfg.remove_dynamic and cfg.dilate_dynamic_px > 0:
            k = int(cfg.dilate_dynamic_px)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        segmenter.load(device)
        try:
            batch = max(1, int(cfg.batch_size))
            for start in range(0, len(keyframes), batch):
                _check_cancel(cancel_token)
                chunk = keyframes[start : start + batch]
                if progress_cb:
                    progress_cb(start, len(keyframes), f"segmenting keyframe {start + 1}/{len(keyframes)}")

                frames = state.video.read_frames([kf.frame_index for kf in chunk])
                images = [f.image for f in frames if f.image is not None]
                if len(images) != len(chunk):
                    logger.warning(
                        "semantics: expected %d frames at offset %d, decoded %d; skipping batch",
                        len(chunk),
                        start,
                        len(images),
                    )
                    continue

                for offset, result in enumerate(segmenter.predict(images)):
                    kf_index = start + offset
                    masks[kf_index] = result
                    label_tally.append(result.labels.ravel()[::97])  # sparse sample for the histogram
                    if cfg.remove_dynamic:
                        m = result.dynamic_mask(EXCLUDED_CLASSES)
                        if kernel is not None:
                            m = cv2.dilate(m.astype(np.uint8), kernel).astype(bool)
                        excluded[kf_index] = m
        except Exception as exc:
            # Release the GPU before this propagates.
            #
            # Python keeps the failing frame alive through the exception's
            # traceback, and after a CUDA OOM that frame's locals still
            # reference multi-gigabyte device tensors -- so the allocator
            # frees nothing and GeometryStage inherits an almost-full GPU,
            # failing for a reason that has nothing to do with geometry.
            # `clear_frames` drops those references so `unload`'s
            # `empty_cache()` can actually reclaim the memory.
            segmenter.unload()
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
            _release_cuda_memory()
            raise
        finally:
            segmenter.unload()
            _release_cuda_memory()

        state.semantic_masks = masks
        state.semantic_excluded_masks = excluded
        state.semantic_model = cfg.checkpoint if cfg.model != "null" else "null"

        # A 2D histogram over sampled pixels, not over 3D points -- the
        # per-point one comes later, from FusionStage, and the two are
        # reported separately because they answer different questions
        # ("what did the camera see" vs "what did we reconstruct").
        pixel_hist: dict[str, float] = {}
        excluded_fraction = 0.0
        if label_tally:
            stacked = np.concatenate(label_tally)
            pixel_hist = class_histogram(stacked)
            excluded_fraction = float(np.isin(stacked, list(EXCLUDED_CLASSES)).mean() * 100.0)

        state.semantic_pixel_histogram = pixel_hist
        state.semantic_excluded_pixel_pct = excluded_fraction

        artifacts = {
            "keyframes_segmented": len(masks),
            "model": state.semantic_model,
            "device": device,
            "excluded_pixel_pct": round(excluded_fraction, 2),
            "pixel_class_pct": pixel_hist,
            "dynamic_removal": bool(cfg.remove_dynamic),
        }
        message = (
            f"segmented {len(masks)}/{len(keyframes)} keyframes with {state.semantic_model} on {device}; "
            f"{excluded_fraction:.1f}% of sampled pixels masked as dynamic/sky"
        )
        return artifacts, message


def _excluded_keep_mask(
    state,
    window_keyframe_positions: list[int],
    target_hw: tuple[int, int],
    n_points_per_view: int,
) -> np.ndarray | None:
    """``(V * H * W,)`` bool: which backbone output points survive masking.

    ``SemanticsStage`` stores masks at full decoded frame resolution, but
    ``GeometryStage`` feeds the backbone a downscaled image, so each mask
    is resampled to the backbone's working size with nearest-neighbour --
    the only correct interpolation for a label mask, since any averaging
    would invent intermediate values that are not classes.

    Returns ``None`` when no mask is available for this window at all, so
    the caller can skip the filtering work entirely rather than building
    and applying an all-True array.
    """
    excluded = getattr(state, "semantic_excluded_masks", None)
    if not excluded:
        return None
    if not any(pos in excluded for pos in window_keyframe_positions):
        return None

    h, w = target_hw
    keeps: list[np.ndarray] = []
    for pos in window_keyframe_positions:
        m = excluded.get(pos)
        if m is None:
            keeps.append(np.ones(n_points_per_view, dtype=bool))
            continue
        resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        keep = ~resized.reshape(-1)
        if keep.size != n_points_per_view:
            # Backbone returned a point grid that is not its input image
            # grid; filtering by pixel would misalign silently, so decline.
            logger.warning(
                "semantics: mask has %d pixels but backbone produced %d points per view; "
                "skipping dynamic masking for this window",
                keep.size,
                n_points_per_view,
            )
            return None
        keeps.append(keep)
    return np.concatenate(keeps)


#: Successive resolution scales tried after a CUDA OOM. Attention cost in a
#: ViT encoder grows with the SQUARE of the token count, and tokens grow
#: with the square of the image's long side -- so halving resolution cuts
#: attention memory ~16x. That steepness is why a couple of backoff steps
#: rescue almost any OOM, and why the first step is already aggressive.
_OOM_BACKOFF_SCALES = (0.75, 0.5, 0.35)


def _predict_with_oom_backoff(
    backbone,
    images: list[np.ndarray],
    intrinsics_list: list,
    poses_list: list,
    *,
    index: int,
    device: str | None = None,
):
    """Run ``backbone.predict``, retrying at lower resolution on CUDA OOM.

    A window that does not fit is not a broken window -- it is a window the
    current resolution cannot afford on this GPU. Reconstructing it smaller
    is strictly better than dropping it: a coarser submap still contributes
    geometry, poses and coverage, while a dropped one leaves a hole that
    every later stage silently inherits.

    This matters most on the deployment target. The benchmark box has 16 GB;
    the 6 GB RTX 4060 this system is meant to run on will hit the ceiling
    far sooner, and it should degrade in quality rather than fail outright.

    The retry is logged at WARNING and recorded by the caller, so a run that
    quietly dropped to half resolution cannot be mistaken for one that did
    not -- the resolution a submap was built at is part of what its accuracy
    means.
    """
    try:
        import torch

        oom_errors: tuple[type[BaseException], ...] = (torch.cuda.OutOfMemoryError,)
        if hasattr(torch, "OutOfMemoryError"):
            oom_errors = (*oom_errors, torch.OutOfMemoryError)
    except ImportError:
        oom_errors = ()

    def _reclaim(exc: BaseException) -> str:
        """Free everything the failed attempt held, and return its message.

        THE critical step in this whole function. Python keeps the failing
        frame alive through the exception's traceback, and after a CUDA OOM
        that frame's locals still reference multi-gigabyte device tensors.
        Holding the exception across retries therefore leaves each attempt
        with LESS memory than the one before it, so the backoff ladder is
        guaranteed to fail however far it climbs down -- which is exactly
        what a 924px/8-view window did on a 16 GB T4: the retry reported
        10.36 GB already allocated before it asked for anything, against
        ~4.6 GB of model weights.

        Only the formatted message is kept; the exception object itself is
        never stored.
        """
        message = f"{type(exc).__name__}: {exc}"
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        _release_cuda_memory()
        return message

    try:
        return backbone.predict(images, intrinsics=intrinsics_list, poses=poses_list)
    except oom_errors as exc:
        last_error = _reclaim(exc)
        logger.warning(
            "geometry: window %d ran out of memory at %dx%d; retrying smaller",
            index,
            images[0].shape[1],
            images[0].shape[0],
        )

    for scale in _OOM_BACKOFF_SCALES:
        long_side = max(images[0].shape[:2])
        target = max(196, int(long_side * scale))
        smaller, scaled_intrinsics = _resize_for_backbone(images, intrinsics_list, target)
        try:
            result = backbone.predict(smaller, intrinsics=scaled_intrinsics, poses=poses_list)
        except oom_errors as exc:
            last_error = _reclaim(exc)
            continue
        logger.warning(
            "geometry: window %d reconstructed at %dx%d (%.0f%% of requested) after OOM backoff",
            index,
            smaller[0].shape[1],
            smaller[0].shape[0],
            scale * 100,
        )
        return result

    raise RuntimeError(
        f"window {index} could not fit on {device or 'this device'} even at "
        f"{_OOM_BACKOFF_SCALES[-1]:.0%} resolution. Lower geometry.max_image_size or "
        f"geometry.window_size: ViT attention cost grows with the SQUARE of the token "
        f"count, so halving the long side cuts memory roughly 16x. "
        f"Last failure: {last_error}"
    )


def _release_cuda_memory() -> None:
    """Empty the caching allocator on every CUDA device. Never raises.

    Called between stages that each load their own model. Without it, a
    stage that failed part-way leaves its allocations resident and the
    next stage OOMs for reasons of its own that are really the previous
    stage's.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    except Exception:
        logger.debug("could not empty the CUDA caching allocator", exc_info=True)


@contextlib.contextmanager
def _device_context(device: str | None):
    """Pin the calling thread to ``device`` for the duration of the block.

    A no-op for ``None``, for non-CUDA devices, and when torch is absent --
    so the single-device, MPS and CPU paths behave exactly as they did
    before multi-device support existed.
    """
    if not device or not str(device).startswith("cuda"):
        yield
        return
    try:
        import torch
    except ImportError:
        yield
        return
    with torch.cuda.device(device):
        yield


def _decode_window(state, window, keyframes, cfg, decode_lock) -> tuple | None:
    """Decode + resize one window's frames. Returns the backbone's inputs, or ``None``.

    Split out of ``GeometryStage.run`` so the same decode path serves the
    sequential, prefetched and multi-GPU schedules -- three copies of this
    would be three chances for them to diverge on, say, whether intrinsics
    get rescaled alongside the image.

    Returns ``None`` when the decoder gave back fewer frames than the window
    asked for, which the caller treats as "skip this window": a window whose
    images and keyframes disagree in length would silently pair each image
    with the wrong camera.
    """
    window_keyframes = [keyframes[idx] for idx in window.keyframe_indices()]
    frame_indices = [kf.frame_index for kf in window_keyframes]

    cache = getattr(state, "keyframe_cache", None)
    cache_scales: list[float] | None = None
    if cache is not None:
        idxs = window.keyframe_indices()
        cached = [cache.get(i) for i in idxs]
        if all(im is not None for im in cached):
            images = cached
            cache_scales = [cache.scale_for(i) for i in idxs]
        else:
            images = []
    else:
        images = []

    if not images:
        with decode_lock:
            frames = state.video.read_frames(frame_indices) if state.video is not None else []
        images = [f.image for f in frames if f.image is not None]

    if len(images) != len(window_keyframes):
        return None

    intrinsics_list = [kf.intrinsics or state.intrinsics for kf in window_keyframes]
    if cache_scales is not None:
        from drishti3d.geometry.mapanything import scale_intrinsics

        intrinsics_list = [
            scale_intrinsics(intr, sc) if intr is not None else None
            for intr, sc in zip(intrinsics_list, cache_scales, strict=True)
        ]
    poses_list = [kf.pose for kf in window_keyframes]
    images, intrinsics_list = _resize_for_backbone(images, intrinsics_list, cfg.max_image_size)
    return images, intrinsics_list, poses_list, len(window_keyframes)


def _anchor_window(state, result, images, intrinsics_list, poses_list, cfg, index: int, backbone=None, window=None):
    """Parallax-anchor one window's depth. Returns the (possibly rescaled) result.

    Diagnostics land in ``state.depth_anchor_diags[index]`` -- a dict
    created by ``GeometryStage.run`` before any worker starts, so writes
    from several window threads are plain key assignments and need no lock.

    A failure inside anchoring is logged and the window continues with the
    backbone's own depth: a window whose scale could not be measured is
    still a window, and losing it entirely would be worse than reporting
    it unanchored.
    """
    if result is None or not getattr(cfg, "depth_anchor", False):
        return result
    from drishti3d.geometry.depth_anchor import anchor_depth_fused

    # Per-view telemetry altitude, for the GPS half of the fusion.
    #
    # ``GeometryConfig.gps_altitude_anchor`` gates this half. It was
    # declared and documented but never read, so turning it off did
    # nothing -- passing no altitudes is what actually disables it, since
    # anchor_depth_fused falls back to parallax alone when it has none.
    altitudes: list[float | None] = []
    if window is not None and getattr(cfg, "gps_altitude_anchor", True):
        for kf_i in window.keyframe_indices():
            kf = state.keyframes[kf_i] if kf_i < len(state.keyframes) else None
            geo = getattr(getattr(kf, "telemetry", None), "geo", None)
            altitudes.append(getattr(geo, "alt_rel", None) if geo is not None else None)

    try:
        anchored, diag = anchor_depth_fused(
            images,
            result,
            poses_list,
            intrinsics_list,
            altitudes,
            max_features=getattr(cfg, "depth_anchor_max_features", 3000),
            min_samples=getattr(cfg, "depth_anchor_min_samples", 30),
            max_ratio=getattr(cfg, "depth_anchor_max_ratio", 20.0),
        )
    except Exception:
        logger.warning("geometry: window %d depth anchoring crashed; using unanchored depth", index, exc_info=True)
        diags = getattr(state, "depth_anchor_diags", None)
        if diags is not None:
            diags[index] = {"applied": False, "failure": "exception (see log)"}
        return result

    # (The GPS-altitude estimate is now fused inside anchor_depth_fused
    # rather than tried as a separate fallback: a hard switch discarded
    # the weaker estimate entirely and made the answer discontinuous at
    # the sample-count threshold.)

    masked = result.metadata.get("masked_fraction") if result.metadata else None
    if masked:
        diag["masked_fraction_mean"] = float(np.mean(masked))
    if not diag.get("applied"):
        logger.warning("geometry: window %d NOT depth-anchored: %s", index, diag.get("failure", "unknown"))

    # Plane-sweep MVS: replace the regressed depth with matched depth.
    #
    # Runs AFTER anchoring on purpose. The sweep is a narrow band around
    # the incoming depth, so it needs that depth to be at roughly the
    # right scale first -- anchoring supplies exactly that, and the sweep
    # then supplies the precision anchoring cannot.
    if getattr(cfg, "plane_sweep", False) and result.images is not None:
        from drishti3d.geometry.plane_sweep import refine_depth_by_plane_sweep

        try:
            sweep = refine_depth_by_plane_sweep(
                np.asarray(anchored.images),
                np.asarray(anchored.depth),
                list(anchored.poses),
                list(anchored.intrinsics),
                device=_plane_sweep_device(),
                n_hypotheses=int(getattr(cfg, "plane_sweep_hypotheses", 48)),
                range_fraction=float(getattr(cfg, "plane_sweep_range_fraction", 0.15)),
                min_ncc=float(getattr(cfg, "plane_sweep_min_ncc", 0.5)),
            )
        except Exception:
            logger.warning("geometry: window %d plane sweep failed; keeping regressed depth", index, exc_info=True)
        else:
            # Rebuild the world points from the refined depth along each
            # pixel's own ray. Depth is the only thing that changed, so
            # re-deriving points from it keeps them exactly consistent.
            anchored = _points_from_depth(anchored, sweep.depth)
            diag["plane_sweep"] = sweep.stats

    # Refinement pass (GeometryConfig.depth_prior_refine): hand the anchored
    # depth back to the backbone as its prior and let it regress detail on
    # a correctly-scaled field. The anchor measurement is repeated on the
    # output as a check -- a ratio near 1.0 means the prior was honoured.
    if (
        diag.get("applied")
        and getattr(cfg, "depth_prior_refine", False)
        and backbone is not None
        and getattr(backbone, "supports_depth_prior", False)
    ):
        try:
            refined = backbone.predict(
                images, intrinsics=intrinsics_list, poses=poses_list, depth_priors=list(anchored.depth)
            )
            _, check = anchor_depth_to_parallax(
                images,
                refined,
                poses_list,
                intrinsics_list,
                max_features=getattr(cfg, "depth_anchor_max_features", 3000),
                min_samples=getattr(cfg, "depth_anchor_min_samples", 30),
                max_ratio=getattr(cfg, "depth_anchor_max_ratio", 20.0),
            )
            diag["refine"] = {
                "ratio_after_prior": check.get("window_ratio"),
                "samples": check.get("samples_total"),
            }
            logger.info(
                "geometry: window %d refined with depth prior; residual ratio %.3f",
                index,
                check.get("window_ratio") or float("nan"),
            )
            refined.metadata = {**(refined.metadata or {}), "depth_anchor": diag}
            anchored = refined
        except Exception:
            logger.warning("geometry: window %d depth-prior refinement failed; keeping anchored pass", index, exc_info=True)
            diag["refine"] = {"failed": True}

    diags = getattr(state, "depth_anchor_diags", None)
    if diags is not None:
        diags[index] = diag
    return anchored


def _match_cfg(state, name: str, fallback):
    """Read a ``MatchingConfig`` field, falling back to this module's constant.

    ``config.MatchingConfig``'s fields were written to mirror these module
    constants 1:1 so the stage could be switched over "mechanically" -- but
    the switch never happened, so every ``quality_profile`` setting for
    matching and bundle adjustment was inert. The "fast" profile has been
    asking for ``detect_scale=0.5``, ``max_points_in_ba=2000`` and
    ``ba_max_iterations=50`` all along while the stage ran at full
    resolution, unbounded points and 100 iterations.
    """
    cfg = getattr(state.config, "matching", None)
    if cfg is None or not hasattr(cfg, name):
        return fallback
    value = getattr(cfg, name)
    return fallback if value is None and name not in ("max_points_in_ba",) else value


def _plane_sweep_device() -> str:
    """Device for the plane sweep: the accelerator when one exists.

    The sweep is grid_sample + unfold over a hypothesis stack -- exactly
    the shape of work a GPU is for, and both ops are available on MPS
    (verified on torch 2.14).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _points_from_depth(result, depth: np.ndarray):
    """Rebuild ``BackboneResult.points`` from refined per-pixel depth.

    Each pixel's world point is its own viewing ray scaled to the new
    depth, so nothing but depth changes -- no resampling, no smoothing,
    and the pixel-to-point correspondence every downstream stage relies
    on (colour, confidence, view index) is preserved exactly.
    """
    from dataclasses import replace

    points = np.array(result.points, dtype=np.float64, copy=True)
    for v, (pose, intr) in enumerate(zip(result.poses, result.intrinsics, strict=False)):
        d = np.asarray(depth[v], dtype=np.float64)
        h, w = d.shape
        ys, xs = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        rays = np.stack([(xs - intr.cx) / intr.fx, (ys - intr.cy) / intr.fy, np.ones_like(xs)], axis=-1)
        cam = rays * d[..., None]
        valid = d > 1e-6
        world = cam @ np.asarray(pose.R, dtype=np.float64).T + np.asarray(pose.t, dtype=np.float64)
        points[v][valid] = world[valid]
    return replace(result, points=points, depth=np.asarray(depth, dtype=np.float64))


def _refine_yaw_from_flow(state, keyframes, cfg_geometry) -> dict:
    """Overwrite each keyframe's conditioning rotation with a flow-measured yaw.

    Mutates ``kf.pose`` in place for keyframes where a yaw was measured;
    keyframes that could not be measured keep their telemetry pose and are
    listed as such in the returned diagnostics. Frames are decoded once,
    downscaled to ``yaw_flow_max_size``, and released.
    """
    if not getattr(cfg_geometry, "yaw_from_flow", False) or state.video is None or not keyframes:
        return {"enabled": False}
    try:
        from drishti3d.geometry.bundle import gimbal_to_R
        from drishti3d.geometry.mapanything import resize_preserving_aspect
        from drishti3d.geometry.yaw_from_flow import estimate_yaw_from_flow
    except ImportError as exc:
        logger.info("geometry: yaw-from-flow unavailable (%s); keeping telemetry yaw", exc)
        return {"enabled": False, "failure": str(exc)}

    max_size = int(getattr(cfg_geometry, "yaw_flow_max_size", 640))
    cache = getattr(state, "keyframe_cache", None)
    images: list = []
    for i, kf in enumerate(keyframes):
        img = cache.get(i) if cache is not None else None
        if img is None:
            try:
                frames = state.video.read_frames([kf.frame_index])
                img = frames[0].image if frames and frames[0].image is not None else None
            except Exception:
                img = None
        images.append(resize_preserving_aspect(img, max_size)[0] if img is not None else None)

    positions = [kf.pose.t if kf.pose is not None else None for kf in keyframes]
    pitch = [kf.telemetry.gimbal_pitch if kf.telemetry is not None else None for kf in keyframes]
    roll = [kf.telemetry.gimbal_roll if kf.telemetry is not None else None for kf in keyframes]
    prior = [kf.telemetry.gimbal_yaw if kf.telemetry is not None else None for kf in keyframes]

    try:
        yaws, diag = estimate_yaw_from_flow(images, positions, pitch, roll, prior)
    except Exception:
        logger.warning("geometry: yaw-from-flow crashed; keeping telemetry yaw", exc_info=True)
        return {"enabled": True, "failure": "exception (see log)"}
    finally:
        images.clear()

    replaced = 0
    for kf, yaw, entry in zip(keyframes, yaws, diag["per_keyframe"], strict=True):
        if entry["source"] != "flow" or yaw is None or kf.pose is None:
            continue
        p = kf.telemetry.gimbal_pitch if kf.telemetry is not None else None
        r = kf.telemetry.gimbal_roll if kf.telemetry is not None else None
        R = gimbal_to_R(yaw_deg=yaw, pitch_deg=p if p is not None else -90.0, roll_deg=r if r is not None else 0.0)
        kf.pose = Pose(R=R, t=kf.pose.t)
        replaced += 1
    diag["enabled"] = True
    diag["poses_replaced"] = replaced
    logger.info("geometry: yaw-from-flow replaced %d/%d conditioning rotations", replaced, len(keyframes))
    return diag


def _submap_from_result(state, window, result, index: int) -> tuple[Submap, int]:
    """Turn one backbone result into a ``Submap``. Returns ``(submap, points_removed)``.

    Pure with respect to shared state -- it reads ``state.semantic_excluded_masks``
    but mutates nothing -- which is what makes it safe to call from several
    worker threads at once.
    """
    # Fix 3: BackboneResult.images (when the backbone reports it) is
    # pixel-aligned with result.points -- same (V, H, W) -- so flattening
    # both the same way (reshape(-1, 3)) lines up each 3D point with its own
    # source-pixel colour. Before this, no code path ever attached colour to
    # a Submap's raw points at all, which is why point_cloud.ply/.las had no
    # red/green/blue properties even though the mesh (coloured separately,
    # downstream, during TSDF integration) did.
    result_rgb = np.asarray(result.images).reshape(-1, 3).astype(np.uint8) if result.images is not None else None

    submap_xyz = np.asarray(result.points).reshape(-1, 3)
    submap_conf = np.asarray(result.confidence).reshape(-1)
    removed = 0
    # Which view each flattened point came from: (V, H, W) -> V blocks of
    # H*W. Carried through every mask below so fusion.reanchor can still
    # find "the points of view v" after the grid is gone.
    pts_shape = np.asarray(result.points).shape
    view_index = (
        np.repeat(np.arange(pts_shape[0], dtype=np.int32), int(np.prod(pts_shape[1:-1])))
        if len(pts_shape) == 4
        else None
    )

    # Dynamic-object removal, applied at the earliest point it can be: these
    # pixels' depths were regressed, but they never become part of any
    # Submap, so nothing downstream -- alignment, TSDF weights, bounding
    # box, rasters -- ever sees a vehicle or a person. `result.points` is
    # pixel-aligned with the backbone's *input* images (same (V, H, W) grid
    # that result.images uses), which is what makes a flat boolean gather
    # legitimate here.
    points_grid = np.asarray(result.points)
    if points_grid.ndim == 4:
        per_view = int(points_grid.shape[1] * points_grid.shape[2])
        view_hw = (int(points_grid.shape[1]), int(points_grid.shape[2]))
        keep = _excluded_keep_mask(state, window.keyframe_indices(), view_hw, per_view)
        if keep is not None and keep.size == submap_xyz.shape[0]:
            removed = int((~keep).sum())
            submap_xyz = submap_xyz[keep]
            submap_conf = submap_conf[keep]
            if result_rgb is not None:
                result_rgb = result_rgb[keep]
            if view_index is not None:
                view_index = view_index[keep]
            if removed:
                logger.info("geometry: window %d dropped %d/%d points as dynamic/sky", index, removed, keep.size)

    # Bound the window's contribution. Memory downstream (merge, outlier
    # removal, photometric verification) is linear in point count, and
    # with the backbone's edge mask off a 956 px window can emit ~4M
    # points -- 19 windows of that was enough to get the process killed
    # during fusion on a 16 GB machine. A uniform random subsample keeps
    # the spatial distribution and the confidence mix intact; the mesh is
    # voxel-capped at ~1.5M points before TSDF anyway, so nothing this
    # removes would have reached the surface.
    budget = int(getattr(state.config.geometry, "max_points_per_window", 0) or 0)
    if budget > 0 and submap_xyz.shape[0] > budget:
        rng = np.random.default_rng(1000 + index)
        keep_idx = np.sort(rng.choice(submap_xyz.shape[0], size=budget, replace=False))
        logger.info(
            "geometry: window %d subsampled %d -> %d points (max_points_per_window)",
            index,
            submap_xyz.shape[0],
            budget,
        )
        submap_xyz = submap_xyz[keep_idx]
        submap_conf = submap_conf[keep_idx]
        if result_rgb is not None:
            result_rgb = result_rgb[keep_idx]
        if view_index is not None:
            view_index = view_index[keep_idx]

    submap = Submap(
        window=window,
        poses=result.poses,
        points=PointCloud(xyz=submap_xyz, rgb=result_rgb),
        confidence=submap_conf,
        keyframe_indices=window.keyframe_indices(),
        local_origin=result.poses[0],
        view_index=view_index,
    )
    return submap, removed


def _run_windows(
    *,
    state,
    windows,
    keyframes,
    cfg,
    backbones,
    submaps: list,
    decode_lock,
    cancel_token,
    progress_cb,
    on_window_done,
    on_cancel,
    devices=None,
) -> None:
    """Reconstruct every window, using every backbone/device provided.

    One backbone means the historical sequential path, with the next
    window's frames decoded on a helper thread while the GPU works on the
    current one. Several backbones means several windows genuinely in
    flight at once, one per device.

    ``submaps`` is appended **in window order** in both modes, regardless of
    the order results actually complete in. That matters: ``merge_submaps``
    chains submaps to each other when it cannot anchor one independently
    (see ``geometry.submap``), and a chain assembled in completion order
    rather than flight order would link windows that do not overlap.
    """
    total = len(windows)
    results: dict[int, tuple[Submap, int]] = {}
    # Per-window failures are collected rather than swallowed. A stage that
    # reconstructs 0 of N windows but still reports "ok" is the worst
    # possible outcome: the run looks healthy, every later stage skips for
    # its own plausible-sounding reason, and the actual error appears only
    # as a log warning that a `| tail` will usually have cut off.
    errors: dict[int, str] = {}

    def reconstruct(index: int, backbone, device: str | None = None) -> None:
        decoded = _decode_window(state, windows[index], keyframes, cfg, decode_lock)
        if decoded is None:
            msg = "decoded too few frames for this window"
            logger.warning("geometry: window %d %s; skipping window", index, msg)
            errors[index] = msg
            return
        images, intrinsics_list, poses_list, _n = decoded

        # Pin this THREAD to the device its backbone lives on before any
        # kernel launches.
        #
        # torch's "current device" is thread-local and defaults to cuda:0 in
        # every new thread. A worker running the cuda:1 replica without this
        # launches kernels on cuda:0 that dereference cuda:1 pointers --
        # `CUDA error: an illegal memory access was encountered`, reported
        # asynchronously at whatever unrelated CUDA call happens next
        # (typically empty_cache() during unload, which is why the traceback
        # blames teardown rather than inference).
        with _device_context(device):
            result = _predict_with_oom_backoff(
                backbone,
                images,
                intrinsics_list,
                poses_list,
                index=index,
                device=device,
            )
        result = _anchor_window(state, result, images, intrinsics_list, poses_list, cfg, index, backbone=backbone, window=windows[index])
        results[index] = _submap_from_result(state, windows[index], result, index)

    def flush_in_order(next_expected: int) -> int:
        """Append every completed window from ``next_expected`` onward, in order."""
        while next_expected in results or next_expected in skipped:
            if next_expected in skipped:
                skipped.discard(next_expected)
            else:
                submap, removed = results.pop(next_expected)
                submaps.append(submap)
                on_window_done(next_expected, submap, removed)
            next_expected += 1
        return next_expected

    skipped: set[int] = set()

    if len(backbones) == 1:
        return _run_windows_sequential(
            state=state,
            windows=windows,
            keyframes=keyframes,
            cfg=cfg,
            backbone=backbones[0],
            device=(devices or [None])[0],
            submaps=submaps,
            decode_lock=decode_lock,
            cancel_token=cancel_token,
            progress_cb=progress_cb,
            on_window_done=on_window_done,
            on_cancel=on_cancel,
        )

    # ---- multi-device ----------------------------------------------------
    # A bounded pool with exactly one worker per device, each pinned to its
    # own backbone. Threads (not processes) because the expensive call
    # releases the GIL inside torch, and because a process pool would have
    # to pickle decoded frames across a pipe -- far more expensive than the
    # inference it is trying to overlap.
    # Each queue entry pairs a backbone with the device it was loaded onto,
    # so a worker always knows which device to pin itself to -- a backbone
    # alone does not carry that reliably across implementations.
    device_queue: queue.Queue = queue.Queue()
    for b, dev in zip(backbones, devices or [None] * len(backbones), strict=False):
        device_queue.put((b, dev))

    def worker(index: int):
        backbone, dev = device_queue.get()
        try:
            reconstruct(index, backbone, dev)
        finally:
            device_queue.put((backbone, dev))

    next_expected = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(backbones)) as pool:
        futures = {pool.submit(worker, i): i for i in range(total)}
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors[index] = traceback.format_exc()
                logger.warning("geometry: window %d failed (%s); continuing", index, exc, exc_info=True)
            if index not in results:
                skipped.add(index)
            if progress_cb:
                progress_cb(done_count, total, f"window {done_count}/{total} ({len(backbones)} devices)")
            next_expected = flush_in_order(next_expected)

            if cancel_token is not None and cancel_token.is_set():
                for f in futures:
                    f.cancel()
                # Raises PipelineCancelled. The `with` block cancels what it
                # can and joins the rest -- an in-flight window finishes its
                # forward pass rather than being killed mid-CUDA-call.
                on_cancel()

    flush_in_order(next_expected)
    return errors


def _run_windows_sequential(
    *,
    state,
    windows,
    keyframes,
    cfg,
    backbone,
    submaps: list,
    decode_lock,
    cancel_token,
    progress_cb,
    on_window_done,
    on_cancel,
    device: str | None = None,
) -> None:
    """One device, with the next window's frames decoded while the GPU works.

    Decode is CPU/IO-bound and inference is GPU-bound, so without the
    prefetch the GPU idles through every seek and H.264 decode. Overlapping
    them changes no result -- only when the bytes arrive -- which is why
    this is on by default and why it applies equally to the CPU/MPS paths.
    """
    total = len(windows)
    prefetch = getattr(cfg, "prefetch_frames", True)
    errors: dict[int, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as decoder:
        pending = decoder.submit(_decode_window, state, windows[0], keyframes, cfg, decode_lock) if total else None

        for i in range(total):
            if cancel_token is not None and cancel_token.is_set():
                on_cancel()

            if progress_cb:
                progress_cb(i, total, f"window {i + 1}/{total} ({windows[i].size} keyframes)")

            # Decode runs on the prefetch thread, so its exception surfaces
            # here, at .result(). Record it like any other window failure
            # rather than letting one unreadable window abort the flight.
            decode_error: str | None = None
            decoded = None
            if pending is not None:
                try:
                    decoded = pending.result()
                except Exception as exc:
                    decode_error = traceback.format_exc()
                    logger.warning("geometry: window %d decode failed (%s); continuing", i, exc, exc_info=True)

            # Kick off the next decode BEFORE running inference, so the two
            # genuinely overlap rather than merely being adjacent.
            if prefetch and i + 1 < total:
                pending = decoder.submit(_decode_window, state, windows[i + 1], keyframes, cfg, decode_lock)
            elif i + 1 < total:
                pending = None
            else:
                pending = None

            if decoded is None:
                if decode_error is not None:
                    errors[i] = decode_error
                else:
                    msg = "decoded too few frames for this window"
                    logger.warning("geometry: window %d %s; skipping window", i, msg)
                    errors[i] = msg
                if not prefetch and i + 1 < total:
                    pending = decoder.submit(_decode_window, state, windows[i + 1], keyframes, cfg, decode_lock)
                continue

            # One bad window must not cost every window after it -- but the
            # failure is recorded, not swallowed, so GeometryStage can fail
            # the stage outright if nothing at all reconstructed.
            try:
                images, intrinsics_list, poses_list, _n = decoded
                with _device_context(device):
                    result = _predict_with_oom_backoff(
                        backbone,
                        images,
                        intrinsics_list,
                        poses_list,
                        index=i,
                        device=device,
                    )
                result = _anchor_window(state, result, images, intrinsics_list, poses_list, cfg, i, backbone=backbone, window=windows[i])
                submap, removed = _submap_from_result(state, windows[i], result, i)
            except PipelineCancelled:
                raise
            except Exception as exc:
                errors[i] = traceback.format_exc()
                logger.warning("geometry: window %d failed (%s); continuing", i, exc, exc_info=True)
                if not prefetch and i + 1 < total:
                    pending = decoder.submit(_decode_window, state, windows[i + 1], keyframes, cfg, decode_lock)
                continue

            submaps.append(submap)
            on_window_done(i, submap, removed)

            if not prefetch and i + 1 < total:
                pending = decoder.submit(_decode_window, state, windows[i + 1], keyframes, cfg, decode_lock)

    return errors


def _resize_for_backbone(
    images: list[np.ndarray],
    intrinsics_list: list[CameraIntrinsics | None],
    max_image_size: int,
) -> tuple[list[np.ndarray], list[CameraIntrinsics | None]]:
    """Downscale ``images`` (long side -> ``max_image_size``) and scale ``intrinsics_list`` to match.

    ``GeometryConfig.max_image_size`` used to only be honoured *inside*
    ``geometry.mapanything.MapAnythingBackbone`` -- any other backbone
    (most importantly ``NullBackbone``, the always-available zero-GPU
    fallback) received native-resolution images straight from the decoded
    video. For real drone footage that is 4K: ``NullBackbone`` ray-casts
    one point per pixel (see its module docstring), so a single window's
    predict call was producing on the order of 8M points *per frame*,
    which makes the pipeline unusable (memory, merge time, export time) --
    not a rare edge case, but the default for any 4K source. Resizing here,
    once, before dispatching to whichever backbone is configured, fixes
    that for every backbone uniformly rather than only for MapAnything.

    Reuses ``geometry.mapanything.resize_preserving_aspect``/
    ``scale_intrinsics`` (already unit-tested, pure numpy/opencv, no torch
    dependency) rather than reimplementing the same resize-and-rescale
    logic a second time.
    """
    resized_images: list[np.ndarray] = []
    resized_intrinsics: list[CameraIntrinsics | None] = []
    for img, intr in zip(images, intrinsics_list, strict=True):
        resized_img, scale = resize_preserving_aspect(img, max_image_size)
        resized_images.append(resized_img)
        resized_intrinsics.append(scale_intrinsics(intr, scale) if intr is not None else None)
    return resized_images, resized_intrinsics


# ---------------------------------------------------------------------------
# GeometryStage
# ---------------------------------------------------------------------------


class PosePriorStage(PipelineStage):
    """Refine the telemetry camera poses against the images BEFORE dense geometry.

    Why this runs first
    -------------------
    Everything the dense stage does is conditioned on the camera poses:
    MapAnything takes them as input, the parallax depth anchor triangulates
    with them, and the submap merge aligns to them. Telemetry gives
    positions to a few metres and (after ``geometry.yaw_from_flow``)
    rotations to a degree or two. A degree is not enough here: triangulated
    depth from a nadir pair is sensitive to rotation as ``Z / B`` -- about
    5x at survey spacing -- so a 1 degree rotation error is ~5% depth error,
    and the ~5 degree error measured at the loop corners was 45%. Measured on
    the sample flight: with flow-yaw poses the anchored ground sat 17 m high
    in one window and 50 m deep in two others; the same bundle adjustment,
    run after geometry, reached 1.9 px reprojection (~0.05 degree).

    So the sparse refinement moves in front of the dense stage. It needs
    only the keyframe images, SIFT tracks and the GPS priors -- none of the
    dense output -- and it costs about seven minutes on the sample flight.
    The post-geometry ``BundleAdjustmentStage`` still runs afterwards; it
    now starts from a good solution and converges quickly.

    This IS the "telemetry alignment" step of the intended architecture
    (MapAnything -> SRT/GPS alignment -> refinement), done where it has the
    most leverage: before the model that depends on it.

    Skips (never fails the run) when any keyframe lacks a conditioning pose
    or when matching cannot triangulate enough tracks; GeometryStage then
    proceeds with the flow-yaw telemetry poses exactly as before.
    """

    name = "pose_prior"

    def _measure_focal(self, state, keyframes) -> None:
        """Replace a guessed focal length with one measured from the flight.

        Skipped when ``intrinsics_provenance`` shows a real source (EXIF, a
        camera database entry the user trusts, an explicit calibration) --
        those are measurements and this estimate is not better than them.
        Only ``default_guess``, the generic-HFOV fallback, is replaced.
        """
        if not getattr(state.config.geometry, "focal_from_flow", True):
            return
        provenance = getattr(state, "intrinsics_provenance", "")
        if provenance != "default_guess":
            logger.info(
                "focal from flow: intrinsics provenance is %r, not a guess; keeping them", provenance
            )
            return

        from drishti3d.geometry.focal_from_flow import estimate_focal_from_flow

        cache = getattr(state, "keyframe_cache", None)
        base = keyframes[0].intrinsics or state.intrinsics
        if base is None:
            return
        images = [cache.get(i) if cache is not None else None for i in range(len(keyframes))]
        if all(im is None for im in images):
            return

        try:
            estimate = estimate_focal_from_flow(
                images,
                [kf.pose.t if kf.pose is not None else None for kf in keyframes],
                [
                    kf.telemetry.geo.alt_rel
                    if kf.telemetry is not None and kf.telemetry.geo is not None
                    else None
                    for kf in keyframes
                ],
                [kf.telemetry.gimbal_pitch if kf.telemetry is not None else None for kf in keyframes],
                native_width=int(base.width),
            )
        except Exception:
            logger.warning("focal from flow: measurement failed; keeping the intrinsics prior", exc_info=True)
            return
        if estimate is None:
            return

        factor = estimate.fx / float(base.fx)
        if not (_MIN_FOCAL_REFINE_FACTOR <= factor <= _MAX_FOCAL_REFINE_FACTOR):
            logger.warning(
                "focal from flow: measured %.0f px against a %.0f px prior (%.2fx), outside "
                "[%.1f, %.1f] -- keeping the prior rather than trusting a measurement that far out.",
                estimate.fx,
                base.fx,
                factor,
                _MIN_FOCAL_REFINE_FACTOR,
                _MAX_FOCAL_REFINE_FACTOR,
            )
            return
        if estimate.spread_fraction > _MAX_FOCAL_SPREAD_FRACTION:
            logger.warning(
                "focal from flow: measured %.0f px but the pair-to-pair spread is %.0f%% of it, "
                "which means the flat-ground/constant-altitude assumption is not holding on this "
                "flight -- keeping the prior.",
                estimate.fx,
                100.0 * estimate.spread_fraction,
            )
            return

        for kf in keyframes:
            existing = kf.intrinsics or state.intrinsics
            if existing is not None:
                kf.intrinsics = _rescale_focal(existing, factor)
        if state.intrinsics is not None:
            state.intrinsics = _rescale_focal(state.intrinsics, factor)
        state.intrinsics_provenance = "measured_from_flow"
        state.focal_estimate = estimate.as_dict()
        logger.info(
            "focal from flow: intrinsics updated %.0f px -> %.0f px (%.3fx). Every downstream "
            "depth, distance and projection now uses the measured lens instead of the HFOV guess.",
            base.fx,
            estimate.fx,
            factor,
        )

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)
        keyframes = state.keyframes
        if not keyframes:
            raise StageUnavailable("pose_prior: no keyframes")

        yaw_diag = _refine_yaw_from_flow(state, keyframes, cfg_geometry=state.config.geometry)
        state.yaw_refinement = yaw_diag

        # Measure focal length before anything is triangulated with it.
        # Only when the intrinsics are a GUESS: a real calibration or EXIF
        # focal is a measurement and must not be overwritten by an
        # estimate. See geometry.focal_from_flow for why bundle adjustment
        # cannot recover this (the points are triangulated with the seed,
        # so the seed is already a minimum).
        self._measure_focal(state, keyframes)

        poses = [kf.pose for kf in keyframes]
        missing = sum(1 for p in poses if p is None)
        if missing:
            raise StageUnavailable(
                f"pose_prior: {missing}/{len(poses)} keyframes have no telemetry pose to refine; "
                "geometry will run on whatever conditioning exists"
            )

        # BundleAdjustmentStage refines whatever is in state.poses. Before
        # geometry that is the conditioned telemetry track; afterwards
        # GeometryStage overwrites it with the merged poses, so lending it
        # here does not leak into later stages.
        state.poses = list(poses)
        # Force the underlying solve: this stage IS the bundle adjustment,
        # so the skip-if-already-refined guard below must not fire here.
        previous_flag = state.config.matching.redundant_ba_after_geometry
        state.config.matching.redundant_ba_after_geometry = True
        try:
            artifacts, message = BundleAdjustmentStage().run(state, cancel_token, progress_cb, partial_cb)
        finally:
            # Restore what the CALLER had, not a hardcoded default: a user
            # who set this true in YAML would otherwise have it silently
            # switched off by the stage that borrowed it.
            state.config.matching.redundant_ba_after_geometry = previous_flag

        refined = state.poses
        if len(refined) != len(keyframes):
            raise StageUnavailable("pose_prior: bundle adjustment returned a pose count that does not match the keyframes")

        shift = float(np.median([np.linalg.norm(r.t - p.t) for r, p in zip(refined, poses, strict=True)]))
        rot = float(np.median([np.degrees(np.arccos(np.clip((np.trace(r.R.T @ p.R) - 1) / 2, -1, 1))) for r, p in zip(refined, poses, strict=True)]))
        for kf, pose in zip(keyframes, refined, strict=True):
            kf.pose = pose
        state.pose_prior_refined = True

        artifacts = {
            **artifacts,
            "median_position_shift_m": round(shift, 3),
            "median_rotation_change_deg": round(rot, 3),
            "yaw_from_flow": {k: v for k, v in (yaw_diag or {}).items() if k not in ("per_keyframe", "pairs")},
        }
        # For the report card (ExportStage cannot read StageResults; see
        # its docstring) -- the telemetry-alignment evidence belongs next
        # to the accuracy figures it underpins.
        state.pose_prior_summary = {
            "tracks_triangulated": artifacts.get("tracks_triangulated"),
            "ba_points": artifacts.get("ba_points"),
            "rmse_after_px": artifacts.get("rmse_after_px"),
            "median_position_shift_m": artifacts["median_position_shift_m"],
            "median_rotation_change_deg": artifacts["median_rotation_change_deg"],
            "yaw_from_flow_keyframes": (yaw_diag or {}).get("from_flow"),
            "yaw_flow_minus_telemetry_median_deg": (yaw_diag or {}).get("median_delta_deg"),
        }
        return artifacts, f"pose prior refined: {message}; poses moved median {shift:.2f} m / {rot:.2f} deg"


def _placement_dir(state) -> Path:
    """Where the placement check writes. Beside ``progress/``, under ``--out``.

    ``config.export.output_dir`` is anchored to ``<out>/output`` by the
    runner's own ``--out`` handling, so its parent is the run directory in
    both the CLI and the GUI. Falls back to the output directory itself
    when that assumption does not hold, which is still somewhere the
    operator will look.
    """
    output_dir = Path(getattr(state.config.export, "output_dir", "output"))
    parent = output_dir.parent
    return (parent / "placement") if parent != Path("") else (output_dir / "placement")



class DronePathStage(PipelineStage):
    """Step 1: where the drone started and how it moved. Telemetry only.

    First of the cheap preview stages that gate the expensive ones (see
    ``export.flightview``'s module docstring). Runs before semantics,
    before any pose refinement and long before the depth backbone,
    because a flight whose track is wrong cannot produce a good model no
    matter what runs after it -- and that is visible in seconds from
    telemetry alone.

    Purely diagnostic: it never blocks the run.
    """

    name = "drone_path"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)
        keyframes = state.keyframes
        if not keyframes:
            return {}, "no keyframes to trace"
        if progress_cb:
            progress_cb(0, 1, "tracing the drone path")

        from drishti3d.export.flightview import write_path_preview

        gps = _gps_enu_by_keyframe(keyframes)
        if not gps:
            return {}, "no keyframe carries a GPS position; cannot trace the path"
        order = sorted(gps)
        positions = np.array([gps[i] for i in order], dtype=np.float64)
        timestamps = [float(keyframes[i].timestamp) for i in order if i < len(keyframes)]

        metrics = write_path_preview(
            _preview_dir(state), positions, timestamps=timestamps if len(timestamps) == len(order) else None
        )
        state.drone_path = metrics
        if progress_cb:
            progress_cb(1, 1, "drone path written")
        if not metrics.get("applied"):
            return metrics, f"path not traced: {metrics.get('reason')}"

        message = (
            f"{metrics['n_points']} cameras, {metrics['path_length_m']:.0f} m flown over "
            f"{metrics['extent_m']['x']:.0f}x{metrics['extent_m']['y']:.0f} m at "
            f"{metrics['altitude_m']['median']:.0f} m; spacing median {metrics['spacing_m']['median']:.1f} m"
        )
        ratio = metrics.get("spacing_ratio")
        if ratio and ratio > 2.0:
            message += f" [WARNING: spacing varies {ratio:.1f}x across the flight]"
        return metrics, message


class CoverageStage(PipelineStage):
    """Step 2: which ground the camera actually saw. Still no depth.

    Projects each frame's field of view onto the telemetry's ground plane
    and rasterises how many frames see each patch. Answers the two
    questions that decide whether a reconstruction is even possible --
    is there a hole, and is any ground seen only once -- before the
    backbone spends minutes on the answer.
    """

    name = "coverage"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)
        keyframes = state.keyframes
        if not keyframes or state.intrinsics is None:
            return {}, "no keyframes or no intrinsics; cannot project a field of view"
        if progress_cb:
            progress_cb(0, 1, "projecting camera coverage")

        from drishti3d.export.flightview import write_coverage_preview
        from drishti3d.export.placement import frame_footprints, ground_z_from_telemetry

        poses = [kf.pose for kf in keyframes]
        placed = [p for p in poses if p is not None]
        if not placed:
            return {}, "no keyframe has a pose yet"

        cams = np.array([p.t for p in placed], dtype=np.float64)
        altitudes = [
            kf.telemetry.geo.alt_rel if kf.telemetry is not None and kf.telemetry.geo is not None else None
            for kf in keyframes
            if kf.pose is not None
        ]
        ground_z = ground_z_from_telemetry(cams, altitudes)
        if ground_z is None:
            return {}, "no telemetry altitude; cannot place a ground plane to project onto"

        footprints = frame_footprints(poses, [state.intrinsics] * len(poses), ground_z)
        metrics = write_coverage_preview(_preview_dir(state), footprints, camera_positions=cams)
        state.coverage = metrics
        if progress_cb:
            progress_cb(1, 1, "coverage written")
        if not metrics.get("applied"):
            return metrics, f"coverage not computed: {metrics.get('reason')}"

        message = (
            f"{metrics['area_covered_ha']:.2f} ha seen; typical patch by "
            f"{metrics['overlap_median']:.0f} frames; {metrics['seen_once_pct']:.1f}% seen once"
        )
        if metrics.get("hole_cells"):
            message += f" [WARNING: {metrics['hole_area_ha']:.2f} ha unseen inside the survey]"
        return metrics, message


class FramePlacementStage(PipelineStage):
    """Place every frame on the ground and render it, BEFORE any geometry runs.

    Runs between ``PosePriorStage`` and ``GeometryStage``, and needs no
    depth: a camera pose, the intrinsics and a ground height fully
    determine which patch of ground a frame sees. See
    ``export.placement``'s "BEFORE geometry" section.

    The point is cost. ``GeometryStage`` spends minutes running the dense
    backbone over every window, and a pose, altitude or coverage problem
    is baked into all of them by the time the first surface exists. This
    stage answers -- in seconds, from data that already exists -- whether
    the frames even land where the flight says they should, so a bad run
    can be stopped at the cheap end rather than the expensive one.

    It never blocks the pipeline: a diagnostic in front of the costly
    stage that could itself abort a reconstruction would be worse than no
    diagnostic at all.
    """

    name = "frame_placement"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)
        keyframes = state.keyframes
        if not keyframes:
            return {}, "no keyframes to place"

        if progress_cb:
            progress_cb(0, 2, "placing frames on the ground plane")

        from drishti3d.export.placement import ground_z_from_telemetry, write_frame_placement

        poses = [kf.pose for kf in keyframes]
        placed = [p for p in poses if p is not None]
        if not placed:
            return {}, "no keyframe has a pose yet; nothing to place"

        cams = np.array([p.t for p in placed], dtype=np.float64)
        altitudes = [
            kf.telemetry.geo.alt_rel if kf.telemetry is not None and kf.telemetry.geo is not None else None
            for kf in keyframes
            if kf.pose is not None
        ]
        ground_z = ground_z_from_telemetry(cams, altitudes)

        # The backbone runs at cfg.max_image_size, but a footprint is a
        # property of the real camera, so this uses the ingest intrinsics
        # unscaled. Resolution cancels out of the ray direction anyway --
        # (u - cx)/fx is unchanged by a uniform resize -- so the choice
        # only matters if it is made inconsistently.
        intrinsics = [state.intrinsics] * len(poses) if state.intrinsics is not None else []

        gps = _gps_enu_by_keyframe(keyframes)
        gps_track = (
            np.array([gps[i] for i in sorted(gps) if i < len(poses) and poses[i] is not None], dtype=np.float64)
            if gps
            else None
        )
        if gps_track is not None and len(gps_track) != len(cams):
            # Only pass a GPS track that lines up one-to-one with the
            # placed cameras; a partial track would silently compare
            # frame i against frame j.
            gps_track = None

        if progress_cb:
            progress_cb(1, 2, "rendering frame placement")

        metrics = write_frame_placement(
            _placement_dir(state), poses, intrinsics, ground_z=ground_z, gps_positions=gps_track
        )
        state.frame_placement = metrics

        if progress_cb:
            progress_cb(2, 2, "frame placement written")

        if not metrics.get("applied"):
            return metrics, f"frames not placed: {metrics.get('reason')}"

        message = (
            f"{metrics['frames_placed']}/{metrics['frames_total']} frames placed on ground "
            f"z={metrics.get('ground_z_m')} m; typical cell seen by {metrics.get('overlap_median')} frames; "
            f"{metrics.get('consecutive_overlap_pct')}% consecutive overlap"
        )
        if metrics.get("pose_vs_gps_median_m") is not None:
            message += f"; poses vs GPS median {metrics['pose_vs_gps_median_m']} m"
        return metrics, message


#: Median per-window collinearity at or above which the merge's rotation
#: fit is treated as unreliable. Uses the same 1 - s1/s0 measure
#: geometry.flight_profile applies to the whole track, so the two numbers
#: are directly comparable; the threshold matches its
#: _COLLINEARITY_LINEAR_THRESHOLD for the same reason.
_WINDOW_COLLINEAR_THRESHOLD = 0.85


def _median_window_collinearity(windows, camera_gps_enu) -> float | None:
    """How straight a typical window's camera track is: 0 spread out, 1 a line.

    The merge fits one transform per window, so this -- not the flight's
    overall shape -- is what decides whether that fit can recover
    rotation. A grid flight is made of straight passes, and a window is a
    piece of one pass.
    """
    scores = []
    for window in windows:
        pts = [camera_gps_enu[i] for i in window.keyframe_indices() if i in camera_gps_enu]
        if len(pts) < 3:
            # Too few cameras to even measure spread; such a window is
            # the worst case for a rotation fit, so count it as a line.
            scores.append(1.0)
            continue
        arr = np.asarray(pts, dtype=np.float64)
        singular = np.linalg.svd(arr - arr.mean(axis=0), compute_uv=False)
        s0 = float(singular[0])
        s1 = float(singular[1]) if singular.shape[0] > 1 else 0.0
        scores.append(float(np.clip(1.0 - (s1 / s0 if s0 > 1e-9 else 0.0), 0.0, 1.0)))
    return float(np.median(scores)) if scores else None


def _merge_scale_per_submap(submaps, poses) -> dict[int, float]:
    """How much the merge rescaled each submap, measured from its own cameras.

    ``merge_submaps`` fits a Sim(3) per submap whose scale comes from the
    camera centres -- local baselines onto GPS baselines -- and then
    applies that scale to the POINTS as well. So whatever correction
    ``depth_anchor`` made to a window's ground depth gets multiplied by
    this factor afterwards.

    That matters because the anchor and the merge derive their scales
    from different evidence: the anchor from how deep the ground is, the
    merge from how far the camera moved. When the backbone returns a
    window whose depth and baseline are inconsistent with each other,
    those two disagree, and the ground ends up at
    ``anchor_ratio * merge_scale`` times its true depth -- a window can
    be anchored perfectly and still land tens of metres out.

    Computed here by comparing each submap's local camera baselines
    against the same cameras' merged positions, which needs no change to
    ``merge_submaps`` and is exact: a similarity scales every baseline by
    the same factor.
    """
    scales: dict[int, float] = {}
    for submap in submaps:
        index = getattr(getattr(submap, "window", None), "index", None)
        if index is None or len(submap.poses) < 2:
            continue
        local, merged = [], []
        kfs = list(submap.keyframe_indices)
        for a, b in zip(range(len(kfs) - 1), range(1, len(kfs)), strict=False):
            if kfs[a] >= len(poses) or kfs[b] >= len(poses):
                continue
            local.append(np.linalg.norm(np.asarray(submap.poses[b].t) - np.asarray(submap.poses[a].t)))
            merged.append(np.linalg.norm(np.asarray(poses[kfs[b]].t) - np.asarray(poses[kfs[a]].t)))
        local_arr, merged_arr = np.asarray(local), np.asarray(merged)
        usable = local_arr > 1e-6
        if usable.sum() >= 1:
            scales[int(index)] = float(np.median(merged_arr[usable] / local_arr[usable]))
    return scales


def _preview_dir(state) -> Path:
    """Where the ordered stage previews are written: ``<out>/preview``."""
    output_dir = Path(getattr(state.config.export, "output_dir", "output"))
    parent = output_dir.parent
    return (parent / "preview") if parent != Path("") else (output_dir / "preview")


def _run_placement_check(state, point_cloud, submap_labels, keyframes, submaps=None, poses=None):
    """Render and measure where the windows landed. Never raises.

    Runs between the merge and fusion. A placement that is metres wrong
    cannot be rescued by any amount of TSDF work downstream, and a
    top-down preview cannot show that it is wrong -- see
    ``export.placement``'s module docstring.
    """
    if point_cloud is None or submap_labels is None:
        return None
    try:
        from drishti3d.export.placement import ground_z_from_telemetry, write_placement_check

        cams = np.array([kf.pose.t for kf in keyframes if kf.pose is not None], dtype=np.float64)
        altitudes = [
            kf.telemetry.geo.alt_rel if kf.telemetry is not None and kf.telemetry.geo is not None else None
            for kf in keyframes
            if kf.pose is not None
        ]
        gps_ground_z = ground_z_from_telemetry(cams, altitudes) if len(cams) else None

        report = write_placement_check(
            _placement_dir(state),
            point_cloud.xyz,
            submap_labels,
            gps_ground_z=gps_ground_z,
            camera_positions=cams if len(cams) else None,
        )

        # Attribute the error. A window's ground depth is off by the
        # product of two independent corrections -- the anchor's (from
        # how deep the ground looked) and the merge's (from how far the
        # camera moved) -- so reporting only the final offset cannot say
        # which one to fix. These put all three numbers side by side.
        if submaps and poses:
            merge_scales = _merge_scale_per_submap(submaps, poses)
            anchors = {
                int(i): d.get("window_ratio")
                for i, d in (getattr(state, "depth_anchor_diags", {}) or {}).items()
                if isinstance(d, dict) and d.get("applied")
            }
            report.metrics["merge_scale_by_submap"] = {str(k): round(v, 4) for k, v in sorted(merge_scales.items())}
            report.metrics["anchor_x_merge_by_submap"] = {
                str(k): round(float(anchors[k]) * v, 4)
                for k, v in sorted(merge_scales.items())
                if anchors.get(k)
            }
            if merge_scales:
                values = np.asarray(list(merge_scales.values()))
                report.metrics["merge_scale_spread"] = round(
                    float(values.max() / max(values.min(), 1e-9)), 3
                )
                logger.info(
                    "placement: the merge rescaled submaps by %.3f-%.3f (median %.3f). This "
                    "multiplies whatever the depth anchor did, because the anchor sets ground "
                    "depth from the ground and the merge sets scale from the camera baselines.",
                    float(values.min()),
                    float(values.max()),
                    float(np.median(values)),
                )

        if not report.passed:
            logger.warning(
                "PLACEMENT CHECK FAILED -- the windows do not agree on where the ground is. "
                "Fusion will now mesh several copies of the same surface, and the result will "
                "look noisy and thick no matter what the fusion settings are. The defect is in "
                "the merge, not the mesh; inspect the elevation panels before changing anything "
                "downstream."
            )
        return report
    except Exception:
        logger.warning("geometry: placement check failed; continuing to fusion", exc_info=True)
        return None


class GeometryStage(PipelineStage):
    """Plan windows, run the backbone per window, build Submaps, merge them.

    Always produces *some* output: if the configured backbone
    (``GeometryConfig.backbone``, or the runner's ``backbone`` override) is
    unavailable, this falls back to ``"null"`` (``NullBackbone``, see its
    module docstring) rather than failing the stage.
    """

    name = "geometry"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)

        keyframes = state.keyframes
        if not keyframes:
            state.windows = []
            state.submaps = []
            state.point_cloud = PointCloud(xyz=np.zeros((0, 3), dtype=np.float64))
            state.poses = []
            artifacts = {"windows": 0, "submaps": 0, "backbone": state.backbone_name, "points": 0}
            return artifacts, "no keyframes; produced an empty point cloud"

        # Independent-anchor submap merging (see geometry.submap's module
        # docstring, "Chained alignment compounds error"): every submap
        # with enough of its own GPS/telemetry coverage gets aligned
        # directly onto GPS instead of chaining onto the previous submap's
        # already-noisy globalized positions, so per-junction estimation
        # noise can no longer compound down the whole submap sequence.
        # Empty (not None) when there's no telemetry at all -- merge_submaps
        # treats an empty dict exactly like "no camera_gps_enu given" (falls
        # back to plain chaining), so this is safe to always compute and
        # pass through unconditionally.
        # Yaw from the footage, not the gimbal log -- see
        # geometry.yaw_from_flow for why the log cannot be trusted on this
        # kind of flight. Done before anything reads kf.pose: window
        # conditioning, conditioned_R, and the parallax anchor all inherit
        # the corrected rotation from here.
        if getattr(state, "yaw_refinement", None) is None:
            yaw_diag = _refine_yaw_from_flow(state, keyframes, cfg_geometry=state.config.geometry)
            state.yaw_refinement = yaw_diag

        # Submaps are anchored to the best camera track available. When
        # PosePriorStage ran, that is the bundle-adjusted track (GPS priors
        # fused with the images, ~metres better than raw GPS); otherwise
        # the raw GPS/ENU positions as before.
        if getattr(state, "pose_prior_refined", False):
            camera_gps_enu = {i: kf.pose.t for i, kf in enumerate(keyframes) if kf.pose is not None}
            logger.info("geometry: anchoring submaps to the bundle-adjusted pose prior (%d cameras)", len(camera_gps_enu))
        else:
            camera_gps_enu = _gps_enu_by_keyframe(keyframes)
        # The rotation each keyframe was conditioned with (see
        # _poses_from_telemetry, which sets Keyframe.pose) -- the
        # "telemetry_rotation" strategy (geometry.submap
        # ._telemetry_rotation_transform) uses this as its trusted world
        # orientation, in preference to ever estimating rotation from a
        # submap's own (possibly near-collinear) camera centres.
        conditioned_R = {i: kf.pose.R for i, kf in enumerate(keyframes) if kf.pose is not None}

        # Telemetry height per keyframe, passed into every merge below.
        #
        # This was previously supplied only by FusionStage, which re-merges
        # the same submaps itself -- so the two merges ran on different
        # inputs. ``merge_submaps`` uses these for two things: the
        # per-submap scale anchor, and (per geometry.overlap_align) the
        # GPS vertical levelling that is "the merge's only absolute
        # vertical reference". Without them this merge had no absolute
        # vertical reference at all, while the one that actually produced
        # the mesh did.
        #
        # That also made the placement check measure a merge that was not
        # the merge being meshed, which is worse than not checking: a gate
        # reporting on the wrong artefact sends every subsequent diagnosis
        # after a phantom.
        keyframe_altitude_m = {
            i: float(kf.telemetry.geo.alt_rel)
            for i, kf in enumerate(keyframes)
            if kf.telemetry is not None
            and kf.telemetry.geo is not None
            and kf.telemetry.geo.alt_rel is not None
            and float(kf.telemetry.geo.alt_rel) > 0.0
        }
        if keyframe_altitude_m:
            logger.info(
                "geometry: %d/%d keyframes carry a telemetry altitude for merge scale anchoring "
                "and GPS vertical levelling",
                len(keyframe_altitude_m),
                len(keyframes),
            )
        else:
            logger.warning(
                "geometry: no keyframe carries a telemetry altitude, so this merge has no absolute "
                "vertical reference -- submap heights will be set by the camera-centre scale fit, "
                "which measures horizontal extent to correct a vertical error"
            )

        # Task 1: classify the capture (gimbal mode, trajectory shape,
        # altitude) from the same telemetry/poses above, and log it
        # prominently -- an operator should be able to see, at a glance,
        # what the system concluded about their flight and which merge
        # strategy that implies, not have to go digging through junction
        # diagnostics after the fact.
        flight_profile: FlightProfile = analyze_flight_profile(keyframes)
        logger.info("=" * 78)
        logger.info("FLIGHT PROFILE: %s", flight_profile.summary())
        for note in flight_profile.notes:
            logger.info("FLIGHT PROFILE note: %s", note)
        logger.info("=" * 78)

        # Task 2: the flight profile's recommendation IS the merge strategy
        # -- geometry.submap.merge_submaps/alignment_residuals/
        # strategy_report take it verbatim as their `strategy` argument.
        # Per-submap fallback within a strategy is handled (and logged)
        # inside geometry.submap itself; what's checked below is whether
        # the *requested* strategy was actually usable at all for this
        # flight's telemetry coverage.
        merge_strategy = flight_profile.recommended_merge_strategy
        merge_strategy_reason = (
            flight_profile.notes[-1] if flight_profile.notes else f"trajectory/orientation support {merge_strategy!r}"
        )

        cfg = state.config.geometry
        # Task 3: derive window_size from the memory budget and keyframe
        # count (geometry.windows.plan_window_size) rather than treating
        # GeometryConfig.window_size as a hard constant -- see that
        # function's docstring. geometry.windows.plan_windows's own overlap
        # floor (2 shared keyframes) is still looser than
        # geometry.submap.merge_submaps' real minimum (3 -- Umeyama's
        # Sim(3) fit needs >= 3 non-collinear correspondences to be
        # well-posed at all), so _MIN_VIABLE_WINDOW_SIZE below is a second,
        # stricter floor bridging that gap.
        if getattr(cfg, "single_inference", False):
            # One backbone call over as many keyframes as will fit. See
            # GeometryConfig.single_inference for the measurement: window
            # boundaries, not the backbone, are what wrecks the surface.
            ceiling = getattr(cfg, "max_views_per_inference", None) or len(keyframes)
            effective_window_size = max(_MIN_VIABLE_WINDOW_SIZE, min(len(keyframes), int(ceiling)))
            planned_window_size = effective_window_size
            logger.info(
                "geometry: single-inference mode -- %d keyframes in %d view(s) per call, so the "
                "reconstruction has %s to merge",
                len(keyframes),
                effective_window_size,
                "nothing" if effective_window_size >= len(keyframes) else "as few boundaries as possible",
            )
        else:
            planned_window_size = plan_window_size(
                len(keyframes), cfg.max_image_size, vram_budget_gb=_window_memory_budget_gb(), requested=cfg.window_size
            )
            effective_window_size = max(planned_window_size, _MIN_VIABLE_WINDOW_SIZE)
        if effective_window_size != cfg.window_size:
            logger.info(
                "geometry: window_size planned at %d (requested %d) for %d keyframes at %dpx "
                "(memory-budget/correspondence-floor adjusted)",
                effective_window_size,
                cfg.window_size,
                len(keyframes),
                cfg.max_image_size,
            )
        overlap = min(max(_MERGE_MIN_OVERLAP, round(0.35 * effective_window_size)), effective_window_size - 1)
        windows = plan_windows(
            keyframes,
            window_size=effective_window_size,
            overlap=overlap,
            max_extent_m=getattr(cfg, "max_window_extent_m", None),
        )
        state.windows = windows
        logger.info(
            "geometry: %d window(s) planned for %d keyframes (window_size=%d, overlap=%d) -> %d junction(s)",
            len(windows),
            len(keyframes),
            effective_window_size,
            overlap,
            max(0, len(windows) - 1),
        )

        # The merge strategy has to be chosen for the geometry the merge
        # ACTUALLY fits, which is one window's cameras -- not the whole
        # flight's.
        #
        # analyze_flight_profile measures collinearity over the entire
        # track. On this footage that is 0.2275 (a grid), so it selects
        # "gps_anchored", which solves each submap's rotation from that
        # submap's own camera centres. But a window is a short, straight
        # segment of the flight: with the extent cap in force a window
        # holds 3-6 cameras in very nearly a line, and rotation ABOUT that
        # line is unconstrained no matter how grid-like the whole flight
        # is. Measured: 376 degenerate fits, and one submap thrown 251 m
        # from where GPS puts it.
        #
        # "telemetry_rotation" does not fit rotation at all. It takes each
        # camera's known world rotation (nadir gimbal, +/-0.0 deg, yaw from
        # image flow) and composes it with the backbone's own camera
        # rotation, then fits only scale and translation from the camera
        # centres -- and those two ARE well constrained by a straight
        # track. So when the windows are collinear and telemetry
        # orientation is trustworthy, it is strictly better here.
        window_collinearity = _median_window_collinearity(windows, camera_gps_enu)
        if (
            window_collinearity is not None
            and window_collinearity >= _WINDOW_COLLINEAR_THRESHOLD
            and flight_profile.orientation_is_reliable
            and conditioned_R
            and merge_strategy != "telemetry_rotation"
        ):
            logger.warning(
                "geometry: overriding merge strategy %r -> 'telemetry_rotation'. The flight as a "
                "whole is not collinear (%.3f), but the individual windows the merge actually fits "
                "are (median %.3f over %d windows), and rotation about a straight camera track "
                "cannot be recovered from the camera centres. Telemetry orientation is reliable "
                "here, so rotation is taken from it rather than estimated.",
                merge_strategy,
                flight_profile.collinearity if flight_profile.collinearity is not None else float("nan"),
                window_collinearity,
                len(windows),
            )
            merge_strategy_reason = (
                f"windows are collinear (median {window_collinearity:.3f}) even though the flight "
                f"({flight_profile.collinearity:.3f}) is not; rotation taken from reliable telemetry "
                "instead of fitted from a straight camera track"
            )
            merge_strategy = "telemetry_rotation"

        backbone_name = state.backbone_name
        try:
            # Fix 2: forward cfg.max_image_size through to the backbone's
            # own constructor -- see get_backbone's docstring for why this
            # matters (MapAnythingBackbone used to always be built with its
            # own hardcoded 518px default, silently overriding whatever
            # GeometryConfig.max_image_size said). Backbones that don't
            # accept this kwarg (e.g. NullBackbone) just ignore it.
            backbone = get_backbone(
                backbone_name,
                max_image_size=cfg.max_image_size,
                mask_edges=cfg.mapanything_mask_edges,
                multiview_confidence=cfg.mapanything_multiview_confidence,
                confidence_percentile=cfg.mapanything_confidence_percentile,
            )
            if not backbone.is_available():
                raise RuntimeError(f"backbone {backbone_name!r} reports unavailable")
        except Exception as exc:  # noqa: BLE001 - any backbone-selection problem falls back to null
            if backbone_name != "null":
                logger.warning("geometry: backbone %r unavailable (%s); falling back to 'null'", backbone_name, exc)
            backbone_name = "null"
            backbone = get_backbone("null")

        # One backbone per available GPU. `available_devices` returns a
        # single entry for MPS/CPU and for a one-GPU box, so this collapses
        # to exactly the old single-device behaviour on every machine that
        # is not genuinely multi-GPU -- including the 6 GB RTX 4060 target.
        devices = available_devices(max_devices=getattr(cfg, "max_devices", 0))
        n_workers = max(1, min(len(devices), len(windows)))
        devices = devices[:n_workers]

        backbones = [backbone]
        backbone.load(device=devices[0])
        # Replicas are constructed only when there is a second device AND a
        # second window to give it -- loading a 4.6 GB checkpoint onto a GPU
        # that will never be handed work is pure latency.
        for device in devices[1:]:
            replica = get_backbone(
                backbone_name,
                max_image_size=cfg.max_image_size,
                mask_edges=cfg.mapanything_mask_edges,
                multiview_confidence=cfg.mapanything_multiview_confidence,
                confidence_percentile=cfg.mapanything_confidence_percentile,
            )
            replica.load(device=device)
            backbones.append(replica)

        if len(devices) > 1:
            logger.info(
                "geometry: running %d windows across %d devices (%s), one window per device at a time",
                len(windows),
                len(devices),
                ", ".join(devices),
            )

        submaps: list[Submap] = []
        state.submaps = submaps  # same list object: appends below are visible on state immediately
        # Per-window parallax anchoring diagnostics (geometry.depth_anchor).
        # Created here, before any worker starts, so window threads only
        # ever assign keys into an existing dict.
        state.depth_anchor_diags = {}

        # Counted across every window so the report card can state, as one
        # number, how much of the raw backbone output was discarded as
        # dynamic content. Windows overlap, so this exceeds the number of
        # distinct points removed from the final cloud -- it is a measure
        # of backbone output filtered, not of cloud points deleted.
        dynamic_points_removed = 0

        # `state.video` wraps a single PyAV container with its own seek
        # position. Two threads decoding through it at once interleave seeks
        # and return each other's frames, so every decode -- sequential or
        # parallel -- goes through this lock. Serialising decode costs
        # nothing in the parallel case anyway: decode is CPU/IO-bound and
        # the thing being overlapped is GPU inference.
        decode_lock = threading.Lock()

        def on_window_done(index: int, submap: Submap, removed: int) -> None:
            """Collector, always called on the main thread."""
            nonlocal dynamic_points_removed
            dynamic_points_removed += removed
            if partial_cb is not None:
                try:
                    partial_pc, _ = merge_submaps(
                        submaps,
                        camera_gps_enu=camera_gps_enu,
                        conditioned_R=conditioned_R,
                        strategy=merge_strategy,
                        keyframe_altitude_m=keyframe_altitude_m,
                    )
                    partial_cb(partial_pc)
                except Exception:
                    logger.debug("geometry: partial merge failed; continuing without a preview update", exc_info=True)

        def on_cancel() -> None:
            if submaps:
                state.point_cloud, state.poses = merge_submaps(
                    submaps,
                    camera_gps_enu=camera_gps_enu,
                    conditioned_R=conditioned_R,
                    strategy=merge_strategy,
                    keyframe_altitude_m=keyframe_altitude_m,
                )
            raise PipelineCancelled()

        try:
            window_errors = (
                _run_windows(
                    state=state,
                    windows=windows,
                    keyframes=keyframes,
                    cfg=cfg,
                    backbones=backbones,
                    submaps=submaps,
                    decode_lock=decode_lock,
                    cancel_token=cancel_token,
                    progress_cb=progress_cb,
                    on_window_done=on_window_done,
                    on_cancel=on_cancel,
                    devices=devices,
                )
                or {}
            )

            # Reconstructing NOTHING is a stage failure, not a stage that
            # happened to produce zero output. Reporting "ok" here sends
            # every downstream stage into its own "nothing to do" skip, and
            # buries the real cause in a log line the operator never sees.
            if windows and not submaps:
                first = next(iter(window_errors.values()), "no error recorded")
                raise RuntimeError(
                    f"geometry reconstructed 0/{len(windows)} windows -- every window failed. "
                    f"First failure:\n{first}"
                )
            if window_errors:
                logger.warning(
                    "geometry: %d/%d windows failed but %d succeeded; the reconstruction is incomplete",
                    len(window_errors),
                    len(windows),
                    len(submaps),
                )
        finally:
            # Unload each replica while pinned to ITS device: empty_cache()
            # on the wrong current device frees the wrong pool, and is the
            # call that surfaces any earlier cross-device error.
            for b, dev in zip(backbones, devices, strict=False):
                try:
                    with _device_context(dev):
                        b.unload()
                except Exception:
                    logger.warning("geometry: unloading backbone on %s failed", dev, exc_info=True)

        # SCALE CONSENSUS. Every window measured its own depth scale
        # independently, and on real footage those measurements have come
        # out a factor of 5.9 apart across one flight. Windows at
        # different scales reconstruct the same ground at different
        # heights, and the merge below -- which fits one Sim(3) per
        # submap onto the CAMERAS -- cannot fix that, so it has to be
        # resolved first. See geometry.scale_consensus.
        state.scale_consensus = {}
        if submaps and getattr(cfg, "scale_consensus", True):
            try:
                from drishti3d.geometry.scale_consensus import harmonise_submap_scales

                state.scale_consensus = harmonise_submap_scales(
                    submaps, getattr(state, "depth_anchor_diags", {}) or {}
                )
            except Exception:
                logger.warning("geometry: scale consensus failed; merging windows at their own scales", exc_info=True)

        submap_labels: np.ndarray | None = None
        try:
            if submaps:
                point_cloud, poses, submap_labels = merge_submaps(
                    submaps,
                    camera_gps_enu=camera_gps_enu,
                    conditioned_R=conditioned_R,
                    strategy=merge_strategy,
                    keyframe_altitude_m=keyframe_altitude_m,
                    return_labels=True,
                )
            else:
                point_cloud, poses = PointCloud(xyz=np.zeros((0, 3), dtype=np.float64)), []
        except Exception:
            logger.warning("geometry: merge_submaps failed; falling back to un-merged submap points", exc_info=True)
            xyz = np.concatenate([sm.points.xyz.reshape(-1, 3) for sm in submaps], axis=0)
            conf = np.concatenate([np.asarray(sm.confidence).reshape(-1) for sm in submaps], axis=0)
            point_cloud = PointCloud(xyz=xyz, confidence=conf)
            poses = [p for sm in submaps for p in sm.poses]

        state.point_cloud = point_cloud
        state.poses = poses

        # PLACEMENT CHECK. Before a single voxel is fused, render and
        # measure where the windows actually landed. See export.placement
        # for why this is its own step: fusion turns a placement error and
        # a meshing error into the same bad-looking mesh, forty minutes
        # later, and every top-down preview this pipeline writes is blind
        # to vertical error by construction.
        state.placement_report = _run_placement_check(
            state, point_cloud, submap_labels, keyframes, submaps=submaps, poses=poses
        )

        # merge_submaps deduplicates near-exact points that overlapping
        # windows reconstruct once per window they share a keyframe in
        # (see geometry.submap.merge_submaps' docstring) -- report how many
        # that was, rather than leaving the inflated pre-dedup count as the
        # only number anyone sees.
        raw_points = sum(sm.points.xyz.reshape(-1, 3).shape[0] for sm in submaps)
        deduplicated = max(0, raw_points - int(point_cloud.xyz.shape[0]))

        # Global consistency check (Task 2's "fail loudly" requirement):
        # compare the just-merged camera positions against the same GPS
        # track that (when available) anchored the merge above, and warn
        # loudly if they disagree by more than a generous threshold. This
        # is deliberately generous (pre-BA, pre-georeference: a real,
        # if bounded, amount of raw backbone/merge error is expected here)
        # -- its job is to catch a merge that is still badly broken (e.g.
        # no telemetry to anchor on, or an insufficiently-covered submap
        # that had to fall back to chaining), not to demand BA-grade
        # accuracy this early in the pipeline.
        merge_gps_rmse_m: float | None = None
        if camera_gps_enu and poses and len(poses) == len(keyframes):
            gps_idx = [i for i in camera_gps_enu if i < len(poses)]
            if gps_idx:
                pred = np.array([poses[i].t for i in gps_idx])
                gps = np.array([camera_gps_enu[i] for i in gps_idx])
                merge_gps_rmse_m = float(np.sqrt(np.mean(np.sum((pred - gps) ** 2, axis=1))))
                if merge_gps_rmse_m > _GEOMETRY_GPS_RMSE_WARN_M:
                    logger.warning(
                        "GEOMETRY/GPS CONSISTENCY WARNING: merged camera positions disagree with "
                        "GPS by RMSE=%.1f m over %d keyframes (warn threshold %.0f m). This usually "
                        "means one or more submaps fell back to chained (not GPS-anchored) "
                        "alignment -- see geometry.submap's degenerate-alignment / chained-drift "
                        "discussion -- or that telemetry coverage is too sparse to anchor every "
                        "submap independently. Downstream matching/bundle-adjustment reprojection "
                        "error is likely to be affected.",
                        merge_gps_rmse_m,
                        len(gps_idx),
                        _GEOMETRY_GPS_RMSE_WARN_M,
                    )
        state.geometry_gps_rmse_m = merge_gps_rmse_m

        # Task 2's other "fail loudly" requirement: report which strategy
        # was actually usable, not just which one was requested.
        # strategy_report inspects every submap's own independent-anchor
        # attempt (geometry.submap already logged a warning per-submap
        # fallback as it happened; this is the aggregate view for the
        # StageResult/accuracy report card).
        strategy_diag = strategy_report(submaps, camera_gps_enu=camera_gps_enu, conditioned_R=conditioned_R, strategy=merge_strategy)
        strategy_warnings: list[str] = []
        if not strategy_diag["fully_used_requested_strategy"]:
            warning = (
                f"MERGE STRATEGY WARNING: requested strategy {merge_strategy!r} could only be applied "
                f"to {strategy_diag['n_anchored']}/{strategy_diag['n_submaps']} submaps; "
                f"{strategy_diag['n_fallback_to_chaining']} fell back to chained Sim(3) alignment "
                "(see per-submap warnings above for why each one couldn't be independently anchored)."
            )
            logger.warning(warning)
            strategy_warnings.append(warning)
        for anchored, diag in strategy_diag["submap_diagnostics"]:
            if anchored and merge_strategy == "telemetry_rotation" and diag is not None:
                spread = diag.get("rotation_spread_deg")
                if spread is not None and spread > _ROTATION_SPREAD_WARN_DEG:
                    warning = (
                        f"MERGE STRATEGY WARNING: telemetry_rotation's rotation samples disagreed by "
                        f"{spread:.1f} deg on average (n={diag.get('n_rotation_samples')}) -- telemetry "
                        "conditioning may not have been consistently honoured for this submap."
                    )
                    logger.warning(warning)
                    strategy_warnings.append(warning)
            # gps_anchored's own per-submap fit (_gps_full_anchor_transform)
            # solves rotation from that submap's camera centres -- unlike
            # telemetry_rotation, it CAN be degenerate (e.g. flight_profile's
            # collinearity threshold was borderline, or one submap's own
            # GPS-tagged keyframes happened to be more collinear than the
            # flight as a whole). Check it here explicitly: this is the one
            # strategy whose own anchor fit -- not just the informational
            # chained cross-check -- can be degenerate, and nothing else in
            # this loop catches that.
            if anchored and merge_strategy == "gps_anchored" and diag is not None and diag.get("degenerate"):
                warning = (
                    f"MERGE STRATEGY WARNING: gps_anchored's own Sim(3) fit for a submap is degenerate "
                    f"(condition_number={diag.get('condition_number')}) -- that submap's own GPS-tagged "
                    "camera track turned out more collinear than the flight-wide classification expected."
                )
                logger.warning(warning)
                strategy_warnings.append(warning)
        # A junction's own ``degenerate`` flag reflects the (near-)collinear
        # shared-camera Umeyama fit -- the transform ``merge_submaps``
        # actually uses when strategy="chained_sim3", but only an
        # *informational* cross-check (see _chain_align's docstring) when
        # a submap was independently anchored (telemetry_rotation /
        # gps_anchored) -- exactly the case those two strategies exist to
        # route around, so flagging it there would warn about the very
        # degeneracy the selected strategy was chosen to avoid. Only
        # chained_sim3's degenerate flags are actionable here.
        if merge_strategy == "chained_sim3":
            for junction in strategy_diag["junction_residuals"]:
                if junction.get("degenerate"):
                    warning = (
                        f"MERGE STRATEGY WARNING: junction {junction.get('junction')} is degenerate "
                        f"(condition_number={junction.get('condition_number')}) under strategy {merge_strategy!r}."
                    )
                    logger.warning(warning)
                    strategy_warnings.append(warning)
        if merge_gps_rmse_m is not None and merge_gps_rmse_m > _GEOMETRY_GPS_RMSE_WARN_M:
            strategy_warnings.append(
                f"merged poses vs GPS RMSE={merge_gps_rmse_m:.2f} m exceeds the {_GEOMETRY_GPS_RMSE_WARN_M:.0f} m "
                f"consistency threshold under strategy {merge_strategy!r}."
            )

        # Stashed (ad-hoc PipelineState attributes, same pattern
        # BundleAdjustmentStage/FusionStage already use -- see their
        # docstrings) so ExportStage's accuracy report card reflects the
        # strategy that actually ran, instead of recomputing junction
        # diagnostics from scratch with no strategy/telemetry context.
        state.geometry_flight_profile = flight_profile
        state.geometry_merge_strategy = merge_strategy
        state.geometry_merge_strategy_reason = merge_strategy_reason
        state.geometry_merge_strategy_warnings = strategy_warnings
        state.geometry_junction_residuals = strategy_diag["junction_residuals"]

        # Depth-anchoring summary for the report card: which windows were
        # rescaled, by how much, and which refused. The per-window ratio is
        # the backbone's measured depth error on this footage and belongs
        # next to the accuracy figures, not buried in a log.
        anchor_diags = getattr(state, "depth_anchor_diags", {}) or {}
        applied = [d for d in anchor_diags.values() if d.get("applied")]
        ratios = [d["window_ratio"] for d in applied]
        depth_anchor_summary = {
            "enabled": bool(getattr(cfg, "depth_anchor", False)),
            "windows_total": len(windows),
            "windows_anchored": len(applied),
            "windows_refused": [
                {"window": int(i), "reason": d.get("failure", "unknown")}
                for i, d in sorted(anchor_diags.items())
                if not d.get("applied")
            ],
            "ratio_median": float(np.median(ratios)) if ratios else None,
            "ratio_min": float(min(ratios)) if ratios else None,
            "ratio_max": float(max(ratios)) if ratios else None,
            "per_window": [
                {
                    "window": int(i),
                    "ratio": d.get("window_ratio"),
                    "samples": d.get("samples_total"),
                    "masked_fraction": d.get("masked_fraction_mean"),
                    # The ratio's two inputs, so a reader can see whether a
                    # window's correction differs because the drone was at a
                    # different height or because the backbone returned a
                    # different scale for the same scene.
                    "altitude_m": d.get("altitude_m"),
                    "backbone_ground_depth_m": d.get("backbone_ground_depth_m"),
                }
                for i, d in sorted(anchor_diags.items())
            ],
        }
        state.depth_anchor_summary = depth_anchor_summary
        if applied:
            logger.info(
                "geometry: depth anchoring rescaled %d/%d windows; ratio median %.3f (min %.3f, max %.3f)",
                len(applied),
                len(windows),
                depth_anchor_summary["ratio_median"],
                depth_anchor_summary["ratio_min"],
                depth_anchor_summary["ratio_max"],
            )
        # FusionStage re-merges state.submaps itself (fusion.tsdf.fuse_submaps
        # needs the per-submap structure, not this stage's already-flattened
        # state.point_cloud) -- stashed so that re-merge uses the same
        # strategy/GPS/telemetry context this one did, instead of silently
        # falling back to plain chaining and re-introducing the exact tilt
        # this stage just fixed. See fuse_submaps' own docstring.
        state.geometry_camera_gps_enu = camera_gps_enu
        state.geometry_conditioned_R = conditioned_R

        artifacts = {
            "windows": len(windows),
            "submaps": len(submaps),
            "backbone": backbone_name,
            "points": int(point_cloud.xyz.shape[0]),
            "raw_points_before_dedup": raw_points,
            "deduplicated_points": deduplicated,
            "merge_gps_rmse_m": merge_gps_rmse_m,
            "flight_profile": vars(flight_profile),
            "merge_strategy": merge_strategy,
            "merge_strategy_reason": merge_strategy_reason,
            "merge_strategy_submaps_anchored": strategy_diag["n_anchored"],
            "merge_strategy_submaps_fallback": strategy_diag["n_fallback_to_chaining"],
            "merge_strategy_warnings": strategy_warnings,
            "dynamic_points_removed": dynamic_points_removed,
            "windows_failed": len(window_errors),
            "depth_anchor": depth_anchor_summary,
            "scale_consensus": getattr(state, "scale_consensus", None),
            "placement": (
                state.placement_report.as_dict() if getattr(state, "placement_report", None) is not None else None
            ),
            "yaw_from_flow": {
                k: v for k, v in (getattr(state, "yaw_refinement", {}) or {}).items() if k not in ("per_keyframe", "pairs")
            },
            "window_errors": {str(k): v for k, v in window_errors.items()},
        }
        state.geometry_dynamic_points_removed = dynamic_points_removed
        message = (
            f"{flight_profile.summary()}; "
            f"{len(submaps)}/{len(windows)} submaps reconstructed via {backbone_name!r} using "
            f"strategy={merge_strategy!r} ({merge_strategy_reason}); "
            f"{point_cloud.xyz.shape[0]} points ({deduplicated} duplicate points removed from overlapping windows)"
        )
        if merge_gps_rmse_m is not None:
            message += f"; merged poses vs GPS RMSE={merge_gps_rmse_m:.2f} m"
            if merge_gps_rmse_m > _GEOMETRY_GPS_RMSE_WARN_M:
                message += " [WARNING: exceeds consistency threshold, see log]"
        if strategy_warnings:
            message += f" [WARNING: {len(strategy_warnings)} merge-strategy issue(s), see log]"
        placement = getattr(state, "placement_report", None)
        if placement is not None:
            message += f"; placement {placement.summary()}"
        return artifacts, message


# ---------------------------------------------------------------------------
# Optional late stages -- owned by other, concurrently-in-progress
# workstreams. See module docstring for the degrade-gracefully contract.
# ---------------------------------------------------------------------------


class MatchingStage(PipelineStage):
    """Feature matching / correspondence stage: detect -> match -> verify -> tracks -> triangulate -> ``BAProblem``.

    This is the link ``BundleAdjustmentStage`` (below) used to be missing:
    ``bundle.bundle_adjust`` needs a full 2D-observation model
    (``BAProblem.obs_camera_idx`` / ``obs_point_idx`` / ``obs_uv``), and
    ``GeometryStage`` does not retain per-pixel correspondence once a
    window's dense backbone output has been folded into a ``Submap`` and
    merged into ``state.point_cloud`` (see ``geometry.submap
    .merge_submaps``'s module docstring -- by design it only tracks a flat
    point cloud + poses). This stage builds that correspondence structure
    from scratch, using ``state.keyframes``' images and ``state.poses``'
    (GeometryStage's merged, backbone+GPS-conditioned) rough poses as the
    camera geometry triangulation needs:

        ``geometry.features.detect_and_describe`` -> ``geometry.features
        .match_features`` -> ``geometry.features.geometric_verify`` ->
        ``geometry.tracks.build_tracks`` -> ``geometry.tracks
        .filter_tracks`` -> ``geometry.triangulate.triangulate_tracks`` ->
        ``geometry.triangulate.filter_by_reprojection`` -> ``geometry
        .triangulate.build_ba_problem``.

    Degrades gracefully at every stage that can legitimately come up
    empty on real (noisy, low-texture, or badly-overlapping) footage:
    too few verified pairs, too few surviving tracks, or too few
    tracks with a usable triangulation angle (see ``geometry.triangulate``'s
    module docstring on why small-angle triangulations are rejected, not
    just flagged) all raise ``StageUnavailable`` with a diagnostic message
    rather than handing a half-built or numerically-thin problem to bundle
    adjustment.

    Not registered directly in ``pipeline.runner``'s stage list --
    ``runner.py`` is outside this workstream's file ownership (see the
    module docstring's top-level note on which stages this workstream
    owns). ``BundleAdjustmentStage`` instead runs this as its own first
    step, so the runner's results table still shows a single
    ``"bundle_adjustment"`` entry; wiring this in as its own numbered
    stage later (so its timing/progress shows up separately) is then a
    one-line addition to ``runner._build_stages``, not a rewrite. This
    class is written as a fully independent, directly testable
    ``PipelineStage`` in the meantime (see ``tests/test_tracks.py``'s
    end-to-end matching -> triangulation -> bundle-adjustment chain test,
    which exercises the same building blocks called here).
    """

    name = "matching"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)

        keyframes = state.keyframes
        poses = state.poses
        if not keyframes or not poses or len(poses) != len(keyframes):
            raise StageUnavailable(
                f"matching: need one pose per keyframe from GeometryStage ({len(keyframes)} keyframes, "
                f"{len(poses)} poses available); skipping"
            )
        if state.video is None:
            raise StageUnavailable("matching: no opened video to read keyframe images from")

        frame_indices = [kf.frame_index for kf in keyframes]
        # Detection and matching depend only on the IMAGES, never on the
        # poses -- so when MatchingStage runs a second time (PosePriorStage
        # before geometry, BundleAdjustmentStage after it) the whole
        # detect + match + verify + track-build phase is identical work.
        # Measured: pose_prior 575-737 s and bundle_adjustment 404-457 s,
        # most of it duplicated. Only triangulation depends on poses, and
        # that is re-run every time against whatever poses now exist.
        cache_key = tuple(kf.frame_index for kf in keyframes)
        cache = getattr(state, "_matching_cache", None)
        if cache is not None and cache.get("key") == cache_key:
            logger.info(
                "matching: reusing %d tracks detected in the earlier pass (same keyframes); re-triangulating",
                len(cache["trackset"]),
            )
            # Re-derive intrinsics from the keyframes as they are NOW,
            # at the resolution the cached images are stored at.
            reuse_intrinsics = [kf.intrinsics or state.intrinsics for kf in keyframes]
            reuse_cache = getattr(state, "keyframe_cache", None)
            if reuse_cache is not None and len(reuse_cache) == len(keyframes):
                from drishti3d.geometry.mapanything import scale_intrinsics

                reuse_intrinsics = [
                    scale_intrinsics(intr, reuse_cache.scale_for(i)) if intr is not None else None
                    for i, intr in enumerate(reuse_intrinsics)
                ]
            return self._triangulate_and_build(
                state, cancel_token, progress_cb, keyframes, poses, reuse_intrinsics,
                cache["trackset"], cache["stats"], len(keyframes), cache["pairs"], cache["verified_pairs"],
            )

        cached = getattr(state, "keyframe_cache", None)
        if cached is not None and len(cached) == len(keyframes):
            images = [cached.get(i) for i in range(len(keyframes))]
            images = [im for im in images if im is not None]
            logger.info("matching: using %d cached keyframe images (no re-decode)", len(images))
        else:
            frames = state.video.read_frames(frame_indices)
            images = [f.image for f in frames if f.image is not None]
        if len(images) != len(keyframes):
            raise StageUnavailable(
                f"matching: could not decode all keyframe images ({len(images)}/{len(keyframes)}); skipping"
            )

        intrinsics_list = [kf.intrinsics or state.intrinsics for kf in keyframes]
        # The cache stores downscaled frames; keypoints are therefore in
        # that frame's pixels, so the intrinsics must be scaled by the SAME
        # factor the image was -- not re-derived from the rounded size.
        if cached is not None and len(cached) == len(keyframes):
            from drishti3d.geometry.mapanything import scale_intrinsics

            intrinsics_list = [
                scale_intrinsics(intr, cached.scale_for(i)) if intr is not None else None
                for i, intr in enumerate(intrinsics_list)
            ]
        if any(k is None for k in intrinsics_list):
            raise StageUnavailable("matching: missing camera intrinsics for one or more keyframes; skipping")

        n = len(keyframes)
        total_steps = n + 1  # one unit per frame's detection, plus one block for matching/tracking/triangulation
        grays = []
        for i, img in enumerate(images):
            _check_cancel(cancel_token)
            if progress_cb:
                progress_cb(i, total_steps, f"detecting features {i + 1}/{n}")
            grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        features_per_frame = [
            features_mod.detect_and_describe(
                gray,
                method=_match_cfg(state, "method", _MATCH_METHOD),
                max_features=int(_match_cfg(state, "max_features", _MATCH_MAX_FEATURES)),
                detect_scale=float(_match_cfg(state, "detect_scale", 1.0)),
            )
            for gray in grays
        ]

        pairs = tracks_mod.select_pairs(
            keyframes, strategy="sequential+loop", window=_MATCH_WINDOW, gps_radius_m=_MATCH_GPS_RADIUS_M
        )

        verified_pairs: list[tuple[int, int, features_mod.Matches]] = []
        for k, (i, j) in enumerate(pairs):
            _check_cancel(cancel_token)
            if progress_cb:
                progress_cb(n, total_steps, f"matching pair {k + 1}/{len(pairs)}")
            raw = features_mod.match_features(features_per_frame[i], features_per_frame[j], ratio=_MATCH_RATIO)
            if len(raw) < _MIN_VERIFIED_INLIERS:
                continue
            verified = features_mod.geometric_verify(
                features_per_frame[i], features_per_frame[j], raw, intrinsics=intrinsics_list[i], method="essential"
            )
            if verified.inlier_mask is not None and int(verified.inlier_mask.sum()) >= _MIN_VERIFIED_INLIERS:
                verified_pairs.append((i, j, verified))

        if not verified_pairs:
            raise StageUnavailable("matching: no keyframe pair passed geometric verification; skipping")

        trackset = tracks_mod.build_tracks(images, features_per_frame, verified_pairs)
        trackset = tracks_mod.filter_tracks(trackset, min_length=_MIN_TRACK_LENGTH)
        stats = tracks_mod.track_statistics(trackset, n_frames=n)

        if stats["count"] < _MIN_TRACKS_FOR_BA:
            raise StageUnavailable(
                f"matching: only {stats['count']} tracks survived filtering (need >= {_MIN_TRACKS_FOR_BA}); "
                "skipping bundle adjustment -- geometry's poses/point cloud carry through unchanged"
            )

        state._matching_cache = {
            "key": cache_key,
            "trackset": trackset,
            "stats": stats,
            # Intrinsics are deliberately NOT cached. Focal-length
            # refinement rewrites Keyframe.intrinsics between the two
            # MatchingStage calls, so a cached copy is stale by the second
            # one. Tracks depend only on the images; intrinsics are re-read
            # from the keyframes on reuse.
            "pairs": len(pairs),
            "verified_pairs": len(verified_pairs),
        }
        return self._triangulate_and_build(
            state, cancel_token, progress_cb, keyframes, poses, intrinsics_list,
            trackset, stats, n, len(pairs), len(verified_pairs),
        )

    def _triangulate_and_build(
        self, state, cancel_token, progress_cb, keyframes, poses, intrinsics_list,
        trackset, stats, n, n_pairs, n_verified,
    ):
        """The pose-dependent half: triangulate, gate, build the BAProblem.

        Split from ``run`` so a second pass can reuse the cached tracks
        (which do not depend on poses) while re-triangulating against
        whatever poses it now has.
        """
        from drishti3d.geometry import tracks as tracks_mod
        from drishti3d.geometry import triangulate as triangulate_mod

        total_steps = n + 1
        _check_cancel(cancel_token)
        points3d, valid_mask, _angles = triangulate_mod.triangulate_tracks(
            trackset, poses, intrinsics_list, min_angle_deg=_MIN_TRIANGULATION_ANGLE_DEG
        )
        kept_tracks = tracks_mod.TrackSet(
            tracks=[t for t, keep in zip(trackset.tracks, valid_mask, strict=True) if keep]
        )
        kept_points = points3d[valid_mask]

        if len(kept_tracks) < _MIN_TRACKS_FOR_BA:
            raise StageUnavailable(
                f"matching: only {len(kept_tracks)} tracks triangulated with a usable angle "
                f"(>= {_MIN_TRIANGULATION_ANGLE_DEG} deg; need >= {_MIN_TRACKS_FOR_BA}); skipping bundle adjustment"
            )

        pre_filter_count = len(kept_tracks)
        # Everything pass 2 needs to re-triangulate from scratch with the
        # refined poses: the angle-filtered trackset and the intrinsics.
        state.matching_trackset = kept_tracks
        state.matching_intrinsics = intrinsics_list
        # Cap the points handed to bundle adjustment. The deliverable here
        # is the POSES; the dense surface comes from the backbone. Solve
        # time scales with point count, and 2,000 well-spread points
        # constrain 77 cameras as well as 7,700 do -- measured: the solver
        # was the whole of pose_prior's remaining cost once decoding was
        # cached. `None` (the "accurate" profile) keeps every track.
        filtered_points, filtered_tracks = triangulate_mod.filter_by_reprojection(
            kept_points,
            kept_tracks,
            poses,
            intrinsics_list,
            max_px=_MAX_REPROJECTION_PX_PREBA,
            max_points=_match_cfg(state, "max_points_in_ba", None),
        )
        # filter_by_reprojection populates Track.reprojection_error_px on
        # every track of its *input* (kept_tracks, still in scope here)
        # before filtering -- see that function's docstring -- so this is
        # the real, measured error distribution the 15px cut just acted
        # on, not a guess. "Fail loudly, not silently" (see this project's
        # brief): a StageUnavailable here used to just say how many tracks
        # were left; it now says which filter removed them, how many, and
        # what the actual reprojection errors looked like, so a badly
        # broken upstream geometry (e.g. GeometryStage's
        # merge_gps_rmse_m warning firing) is diagnosable from this
        # message alone instead of a bare "skipping bundle adjustment."
        pre_filter_errors = np.array([t.reprojection_error_px for t in kept_tracks.tracks], dtype=np.float64)
        kept_points, kept_tracks = filtered_points, filtered_tracks
        if len(kept_tracks) < _MIN_TRACKS_FOR_BA:
            finite = np.isfinite(pre_filter_errors)
            if finite.any():
                fe = pre_filter_errors[finite]
                error_summary = (
                    f"reprojection error over those {pre_filter_count} tracks (before the "
                    f"{_MAX_REPROJECTION_PX_PREBA:.0f}px cut): median={np.median(fe):.0f}px, "
                    f"p90={np.percentile(fe, 90):.0f}px, max={fe.max():.0f}px "
                    f"({int((~finite).sum())} landed behind a camera entirely). Errors this large "
                    "mean the cameras' relative geometry is broken (e.g. submaps merged with "
                    "inconsistent poses), not that the 15px threshold is merely a bit tight -- "
                    "see geometry.submap's chained-Sim3-drift discussion and "
                    "GeometryStage's merge_gps_rmse_m artifact."
                )
            else:
                error_summary = f"all {pre_filter_count} candidate tracks landed behind a camera (infinite reprojection error)"
            raise StageUnavailable(
                f"matching: the {_MAX_REPROJECTION_PX_PREBA:.0f}px reprojection filter removed "
                f"{pre_filter_count - len(kept_tracks)}/{pre_filter_count} triangulated tracks, "
                f"leaving only {len(kept_tracks)} (need >= {_MIN_TRACKS_FOR_BA}); skipping bundle "
                f"adjustment. {error_summary}"
            )

        problem = triangulate_mod.build_ba_problem(kept_tracks, poses, intrinsics_list, kept_points)

        # Handed to BundleAdjustmentStage. Deliberately not a declared
        # ``PipelineState`` field: this workstream's file ownership covers
        # this stage (and BundleAdjustmentStage) but not PipelineState's
        # own definition -- see this class's docstring. Plain instance
        # attributes on a non-``__slots__`` dataclass work fine for this.
        state.matching_problem = problem
        state.matching_stats = stats

        if progress_cb:
            progress_cb(total_steps, total_steps, f"{len(kept_tracks)} tracks triangulated")

        artifacts = {
            "keyframes": n,
            "pairs_attempted": n_pairs,
            "pairs_verified": n_verified,
            "tracks_built": stats["count"],
            "tracks_triangulated": len(kept_tracks),
            "mean_track_length": stats["mean_length"],
        }
        message = (
            f"{n_pairs} pairs attempted, {n_verified} verified; {stats['count']} tracks built, "
            f"{len(kept_tracks)} triangulated with a usable angle"
        )
        return artifacts, message


def _rescale_focal(intr: CameraIntrinsics, factor: float) -> CameraIntrinsics:
    """Correct focal length only. NOT ``scale_intrinsics``.

    ``geometry.mapanything.scale_intrinsics`` exists for RESIZING an image:
    it scales width, height and the principal point along with the focal
    length, because all of those move together when pixels are resampled.
    Correcting a wrong focal-length *estimate* is a different operation --
    the sensor is still the same size and the optical axis still hits the
    same pixel; only the assumed focal length was wrong. Using the resize
    helper here turned a 3840 px frame into a 6029 px one (caught by
    tests/test_pipeline.py::test_focal_refinement_is_applied_as_a_ratio_at_native_resolution).
    """
    return CameraIntrinsics(
        fx=intr.fx * factor,
        fy=intr.fy * factor,
        cx=intr.cx,
        cy=intr.cy,
        width=intr.width,
        height=intr.height,
        dist_coeffs=intr.dist_coeffs,
    )


def _camera_priors_from_telemetry(keyframes: list[Keyframe]) -> list:
    """Build ``bundle.CameraPrior`` entries from each keyframe's own flight telemetry.

    Two independent priors, attached per keyframe only when their own
    source data is present -- see ``geometry.bundle``'s module docstring
    for why both matter for a single-pass flight strip:

    - **Gravity/tilt** (from ``TelemetrySample.gimbal_pitch``/
      ``gimbal_roll``): a direct IMU/gimbal attitude reading, always
      expressed relative to the local horizon independent of the
      reconstruction's own coordinate frame -- safe to attach whenever a
      keyframe has one, with no dependency on GPS/world-frame alignment.
    - **GPS position** (from ``TelemetrySample.geo``): converted to a
      local ENU frame anchored at the first GPS-tagged keyframe
      (``ingest.telemetry.telemetry_to_enu``), matching ``types``'
      documented world-frame convention. This is only a physically
      *accurate* position prior to the extent the reconstruction's own
      world frame is already GPS-aligned -- true once a real,
      GPS/IMU-conditioned backbone is wired up to honour the poses it's
      conditioned with, not true of the zero-weight ``NullBackbone``
      fallback (see its module docstring: it always synthesizes its own
      arbitrary flight line). Attaching it regardless is still safe --
      it is one more soft, several-metre-sigma constraint among several,
      never a hard one -- but its value depends on that upstream
      alignment actually existing.
    """
    from drishti3d.geometry.bundle import CameraPrior

    enu_by_idx = _gps_enu_by_keyframe(keyframes)

    priors = []
    for i, kf in enumerate(keyframes):
        telemetry = kf.telemetry
        gps_position = enu_by_idx.get(i)
        gimbal_pitch = telemetry.gimbal_pitch if telemetry is not None else None
        gimbal_roll = telemetry.gimbal_roll if telemetry is not None else None
        accuracy_h = telemetry.geo.accuracy_h if telemetry is not None and telemetry.geo is not None else None

        if gps_position is None and gimbal_pitch is None:
            continue

        priors.append(
            CameraPrior(
                camera_idx=i,
                gps_position=gps_position,
                gps_sigma_m=accuracy_h,
                gimbal_pitch_deg=gimbal_pitch,
                gimbal_roll_deg=gimbal_roll,
            )
        )

    return priors


class BundleAdjustmentStage(PipelineStage):
    """Refine ``state.poses`` by running ``drishti3d.geometry.bundle.bundle_adjust`` over real 2D observations.

    Runs ``MatchingStage`` (above) as its own first step to build the
    ``bundle.BAProblem`` bundle adjustment needs, attaches GPS/gravity
    priors from each keyframe's own telemetry (``_camera_priors_from_telemetry``),
    then calls ``bundle.bundle_adjust``. If matching cannot produce a
    usable problem (see ``MatchingStage``'s degrade-gracefully conditions),
    this stage itself raises ``StageUnavailable`` and ``state.poses``/
    ``state.point_cloud`` carry through from ``GeometryStage`` unchanged --
    a skipped bundle-adjustment stage is not a failed run.

    Only ``state.poses`` is overwritten with the refined result (matching
    ``BAProblem.cameras``' indexing 1:1). ``state.point_cloud`` -- the
    dense, per-pixel merged cloud from ``GeometryStage`` -- is a different,
    much larger point set than ``BAProblem.points`` (the sparse triangulated
    tracks this stage builds) and is deliberately left untouched here;
    re-densifying against refined poses is a fusion-stage-shaped problem,
    not this one's.
    """

    name = "bundle_adjustment"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)

        try:
            from drishti3d.geometry import bundle
        except ImportError as exc:
            raise StageUnavailable(f"drishti3d.geometry.bundle not available yet: {exc}") from exc

        if not hasattr(bundle, "bundle_adjust"):
            raise StageUnavailable("drishti3d.geometry.bundle has no bundle_adjust entry point yet")

        # Skip when the pose prior already did this work.
        #
        # PosePriorStage runs matching + bundle adjustment BEFORE geometry,
        # and GeometryStage then anchors its submap merge on those refined
        # poses -- so by the time this stage runs, state.poses is already
        # the bundle-adjusted track, merged. Re-solving it gains almost
        # nothing and is not cheap: measured on the sample flight, the
        # pose prior converged to 0.63 px in 168 s and this second pass
        # then ran 66 minutes in the sparse solver without finishing.
        #
        # Set `redundant_ba_after_geometry: true` to run it anyway (useful
        # when geometry's merge is suspected of having moved the cameras).
        if getattr(state, "pose_prior_refined", False) and not getattr(
            state.config.matching, "redundant_ba_after_geometry", False
        ):
            prior = getattr(state, "pose_prior_summary", {}) or {}
            raise StageUnavailable(
                "bundle_adjustment: skipped -- PosePriorStage already refined these poses before geometry "
                f"(reprojection {prior.get('rmse_after_px', 'n/a')} px over {prior.get('ba_points', 'n/a')} points) "
                "and geometry anchored its merge on them; re-solving would re-derive the same answer. "
                "Set matching.redundant_ba_after_geometry=true to force it."
            )

        matching_stage = MatchingStage()
        match_artifacts, match_message = matching_stage.run(state, cancel_token, progress_cb, partial_cb)

        problem = getattr(state, "matching_problem", None)
        if problem is None:
            raise StageUnavailable("matching produced no BAProblem; skipping bundle adjustment")

        _check_cancel(cancel_token)

        problem.camera_priors = _camera_priors_from_telemetry(state.keyframes)

        # Pass 1: loose-gated tracks, telemetry rotations. Its job is to
        # fix the rotations, not to produce the final answer.
        # Refine focal length when the intrinsics are a GUESS rather than a
        # measurement. Drone video carries no intrinsics, so ingest falls
        # back to a generic 84-degree-HFOV prior: 2132 px on this footage,
        # where COLMAP refined the same frames to 3355 px -- 1.57x out.
        # That error is not cosmetic. GSD goes 3.58 -> 5.63 cm/px, every
        # horizontal distance scales by 1.57, and because parallax depth
        # is fx * B / disparity, the depth anchor then rescales the whole
        # backbone to match a wrong reference.
        #
        # The classic danger here (BAConfig.refine_intrinsics' own warning)
        # is that focal length and depth trade off against each other when
        # a camera is refined from its own observations alone. GPS position
        # priors on every camera break exactly that ambiguity, which is why
        # this is enabled in the pose prior -- where those priors exist and
        # there are thousands of multi-view tracks -- and nowhere else.
        # Only refine when GPS POSITION priors exist -- not merely when a
        # prior object exists. _camera_priors_from_telemetry emits a prior
        # carrying only gimbal attitude when GPS is absent, and attitude
        # does not pin depth, so it does not break the focal/depth
        # ambiguity that BAConfig.refine_intrinsics warns about.
        n_gps_priors = sum(1 for cp in problem.camera_priors if getattr(cp, "gps_position", None) is not None)
        refine = bool(_match_cfg(state, "refine_intrinsics", False)) and n_gps_priors >= len(problem.cameras) // 2
        ba_config = bundle.BAConfig(
            max_iterations=int(_match_cfg(state, "ba_max_iterations", 100)),
            refine_intrinsics=refine,
        )
        fx_before = float(problem.intrinsics[0].fx) if problem.intrinsics else None
        logger.info(
            "bundle adjustment: refine_intrinsics=%s (config=%s, %d/%d cameras carry a GPS position "
            "prior, fx_before=%s)",
            refine,
            _match_cfg(state, "refine_intrinsics", False),
            n_gps_priors,
            len(problem.cameras),
            f"{fx_before:.1f}" if fx_before else None,
        )
        result = bundle.bundle_adjust(problem, ba_config)
        pass1 = {
            "ba_pass1_points": int(problem.points.shape[0]),
            "ba_pass1_rmse_before_px": result.rmse_before_px,
            "ba_pass1_rmse_after_px": result.rmse_after_px,
            "ba_pass1_converged": result.converged,
        }

        # Pass 2: re-triangulate every angle-filtered track against the
        # refined poses, apply the strict gate, and solve again. With the
        # rotations corrected, tracks that failed the strict gate for
        # geometric reasons (not because they were wrong) now pass it, so
        # the second solve sees the full track set instead of the ~1% that
        # survived gating against unrefined telemetry.
        trackset = getattr(state, "matching_trackset", None)
        intrinsics_list = getattr(state, "matching_intrinsics", None)
        pass2: dict = {}
        if trackset is not None and intrinsics_list is not None:
            from drishti3d.geometry import triangulate as triangulate_mod

            _check_cancel(cancel_token)
            pts, valid, _ = triangulate_mod.triangulate_tracks(
                trackset, result.poses, intrinsics_list, min_angle_deg=_MIN_TRIANGULATION_ANGLE_DEG
            )
            from drishti3d.geometry.tracks import TrackSet

            kept = TrackSet(tracks=[t for t, k in zip(trackset.tracks, valid, strict=True) if k])
            pts2, kept2 = triangulate_mod.filter_by_reprojection(
                pts[valid],
                kept,
                result.poses,
                intrinsics_list,
                max_px=_MAX_REPROJECTION_PX,
                max_points=_match_cfg(state, "max_points_in_ba", None),
            )
            if len(kept2) >= _MIN_TRACKS_FOR_BA:
                problem2 = triangulate_mod.build_ba_problem(kept2, result.poses, intrinsics_list, pts2)
                problem2.camera_priors = problem.camera_priors
                result2 = bundle.bundle_adjust(problem2, ba_config)
                pass2 = {
                    "ba_pass2_tracks_retriangulated": int(valid.sum()),
                    "ba_pass2_points": int(problem2.points.shape[0]),
                    "ba_pass2_rmse_before_px": result2.rmse_before_px,
                    "ba_pass2_rmse_after_px": result2.rmse_after_px,
                    "ba_pass2_converged": result2.converged,
                }
                problem, result = problem2, result2
                state.matching_problem = problem2
            else:
                logger.warning(
                    "bundle adjustment: pass 2 kept only %d tracks under the %.0f px gate; keeping pass 1",
                    len(kept2),
                    _MAX_REPROJECTION_PX,
                )
                pass2 = {"ba_pass2_points": int(len(kept2)), "ba_pass2_skipped": True}

        if refine and result.intrinsics and fx_before:
            # Apply the refinement as a RATIO, never as an absolute value.
            # `problem.intrinsics` are whatever resolution matching ran at
            # -- with the keyframe cache that is 1920 px, half of native --
            # so result.intrinsics carry that width/height too. Writing
            # them onto native-resolution keyframes would halve fx. The
            # ratio is scale-invariant and is the actual finding.
            # Read the FREE cameras only, and take their median.
            #
            # Two cameras are gauge-fixed (bundle._build_layout skips them),
            # and _unpack_x returns a fixed camera's intrinsics UNCHANGED.
            # Reading result.intrinsics[0] therefore always reports the
            # seed value and makes refinement look like a no-op -- measured
            # 1.000x while the free cameras had in fact moved to 1.206x.
            #
            # Median across free cameras, not one of them: BA refines each
            # camera's intrinsics independently, but this is ONE physical
            # lens, so the per-camera spread (2418-2759 px in a controlled
            # test) is estimation noise around a single true value.
            free = [
                i for i in range(len(result.intrinsics)) if i not in problem.fixed_camera_indices
            ]
            if not free:
                logger.info("bundle adjustment: every camera is gauge-fixed; focal length not refined")
                factor = 1.0
            else:
                factor = float(np.median([result.intrinsics[i].fx for i in free])) / fx_before
            if factor == 1.0:
                pass  # nothing free to refine; already reported above
            elif not (_MIN_FOCAL_REFINE_FACTOR <= factor <= _MAX_FOCAL_REFINE_FACTOR):
                logger.warning(
                    "bundle adjustment: focal refinement returned %.2fx, outside [%.1f, %.1f] -- "
                    "rejecting it and keeping the intrinsics prior. A correction this large means the "
                    "solve went somewhere unphysical, not that the lens is that different.",
                    factor,
                    _MIN_FOCAL_REFINE_FACTOR,
                    _MAX_FOCAL_REFINE_FACTOR,
                )
            else:
                native_before = float((state.keyframes[0].intrinsics or state.intrinsics).fx)
                for kf in state.keyframes:
                    base = kf.intrinsics or state.intrinsics
                    if base is not None:
                        kf.intrinsics = _rescale_focal(base, factor)
                if state.intrinsics is not None:
                    state.intrinsics = _rescale_focal(state.intrinsics, factor)
                state.refined_focal_px = native_before * factor
                state.focal_refine_factor = factor
                spread = [result.intrinsics[i].fx for i in free]
                logger.info(
                    "bundle adjustment: focal length refined %.0f px -> %.0f px (%.3fx) at native "
                    "resolution. Median over %d free cameras (%.0f-%.0f px spread; %d gauge-fixed "
                    "cameras excluded), from %d tracks with %d GPS position priors. The prior was a "
                    "generic HFOV guess (provenance %r), not a measurement.",
                    native_before,
                    state.refined_focal_px,
                    factor,
                    len(free),
                    min(spread),
                    max(spread),
                    len(problem.fixed_camera_indices),
                    int(problem.points.shape[0]),
                    n_gps_priors,
                    getattr(state, "intrinsics_provenance", "unknown"),
                )

        state.poses = result.poses
        # The refined sparse points and which camera saw which: the
        # parallax truth fusion.reanchor rescales each dense view against.
        state.ba_points = np.asarray(result.points, dtype=np.float64)
        state.ba_obs_camera_idx = np.asarray(problem.obs_camera_idx)
        state.ba_obs_point_idx = np.asarray(problem.obs_point_idx)
        # Stashed for ExportStage's accuracy report card (see that class's
        # docstring on why it can't just read this stage's own returned
        # ``artifacts``/``StageResult`` -- the runner only assembles those
        # after every stage, including export, has already run). Same
        # ad-hoc-attribute pattern as ``state.matching_problem`` above.
        state.mean_reprojection_error_px = result.rmse_after_px

        artifacts = {
            **match_artifacts,
            "ba_cameras": len(problem.cameras),
            "ba_points": int(problem.points.shape[0]),
            "ba_observations": int(problem.obs_uv.shape[0]),
            "ba_camera_priors": len(problem.camera_priors),
            "rmse_before_px": result.rmse_before_px,
            "rmse_after_px": result.rmse_after_px,
            "converged": result.converged,
            **pass1,
            **pass2,
        }
        message = (
            f"{match_message}; bundle adjustment over {len(problem.cameras)} cameras / "
            f"{problem.points.shape[0]} points ({len(problem.camera_priors)} camera priors): "
            f"reprojection RMSE {result.rmse_before_px:.2f}px -> {result.rmse_after_px:.2f}px"
        )
        return artifacts, message


class FusionStage(PipelineStage):
    """Wrapper over ``drishti3d.fusion.tsdf.fuse_submaps`` (owned by another workstream).

    ``fuse_submaps`` re-merges ``state.submaps`` itself (it needs the
    per-submap structure, not our already-flattened ``state.point_cloud``),
    cleans the result (``fusion.filters``), and volumetrically fuses it via
    a confidence-weighted TSDF into a ``(vertices, faces, colors,
    confidence, raw_point_cloud)`` mesh + full-detail-fallback pair.
    ``PipelineResult`` has no dedicated mesh field (see ``pipeline.result``),
    so the mesh's vertices/colors/tiered confidence become
    ``state.point_cloud`` here -- face connectivity is not currently
    threaded through beyond this stage's own artifacts (see
    ``artifacts["n_faces"]``). ``raw_point_cloud`` (the merged, cleaned, but
    never coarsely voxel-downsampled dense cloud TSDF meshing itself
    consumed) is stashed separately as ``state.fusion_raw_point_cloud`` --
    see that attribute's assignment below -- so ``ExportStage`` can write
    it as ``point_cloud.ply``/``point_cloud.las`` independently of whatever
    the mesh above does or doesn't produce (meshing is lossy; TSDF voxel
    size is itself derived from ground sample distance, clamped against a
    voxel-count budget for large scenes -- see
    ``fusion.tsdf._derive_voxel_size`` -- so the raw cloud is the only
    deliverable guaranteed to carry full source detail regardless of scene
    size).

    ``fuse_submaps`` itself lazily imports ``drishti3d.fusion.mesh``
    (normal estimation), which may land after ``drishti3d.fusion.tsdf``
    does -- an ``ImportError`` raised *from inside* that call is therefore
    also treated as "not available yet", not a genuine failure.
    """

    name = "fusion"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)

        try:
            from drishti3d.fusion.tsdf import fuse_submaps
        except ImportError as exc:
            raise StageUnavailable(f"drishti3d.fusion.tsdf not available yet: {exc}") from exc

        if not state.submaps:
            raise StageUnavailable("fusion: no submaps to fuse (geometry stage produced none)")

        _check_cancel(cancel_token)

        # Native-resolution intrinsics per keyframe -- fuse_submaps needs
        # these to derive a real ground-sample-distance voxel size (Fix 1;
        # see fusion.tsdf._derive_voxel_size's docstring on why it must be
        # the *native* resolution, not the backbone's working-resolution
        # copy that _resize_for_backbone builds separately in GeometryStage
        # without ever touching Keyframe.intrinsics).
        keyframe_intrinsics = {i: kf.intrinsics for i, kf in enumerate(state.keyframes) if kf.intrinsics is not None}

        # Second anchoring pass, against bundle adjustment's sparse points
        # (see fusion.reanchor). The submaps are rescaled per view here,
        # before merging, so the merge sees geometry at BA-grade scale.
        submaps_for_fusion = state.submaps
        reanchor_diag: dict = {}
        if getattr(state.config.fusion, "ba_reanchor", True) and getattr(state, "ba_points", None) is not None:
            try:
                from drishti3d.fusion.reanchor import reanchor_submaps_to_ba

                submaps_for_fusion, reanchor_diag = reanchor_submaps_to_ba(
                    state.submaps,
                    state.ba_points,
                    state.ba_obs_camera_idx,
                    state.ba_obs_point_idx,
                    state.poses,
                )
            except Exception:
                logger.warning("fusion: BA re-anchoring failed; fusing the geometry-stage submaps", exc_info=True)
                submaps_for_fusion = state.submaps
        state.ba_reanchor_summary = dict(reanchor_diag)  # per_view included: ~150 small dicts, worth having in meta.json

        stats: dict = {}
        try:
            vertices, faces, colors, confidence, raw_point_cloud = fuse_submaps(
                submaps_for_fusion,
                state.config.fusion,
                stats=stats,
                camera_gps_enu=getattr(state, "geometry_camera_gps_enu", None),
                conditioned_R=getattr(state, "geometry_conditioned_R", None),
                strategy=getattr(state, "geometry_merge_strategy", "chained_sim3"),
                keyframe_intrinsics=keyframe_intrinsics,
                point_filter=self._pre_mesh_point_filter(state),
                # Always supplied now. Altitudes drive two different things:
                # the optional per-submap SCALE anchor (altitude_anchor,
                # off by default), and GPS vertical LEVELLING in
                # geometry.overlap_align, which is the merge's only
                # absolute vertical reference and must always have them.
                keyframe_altitude_m=self._keyframe_altitudes(state),
            )
        except ImportError as exc:
            raise StageUnavailable(f"drishti3d.fusion.tsdf.fuse_submaps not fully available yet: {exc}") from exc

        # Stashed for ExportStage's accuracy report card -- see
        # BundleAdjustmentStage.run's identical comment on why a plain
        # ad-hoc PipelineState attribute, not this stage's own returned
        # artifacts, is what a later stage in the same run can actually
        # read. Set even when fusion produced no vertices (below) so the
        # report card can still say which source *would* have been used.
        state.fusion_confidence_source = stats.get("confidence_source")

        # Fix 2: the raw, cleaned (pre-TSDF, never coarsely voxelised) dense
        # point cloud, stashed independently of whether meshing below
        # succeeded -- ExportStage writes this as point_cloud.ply/.las
        # regardless of mesh outcome (see fuse_submaps' own docstring: this
        # is the "meshing is lossy" full-detail fallback that matches what
        # the reference dense-point-cloud pipeline this fix was compared
        # against produces).
        state.fusion_raw_point_cloud = raw_point_cloud

        # Lift the 2D masks onto both clouds. Done here, not in
        # SemanticsStage, because this is the first point at which final
        # world-frame geometry exists -- and it must happen *before*
        # ExportStage's georeferencing, since the poses the vote projects
        # through are still in the same ENU frame as these points. Labelling
        # after georeferencing would need the Sim(3) applied to the poses
        # too, which is a second chance to get a transform wrong for no
        # benefit.
        semantic_stats = self._label_clouds(state, raw_point_cloud, vertices)

        # Grade the geometry against the frames that produced it. Runs in
        # the same pre-georeference frame as labelling, for the same reason:
        # points and camera poses are still in one consistent frame here.
        photometric_stats = self._verify_photometrically(state, raw_point_cloud, vertices, confidence)

        voxel_summary = (
            f"voxel_size={stats.get('voxel_size', 'n/a')} "
            f"(source={stats.get('voxel_size_source', 'n/a')}, "
            f"gsd={stats.get('voxel_size_gsd_m', 'n/a')}, "
            f"ideal={stats.get('voxel_size_ideal_m', 'n/a')}, "
            f"budget_exceeded={stats.get('voxel_size_budget_exceeded', 'n/a')}, "
            f"grid_dims={stats.get('grid_dims', 'n/a')})"
        )

        if vertices.shape[0] == 0:
            artifacts = {"applied": False, "n_vertices": 0, "n_faces": 0, "n_raw_points": int(raw_point_cloud.xyz.shape[0]), **stats}
            survivors = ", ".join(
                f"{step}={stats[step]}"
                for step in ("raw_submap_points", "merged", "after_outlier_removal", "after_confidence_filter", "after_voxel_downsample")
                if step in stats
            )
            reason = stats.get("empty_at_step", "unknown step")
            message = (
                f"fusion produced no vertices -- emptied at {reason}; points surviving each stage: {survivors}; "
                f"{voxel_summary}; keeping geometry's point cloud "
                f"({raw_point_cloud.xyz.shape[0]} raw fused points still available for point_cloud.ply/.las)"
            )
            return artifacts, message

        # Strip Poisson spikes and speck islands BEFORE anything downstream
        # sees the mesh -- texture baking especially, since a sliver thrown
        # across a depth discontinuity gets its own UV chart and smears
        # whatever it happened to cross into the atlas.
        #
        # This only deletes faces; it never smooths or moves a vertex, so
        # every measurement taken off the mesh is unchanged. Per-vertex
        # arrays are re-indexed through `keep` because clean_mesh compacts
        # the vertex list -- dropping that step would silently mis-pair
        # colour and confidence with geometry.
        clean_stats: dict = {}
        if faces.size and getattr(state.config.fusion, "mesh_cleanup", True):
            from drishti3d.fusion.mesh import clean_mesh

            cfg_fusion = state.config.fusion
            cleaned_v, cleaned_f, clean_stats = clean_mesh(
                vertices,
                faces,
                max_edge_factor=getattr(cfg_fusion, "mesh_max_edge_factor", 6.0),
                min_component_faces=getattr(cfg_fusion, "mesh_min_component_faces", 64),
            )
            if cleaned_f.size:
                # Gather every parallel per-vertex channel through the same
                # mapping clean_mesh used to compact the vertices. Skipping
                # this leaves colour and confidence attached to the wrong
                # points -- a corruption that looks like a texturing bug.
                kept = clean_stats["kept_vertices"]
                vertices, faces = cleaned_v, cleaned_f
                if colors.size:
                    colors = colors[kept]
                if confidence.size:
                    confidence = confidence[kept]
                for attr in ("_mesh_semantic_class", "_mesh_semantic_confidence", "_mesh_photometric_confidence"):
                    arr = getattr(state, attr, None)
                    if arr is not None and len(arr) > kept.max():
                        setattr(state, attr, np.asarray(arr)[kept])
                logger.info(
                    "fusion: mesh cleanup removed %d spike + %d speck faces (%.2f%% of %d)",
                    clean_stats.get("removed_spike_faces", 0),
                    clean_stats.get("removed_component_faces", 0),
                    clean_stats.get("removed_faces_pct", 0.0),
                    clean_stats.get("faces_in", 0),
                )
            # `kept_vertices` is an array; keep it out of the JSON artifacts.
            clean_stats = {k: v for k, v in clean_stats.items() if k != "kept_vertices"}

        # Cap the mesh size. A fine TSDF voxel over a survey-scale site
        # yields tens of millions of marching-cubes triangles (34M vertices
        # / 69M faces measured at 0.65 m) -- a 1.4 GB GLB that no viewer
        # opens and a texture bake that runs for hours. Quadric decimation
        # keeps the surface shape; every per-vertex channel is remapped by
        # nearest original vertex so nothing is left mis-paired.
        max_faces = int(getattr(state.config.fusion, "max_mesh_faces", 0) or 0)
        decimation_stats: dict = {}
        if max_faces > 0 and faces.shape[0] > max_faces:
            from drishti3d.fusion.mesh import decimate_to_cap

            new_v, new_f, nn_index = decimate_to_cap(vertices, faces, max_faces)
            decimation_stats = {"faces_before": int(faces.shape[0]), "faces_after": int(new_f.shape[0])}
            vertices, faces = new_v, new_f
            if colors.size:
                colors = colors[nn_index]
            if confidence.size:
                confidence = confidence[nn_index]
            for attr in ("_mesh_semantic_class", "_mesh_semantic_confidence", "_mesh_photometric_confidence"):
                arr = getattr(state, attr, None)
                if arr is not None and len(arr) > nn_index.max():
                    setattr(state, attr, np.asarray(arr)[nn_index])
            logger.info(
                "fusion: decimated mesh %d -> %d faces (max_mesh_faces=%d)",
                decimation_stats["faces_before"],
                decimation_stats["faces_after"],
                max_faces,
            )

        mesh_conf = getattr(state, "_mesh_photometric_confidence", None)
        if mesh_conf is not None and len(mesh_conf) == len(vertices):
            confidence = mesh_conf

        state.point_cloud = PointCloud(
            xyz=vertices,
            rgb=colors if colors.size else None,
            confidence=confidence if confidence.size else None,
            semantic_class=getattr(state, "_mesh_semantic_class", None),
            semantic_confidence=getattr(state, "_mesh_semantic_confidence", None),
        )
        state.mesh_faces = faces if faces.size else None

        artifacts = {
            "applied": True,
            "n_vertices": int(vertices.shape[0]),
            "n_faces": int(faces.shape[0]),
            "n_raw_points": int(raw_point_cloud.xyz.shape[0]),
            **semantic_stats,
            **photometric_stats,
            **clean_stats,
            **stats,
        }
        message = (
            f"fusion produced a {vertices.shape[0]}-vertex / {faces.shape[0]}-face confidence-weighted TSDF mesh "
            f"({voxel_summary}, {stats.get('deduplicated', 0)} duplicate points removed on merge); "
            f"{raw_point_cloud.xyz.shape[0]}-point raw fused cloud also exported as point_cloud.ply/.las"
        )
        if semantic_stats.get("semantic_labelled_pct"):
            message += (
                f"; {semantic_stats['semantic_labelled_pct']:.1f}% of points classified "
                f"({semantic_stats.get('semantic_views_voted', 0)} views voted)"
            )
        return artifacts, message

    # ------------------------------------------------------------------
    def _keyframe_altitudes(self, state) -> dict[int, float]:
        """Telemetry height-above-takeoff per keyframe, for scale anchoring.

        Returns ``{}`` when telemetry carries no relative altitude, which
        disables altitude anchoring rather than substituting a guess --
        ``geometry.submap.altitude_anchor_scale`` then falls through to the
        backbone's own metric claim.

        ``alt_rel``, not ``alt_msl``: the anchor needs height above the
        ground being photographed, and MSL altitude carries the takeoff
        site's elevation with it, which would offset every measurement by
        that constant.
        """
        altitudes: dict[int, float] = {}
        for i, kf in enumerate(state.keyframes or []):
            geo = getattr(getattr(kf, "telemetry", None), "geo", None)
            alt = getattr(geo, "alt_rel", None) if geo is not None else None
            if alt is not None and float(alt) > 0.0:
                altitudes[i] = float(alt)
        if not altitudes:
            logger.info("fusion: no relative altitude in telemetry; altitude anchoring disabled")
        return altitudes

    #: Cap on keyframes decoded for any photometric pass. Beyond a few
    #: dozen well-spread views, additional ones re-observe surfaces already
    #: covered, and each one costs a full-resolution decode.
    _PHOTOMETRIC_MAX_VIEWS = 40

    def _photometric_views(self, state) -> list:
        """Decode the keyframes used for photometric work, once per run.

        Cached on ``state`` because both the pre-mesh rejection filter and
        the post-mesh re-grading need the same frames, and decoding 40
        4K frames twice is a minute of wall clock spent re-reading the same
        video. Returns ``[]`` when photometric work cannot run at all, so
        callers branch on emptiness rather than each re-deriving the
        preconditions.
        """
        cached = getattr(state, "_photometric_view_cache", None)
        if cached is not None:
            return cached

        views: list = []
        if state.video is not None and state.keyframes:
            poses = state.poses if len(state.poses) == len(state.keyframes) else None
            if poses is None:
                logger.info("photometric: poses not index-aligned with keyframes; skipping")
            else:
                indices = list(range(len(state.keyframes)))
                if len(indices) > self._PHOTOMETRIC_MAX_VIEWS:
                    step = len(indices) / float(self._PHOTOMETRIC_MAX_VIEWS)
                    indices = [indices[int(i * step)] for i in range(self._PHOTOMETRIC_MAX_VIEWS)]
                # Same shared cache the geometry and matching stages use
                # (pipeline.framecache): without it this re-decodes up to
                # 40 keyframes from the 4K source at ~1.87 s each, purely
                # to sample colours it already had. Intrinsics are scaled
                # by the cache's own factor so the projection stays exact.
                cache = getattr(state, "keyframe_cache", None)
                from drishti3d.geometry.mapanything import scale_intrinsics

                for i in indices:
                    kf = state.keyframes[i]
                    intr = kf.intrinsics or state.intrinsics
                    if intr is None:
                        continue
                    image = cache.get(i) if cache is not None else None
                    if image is not None:
                        views.append((poses[i], scale_intrinsics(intr, cache.scale_for(i)), image))
                        continue
                    frames = state.video.read_frames([kf.frame_index])
                    if not frames or frames[0].image is None:
                        continue
                    views.append((poses[i], intr, frames[0].image))

        state._photometric_view_cache = views
        return views

    def _pre_mesh_point_filter(self, state):
        """Compose every pre-mesh keep-mask into one callable, or ``None``.

        Two independent checks, deliberately in this order:

        1. **Ground band** (fusion.ground_bounds) -- geometric, needs only
           telemetry, and catches the failure the mesh cleanup provably
           cannot see: dense well-formed geometry standing where terrain
           physically is not. Cheap, so it runs first and shrinks the
           cloud the expensive check has to look at.
        2. **Photometric** -- deletes points the images actively disagree
           about.

        A point must pass both. Either check may decline (no telemetry
        ground, photometric disabled), in which case the other still runs.
        """
        checks = []

        ground_z = self._ground_z(state)
        if ground_z is not None:
            from drishti3d.fusion.ground_bounds import filter_to_ground_band

            def _ground(cloud, _z=ground_z):
                keep, diag = filter_to_ground_band(cloud.xyz, _z)
                state.ground_band = diag
                return keep

            checks.append(_ground)

        photometric = self._photometric_point_filter(state)
        if photometric is not None:
            checks.append(photometric)

        if not checks:
            return None
        if len(checks) == 1:
            return checks[0]

        def _combined(cloud):
            keep = np.ones(cloud.xyz.shape[0], dtype=bool)
            for check in checks:
                keep &= np.asarray(check(cloud), dtype=bool)
            return keep

        return _combined

    def _ground_z(self, state) -> float | None:
        """Telemetry ground height, the same reference the placement check uses."""
        try:
            from drishti3d.export.placement import ground_z_from_telemetry

            keyframes = state.keyframes or []
            cams = np.array([kf.pose.t for kf in keyframes if kf.pose is not None], dtype=np.float64)
            if not len(cams):
                return None
            altitudes = [
                kf.telemetry.geo.alt_rel if kf.telemetry is not None and kf.telemetry.geo is not None else None
                for kf in keyframes
                if kf.pose is not None
            ]
            return ground_z_from_telemetry(cams, altitudes)
        except Exception:
            logger.warning("fusion: could not derive a telemetry ground height", exc_info=True)
            return None

    def _photometric_point_filter(self, state):
        """Build the pre-mesh keep-mask callable, or ``None`` to skip it.

        Returns a callable because ``fuse_submaps`` is the only place the
        merged world-frame cloud exists, and it is deliberately ignorant of
        video and poses (see the ``point_filter`` block in
        ``fusion.tsdf.fuse_submaps`` for why the dependency points this
        way).

        The mask keeps a point unless the frames that can see it actively
        disagree about it. ``unverifiable`` points -- too few views, or too
        little texture for agreement to carry information -- are kept. That
        asymmetry is the whole design: this deletes on *evidence of error*,
        never on *absence of evidence*, so a featureless road survives and
        a depth spike floating above a driveway does not.
        """
        fcfg = state.config.fusion
        if not getattr(fcfg, "photometric_verify", False):
            return None
        if not getattr(fcfg, "photometric_reject_before_mesh", False):
            return None

        views = self._photometric_views(state)
        if len(views) < fcfg.photometric_min_views:
            logger.info("photometric: only %d usable views; not filtering before mesh", len(views))
            return None

        from drishti3d.fusion.photometric import photometric_consistency

        def _filter(cloud):
            try:
                result = photometric_consistency(
                    cloud.xyz,
                    views,
                    min_views=fcfg.photometric_min_views,
                    min_contrast=fcfg.photometric_min_contrast,
                    agree_threshold=fcfg.photometric_agree_threshold,
                )
            except Exception:
                # Keeping everything is the honest failure mode: it leaves
                # the cloud exactly as it would have been without this
                # feature, rather than deleting on a check that crashed.
                logger.warning("photometric: pre-mesh filter failed; keeping all points", exc_info=True)
                return np.ones(cloud.xyz.shape[0], dtype=bool)

            inconsistent = (~result.unverifiable) & (~result.verified) & np.isfinite(result.error)
            keep = ~inconsistent
            logger.info(
                "photometric pre-mesh: %d/%d points rejected as view-inconsistent "
                "(%d verified, %d unverifiable kept)",
                int(inconsistent.sum()),
                len(keep),
                int(result.verified.sum()),
                int(result.unverifiable.sum()),
            )
            return keep

        return _filter

    def _verify_photometrically(self, state, raw_point_cloud, mesh_vertices, mesh_confidence) -> dict:
        """Re-grade confidence against the source frames. Returns stats.

        This replaces a self-reported confidence with an externally checked
        one: ``confidence_source`` becomes ``photometric_verification``
        rather than ``backbone_confidence_and_view_count``. The distinction
        matters -- the first is evidence, the second is the model's opinion
        of itself.

        Returns ``{}`` when verification is disabled or there is nothing to
        verify against, so the report card reports the backbone source
        rather than implying a check that never ran.
        """
        cfg = getattr(state.config.fusion, "photometric_verify", False)
        if not cfg or state.video is None or not state.keyframes:
            return {}

        from drishti3d.fusion.photometric import (
            photometric_consistency,
            verify_confidence,
        )

        fcfg = state.config.fusion
        # Same decoded frames the pre-mesh filter used -- see
        # _photometric_views on why this is cached rather than re-read.
        views = self._photometric_views(state)

        if len(views) < fcfg.photometric_min_views:
            logger.info("photometric: only %d usable views; skipping verification", len(views))
            return {}

        out: dict = {}
        try:
            result = photometric_consistency(
                raw_point_cloud.xyz,
                views,
                min_views=fcfg.photometric_min_views,
                min_contrast=fcfg.photometric_min_contrast,
                agree_threshold=fcfg.photometric_agree_threshold,
            )
        except Exception:
            logger.warning("photometric: verification failed; keeping backbone confidence", exc_info=True)
            return {}

        if raw_point_cloud.confidence is not None:
            raw_point_cloud.confidence, regrade = verify_confidence(raw_point_cloud.confidence, result)
            out.update(regrade)

        out.update({f"photometric_{k}": v for k, v in result.stats.items()})
        state.fusion_confidence_source = "photometric_verification"

        # The mesh is a different vertex set, so it needs its own pass.
        if mesh_vertices is not None and len(mesh_vertices) and mesh_confidence is not None and mesh_confidence.size:
            try:
                mesh_result = photometric_consistency(
                    mesh_vertices,
                    views,
                    min_views=fcfg.photometric_min_views,
                    min_contrast=fcfg.photometric_min_contrast,
                    agree_threshold=fcfg.photometric_agree_threshold,
                )
                verified_conf, _ = verify_confidence(mesh_confidence, mesh_result)
                state._mesh_photometric_confidence = verified_conf
            except Exception:
                logger.debug("photometric: mesh-vertex pass failed", exc_info=True)
        return out

    def _label_clouds(self, state, raw_point_cloud, mesh_vertices) -> dict:
        """Vote per-point ``SemanticClass`` onto the raw cloud and the mesh.

        Both are labelled from the same masks and the same poses, so a
        point in the exported ``point_cloud.las`` and the mesh vertex
        nearest to it cannot disagree for any reason other than the
        genuine geometric difference between them.

        Returns a stats dict merged into the stage's artifacts. Returns an
        empty dict -- not zeroed stats -- when semantics never ran, so the
        report card can tell "not classified" from "classified as nothing".
        """
        masks = getattr(state, "semantic_masks", None)
        if not masks:
            return {}

        from drishti3d.semantics.classes import class_histogram
        from drishti3d.semantics.labelling import label_points

        cfg = state.config.semantics

        # Prefer bundle-adjusted poses when BA ran and kept the keyframe
        # ordering; fall back to each keyframe's own pose otherwise. A
        # pose/keyframe length mismatch means the two are not index-aligned
        # and pairing them anyway would project through the wrong camera.
        poses = state.poses if len(state.poses) == len(state.keyframes) else None

        views = []
        for idx, kf in enumerate(state.keyframes):
            result = masks.get(idx)
            if result is None:
                continue
            pose = poses[idx] if poses is not None else kf.pose
            intr = kf.intrinsics or state.intrinsics
            if pose is None or intr is None:
                continue
            views.append((pose, intr, result.labels, result.confidence))

        if not views:
            logger.warning("semantics: masks exist but no keyframe had both a pose and intrinsics; not labelling")
            return {}

        out: dict = {}
        vote = label_points(
            raw_point_cloud.xyz,
            views,
            min_vote_ratio=cfg.min_vote_ratio,
            min_views=cfg.min_views,
            occlusion_tolerance_m=cfg.occlusion_tolerance_m,
            use_occlusion=cfg.use_occlusion,
        )
        raw_point_cloud.semantic_class = vote.semantic_class
        raw_point_cloud.semantic_confidence = vote.vote_ratio

        if mesh_vertices is not None and len(mesh_vertices):
            mesh_vote = label_points(
                mesh_vertices,
                views,
                min_vote_ratio=cfg.min_vote_ratio,
                min_views=cfg.min_views,
                occlusion_tolerance_m=cfg.occlusion_tolerance_m,
                use_occlusion=cfg.use_occlusion,
            )
            state._mesh_semantic_class = mesh_vote.semantic_class
            state._mesh_semantic_confidence = mesh_vote.vote_ratio

        histogram = class_histogram(vote.semantic_class)
        labelled_pct = 100.0 * vote.stats["labelled_points"] / max(vote.stats["points"], 1)

        state.semantic_point_histogram = histogram
        state.semantic_vote_stats = vote.stats

        out["semantic_class_pct"] = histogram
        out["semantic_labelled_pct"] = round(labelled_pct, 2)
        out["semantic_views_voted"] = vote.stats["views_voted"]
        out["semantic_unseen_points"] = vote.stats["unseen_points"]
        out["semantic_disputed_points"] = vote.stats["disputed_points"]
        out["semantic_mean_views_per_point"] = round(vote.stats["mean_views_per_point"], 2)
        return out


def _transform_covariance(covariance: np.ndarray | None, transform) -> np.ndarray | None:
    """Propagate per-point covariance through a ``geometry.submap.Sim3`` transform.

    Same rule ``geometry.submap.merge_submaps`` already uses for exactly
    this reason: ``Cov' = scale^2 * R @ Cov @ R^T`` (standard linear/
    similarity-transform covariance propagation).
    """
    if covariance is None:
        return None
    cov = np.asarray(covariance).reshape(-1, 3, 3)
    return (transform.scale**2) * np.einsum("ij,njk,lk->nil", transform.R, cov, transform.R)


def _apply_georeferencing(state: PipelineState) -> dict:
    """Georeference ``state.point_cloud``/``state.poses`` against GPS, in place, and report the result.

    This is the second half of the metric-scale fix (see
    ``_poses_from_telemetry`` for the first half): ``geometry.georef
    .georeference`` computes a Sim(3) transform from the reconstruction's
    local metric frame into the GPS-derived ENU frame -- correcting
    residual rotation/translation *and*, critically, any global scale
    error bundle adjustment's soft GPS priors didn't fully pin down.
    ``georeference``'s own docstring says as much about its ``point_cloud``
    parameter: "used only for its size; the transform is what a caller
    then applies to it." Previously, no caller ever did -- this function's
    predecessor (``_georeference_for_report``) called ``georeference``
    purely to read off the accuracy report card's numbers and threw the
    transform away, so a reconstruction that was internally, say, 14x too
    small stayed exactly that small in every exported deliverable (PLY/LAS/
    GLB) even though this same stage had already computed the correct
    Sim(3) to fix it. This function actually applies it.

    Returns a ``report_artifacts``-shaped dict (``relative_rmse_m``/
    ``absolute_rmse_m``/``scale_error_pct``/``georef_notes``, each ``None``
    -- i.e. "not computed" once ``export.report.build_report`` sees it --
    when georeferencing genuinely can't run: no telemetry sidecar, a
    keyframe missing a telemetry sample, poses not lined up 1:1 with
    keyframes, or fewer than 3 geo-tagged keyframes). Never raises --
    georeferencing is best-effort, matching every other optional stage's
    degrade-gracefully contract in this module; when it can't run,
    ``state.point_cloud``/``state.poses`` are left exactly as handed in
    (still whatever local metric frame upstream produced -- not
    georeferenced, but not corrupted either).
    """
    not_computed = {"relative_rmse_m": None, "absolute_rmse_m": None, "scale_error_pct": None, "georef_notes": None}

    if state.telemetry_path is None or not state.keyframes or len(state.poses) != len(state.keyframes):
        return not_computed
    if state.point_cloud is None:
        return not_computed

    telemetry = [kf.telemetry for kf in state.keyframes]
    if any(t is None for t in telemetry):
        return not_computed

    try:
        from drishti3d.geometry.georef import georeference
    except ImportError:
        return not_computed

    try:
        # relative_accuracy_m is deliberately left None: converting BA's
        # px reprojection RMSE to a metre figure needs per-point depth (see
        # georeference()'s own docstring on why that's the *right* source
        # once available) which this stage doesn't have on hand -- passing
        # a made-up conversion here would violate this report's one hard
        # rule (see export.report's module docstring: never fabricate a
        # number that wasn't actually measured). georeference() falls back
        # to its own GPS-alignment-residual proxy instead, which is a
        # real, if conservative, measurement (and now notes explicitly
        # when that proxy has absorbed a large scale correction -- see
        # georeference()'s own comment on relative_accuracy_is_fallback).
        # Fix rotation (identity -- state.poses are already expressed in
        # the world/ENU frame by this point) instead of letting
        # georeference() re-derive it via Umeyama on (post-BA) camera
        # centres, whenever GeometryStage's own merge already had to solve
        # exactly this problem via strategy="telemetry_rotation" (a
        # near-collinear single-flight-strip track -- see
        # geometry.submap's module docstring). Skipping this would silently
        # re-introduce the same rotation-from-collinear-points degeneracy
        # at the georeferencing step, undoing the submap-merge fix on the
        # one number (the exported, georeferenced point cloud) that
        # actually matters -- see geometry.georef.align_to_gps's
        # fixed_rotation docstring for the full reasoning.
        fixed_rotation = np.eye(3) if getattr(state, "geometry_merge_strategy", None) == "telemetry_rotation" else None
        result = georeference(
            state.point_cloud, state.poses, telemetry, relative_accuracy_m=None, fixed_rotation=fixed_rotation
        )
    except ValueError:
        # Fewer than 3 geo-tagged samples, or some other precondition
        # georeference() itself enforces -- a legitimate "can't compute
        # this," not a bug.
        logger.info("export: georeferencing not computed for the report card", exc_info=True)
        return not_computed

    state.point_cloud = PointCloud(
        xyz=result.transform.apply(np.asarray(state.point_cloud.xyz).reshape(-1, 3)),
        rgb=state.point_cloud.rgb,
        covariance=_transform_covariance(state.point_cloud.covariance, result.transform),
        confidence=state.point_cloud.confidence,
        # Carried, not recomputed: a Sim(3) moves points, it does not change
        # what they ARE. Omitting these here silently dropped every semantic
        # label between fusion and export -- the report card still showed a
        # class breakdown (computed pre-georeference) while the exported LAS
        # had no semantic_class dimension and an all-zero ASPRS
        # classification field. Any new per-point channel must be added
        # here and in the raw-cloud rebuild below, or it dies at this line.
        semantic_class=state.point_cloud.semantic_class,
        semantic_confidence=state.point_cloud.semantic_confidence,
    )
    state.poses = [result.transform.apply_pose(p) for p in state.poses]

    # FusionStage's raw (pre-TSDF) fused point cloud (Fix 2) lives in the
    # exact same local frame state.point_cloud was just transformed out of
    # (both came out of the same fuse_submaps/merge_submaps call) -- apply
    # the identical Sim(3) correction so point_cloud.ply/.las lines up with
    # every other georeferenced deliverable instead of staying in the
    # pre-georeference local frame.
    raw_pc = getattr(state, "fusion_raw_point_cloud", None)
    if raw_pc is not None and raw_pc.xyz.shape[0] > 0:
        state.fusion_raw_point_cloud = PointCloud(
            xyz=result.transform.apply(np.asarray(raw_pc.xyz).reshape(-1, 3)),
            rgb=raw_pc.rgb,
            covariance=_transform_covariance(raw_pc.covariance, result.transform),
            confidence=raw_pc.confidence,
            semantic_class=raw_pc.semantic_class,
            semantic_confidence=raw_pc.semantic_confidence,
        )

    # Silent scale error is the worst failure mode this system can have
    # (see this project's own priorities): if georeferencing's own Sim(3)
    # fit still needed a substantial scale correction, that means whatever
    # upstream produced (backbone conditioning, BA priors, submap merging)
    # left real, uncorrected metric-scale error -- worth a loud, impossible
    # -to-miss warning even though georeferencing has now patched the
    # *exported* geometry's scale.
    tolerance = state.config.export.scale_warning_tolerance
    scale_deviation = abs(result.transform.scale - 1.0)
    if scale_deviation > tolerance:
        # Note scale_error_pct (the raw, pre-alignment, rotation-free pairwise-
        # distance-ratio measurement) alongside the Sim(3) fit's own scale --
        # the two can legitimately disagree, and the size of that gap is
        # itself diagnostic: this fit's rotation component is exactly what
        # geometry.submap's near-collinear-flight-strip degeneracy warning
        # (see that module's docstring) is about, so a much larger
        # transform.scale correction than scale_error_pct alone would
        # suggest often means the *alignment* (not necessarily the
        # underlying reconstruction) is what's poorly constrained here --
        # both figures are worth investigating together, not just the
        # larger one in isolation.
        warning = (
            f"post-georeference scale factor {result.transform.scale:.4f} deviates from 1.0 by "
            f"{scale_deviation * 100.0:.1f}% (tolerance {tolerance * 100.0:.1f}%). The raw, "
            f"pre-alignment scale_error_pct is {result.scale_error_pct:+.2f}% -- if these two "
            "figures disagree substantially, that gap itself is worth investigating (possible "
            "causes: genuine upstream metric-scale error from backbone pose conditioning/BA "
            "priors/submap merging, OR a poorly-constrained Sim(3) rotation fit on a "
            "near-collinear single-flight-strip camera track, which biases scale together with "
            "rotation -- see geometry.submap's degenerate-alignment discussion). Either way, "
            "georeferencing has applied this correction to the exported geometry, but the "
            "discrepancy should not be treated as resolved without checking which it was."
        )
        logger.warning("GEOREFERENCE SCALE WARNING: %s", warning)
        result.notes.append(f"WARNING: {warning}")

    return {
        "relative_rmse_m": result.relative_accuracy_m,
        "absolute_rmse_m": result.absolute_accuracy_m,
        "scale_error_pct": result.scale_error_pct,
        "georef_notes": list(result.notes) if result.notes else None,
    }


class ExportStage(PipelineStage):
    """Writes the final deliverables (PLY/LAS/GLB/OBJ/XYZ, rasters, report card) via ``drishti3d.export.export_all``.

    Uses ``state.mesh_faces`` (set by ``FusionStage`` when it produced a
    real mesh) alongside ``state.point_cloud`` so a genuine TSDF mesh gets
    exported as a faceted OBJ/GLB/PLY rather than a points-only file --
    ``state.point_cloud`` alone never carries face connectivity (see
    ``FusionStage``'s own docstring on why).
    """

    name = "export"

    def run(self, state, cancel_token, progress_cb, partial_cb=None):
        _check_cancel(cancel_token)

        try:
            from drishti3d.export import export_all
        except ImportError as exc:
            raise StageUnavailable(f"drishti3d.export not available yet: {exc}") from exc

        if state.point_cloud is None or state.point_cloud.xyz.shape[0] == 0:
            raise StageUnavailable("export: no point cloud to export (geometry/fusion stage produced none)")

        _check_cancel(cancel_token)

        # Georeference *before* building the export payload below: this
        # both fills in the accuracy report card's relative/absolute/scale
        # figures AND (see _apply_georeferencing's docstring) actually
        # rewrites state.point_cloud/state.poses into the GPS-aligned,
        # metrically-corrected frame -- everything read from `state` from
        # this point on (result_or_pointcloud, `poses=state.poses` below)
        # must see the georeferenced version, not the raw local-frame one.
        georef_report = _apply_georeferencing(state)

        export_cfg = state.config.export
        formats = {"glb", "xyz"}
        if export_cfg.export_ply:
            formats.add("ply")
        if export_cfg.export_las:
            formats.add("las")
        if state.mesh_faces is not None:
            formats.add("obj")
            # FBX is named explicitly in the problem statement's output
            # format list. export.bundle_export has supported it since it
            # was written; this stage simply never asked for it, so every
            # run silently shipped five of the six required formats.
            formats.add("fbx")

        result_or_pointcloud = (
            (state.point_cloud.xyz, state.mesh_faces, state.point_cloud.rgb, state.point_cloud.confidence)
            if state.mesh_faces is not None
            else state.point_cloud
        )

        # PipelineState.report (the informal run summary -- distinct from
        # this accuracy report card, see pipeline.runner) is only filled in
        # by the runner once every stage has a terminal StageResult, so it
        # isn't useful here -- but several *other* stages already computed
        # real numbers this report card was previously discarding, stashed
        # as plain PipelineState attributes for exactly this reason (see
        # BundleAdjustmentStage.run's comment): reprojection error from
        # bundle adjustment, per-stage timings tracked live by the runner,
        # timeline coverage from triage, submap junction alignment from
        # geometry.submap, and relative/absolute accuracy + scale error
        # from georeferencing (best-effort -- see
        # _apply_georeferencing, called above). Every value below is either
        # something a real stage actually measured, or left out of the
        # dict entirely so build_report's NOT_COMPUTED default takes over
        # -- never a fabricated number.
        report_artifacts: dict = {
            "keyframe_count": len(state.keyframes),
            "confidence_source": getattr(state, "fusion_confidence_source", None),
        }

        mean_reprojection_error_px = getattr(state, "mean_reprojection_error_px", None)
        if mean_reprojection_error_px is not None:
            report_artifacts["mean_reprojection_error_px"] = mean_reprojection_error_px

        coverage_fraction = state.triage_report.get("timeline_coverage_fraction")
        if coverage_fraction is not None:
            report_artifacts["coverage_pct"] = float(coverage_fraction) * 100.0

        # Telemetry/video time-sync provenance (see ingest.telemetry
        # .load_telemetry's docstring): a georeferenced product whose
        # video-start offset was guessed ("assumed_zero"), not measured or
        # explicitly given, needs to say so on its own accuracy report --
        # see export.report's module docstring on never letting a report
        # card look more certain than the pipeline actually was.
        if "time_offset_s" in state.telemetry_stats:
            report_artifacts["telemetry_offset_s"] = state.telemetry_stats["time_offset_s"]
            report_artifacts["telemetry_offset_source"] = state.telemetry_stats.get("offset_source")
            report_artifacts["telemetry_format"] = state.telemetry_stats.get("format")
        if "telemetry_video_coverage_fraction" in state.telemetry_stats:
            report_artifacts["telemetry_video_coverage_fraction"] = state.telemetry_stats[
                "telemetry_video_coverage_fraction"
            ]

        stage_timings = getattr(state, "stage_timings_s", None)
        if stage_timings:
            report_artifacts["stage_timings_s"] = dict(stage_timings)

        # Prefer GeometryStage's own already-computed junction diagnostics
        # (state.geometry_junction_residuals -- computed with the actual
        # merge_strategy/camera_gps_enu/conditioned_R that run used, see
        # GeometryStage.run) over recomputing from scratch: a bare
        # ``alignment_residuals(state.submaps)`` call here would silently
        # default back to strategy="chained_sim3" with no GPS/telemetry
        # context, showing a different (and misleading) picture than what
        # actually merged this model.
        # Parallax depth anchoring: the backbone's measured depth error per
        # window and whether it was corrected. Belongs on the card because
        # a reader of relative_rmse_m needs to know the depth was anchored
        # to parallax rather than taken from the backbone's own scale.
        report_artifacts["depth_anchor"] = getattr(state, "depth_anchor_summary", None)
        report_artifacts["pose_prior"] = getattr(state, "pose_prior_summary", None)
        # Where the windows landed, measured before fusion ran (see
        # export.placement). On the card because a reader of
        # relative_rmse_m needs to know whether the mesh is thick because
        # meshing was poor or because the windows were never in the same
        # place to begin with -- those have entirely different fixes.
        placement_report = getattr(state, "placement_report", None)
        report_artifacts["placement"] = placement_report.as_dict() if placement_report is not None else None
        report_artifacts["scale_consensus"] = getattr(state, "scale_consensus", None)

        geometry_junction_residuals = getattr(state, "geometry_junction_residuals", None)
        if geometry_junction_residuals is not None:
            report_artifacts["junction_residuals"] = geometry_junction_residuals
        elif len(state.submaps) >= 2:
            try:
                from drishti3d.geometry.submap import alignment_residuals

                report_artifacts["junction_residuals"] = alignment_residuals(state.submaps)
            except Exception:
                logger.info("export: submap junction alignment not computed for the report card", exc_info=True)

        flight_profile = getattr(state, "geometry_flight_profile", None)
        if flight_profile is not None:
            report_artifacts["flight_profile"] = vars(flight_profile)
        merge_strategy = getattr(state, "geometry_merge_strategy", None)
        if merge_strategy is not None:
            report_artifacts["merge_strategy"] = merge_strategy
            report_artifacts["merge_strategy_reason"] = getattr(state, "geometry_merge_strategy_reason", None)
        merge_strategy_warnings = getattr(state, "geometry_merge_strategy_warnings", None)
        if merge_strategy_warnings:
            report_artifacts["merge_strategy_warnings"] = merge_strategy_warnings

        # Scene composition. Every key is set only when semantics actually
        # ran -- an absent key is what makes build_report render
        # "not computed" instead of a fabricated all-zero class breakdown.
        semantic_histogram = getattr(state, "semantic_point_histogram", None)
        if semantic_histogram:
            report_artifacts["semantic_class_pct"] = semantic_histogram
            report_artifacts["semantic_model"] = getattr(state, "semantic_model", None)
            vote_stats = getattr(state, "semantic_vote_stats", None) or {}
            if vote_stats:
                points = max(int(vote_stats.get("points", 0)), 1)
                report_artifacts["semantic_labelled_pct"] = round(
                    100.0 * int(vote_stats.get("labelled_points", 0)) / points, 2
                )
                report_artifacts["semantic_unseen_points"] = vote_stats.get("unseen_points")
                report_artifacts["semantic_disputed_points"] = vote_stats.get("disputed_points")
                report_artifacts["semantic_views_voted"] = vote_stats.get("views_voted")
                report_artifacts["semantic_mean_views_per_point"] = round(
                    float(vote_stats.get("mean_views_per_point", 0.0)), 2
                )
        dynamic_removed = getattr(state, "geometry_dynamic_points_removed", None)
        if dynamic_removed:
            report_artifacts["dynamic_points_removed"] = dynamic_removed

        report_artifacts.update(georef_report)

        # Fix 2: the raw, cleaned (pre-TSDF, never coarsely voxelised) dense
        # point cloud FusionStage stashed -- written independently of
        # whatever result_or_pointcloud above ended up being (mesh or
        # point-cloud-only), so meshing parameters can never be the ceiling
        # on how much detail actually reaches the operator. Already
        # georeferenced above (_apply_georeferencing transforms it in
        # lockstep with state.point_cloud) when georeferencing ran at all.
        raw_point_cloud = getattr(state, "fusion_raw_point_cloud", None)
        if raw_point_cloud is not None and raw_point_cloud.xyz.shape[0] == 0:
            raw_point_cloud = None

        output_dir = Path(export_cfg.output_dir)
        # ``export_all`` builds the accuracy report card on its way to
        # writing report.html/report.txt. Capture it: it is the ONLY
        # place those numbers are computed, and a GUI that cannot show
        # them is a survey tool that will not state its own accuracy.
        report_card: dict = {}
        written = export_all(
            result_or_pointcloud,
            output_dir,
            formats=formats,
            crs=export_cfg.crs,
            poses=state.poses,
            report_artifacts=report_artifacts,
            raw_point_cloud=raw_point_cloud,
            report_out=report_card,
        )

        texture_stats = self._bake_texture(state, output_dir, written)

        artifacts = {"applied": True, "output_dir": str(output_dir), "files": {k: str(v) for k, v in written.items()}}
        if texture_stats:
            artifacts["texture"] = texture_stats
        if report_card:
            artifacts["report_card"] = report_card
            # Also promoted onto PipelineState.report so it survives into
            # PipelineResult.report (runner copies dict(state.report))
            # and therefore into a saved-and-reloaded run.
            state.report.update(report_card)
        message = f"wrote {len(written)} deliverable(s) to {output_dir}: {', '.join(sorted(written))}"
        if texture_stats:
            message += (
                f"; baked a {texture_stats['texture_size']}px texture atlas from "
                f"{texture_stats['views_used']} views ({texture_stats['texel_coverage_pct']:.1f}% texel coverage)"
            )
        return artifacts, message

    # ------------------------------------------------------------------
    def _bake_texture(self, state, output_dir: Path, written: dict) -> dict | None:
        """Bake and write ``model_textured.obj`` + ``.mtl`` + ``.png``.

        Runs last, after ``export_all``, and never raises into the stage:
        a failed bake must not cost the operator the point clouds, rasters
        and report that already wrote successfully. Returns ``None`` when
        texturing was disabled, impossible, or unsuccessful -- in which
        case ``model.obj``'s per-vertex colour remains the mesh
        deliverable.

        Texturing happens here rather than in FusionStage because it needs
        the *georeferenced* vertices: ``_apply_georeferencing`` has already
        transformed ``state.point_cloud`` and ``state.poses`` together by
        this point, so points and cameras are still in one consistent
        frame, and the atlas is baked against the geometry actually being
        exported.
        """
        cfg = getattr(state.config, "texture", None)
        if cfg is None or not cfg.enabled:
            return None
        faces = getattr(state, "mesh_faces", None)
        if faces is None or state.point_cloud is None or not len(state.point_cloud.xyz):
            return None
        if state.video is None:
            return None

        try:
            from drishti3d.export.formats import export_obj_textured
            from drishti3d.fusion.texture import bake_texture, is_available
        except ImportError:
            return None

        if not is_available():
            logger.info(
                "texture: xatlas not installed -- exporting per-vertex-coloured model.obj instead of a "
                "texture-mapped mesh. Install the 'texture' extra for a photographic atlas."
            )
            return None

        poses = state.poses if len(state.poses) == len(state.keyframes) else None
        if poses is None:
            logger.info("texture: poses are not index-aligned with keyframes; skipping texture bake")
            return None

        candidates = [
            (i, kf) for i, kf in enumerate(state.keyframes) if (kf.intrinsics or state.intrinsics) is not None
        ]
        if cfg.max_views and len(candidates) > cfg.max_views:
            # Keep an evenly-spaced subset rather than the first N: the
            # first N are the start of the flight and would texture only
            # one end of the scene.
            step = len(candidates) / float(cfg.max_views)
            candidates = [candidates[int(i * step)] for i in range(cfg.max_views)]

        views = []
        for i, kf in candidates:
            frames = state.video.read_frames([kf.frame_index])
            if not frames or frames[0].image is None:
                continue
            views.append(
                (
                    poses[i],
                    kf.intrinsics or state.intrinsics,
                    frames[0].image,
                    float(kf.metrics.blur_score) if kf.metrics is not None else 1.0,
                )
            )

        if not views:
            return None

        try:
            baked = bake_texture(
                state.point_cloud.xyz,
                faces,
                views,
                texture_size=cfg.texture_size,
                occlusion_tolerance_m=cfg.occlusion_tolerance_m,
                blend_views=cfg.blend_views,
            )
        except Exception:
            logger.warning("texture: bake failed; model.obj keeps per-vertex colour", exc_info=True)
            return None

        if baked is None:
            return None

        try:
            paths = export_obj_textured(
                output_dir / "model_textured.obj",
                baked.vertices,
                baked.faces,
                baked.uv,
                baked.texture,
            )
        except Exception:
            logger.warning("texture: writing the textured OBJ failed", exc_info=True)
            return None

        written.update(paths)
        return baked.stats
