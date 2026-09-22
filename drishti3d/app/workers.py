"""Qt-side pipeline execution handle.

This module runs the DRISHTI-3D pipeline off the UI thread so the window
never freezes while ingest / triage / geometry / fusion / export run.

Threading model, now vs. later
-------------------------------
``PipelineWorker`` is a ``QObject`` intended to be moved to a ``QThread``
via ``worker.moveToThread(thread)``. Today it just calls a plain Python
callable and reports progress through Qt signals.

IMPORTANT: once the real pipeline (drishti3d.ingest / drishti3d.triage /
drishti3d.geometry / drishti3d.fusion) lands, the heavy stages will hold
the GIL for long stretches and will depend on ``torch``. At that point the
pipeline should run in a separate **process** (e.g. via
``multiprocessing.Process`` or ``concurrent.futures.ProcessPoolExecutor``),
not merely a background thread — a Python thread cannot keep the Qt event
loop responsive against a GIL-bound numeric/torch workload the way a
separate process can, and a crash in native code (CUDA/torch) would take
the whole application down if it shared the main process.

``PipelineWorker`` is written so that swap is transparent to the UI: it is
already just a thin Qt-side handle around "some pipeline callable that
reports progress and eventually produces a result or an error". When the
process-based runner is introduced, only the innards of ``run()`` need to
change (e.g. polling a ``multiprocessing.Queue`` fed by the child process
instead of calling the callable directly) — the signal contract
(``progress``, ``stageChanged``, ``partialResult``, ``finished``,
``failed``) and the cancellation flag stay the same, so ``main_window.py``
does not need to change.

Demo mode
---------
``DemoPipeline`` is a callable with the same "pipeline callable" signature
that fabricates a plausible run: staged progress across the six
documented pipeline stages, and a synthetic ``PointCloud`` shaped like a
couple of buildings on a ground plane with a realistic mix of
``Confidence`` values. It lets the whole UI (progress panel, viewport
color modes, report panel) be exercised end-to-end with no video, no GPU,
and no wait, and doubles as an offline "cached demo" safety net for live
demonstrations.

Real mode
---------
``RealPipeline`` is the production callable: it drives
``drishti3d.pipeline.runner.run_pipeline`` over an actual video (+
optional telemetry), translating that function's ``(stage_name, current,
total, message)`` progress callback and ``PointCloud`` partial-result
callback onto the ``report_progress``/``report_stage``/``report_partial``
trio ``PipelineWorker`` expects. It returns a
``drishti3d.pipeline.result.PipelineResult`` (not a plain dict like
``DemoPipeline``) as its "finished" payload -- ``main_window.py``
distinguishes the two by type.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, Signal

from drishti3d.types import Confidence, PointCloud

if TYPE_CHECKING:
    from drishti3d.config import Config
    from drishti3d.pipeline.result import PipelineResult

# Canonical pipeline stage names, in execution order.
#
# Derived from ``panels.stage_rail.STAGE_SPECS`` rather than re-listed
# here. The hardcoded copy this replaces named six stages and claimed to
# "match pipeline.runner's six stages exactly" -- the runner has run
# ELEVEN since the preview stages were added, so drone_path, coverage,
# semantics, pose_prior and frame_placement had no row in the UI and
# logged nothing. pose_prior alone is 427 s of a 2108 s run on this
# project's sample flight: seven minutes during which the app showed the
# operator nothing at all.
from drishti3d.app.panels.stage_rail import STAGE_SPECS

STAGES: tuple[str, ...] = tuple(spec.name for spec in STAGE_SPECS)


class CancelToken:
    """Thread-safe cooperative cancellation flag.

    The running pipeline callable is expected to check ``is_set()`` between
    stages (and ideally periodically within a long stage) and raise
    ``PipelineCancelled`` or simply return early when set.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


class PipelineCancelled(Exception):
    """Raised by a pipeline callable to unwind cleanly on cancellation."""


# A pipeline callable receives (report_progress, report_stage, report_partial,
# cancel_token) and returns an arbitrary result object (e.g. a PointCloud or
# a dict of results) when it completes normally.
ProgressFn = Callable[[int, int, str], None]
StageFn = Callable[[str], None]
PartialFn = Callable[[object], None]
# ``report_preview(stage_name, snapshot)`` -- see
# ``pipeline.runner._preview_snapshot``. Optional and last so that a
# pipeline callable written against the original four-argument contract
# (as the tests are) still works unchanged.
PreviewFn = Callable[[str, object], None]
PipelineCallable = Callable[..., object]


class PipelineWorker(QObject):
    """Qt-side handle for running a pipeline callable off the UI thread.

    Intended usage::

        thread = QThread()
        worker = PipelineWorker(DemoPipeline())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.start()

    Signals
    -------
    progress(int, int, str):
        (current_step, total_steps, message) — a monotonically
        non-decreasing progress update within the current stage.
    stageChanged(str):
        Emitted when the pipeline moves to a new named stage.
    partialResult(object):
        Emitted whenever an intermediate/partial result is available
        (e.g. a preview PointCloud) so the UI can update before the run
        finishes.
    finished(object):
        Emitted once with the final result object on success.
    failed(str):
        Emitted with a formatted traceback string if the callable raises.
        The worker never lets an exception propagate out of ``run()``.
    """

    progress = Signal(int, int, str)
    stageChanged = Signal(str)
    partialResult = Signal(object)
    previewUpdate = Signal(str, object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, pipeline: PipelineCallable, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._cancel_token = CancelToken()

    def cancel(self) -> None:
        """Request cooperative cancellation. Safe to call from any thread."""
        self._cancel_token.cancel()

    def run(self) -> None:
        """Execute the pipeline callable, catching all exceptions.

        Connect this to ``QThread.started`` (or invoke via
        ``QMetaObject.invokeMethod``) rather than calling it directly from
        the UI thread.
        """
        try:
            result = self._pipeline(
                self.progress.emit,
                self.stageChanged.emit,
                self.partialResult.emit,
                self._cancel_token,
                self.previewUpdate.emit,
            )
        except PipelineCancelled:
            self.failed.emit("Pipeline cancelled by user.")
            return
        except Exception:  # noqa: BLE001 - by design: never let the worker thread crash the app
            self.failed.emit(traceback.format_exc())
            return

        self.finished.emit(result)


def _make_building(
    origin: np.ndarray,
    width: float,
    depth: float,
    height: float,
    density: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on the four walls + roof of a box "building".

    The front wall (facing -Y, i.e. toward the camera/operator) is marked
    MEASURED (directly observed), the back wall is INFERRED (never seen by
    any camera, filled in by the model's prior), and the side walls / roof
    are LOW_CONFIDENCE (partially observed at grazing angles).
    """
    pts: list[np.ndarray] = []
    conf: list[int] = []

    def wall(u_range, v_range, fixed_axis: str, fixed_val: float, n: int, confidence: int):
        u = rng.uniform(*u_range, size=n)
        v = rng.uniform(*v_range, size=n)
        pack = np.zeros((n, 3), dtype=np.float64)
        if fixed_axis == "x":
            pack[:, 0] = fixed_val
            pack[:, 1] = u
            pack[:, 2] = v
        elif fixed_axis == "y":
            pack[:, 0] = u
            pack[:, 1] = fixed_val
            pack[:, 2] = v
        else:
            pack[:, 0] = u
            pack[:, 1] = v
            pack[:, 2] = fixed_val
        pts.append(pack + origin)
        conf.append(np.full(n, confidence, dtype=np.uint8))

    n = density
    # Front wall (-Y face): directly observed by the drone pass.
    wall((0, width), (0, height), "y", 0.0, n, int(Confidence.MEASURED))
    # Back wall (+Y face): never seen, purely inferred.
    wall((0, width), (0, height), "y", depth, n, int(Confidence.INFERRED))
    # Side walls: seen at grazing incidence only.
    wall((0, depth), (0, height), "x", 0.0, n // 2, int(Confidence.LOW_CONFIDENCE))
    wall((0, depth), (0, height), "x", width, n // 2, int(Confidence.LOW_CONFIDENCE))
    # Roof: partially observed from above.
    wall((0, width), (0, depth), "z", height, n // 2, int(Confidence.LOW_CONFIDENCE))

    xyz = np.concatenate(pts, axis=0)
    confidence = np.concatenate(conf, axis=0)
    return xyz, confidence


def _make_demo_point_cloud(seed: int = 42) -> PointCloud:
    """Build a synthetic PointCloud: a couple of buildings on a ground plane."""
    rng = np.random.default_rng(seed)

    xyz_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []

    # Ground plane: mostly measured, a patch of low-confidence "vegetation".
    ground_n = 2500
    gx = rng.uniform(-40, 40, size=ground_n)
    gy = rng.uniform(-40, 40, size=ground_n)
    gz = rng.normal(0.0, 0.05, size=ground_n)
    ground = np.stack([gx, gy, gz], axis=1)
    ground_conf = np.full(ground_n, int(Confidence.MEASURED), dtype=np.uint8)

    # Vegetation patch: noisy, low confidence, slightly raised.
    veg_n = 600
    veg_x = rng.uniform(10, 30, size=veg_n)
    veg_y = rng.uniform(-30, -10, size=veg_n)
    veg_z = np.abs(rng.normal(1.5, 1.0, size=veg_n))
    veg = np.stack([veg_x, veg_y, veg_z], axis=1)
    veg_conf = np.full(veg_n, int(Confidence.LOW_CONFIDENCE), dtype=np.uint8)

    xyz_parts.append(ground)
    conf_parts.append(ground_conf)
    xyz_parts.append(veg)
    conf_parts.append(veg_conf)

    b1_xyz, b1_conf = _make_building(
        np.array([-25.0, -5.0, 0.0]), width=14.0, depth=10.0, height=12.0, density=900, rng=rng
    )
    b2_xyz, b2_conf = _make_building(
        np.array([5.0, 5.0, 0.0]), width=10.0, depth=8.0, height=7.0, density=700, rng=rng
    )
    xyz_parts += [b1_xyz, b2_xyz]
    conf_parts += [b1_conf, b2_conf]

    xyz = np.concatenate(xyz_parts, axis=0).astype(np.float64)
    confidence = np.concatenate(conf_parts, axis=0).astype(np.uint8)

    # RGB: green-ish ground/vegetation, grey-ish buildings, tinted by height.
    n_total = xyz.shape[0]
    rgb = np.empty((n_total, 3), dtype=np.uint8)
    height_norm = np.clip((xyz[:, 2] - xyz[:, 2].min()) / max(xyz[:, 2].max() - xyz[:, 2].min(), 1e-6), 0, 1)
    rgb[:, 0] = (120 + 100 * height_norm).astype(np.uint8)
    rgb[:, 1] = (140 + 60 * height_norm).astype(np.uint8)
    rgb[:, 2] = (150 + 40 * (1 - height_norm)).astype(np.uint8)

    return PointCloud(xyz=xyz, rgb=rgb, confidence=confidence)


def _demo_poses(n: int = 24):
    """A plausible lawnmower survey track, as ``Pose`` objects.

    Gives the demo a real flight path and camera frustums in the 3D view
    from the first seconds, the same way a real run gets them out of
    triage -- so "Run Demo" shows the whole journey, not just the final
    cloud.
    """
    from drishti3d.types import Pose

    poses = []
    legs = 3
    per_leg = max(2, n // legs)
    for index in range(n):
        leg = index // per_leg
        along = (index % per_leg) / max(1, per_leg - 1)
        if leg % 2:
            along = 1.0 - along
        x = -40.0 + 80.0 * along
        y = -30.0 + 30.0 * leg
        # Nadir camera: OpenCV +Z (forward) points down -Z in world, +Y
        # (image down) points along world -Y.
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        poses.append(Pose(R=R, t=np.array([x, y, 45.0])))
    return poses


class DemoPipeline:
    """A callable that fakes a full pipeline run with realistic staged progress.

    Matches the ``PipelineCallable`` signature so it can be handed straight
    to ``PipelineWorker``. Produces a synthetic ``PointCloud`` (see
    ``_make_demo_point_cloud``) as its final result and a plain dict of
    accuracy-report-shaped numbers as a partial result, so
    ``report_panel.py`` has something to display without a real fusion
    stage.
    """

    #: number of progress ticks simulated per stage
    steps_per_stage: int = 6
    #: seconds to sleep between ticks (kept short so tests run fast; override in tests)
    tick_seconds: float = 0.05

    def __init__(self, steps_per_stage: int | None = None, tick_seconds: float | None = None) -> None:
        if steps_per_stage is not None:
            self.steps_per_stage = steps_per_stage
        if tick_seconds is not None:
            self.tick_seconds = tick_seconds

    def __call__(
        self,
        report_progress: ProgressFn,
        report_stage: StageFn,
        report_partial: PartialFn,
        cancel_token: CancelToken,
        report_preview: PreviewFn | None = None,
    ) -> dict[str, Any]:
        import time

        total = len(STAGES) * self.steps_per_stage
        step = 0

        for stage in STAGES:
            if cancel_token.is_set():
                raise PipelineCancelled()
            report_stage(stage)
            for i in range(self.steps_per_stage):
                if cancel_token.is_set():
                    raise PipelineCancelled()
                step += 1
                report_progress(step, total, f"{stage}: step {i + 1}/{self.steps_per_stage}")
                if self.tick_seconds:
                    time.sleep(self.tick_seconds)

            if stage == "triage" and report_preview is not None:
                # The demo has to exercise the same live-preview path the
                # real pipeline uses, or the path only ever gets tested
                # by a thirty-minute run.
                report_preview(stage, {"stage": stage, "poses": _demo_poses()})

            if stage == "fusion":
                point_cloud = _make_demo_point_cloud()
                report_partial(point_cloud)
                if report_preview is not None:
                    report_preview(stage, {"stage": stage, "point_cloud": point_cloud})

        point_cloud = _make_demo_point_cloud()

        confidence = point_cloud.confidence
        assert confidence is not None
        n = confidence.shape[0]
        pct_measured = float(np.count_nonzero(confidence == int(Confidence.MEASURED))) / n * 100.0
        pct_low = float(np.count_nonzero(confidence == int(Confidence.LOW_CONFIDENCE))) / n * 100.0
        pct_inferred = float(np.count_nonzero(confidence == int(Confidence.INFERRED))) / n * 100.0

        report = {
            "relative_rmse": 0.021,
            "absolute_rmse_m": 0.34,
            "scale_error_pct": 1.2,
            "reprojection_error_px": 0.87,
            "coverage_pct": 92.5,
            "pct_measured": pct_measured,
            "pct_low_confidence": pct_low,
            "pct_inferred": pct_inferred,
        }

        return {"point_cloud": point_cloud, "report": report}


class RealPipeline:
    """A ``PipelineCallable`` that drives the real pipeline (``pipeline.runner.run_pipeline``).

    Matches ``DemoPipeline``'s calling convention exactly (so
    ``PipelineWorker``/``main_window.py`` don't need to special-case which
    one is running until the *result* comes back) but runs actual
    ingest/triage/geometry/bundle/fusion/export stages over a real video.

    ``backbone`` defaults to ``None`` (defer to ``config.geometry.backbone``,
    which itself defaults to ``"mapanything"`` -- see ``drishti3d.config
    .GeometryConfig``): the Run button should attempt the real
    reconstruction backbone when it's installed and fall back to
    ``NullBackbone`` automatically when it isn't (see
    ``pipeline.stages.GeometryStage``), never fail outright just because a
    heavy optional dependency is missing.
    """

    def __init__(
        self,
        video_path: str,
        telemetry_path: str | None = None,
        config: Config | None = None,
        backbone: str | None = None,
    ) -> None:
        self.video_path = video_path
        self.telemetry_path = telemetry_path or None
        self.config = config
        self.backbone = backbone

    def __call__(
        self,
        report_progress: ProgressFn,
        report_stage: StageFn,
        report_partial: PartialFn,
        cancel_token: CancelToken,
        report_preview: PreviewFn | None = None,
    ) -> PipelineResult:
        # Imported lazily so constructing/importing this module never pays
        # for numpy-heavy pipeline-stage imports unless a real run actually
        # happens (mirrors the lazy-import convention used throughout this
        # codebase, e.g. drishti3d.app.main / drishti3d.device).
        from drishti3d.pipeline.runner import run_pipeline

        last_stage: str | None = None

        def _progress_cb(stage_name: str, current: int, total: int, message: str) -> None:
            nonlocal last_stage
            if stage_name != last_stage:
                last_stage = stage_name
                report_stage(stage_name)
            report_progress(current, total, message)

        # ``cancel_token`` here is a Qt-side ``CancelToken`` (this module's
        # class, above) -- run_pipeline only ever calls ``.is_set()`` on it,
        # so it's usable as-is without adapting to
        # ``pipeline.stages.CancelToken``; see that class's docstring.
        return run_pipeline(
            self.video_path,
            telemetry_path=self.telemetry_path,
            config=self.config,
            progress_cb=_progress_cb,
            cancel_token=cancel_token,
            backbone=self.backbone,
            partial_cb=report_partial,
            preview_cb=report_preview,
        )
