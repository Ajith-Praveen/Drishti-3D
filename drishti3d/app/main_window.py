"""The DRISHTI-3D main window: one screen, no wizard.

Layout
------
::

    ┌────────────────────────────────────────────────────────────┐
    │ ◆ DRISHTI-3D │ video │ telemetry │        ▶ Run  ■ Cancel   │
    ├──────────┬──────────────────────────────────┬──────────────┤
    │ MISSION  │                                  │  INSPECTOR   │
    │  stage   │        3D VIEWPORT               │  Layers      │
    │  rail    │   (flight path → cameras →       │  Settings    │
    │  (live)  │    point cloud → mesh)           │  Report      │
    │          ├──────────────────────────────────┤              │
    │          │ Stage renders │ Log │ Stage data │              │
    ├──────────┴──────────────────────────────────┴──────────────┤
    │ device · stage · elapsed · points · triangles              │
    └────────────────────────────────────────────────────────────┘

Why this replaces the four-page wizard
--------------------------------------
The window used to be a ``QStackedWidget`` of four full-page panels --
Ingest, Process, Workspace, Report -- selected by a tab bar. During a
thirty-minute run that put the three things an operator needs at once
(what stage is running, what the reconstruction looks like so far, and
whether a gate has failed) on three different pages. Worse, the
Workspace page's own docstring already claimed to have *replaced* the
wizard, so the app carried both designs at the same time, with the
viewport owned by one and referenced by the other.

There is now one screen. Nothing the pipeline learns is more than a
glance away, and the 3D view -- the actual product -- is never hidden.

The pipeline graph is gone with it. It presented itself as editable
(drag stages in, wire ports, delete nodes) but ``run_pipeline`` has no
graph parameter and calls a hardcoded ``_build_stages()``; deleting a
node changed nothing about what ran. A control that lies about its
effect is worse than no control, so the stage rail shows the real fixed
order and promises only status, timing and each stage's findings.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import icons, theme
from drishti3d.app.panels.diagnostics import DiagnosticsPanel
from drishti3d.app.panels.layers_panel import LayersPanel
from drishti3d.app.panels.preferences import PreferencesDialog
from drishti3d.app.panels.report_panel import ReportPanel
from drishti3d.app.panels.settings_panel import SettingsPanel
from drishti3d.app.panels.stage_rail import STAGE_SPECS, StageRail
from drishti3d.app.panels.topbar import TopBar
from drishti3d.app.panels.viewer import Viewer
from drishti3d.app.viewport import COLOR_MODES, Viewport
from drishti3d.app.workers import STAGES, DemoPipeline, PipelineWorker, RealPipeline
from drishti3d.device import get_device

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
TELEMETRY_EXTENSIONS = (".csv", ".srt", ".json", ".gpx", ".txt")

#: Fraction of a run each stage typically accounts for, measured on this
#: project's own 4-minute 4K sample flight (results/v22, 2108 s total).
#: The progress bar is weighted by these because a bar fed raw
#: ``current/total`` sweeps 0->100% once per stage -- eleven times a run,
#: hitting 100% after the first half-second stage.
_STAGE_WEIGHTS: dict[str, float] = {
    "ingest": 0.5,
    "triage": 247.0,
    "time_sync": 5.0,
    "drone_path": 0.1,
    "coverage": 0.1,
    "semantics": 1.0,
    "pose_prior": 427.0,
    "frame_placement": 0.1,
    "geometry": 192.0,
    "bundle_adjustment": 1.0,
    "fusion": 1189.0,
    "export": 262.0,
}
_TOTAL_WEIGHT = sum(_STAGE_WEIGHTS.values())


class MainWindow(QMainWindow):
    """Top-level DRISHTI-3D application window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DRISHTI-3D")
        self.setWindowIcon(icons.app_icon())
        self.resize(1560, 960)
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)

        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_result = None
        self._run_started: float | None = None
        self._current_stage: str | None = None
        self._completed_weight = 0.0
        self._closing = False
        self._video_path = ""
        self._telemetry_path = ""
        self._run_dir: Path | None = None
        self._pending_appearance = False

        self._build_ui()
        self._build_menus()
        self._wire_signals()

        self.topbar.set_device(get_device())
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._tick_elapsed)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self.topbar = TopBar()
        column.addWidget(self.topbar)

        self.progress_line = _ProgressLine()
        column.addWidget(self.progress_line)

        # --- left: mission rail -----------------------------------------
        self.stage_rail = StageRail()
        self.stage_rail.setMinimumWidth(210)
        self.stage_rail.setMaximumWidth(360)

        # --- centre: 3D view over the diagnostics drawer ------------------
        self.viewport = Viewport()
        self.viewer = Viewer(self.viewport)
        self.diagnostics = DiagnosticsPanel()

        centre = QSplitter(Qt.Vertical)
        centre.addWidget(self._framed(self.viewer, None))
        centre.addWidget(self._framed(self.diagnostics, None))
        centre.setStretchFactor(0, 4)
        centre.setStretchFactor(1, 2)
        centre.setSizes([620, 280])
        self._centre_splitter = centre

        # --- right: inspector ---------------------------------------------
        self.layers_panel = LayersPanel()
        self.settings_panel = SettingsPanel()
        self.report_panel = ReportPanel()

        self.inspector = QTabWidget()
        self.inspector.setDocumentMode(True)
        self.inspector.addTab(self._scrolled(self.layers_panel), icons.icon("layers", 15), "")
        self.inspector.addTab(self._scrolled(self.settings_panel), icons.icon("settings", 15), "")
        self.inspector.addTab(self.report_panel, icons.icon("report", 15), "")
        self.inspector.setTabToolTip(0, "Layers — what is drawn in the 3D view")
        self.inspector.setTabToolTip(1, "Settings — reconstruction quality and cost")
        self.inspector.setTabToolTip(2, "Report — accuracy and provenance")
        self.inspector.setMinimumWidth(310)
        self.inspector.setMaximumWidth(480)

        root = QSplitter(Qt.Horizontal)
        root.addWidget(self._framed(self.stage_rail, theme.BG_PANEL))
        root.addWidget(centre)
        root.addWidget(self._framed(self.inspector, theme.BG_PANEL))
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setStretchFactor(2, 0)
        root.setSizes([250, 970, 340])
        root.setCollapsible(1, False)
        column.addWidget(root, 1)

        self.setCentralWidget(central)
        self._build_statusbar()

    def _framed(self, widget: QWidget, background: str | None) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        if background:
            frame.setStyleSheet(f"QFrame{{background:{background};}}")
        box = QVBoxLayout(frame)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(widget)
        return frame

    def _scrolled(self, widget: QWidget) -> QWidget:
        from PySide6.QtWidgets import QScrollArea

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setWidget(widget)
        return area

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)

        self.stage_label = QLabel("Idle")
        self.elapsed_label = QLabel("")
        self.elapsed_label.setFont(theme.mono_font(10))
        self.counts_label = QLabel("no reconstruction loaded")
        self.counts_label.setFont(theme.mono_font(10))

        bar.addWidget(self.stage_label)
        bar.addWidget(self.elapsed_label)
        bar.addPermanentWidget(self.counts_label)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        # --- File ---------------------------------------------------------
        file_menu = menu_bar.addMenu("&File")
        self._add_action(file_menu, "Open &Video…", self._browse_video, "Ctrl+O")
        self._add_action(file_menu, "Open &Telemetry…", self._browse_telemetry)
        file_menu.addSeparator()
        self._add_action(file_menu, "Open &Run…", self._load_result, "Ctrl+Shift+O")
        self._add_action(file_menu, "&Save Run…", self._save_result, "Ctrl+S")
        file_menu.addSeparator()
        self._add_action(file_menu, "&Export Model…", self._export_model, "Ctrl+E")
        self._add_action(file_menu, "Export &Screenshot…", self._export_screenshot)
        self._add_action(file_menu, "Reveal Output &Folder", self._reveal_output)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Quit", self.close, QKeySequence.Quit)

        # --- Run -----------------------------------------------------------
        run_menu = menu_bar.addMenu("&Run")
        self.run_action = self._add_action(run_menu, "&Run Reconstruction", self._start_real_pipeline, "Ctrl+R")
        self._add_action(run_menu, "Run &Demo", self._start_demo_pipeline, "Ctrl+D")
        self.cancel_action = self._add_action(run_menu, "&Cancel Run", self._cancel_pipeline, "Ctrl+.")
        self.cancel_action.setEnabled(False)

        # --- View ------------------------------------------------------------
        view_menu = menu_bar.addMenu("&View")

        color_menu = view_menu.addMenu("Colour Mode")
        self.color_mode_group = QActionGroup(self)
        self.color_mode_group.setExclusive(True)
        self._color_mode_actions: dict[str, QAction] = {}
        for index, mode in enumerate(COLOR_MODES):
            action = QAction(mode.capitalize(), self, checkable=True)
            action.setChecked(mode == "rgb")
            action.setShortcut(f"Ctrl+{index + 1}")
            action.triggered.connect(lambda _checked=False, m=mode: self._set_color_mode(m))
            color_menu.addAction(action)
            self.color_mode_group.addAction(action)
            self._color_mode_actions[mode] = action

        camera_menu = view_menu.addMenu("Camera")
        for name, shortcut in (("top", "1"), ("front", "2"), ("side", "3"), ("iso", "4")):
            action = QAction(name.capitalize(), self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _checked=False, n=name: self.viewport.set_view(n))
            camera_menu.addAction(action)
        camera_menu.addSeparator()
        self._add_action(camera_menu, "Fit to Reconstruction", self.viewport.reset_camera, "0")

        view_menu.addSeparator()
        self._add_action(view_menu, "Toggle Diagnostics Drawer", self._toggle_diagnostics, "Ctrl+`")
        self._add_action(view_menu, "Toggle Inspector", self._toggle_inspector, "Ctrl+I")

        # --- Tools --------------------------------------------------------------
        tools_menu = menu_bar.addMenu("&Tools")
        self._add_action(tools_menu, "Measure &Point (coordinates)", lambda: self._start_measure("point"), "P")
        self._add_action(tools_menu, "Measure &Distance", self._measure_distance, "M")
        self._add_action(tools_menu, "Measure &Area", self._measure_area, "Shift+M")
        self._add_action(tools_menu, "Measure &Volume (cut / fill)", lambda: self._start_measure("volume"), "V")
        self._add_action(tools_menu, "Elevation Pro&file", lambda: self._start_measure("profile"), "L")
        self._add_action(tools_menu, "&Export Measurements… (GeoJSON / KML)", self._export_measurements)
        self._add_action(tools_menu, "&Clear Measurements", self._clear_measurements)
        tools_menu.addSeparator()
        # NoRole keeps it under Tools: on macOS Qt otherwise moves any
        # "Preferences…" item into the application menu by text heuristic.
        prefs = QAction("&Preferences…", self)
        prefs.setMenuRole(QAction.NoRole)
        prefs.setShortcut(QKeySequence("Ctrl+,"))
        prefs.triggered.connect(self._open_preferences)
        tools_menu.addAction(prefs)
        self.addAction(prefs)

        # --- Help ------------------------------------------------------------------
        help_menu = menu_bar.addMenu("&Help")
        self._add_action(help_menu, "&About DRISHTI-3D", self._show_about)

    def _add_action(self, menu, text: str, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        # Also add to the window so the shortcut fires when the native
        # macOS menu bar is not the focus owner.
        self.addAction(action)
        return action

    def _wire_signals(self) -> None:
        self.topbar.videoRequested.connect(self._browse_video)
        self.topbar.telemetryRequested.connect(self._browse_telemetry)
        self.topbar.videoCleared.connect(lambda: self._set_video(""))
        self.topbar.telemetryCleared.connect(lambda: self._set_telemetry(""))
        self.topbar.runRequested.connect(self._start_real_pipeline)
        self.topbar.demoRequested.connect(self._start_demo_pipeline)
        self.topbar.cancelRequested.connect(self._cancel_pipeline)

        self.viewer.colorModeChanged.connect(self._set_color_mode)

        self.layers_panel.pathToggled.connect(self.viewport.set_flight_path_visible)
        self.layers_panel.camerasToggled.connect(self.viewport.set_cameras_visible)
        self.layers_panel.pointsToggled.connect(self.viewport.set_point_cloud_visible)
        self.layers_panel.meshToggled.connect(self.viewport.set_mesh_visible)
        self.layers_panel.gridToggled.connect(self.viewport.set_grid_visible)
        self.layers_panel.axesToggled.connect(self.viewport.set_axes_visible)
        self.layers_panel.scaleToggled.connect(self.viewport.set_scale_bar_visible)
        self.layers_panel.measurementsToggled.connect(self.viewport.set_measurements_visible)
        self.layers_panel.pointSizeChanged.connect(self.viewport.set_point_size)

        self.viewport.measurementMade.connect(self._on_measurement_made)
        self.viewport.measurementCompleted.connect(self._on_measurement_completed)
        self.viewport.sceneChanged.connect(lambda: self.layers_panel.sync_from_viewport(self.viewport))

        self.settings_panel.settingsChanged.connect(self._on_settings_changed)
        self.report_panel.exportRequested.connect(
            lambda path: self.diagnostics.append_log(f"report exported to {path}")
        )
        self.stage_rail.stageSelected.connect(self._on_stage_selected)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def telemetry_path(self) -> str:
        return self._telemetry_path

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Drone Video", "", "Video files (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)"
        )
        if path:
            self._set_video(path)

    def _browse_telemetry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Flight Telemetry",
            "",
            "Telemetry (*.csv *.srt *.json *.gpx *.txt);;All files (*)",
        )
        if path:
            self._set_telemetry(path)

    def load_video(self, path: str) -> None:
        """Public entry point used by ``--video`` and drag-and-drop."""
        self._set_video(path)

    def load_telemetry(self, path: str) -> None:
        self._set_telemetry(path)

    def _set_video(self, path: str) -> None:
        self._video_path = path or ""
        self.topbar.set_video(self._video_path, self._probe_video(self._video_path))
        if path:
            self.diagnostics.append_log(f"video: {path}")

    def _set_telemetry(self, path: str) -> None:
        self._telemetry_path = path or ""
        detail = ""
        if path:
            suffix = Path(path).suffix.lstrip(".").upper()
            detail = suffix or ""
        self.topbar.set_telemetry(self._telemetry_path, detail)
        if path:
            self.diagnostics.append_log(f"telemetry: {path}")

    def apply_config_file(self, path: str) -> None:
        """Adopt a pipeline config YAML into the Settings panel.

        ``--config`` used to be parsed and then only logged as "not yet
        wired to the real pipeline", so a config file passed on the
        command line silently did nothing.
        """
        from drishti3d.config import load_config

        try:
            config = load_config(path)
        except Exception as exc:  # noqa: BLE001 - a bad config is a message, not a crash
            QMessageBox.warning(self, "Config not loaded", f"{path}\n\n{exc}")
            return
        self.settings_panel.apply_config(config)
        self.diagnostics.append_log(f"config loaded from {path}")

    @staticmethod
    def _probe_video(path: str) -> str:
        """A one-line summary for the source chip. Never raises."""
        if not path:
            return ""
        try:
            import cv2

            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                capture.release()
                return "unreadable"
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if width <= 0 or height <= 0:
                return "unreadable"
            duration = frames / fps if fps > 0 else 0.0
            return f"{width}×{height} · {fps:.0f} fps · {duration:.0f} s"
        except Exception:  # noqa: BLE001 - a probe must never block loading
            return ""

    # ------------------------------------------------------------------
    # Run control
    # ------------------------------------------------------------------
    def _busy(self) -> bool:
        """True (and complains) if a run is already in progress.

        Checked BEFORE any side effect. The previous version reset the
        node graph and cleared the preview tabs first and only then
        discovered it could not start -- destroying the live run's state
        to tell the operator that nothing happened.
        """
        if self._thread is None:
            return False
        QMessageBox.warning(
            self,
            "Run in progress",
            "A reconstruction is already running.\n\nCancel it first, or wait for it to finish.",
        )
        return True

    def _start_real_pipeline(self) -> None:
        if self._busy():
            return
        if not self._video_path:
            QMessageBox.warning(
                self,
                "No video loaded",
                "Choose a drone video first — click the video chip in the toolbar, "
                "or use File ▸ Open Video.",
            )
            return

        # One stamp for the whole run: the config, the renders panel and
        # Reveal Output Folder must all name the same directory.
        self.settings_panel.begin_run()
        config = self.settings_panel.build_config()
        run_dir = self._preview_run_dir(config)
        try:
            import tempfile

            run_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=run_dir):
                pass
        except OSError as exc:
            QMessageBox.warning(self, "Output folder unavailable", f"Choose a writable output folder in Settings.\n\n{exc}")
            return
        self._run_started = time.time()

        self._reset_for_run()
        self._run_dir = run_dir
        self.diagnostics.set_run_dir(run_dir, epoch=self._run_started)
        self.diagnostics.append_log(f"run started — output: {Path(run_dir).resolve()}")
        if not self._telemetry_path:
            self.diagnostics.append_log(
                "WARNING: no telemetry — output stays in a local frame, not georeferenced"
            )

        self._start_pipeline(
            RealPipeline(
                self._video_path,
                telemetry_path=self._telemetry_path or None,
                config=config,
                telemetry_offset_s=self.settings_panel.telemetry_offset_s,
            )
        )

    def _start_demo_pipeline(self) -> None:
        if self._busy():
            return
        self._run_started = time.time()
        self._reset_for_run()
        self.diagnostics.append_log("demo run started (synthetic; no video, no GPU)")
        self._start_pipeline(DemoPipeline())

    def _reset_for_run(self) -> None:
        self.stage_rail.reset()
        self.stage_rail.set_summary("Starting…", "accent")
        self.diagnostics.reset()
        self.progress_line.set_fraction(0.0)
        self.viewport.clear_scene()
        self.report_panel.set_report({})
        self.layers_panel.sync_from_viewport(self.viewport)
        self._completed_weight = 0.0
        self._current_stage = None
        self.counts_label.setText("no reconstruction loaded")

    @staticmethod
    def _preview_run_dir(config) -> Path:
        """Where a run writes its renders: the parent of ``export.output_dir``.

        ``DronePathStage``/``CoverageStage`` write to ``<run>/preview``
        and the placement checks to ``<run>/placement``, both siblings of
        ``output`` -- see ``pipeline.stages._preview_dir``. Deriving it
        here keeps the panel pointed at whatever the operator configured
        rather than assuming a fixed layout.
        """
        output_dir = Path(getattr(getattr(config, "export", None), "output_dir", "output") or "output")
        parent = output_dir.parent
        return parent if str(parent) not in ("", ".") else output_dir

    def _start_pipeline(self, pipeline_callable) -> None:
        # No parent: a QThread owned by the window is destroyed with it,
        # which is fatal if it is still running. Lifetime is managed by
        # the finished -> deleteLater chain below instead.
        thread = QThread()
        worker = PipelineWorker(pipeline_callable)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.stageChanged.connect(self._on_stage_changed)
        worker.partialResult.connect(self._on_partial_result)
        worker.previewUpdate.connect(self._on_preview_update)
        worker.finished.connect(self._on_pipeline_finished)
        worker.failed.connect(self._on_pipeline_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker

        self.topbar.set_running(True)
        self.run_action.setEnabled(False)
        self.cancel_action.setEnabled(True)
        self._tick_timer.start()

        thread.start()

    def _cancel_pipeline(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.stage_rail.set_summary("Cancelling — finishing the current step…", "warn")
            self.diagnostics.append_log("cancellation requested")

    # ------------------------------------------------------------------
    # Worker signals
    # ------------------------------------------------------------------
    def _on_progress(self, current: int, total: int, message: str) -> None:
        stage = self._current_stage
        if stage is None:
            return
        self.stage_rail.set_progress(stage, current, total, message)

        within = (current / total) if total else 0.0
        weight = _STAGE_WEIGHTS.get(stage, 1.0)
        self.progress_line.set_fraction(
            min(1.0, (self._completed_weight + weight * within) / _TOTAL_WEIGHT)
        )

    def _on_stage_changed(self, stage: str) -> None:
        if self._current_stage and self._current_stage != stage:
            self._completed_weight += _STAGE_WEIGHTS.get(self._current_stage, 1.0)
        self._current_stage = stage
        self.stage_rail.set_running(stage)
        self.stage_label.setText(f"Running: {stage.replace('_', ' ')}")
        self.diagnostics.append_log(f"── {stage} ──")

    def _on_partial_result(self, result: object) -> None:
        from drishti3d.types import PointCloud

        if isinstance(result, PointCloud):
            self.viewport.set_point_cloud(result)
            # A partial has no photo colour yet: show it as a heatmap of how
            # well each spot is measured (metres when the stereo measured it)
            # as it grows -- but only switch away from plain RGB, never
            # override a mode the operator picked.
            if result.rgb is None and self.viewport.color_mode() == "rgb":
                if result.uncertainty_m is not None:
                    self._set_color_mode("uncertainty")
                elif result.confidence is not None:
                    self._set_color_mode("confidence")
            self._update_counts()

    def _on_preview_update(self, stage: str, snapshot: object) -> None:
        """Draw whatever the pipeline has learned, the moment it learns it.

        This is the live preview. ``snapshot`` is
        ``pipeline.runner._preview_snapshot``'s dict -- references, not
        copies -- so the drone track appears seconds into a run, refined
        poses and the sparse triangulated cloud minutes before geometry
        finishes, and the fused surface as soon as fusion extracts it.
        """
        if not isinstance(snapshot, dict):
            return

        datum = snapshot.get("height_datum_m")
        if datum is not None:
            self.viewport.set_height_datum(datum, "Elevation (m)")

        poses = snapshot.get("poses")
        if poses:
            centres = np.array([np.asarray(p.t, dtype=np.float64).reshape(3) for p in poses])
            self.viewport.set_flight_path(centres)
            self.viewport.set_cameras(poses, intrinsics=snapshot.get("intrinsics"))

        sparse = snapshot.get("sparse_points")
        if sparse is not None and snapshot.get("point_cloud") is None:
            from drishti3d.types import PointCloud

            self.viewport.set_point_cloud(PointCloud(xyz=np.asarray(sparse, dtype=np.float64)))

        point_cloud = snapshot.get("point_cloud")
        faces = snapshot.get("mesh_faces")
        if point_cloud is not None:
            self.viewport.set_point_cloud(point_cloud)
            if faces is not None and len(faces):
                self.viewport.set_mesh(
                    point_cloud.xyz,
                    faces,
                    point_cloud.rgb,
                    confidence=point_cloud.confidence,
                    uncertainty_m=point_cloud.uncertainty_m,
                )
                # The surface supersedes its own vertices; showing both
                # just makes the mesh look noisy.
                self.viewport.set_point_cloud_visible(False)
                self.layers_panel._rows["points"].check.setChecked(False)

        artifacts = snapshot.get("artifacts")
        if artifacts:
            self.diagnostics.set_stage_artifacts(stage, artifacts)
            self._flag_failed_gates(stage, artifacts)

        self._update_counts()
        self.layers_panel.sync_from_viewport(self.viewport)

    def _flag_failed_gates(self, stage: str, artifacts: dict) -> None:
        """Surface a gate that failed, instead of burying it in a dict.

        The placement check exists precisely to catch a run whose windows
        disagree about where the ground is, and on this project's own
        sample flight it FAILED by 18 m while the app said nothing.
        """
        placement = artifacts.get("placement")
        if isinstance(placement, dict) and placement.get("verdict", "FAIL" if placement.get("passed") is False else "PASS") == "FAIL":
            reason = placement.get("reason") or "windows disagree on the ground plane"
            self.diagnostics.append_log(f"GATE FAILED [{stage}]: {reason}")
            self.stage_rail.set_summary(f"Placement check failed: {reason}", "warn")

        holes = artifacts.get("hole_cells")
        if isinstance(holes, (int, float)) and holes > 0:
            area = artifacts.get("hole_area_ha")
            suffix = f" ({area:.2f} ha)" if isinstance(area, (int, float)) else ""
            self.diagnostics.append_log(f"coverage holes: {int(holes)} cells{suffix}")

    def _on_pipeline_finished(self, result: object) -> None:
        self._tick_timer.stop()
        self.diagnostics.stop()

        from drishti3d.pipeline.result import PipelineResult
        from drishti3d.types import PointCloud

        if isinstance(result, PipelineResult):
            self._last_result = result
            self._apply_result(result)
        elif isinstance(result, dict):
            point_cloud = result.get("point_cloud")
            if isinstance(point_cloud, PointCloud):
                self.viewport.set_point_cloud(point_cloud, reset_camera=True)
            report = result.get("report")
            if isinstance(report, dict):
                self.report_panel.set_report(report)

        self.progress_line.set_fraction(1.0)
        self._finish_ui(result.outcome_label if isinstance(result, PipelineResult) else "Unverified — diagnostic output")
        self._update_counts()
        self.layers_panel.sync_from_viewport(self.viewport)

    def _apply_result(self, result) -> None:
        """Show everything a completed (or partial) run produced."""
        for stage_result in result.stage_results or []:
            self.stage_rail.apply_stage_result(
                stage_result.name, stage_result.status, stage_result.message, stage_result.elapsed_s
            )
            self.diagnostics.set_stage_artifacts(stage_result.name, stage_result.artifacts)
            first_line = stage_result.message if stage_result.status == "failed" else (
                stage_result.message.splitlines()[0] if stage_result.message else ""
            )
            self.diagnostics.append_log(
                f"[{stage_result.name}] {stage_result.status}" + (f": {first_line}" if first_line else "")
            )

        datum = _result_height_datum(result)
        self.viewport.set_height_datum(datum if datum is not None else 0.0, "Elevation (m)")

        if result.poses:
            centres = np.array([np.asarray(p.t, dtype=np.float64).reshape(3) for p in result.poses])
            self.viewport.set_flight_path(centres)
            intrinsics = next(
                (kf.intrinsics for kf in (result.keyframes or []) if kf.intrinsics is not None), None
            )
            self.viewport.set_cameras(result.poses, intrinsics=intrinsics)

        if result.point_cloud is not None:
            self.viewport.set_point_cloud(result.point_cloud, reset_camera=True)
            faces = getattr(result, "mesh_faces", None)
            if faces is not None and len(faces):
                pc = result.point_cloud
                self.viewport.set_mesh(
                    pc.xyz, faces, pc.rgb, confidence=pc.confidence, uncertainty_m=pc.uncertainty_m
                )
                self.viewport.set_point_cloud_visible(False)
                self.layers_panel._rows["points"].check.setChecked(False)

        # ``report_card()`` prefers the card ExportStage published and
        # falls back to the stage that measured each number, so a run
        # saved before the card existed -- or one cancelled before
        # export -- still shows real figures rather than a blank table.
        self.report_panel.set_report(result.report_card())
        self.stage_label.setText(result.outcome_label)
        self.stage_rail.set_summary(result.outcome_label, "ok" if result.outcome == "valid" else "warn")
        if (result.report or {}).get("cancelled"):
            self.diagnostics.append_log("run cancelled by the operator; showing the partial result")

    def _on_pipeline_failed(self, message: str) -> None:
        self._tick_timer.stop()
        self.diagnostics.stop()
        self.stage_rail.fail(self._current_stage, message)
        self.diagnostics.append_log(f"ERROR: {message.splitlines()[-1] if message else 'failed'}")
        self.report_panel.set_report({"outcome": "failed", "diagnostic_output": True,
                                      "quality_reasons": ["Reconstruction failed; retained geometry is diagnostic only."]})
        self._finish_ui("Failed — diagnostic output")
        if not self._closing:
            QMessageBox.critical(
                self,
                "Reconstruction failed",
                (message.strip().splitlines() or ["Unknown error"])[-1]
                + "\n\nThe full traceback is in the Log tab.",
            )

    def _finish_ui(self, status: str) -> None:
        elapsed = self._elapsed_text()
        self.stage_label.setText(status)
        if status == "Valid":
            self.stage_rail.finish(f"Valid — completed in {elapsed}" if elapsed else "Valid")
        else:
            self.stage_rail.finish(status, successful=False)
            self.stage_rail.set_summary(status, "error" if status.startswith("Failed") else "warn")
        self.topbar.set_running(False)
        self.run_action.setEnabled(bool(self._video_path))
        self.cancel_action.setEnabled(False)
        self._current_stage = None

    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        if self._closing:
            self.close()
            return
        if self._pending_appearance:
            QTimer.singleShot(0, self._rebuild_for_appearance)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _elapsed_text(self) -> str:
        if self._run_started is None:
            return ""
        seconds = int(time.time() - self._run_started)
        minutes, rest = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{rest:02d}" if hours else f"{minutes}:{rest:02d}"

    def _tick_elapsed(self) -> None:
        self.elapsed_label.setText(self._elapsed_text())

    def _update_counts(self) -> None:
        points = self.viewport.point_count()
        faces = self.viewport.mesh_face_count()
        cameras = self.viewport.camera_count()
        if not (points or faces or cameras):
            self.counts_label.setText("no reconstruction loaded")
            return
        self.counts_label.setText(
            f"{cameras:,} cameras   {points:,} points   {faces:,} triangles"
        )

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------
    def _save_result(self) -> None:
        if self._last_result is None:
            QMessageBox.information(
                self, "Save Run", "No reconstruction to save yet — run the pipeline first."
            )
            return
        path = QFileDialog.getExistingDirectory(self, "Save Run To Directory")
        if not path:
            return
        try:
            self._last_result.save(path)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.diagnostics.append_log(f"run saved to {path}")
        self.statusBar().showMessage(f"Saved run to {path}", 5000)

    def _load_result(self, path: str | None = None) -> None:
        """Open a run previously written by Save Run — instant, no GPU. ``path`` skips the folder dialog."""
        if self._busy():
            return
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Open Run Directory")
        if not path:
            return

        from drishti3d.pipeline.result import PipelineResult

        try:
            result = PipelineResult.load(path)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Open failed",
                f"{exc}\n\nChoose the directory that contains meta.json and arrays.npz.",
            )
            return

        self._last_result = result
        self._run_started = None
        self.elapsed_label.setText("")
        self._reset_for_run()

        self._set_video(result.video_path or "")
        self._set_telemetry(result.telemetry_path or "")

        # Point the renders at the loaded run. epoch=None: the run is
        # finished, so everything already on disk belongs to it.
        run_dir = self._preview_run_dir(result.config) if result.config else Path(path)
        if not (Path(run_dir) / "preview").exists():
            run_dir = Path(path)
        self._run_dir = run_dir
        self.diagnostics.set_run_dir(run_dir, epoch=None)

        self._apply_result(result)
        self._finish_ui(result.outcome_label)
        self.progress_line.set_fraction(1.0)
        self._update_counts()
        self.layers_panel.sync_from_viewport(self.viewport)
        self.stage_rail.set_summary(f"{result.outcome_label} — {Path(path).name}", "ok" if result.outcome == "valid" else "warn")
        self.diagnostics.append_log(f"opened saved run {path}")

    def _export_model(self) -> None:
        """Write the loaded reconstruction in a chosen format."""
        if self._last_result is None or self._last_result.point_cloud is None:
            QMessageBox.information(
                self, "Export Model", "No reconstruction to export yet — run the pipeline first."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            "model.ply" if self._last_result.outcome == "valid" else "model.diagnostic.ply",
            "PLY point cloud (*.ply);;LAS point cloud (*.las);;OBJ mesh (*.obj);;glTF binary (*.glb);;XYZ text (*.xyz)",
        )
        if not path:
            return

        from drishti3d.export import formats

        path = Path(path)
        if self._last_result.outcome != "valid" and not path.stem.endswith(".diagnostic"):
            path = path.with_name(f"{path.stem}.diagnostic{path.suffix}")

        point_cloud = self._last_result.point_cloud
        faces = getattr(self._last_result, "mesh_faces", None)
        has_mesh = faces is not None and len(faces) > 0
        payload = (point_cloud.xyz, faces, point_cloud.rgb, point_cloud.confidence) if has_mesh else point_cloud
        ext = path.suffix.lower()
        if ext == ".obj" and not has_mesh:
            QMessageBox.information(
                self, "Export Model", "OBJ is a mesh format and this run has no mesh -- choose PLY, LAS or XYZ."
            )
            return
        try:
            # Each writer has its own signature: PLY takes the mesh bundle,
            # LAS/XYZ the point cloud, OBJ/GLB vertices and faces separately
            # (passing the bundle to those two failed for every run).
            if ext == ".ply":
                formats.export_ply(path, payload)
            elif ext == ".las":
                formats.export_las(path, point_cloud)
            elif ext == ".xyz":
                formats.export_xyz(path, point_cloud)
            elif ext == ".obj":
                formats.export_obj(path, point_cloud.xyz, faces, colors=point_cloud.rgb)
            elif ext == ".glb":
                formats.export_glb(
                    path, point_cloud.xyz, faces=faces if has_mesh else None,
                    colors=point_cloud.rgb, confidence=point_cloud.confidence,
                )
            else:
                raise ValueError(f"unsupported export format {ext!r} -- use .ply, .las, .obj, .glb or .xyz")
            import json

            path.with_suffix(path.suffix + ".quality.json").write_text(json.dumps(self._last_result.quality, indent=2))
        except Exception as exc:  # noqa: BLE001 - report, never crash the app
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.diagnostics.append_log(f"exported model to {path}")
        self.statusBar().showMessage(f"Exported to {path}", 5000)

    def _export_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Screenshot", "drishti3d_view.png", "PNG image (*.png)"
        )
        if not path:
            return
        self.viewport.screenshot(path)
        self.diagnostics.append_log(f"screenshot written to {path}")
        self.statusBar().showMessage(f"Screenshot written to {path}", 5000)

    def _reveal_output(self) -> None:
        """Open the run's output directory in the platform file manager."""
        import subprocess
        import sys

        config = self.settings_panel.build_config()
        target = Path(self._preview_run_dir(config)).resolve()
        if not target.exists():
            QMessageBox.information(
                self, "Output folder", f"Nothing written yet.\n\nIt will appear at:\n{target}"
            )
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        elif sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About DRISHTI-3D",
            "<b>DRISHTI-3D</b><br><br>"
            "Turns a single-pass drone video into a georeferenced, measurable "
            "3D model.<br><br>"
            "Feed-forward geometry anchored to flight telemetry, then refined — "
            "not classical structure-from-motion.<br><br>"
            "Fully offline. Native desktop (PySide6 + VTK). No network calls.",
        )

    # ------------------------------------------------------------------
    # View / tools
    # ------------------------------------------------------------------
    def _set_color_mode(self, mode: str) -> None:
        self.viewer.set_color_mode(mode)
        action = self._color_mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)

    def _toggle_diagnostics(self) -> None:
        sizes = self._centre_splitter.sizes()
        if sizes[1] > 0:
            self._drawer_height = sizes[1]
            self._centre_splitter.setSizes([sizes[0] + sizes[1], 0])
        else:
            restore = getattr(self, "_drawer_height", 280)
            self._centre_splitter.setSizes([max(100, sizes[0] - restore), restore])

    def _toggle_inspector(self) -> None:
        self.inspector.parentWidget().setVisible(not self.inspector.parentWidget().isVisible())

    _MEASURE_HINTS = {
        "point": "Click a point on the model to read its coordinates and elevation.",
        "distance": "Click two points on the model to measure a distance.",
        "area": "Click the corners of an area; right-click, double-click or Enter closes it. Esc cancels.",
        "volume": "Click around a stockpile, pit or debris; right-click or Enter closes the outline. "
                  "The base is the plane through your corners.",
        "profile": "Click along a route; right-click or Enter finishes the elevation profile. Backspace undoes a point.",
    }

    def _start_measure(self, kind: str) -> None:
        self.viewport.start_measurement(kind)
        self.statusBar().showMessage(self._MEASURE_HINTS[kind], 12000)
        self.layers_panel.sync_from_viewport(self.viewport)

    def _measure_distance(self) -> None:
        self._start_measure("distance")

    def _measure_area(self) -> None:
        self._start_measure("area")

    def _clear_measurements(self) -> None:
        self.viewport.clear_measurements()
        self.layers_panel.sync_from_viewport(self.viewport)

    def _on_measurement_made(self, kind: str, value: float, warning: str) -> None:
        """Failures and warnings; finished readings are reported by ``_on_measurement_completed``."""
        if not np.isfinite(value):
            self.statusBar().showMessage(f"{kind.capitalize()} not measured: {warning}", 10000)
            self.diagnostics.append_log(f"{kind} not measured: {warning}")
            return
        if warning:
            self.diagnostics.append_log(f"WARNING: {warning}")
            QMessageBox.warning(self, "Measurement on inferred geometry", warning)

    def _geo_origin(self):
        """The run's local-frame origin as a GeoPoint, or None if it was never georeferenced."""
        from drishti3d.types import GeoPoint

        result = self._last_result
        origin = (getattr(result, "report", None) or {}).get("georef_origin") if result is not None else None
        if not isinstance(origin, dict):
            out_dir = getattr(getattr(getattr(result, "config", None), "export", None), "output_dir", None)
            try:
                import json

                origin = json.loads((Path(out_dir) / "georef.json").read_text())["origin"] if out_dir else None
            except (OSError, KeyError, TypeError, ValueError):
                origin = None
        if not isinstance(origin, dict):
            return None
        return GeoPoint(lat=float(origin["lat"]), lon=float(origin["lon"]), alt_msl=float(origin["alt_msl"]))

    def _on_measurement_completed(self, m) -> None:
        from drishti3d.app.viewport import _measure_text

        message = f"{m.label}: {_measure_text(m)}"
        origin = self._geo_origin()
        if m.kind == "point" and origin is not None:
            from drishti3d.geometry.georef import enu_to_wgs84

            lon, lat, alt = enu_to_wgs84(m.points[:1], origin)[0]
            m.extra.update({"lat": float(lat), "lon": float(lon)})
            message = f"{m.label}: {lat:.7f}, {lon:.7f}, elevation {alt:,.1f} m"
        self.diagnostics.append_log(message)
        self.statusBar().showMessage(message, 15000)
        self.layers_panel.sync_from_viewport(self.viewport)
        if m.kind == "profile":
            from drishti3d.app.panels.profile_dialog import ProfileDialog

            ProfileDialog(m, datum_m=self.viewport._height_datum_m, parent=self).show()

    def _export_measurements(self) -> None:
        measurements = self.viewport.measurements()
        if not measurements:
            QMessageBox.information(self, "Export Measurements", "Take a measurement first (Tools menu).")
            return
        origin = self._geo_origin()
        if origin is None:
            QMessageBox.information(
                self, "Export Measurements",
                "This run has no georeferencing (no GPS), so its measurements cannot be placed on a map.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Measurements", "measurements.geojson", "GeoJSON (*.geojson);;KML — Google Earth (*.kml)"
        )
        if not path:
            return
        from drishti3d.app.measure import write_measurements

        if Path(path).suffix.lower() not in (".geojson", ".json", ".kml"):
            path += ".geojson"
        try:
            write_measurements(path, measurements, origin)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.diagnostics.append_log(f"exported {len(measurements)} measurement(s) to {path}")
        self.statusBar().showMessage(f"Exported {len(measurements)} measurement(s) to {path}", 8000)

    def _on_settings_changed(self) -> None:
        config = self.settings_panel.build_config()
        self.diagnostics.append_log(
            f"settings: {config.quality_profile} profile, "
            f"{config.geometry.max_image_size}px, "
            f"{config.geometry.window_size} frames/window, "
            f"{config.fusion.max_mesh_faces / 1e6:g}M faces"
        )

    def _on_stage_selected(self, stage: str) -> None:
        """Clicking a stage in the rail shows what it measured."""
        self.diagnostics.tabs.setCurrentIndex(2)
        from PySide6.QtCore import Qt as _Qt

        matches = self.diagnostics.artifacts.findItems(stage, _Qt.MatchExactly, 0)
        if matches:
            item = matches[0]
            item.setExpanded(True)
            self.diagnostics.artifacts.scrollToItem(item)
            self.diagnostics.artifacts.setCurrentItem(item)

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(
                    VIDEO_EXTENSIONS + TELEMETRY_EXTENSIONS
                ):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        accepted = False
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            lowered = path.lower()
            if lowered.endswith(VIDEO_EXTENSIONS):
                self._set_video(path)
                accepted = True
            elif lowered.endswith(TELEMETRY_EXTENSIONS):
                self._set_telemetry(path)
                accepted = True
        if accepted:
            event.acceptProposedAction()
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Appearance (Tools > Preferences)
    # ------------------------------------------------------------------
    def _open_preferences(self) -> None:
        dialog = PreferencesDialog(self, run_in_progress=self._thread is not None)
        dialog.appearanceApplied.connect(self._apply_appearance)
        dialog.exec()
        dialog.deleteLater()

    def _apply_appearance(self, palette: str, typeface: str) -> None:
        """Adopt and save a theme/typeface, then re-theme the window.

        Many panels bake colours into per-widget stylesheets when they are
        built, so re-theming means rebuilding the window's contents. That
        is never done under a live run -- the worker thread holds
        references to these widgets through its signal connections -- so
        during a run the choice is saved and applied when it finishes.
        """
        app = QApplication.instance()
        theme.apply_theme(app, palette, typeface)
        theme.save_preferences()
        app.setWindowIcon(icons.app_icon())
        self.setWindowIcon(icons.app_icon())
        if self._thread is not None:
            self._pending_appearance = True
            self.statusBar().showMessage("Theme saved — the window re-themes when this run finishes.", 8000)
            return
        self._rebuild_for_appearance()

    def _capture_state(self) -> dict:
        sp = self.settings_panel
        return {
            "video": self._video_path,
            "telemetry": self._telemetry_path,
            "log": self.diagnostics.log_view.toPlainText(),
            "inspector_tab": self.inspector.currentIndex(),
            "inspector_visible": not self.inspector.parentWidget().isHidden(),
            "centre_sizes": self._centre_splitter.sizes(),
            "color_mode": next(
                (m for m, a in self._color_mode_actions.items() if a.isChecked()), "rgb"
            ),
            "settings": {
                "resolution": sp.resolution_combo.currentIndex(),
                "window": sp.window_spin.value(),
                "profile": sp.profile_combo.currentIndex(),
                "reconstruction": sp.reconstruction_combo.currentData(),
                "mesh": sp.mesh_combo.currentIndex(),
                "semantics": sp.semantics_check.isChecked(),
                "output": sp.output_edit.text(),
                "timestamp": sp.timestamp_check.isChecked(),
            },
        }

    def _restore_state(self, state: dict) -> None:
        sp = self.settings_panel
        s = state["settings"]
        sp.resolution_combo.setCurrentIndex(s["resolution"])
        sp.window_spin.setValue(s["window"])
        sp.profile_combo.setCurrentIndex(s["profile"])
        mode_index = sp.reconstruction_combo.findData(s.get("reconstruction", "auto"))
        sp.reconstruction_combo.setCurrentIndex(max(0, mode_index))
        sp.mesh_combo.setCurrentIndex(s["mesh"])
        sp.semantics_check.setChecked(s["semantics"])
        sp.output_edit.setText(s["output"])
        sp.timestamp_check.setChecked(s["timestamp"])

        # Paths first, without re-logging them.
        self._video_path = state["video"]
        self.topbar.set_video(self._video_path, self._probe_video(self._video_path))
        self._telemetry_path = state["telemetry"]
        self.topbar.set_telemetry(
            self._telemetry_path,
            Path(self._telemetry_path).suffix.lstrip(".").upper() if self._telemetry_path else "",
        )
        self.run_action.setEnabled(bool(self._video_path))

        if self._last_result is not None:
            self._apply_result(self._last_result)
            self._update_counts()
            self.layers_panel.sync_from_viewport(self.viewport)
            self.stage_label.setText(self._last_result.outcome_label)
        if self._run_dir is not None:
            self.diagnostics.set_run_dir(self._run_dir, epoch=None)

        # The log last, so the replayed result lines above are replaced by
        # the original history rather than appended to it.
        self.diagnostics.clear_log()
        if state["log"]:
            self.diagnostics.log_view.setPlainText(state["log"])
            bar = self.diagnostics.log_view.verticalScrollBar()
            bar.setValue(bar.maximum())

        self.inspector.setCurrentIndex(state["inspector_tab"])
        self.inspector.parentWidget().setVisible(state["inspector_visible"])
        if state["centre_sizes"] and state["centre_sizes"][1] == 0:
            self._centre_splitter.setSizes([sum(state["centre_sizes"]), 0])
        if state["color_mode"] != "rgb":
            self._set_color_mode(state["color_mode"])

    def _rebuild_for_appearance(self) -> None:
        """Tear down and rebuild the window contents in the active theme."""
        self._pending_appearance = False
        if self._thread is not None:  # a run started in the meantime
            self._pending_appearance = True
            return
        state = self._capture_state()

        self.diagnostics.stop()
        self.viewport.close()
        for action in self.actions():
            self.removeAction(action)
            action.deleteLater()
        self.color_mode_group.deleteLater()
        self.menuBar().clear()

        self._build_ui()  # replaces (and deletes) the old central widget and status bar
        self._build_menus()
        self._wire_signals()
        self.topbar.set_device(get_device())
        self._restore_state(state)
        # Like at startup, VTK needs a realised window before it renders.
        QTimer.singleShot(0, self.viewport.start)
        self.statusBar().showMessage(
            f"Theme: {theme.PALETTES[theme.current_palette()].name} · "
            f"Type: {theme.TYPEFACES[theme.current_typeface()].name}",
            5000,
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Never destroy a running worker thread.

        ``QThread.quit()`` only asks the thread's event loop to exit, and
        the worker's ``run()`` slot is executing *inside* that loop -- so
        quit() cannot take effect until the current stage returns, which
        for geometry or fusion is minutes. The previous implementation
        waited two seconds and then accepted the close regardless, which
        destroyed a live QThread and aborted the process.

        So: request cancellation, refuse the close, and let
        ``_on_thread_finished`` re-issue it once the thread is genuinely
        done.
        """
        if self._thread is not None and self._thread.isRunning():
            if not self._closing:
                self._closing = True
                if self._worker is not None:
                    self._worker.cancel()
                self.stage_rail.set_summary(
                    "Closing — waiting for the current step to finish…", "warn"
                )
                self.statusBar().showMessage(
                    "Stopping the run… the window will close when the current step finishes."
                )
            event.ignore()
            return

        self.diagnostics.stop()
        self._tick_timer.stop()
        self.viewport.close()
        super().closeEvent(event)


class _ProgressLine(QFrame):
    """A 3 px run-progress rule directly under the command bar.

    Weighted by measured per-stage cost (see ``_STAGE_WEIGHTS``) rather
    than fed each stage's own counter, because a bar driven by raw
    ``current/total`` sweeps 0->100% once per stage -- eleven times in a
    run, and reaching 100% after the first half-second stage.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(3)
        self.setStyleSheet(f"background:{theme.BG_DARK};")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self._fill = QFrame()
        self._fill.setStyleSheet(f"background:{theme.ACCENT};")
        self._rest = QFrame()
        self._rest.setStyleSheet("background:transparent;")
        row.addWidget(self._fill, 0)
        row.addWidget(self._rest, 1000)

    def set_fraction(self, fraction: float) -> None:
        filled = max(0, min(1000, int(round(fraction * 1000))))
        layout = self.layout()
        layout.setStretch(0, filled)
        layout.setStretch(1, 1000 - filled)


def _result_height_datum(result) -> float | None:
    """The local frame's origin altitude for a finished or saved run (heights shown as elevations)."""
    import json

    origin = (getattr(result, "report", None) or {}).get("georef_origin")
    if isinstance(origin, dict) and origin.get("alt_msl") is not None:
        return float(origin["alt_msl"])
    # Runs saved before the report carried it: the export's sidecar.
    out_dir = getattr(getattr(getattr(result, "config", None), "export", None), "output_dir", None)
    for candidate in [Path(out_dir) / "georef.json"] if out_dir else []:
        try:
            return float(json.loads(candidate.read_text())["origin"]["alt_msl"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
    for kf in getattr(result, "keyframes", None) or []:
        geo = getattr(getattr(kf, "telemetry", None), "geo", None)
        if geo is not None and geo.alt_msl is not None:
            return float(geo.alt_msl)
    return None
