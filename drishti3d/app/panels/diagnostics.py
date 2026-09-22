"""The diagnostics drawer: stage renders, the run log, and stage artifacts.

Three tabs along the bottom of the window, because all three answer
questions you ask *while looking at the 3D view*, not instead of it.

**Renders** -- the PNGs the pipeline writes as it goes: the flight path,
the ground the camera saw, where each frame landed, and whether the
reconstructed windows agree on where the ground is. These are the cheap
gates in front of the expensive stages; they existed before this panel
but were written to a folder nobody opened during a run.

Two correctness rules this panel did not previously obey:

1. *Nine renders, not six.* The pipeline writes nine PNGs; the old list
   hardcoded six. The worst omission was ``placement_elevation_y.png``:
   the placement check writes the X-Z and Y-Z cuts as a matched pair
   precisely because every top-down view is blind to vertical error by
   construction, and showing only one cut hides cross-track disagreement.

2. *Never show the previous run's answer as this one's.* The GUI always
   writes to the same ``output/`` directory and nothing clears it, so a
   new run would immediately display the last run's flight path and
   placement verdict as live -- for the fourteen minutes before the
   current run overwrites them. Each tab now refuses any file older than
   the run epoch and says "waiting" instead.

**Log** -- the run's own messages, rate-limited. Triage alone emits one
progress message per scanned frame (7024 on this project's sample
flight), which buried every line that mattered; only stage transitions
and changed messages are appended now.

**Artifacts** -- the measured values each stage returned. Eleven stages
produce 277 of them (coverage holes, per-window depth-anchor ratios, the
placement verdict, the list of 19 written deliverables) and the app used
to discard every one.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import theme

logger = logging.getLogger(__name__)

__all__ = ["DiagnosticsPanel", "PreviewPanel"]

#: (tab label, PNG path relative to the run dir, companion JSON or "").
#: Ordered the way the pipeline produces them, so the tab strip doubles
#: as a progress indicator.
_RENDERS: tuple[tuple[str, str, str], ...] = (
    ("Flight path", "preview/1_drone_path.png", "preview/1_drone_path.json"),
    ("Coverage", "preview/2_coverage.png", "preview/2_coverage.json"),
    ("Footprints", "placement/frames_topdown.png", "placement/frame_placement.json"),
    ("Frame elevation", "placement/frames_elevation.png", ""),
    ("Frame placement", "placement/frame_placement.png", "placement/frame_placement.json"),
    ("Placement check", "placement/placement_check.png", "placement/placement.json"),
    ("Windows top-down", "placement/placement_topdown.png", ""),
    ("Elevation X-Z", "placement/placement_elevation_x.png", ""),
    ("Elevation Y-Z", "placement/placement_elevation_y.png", ""),
)

#: How often to look for new renders. Slow enough to be free, fast enough
#: that a stage taking seconds still feels live.
_POLL_MS = 1200

#: Log lines retained. Enough to hold a whole run's stage transitions.
_LOG_LIMIT = 4000


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


class _RenderTab(QScrollArea):
    """One preview image plus the metrics its companion JSON carries."""

    def __init__(self, image_path: Path, json_path: Path | None) -> None:
        super().__init__()
        self.image_path = image_path
        self.json_path = json_path
        self._mtime: float | None = None
        self._json_mtime: float | None = None

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setStyleSheet(f"background:{theme.BG_VOID};")

        body = QWidget()
        body.setStyleSheet(f"background:{theme.BG_VOID};")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.metrics = QLabel("")
        self.metrics.setWordWrap(True)
        self.metrics.setFont(theme.mono_font(10))
        self.metrics.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; background:{theme.BG_PANEL};"
            f" border-bottom:1px solid {theme.BORDER}; padding:6px 10px;"
        )
        self.metrics.setVisible(False)
        layout.addWidget(self.metrics)

        self.image = QLabel("waiting for this stage…")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet(f"color:{theme.TEXT_FAINT}; background:{theme.BG_VOID}; padding:24px;")
        layout.addWidget(self.image, 1)

        self.setWidget(body)

    def refresh(self, epoch: float | None) -> bool:
        """Reload if changed. Returns whether a current image is shown.

        ``epoch`` is the wall-clock time the run started; a file written
        before it belongs to a previous run and is ignored.
        """
        try:
            if not self.image_path.exists():
                return False
            mtime = self.image_path.stat().st_mtime
            if epoch is not None and mtime < epoch:
                return False  # a previous run's render; not ours to show
            if self._mtime is not None and mtime == self._mtime:
                self._refresh_metrics(epoch)
                return True

            pixmap = QPixmap(str(self.image_path))
            if pixmap.isNull():
                # Half-written file: leave the previous image up and try
                # again on the next tick rather than blanking the view.
                return self._mtime is not None

            self._mtime = mtime
            self.image.setPixmap(pixmap)
            self.image.setStyleSheet(f"background:{theme.BG_VOID};")
            self._refresh_metrics(epoch)
            return True
        except OSError:
            return self._mtime is not None

    def _refresh_metrics(self, epoch: float | None) -> None:
        """Show the numbers behind the picture, from the companion JSON."""
        if self.json_path is None or not self.json_path.exists():
            return
        try:
            mtime = self.json_path.stat().st_mtime
            if epoch is not None and mtime < epoch:
                return
            if self._json_mtime is not None and mtime == self._json_mtime:
                return
            with self.json_path.open() as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return

        self._json_mtime = mtime
        if not isinstance(data, dict):
            return

        parts = [
            f"{key}={_format_scalar(value)}"
            for key, value in data.items()
            if isinstance(value, (int, float, str, bool))
        ]
        if parts:
            self.metrics.setText("   ".join(parts))
            self.metrics.setVisible(True)


class DiagnosticsPanel(QWidget):
    """Renders / Log / Artifacts, as a bottom drawer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run_dir: Path | None = None
        self._epoch: float | None = None
        self._last_log_line = ""
        self._showing_renders = False

        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background:{theme.BG_PANEL};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs)

        # --- renders ------------------------------------------------------
        self.renders = QTabWidget()
        self.renders.setDocumentMode(True)
        self.renders.setTabPosition(QTabWidget.South)

        self.render_placeholder = QLabel()
        self.render_placeholder.setAlignment(Qt.AlignCenter)
        self.render_placeholder.setWordWrap(True)
        self.render_placeholder.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; background:{theme.BG_VOID}; padding:24px;"
        )

        render_page = QWidget()
        render_layout = QVBoxLayout(render_page)
        render_layout.setContentsMargins(0, 0, 0, 0)
        render_layout.setSpacing(0)
        render_layout.addWidget(self.renders, 1)
        render_layout.addWidget(self.render_placeholder, 1)
        self.renders.hide()
        self._set_placeholder_idle()
        self.tabs.addTab(render_page, "Stage renders")

        # --- log ------------------------------------------------------------
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(_LOG_LIMIT)
        self.log_view.setFont(theme.mono_font(10))
        self.log_view.setStyleSheet(
            f"background:{theme.BG_VOID}; border:none; color:{theme.TEXT_SECONDARY};"
        )
        self.tabs.addTab(self.log_view, "Log")

        # --- artifacts ---------------------------------------------------------
        self.artifacts = QTreeWidget()
        self.artifacts.setHeaderLabels(["Stage / key", "Value"])
        self.artifacts.setAlternatingRowColors(True)
        self.artifacts.setColumnWidth(0, 300)
        self.artifacts.setStyleSheet("border:none;")
        self.tabs.addTab(self.artifacts, "Stage data")

        self._tabs: dict[str, _RenderTab] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self.refresh)

    # ------------------------------------------------------------------
    # Run selection
    # ------------------------------------------------------------------
    def set_run_dir(self, run_dir: str | Path | None, epoch: float | None = None) -> None:
        """Point at a run directory (the parent of ``output``/``preview``).

        ``epoch`` is the run's start time; renders older than it are
        treated as a previous run's and not displayed. Pass ``None`` when
        opening a *finished* run so everything already on disk shows.
        """
        self.renders.clear()
        self._tabs.clear()
        self._showing_renders = False
        self.renders.hide()
        self.render_placeholder.show()
        self._run_dir = Path(run_dir) if run_dir else None
        self._epoch = epoch

        if self._run_dir is None:
            self._set_placeholder_idle()
            self._timer.stop()
            return

        for label, image_rel, json_rel in _RENDERS:
            self._tabs[label] = _RenderTab(
                self._run_dir / image_rel,
                (self._run_dir / json_rel) if json_rel else None,
            )

        self.render_placeholder.setText(
            (
                "Run in progress — each stage's render appears here as it completes."
                if epoch is not None
                else "This run wrote no stage renders."
            )
            + f"\n\n{self._run_dir.resolve()}"
        )
        self.refresh()
        if epoch is not None:
            self._timer.start()

    def _set_placeholder_idle(self) -> None:
        self.render_placeholder.setText(
            "No run loaded.\n\n"
            "Start a reconstruction, or open a previous run, to see the flight path,\n"
            "ground coverage and frame placement as the pipeline produces them."
        )

    def stop(self) -> None:
        """Stop polling. Called when a run finishes or the window closes."""
        self._timer.stop()
        self._epoch = None  # the run is over; show everything it wrote
        self.refresh()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Add tabs for renders that now exist; reload the ones that changed."""
        if self._run_dir is None:
            return

        order = [label for label, _image, _json in _RENDERS]
        shown = 0
        for label in order:
            tab = self._tabs.get(label)
            if tab is None or not tab.refresh(self._epoch):
                continue
            shown += 1
            if self.renders.indexOf(tab) == -1:
                # Insert in declaration order so tabs stay in pipeline
                # order regardless of which file lands first.
                position = sum(
                    1
                    for earlier in order[: order.index(label)]
                    if self.renders.indexOf(self._tabs[earlier]) != -1
                )
                self.renders.insertTab(position, tab, label)

        if shown and not self._showing_renders:
            self._showing_renders = True
            self.render_placeholder.hide()
            self.renders.show()
        elif not shown and self._showing_renders:
            self._showing_renders = False
            self.renders.hide()
            self.render_placeholder.show()

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------
    def append_log(self, message: str, tone: str = "") -> None:
        if not message or message == self._last_log_line:
            return
        self._last_log_line = message
        stamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{stamp}  {message}")
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear_log(self) -> None:
        self.log_view.clear()
        self._last_log_line = ""

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    def set_stage_artifacts(self, stage: str, artifacts: dict | None) -> None:
        """Insert or replace one stage's measured values."""
        if not artifacts:
            return

        existing = self.artifacts.findItems(stage, Qt.MatchExactly, 0)
        for item in existing:
            self.artifacts.takeTopLevelItem(self.artifacts.indexOfTopLevelItem(item))

        root = QTreeWidgetItem([stage, f"{len(artifacts)} values"])
        root.setForeground(1, Qt.gray)
        self._populate(root, artifacts)
        self.artifacts.addTopLevelItem(root)
        root.setExpanded(False)

    def _populate(self, parent: QTreeWidgetItem, data: Any) -> None:
        """Render nested dicts/lists as a tree, flattening scalars inline."""
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, (list, tuple)):
            items = ((str(index), value) for index, value in enumerate(data))
        else:
            return

        for key, value in items:
            if isinstance(value, dict) or (
                isinstance(value, (list, tuple)) and value and isinstance(value[0], (dict, list, tuple))
            ):
                child = QTreeWidgetItem([str(key), f"{len(value)} entries"])
                self._populate(child, value)
            elif isinstance(value, (list, tuple)):
                preview = ", ".join(_format_scalar(v) for v in value[:6])
                suffix = f" … (+{len(value) - 6})" if len(value) > 6 else ""
                child = QTreeWidgetItem([str(key), f"[{preview}{suffix}]"])
            else:
                child = QTreeWidgetItem([str(key), _format_scalar(value)])
            parent.addChild(child)

    def clear_artifacts(self) -> None:
        self.artifacts.clear()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.clear_log()
        self.clear_artifacts()
        self.tabs.setCurrentIndex(0)


#: The panel was called ``PreviewPanel`` when it only showed the renders.
PreviewPanel = DiagnosticsPanel
