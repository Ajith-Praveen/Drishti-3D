"""The mission rail: every pipeline stage, live, down the left edge.

Why this replaces both the progress panel and the node graph
------------------------------------------------------------
The app previously had two views of the same eleven stages -- a
``QTreeWidget`` of rows on a "Process" page, and a draggable node graph on
a "Workspace" page -- and neither was on screen when the operator was
looking at the 3D view. During a thirty-minute run the single most
important question is "which stage is running, and is it going wrong",
and the answer was one or two clicks away on a page that hid the result.

So there is one rail, it is always visible, and it is ordered exactly like
``pipeline.runner._build_stages()``. A stage that is running pulses; a
stage that failed is red and keeps its message; an optional stage that
was skipped is dimmed rather than removed, because "semantics did not
run" is information.

The node graph was a picture of a pipeline the operator cannot actually
rewire -- ``_build_stages()`` is a fixed list -- so it promised an
editing capability that does not exist. The rail promises only what it
delivers: status, timing, and what each stage concluded.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import theme

__all__ = ["STAGE_SPECS", "StageRail", "StageSpec"]

PENDING = "pending"
RUNNING = "running"
DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"

#: Statuses a ``StageResult`` can carry, mapped onto rail states.
_RESULT_STATUS = {"ok": DONE, "skipped": SKIPPED, "failed": FAILED}


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage, as the operator needs to understand it."""

    name: str      # must match pipeline.stages.<Stage>.name exactly
    title: str
    subtitle: str
    optional: bool = False


#: Exactly ``pipeline.runner._build_stages()``, in execution order. The
#: ``name`` fields are the contract: a typo here silently produces a rail
#: row that never lights up, which is how the old node graph's
#: "bundle_adjustment" vs "bundle" mismatch hid for so long.
STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec("ingest", "Ingest", "decode video, align telemetry"),
    StageSpec("triage", "Triage", "select sharp, well-spaced keyframes"),
    StageSpec("drone_path", "Flight path", "where the drone flew"),
    StageSpec("coverage", "Coverage", "which ground the camera saw"),
    StageSpec("semantics", "Semantics", "per-pixel classes, dynamic masking", optional=True),
    StageSpec("pose_prior", "Pose prior", "refine poses from features + GPS"),
    StageSpec("frame_placement", "Placement", "check every frame lands where it should"),
    StageSpec("geometry", "Geometry", "depth + poses per window"),
    StageSpec("bundle_adjustment", "Bundle adjust", "joint refinement", optional=True),
    StageSpec("fusion", "Fusion", "confidence-weighted TSDF surface"),
    StageSpec("export", "Export", "georeferenced deliverables"),
)


class _StageRow(QWidget):
    """One stage: status dot on a spine, title, timing, latest message."""

    clicked = Signal(str)

    DOT_X = 17
    LEFT_PAD = 34

    def __init__(self, spec: StageSpec, is_last: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._is_last = is_last
        self.state = PENDING
        self.elapsed: float | None = None
        self.message = ""
        self.detail = ""
        self._pulse = 0.0
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumHeight(52)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{spec.title} — {spec.subtitle}")

    # -- state ----------------------------------------------------------

    def set_state(self, state: str, message: str = "", elapsed: float | None = None) -> None:
        self.state = state
        if message:
            self.message = message.splitlines()[0]
        if elapsed is not None:
            self.elapsed = elapsed
        self.updateGeometry()
        self.update()

    def set_detail(self, detail: str) -> None:
        """The stage's own live progress line, shown while it runs."""
        self.detail = detail
        self.update()

    def reset(self) -> None:
        self.state = PENDING
        self.elapsed = None
        self.message = ""
        self.detail = ""
        self.update()

    def set_pulse(self, phase: float) -> None:
        if self.state == RUNNING:
            self._pulse = phase
            self.update()

    # -- painting -------------------------------------------------------

    def _accent(self) -> QColor:
        return {
            PENDING: QColor(theme.TEXT_FAINT),
            RUNNING: QColor(theme.ACCENT),
            DONE: QColor(theme.OK),
            SKIPPED: QColor(theme.TEXT_FAINT),
            FAILED: QColor(theme.ERR),
        }[self.state]

    def sizeHint(self) -> QSize:
        body = self.detail if self.state == RUNNING and self.detail else self.message
        return QSize(240, 52 if body else 40)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        width = self.width()
        height = self.height()
        accent = self._accent()
        running = self.state == RUNNING

        if running:
            painter.fillRect(0, 0, width, height, QColor(theme.BG_RAISED))
            painter.fillRect(0, 0, 2, height, accent)

        # --- spine ------------------------------------------------------
        spine = QPen(QColor(theme.BORDER), 1)
        painter.setPen(spine)
        painter.drawLine(self.DOT_X, 0, self.DOT_X, 14)
        if not self._is_last:
            painter.drawLine(self.DOT_X, 26, self.DOT_X, height)

        # --- status dot -------------------------------------------------
        centre = QRectF(self.DOT_X - 6.0, 14.0, 12.0, 12.0)
        painter.setPen(Qt.NoPen)

        if running:
            # A ring that breathes: motion is what makes "this is the one
            # that is working" readable at a glance across the window.
            halo = QColor(accent)
            halo.setAlphaF(0.10 + 0.14 * self._pulse)
            radius = 6.0 + 3.0 * self._pulse
            painter.setBrush(halo)
            painter.drawEllipse(centre.center(), radius, radius)
            painter.setBrush(accent)
            painter.drawEllipse(centre.center(), 4.5, 4.5)
        elif self.state == DONE:
            painter.setBrush(accent)
            painter.drawEllipse(centre)
            pen = QPen(QColor(theme.BG_DARKEST), 1.8)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            check = QPainterPath()
            check.moveTo(centre.left() + 3.0, centre.top() + 6.2)
            check.lineTo(centre.left() + 5.0, centre.top() + 8.4)
            check.lineTo(centre.left() + 9.0, centre.top() + 3.6)
            painter.drawPath(check)
        elif self.state == FAILED:
            painter.setBrush(accent)
            painter.drawEllipse(centre)
            pen = QPen(QColor(theme.BG_DARKEST), 1.8)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(centre.left() + 4.0, centre.top() + 4.0, centre.right() - 4.0, centre.bottom() - 4.0)
            painter.drawLine(centre.right() - 4.0, centre.top() + 4.0, centre.left() + 4.0, centre.bottom() - 4.0)
        elif self.state == SKIPPED:
            painter.setPen(QPen(QColor(theme.TEXT_FAINT), 1.4))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(centre.adjusted(1, 1, -1, -1))
            painter.drawLine(centre.left() + 3.5, centre.center().y(), centre.right() - 3.5, centre.center().y())
        else:
            painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1.4))
            painter.setBrush(QColor(theme.BG_DARK))
            painter.drawEllipse(centre.adjusted(1.5, 1.5, -1.5, -1.5))

        # --- title ------------------------------------------------------
        title_color = {
            PENDING: QColor(theme.TEXT_MUTED),
            RUNNING: QColor(theme.TEXT_PRIMARY),
            DONE: QColor(theme.TEXT_SECONDARY),
            SKIPPED: QColor(theme.TEXT_FAINT),
            FAILED: QColor(theme.ERR),
        }[self.state]

        font = painter.font()
        font.setPixelSize(12)
        font.setBold(running or self.state == FAILED)
        painter.setFont(font)

        metrics = QFontMetrics(font)
        timing = ""
        if self.elapsed is not None and self.state in (DONE, FAILED):
            timing = _format_elapsed(self.elapsed)
        timing_width = metrics.horizontalAdvance(timing) + 10 if timing else 6

        title_rect = QRectF(self.LEFT_PAD, 11, width - self.LEFT_PAD - timing_width, 16)
        painter.setPen(title_color)
        painter.drawText(
            title_rect,
            Qt.AlignLeft | Qt.AlignVCenter,
            metrics.elidedText(self.spec.title, Qt.ElideRight, int(title_rect.width())),
        )

        if timing:
            painter.setPen(QColor(theme.TEXT_FAINT))
            mono = theme.mono_font(10)
            painter.setFont(mono)
            painter.drawText(
                QRectF(width - timing_width - 4, 11, timing_width, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                timing,
            )

        # --- body -------------------------------------------------------
        body = self.detail if running and self.detail else self.message
        if not body and self.state == PENDING:
            body = self.spec.subtitle
        if body:
            small = painter.font()
            small.setPixelSize(10)
            small.setBold(False)
            painter.setFont(small)
            painter.setPen(QColor(theme.ERR if self.state == FAILED else theme.TEXT_FAINT))
            body_rect = QRectF(self.LEFT_PAD, 27, width - self.LEFT_PAD - 8, 16)
            painter.drawText(
                body_rect,
                Qt.AlignLeft | Qt.AlignTop,
                QFontMetrics(small).elidedText(body, Qt.ElideRight, int(body_rect.width())),
            )

        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.clicked.emit(self.spec.name)
        super().mousePressEvent(event)


def _format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class StageRail(QWidget):
    """The whole rail: a header, eleven stage rows, and a run summary."""

    stageSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._rows: dict[str, _StageRow] = {}
        self._pulse_phase = 0.0
        self._pulse_up = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QLabel("MISSION")
        header.setObjectName("SectionHeader")
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet(f"QScrollArea{{background:{theme.BG_PANEL};}}")
        scroll.viewport().setStyleSheet(f"background:{theme.BG_PANEL};")

        body = QWidget()
        body.setStyleSheet(f"background:{theme.BG_PANEL};")
        rows = QVBoxLayout(body)
        rows.setContentsMargins(0, 6, 0, 6)
        rows.setSpacing(0)

        for index, spec in enumerate(STAGE_SPECS):
            row = _StageRow(spec, is_last=index == len(STAGE_SPECS) - 1)
            row.clicked.connect(self.stageSelected)
            rows.addWidget(row)
            self._rows[spec.name] = row
        rows.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        self.summary = QLabel("Idle")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; padding:8px 10px; "
            f"background:{theme.BG_RAISED}; border-top:1px solid {theme.BORDER};"
        )
        outer.addWidget(self.summary)

        # One timer drives every running row's pulse, rather than a timer
        # per row: eleven timers firing at 60 Hz to animate one dot is not
        # a reasonable thing to do to a laptop battery.
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(45)
        self._pulse_timer.timeout.connect(self._tick_pulse)

    # -- live updates ---------------------------------------------------

    def reset(self) -> None:
        for row in self._rows.values():
            row.reset()
        self.summary.setText("Idle")
        self._pulse_timer.stop()

    def set_running(self, stage: str) -> None:
        """Mark ``stage`` running; anything still running before it is done."""
        for name, row in self._rows.items():
            if name != stage and row.state == RUNNING:
                row.set_state(DONE)
        row = self._rows.get(stage)
        if row is not None:
            row.set_state(RUNNING)
        if not self._pulse_timer.isActive():
            self._pulse_timer.start()

    def set_progress(self, stage: str, current: int, total: int, message: str) -> None:
        row = self._rows.get(stage)
        if row is None:
            return
        prefix = f"{current}/{total}  " if total else ""
        row.set_detail(f"{prefix}{message}")

    def apply_stage_result(self, name: str, status: str, message: str, elapsed: float) -> None:
        row = self._rows.get(name)
        if row is None:
            return
        row.set_state(_RESULT_STATUS.get(status, DONE), message=message, elapsed=elapsed)

    def finish(self, note: str = "") -> None:
        for row in self._rows.values():
            if row.state == RUNNING:
                row.set_state(DONE)
        self._pulse_timer.stop()
        for row in self._rows.values():
            row.set_pulse(0.0)
        if note:
            self.summary.setText(note)

    def fail(self, stage: str | None, message: str) -> None:
        if stage and stage in self._rows:
            self._rows[stage].set_state(FAILED, message=message)
        else:
            for row in self._rows.values():
                if row.state == RUNNING:
                    row.set_state(FAILED, message=message)
        self._pulse_timer.stop()
        self.summary.setText(message.splitlines()[0] if message else "Failed")
        self.summary.setStyleSheet(
            f"color:{theme.ERR}; padding:8px 10px; "
            f"background:{theme.BG_RAISED}; border-top:1px solid {theme.BORDER};"
        )

    def set_summary(self, text: str, tone: str = "muted") -> None:
        color = {
            "muted": theme.TEXT_MUTED,
            "ok": theme.OK,
            "warn": theme.WARN,
            "error": theme.ERR,
            "accent": theme.ACCENT,
        }.get(tone, theme.TEXT_MUTED)
        self.summary.setText(text)
        self.summary.setStyleSheet(
            f"color:{color}; padding:8px 10px; "
            f"background:{theme.BG_RAISED}; border-top:1px solid {theme.BORDER};"
        )

    def state_of(self, stage: str) -> str | None:
        row = self._rows.get(stage)
        return None if row is None else row.state

    def message_of(self, stage: str) -> str:
        row = self._rows.get(stage)
        return "" if row is None else row.message

    # -- animation ------------------------------------------------------

    def _tick_pulse(self) -> None:
        step = 0.07
        self._pulse_phase += step if self._pulse_up else -step
        if self._pulse_phase >= 1.0:
            self._pulse_phase, self._pulse_up = 1.0, False
        elif self._pulse_phase <= 0.0:
            self._pulse_phase, self._pulse_up = 0.0, True

        any_running = False
        for row in self._rows.values():
            if row.state == RUNNING:
                any_running = True
                row.set_pulse(self._pulse_phase)
        if not any_running:
            self._pulse_timer.stop()
