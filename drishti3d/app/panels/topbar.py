"""The command bar: identity, inputs, the run control, and the device.

One row, read left to right, in the order an operator works:

    ◆ DRISHTI-3D │ [video] [telemetry] │ ......... │ ▶ Run  ■ Cancel │ mps

The product mark is on the left because this is the app's only branded
surface -- there is no splash screen and no about-box chrome to carry it.
The source chips sit next to it because a reconstruction is only ever
about one video, and which one must never be a question. The run control
is right-aligned and is the only filled, accented button anywhere in the
window, so "the thing that starts the expensive operation" is
unambiguous. The device badge is last because it is the one piece of
state the operator cannot change from here, only verify.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import icons, theme

__all__ = ["SourceChip", "TopBar"]


class SourceChip(QFrame):
    """One input file: icon, label, value, and a click to replace it."""

    clicked = Signal()
    cleared = Signal()

    def __init__(
        self,
        icon_name: str,
        label: str,
        empty_text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._empty_text = empty_text
        self._path: str = ""
        self._required = False

        # QLabel derives from QFrame, so an unscoped `QFrame{border:...}`
        # rule on this widget also draws a border around every label
        # inside it. Scope every rule below to this object name.
        self.setObjectName("SourceChip")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(38)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(9, 4, 11, 4)
        row.setSpacing(8)

        self._icon_name = icon_name
        self._icon_tone = theme.TEXT_FAINT
        self.icon_label = QLabel()
        self.icon_label.setPixmap(icons.pixmap(icon_name, 17, theme.TEXT_FAINT))
        self.icon_label.setFixedSize(QSize(18, 18))
        row.addWidget(self.icon_label, 0, Qt.AlignVCenter)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(0)

        self.kind_label = QLabel(label.upper())
        self.kind_label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:9px; font-weight:700; letter-spacing:0.6px;"
        )
        text_column.addWidget(self.kind_label)

        self.value_label = QLabel(empty_text)
        self.value_label.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
        text_column.addWidget(self.value_label)

        row.addLayout(text_column, 1)
        self._restyle()

    # -- content --------------------------------------------------------

    def set_path(self, path: str, detail: str = "") -> None:
        self._path = path or ""
        if self._path:
            name = Path(self._path).name
            self.value_label.setText(f"{name}   {detail}" if detail else name)
            self.value_label.setToolTip(self._path)
            self.setToolTip(self._path)
        else:
            self.value_label.setText(self._empty_text)
            self.value_label.setToolTip("")
            self.setToolTip(f"Choose a {self._label.lower()} file")
        self._restyle()

    def set_required(self, required: bool) -> None:
        """Mark the chip as a missing prerequisite (amber, not red).

        Telemetry is optional-but-degrading: without it the output stays
        in a local frame. That is a warning about the *result*, not an
        error in the app, so it is never red.
        """
        self._required = required
        self._restyle()

    def path(self) -> str:
        return self._path

    def _restyle(self) -> None:
        if self._path:
            border, text = theme.BORDER_STRONG, theme.TEXT_SECONDARY
            icon_tone = theme.ACCENT
        elif self._required:
            border, text = theme.WARN_DIM, theme.WARN
            icon_tone = theme.WARN
        else:
            border, text = theme.BORDER, theme.TEXT_MUTED
            icon_tone = theme.TEXT_FAINT

        self.setStyleSheet(
            f"QFrame#SourceChip{{background:{theme.BG_PANEL}; border:1px solid {border};"
            f" border-radius:{theme.RADIUS}px;}}"
            f"QFrame#SourceChip:hover{{border-color:{theme.BORDER_STRONG};"
            f" background:{theme.BG_RAISED};}}"
            f"QFrame#SourceChip QLabel{{border:none; background:transparent;}}"
        )
        self.value_label.setStyleSheet(
            f"QLabel{{color:{text}; font-size:11px; background:transparent; border:none;}}"
        )
        self.kind_label.setStyleSheet(
            f"QLabel{{color:{theme.TEXT_FAINT}; font-size:9px; font-weight:600;"
            f" letter-spacing:0.6px; background:transparent; border:none;}}"
        )
        self._icon_tone = icon_tone
        self.icon_label.setPixmap(icons.pixmap(self._icon_name, 17, icon_tone))

    def set_icon(self, icon_name: str) -> None:
        self._icon_name = icon_name
        self.icon_label.setPixmap(icons.pixmap(icon_name, 17, self._icon_tone))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        elif event.button() == Qt.RightButton and self._path:
            self.cleared.emit()
        super().mousePressEvent(event)


class _Badge(QLabel):
    """A small monospace status pill, e.g. the compute device."""

    def __init__(self, text: str = "", tone: str = theme.TEXT_MUTED, parent=None) -> None:
        super().__init__(text, parent)
        self.setFont(theme.mono_font(10, weight=700))
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        self.setStyleSheet(
            f"QLabel{{color:{tone}; background:{theme.BG_DARK};"
            f" border:1px solid {theme.BORDER}; border-radius:{theme.RADIUS}px;"
            f" padding:4px 8px;}}"
        )


class TopBar(QFrame):
    """The application command bar."""

    videoRequested = Signal()
    telemetryRequested = Signal()
    videoCleared = Signal()
    telemetryCleared = Signal()
    runRequested = Signal()
    demoRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(54)
        self.setStyleSheet(
            f"QFrame#TopBar{{background:{theme.BG_DARKEST};"
            f" border-bottom:1px solid {theme.BORDER};}}"
        )
        self.setObjectName("TopBar")

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(10)

        # --- identity ---------------------------------------------------
        mark = QLabel()
        mark.setPixmap(icons.mark(24))
        mark.setFixedSize(QSize(26, 26))
        row.addWidget(mark, 0, Qt.AlignVCenter)

        wordmark = QLabel("DRISHTI<span style='color:%s'>-3D</span>" % theme.ACCENT)
        wordmark.setTextFormat(Qt.RichText)
        wordmark.setStyleSheet(
            f"color:{theme.TEXT_PRIMARY}; font-size:14px; font-weight:800; letter-spacing:1.1px;"
        )
        row.addWidget(wordmark, 0, Qt.AlignVCenter)

        row.addSpacing(6)
        row.addWidget(self._rule())
        row.addSpacing(6)

        # --- sources ----------------------------------------------------
        self.video_chip = SourceChip("video", "Drone video", "No video — click to choose")
        self.video_chip.clicked.connect(self.videoRequested)
        self.video_chip.cleared.connect(self.videoCleared)
        row.addWidget(self.video_chip, 0, Qt.AlignVCenter)

        self.telemetry_chip = SourceChip("telemetry", "Telemetry", "None — output stays local-frame")
        self.telemetry_chip.set_required(True)
        self.telemetry_chip.clicked.connect(self.telemetryRequested)
        self.telemetry_chip.cleared.connect(self.telemetryCleared)
        row.addWidget(self.telemetry_chip, 0, Qt.AlignVCenter)

        row.addStretch(1)

        # --- run control ------------------------------------------------
        self.demo_button = QPushButton(icons.icon("demo", 15), "  Demo")
        self.demo_button.setObjectName("Chip")
        self.demo_button.setToolTip(
            "Run a synthetic reconstruction: no video, no GPU, a few seconds.\n"
            "Exercises every panel end to end."
        )
        self.demo_button.clicked.connect(self.demoRequested)
        row.addWidget(self.demo_button, 0, Qt.AlignVCenter)

        self.run_button = QPushButton(icons.icon("run", 15, "#ffffff", "#ffffff"), "  Run")
        self.run_button.setObjectName("Primary")
        self.run_button.setMinimumWidth(96)
        self.run_button.setEnabled(False)
        self.run_button.setToolTip("Reconstruct the loaded video (⌘R)")
        self.run_button.clicked.connect(self.runRequested)
        row.addWidget(self.run_button, 0, Qt.AlignVCenter)

        self.cancel_button = QPushButton(icons.icon("stop", 14, theme.ERR, theme.ERR), "  Cancel")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip("Stop after the current step; keeps the partial result")
        self.cancel_button.clicked.connect(self.cancelRequested)
        row.addWidget(self.cancel_button, 0, Qt.AlignVCenter)

        row.addSpacing(6)
        row.addWidget(self._rule())
        row.addSpacing(6)

        self.device_badge = _Badge("cpu")
        self.device_badge.setToolTip("Compute device the geometry backbone will use")
        row.addWidget(self.device_badge, 0, Qt.AlignVCenter)

    # -- helpers ---------------------------------------------------------

    def _rule(self) -> QFrame:
        rule = QFrame()
        rule.setFrameShape(QFrame.VLine)
        rule.setFixedWidth(1)
        rule.setStyleSheet(f"background:{theme.BORDER}; border:none;")
        return rule

    # -- state -----------------------------------------------------------

    def set_device(self, device: str) -> None:
        tone = theme.OK if device in ("cuda", "mps") else theme.TEXT_MUTED
        self.device_badge.setText(device)
        self.device_badge.set_tone(tone)
        self.device_badge.setToolTip(
            f"Compute device: {device}"
            + ("" if device in ("cuda", "mps") else "\nNo GPU detected — runs will be slow.")
        )

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running and bool(self.video_chip.path()))
        self.demo_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.video_chip.setEnabled(not running)
        self.telemetry_chip.setEnabled(not running)

    def set_video(self, path: str, detail: str = "") -> None:
        self.video_chip.set_path(path, detail)
        self.run_button.setEnabled(bool(path))

    def set_telemetry(self, path: str, detail: str = "") -> None:
        self.telemetry_chip.set_path(path, detail)
        self.telemetry_chip.set_required(not path)
