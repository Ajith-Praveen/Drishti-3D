"""The 3D stage: the viewport plus the controls that float on top of it.

The controls sit *over* the render, not in a side panel, for the same
reason a camera's controls are on the camera: changing the view and
judging the view are the same action, and a round trip to a dock on the
far side of a 27-inch window breaks it.

Two clusters, both translucent so the render stays the subject:

- **top-right** -- projection presets (top / front / side / iso) and
  "fit", i.e. where you are looking from.
- **bottom-left** -- what is being drawn: the colour mode, and a live
  caption naming the layers currently in the scene.

Everything here is a thin shell over :class:`drishti3d.app.viewport.Viewport`;
no rendering logic lives in this module.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import icons, theme
from drishti3d.app.viewport import COLOR_MODES, Viewport

__all__ = ["Viewer"]

_COLOR_MODE_LABELS = {
    "rgb": "RGB",
    "confidence": "Confidence",
    "uncertainty": "Uncertainty",
    "height": "Height",
}

_COLOR_MODE_HELP = {
    "rgb": "Photographic colour sampled from the source frames.",
    "confidence": (
        "Green = MEASURED (directly observed).\n"
        "Amber = LOW_CONFIDENCE (grazing angles, few views).\n"
        "Red = INFERRED (never observed; filled in by the model's prior)."
    ),
    "uncertainty": "The inverse framing of confidence: green is most certain.",
    "height": "Elevation ramp over the scene's own Z range.",
}

_VIEW_LABELS = (("top", "Top"), ("front", "Front"), ("side", "Side"), ("iso", "Iso"))


class _OverlayBar(QFrame):
    """A translucent floating control cluster."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("OverlayBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame#OverlayBar{{background:rgba(17,19,22,214);"
            f" border:1px solid {theme.BORDER}; border-radius:{theme.RADIUS + 1}px;}}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)
        self.row = layout


class Viewer(QWidget):
    """Viewport + floating overlay controls."""

    colorModeChanged = Signal(str)

    def __init__(self, viewport: Viewport | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.viewport = viewport or Viewport()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewport)

        # Overlays are children of *this* widget, raised above the VTK
        # widget and positioned in resizeEvent. They cannot be laid out
        # by a QLayout on top of the render surface, and on macOS the VTK
        # widget is the one native NSView in the tree -- anything meant
        # to sit above it has to be an explicitly raised sibling.
        self._view_bar = self._build_view_bar()
        self._display_bar = self._build_display_bar()
        self.caption = self._build_caption()

        self.viewport.sceneChanged.connect(self._refresh_caption)
        self._refresh_caption()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_view_bar(self) -> _OverlayBar:
        bar = _OverlayBar(self)
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)

        for name, label in _VIEW_LABELS:
            button = QPushButton(label)
            button.setObjectName("Chip")
            button.setCheckable(True)
            button.setChecked(name == "iso")
            button.setToolTip(f"Look from {label.lower()}")
            button.clicked.connect(lambda _checked=False, n=name: self.viewport.set_view(n))
            self._view_group.addButton(button)
            bar.row.addWidget(button)

        separator = QFrame()
        separator.setFixedWidth(1)
        separator.setStyleSheet(f"background:{theme.BORDER}; border:none;")
        bar.row.addWidget(separator)

        fit = QPushButton(icons.icon("reset", 14), "")
        fit.setObjectName("Chip")
        fit.setToolTip("Fit the view to the reconstruction")
        fit.clicked.connect(self.viewport.reset_camera)
        bar.row.addWidget(fit)

        return bar

    def _build_display_bar(self) -> _OverlayBar:
        bar = _OverlayBar(self)
        self._color_group = QButtonGroup(self)
        self._color_group.setExclusive(True)
        self._color_buttons: dict[str, QPushButton] = {}

        for mode in COLOR_MODES:
            button = QPushButton(_COLOR_MODE_LABELS.get(mode, mode))
            button.setObjectName("Chip")
            button.setCheckable(True)
            button.setChecked(mode == "rgb")
            button.setToolTip(_COLOR_MODE_HELP.get(mode, ""))
            button.clicked.connect(lambda _checked=False, m=mode: self._choose_color_mode(m))
            self._color_group.addButton(button)
            self._color_buttons[mode] = button
            bar.row.addWidget(button)

        return bar

    def _build_caption(self) -> QLabel:
        caption = QLabel(self)
        caption.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; background:transparent; font-size:10px;"
        )
        caption.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        return caption

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def _choose_color_mode(self, mode: str) -> None:
        self.viewport.set_color_mode(mode)
        self.colorModeChanged.emit(mode)

    def set_color_mode(self, mode: str) -> None:
        """Adopt a colour mode chosen elsewhere, without re-emitting."""
        button = self._color_buttons.get(mode)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        if self.viewport.color_mode() != mode:
            self.viewport.set_color_mode(mode)

    def _refresh_caption(self) -> None:
        parts: list[str] = []
        if self.viewport.has_flight_path():
            parts.append("flight path")
        if self.viewport.camera_count():
            parts.append(f"{self.viewport.camera_count()} cameras")
        if self.viewport.point_count():
            parts.append("point cloud")
        if self.viewport.has_mesh():
            parts.append("mesh")
        self.caption.setText("  ·  ".join(parts))
        self.caption.adjustSize()
        self._position_overlays()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _position_overlays(self) -> None:
        margin = 10
        width = self.width()
        height = self.height()

        self._view_bar.adjustSize()
        self._view_bar.move(max(margin, width - self._view_bar.width() - margin), margin)
        self._view_bar.raise_()

        self._display_bar.adjustSize()
        self._display_bar.move(margin, height - self._display_bar.height() - margin)
        self._display_bar.raise_()

        self.caption.adjustSize()
        self.caption.move(
            margin + 3,
            height - self._display_bar.height() - self.caption.height() - margin - 4,
        )
        self.caption.raise_()

    def resizeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._position_overlays()

    def showEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._position_overlays()
