"""Scene layers: what is drawn in the 3D view, and how.

Every control here is bound to something that exists. The previous
version shipped "Mesh" and "Cameras" checkboxes whose signals were
connected to nothing and whose actors were never built -- a control that
does nothing reads as a broken app, not as an unimplemented feature. Now
each row is backed by a real ``Viewport`` layer, and a row whose layer
has no data yet is *disabled with a reason in its tooltip* rather than
left live and inert.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import icons, theme

__all__ = ["LayersPanel"]

#: (key, icon, label, tooltip-when-empty). ``key`` is the signal name
#: stem and the identity used by ``set_layer_available``.
_LAYERS: tuple[tuple[str, str, str, str], ...] = (
    ("path", "path", "Flight path", "Available once ingest has read the telemetry."),
    ("cameras", "camera", "Cameras", "Available once triage has selected keyframes."),
    ("points", "cloud", "Point cloud", "Available once geometry produces its first window."),
    ("mesh", "mesh", "Mesh surface", "Available once fusion extracts the TSDF surface."),
    ("grid", "grid", "Ground grid", ""),
    ("axes", "axes", "Orientation axes", ""),
    ("scale", "ruler", "Scale bar", ""),
    ("measurements", "area", "Measurements", "Appears once you take a measurement."),
)


class _LayerRow(QWidget):
    """A checkbox with a glyph, plus a right-aligned count."""

    toggled = Signal(bool)

    def __init__(self, key: str, icon_name: str, label: str, empty_hint: str, parent=None) -> None:
        super().__init__(parent)
        self.key = key
        self._empty_hint = empty_hint

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(7)

        self.glyph = QLabel()
        self.glyph.setPixmap(icons.pixmap(icon_name, 14, theme.TEXT_FAINT))
        self.glyph.setFixedWidth(16)
        row.addWidget(self.glyph)

        self.check = QCheckBox(label)
        self.check.setChecked(True)
        self.check.toggled.connect(self.toggled)
        row.addWidget(self.check, 1)

        self.count = QLabel("")
        self.count.setFont(theme.mono_font(9))
        self.count.setStyleSheet(f"color:{theme.TEXT_FAINT};")
        row.addWidget(self.count, 0, Qt.AlignRight)

        self._icon_name = icon_name
        self.set_available(False)

    def set_available(self, available: bool, count: int | None = None) -> None:
        self.check.setEnabled(available)
        self.glyph.setPixmap(
            icons.pixmap(
                self._icon_name, 14, theme.TEXT_SECONDARY if available else theme.TEXT_FAINT
            )
        )
        # Set the label colour on the widget itself. Leaving it to the
        # application stylesheet's :disabled rule did not survive the
        # per-row restyle and rendered available rows fainter than
        # unavailable ones -- the exact opposite of the intent.
        self.check.setStyleSheet(
            f"QCheckBox{{color:{theme.TEXT_PRIMARY if available else theme.TEXT_FAINT};}}"
        )
        if count is None or not available:
            self.count.setText("")
        else:
            self.count.setText(f"{count:,}")
        self.count.setStyleSheet(
            f"color:{theme.TEXT_MUTED if available else theme.TEXT_FAINT};"
        )
        self.setToolTip("" if available else self._empty_hint)


class LayersPanel(QWidget):
    """Layer visibility, point size, and the confidence legend."""

    pathToggled = Signal(bool)
    camerasToggled = Signal(bool)
    pointsToggled = Signal(bool)
    meshToggled = Signal(bool)
    gridToggled = Signal(bool)
    axesToggled = Signal(bool)
    scaleToggled = Signal(bool)
    measurementsToggled = Signal(bool)
    pointSizeChanged = Signal(int)

    #: Back-compat alias: earlier code connected ``pointCloudToggled``.
    pointCloudToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        signals = {
            "path": self.pathToggled,
            "cameras": self.camerasToggled,
            "points": self.pointsToggled,
            "mesh": self.meshToggled,
            "grid": self.gridToggled,
            "axes": self.axesToggled,
            "scale": self.scaleToggled,
            "measurements": self.measurementsToggled,
        }

        layout.addWidget(self._heading("LAYERS"))
        self._rows: dict[str, _LayerRow] = {}
        for key, icon_name, label, hint in _LAYERS:
            row = _LayerRow(key, icon_name, label, hint)
            row.toggled.connect(signals[key])
            if key == "points":
                row.toggled.connect(self.pointCloudToggled)
            layout.addWidget(row)
            self._rows[key] = row

        # Furniture is always available; only data layers gate.
        for key in ("grid", "axes", "scale"):
            self._rows[key].set_available(True)
        self._rows["scale"].check.setChecked(True)

        # --- point size ---------------------------------------------------
        layout.addWidget(self._heading("POINT SIZE"))
        size_row = QHBoxLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(8)
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(1, 12)
        self.point_size_slider.setValue(3)
        self.point_size_slider.valueChanged.connect(self.pointSizeChanged)
        self.point_size_slider.valueChanged.connect(self._update_size_label)
        size_row.addWidget(self.point_size_slider, 1)
        self.size_label = QLabel("3 px")
        self.size_label.setFont(theme.mono_font(10))
        self.size_label.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.size_label.setFixedWidth(34)
        size_row.addWidget(self.size_label)
        layout.addLayout(size_row)

        # --- legend ---------------------------------------------------------
        layout.addWidget(self._heading("CONFIDENCE LEGEND"))
        layout.addWidget(self._legend())

        layout.addStretch(1)

    # ------------------------------------------------------------------
    def _heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:9px; font-weight:700; letter-spacing:0.7px;"
        )
        return label

    def _legend(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Legend")
        frame.setStyleSheet(
            f"QFrame#Legend{{background:{theme.BG_DARK}; border:1px solid {theme.BORDER};"
            f" border-radius:{theme.RADIUS}px;}}"
            f"QFrame#Legend QLabel{{border:none; background:transparent;}}"
        )
        grid = QGridLayout(frame)
        grid.setContentsMargins(9, 7, 9, 7)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)

        entries = (
            (theme.CONF_MEASURED, "Measured", "directly observed by 2+ cameras"),
            (theme.CONF_LOW, "Low confidence", "grazing incidence, or few views"),
            (theme.CONF_INFERRED, "Inferred", "never observed — model prior only"),
        )
        for row, (color, name, note) in enumerate(entries):
            swatch = QLabel("●")
            swatch.setStyleSheet(f"color:{color}; background:transparent; font-size:13px;")
            grid.addWidget(swatch, row, 0)

            text = QLabel(f"<b>{name}</b> — <span style='color:{theme.TEXT_FAINT}'>{note}</span>")
            text.setTextFormat(Qt.RichText)
            text.setWordWrap(True)
            text.setStyleSheet(
                f"color:{theme.TEXT_SECONDARY}; background:transparent; font-size:10px;"
            )
            grid.addWidget(text, row, 1)

        return frame

    def _update_size_label(self, value: int) -> None:
        self.size_label.setText(f"{value} px")

    # ------------------------------------------------------------------
    # Availability, driven by the viewport's actual contents
    # ------------------------------------------------------------------
    def set_layer_available(self, key: str, available: bool, count: int | None = None) -> None:
        row = self._rows.get(key)
        if row is not None:
            row.set_available(available, count)

    def sync_from_viewport(self, viewport) -> None:
        """Enable exactly the layers the viewport currently has data for."""
        self.set_layer_available("path", viewport.has_flight_path())
        self.set_layer_available("cameras", viewport.camera_count() > 0, viewport.camera_count())
        self.set_layer_available("points", viewport.point_count() > 0, viewport.point_count())
        self.set_layer_available("mesh", viewport.has_mesh(), viewport.mesh_face_count())
        self.set_layer_available(
            "measurements", viewport.measurement_count() > 0, viewport.measurement_count()
        )

    def is_checked(self, key: str) -> bool:
        row = self._rows.get(key)
        return bool(row and row.check.isChecked())
