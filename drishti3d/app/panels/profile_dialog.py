"""Elevation profile chart for a profile measurement (distance along the line vs surface elevation).

Drawn with QPainter rather than a charting library: matplotlib is excluded
from the packaged app, and a line chart with two labelled axes needs
nothing more.
"""

from __future__ import annotations

import csv

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from drishti3d.app import theme


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values spanning [lo, hi]."""
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    start = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(start, hi + step * 0.5, step)]


class ProfileChart(QWidget):
    """Distance (m) against elevation (m); gaps where the line crosses no surface."""

    def __init__(self, distance_m: np.ndarray, elevation_m: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.distance = np.asarray(distance_m, dtype=np.float64)
        self.elevation = np.asarray(elevation_m, dtype=np.float64)
        self.setMinimumSize(560, 260)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.BG_VOID))
        ok = np.isfinite(self.elevation)
        if ok.sum() < 2:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, "The line crosses no reconstructed surface.")
            return
        left, right, top, bottom = 58, 16, 14, 34
        plot = QRectF(left, top, self.width() - left - right, self.height() - top - bottom)
        d0, d1 = float(self.distance[0]), float(self.distance[-1])
        z = self.elevation[ok]
        pad = max(0.5, 0.08 * float(z.max() - z.min()))
        z0, z1 = float(z.min()) - pad, float(z.max()) + pad

        def to_px(d: float, h: float) -> QPointF:
            return QPointF(plot.left() + (d - d0) / max(d1 - d0, 1e-9) * plot.width(),
                           plot.bottom() - (h - z0) / max(z1 - z0, 1e-9) * plot.height())

        grid_pen = QPen(QColor(theme.TEXT_FAINT))
        grid_pen.setWidthF(0.6)
        text = QColor(theme.TEXT_MUTED)
        for v in _nice_ticks(z0, z1):
            p = to_px(d0, v)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot.left(), p.y()), QPointF(plot.right(), p.y()))
            painter.setPen(text)
            painter.drawText(QRectF(0, p.y() - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:,.1f}")
        for v in _nice_ticks(d0, d1):
            p = to_px(v, z0)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(p.x(), plot.top()), QPointF(p.x(), plot.bottom()))
            painter.setPen(text)
            painter.drawText(QRectF(p.x() - 30, plot.bottom() + 4, 60, 16), Qt.AlignHCenter, f"{v:,.0f}")
        painter.drawText(QRectF(plot.left(), self.height() - 16, plot.width(), 16), Qt.AlignHCenter,
                         "distance along the line (m)")
        painter.save()
        painter.translate(12, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-60, -8, 120, 16), Qt.AlignCenter, "elevation (m)")
        painter.restore()

        # The profile, broken where the line leaves the surface; filled below.
        fill = QColor(theme.ACCENT)
        fill.setAlpha(60)
        line_pen = QPen(QColor(theme.ACCENT))
        line_pen.setWidthF(2.0)
        runs, run = [], []
        for d, h in zip(self.distance, self.elevation, strict=True):
            if np.isfinite(h):
                run.append((float(d), float(h)))
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)
        for r in runs:
            if len(r) < 2:
                continue
            path = QPainterPath(to_px(*r[0]))
            for d, h in r[1:]:
                path.lineTo(to_px(d, h))
            area = QPainterPath(path)
            area.lineTo(to_px(r[-1][0], z0))
            area.lineTo(to_px(r[0][0], z0))
            area.closeSubpath()
            painter.fillPath(area, fill)
            painter.setPen(line_pen)
            painter.drawPath(path)


class ProfileDialog(QDialog):
    """The chart, the profile's summary, and CSV export."""

    def __init__(self, measurement, datum_m: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(measurement.label)
        self._distance = np.asarray(measurement.extra.get("_distance"), dtype=np.float64)
        self._elevation = np.asarray(measurement.extra.get("_z"), dtype=np.float64) + float(datum_m)
        self._xy = np.asarray(measurement.extra.get("_xy"), dtype=np.float64)
        e = measurement.extra
        summary = QLabel(
            f"Length {e.get('length_m', 0.0):,.1f} m  ·  elevation {e.get('min_elevation_m', float('nan')):,.1f}"
            f"–{e.get('max_elevation_m', float('nan')):,.1f} m  ·  climb {e.get('climb_m', 0.0):,.1f} m, "
            f"descent {e.get('descent_m', 0.0):,.1f} m  ·  steepest {e.get('max_slope_deg', 0.0):.0f}°"
            + ("" if e.get("valid_fraction", 1.0) >= 0.999 else
               f"  ·  {100 * (1 - e.get('valid_fraction', 1.0)):.0f}% of the line crosses no surface")
        )
        summary.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(ProfileChart(self._distance, self._elevation, self))
        layout.addWidget(summary)
        if measurement.warning:
            warn = QLabel(measurement.warning)
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {theme.WARN};")
            layout.addWidget(warn)
        row = QHBoxLayout()
        row.addStretch(1)
        save = QPushButton("Save CSV…")
        save.clicked.connect(self._save_csv)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(save)
        row.addWidget(close)
        layout.addLayout(row)
        self.resize(720, 420)

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Profile", f"{self.windowTitle()}.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["distance_m", "x_m", "y_m", "elevation_m"])
            for d, (x, y), h in zip(self._distance, self._xy, self._elevation, strict=False):
                w.writerow([f"{d:.3f}", f"{x:.3f}", f"{y:.3f}", "" if not np.isfinite(h) else f"{h:.3f}"])
