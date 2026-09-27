"""The accuracy report card.

What changed and why
--------------------
This panel used to look up eight keys -- ``relative_rmse``,
``reprojection_error_px``, ``pct_measured`` and friends -- that *nothing
in the pipeline has ever produced*. The only code that emitted them was
``workers.DemoPipeline``. So the report card was fully populated in the
demo and blank after every real run, while ``results/v22/output/report.txt``
on disk held the real figures the whole time (relative RMSE 2.257 m,
scale error -0.80%, 54.8% measured).

The real schema is ``export.report.build_report``'s, and it is now
plumbed to the GUI (``ExportStage`` publishes it as
``artifacts["report_card"]`` and merges it into ``PipelineState.report``).
This panel reads that schema directly, so the numbers on screen and the
numbers in ``report.html`` are the same numbers.

Honesty rules, carried over from ``export.report``
--------------------------------------------------
- A metric that was not computed says **"not computed"**, greyed. It is
  never shown as 0, "n/a", or an em dash that could be mistaken for a
  measured zero.
- Absolute accuracy is displayed with its provenance note, because on a
  standalone-GPS flight it is bounded by an assumed bias rather than
  measured.
- A telemetry offset that was *assumed* rather than measured is flagged
  amber, since every georeferenced number below it inherits that guess.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import icons, theme

NOT_COMPUTED = "not computed"

#: (label, key, unit, format, tooltip). Matches ``export.report``'s own
#: key names exactly -- see the module docstring for why that matters.
_HEADLINE: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "Relative RMSE", "relative_rmse_m", "m", "{:.3f}",
        "Internal consistency of the reconstruction against the GPS track,\n"
        "after removing any global similarity transform.",
    ),
    (
        "Absolute RMSE", "absolute_rmse_m", "m", "{:.3f}",
        "Agreement with the georeferenced GPS positions.\n"
        "On a standalone-GPS flight this is bounded by the receiver's own\n"
        "bias, not by the reconstruction — see the georeferencing notes.",
    ),
    (
        "Scale error", "scale_error_pct", "%", "{:+.2f}",
        "How far the reconstruction's metric scale departs from the\n"
        "GPS-derived scale. Near zero is the whole point of anchoring to\n"
        "telemetry rather than reconstructing up to an unknown scale.",
    ),
    (
        "Reprojection", "mean_reprojection_error_px", "px", "{:.2f}",
        "Mean reprojection residual after bundle adjustment.\n"
        "Sub-pixel means the poses and the features agree.",
    ),
    (
        "Timeline coverage", "coverage_pct", "%", "{:.1f}",
        "Fraction of the flight timeline represented by a selected keyframe.",
    ),
    (
        "Keyframes", "keyframe_count", "", "{:.0f}",
        "Frames triage selected for reconstruction.",
    ),
)

#: Confidence tiers, read out of the nested ``confidence_breakdown_pct``.
_CONFIDENCE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Measured", "measured_pct", theme.CONF_MEASURED),
    ("Low confidence", "low_confidence_pct", theme.CONF_LOW),
    ("Inferred", "inferred_pct", theme.CONF_INFERRED),
)

#: Flat provenance/diagnostic rows shown under the headline tiles.
_DETAIL_ROWS: tuple[tuple[str, str], ...] = (
    ("Representation", "reconstruction_representation"),
    ("Confidence source", "confidence_source"),
    ("Merge strategy", "merge_strategy"),
    ("Merge reason", "merge_strategy_reason"),
    ("Telemetry format", "telemetry_format"),
    ("Telemetry offset (s)", "telemetry_offset_s"),
    ("Offset provenance", "telemetry_offset_source"),
    ("Telemetry coverage", "telemetry_video_coverage_fraction"),
    ("DTM method", "dtm_method"),
    ("Ground points (%)", "ground_point_pct"),
    ("DTM interpolated (%)", "dtm_interpolated_pct"),
    ("Dynamic points removed", "dynamic_points_removed"),
    ("Semantic labelled (%)", "semantic_labelled_pct"),
    ("Backbone", "backbone"),
)


def _is_computed(value: Any) -> bool:
    return value is not None and value != NOT_COMPUTED


#: Zero-width space. Inserted after underscores so a long snake_case
#: value ("backbone_confidence_and_view_count") has somewhere to wrap.
#: Without it QLabel cannot break the token at all and the value sets the
#: minimum width of the whole inspector column.
_ZWSP = "\u200b"


def _wrappable(text: str) -> str:
    return text.replace("_", "_" + _ZWSP)


def _format(value: Any, fmt: str = "{}") -> str:
    if not _is_computed(value):
        return NOT_COMPUTED
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        try:
            return fmt.format(value)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


class _Tile(QFrame):
    """One headline metric: big number, unit, label."""

    def __init__(self, label: str, unit: str, tooltip: str, parent=None) -> None:
        super().__init__(parent)
        self._unit = unit
        # Scoped by object name: QLabel derives from QFrame, so a bare
        # `QFrame{border:...}` rule would draw a box around the caption
        # and the value as well as around the tile.
        self.setObjectName("MetricTile")
        self.setStyleSheet(
            f"QFrame#MetricTile{{background:{theme.BG_RAISED}; border:1px solid {theme.BORDER};"
            f" border-radius:{theme.RADIUS}px;}}"
            f"QFrame#MetricTile QLabel{{border:none; background:transparent;}}"
        )
        self.setToolTip(tooltip)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        box = QVBoxLayout(self)
        box.setContentsMargins(11, 9, 11, 9)
        box.setSpacing(1)

        self.caption = QLabel(label.upper())
        self.caption.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:9px; font-weight:700;"
            f" letter-spacing:0.6px; background:transparent;"
        )
        box.addWidget(self.caption)

        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(3)

        self.value = QLabel(NOT_COMPUTED)
        self.value.setFont(theme.mono_font(17, weight=700))
        self.value.setStyleSheet(f"color:{theme.TEXT_FAINT}; background:transparent;")
        value_row.addWidget(self.value, 0, Qt.AlignBottom)

        self.unit = QLabel(unit)
        self.unit.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:10px; background:transparent; padding-bottom:2px;"
        )
        value_row.addWidget(self.unit, 0, Qt.AlignBottom)
        value_row.addStretch(1)
        box.addLayout(value_row)

    def set_value(self, raw: Any, fmt: str) -> None:
        if _is_computed(raw):
            self.value.setText(_format(raw, fmt))
            self.value.setFont(theme.mono_font(17, weight=700))
            self.value.setStyleSheet(f"color:{theme.TEXT_PRIMARY}; background:transparent;")
            self.unit.setText(self._unit)
            self.unit.setStyleSheet(
                f"color:{theme.TEXT_MUTED}; font-size:10px; background:transparent; padding-bottom:2px;"
            )
        else:
            self.value.setText(NOT_COMPUTED)
            self.value.setFont(theme.ui_font(11))
            self.value.setStyleSheet(f"color:{theme.TEXT_FAINT}; background:transparent;")
            self.unit.setText("")


class _ConfidenceBar(QWidget):
    """A single stacked bar: measured / low-confidence / inferred."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pct: dict[str, float] = {}

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)

        self.bar = QFrame()
        self.bar.setObjectName("ConfidenceBar")
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet(
            f"QFrame#ConfidenceBar{{background:{theme.BG_DARK};"
            f" border:1px solid {theme.BORDER}; border-radius:{theme.RADIUS}px;}}"
        )
        self._bar_row = QHBoxLayout(self.bar)
        self._bar_row.setContentsMargins(1, 1, 1, 1)
        self._bar_row.setSpacing(0)
        box.addWidget(self.bar)

        self._segments: dict[str, QFrame] = {}
        for _label, key, color in _CONFIDENCE_ROWS:
            segment = QFrame()
            segment.setStyleSheet(f"background:{color}; border:none;")
            self._bar_row.addWidget(segment, 0)
            self._segments[key] = segment
        self._spacer = QFrame()
        self._spacer.setStyleSheet("background:transparent; border:none;")
        self._bar_row.addWidget(self._spacer, 1)

        legend = QHBoxLayout()
        legend.setContentsMargins(0, 0, 0, 0)
        legend.setSpacing(14)
        self._legend_labels: dict[str, QLabel] = {}
        for label, key, color in _CONFIDENCE_ROWS:
            item = QLabel(f"● {label} —")
            item.setStyleSheet(f"color:{color}; font-size:10px;")
            legend.addWidget(item)
            self._legend_labels[key] = item
        legend.addStretch(1)
        box.addLayout(legend)

    def set_breakdown(self, breakdown: Any) -> None:
        computed = isinstance(breakdown, dict)
        for label, key, color in _CONFIDENCE_ROWS:
            pct = float(breakdown.get(key, 0.0)) if computed else 0.0
            # Integer stretch factors: a stacked bar built from layout
            # stretch needs whole numbers, and 0.1% resolution is finer
            # than a 200 px bar can show anyway.
            self._bar_row.setStretch(
                self._bar_row.indexOf(self._segments[key]), max(0, int(round(pct * 10)))
            )
            self._legend_labels[key].setText(
                f"● {label} {pct:.1f}%" if computed else f"● {label} —"
            )
            self._legend_labels[key].setStyleSheet(
                f"color:{color if computed else theme.TEXT_FAINT}; font-size:10px;"
            )
        self._bar_row.setStretch(self._bar_row.indexOf(self._spacer), 0 if computed else 1)


class ReportPanel(QWidget):
    """Accuracy report card, rendered from ``export.report.build_report``'s schema."""

    exportRequested = Signal(str)  # export path chosen by the operator

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # --- headline tiles ---------------------------------------------
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._tiles: dict[str, tuple[_Tile, str]] = {}
        for index, (label, key, unit, fmt, tip) in enumerate(_HEADLINE):
            tile = _Tile(label, unit, tip)
            grid.addWidget(tile, index // 2, index % 2)
            self._tiles[key] = (tile, fmt)
        layout.addLayout(grid)

        # --- provenance banner -------------------------------------------
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        layout.addWidget(self.banner)

        # --- confidence ----------------------------------------------------
        layout.addWidget(self._heading("CONFIDENCE BREAKDOWN"))
        self.confidence = _ConfidenceBar()
        self.confidence.setToolTip(
            "MEASURED: directly observed by at least two cameras.\n"
            "LOW_CONFIDENCE: seen at grazing incidence or by few views.\n"
            "INFERRED: never observed; filled in by the model's prior.\n"
            "Measuring inferred geometry is disallowed by default."
        )
        layout.addWidget(self.confidence)

        # --- details -------------------------------------------------------
        layout.addWidget(self._heading("PROVENANCE & DIAGNOSTICS"))
        self._detail_grid = QGridLayout()
        self._detail_grid.setContentsMargins(0, 0, 0, 0)
        self._detail_grid.setHorizontalSpacing(10)
        self._detail_grid.setVerticalSpacing(3)
        self._detail_grid.setColumnStretch(1, 1)
        self._detail_values: dict[str, QLabel] = {}
        for row, (label, key) in enumerate(_DETAIL_ROWS):
            name = QLabel(label)
            name.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
            name.setMinimumWidth(0)
            value = QLabel(NOT_COMPUTED)
            value.setFont(theme.mono_font(10))
            value.setStyleSheet(f"color:{theme.TEXT_FAINT};")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            # Wrap rather than widen: a value like
            # "backbone_confidence_and_view_count" is 34 monospace
            # characters and would otherwise set the minimum width of the
            # entire inspector column.
            value.setWordWrap(True)
            value.setMinimumWidth(0)
            self._detail_grid.addWidget(name, row, 0, Qt.AlignLeft | Qt.AlignVCenter)
            self._detail_grid.addWidget(value, row, 1, Qt.AlignRight | Qt.AlignVCenter)
            self._detail_values[key] = value
        layout.addLayout(self._detail_grid)

        # --- notes ----------------------------------------------------------
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:10px;")
        self.notes.setVisible(False)
        layout.addWidget(self.notes)

        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # --- actions ---------------------------------------------------------
        actions = QHBoxLayout()
        actions.setContentsMargins(10, 8, 10, 10)
        actions.setSpacing(6)
        self.export_button = QPushButton(icons.icon("export", 14), "  Export report")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._browse_export)
        actions.addWidget(self.export_button)
        actions.addStretch(1)
        outer.addLayout(actions)

    # ------------------------------------------------------------------
    def _heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:9px; font-weight:700;"
            f" letter-spacing:0.7px; padding-top:4px;"
        )
        return label

    # ------------------------------------------------------------------
    def set_report(self, report: dict) -> None:
        """Populate from a ``build_report``-shaped dict (extra keys ignored)."""
        self._report = dict(report or {})

        for key, (tile, fmt) in self._tiles.items():
            tile.set_value(self._report.get(key), fmt)

        self.confidence.set_breakdown(self._report.get("confidence_breakdown_pct"))

        for key, label in self._detail_values.items():
            value = self._report.get(key)
            label.setText(_wrappable(_format(value, "{:.4g}")))
            label.setToolTip(_format(value, "{}"))
            label.setStyleSheet(
                f"color:{theme.TEXT_SECONDARY};" if _is_computed(value) else f"color:{theme.TEXT_FAINT};"
            )

        self._update_banner()
        self._update_notes()
        self.export_button.setEnabled(bool(self._report))

    def _update_banner(self) -> None:
        """Flag a run whose georeferencing rests on a guess, not a measurement."""
        source = self._report.get("telemetry_offset_source")
        fmt = self._report.get("telemetry_format")
        cancelled = self._report.get("cancelled")

        outcome = self._report.get("outcome")
        if outcome in {"failed", "unverified", "cancelled"}:
            self._show_banner(
                f"{outcome.upper()} — DIAGNOSTIC OUTPUT. "
                + " ".join(self._report.get("quality_reasons") or []),
                theme.WARN, theme.WARN_DIM,
            )
            return

        if cancelled:
            self._show_banner(
                "This run was cancelled. Every figure below describes the partial "
                "result that existed when it stopped.",
                theme.WARN,
                theme.WARN_DIM,
            )
            return
        if source == "assumed_zero" and fmt == "csv":
            self._show_banner(
                "Telemetry/video sync was ASSUMED, not measured: no isVideo column "
                "matched this clip. Every georeferenced figure below inherits that "
                "assumption.",
                theme.WARN,
                theme.WARN_DIM,
            )
            return
        self.banner.setVisible(False)

    def _show_banner(self, text: str, color: str, border: str) -> None:
        self.banner.setText(text)
        self.banner.setStyleSheet(
            f"color:{color}; background:{theme.BG_PANEL}; border:1px solid {border};"
            f" border-radius:{theme.RADIUS}px; padding:7px 9px; font-size:10px;"
        )
        self.banner.setVisible(True)

    def _update_notes(self) -> None:
        notes = self._report.get("georef_notes")
        placement = self._report.get("placement")

        lines: list[str] = []
        if _is_computed(notes):
            lines.extend(notes if isinstance(notes, list) else [str(notes)])
        if isinstance(placement, dict) and placement.get("verdict", "FAIL" if placement.get("passed") is False else "PASS") == "FAIL":
            reason = placement.get("reason") or "windows disagree on where the ground is"
            lines.append(f"Placement check FAILED: {reason}")

        if lines:
            self.notes.setText("\n".join(f"· {line}" for line in lines))
            self.notes.setVisible(True)
        else:
            self.notes.setVisible(False)

    def report(self) -> dict:
        return dict(self._report)

    # ------------------------------------------------------------------
    def _browse_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "drishti3d_report.txt", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            self.export_report(path)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.exportRequested.emit(path)

    def export_report(self, path: str) -> None:
        """Write the current report to a plain-text file.

        Delegates to ``export.report.render_report_text`` so the file the
        operator saves from the GUI is byte-identical to the
        ``report.txt`` the pipeline writes.
        """
        from drishti3d.export.report import render_report_text

        with open(path, "w") as handle:
            handle.write(render_report_text(self._report))
