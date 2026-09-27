"""Tools > Preferences: the chrome theme and the typeface pairing.

Two tabs, Themes and Type, each a grid of cards that preview the choice
in its own colours or faces, so the operator picks by looking rather than
by reading names. Apply re-themes the running window; OK applies and
closes. Choices persist through ``theme.save_preferences`` (QSettings).
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drishti3d.app import theme

__all__ = ["PreferencesDialog"]


class _Card(QAbstractButton):
    """A checkable, self-painted preview card."""

    WIDTH = 300
    HEIGHT = 150

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(self.WIDTH, self.HEIGHT)
        self.setFocusPolicy(Qt.StrongFocus)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(self.WIDTH, self.HEIGHT)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().leaveEvent(event)

    def _frame(self, painter: QPainter) -> QRectF:
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self.isChecked():
            border = QColor(theme.ACCENT)
        elif self.underMouse() or self.hasFocus():
            border = QColor(theme.TEXT_FAINT)
        else:
            border = QColor(theme.BORDER_STRONG)
        painter.setPen(QPen(border, 2 if self.isChecked() else 1))
        painter.setBrush(QColor(theme.BG_RAISED if self.isChecked() else theme.BG_PANEL))
        painter.drawRoundedRect(rect, 5, 5)
        return rect

    def _tick(self, painter: QPainter, rect: QRectF) -> None:
        if not self.isChecked():
            return
        dot = QRectF(rect.right() - 26, rect.top() + 10, 16, 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawEllipse(dot)
        pen = QPen(QColor(theme.ON_ACCENT), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(dot.left() + 4.5, dot.center().y(), dot.left() + 7, dot.bottom() - 4.5)
        painter.drawLine(dot.left() + 7, dot.bottom() - 4.5, dot.right() - 4, dot.top() + 5)


class _PaletteCard(_Card):
    """A palette, previewed in its own colours."""

    def __init__(self, palette: theme.Palette, parent: QWidget | None = None) -> None:
        super().__init__(palette.key, parent)
        self.palette_ = palette
        self.setToolTip(palette.description)
        self.setAccessibleName(f"{palette.name} theme")

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        t = self.palette_.tokens
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self._frame(painter)

        # Mini window: the palette's own chrome.
        win = QRectF(rect.left() + 12, rect.top() + 12, 132, rect.height() - 24)
        painter.setPen(QPen(QColor(t["BORDER_STRONG"]), 1))
        painter.setBrush(QColor(t["BG_DARKEST"]))
        painter.drawRoundedRect(win, 3, 3)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t["BG_PANEL"]))
        painter.drawRect(QRectF(win.left() + 1, win.top() + 16, 30, win.height() - 17))
        painter.drawRect(QRectF(win.right() - 31, win.top() + 16, 30, win.height() - 17))
        painter.setBrush(QColor(t["BG_VOID"]))
        painter.drawRect(QRectF(win.left() + 32, win.top() + 16, win.width() - 64, win.height() - 17))
        # Rail rows, the running one lit.
        for i in range(5):
            y = win.top() + 24 + i * 11
            painter.setBrush(QColor(theme.OK if i < 3 else (t["ACCENT"] if i == 3 else t["TEXT_FAINT"])))
            painter.drawEllipse(QRectF(win.left() + 5, y, 4, 4))
            painter.setBrush(QColor(t["TEXT_SECONDARY"] if i < 4 else t["TEXT_FAINT"]))
            painter.drawRect(QRectF(win.left() + 12, y + 1, 14, 2))
        # Run button + accent path in the viewport.
        painter.setBrush(QColor(t["ACCENT"]))
        painter.drawRoundedRect(QRectF(win.right() - 30, win.top() + 4, 24, 8), 2, 2)
        pen = QPen(QColor(t["ACCENT"]), 1.4)
        painter.setPen(pen)
        vx, vy, vw, vh = win.left() + 36, win.top() + 22, win.width() - 72, win.height() - 30
        painter.drawLine(vx, vy + vh * 0.7, vx + vw * 0.35, vy + vh * 0.35)
        painter.drawLine(vx + vw * 0.35, vy + vh * 0.35, vx + vw * 0.7, vy + vh * 0.55)
        painter.drawLine(vx + vw * 0.7, vy + vh * 0.55, vx + vw, vy + vh * 0.15)

        # Ramp.
        ramp_keys = ("BG_VOID", "BG_DARKEST", "BG_DARK", "BG_PANEL", "BG_RAISED", "BG_HOVER", "BG_ACTIVE")
        x0 = win.right() + 14
        ramp_w = rect.right() - 14 - x0
        step = ramp_w / len(ramp_keys)
        painter.setPen(Qt.NoPen)
        for i, key in enumerate(ramp_keys):
            painter.setBrush(QColor(t[key]))
            painter.drawRect(QRectF(x0 + i * step, rect.bottom() - 26, step + 0.5, 14))
        painter.setBrush(QColor(t["ACCENT"]))
        painter.drawRect(QRectF(x0, rect.bottom() - 32, ramp_w * 0.55, 3))
        painter.setBrush(QColor(t["ACCENT_DIM"]))
        painter.drawRect(QRectF(x0 + ramp_w * 0.55 + 3, rect.bottom() - 32, ramp_w * 0.2, 3))

        # Words, in the dialog's own theme.
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.setFont(theme.ui_font(14, QFont.Weight.DemiBold))
        painter.drawText(QRectF(x0, rect.top() + 12, ramp_w - 20, 20), Qt.AlignLeft | Qt.AlignVCenter, self.palette_.name)
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.setFont(theme.ui_font(11))
        painter.drawText(QRectF(x0, rect.top() + 32, ramp_w, 16), Qt.AlignLeft | Qt.AlignVCenter, self.palette_.reference)
        painter.setFont(theme.mono_font(10))
        painter.drawText(
            QRectF(x0, rect.top() + 52, ramp_w, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            t["ACCENT"].upper(),
        )
        self._tick(painter, rect)
        painter.end()


class _TypeCard(_Card):
    """A typeface pairing, previewed in its own faces."""

    def __init__(self, face: theme.Typeface, parent: QWidget | None = None) -> None:
        super().__init__(face.key, parent)
        self.face = face
        self.setToolTip(face.description)
        self.setAccessibleName(f"{face.name} typeface")

    def _font(self, role: str, size: int, weight=QFont.Weight.Normal) -> QFont:
        family = self.face.ui_family if role == "ui" else self.face.mono_family
        if family:
            font = QFont(family)
            if role == "mono":
                font.setStyleHint(QFont.Monospace)
        else:
            from PySide6.QtGui import QFontDatabase

            font = QFontDatabase.systemFont(
                QFontDatabase.GeneralFont if role == "ui" else QFontDatabase.FixedFont
            )
        font.setPixelSize(size)
        font.setWeight(weight)
        return font

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        rect = self._frame(painter)
        inner = rect.adjusted(14, 12, -14, -12)

        painter.setPen(QColor(theme.ACCENT))
        painter.setFont(theme.ui_font(10, QFont.Weight.DemiBold))
        painter.drawText(QRectF(inner.left(), inner.top(), inner.width(), 14), Qt.AlignLeft, self.face.name.upper())

        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.setFont(self._font("ui", 22, QFont.Weight.DemiBold))
        painter.drawText(QRectF(inner.left(), inner.top() + 18, inner.width(), 30), Qt.AlignLeft | Qt.AlignVCenter, "Fusion · TSDF surface")

        # A stage-rail row: label in the UI face, figures in the mono.
        well = QRectF(inner.left(), inner.top() + 56, inner.width(), 40)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.setBrush(QColor(theme.BG_DARK))
        painter.drawRoundedRect(well, 3, 3)
        rows = (("Pose prior", "07:07.0"), ("Point cloud", "3,241,128"))
        for i, (label, value) in enumerate(rows):
            y = well.top() + 4 + i * 16
            painter.setPen(QColor(theme.TEXT_PRIMARY))
            painter.setFont(self._font("ui", 12))
            painter.drawText(QRectF(well.left() + 8, y, 120, 16), Qt.AlignLeft | Qt.AlignVCenter, label)
            painter.setPen(QColor(theme.TEXT_SECONDARY))
            painter.setFont(self._font("mono", 11))
            painter.drawText(QRectF(well.left(), y, well.width() - 8, 16), Qt.AlignRight | Qt.AlignVCenter, value)

        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.setFont(theme.mono_font(10))
        pair = (
            f"{self.face.ui_family} + {self.face.mono_family}"
            if self.face.ui_family
            else "Platform UI + platform mono"
        )
        painter.drawText(QRectF(inner.left(), inner.bottom() - 16, inner.width(), 16), Qt.AlignLeft | Qt.AlignVCenter, pair)
        self._tick(painter, rect)
        painter.end()


class PreferencesDialog(QDialog):
    """Themes and Type. Emits ``appearanceApplied(palette_key, typeface_key)``."""

    appearanceApplied = Signal(str, str)

    def __init__(self, parent: QWidget | None = None, run_in_progress: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.palette_group, themes_page = self._grid(
            [_PaletteCard(p) for p in theme.PALETTES.values()], theme.current_palette()
        )
        self.type_group, type_page = self._grid(
            [_TypeCard(t) for t in theme.TYPEFACES.values()], theme.current_typeface()
        )
        self.tabs.addTab(themes_page, "Themes")
        self.tabs.addTab(type_page, "Type")

        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setObjectName("Hint")
        self.description.setMinimumHeight(34)

        self.note = QLabel(
            "A reconstruction is running. Your choice is saved now and the window "
            "re-themes as soon as the run finishes."
            if run_in_progress
            else "Colours for Measured, Low and Inferred confidence stay the same in every theme."
        )
        self.note.setWordWrap(True)
        self.note.setProperty("warning" if run_in_progress else "muted", True)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.RestoreDefaults
            | QDialogButtonBox.Cancel
            | QDialogButtonBox.Apply
            | QDialogButtonBox.Ok
        )
        self.buttons.button(QDialogButtonBox.Ok).setObjectName("Primary")
        self.buttons.button(QDialogButtonBox.RestoreDefaults).setText("Reset to Defaults")
        self.buttons.accepted.connect(self._ok)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._defaults)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 14)
        layout.setSpacing(10)
        layout.addWidget(self.tabs)
        for widget in (self.description, self.note):
            wrapper = QVBoxLayout()
            wrapper.setContentsMargins(18, 0, 18, 0)
            wrapper.addWidget(widget)
            layout.addLayout(wrapper)
        button_row = QVBoxLayout()
        button_row.setContentsMargins(18, 4, 18, 0)
        button_row.addWidget(self.buttons)
        layout.addLayout(button_row)

        self.palette_group.buttonToggled.connect(self._describe)
        self.type_group.buttonToggled.connect(self._describe)
        self.tabs.currentChanged.connect(lambda _i: self._describe())
        self._describe()

    # ------------------------------------------------------------------
    def _grid(self, cards: list[_Card], selected: str) -> tuple[QButtonGroup, QWidget]:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(18, 18, 18, 8)
        grid.setSpacing(12)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for index, card in enumerate(cards):
            group.addButton(card)
            grid.addWidget(card, index // 2, index % 2)
            card.setChecked(card.key == selected)
        grid.setRowStretch(grid.rowCount(), 1)
        return group, page

    @staticmethod
    def _key(group: QButtonGroup, fallback: str) -> str:
        button = group.checkedButton()
        return button.key if button is not None else fallback

    @property
    def selected_palette(self) -> str:
        return self._key(self.palette_group, theme.DEFAULT_PALETTE)

    @property
    def selected_typeface(self) -> str:
        return self._key(self.type_group, theme.DEFAULT_TYPEFACE)

    def _describe(self, *_args) -> None:
        if self.tabs.currentIndex() == 0:
            p = theme.PALETTES[self.selected_palette]
            self.description.setText(f"<b>{p.name}</b> — {p.description}")
        else:
            t = theme.TYPEFACES[self.selected_typeface]
            self.description.setText(f"<b>{t.name}</b> — {t.description}")

    def _select(self, group: QButtonGroup, key: str) -> None:
        for button in group.buttons():
            if button.key == key:
                button.setChecked(True)

    def _defaults(self) -> None:
        self._select(self.palette_group, theme.DEFAULT_PALETTE)
        self._select(self.type_group, theme.DEFAULT_TYPEFACE)

    def _apply(self) -> None:
        self.appearanceApplied.emit(self.selected_palette, self.selected_typeface)
        # The dialog itself is re-styled by the new app stylesheet; the
        # cards paint from theme tokens, so a repaint is enough.
        for button in self.palette_group.buttons() + self.type_group.buttons():
            button.update()
        self._describe()

    def _ok(self) -> None:
        if (self.selected_palette, self.selected_typeface) != (
            theme.current_palette(),
            theme.current_typeface(),
        ):
            self._apply()
        self.accept()
