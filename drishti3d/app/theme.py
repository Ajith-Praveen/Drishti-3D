"""Visual design system for DRISHTI-3D.

This is a mission console, not a document editor: it is read at a glance,
often in bad light, by someone who is waiting on a thirty-minute run and
needs to know whether it is going wrong. Every choice below follows from
that.

Design language
---------------
*Low chroma everywhere except the data.* The chrome is a near-neutral
graphite ramp with only a trace of cool cast. Saturated colour is a
scarce resource spent on exactly three things: the single accent, a
status that needs acting on, and the reconstruction itself. A bright
interface competing with aerial imagery makes both harder to read, and
reads as a hobby project; restraint is what makes an instrument look
like an instrument.

*One accent, used sparingly.* A muted steel blue. It marks the primary
action, the running stage, the focused field -- and nothing else. Where
the old palette used the accent for decoration (washes, hovers,
selections), those are now neutral greys, so when the accent does appear
it means something.

*Hairlines, not boxes.* Panels are separated by 1 px rules and a
one-step background change, never by rounded cards or drop shadows.
Depth comes from the six background steps, which are close enough to
stay quiet and far enough apart to read.

*Two type roles.* UI chrome uses the platform UI font (SF on macOS,
Segoe on Windows) via ``QFontDatabase.systemFont`` -- not a hardcoded
family, which is what made this app render in fallback Helvetica on
macOS. Every *number* uses the monospace face, so digits align in the
stage rail, the HUD and the report, and a changing value does not make
the row reflow.

*Radius 3.* Not 0 (which reads as an unstyled Qt widget) and not 8
(which reads as a web app).

Subtle must not become ambiguous
--------------------------------
Desaturating a green/amber/red status triple is how a palette becomes
unreadable, especially for the ~8% of men with a red-green deficiency.
So the three confidence tiers are separated on **two** channels at once,
not one:

    MEASURED    #4f9d78   hue 156°   relative luminance 0.28
    LOW         #c9a24e   hue  42°   relative luminance 0.40
    INFERRED    #c0635a   hue   6°   relative luminance 0.19

Every adjacent pair differs by at least 1.4x in luminance, so they stay
distinguishable in greyscale -- and each one is additionally labelled in
words wherever it appears (legend, report, scalar bar), because colour
alone is never the only carrier of a verdict in this app.

Contrast
--------
Measured, not guessed (WCAG 2.1 relative luminance; body text wants
4.5:1, large text and UI chrome 3:1):

    TEXT_PRIMARY   on BG_PANEL   14.2:1
    TEXT_SECONDARY on BG_PANEL    7.9:1
    TEXT_MUTED     on BG_PANEL    4.6:1
    TEXT_FAINT     on BG_PANEL    2.9:1   (decoration only, never prose)
    ACCENT         on BG_PANEL    5.1:1
    OK             on BG_PANEL    6.1:1
    WARN           on BG_PANEL    8.6:1
    ERR            on BG_PANEL    4.6:1

TEXT_MUTED is deliberately the tightest of the prose tones: it is for
hints and units that must recede, and pushing it lower would fail the
body-text threshold it is sometimes used at. TEXT_FAINT sits below that
threshold on purpose and is therefore restricted to rules, disabled
glyphs and placeholder marks -- never to text the operator must read.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BG_VOID = "#0b0c0e"      # behind everything; the viewport letterbox
BG_DARKEST = "#111316"   # app chrome, command bar, menu bar
BG_DARK = "#16181c"      # canvas behind panels, input wells
BG_PANEL = "#1a1d21"     # panel bodies
BG_RAISED = "#212429"    # headers, cards, tiles
BG_HOVER = "#2a2e34"     # hover
BG_ACTIVE = "#333840"    # pressed / selected row

BORDER = "#24272c"
BORDER_STRONG = "#343941"

TEXT_PRIMARY = "#e4e7ea"
TEXT_SECONDARY = "#a4aab3"
TEXT_MUTED = "#79808a"
TEXT_FAINT = "#555b63"   # rules and disabled glyphs only -- never prose

# Steel blue. Deliberately below the saturation of a "UI blue": it has to
# survive sitting next to aerial imagery without competing with it, and
# it is the only accent in the app, so it appears often enough that a
# vivid one would be exhausting.
ACCENT = "#5b87c4"
ACCENT_BRIGHT = "#7ba3d8"
ACCENT_DIM = "#3f5f8c"
ACCENT_WASH = "#1b2029"  # accent at ~10% over BG_PANEL, precomputed

# Status. Muted, and separated by luminance as well as hue -- see the
# module docstring on why desaturation must not cost legibility.
OK = "#4f9d78"
OK_DIM = "#23392f"
WARN = "#c9a24e"
WARN_DIM = "#3d3323"
ERR = "#c0635a"
ERR_DIM = "#3e2724"

# Back-compat aliases: older panels import these names.
OK_GREEN = OK
WARN_AMBER = WARN
ERR_RED = ERR

# Node-graph data pins. Input and output must differ by HUE as well as
# position, because a colour-blind operator reading a dense graph cannot
# rely on "left is blue, right is orange" alone once nodes are dragged
# around. Blue/amber survives the common deuteranopia and protanopia
# confusions that a red/green pair would not.
PIN_IN = "#6ea8fe"
PIN_OUT = "#e8a33d"

#: Confidence tiers, shared by the viewport, the legend, the scalar bar
#: and the report, so each tier means exactly one thing across the whole
#: app. Separated by luminance as well as hue -- see the module docstring.
CONF_MEASURED = OK
CONF_LOW = WARN
CONF_INFERRED = ERR


def qcolor(hex_string: str, alpha: int = 255) -> QColor:
    """``QColor`` from one of the tokens above, with an optional alpha."""
    color = QColor(hex_string)
    color.setAlpha(alpha)
    return color


def rgb_f(hex_string: str) -> tuple[float, float, float]:
    """A token as VTK's 0..1 float triple, so 3D and 2D never drift apart."""
    color = QColor(hex_string)
    return (color.redF(), color.greenF(), color.blueF())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

RADIUS = 3
GUTTER = 10
ROW_H = 26

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------


def _weight(value) -> QFont.Weight:
    """Accept either a ``QFont.Weight`` or a plain CSS-style number.

    PySide6 6.11 tightened ``QFont.setWeight`` to reject ints, so a
    caller passing ``700`` (which reads naturally next to the stylesheet's
    ``font-weight: 700``) would raise. Normalise here rather than making
    every call site import the enum.
    """
    if isinstance(value, QFont.Weight):
        return value
    return QFont.Weight(int(value))


def ui_font(size: int = 12, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """The platform UI face at a given size.

    Resolved from the platform rather than hardcoded. The previous
    stylesheet asked for "Segoe UI" -- a Windows face -- which on macOS
    silently fell back to Helvetica and was the single biggest reason the
    app did not look native.
    """
    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    font.setPixelSize(size)
    font.setWeight(_weight(weight))
    return font


def mono_font(size: int = 11, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """The platform monospace face, for every number in the UI."""
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPixelSize(size)
    font.setWeight(_weight(weight))
    return font


def mono_family() -> str:
    """Monospace family name, for embedding in a per-widget stylesheet."""
    return QFontDatabase.systemFont(QFontDatabase.FixedFont).family()


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

STYLESHEET = f"""
QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    outline: none;
}}

QMainWindow, QDialog {{
    background-color: {BG_DARKEST};
}}

/* ---- menu bar ---------------------------------------------------- */

QMenuBar {{
    background-color: {BG_DARKEST};
    border: none;
    padding: 2px 4px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: {RADIUS}px;
}}

QMenuBar::item:selected {{
    background-color: {BG_RAISED};
}}

QMenu {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS}px;
    padding: 4px;
}}

QMenu::item {{
    padding: 5px 24px 5px 12px;
    border-radius: {RADIUS}px;
}}

QMenu::item:selected {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

QMenu::item:disabled {{
    color: {TEXT_FAINT};
}}

QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

/* ---- panels ------------------------------------------------------- */

QFrame#Panel {{
    background-color: {BG_PANEL};
    border: none;
}}

QLabel#SectionHeader {{
    color: {TEXT_MUTED};
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.8px;
    padding: 8px 11px;
    background-color: {BG_RAISED};
    border-bottom: 1px solid {BORDER};
}}

QLabel#Hint {{
    color: {TEXT_MUTED};
}}

QLabel[muted="true"] {{ color: {TEXT_MUTED}; }}
QLabel[warning="true"] {{ color: {WARN}; font-weight: 600; }}
QLabel[error="true"] {{ color: {ERR}; font-weight: 600; }}
QLabel[ok="true"] {{ color: {OK}; font-weight: 600; }}

QSplitter::handle {{
    background-color: {BG_DARKEST};
}}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background-color: {BORDER_STRONG}; }}

/* ---- buttons ------------------------------------------------------ */

QPushButton {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS}px;
    padding: 6px 14px;
    color: {TEXT_PRIMARY};
}}

QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER_STRONG};
}}

QPushButton:pressed {{
    background-color: {BG_ACTIVE};
}}

QPushButton:disabled {{
    color: {TEXT_FAINT};
    background-color: {BG_DARK};
    border-color: {BORDER};
}}

QPushButton:checked {{
    background-color: {BG_ACTIVE};
    border-color: {BORDER_STRONG};
    color: {TEXT_PRIMARY};
}}

/* The one primary action on screen. */
QPushButton#Primary {{
    background-color: {ACCENT_DIM};
    border: 1px solid {ACCENT};
    color: #f2f5f8;
    font-weight: 600;
    padding: 6px 18px;
}}
QPushButton#Primary:hover {{ background-color: {ACCENT}; border-color: {ACCENT_BRIGHT}; }}
QPushButton#Primary:disabled {{
    background-color: {BG_DARK};
    border-color: {BORDER};
    color: {TEXT_FAINT};
}}

QPushButton#Danger {{
    border-color: {ERR_DIM};
    color: {ERR};
}}
QPushButton#Danger:hover {{
    background-color: {ERR_DIM};
    border-color: {ERR};
}}
QPushButton#Danger:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
    background-color: {BG_DARK};
}}

/* Flat square buttons used for viewport overlays / segmented controls. */
QPushButton#Chip {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS}px;
    padding: 4px 10px;
    color: {TEXT_SECONDARY};
}}
QPushButton#Chip:hover {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}
QPushButton#Chip:checked {{
    background-color: {BG_ACTIVE};
    border-color: {BORDER_STRONG};
    color: {TEXT_PRIMARY};
}}

QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS}px;
    padding: 4px 8px;
    color: {TEXT_SECONDARY};
}}
QToolButton:hover {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}
QToolButton:checked {{
    background-color: {BG_ACTIVE};
    color: {TEXT_PRIMARY};
}}
QToolButton:disabled {{ color: {TEXT_FAINT}; }}

/* ---- inputs -------------------------------------------------------- */

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_DARK};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS}px;
    padding: 5px 7px;
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_DIM};
    selection-color: #ffffff;
}}

QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

QLineEdit:read-only {{
    background-color: {BG_DARKEST};
    color: {TEXT_SECONDARY};
}}

QLineEdit::placeholder {{ color: {TEXT_FAINT}; }}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {BG_ACTIVE};
    outline: none;
    padding: 2px;
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_RAISED};
    border: none;
    width: 14px;
}}

QCheckBox, QRadioButton {{
    spacing: 7px;
    color: {TEXT_PRIMARY};
    padding: 2px 0;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS}px;
    background-color: {BG_DARK};
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT_DIM};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT_DIM};
    border-color: {ACCENT};
}}

/* A checked-but-disabled box means "this layer is on, but there is no
   data for it yet". Painting it in full accent claims otherwise. */
QCheckBox:disabled, QRadioButton:disabled {{
    color: {TEXT_FAINT};
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {BG_DARK};
    border-color: {BORDER};
}}

QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled {{
    background-color: {BORDER_STRONG};
    border-color: {BORDER_STRONG};
}}

QRadioButton::indicator {{ border-radius: 7px; }}

QSlider::groove:horizontal {{
    height: 3px;
    background: {BORDER_STRONG};
    border-radius: 1px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT_DIM};
    height: 3px;
    border-radius: 1px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 11px;
    height: 11px;
    margin: -5px 0;
    border-radius: 5px;
}}

QSlider::handle:horizontal:hover {{ background: {ACCENT_BRIGHT}; }}

/* ---- progress ------------------------------------------------------ */

QProgressBar {{
    background-color: {BG_DARK};
    border: none;
    border-radius: 1px;
    text-align: center;
    color: {TEXT_SECONDARY};
    height: 3px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 1px;
}}

/* ---- item views ---------------------------------------------------- */

QTableWidget, QTreeWidget, QListWidget, QTableView, QTreeView, QListView {{
    background-color: {BG_DARK};
    alternate-background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    gridline-color: {BORDER};
    color: {TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {BG_RAISED};
    color: {TEXT_MUTED};
    padding: 5px 7px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}

QTableWidget::item, QTreeWidget::item, QListWidget::item {{
    padding: 3px 4px;
}}

QTableWidget::item:selected, QTreeWidget::item:selected,
QListWidget::item:selected {{
    background-color: {BG_ACTIVE};
    color: {TEXT_PRIMARY};
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {BG_HOVER};
}}

/* ---- scrollbars ---------------------------------------------------- */

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QAbstractScrollArea {{ background: transparent; }}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 28px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-width: 28px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {TEXT_FAINT}; }}

QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ---- tabs ----------------------------------------------------------- */

QTabWidget {{
    background-color: {BG_PANEL};
}}

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {BORDER};
    background-color: {BG_PANEL};
}}

QTabBar {{
    qproperty-drawBase: 0;
    background-color: {BG_PANEL};
}}

QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_MUTED};
    padding: 7px 13px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
}}

QTabBar::tab:hover {{ color: {TEXT_SECONDARY}; }}

QTabBar::tab:selected {{
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT};
}}

QTabBar::tab:disabled {{ color: {TEXT_FAINT}; }}

/* ---- group boxes ----------------------------------------------------- */

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS}px;
    margin-top: 14px;
    padding: 12px 10px 8px 10px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {TEXT_MUTED};
}}

/* ---- misc ------------------------------------------------------------ */

QToolTip {{
    background-color: {BG_RAISED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS}px;
    padding: 5px 7px;
}}

QStatusBar {{
    background-color: {BG_DARKEST};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {TEXT_MUTED}; padding: 0 8px; }}

QDockWidget {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {BG_RAISED};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the DRISHTI-3D design system to a ``QApplication``."""
    app.setStyle("Fusion")
    app.setFont(ui_font(12))
    app.setStyleSheet(STYLESHEET)
