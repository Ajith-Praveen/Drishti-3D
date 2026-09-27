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

Themes and type
---------------
The chrome palette and the typeface pairing are operator preferences
(Tools > Preferences), stored with ``QSettings``. See ``PALETTES`` and
``TYPEFACES`` below. The confidence and status colours are deliberately
outside any palette.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
#
# The chrome palette is switchable (Tools > Preferences > Themes). Every
# palette is a seven-step ground ramp, four text tones and one accent;
# nothing else changes between them. In particular the status and
# confidence colours further down are NOT part of a palette: a verdict
# must mean the same colour whichever theme the operator picked.
#
# Panels read tokens as ``theme.BG_PANEL`` at the moment they build or
# paint, never ``from theme import BG_PANEL``, so ``set_palette`` only has
# to rewrite this module's globals and rebuild the widgets.

_TOKEN_NAMES: tuple[str, ...] = (
    "BG_VOID", "BG_DARKEST", "BG_DARK", "BG_PANEL", "BG_RAISED", "BG_HOVER", "BG_ACTIVE",
    "BORDER", "BORDER_STRONG",
    "TEXT_PRIMARY", "TEXT_SECONDARY", "TEXT_MUTED", "TEXT_FAINT",
    "ACCENT", "ACCENT_BRIGHT", "ACCENT_DIM", "ACCENT_WASH", "ON_ACCENT",
)


@dataclass(frozen=True)
class Palette:
    key: str
    name: str
    reference: str
    description: str
    tokens: dict[str, str]


def _palette(key: str, name: str, reference: str, description: str, *values: str) -> Palette:
    assert len(values) == len(_TOKEN_NAMES), key
    return Palette(key, name, reference, description, dict(zip(_TOKEN_NAMES, values)))


PALETTES: dict[str, Palette] = {
    p.key: p
    for p in (
        _palette(
            "glacier", "Glacier", "Survey cyan",
            "Cyan reads as optics and survey instruments, and stays clear of all "
            "three confidence hues, so the accent never looks like a verdict.",
            "#070a0b", "#0b0f11", "#101518", "#141a1d", "#1a2226", "#232d31", "#2c383d",
            "#1d262a", "#2c383d",
            "#e3ecee", "#9fb0b4", "#728488", "#4b5a5e",
            "#36c2d4", "#6adbe8", "#16606b", "#122429", "#041215",
        ),
        _palette(
            "aperture", "Aperture", "Pro-video blue",
            "The familiar editing-suite blue on cool slate. The safest choice, and "
            "the closest to the original steel blue.",
            "#08090c", "#0d0f13", "#12151a", "#171a20", "#1e2229", "#272c34", "#313740",
            "#22262d", "#333942",
            "#e6e9ee", "#a3abb6", "#77808c", "#525a65",
            "#3d9bff", "#6cb4ff", "#1f4f86", "#152234", "#04101f",
        ),
        _palette(
            "violet", "Render Violet", "Compositor violet",
            "Soft lavender over violet-tinted greys. Reads as a motion or "
            "compositing tool; the most personality of the set.",
            "#09080d", "#0e0d14", "#13121b", "#181721", "#1f1d2a", "#292736", "#333043",
            "#221f2d", "#353146",
            "#ebe9f3", "#aba7bd", "#7e7a92", "#57536a",
            "#9d8cff", "#bcb0ff", "#4e4394", "#211d36", "#110c2a",
        ),
        _palette(
            "carbon", "Carbon", "Grading-suite neutral",
            "True neutral greys and a near-white accent. The most restrained: the "
            "aerial imagery supplies all of the colour.",
            "#080808", "#0e0e0f", "#131314", "#18181a", "#1f1f21", "#29292c", "#333336",
            "#222224", "#343437",
            "#ededee", "#a8a8ad", "#7b7b81", "#545458",
            "#e8e8ea", "#ffffff", "#5c5c62", "#232325", "#111113",
        ),
        _palette(
            "classic", "Classic Steel", "The original theme",
            "The muted steel blue DRISHTI-3D shipped with, kept for anyone who "
            "prefers it.",
            "#0b0c0e", "#111316", "#16181c", "#1a1d21", "#212429", "#2a2e34", "#333840",
            "#24272c", "#343941",
            "#e4e7ea", "#a4aab3", "#79808a", "#555b63",
            "#5b87c4", "#7ba3d8", "#3f5f8c", "#1b2029", "#0a111c",
        ),
    )
}
DEFAULT_PALETTE = "glacier"

# Active chrome tokens. These are the Glacier values; ``set_palette``
# overwrites them in place. Declared explicitly (rather than only through
# ``globals().update``) so editors and linters can see them.
BG_VOID = "#070a0b"      # behind everything; the viewport letterbox
BG_DARKEST = "#0b0f11"   # app chrome, command bar, menu bar
BG_DARK = "#101518"      # canvas behind panels, input wells
BG_PANEL = "#141a1d"     # panel bodies
BG_RAISED = "#1a2226"    # headers, cards, tiles
BG_HOVER = "#232d31"     # hover
BG_ACTIVE = "#2c383d"    # pressed / selected row

BORDER = "#1d262a"
BORDER_STRONG = "#2c383d"

TEXT_PRIMARY = "#e3ecee"
TEXT_SECONDARY = "#9fb0b4"
TEXT_MUTED = "#728488"
TEXT_FAINT = "#4b5a5e"   # rules and disabled glyphs only -- never prose

ACCENT = "#36c2d4"
ACCENT_BRIGHT = "#6adbe8"
ACCENT_DIM = "#16606b"
ACCENT_WASH = "#122429"  # accent at ~10% over BG_PANEL, precomputed
ON_ACCENT = "#041215"    # label colour on a filled-accent button

_active_palette = DEFAULT_PALETTE


def set_palette(key: str) -> None:
    """Make ``key`` the active chrome palette (module globals only).

    Widgets that already exist keep the colours they were built with;
    call :func:`apply_theme` and rebuild the window to see the change.
    """
    global _active_palette
    if key not in PALETTES:
        key = DEFAULT_PALETTE
    globals().update(PALETTES[key].tokens)
    _active_palette = key


def current_palette() -> str:
    return _active_palette


# Status. Muted, and separated by luminance as well as hue -- see the
# module docstring on why desaturation must not cost legibility. These
# are constant across palettes on purpose.
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
#
# Two roles: a UI face for chrome and a monospace face for every number.
# The pairings below are bundled as .ttf files in ``app/fonts`` (all SIL
# OFL), so the app looks the same on every machine and needs no network.
# "System" keeps the platform faces (SF Pro / SF Mono on macOS).


@dataclass(frozen=True)
class Typeface:
    key: str
    name: str
    ui_family: str | None
    mono_family: str | None
    description: str


TYPEFACES: dict[str, Typeface] = {
    t.key: t
    for t in (
        Typeface(
            "plex", "Instrument", "IBM Plex Sans", "IBM Plex Mono",
            "Engineered and very legible at 11 px. Feels like a measuring "
            "instrument rather than a web app.",
        ),
        Typeface(
            "geist", "Studio", "Geist", "Geist Mono",
            "Crisp and contemporary, with a mono cut on the same skeleton.",
        ),
        Typeface(
            "barlow", "Broadcast", "Barlow", "Roboto Mono",
            "Slightly narrow, so more fits on each row. Broadcast and grading-panel "
            "flavour.",
        ),
        Typeface(
            "manrope", "Modern", "Manrope", "JetBrains Mono",
            "Rounder and friendlier; good for a demo build.",
        ),
        Typeface(
            "system", "System", None, None,
            "The platform faces: SF Pro and SF Mono on macOS, Segoe UI and "
            "Consolas on Windows.",
        ),
    )
}
DEFAULT_TYPEFACE = "plex"
_active_typeface = DEFAULT_TYPEFACE
_fonts_loaded = False
_loaded_families: set[str] = set()


def _font_dirs() -> list[Path]:
    """Where the bundled .ttf files live, from source and when frozen."""
    dirs = [Path(__file__).resolve().parent / "fonts"]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        dirs.append(Path(frozen_root) / "drishti3d" / "app" / "fonts")
        # macOS .app bundles keep data under Contents/Resources.
        dirs.append(Path(frozen_root).parent / "Resources" / "drishti3d" / "app" / "fonts")
    return dirs


def load_bundled_fonts() -> set[str]:
    """Register the bundled faces with Qt. Needs a QGuiApplication; idempotent."""
    global _fonts_loaded
    if _fonts_loaded:
        return _loaded_families
    for folder in _font_dirs():
        if not folder.is_dir():
            continue
        for ttf in sorted(folder.glob("*.ttf")):
            font_id = QFontDatabase.addApplicationFont(str(ttf))
            if font_id >= 0:
                _loaded_families.update(QFontDatabase.applicationFontFamilies(font_id))
        if _loaded_families:
            break
    _fonts_loaded = True
    return _loaded_families


def set_typeface(key: str) -> None:
    global _active_typeface
    _active_typeface = key if key in TYPEFACES else DEFAULT_TYPEFACE


def current_typeface() -> str:
    return _active_typeface


def _family(role: str) -> str | None:
    """The active family for ``role`` ("ui"/"mono"), if it is installed."""
    face = TYPEFACES.get(_active_typeface)
    family = None if face is None else (face.ui_family if role == "ui" else face.mono_family)
    if not family:
        return None
    if family in _loaded_families or family in QFontDatabase.families():
        return family
    return None


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
    """The active UI face at a given size.

    Falls back to the platform UI face -- resolved from the platform, not
    hardcoded; asking for "Segoe UI" on macOS is what once made this app
    render in fallback Helvetica.
    """
    family = _family("ui")
    if family:
        font = QFont(family)
    else:
        font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    font.setPixelSize(size)
    font.setWeight(_weight(weight))
    return font


def mono_font(size: int = 11, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """The active monospace face, for every number in the UI."""
    family = _family("mono")
    if family:
        font = QFont(family)
        font.setStyleHint(QFont.Monospace)
    else:
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPixelSize(size)
    font.setWeight(_weight(weight))
    return font


def mono_family() -> str:
    """Monospace family name, for embedding in a per-widget stylesheet."""
    return _family("mono") or QFontDatabase.systemFont(QFontDatabase.FixedFont).family()


def ui_family() -> str:
    return _family("ui") or QFontDatabase.systemFont(QFontDatabase.GeneralFont).family()


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

_PREF_PALETTE = "appearance/palette"
_PREF_TYPEFACE = "appearance/typeface"


def _settings() -> QSettings:
    return QSettings("DRISHTI-3D", "DRISHTI-3D")


def load_preferences() -> tuple[str, str]:
    """Adopt the saved palette and typeface (defaults if none saved)."""
    settings = _settings()
    set_palette(str(settings.value(_PREF_PALETTE, DEFAULT_PALETTE)))
    set_typeface(str(settings.value(_PREF_TYPEFACE, DEFAULT_TYPEFACE)))
    return _active_palette, _active_typeface


def save_preferences() -> None:
    settings = _settings()
    settings.setValue(_PREF_PALETTE, _active_palette)
    settings.setValue(_PREF_TYPEFACE, _active_typeface)
    settings.sync()


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def build_stylesheet() -> str:
    """The application stylesheet for the active palette."""
    return f"""
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

/* The one primary action on screen: filled accent, dark label. */
QPushButton#Primary {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: {ON_ACCENT};
    font-weight: 600;
    padding: 6px 18px;
}}
QPushButton#Primary:hover {{ background-color: {ACCENT_BRIGHT}; border-color: {ACCENT_BRIGHT}; }}
QPushButton#Primary:pressed {{ background-color: {ACCENT_DIM}; border-color: {ACCENT}; color: {TEXT_PRIMARY}; }}
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
    background-color: {ACCENT};
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



STYLESHEET = build_stylesheet()  # back-compat: the stylesheet for the default palette


def _qpalette() -> QPalette:
    """A Fusion ``QPalette`` matching the tokens, for anything the stylesheet misses."""
    pal = QPalette()
    roles = {
        QPalette.Window: BG_DARKEST,
        QPalette.WindowText: TEXT_PRIMARY,
        QPalette.Base: BG_DARK,
        QPalette.AlternateBase: BG_PANEL,
        QPalette.Text: TEXT_PRIMARY,
        QPalette.Button: BG_RAISED,
        QPalette.ButtonText: TEXT_PRIMARY,
        QPalette.ToolTipBase: BG_RAISED,
        QPalette.ToolTipText: TEXT_PRIMARY,
        QPalette.PlaceholderText: TEXT_FAINT,
        QPalette.Highlight: ACCENT_DIM,
        QPalette.HighlightedText: "#ffffff",
        QPalette.Link: ACCENT_BRIGHT,
        QPalette.BrightText: ERR,
        QPalette.Mid: BORDER_STRONG,
        QPalette.Dark: BG_VOID,
        QPalette.Light: BG_HOVER,
    }
    for role, value in roles.items():
        pal.setColor(role, QColor(value))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(TEXT_FAINT))
    return pal


def apply_theme(
    app: QApplication, palette: str | None = None, typeface: str | None = None
) -> None:
    """Apply the DRISHTI-3D design system to a ``QApplication``.

    With no arguments the saved preferences are used (defaults if none).
    Passing ``palette`` / ``typeface`` applies those without saving; call
    :func:`save_preferences` to persist them.
    """
    global STYLESHEET
    load_bundled_fonts()
    if palette is None and typeface is None:
        load_preferences()
    else:
        if palette is not None:
            set_palette(palette)
        if typeface is not None:
            set_typeface(typeface)
    app.setStyle("Fusion")
    app.setPalette(_qpalette())
    app.setFont(ui_font(12))
    STYLESHEET = build_stylesheet()
    app.setStyleSheet(STYLESHEET)
