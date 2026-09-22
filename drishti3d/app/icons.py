"""The DRISHTI-3D icon set and product mark, drawn in code.

No image files. Every glyph here is an SVG path rendered to a ``QIcon``
at the requested size and tint, which means:

- the icons stay crisp on a Retina display without shipping @2x assets,
- they can be recoloured per state (muted / accent / danger) from the
  same source, so a toolbar button's icon and its label always agree,
- and the PyInstaller bundle has no icon directory to forget to include,
  which is one of the ways the packaged app previously shipped without
  its resources.

All paths are authored on a 24x24 grid with a 2 px stroke, stroke-only
(no fills) except where a filled shape reads better at 16 px. Keeping one
grid and one stroke weight is what makes a set of icons look like a set.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPixmap
from PySide6.QtSvg import QSvgRenderer

from drishti3d.app import theme

#: name -> the inner markup of a 24x24 SVG (stroke is applied by the
#: wrapper below, so a path never hardcodes its own colour).
_PATHS: dict[str, str] = {
    # transport / run control
    "run": '<path d="M7 4.5 L19 12 L7 19.5 Z" fill="CURRENT" stroke="none"/>',
    "demo": '<path d="M7 4.5 L19 12 L7 19.5 Z" fill="none"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5"/>',
    "pause": '<path d="M9 5.5v13M15 5.5v13"/>',
    # files
    "video": (
        '<rect x="2.5" y="6" width="13" height="12" rx="2"/>'
        '<path d="M15.5 11 L21.5 7.5 v9 L15.5 13 Z"/>'
    ),
    "telemetry": (
        '<path d="M3 17 L8 10 L12 14 L16 6 L21 12"/>'
        '<circle cx="8" cy="10" r="1.4" fill="CURRENT" stroke="none"/>'
        '<circle cx="16" cy="6" r="1.4" fill="CURRENT" stroke="none"/>'
    ),
    "folder": '<path d="M3 6.5h6l2 2.5h10v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18Z"/>',
    "save": (
        '<path d="M4 4.5h12l4 4v11a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 19.5Z"/>'
        '<path d="M8 4.5v5h7v-5"/><path d="M7.5 20v-6h9v6"/>'
    ),
    "export": '<path d="M12 15.5V4M12 4 L8 8M12 4 L16 8"/><path d="M4 15v4.5h16V15"/>',
    "open": '<path d="M12 4v11.5M12 15.5 L8 11.5M12 15.5 L16 11.5"/><path d="M4 15v4.5h16V15"/>',
    # view
    "layers": (
        '<path d="M12 3 L21.5 8 L12 13 L2.5 8 Z"/>'
        '<path d="M2.5 12.5 L12 17.5 L21.5 12.5"/>'
        '<path d="M2.5 17 L12 22 L21.5 17"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2.1 2.1M16.9 16.9L19 19M19 5l-2.1 2.1M7.1 16.9L5 19"/>'
    ),
    "report": (
        '<rect x="4" y="3" width="16" height="18" rx="2"/>'
        '<path d="M8 9h8M8 13h8M8 17h5"/>'
    ),
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "grid": '<path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
    "axes": '<path d="M5 19V5M5 19h14"/><path d="M5 19 L13 11"/>',
    "cube": (
        '<path d="M12 2.5 L21 7.25 v9.5 L12 21.5 L3 16.75 v-9.5 Z"/>'
        '<path d="M3 7.25 L12 12 L21 7.25M12 12v9.5"/>'
    ),
    "camera": (
        '<path d="M4 8.5h3.5L9 6h6l1.5 2.5H20a1.5 1.5 0 0 1 1.5 1.5v8a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 18v-8A1.5 1.5 0 0 1 4 8.5Z"/>'
        '<circle cx="12" cy="13.5" r="3.5"/>'
    ),
    "path": (
        '<path d="M5 19c6 0 4-7 9-7s5 5 5 5"/>'
        '<circle cx="5" cy="19" r="2" fill="CURRENT" stroke="none"/>'
        '<circle cx="19" cy="17" r="2"/>'
    ),
    "cloud": (
        '<circle cx="6" cy="8" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="11" cy="6" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="17" cy="9" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="8" cy="13" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="14" cy="12" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="19" cy="15" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="5" cy="18" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="11" cy="17" r="1.2" fill="CURRENT" stroke="none"/>'
        '<circle cx="16" cy="19" r="1.2" fill="CURRENT" stroke="none"/>'
    ),
    "mesh": (
        '<path d="M12 3 L21 8.5 L17 20 H7 L3 8.5 Z"/>'
        '<path d="M3 8.5 L12 12.5 L21 8.5M12 12.5 L7 20M12 12.5 L17 20"/>'
    ),
    # tools
    "ruler": (
        '<rect x="1.5" y="8.5" width="21" height="7" rx="1.5" transform="rotate(-20 12 12)"/>'
        '<path d="M6.6 8.4l1 2.4M10.3 7l1 2.4M14 5.6l1 2.4M17.7 4.2l1 2.4"/>'
    ),
    "area": (
        '<path d="M3.5 17.5 L8 6.5 L20.5 9 L16.5 20 Z"/>'
        '<circle cx="3.5" cy="17.5" r="1.6" fill="CURRENT" stroke="none"/>'
        '<circle cx="8" cy="6.5" r="1.6" fill="CURRENT" stroke="none"/>'
        '<circle cx="20.5" cy="9" r="1.6" fill="CURRENT" stroke="none"/>'
        '<circle cx="16.5" cy="20" r="1.6" fill="CURRENT" stroke="none"/>'
    ),
    "clear": '<path d="M5 5l14 14M19 5L5 19"/>',
    "reset": '<path d="M4 10a8 8 0 1 1 1.2 6"/><path d="M3.2 4.5v5.7h5.7"/>',
    # status
    "check": '<path d="M4.5 12.5 L9.5 17.5 L19.5 6.5"/>',
    "cross": '<path d="M6 6l12 12M18 6L6 18"/>',
    "dot": '<circle cx="12" cy="12" r="4" fill="CURRENT" stroke="none"/>',
    "ring": '<circle cx="12" cy="12" r="5"/>',
    "skip": '<path d="M6 12h12"/>',
    "warning": (
        '<path d="M12 3.5 L22 20 H2 Z"/>'
        '<path d="M12 10v4.5"/>'
        '<circle cx="12" cy="17.3" r="1.1" fill="CURRENT" stroke="none"/>'
    ),
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="7.8" r="1.1" fill="CURRENT" stroke="none"/>',
    # navigation
    "chevron_down": '<path d="M6 9.5 L12 15.5 L18 9.5"/>',
    "chevron_right": '<path d="M9.5 6 L15.5 12 L9.5 18"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.2 15.2 L20.5 20.5"/>',
    "filmstrip": (
        '<rect x="2.5" y="5.5" width="19" height="13" rx="1.5"/>'
        '<path d="M6.5 5.5v13M17.5 5.5v13M2.5 12h4M17.5 12h4"/>'
    ),
    "eye": '<path d="M2 12s3.8-6.5 10-6.5S22 12 22 12s-3.8 6.5-10 6.5S2 12 2 12Z"/><circle cx="12" cy="12" r="2.8"/>',
}

_SVG_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'width="{size}" height="{size}" fill="none" stroke="{color}" '
    'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
    "{body}</svg>"
)


@lru_cache(maxsize=512)
def _pixmap(name: str, size: int, color: str, stroke: float, ratio: float) -> QPixmap:
    body = _PATHS.get(name)
    if body is None:
        body = _PATHS["dot"]
    markup = _SVG_TEMPLATE.format(
        size=size, color=color, stroke=stroke, body=body.replace("CURRENT", color)
    )
    renderer = QSvgRenderer(QByteArray(markup.encode("utf-8")))

    # The backing store is `size * ratio` real pixels, but once
    # devicePixelRatio is set the QPainter works in LOGICAL coordinates --
    # so the target rect is `size`, not `device`. Painting into a
    # `device`-sized rect draws at 2x and clips every glyph to its own
    # top-left quarter, which is exactly what the first build of this
    # module did.
    device = max(1, int(round(size * ratio)))
    pixmap = QPixmap(device, device)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pixmap


def icon(
    name: str,
    size: int = 18,
    color: str = theme.TEXT_SECONDARY,
    active_color: str | None = None,
    stroke: float = 1.8,
) -> QIcon:
    """A themed ``QIcon``.

    ``active_color`` supplies the Active/Selected states so a toolbar
    button lights up in the accent on hover without a second asset. Left
    to ``None`` it defaults to :data:`theme.ACCENT_BRIGHT`.
    """
    active = active_color or theme.ACCENT_BRIGHT
    result = QIcon()
    result.addPixmap(_pixmap(name, size, color, stroke, 2.0), QIcon.Normal, QIcon.Off)
    result.addPixmap(_pixmap(name, size, active, stroke, 2.0), QIcon.Active, QIcon.Off)
    result.addPixmap(_pixmap(name, size, active, stroke, 2.0), QIcon.Selected, QIcon.Off)
    result.addPixmap(_pixmap(name, size, active, stroke, 2.0), QIcon.Normal, QIcon.On)
    result.addPixmap(
        _pixmap(name, size, theme.TEXT_FAINT, stroke, 2.0), QIcon.Disabled, QIcon.Off
    )
    return result


def pixmap(name: str, size: int = 18, color: str = theme.TEXT_SECONDARY, stroke: float = 1.8) -> QPixmap:
    """A single themed pixmap, for a ``QLabel`` that needs a glyph."""
    return _pixmap(name, size, color, stroke, 2.0)


# ---------------------------------------------------------------------------
# Product mark
# ---------------------------------------------------------------------------


def mark(size: int = 22, ratio: float = 2.0) -> QPixmap:
    """The DRISHTI-3D mark: an aperture over a survey grid.

    *Drishti* is sight. The mark is a stylised aperture -- a hexagonal
    iris -- with a horizon line through it and a single bright vertex at
    the top, reading as both a lens and a triangulated point. Drawn here
    rather than shipped as a file so it tints with the theme and survives
    packaging.
    """
    device = max(1, int(round(size * ratio)))
    pm = QPixmap(device, device)
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.transparent)

    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    # Logical coordinates, not device -- see _pixmap above.
    painter.scale(size / 24.0, size / 24.0)

    # Hexagonal aperture.
    hexagon = QPainterPath()
    points = [(12, 2.2), (20.5, 7.1), (20.5, 16.9), (12, 21.8), (3.5, 16.9), (3.5, 7.1)]
    hexagon.moveTo(*points[0])
    for x, y in points[1:]:
        hexagon.lineTo(x, y)
    hexagon.closeSubpath()

    gradient = QLinearGradient(3.5, 2.2, 20.5, 21.8)
    gradient.setColorAt(0.0, QColor(theme.ACCENT_BRIGHT))
    gradient.setColorAt(1.0, QColor(theme.ACCENT_DIM))

    pen = painter.pen()
    pen.setWidthF(1.9)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setBrush(gradient)
    painter.setPen(pen)
    painter.drawPath(hexagon)

    # Horizon + descending sight lines: the survey, seen through the lens.
    pen.setWidthF(1.5)
    pen.setBrush(QColor(theme.ACCENT))
    painter.setPen(pen)
    painter.drawLine(6.2, 13.6, 17.8, 13.6)
    painter.drawLine(12, 2.2, 8.4, 13.6)
    painter.drawLine(12, 2.2, 15.6, 13.6)

    # The measured vertex.
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(theme.OK))
    painter.drawEllipse(QRectF(10.4, 0.6, 3.2, 3.2))

    painter.end()
    return pm


def app_icon() -> QIcon:
    """Window/dock icon, at the sizes macOS and Windows actually ask for."""
    result = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        result.addPixmap(mark(size, ratio=1.0))
    return result
