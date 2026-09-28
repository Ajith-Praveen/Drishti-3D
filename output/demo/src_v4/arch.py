"""DRISHTI-3D system architecture, AWS-reference style, drawn with QPainter at 3840 x 2160.

draw(painter, show=None, focus=None): ``show`` limits what is drawn (element ids, for build-up
animation), ``focus`` dims everything else. Elements: groups, tiles, flows (see ARCH below).
"""
import math
import os
import sys

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtCore import QRect, QSize
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QImage, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPolygonF)

W, H = 3840, 2160
TRANSPARENT = False   # no canvas; group fills become light tints (for slides)
FONTS = "/Users/ajith/sih/drishti3d/app/fonts"

NAVY = QColor("#0B1F44")
INK = QColor("#232F3E")
SUB = QColor("#26344D")
LINE = QColor("#3B4A63")
ORANGE = QColor("#ED7100")
TEAL = QColor("#01A88D")
BLUE = QColor("#2E73B8")
PURPLE = QColor("#8C4FFF")
SLATE = QColor("#5A6B85")
TILE = {"ours": ORANGE, "app": TEAL, "ext": BLUE, "input": PURPLE, "runtime": SLATE}

_fams = {}


def fonts():
    if not _fams:
        for name in ("Manrope-Bold", "Manrope-SemiBold", "IBMPlexSans-Medium", "IBMPlexSans-Regular", "IBMPlexSans-SemiBold"):
            fid = QFontDatabase.addApplicationFont(f"{FONTS}/{name}.ttf")
            _fams[name] = QFontDatabase.applicationFontFamilies(fid)[0]
    return _fams


def qfont(name, px, weight=None):
    f = QFont(fonts()[name]); f.setPixelSize(int(px))
    if weight is not None:
        f.setWeight(weight)
    return f


def _flag(v):
    return v.value if hasattr(v, "value") else int(v)


def dtext(p, rect, flags, text):
    """drawText as filled vector outlines (identical in PNG, SVG and PowerPoint; no font needed)."""
    f = _flag(flags); fm = QFontMetricsF(p.font()); color = p.pen().color()
    lines = [text]
    if f & _flag(Qt.TextWordWrap):
        lines, cur = [], ""
        for w in text.split():
            trial = (cur + " " + w).strip()
            if fm.horizontalAdvance(trial) > rect.width() and cur:
                lines.append(cur); cur = w
            else:
                cur = trial
        lines.append(cur)
    lh = fm.height(); total = lh * len(lines)
    if f & _flag(Qt.AlignVCenter):
        y0 = rect.top() + (rect.height() - total) / 2
    elif f & _flag(Qt.AlignBottom):
        y0 = rect.bottom() - total
    else:
        y0 = rect.top()
    for i, line in enumerate(lines):
        w = fm.horizontalAdvance(line)
        if f & _flag(Qt.AlignHCenter):
            x = rect.left() + (rect.width() - w) / 2
        elif f & _flag(Qt.AlignRight):
            x = rect.right() - w
        else:
            x = rect.left()
        path = QPainterPath(); path.addText(x, y0 + i * lh + fm.ascent(), p.font(), line)
        p.fillPath(path, color)


# ------------------------------------------------------------------ glyphs (white line icons)
def glyph(p, kind, cx, cy, s):
    pen = QPen(QColor("white"), s * 0.065, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.NoBrush)
    u = s / 2

    def R(x, y, w, h, r=0.0):
        p.drawRoundedRect(QRectF(cx + x * u, cy + y * u, w * u, h * u), r * u, r * u)

    def L(*pts):
        path = QPainterPath(QPointF(cx + pts[0][0] * u, cy + pts[0][1] * u))
        for x, y in pts[1:]:
            path.lineTo(cx + x * u, cy + y * u)
        p.drawPath(path)

    def C(x, y, r):
        p.drawEllipse(QPointF(cx + x * u, cy + y * u), r * u, r * u)

    if kind == "video":
        R(-0.55, -0.35, 0.8, 0.7, 0.12); L((0.25, -0.12), (0.55, -0.3), (0.55, 0.3), (0.25, 0.12))
    elif kind == "gps":
        path = QPainterPath(); path.moveTo(cx, cy + 0.55 * u)
        path.cubicTo(cx - 0.55 * u, cy - 0.05 * u, cx - 0.4 * u, cy - 0.6 * u, cx, cy - 0.6 * u)
        path.cubicTo(cx + 0.4 * u, cy - 0.6 * u, cx + 0.55 * u, cy - 0.05 * u, cx, cy + 0.55 * u); p.drawPath(path)
        C(0, -0.2, 0.15)
    elif kind == "lens":
        C(0, 0, 0.5); C(0, 0, 0.22)
        for k in range(6):
            a = k * math.pi / 3
            L((0.22 * math.cos(a), 0.22 * math.sin(a)), (0.5 * math.cos(a + 0.9), 0.5 * math.sin(a + 0.9)))
    elif kind == "film":
        R(-0.5, -0.45, 1.0, 0.9, 0.08); L((-0.28, -0.45), (-0.28, 0.45)); L((0.28, -0.45), (0.28, 0.45))
        for y in (-0.28, 0.0, 0.28):
            L((-0.44, y), (-0.34, y)); L((0.34, y), (0.44, y))
    elif kind == "filter":
        L((-0.55, -0.45), (0.55, -0.45), (0.12, 0.08), (0.12, 0.5), (-0.12, 0.38), (-0.12, 0.08), (-0.55, -0.45))
    elif kind == "clock":
        C(0, 0, 0.52); L((0, -0.3), (0, 0), (0.24, 0.14))
    elif kind == "chart":
        L((-0.5, 0.45), (0.55, 0.45)); L((-0.5, 0.45), (-0.5, -0.5))
        for x, h in ((-0.3, 0.35), (-0.02, 0.62), (0.26, 0.48)):
            R(x, 0.38 - h, 0.16, h, 0.03)
    elif kind == "mask":
        R(-0.5, -0.5, 0.7, 0.7, 0.1); R(-0.2, -0.2, 0.7, 0.7, 0.1)
    elif kind == "graph":
        pts = [(-0.45, -0.3), (0.05, -0.45), (0.45, -0.05), (-0.1, 0.1), (-0.35, 0.45), (0.3, 0.45)]
        for a, b in ((0, 1), (1, 2), (0, 3), (1, 3), (3, 4), (3, 5), (2, 5)):
            L(pts[a], pts[b])
        p.setBrush(QColor("white"))
        for x, y in pts:
            C(x, y, 0.08)
        p.setBrush(Qt.NoBrush)
    elif kind == "target":
        C(0, 0, 0.5); C(0, 0, 0.25); L((0, -0.62), (0, -0.35)); L((0, 0.35), (0, 0.62)); L((-0.62, 0), (-0.35, 0)); L((0.35, 0), (0.62, 0))
    elif kind == "planes":
        for k, y in enumerate((-0.35, -0.02, 0.31)):
            L((-0.55, y + 0.12), (-0.2, y - 0.12), (0.55, y - 0.12), (0.2, y + 0.12), (-0.55, y + 0.12))
    elif kind == "shield":
        L((0, -0.55), (0.45, -0.38), (0.4, 0.12), (0, 0.55), (-0.4, 0.12), (-0.45, -0.38), (0, -0.55))
        L((-0.2, 0.0), (-0.04, 0.16), (0.24, -0.16))
    elif kind == "cube":
        L((0, -0.55), (0.5, -0.28), (0.5, 0.3), (0, 0.57), (-0.5, 0.3), (-0.5, -0.28), (0, -0.55))
        L((-0.5, -0.28), (0, 0.0), (0.5, -0.28)); L((0, 0.0), (0, 0.57))
    elif kind == "globe":
        C(0, 0, 0.52); L((-0.52, 0), (0.52, 0))
        p.drawEllipse(QPointF(cx, cy), 0.22 * u, 0.52 * u)
        L((-0.45, -0.26), (0.45, -0.26)); L((-0.45, 0.26), (0.45, 0.26))
    elif kind == "files":
        L((-0.3, -0.55), (0.2, -0.55), (0.42, -0.33), (0.42, 0.45), (-0.3, 0.45), (-0.3, -0.55)); L((0.2, -0.55), (0.2, -0.33), (0.42, -0.33))
        L((-0.45, -0.38), (-0.45, 0.6), (0.28, 0.6))
        L((-0.14, -0.1), (0.26, -0.1)); L((-0.14, 0.1), (0.26, 0.1)); L((-0.14, 0.28), (0.12, 0.28))
    elif kind == "pin":
        L((-0.55, -0.3), (-0.2, -0.45), (0.2, -0.3), (0.55, -0.45), (0.55, 0.35), (0.2, 0.5), (-0.2, 0.35), (-0.55, 0.5), (-0.55, -0.3))
        L((-0.2, -0.45), (-0.2, 0.35)); L((0.2, -0.3), (0.2, 0.5))
    elif kind == "monitor":
        R(-0.58, -0.45, 1.16, 0.75, 0.08); L((-0.2, 0.55), (0.2, 0.55)); L((0, 0.3), (0, 0.55))
        L((0, -0.3), (0.2, -0.19), (0.2, 0.05), (0, 0.16), (-0.2, 0.05), (-0.2, -0.19), (0, -0.3)); L((-0.2, -0.19), (0, -0.08), (0.2, -0.19)); L((0, -0.08), (0, 0.16))
    elif kind == "sliders":
        for x, k in ((-0.32, -0.15), (0.0, 0.25), (0.32, -0.3)):
            L((x, -0.55), (x, 0.55)); p.setBrush(QColor("white")); C(x, k, 0.1); p.setBrush(Qt.NoBrush)
    elif kind == "ruler":
        L((-0.6, 0.15), (0.15, -0.6), (0.6, -0.15), (-0.15, 0.6), (-0.6, 0.15))
        for k in range(4):
            a = -0.38 + k * 0.19
            L((a, 0.38 + a * 0 - (a + 0.38) * 1.0 + 0.0), (a + 0.12, 0.26 - (a + 0.38) * 1.0))
    elif kind == "chip":
        R(-0.36, -0.36, 0.72, 0.72, 0.08); R(-0.16, -0.16, 0.32, 0.32, 0.04)
        for k in (-0.2, 0.0, 0.2):
            L((k, -0.36), (k, -0.55)); L((k, 0.36), (k, 0.55)); L((-0.36, k), (-0.55, k)); L((0.36, k), (0.55, k))
    elif kind == "python":
        R(-0.5, -0.5, 1.0, 1.0, 0.25); L((-0.2, -0.12), (0.05, 0.1), (-0.2, 0.3)); L((0.12, 0.3), (0.38, 0.3))
    elif kind == "sigma":
        L((0.4, -0.45), (-0.4, -0.45), (0.1, 0.0), (-0.4, 0.45), (0.4, 0.45))
    elif kind == "flame":
        path = QPainterPath(); path.moveTo(cx, cy + 0.55 * u)
        path.cubicTo(cx - 0.55 * u, cy + 0.5 * u, cx - 0.5 * u, cy - 0.1 * u, cx - 0.05 * u, cy - 0.58 * u)
        path.cubicTo(cx + 0.0 * u, cy - 0.2 * u, cx + 0.5 * u, cy - 0.1 * u, cx + 0.42 * u, cy + 0.2 * u)
        path.cubicTo(cx + 0.38 * u, cy + 0.45 * u, cx + 0.2 * u, cy + 0.56 * u, cx, cy + 0.55 * u); p.drawPath(path)
    elif kind == "brain":
        C(-0.18, -0.12, 0.3); C(0.18, -0.12, 0.3); C(0.0, 0.2, 0.3); L((0, -0.4), (0, 0.45))
    elif kind == "folder":
        L((-0.58, -0.4), (-0.15, -0.4), (0.0, -0.25), (0.58, -0.25), (0.58, 0.45), (-0.58, 0.45), (-0.58, -0.4)); L((-0.58, -0.1), (0.58, -0.1))
    elif kind == "window":
        R(-0.58, -0.48, 1.16, 0.96, 0.1); L((-0.58, -0.22), (0.58, -0.22)); L((-0.1, -0.22), (-0.1, 0.48))
    elif kind == "package":
        L((0, -0.58), (0.52, -0.32), (0.52, 0.32), (0, 0.58), (-0.52, 0.32), (-0.52, -0.32), (0, -0.58))
        L((-0.52, -0.32), (0, -0.06), (0.52, -0.32)); L((0, -0.06), (0, 0.58)); L((-0.26, -0.45), (0.26, -0.19))
    elif kind == "eye":
        path = QPainterPath(); path.moveTo(cx - 0.6 * u, cy)
        path.quadTo(cx, cy - 0.62 * u, cx + 0.6 * u, cy); path.quadTo(cx, cy + 0.62 * u, cx - 0.6 * u, cy); p.drawPath(path)
        C(0, 0, 0.18)
    elif kind == "points":
        p.setBrush(QColor("white"))
        for x, y in ((-0.4, 0.3), (-0.15, -0.1), (0.1, 0.2), (0.35, -0.3), (0.4, 0.35), (-0.35, -0.4), (0.05, -0.45), (-0.05, 0.5)):
            C(x, y, 0.09)
        p.setBrush(Qt.NoBrush)
    elif kind == "grid":
        for k in (-0.5, -0.17, 0.17, 0.5):
            L((k, -0.5), (k, 0.5)); L((-0.5, k), (0.5, k))
    elif kind == "bolt":
        L((0.12, -0.6), (-0.35, 0.08), (0.02, 0.08), (-0.12, 0.6), (0.35, -0.08), (-0.02, -0.08), (0.12, -0.6))
    elif kind == "check":
        C(0, 0, 0.52); L((-0.24, 0.0), (-0.06, 0.2), (0.26, -0.2))
    elif kind == "branch":
        C(-0.3, -0.4, 0.12); C(-0.3, 0.42, 0.12); C(0.32, -0.12, 0.12)
        L((-0.3, -0.28), (-0.3, 0.3)); path = QPainterPath(QPointF(cx + 0.32 * u, cy + 0.0 * u))
        path.quadTo(cx + 0.32 * u, cy + 0.3 * u, cx - 0.2 * u, cy + 0.34 * u); p.drawPath(path)
    elif kind == "container":
        for x, y in ((-0.45, 0.05), (-0.12, 0.05), (0.21, 0.05), (-0.28, -0.3), (0.05, -0.3)):
            R(x, y, 0.3, 0.3, 0.03)
        L((-0.6, 0.45), (0.6, 0.45))
    elif kind == "user":
        C(0, -0.25, 0.22); path = QPainterPath(QPointF(cx - 0.5 * u, cy + 0.55 * u))
        path.quadTo(cx - 0.5 * u, cy + 0.08 * u, cx, cy + 0.08 * u); path.quadTo(cx + 0.5 * u, cy + 0.08 * u, cx + 0.5 * u, cy + 0.55 * u); p.drawPath(path)
    elif kind == "gear":
        C(0, 0, 0.2)
        for k in range(8):
            a = k * math.pi / 4
            L((0.34 * math.cos(a), 0.34 * math.sin(a)), (0.55 * math.cos(a), 0.55 * math.sin(a)))
        C(0, 0, 0.36)
    elif kind == "scissors":
        C(-0.28, 0.3, 0.16); C(0.28, 0.3, 0.16); L((-0.18, 0.18), (0.35, -0.55)); L((0.18, 0.18), (-0.35, -0.55))
    elif kind == "app":
        R(-0.55, -0.45, 1.1, 0.9, 0.1); L((-0.55, -0.2), (0.55, -0.2)); C(-0.4, -0.33, 0.04); C(-0.28, -0.33, 0.04)


# ------------------------------------------------------------------ building blocks
def tile(p, kind, cat, cx, cy, s=132, dashed=False, alpha=1.0):
    col = QColor(TILE[cat])
    g = QLinearGradient(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
    g.setColorAt(0, col.lighter(118)); g.setColorAt(1, col.darker(108))
    p.setOpacity(alpha)
    p.setPen(Qt.NoPen); p.setBrush(QBrush(g))
    p.drawRoundedRect(QRectF(cx - s / 2, cy - s / 2, s, s), s * 0.16, s * 0.16)
    if dashed:
        p.setBrush(Qt.NoBrush); pen = QPen(col.darker(130), 5, Qt.DashLine); p.setPen(pen)
        p.drawRoundedRect(QRectF(cx - s / 2 - 10, cy - s / 2 - 10, s + 20, s + 20), s * 0.2, s * 0.2)
    glyph(p, kind, cx, cy, s * 0.72)
    p.setOpacity(1.0)


def label(p, cx, y, lines, alpha=1.0, width=330, bold_first=True, size=40, color=None):
    p.setOpacity(alpha)
    for k, text in enumerate(lines):
        f = qfont("IBMPlexSans-SemiBold" if (k == 0 and bold_first) else "IBMPlexSans-Regular", size if k == 0 else size - 8)
        p.setFont(f); p.setPen(color or (NAVY if k == 0 else SUB))
        dtext(p, QRectF(cx - width / 2, y + k * (size + 8), width, size + 12), Qt.AlignHCenter | Qt.AlignTop, text)
    p.setOpacity(1.0)


def group(p, rect, title, color, fill, icon=None, dashed=False, alpha=1.0, header=True, title_color=None, width=4, hpx=38, hs=58):
    x, y, w, h = rect
    p.setOpacity(alpha)
    fc = QColor(fill)
    if TRANSPARENT:
        fc.setAlpha(0 if fill.upper() == "#FFFFFF" else 150)
    p.setBrush(fc); pen = QPen(QColor(color), width, Qt.DashLine if dashed else Qt.SolidLine); p.setPen(pen)
    p.drawRect(QRectF(x, y, w, h))
    if header:
        s = hs
        p.setPen(Qt.NoPen); p.setBrush(QColor(color)); p.drawRect(QRectF(x, y, s, s))
        if icon:
            glyph(p, icon, x + s / 2, y + s / 2, s * 0.62)
        p.setFont(qfont("IBMPlexSans-SemiBold", hpx)); p.setPen(title_color or NAVY)
        dtext(p, QRectF(x + s + 18, y + 6, w - s - 30, s), Qt.AlignLeft | Qt.AlignVCenter, title)
    p.setOpacity(1.0)


def badge(p, x, y, n, alpha=1.0, r=29):
    p.setOpacity(alpha)
    p.setPen(Qt.NoPen); p.setBrush(NAVY); p.drawEllipse(QPointF(x, y), r, r)
    p.setFont(qfont("Manrope-Bold", 32)); p.setPen(QColor("white"))
    dtext(p, QRectF(x - r, y - r, 2 * r, 2 * r), Qt.AlignCenter, str(n))
    p.setOpacity(1.0)


def arrow(p, pts, n=None, text=None, style="solid", alpha=1.0, color=LINE, badge_at=0.5, text_side=-1, text_dx=0, text_dy=0, width=5):
    p.setOpacity(alpha)
    pen = QPen(color, width, {"solid": Qt.SolidLine, "dotted": Qt.DotLine, "dashed": Qt.DashLine}[style], Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.NoBrush)
    path = QPainterPath(QPointF(*pts[0]))
    for q in pts[1:]:
        path.lineTo(QPointF(*q))
    p.drawPath(path)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    a = math.atan2(y1 - y0, x1 - x0); L_ = 26
    head = QPolygonF([QPointF(x1, y1), QPointF(x1 - L_ * math.cos(a - 0.42), y1 - L_ * math.sin(a - 0.42)),
                      QPointF(x1 - L_ * math.cos(a + 0.42), y1 - L_ * math.sin(a + 0.42))])
    p.setPen(Qt.NoPen); p.setBrush(color); p.drawPolygon(head)
    # badge / text at a fraction of the path length
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    lens = [math.dist(a_, b_) for a_, b_ in segs]; tot = sum(lens); d = tot * badge_at
    for (a_, b_), l_ in zip(segs, lens):
        if d <= l_ or (a_, b_) == segs[-1]:
            f = d / l_ if l_ else 0
            bx, by = a_[0] + (b_[0] - a_[0]) * f, a_[1] + (b_[1] - a_[1]) * f
            horizontal = abs(b_[0] - a_[0]) >= abs(b_[1] - a_[1])
            break
        d -= l_
    p.setOpacity(1.0)
    if n is not None:
        badge(p, bx, by, n, alpha)
    if text:
        p.setOpacity(alpha); p.setFont(qfont("IBMPlexSans-Medium", 31)); p.setPen(SUB)
        if horizontal:
            r = QRectF(bx - 260 + text_dx, by + (-72 if text_side < 0 else 34) + text_dy, 520, 36)
            dtext(p, r, Qt.AlignHCenter | Qt.AlignVCenter, text)
        else:
            r = QRectF(bx + (40 if text_side > 0 else -560) + text_dx, by - 18 + text_dy, 520, 36)
            dtext(p, r, (Qt.AlignLeft if text_side > 0 else Qt.AlignRight) | Qt.AlignVCenter, text)
        p.setOpacity(1.0)


# ------------------------------------------------------------------ the architecture
# tiles: id -> (glyph, category, cx, cy, [label lines], dashed)
BX0, BX1 = 700, 3420           # app boundary x-range
ROW = 1150                      # core row centre
PX, DX = 970, 3200              # preparation / delivery tile columns
TILES = {
    "in_video": ("video", "input", 370, 560, ["Drone video", "MP4 / MOV, one pass"], False),
    "in_telem": ("gps", "input", 370, 880, ["Flight telemetry", "GPS + attitude log"], False),
    "in_calib": ("lens", "input", 370, 1200, ["Camera calibration", "optional intrinsics"], True),
    "ui_ctrl": ("sliders", "app", 1000, 468, ["Controls + stage rail", "Qt 6 (PySide6)"], False),
    "ui_diag": ("chart", "app", 1660, 468, ["Diagnostics + report", "stage renders · quality"], False),
    "ui_view": ("monitor", "ext", 2330, 468, ["3D model viewer", "VTK 9"], False),
    "ui_meas": ("ruler", "app", 2750, 468, ["Measurement tools", "surface-snapped"], False),
    "ui_gis": ("pin", "app", 3170, 468, ["GIS export", "GeoJSON · KML"], False),
    "prep_dec": ("film", "ext", PX, 1000, ["Video decode", "PyAV / FFmpeg"], False),
    "prep_key": ("filter", "app", PX, 1260, ["Keyframes", "sharpness · overlap"], False),
    "prep_clk": ("clock", "app", PX, 1520, ["Clock alignment", "video ↔ GPS"], False),
    "core_feat": ("graph", "ext", 1400, ROW, ["Feature tracks", "DISK + LightGlue"], False),
    "core_ba": ("target", "ours", 1720, ROW, ["Camera solver", "exact camera poses"], False),
    "core_mvs": ("planes", "ours", 2040, ROW, ["Depth engine", "depth per pixel"], False),
    "core_chk": ("shield", "ours", 2360, ROW, ["Depth check", "views must agree"], False),
    "core_tsdf": ("cube", "ext", 2680, ROW, ["3D fusion", "Open3D TSDF"], False),
    "opt_sem": ("mask", "ext", 1880, 1620, ["Semantic masks", "SegFormer · optional"], True),
    "opt_ml": ("brain", "ext", 2560, 1620, ["Learned-depth fallback", "MapAnything · optional"], True),
    "del_geo": ("globe", "app", DX, 1000, ["Georeferencing", "WGS84 / UTM"], False),
    "del_exp": ("files", "app", DX, 1330, ["Exporters", "OBJ · PLY · LAS", "GeoTIFF · glTF · FBX"], False),
    "out_disk": ("folder", "runtime", 3650, 1330, ["Local run folder", "models · maps · report"], False),
    "rt_py": ("python", "runtime", 880, 1962, ["Python 3.12"], False),
    "rt_np": ("sigma", "runtime", 1480, 1962, ["NumPy · SciPy"], False),
    "rt_torch": ("flame", "runtime", 2080, 1962, ["PyTorch"], False),
    "rt_dev": ("chip", "runtime", 2680, 1962, ["Apple MPS · NVIDIA CUDA · CPU"], False),
}
T = 132
HALF = T / 2 + 10
LBL = 48 + 32 + 14 + 10           # label block below a tile, to where arrows may end

GROUPS = {   # id -> (rect, title, border, fill, icon, dashed)
    "g_inputs": ((150, 360, 440, 1060), "Inputs", PURPLE, "#FAF7FF", "video", False),
    "g_app": ((BX0, 250, BX1 - BX0, 1830), "DRISHTI-3D desktop application  ·  runs locally on the laptop  ·  no cloud", INK, "#FFFFFF", "app", False),
    "g_pres": ((BX0 + 40, 340, BX1 - BX0 - 80, 300), "Presentation  ·  Qt 6 + VTK 9", BLUE, "#F3F8FE", "monitor", False),
    "g_work": ((BX0 + 40, 780, BX1 - BX0 - 80, 1020), "Pipeline worker  ·  QThread  ·  sequential stages  ·  cooperative cancel", TEAL, "#F4FBF9", "chip", False),
    "g_prep": ((780, 860, 380, 910), "Preparation", QColor("#8A99AD"), "#FFFFFF", "film", False),
    "g_core": ((1230, 860, 1670, 560), "Our reconstruction core", ORANGE, "#FFF7F0", "target", False),
    "g_del": ((2970, 860, 410, 910), "Delivery", QColor("#8A99AD"), "#FFFFFF", "files", False),
    "g_run": ((BX0 + 40, 1850, BX1 - BX0 - 80, 200), "Shared runtime", SLATE, "#F6F8FA", "chip", False),
}

# flows: id -> (points, number, text, style, text_side, badge_at)
FLOWS = {
    "f1": ([(370 + HALF, 560), (675, 560), (675, 1000), (PX - HALF, 1000)], 1, None, "solid", -1, 0.12),
    "f2": ([(370 + HALF, 880), (645, 880), (645, 1520), (PX - HALF, 1520)], 2, None, "solid", -1, 0.1),
    "f3": ([(370 + HALF, 1200), (612, 1200), (612, 1825), (1720, 1825), (1720, 1426)], 3, "intrinsics (optional)", "dashed", -1, 0.71),
    "p1": ([(PX, 1000 + HALF + LBL + 22), (PX, 1260 - HALF)], None, None, "solid", 1, 0.5),
    "p2": ([(PX, 1260 + HALF + LBL + 22), (PX, 1520 - HALF)], None, None, "solid", 1, 0.5),
    "f4": ([(PX + HALF, 1520), (1195, 1520), (1195, ROW), (1400 - HALF, ROW)], 4, None, "solid", 1, 0.5),
    "f5": ([(1400 + HALF, ROW), (1720 - HALF, ROW)], 5, "tracks", "solid", -1, 0.5),
    "f6": ([(1720 + HALF, ROW), (2040 - HALF, ROW)], 6, "poses + lens", "solid", -1, 0.5),
    "f7": ([(2040 + HALF, ROW), (2360 - HALF, ROW)], 7, "depth maps", "solid", -1, 0.5),
    "f8": ([(2360 + HALF, ROW), (2680 - HALF, ROW)], 8, "confirmed", "solid", -1, 0.5),
    "f9": ([(2680 + HALF, ROW), (2935, ROW), (2935, 1000), (DX - HALF, 1000)], 9, None, "solid", -1, 0.2),
    "f10": ([(DX, 1000 + HALF + LBL - 6), (DX, 1330 - HALF)], 10, None, "solid", 1, 0.55),
    "f11": ([(DX + HALF, 1330), (3650 - HALF, 1330)], 11, "files", "solid", -1, 0.66),
    "f12": ([(DX, 1000 - HALF), (DX, 720), (2330, 720), (2330, 468 + HALF + LBL - 10)], 12, "final geometry + confidence", "solid", -1, 0.55),
    "f13": ([(2750 + HALF, 468), (3170 - HALF, 468)], 13, None, "solid", -1, 0.5),
    "f14": ([(3170 + HALF, 468), (3650, 468), (3650, 1330 - HALF)], 14, None, "solid", 1, 0.62),
    "b1": ([(1880, 1620 - HALF), (1880, 1426)], None, "masks: sky · people", "dashed", 1, 0.5),
    "b2": ([(2560, 1620 - HALF), (2560, 1426)], None, "if cameras unsolved", "dashed", 1, 0.5),
    "c1": ([(1000, 468 + HALF + LBL - 4), (1000, 774)], None, "Run · settings · cancel", "dotted", 1, 0.62),
    "c2": ([(1660, 774), (1660, 468 + HALF + LBL - 4)], None, "stage status · live previews · results", "dotted", 1, 0.62),
}
FLOW_TEXT = {}

LEGEND = (150, 1540, 440, 540)
TITLE = ("DRISHTI-3D  ·  system architecture", "Single drone pass  →  georeferenced, measurable 3D model")

ALL = set(TILES) | set(GROUPS) | set(FLOWS) | {"legend", "title", "core_note"}
PARENT = {"prep_": "g_prep", "core_": "g_core", "del_": "g_del", "ui_": "g_pres", "rt_": "g_run", "in_": "g_inputs"}


def draw(p, show=None, focus=None, dim=0.18, title=True):
    """Draw the diagram. show: ids to draw (None = all). focus: ids at full strength (others dimmed)."""
    show = ALL if show is None else set(show)
    if not title:
        show = show - {"title"}
    alpha = (lambda i: 1.0 if (focus is None or i in focus) else dim)
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    if not TRANSPARENT:
        p.fillRect(QRectF(0, 0, W, H), QColor("#FFFFFF"))
    if "title" in show:
        p.setOpacity(alpha("title"))
        p.setFont(qfont("Manrope-Bold", 64)); p.setPen(NAVY); dtext(p, QRectF(150, 70, 2600, 90), Qt.AlignLeft | Qt.AlignVCenter, TITLE[0])
        p.setFont(qfont("IBMPlexSans-Regular", 34)); p.setPen(SUB); dtext(p, QRectF(150, 158, 2600, 60), Qt.AlignLeft | Qt.AlignVCenter, TITLE[1])
        p.setFont(qfont("IBMPlexSans-Medium", 30)); dtext(p, QRectF(2200, 100, 1490, 60), Qt.AlignRight | Qt.AlignVCenter, "Team Robos.Inc  ·  SIH26158  ·  NTRO")
        p.setOpacity(1.0)
    order = ["g_app", "g_inputs", "g_pres", "g_work", "g_prep", "g_core", "g_del", "g_run"]
    for gid in order:
        if gid in show:
            rect, title, col, fill, icon, dashed = GROUPS[gid]
            group(p, rect, title, col, fill, icon, dashed, alpha(gid), title_color=(ORANGE.darker(115) if gid == "g_core" else None),
                  width=5 if gid in ("g_app", "g_core") else 4)
    if "core_note" in show:
        x, y, w, h = GROUPS["g_core"][0]
        p.setOpacity(alpha("core_note"))
        p.setFont(qfont("IBMPlexSans-Medium", 32)); p.setPen(ORANGE.darker(125))
        dtext(p, QRectF(x + 30, y + 66, w - 60, 132), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
                   "Written by us. Camera solver: where every frame was taken (13× faster than our first version). "
                   "Depth engine: distance to every pixel, on the GPU. Depth check: keeps only depth other views confirm.")
        p.setOpacity(1.0)
    for fid, (pts, n, text, style, side, at) in FLOWS.items():
        if fid in show:
            arrow(p, pts, n, text, style, alpha(fid), text_side=side, badge_at=at,
                  color=(ORANGE.darker(110) if fid in ("f5", "f6", "f7", "f8") else LINE))
    for tid, (kind, cat, cx, cy, lines, dashed) in TILES.items():
        if tid in show:
            a = alpha(tid)
            if tid.startswith("rt_"):
                tile(p, kind, cat, cx, cy, 104, dashed, a)
                p.setOpacity(a); p.setFont(qfont("IBMPlexSans-SemiBold", 38)); p.setPen(NAVY)
                dtext(p, QRectF(cx + 70, cy - 30, 620, 60), Qt.AlignLeft | Qt.AlignVCenter, lines[0]); p.setOpacity(1.0)
                continue
            tile(p, kind, cat, cx, cy, T, dashed, a)
            label(p, cx, cy + T / 2 + 14, lines, a, width=420 if tid in ("del_exp", "opt_ml", "out_disk") else 330)
    if "legend" in show:
        x, y, w, h = LEGEND
        p.setOpacity(alpha("legend"))
        p.setPen(QPen(QColor("#B6C2D1"), 3)); p.setBrush(QColor("#FFFFFF")); p.drawRect(QRectF(x, y, w, h))
        p.setFont(qfont("IBMPlexSans-SemiBold", 36)); p.setPen(NAVY); dtext(p, QRectF(x + 28, y + 14, w, 48), Qt.AlignLeft, "Legend")
        rows = [("ours", "Our own algorithm"), ("app", "Our application code"), ("ext", "External library"), ("input", "Input data")]
        for k, (cat, text) in enumerate(rows):
            yy = y + 96 + k * 60
            p.setPen(Qt.NoPen); p.setBrush(TILE[cat]); p.drawRoundedRect(QRectF(x + 30, yy - 22, 44, 44), 8, 8)
            p.setFont(qfont("IBMPlexSans-Regular", 31)); p.setPen(NAVY); dtext(p, QRectF(x + 92, yy - 22, w - 100, 44), Qt.AlignVCenter, text)
        yy = y + 96 + 4 * 60
        badge(p, x + 52, yy, 1, alpha("legend"), r=20)
        p.setOpacity(alpha("legend"))
        p.setPen(QPen(LINE, 4)); p.drawLine(QPointF(x + 80, yy), QPointF(x + 110, yy))
        p.setFont(qfont("IBMPlexSans-Regular", 31)); p.setPen(NAVY); dtext(p, QRectF(x + 120, yy - 22, w - 130, 44), Qt.AlignVCenter, "Data flow, in order")
        yy += 60
        p.setPen(QPen(LINE, 4, Qt.DotLine)); p.drawLine(QPointF(x + 30, yy), QPointF(x + 110, yy))
        p.setPen(NAVY); dtext(p, QRectF(x + 120, yy - 22, w - 130, 44), Qt.AlignVCenter, "Control and events")
        yy += 60
        p.setPen(QPen(LINE, 4, Qt.DashLine)); p.drawLine(QPointF(x + 30, yy), QPointF(x + 110, yy))
        p.setPen(NAVY); dtext(p, QRectF(x + 120, yy - 22, w - 130, 44), Qt.AlignVCenter, "Optional path")
        p.setOpacity(1.0)


def render(path, show=None, focus=None, scale=1.0, drawer=None, size=(W, H), **kw):
    drawer = drawer or draw
    if str(path).endswith(".svg"):
        from PySide6.QtSvg import QSvgGenerator
        gen = QSvgGenerator(); gen.setFileName(str(path)); gen.setSize(QSize(*size)); gen.setViewBox(QRect(0, 0, *size))
        gen.setTitle("DRISHTI-3D"); p = QPainter(gen)
    else:
        img = QImage(int(size[0] * scale), int(size[1] * scale), QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        p = QPainter(img)
        if scale != 1.0:
            p.scale(scale, scale)
    drawer(p, show, focus, **kw) if drawer is draw else drawer(p, **kw)
    p.end()
    if not str(path).endswith(".svg"):
        img.save(str(path))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication(sys.argv[:1])
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo4/arch.png"
    if "--slide" in sys.argv:
        TRANSPARENT = True
        render(out, title=False)
    else:
        render(out)
    print("ok")
