"""Shared bits for the slide overlays: 13.333 x 7.5 in, 300 dpi, transparent, text as vector outlines."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/tmp/demo4")
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QGuiApplication, QImage, QPainter, QPen, QPolygonF
from PySide6.QtSvg import QSvgGenerator

app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
import arch  # noqa: E402
from arch import dtext  # noqa: E402

arch.TRANSPARENT = True
S = 300.0                      # px per inch
PT = S / 72.0                  # px per point
W, H = int(13.333 * S), int(7.5 * S)
BODY = QColor("#1F497D")       # template body colour
MUTED = QColor("#5A6B85")
OUTROOT = "/Users/ajith/sih/output/slides"


def I(v):
    return v * S


def font(family, pt, bold=False):
    f = QFont(family); f.setPixelSize(int(round(pt * PT))); f.setBold(bold); return f


def text(p, rect_in, s, pt=14, bold=False, color=BODY, align=None, family="Arial", wrap=False):
    x, y, w, h = rect_in
    p.setFont(font(family, pt, bold)); p.setPen(color)
    f = arch._flag(align if align is not None else (Qt.AlignLeft | Qt.AlignVCenter))
    if wrap:
        f |= arch._flag(Qt.TextWordWrap)
    dtext(p, QRectF(I(x), I(y), I(w), I(h)), f, s)


_TITLE = [None]


def chrome(p, title):
    """Record the slide title for the preview. The overlay itself draws neither the title nor the team name:
    the template's own title placeholder and team oval (white, purple outline, black text) stay native."""
    _TITLE[0] = title


def template_look(q, s, title):
    """Preview only: the SIH template as it really renders (qlmanage render of the template, 2026-09-27)."""
    q.setPen(QPen(QColor("#8064A2"), 2.0 * s / 72)); q.setBrush(QColor("white"))
    q.drawEllipse(QRectF(0.36 * s, 0.28 * s, 1.37 * s, 0.88 * s))
    f = QFont("Arial"); f.setPixelSize(int(round(16 * s / 72))); f.setBold(True); q.setFont(f); q.setPen(QColor("black"))
    q.drawText(QRectF(0.36 * s, 0.28 * s, 1.37 * s, 0.88 * s), Qt.AlignCenter, "Robos.Inc")
    if title:
        f = QFont("Times New Roman"); f.setPixelSize(int(round(36 * s / 72))); f.setBold(True); q.setFont(f)
        q.drawText(QRectF(1.8 * s, 0.1 * s, 10.0 * s, 1.05 * s), Qt.AlignCenter, title)


def heading(p, y, s, x=0.45, w=12.45):
    text(p, (x, y, w, 0.36), s, 16, True, BODY)


def arrow(p, pts, color=QColor("#3B4A63"), width=7, head=30, style=Qt.SolidLine):
    p.setPen(QPen(color, width, style, Qt.RoundCap, Qt.RoundJoin)); p.setBrush(Qt.NoBrush)
    for a, b in zip(pts[:-1], pts[1:]):
        p.drawLine(QPointF(*a), QPointF(*b))
    import math
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    t = math.atan2(y1 - y0, x1 - x0)
    p.setPen(Qt.NoPen); p.setBrush(color)
    p.drawPolygon(QPolygonF([QPointF(x1, y1), QPointF(x1 - head * math.cos(t - 0.45), y1 - head * math.sin(t - 0.45)),
                             QPointF(x1 - head * math.cos(t + 0.45), y1 - head * math.sin(t + 0.45))]))


def export(draw, name):
    out = f"{OUTROOT}/{name.lower()}"
    os.makedirs(out, exist_ok=True)
    gen = QSvgGenerator(); gen.setFileName(f"{out}/{name}.svg"); gen.setSize(QSize(W, H)); gen.setViewBox(QRect(0, 0, W, H))
    gen.setResolution(int(S)); gen.setTitle(f"DRISHTI-3D {name}")
    p = QPainter(gen); p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform); draw(p); p.end()
    img = QImage(W, H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform); draw(p); p.end()
    img.save(f"{out}/{name}.png")
    # a check render over the template look
    from PySide6.QtSvg import QSvgRenderer
    chk = QImage(1920, 1080, QImage.Format_ARGB32_Premultiplied); chk.fill(QColor("white"))
    q = QPainter(chk); s = 1920 / 13.333
    q.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    template_look(q, s, _TITLE[0])
    q.drawImage(QRectF(10.7 * s, 0, 2.46 * s, 1.16 * s), QImage("/tmp/tpl/image2_s.png"))
    q.setBrush(QColor("#0070C0")); q.drawRect(QRectF(0, 6.95 * s, 1920, 0.55 * s))
    QSvgRenderer(f"{out}/{name}.svg").render(q, QRectF(0, 0, 1920, 1080)); q.end()
    chk.save(f"/tmp/demo4/{name}_check.png")
    print(name, os.path.getsize(f"{out}/{name}.svg") // 1024, "KB svg")
    return out
