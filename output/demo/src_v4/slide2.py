"""Slide 2 (Proposed Solution) diagrams, light theme, transparent. Each is 1890 x 520 px = 6.3 x 1.73 in at
300 dpi, so text lands at 14 pt (58 px) and 16 pt (67 px) when placed 6.3 in wide."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch  # noqa: E402
from arch import NAVY, ORANGE, QColor, QPainter, QPainterPath, QRectF, Qt, dtext, glyph, qfont, tile  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QImage, QPen, QPolygonF  # noqa: E402

PT14, PT16 = 58, 67            # px at 300 dpi
BODY = QColor("#1F497D")       # the template's body text colour
GREEN = QColor("#2E9E5B")
OUT = "/Users/ajith/sih/output/slides/slide2"
APP_SHOT = "/tmp/demo3/feat/f0000.jpg"
MODEL_SHOT = "/tmp/demo/assets/fly/0150.jpg"


def text(p, x, y, w, h, s, px, color=NAVY, bold=False, align=Qt.AlignHCenter | Qt.AlignTop, wrap=False):
    p.setFont(qfont("IBMPlexSans-SemiBold" if bold else "IBMPlexSans-Regular", px)); p.setPen(color)
    dtext(p, QRectF(x, y, w, h), arch._flag(align) | (arch._flag(Qt.TextWordWrap) if wrap else 0), s)


def arrow_right(p, x0, x1, y, color=QColor("#3B4A63")):
    p.setPen(QPen(color, 9, Qt.SolidLine, Qt.RoundCap)); p.drawLine(QPointF(x0, y), QPointF(x1 - 26, y))
    p.setPen(Qt.NoPen); p.setBrush(color)
    p.drawPolygon(QPolygonF([QPointF(x1, y), QPointF(x1 - 36, y - 22), QPointF(x1 - 36, y + 22)]))


def picture(p, path, rect, radius=22, crop=None):
    img = QImage(path)
    if crop:
        img = img.copy(*crop)
    x, y, w, h = rect
    img = img.scaled(int(w * 2.2), int(h * 2.2), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    # cover-fit
    s = max(w / img.width(), h / img.height()); sw, sh = w / s, h / s
    src = QRectF((img.width() - sw) / 2, (img.height() - sh) / 2, sw, sh)
    clip = QPainterPath(); clip.addRoundedRect(QRectF(x, y, w, h), radius, radius)
    p.save(); p.setClipPath(clip); p.drawImage(QRectF(x, y, w, h), img, src); p.restore()


def laptop(p, cx, cy, w):
    """A laptop outline with the real app screenshot on its screen."""
    h = w * 0.58
    x, y = cx - w / 2, cy - h / 2 - 18
    p.setPen(QPen(QColor("#2B3440"), 10)); p.setBrush(QColor("#2B3440"))
    p.drawRoundedRect(QRectF(x - 14, y - 14, w + 28, h + 28), 22, 22)
    picture(p, APP_SHOT, (x, y, w, h), 8)
    base = QPolygonF([QPointF(x - 60, y + h + 22), QPointF(x + w + 60, y + h + 22), QPointF(x + w + 20, y + h + 50), QPointF(x - 20, y + h + 50)])
    p.setPen(Qt.NoPen); p.setBrush(QColor("#8A96A8")); p.drawPolygon(base)


# ------------------------------------------------------------------ A: what it does
def flow(p, **_):
    W, H = 1890, 520
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    y_mid = 200; T = 200
    # inputs: video + GPS (stacked small tiles)
    tile(p, "video", "input", 175, y_mid - 72, 140)
    tile(p, "gps", "input", 175, y_mid + 88, 140)
    text(p, 0, y_mid + 180, 350, 80, "Video + GPS", PT14, NAVY, True)
    arrow_right(p, 265, 405, y_mid)
    # the app on a laptop
    laptop(p, 690, y_mid, 480)
    text(p, 420, y_mid + 180, 540, 80, "DRISHTI-3D app", PT14, NAVY, True)
    text(p, 420, y_mid + 244, 540, 80, "offline · one laptop", PT14, arch.SUB)
    arrow_right(p, 975, 1135, y_mid)
    # the 3D model (real render)
    picture(p, MODEL_SHOT, (1150, y_mid - 140, 320, 280), 26)
    p.setPen(QPen(QColor("#2E73B8"), 6)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(1150, y_mid - 140, 320, 280), 26, 26)
    text(p, 1090, y_mid + 180, 440, 80, "Measured 3D", PT14, NAVY, True)
    arrow_right(p, 1490, 1620, y_mid)
    # measure + export
    tile(p, "ruler", "app", 1745, y_mid - 72, 140)
    tile(p, "files", "app", 1745, y_mid + 88, 140)
    text(p, 1600, y_mid + 180, 290, 80, "Measure", PT14, NAVY, True)
    text(p, 1600, y_mid + 244, 290, 80, "+ export", PT14, arch.SUB)
    return W, H


# ------------------------------------------------------------------ B: meets the problem statement
TARGETS = [("0.29 m*", "IGN agreement", "target"), ("6.9 min", "11.4-min video", "clock"), ("1 pass", "single flight", "video"),
           ("86 %", "measured", "shield"), ("6 / 6", "output formats", "files"), ("Desktop", "viewer + tools", "monitor")]


def targets(p, **_):
    W, H = 1890, 520
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    cw, ch, gx, gy = 610, 235, 30, 30
    for k, (value, label, icon) in enumerate(TARGETS):
        r, c = divmod(k, 3)
        x, y = c * (cw + gx), r * (ch + gy) + 10
        p.setPen(QPen(QColor("#CFE3D6"), 4)); p.setBrush(QColor(236, 248, 240, 220))
        p.drawRoundedRect(QRectF(x, y, cw, ch), 30, 30)
        # green check disc with the metric's icon
        cx, cy = x + 95, y + ch / 2
        p.setPen(Qt.NoPen); p.setBrush(GREEN); p.drawEllipse(QPointF(cx, cy), 68, 68)
        glyph(p, icon, cx, cy, 84)
        p.setBrush(QColor("white")); p.setPen(QPen(GREEN, 6)); p.drawEllipse(QPointF(cx + 58, cy - 58), 30, 30)
        p.setPen(QPen(GREEN, 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)); path = QPainterPath(QPointF(cx + 44, cy - 58))
        path.lineTo(cx + 54, cy - 46); path.lineTo(cx + 73, cy - 70); p.setBrush(Qt.NoBrush); p.drawPath(path)
        text(p, x + 190, y + 34, cw - 200, 90, value, PT16 + 12, GREEN.darker(135), True, Qt.AlignLeft | Qt.AlignTop)
        text(p, x + 190, y + 128, cw - 200, 80, label, PT14, NAVY, False, Qt.AlignLeft | Qt.AlignTop)
    return W, H


# ------------------------------------------------------------------ C: what makes it unique
UNIQUE = [("target", "Own engine", "no COLMAP / ODM wrapper"), ("eye", "Live preview", "spot gaps, re-fly on site"),
          ("chart", "3.5× faster", "than COLMAP, and dense*"), ("shield", "Honest 3D", "measured vs inferred")]


def unique(p, **_):
    W, H = 1890, 520
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    cw, gx = 445, 36
    for k, (icon, title, sub) in enumerate(UNIQUE):
        x = k * (cw + gx)
        p.setPen(QPen(QColor("#F6C9A3"), 4)); p.setBrush(QColor(255, 246, 238, 225))
        p.drawRoundedRect(QRectF(x, 10, cw, H - 20), 30, 30)
        tile(p, icon, "ours", x + cw / 2, 118, 150)
        text(p, x + 10, 212, cw - 20, 90, title, PT16, ORANGE.darker(125), True)
        text(p, x + 20, 292, cw - 40, 220, sub, PT14, NAVY, False, Qt.AlignHCenter, wrap=True)
    return W, H


def render(fn, name):
    from PySide6.QtCore import QRect, QSize
    from PySide6.QtSvg import QSvgGenerator
    probe = QImage(10, 10, QImage.Format_ARGB32_Premultiplied); pp = QPainter(probe); W, H = fn(pp); pp.end()
    img = QImage(W, H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); fn(p); p.end(); img.save(f"{OUT}/{name}.png")
    gen = QSvgGenerator(); gen.setFileName(f"{OUT}/{name}.svg"); gen.setSize(QSize(W, H)); gen.setViewBox(QRect(0, 0, W, H))
    p = QPainter(gen); fn(p); p.end()
    return W, H


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication(sys.argv[:1])
    os.makedirs(OUT, exist_ok=True)
    arch.TRANSPARENT = True
    for fn, name in ((flow, "s2_A_flow"), (targets, "s2_B_targets"), (unique, "s2_C_unique")):
        print(name, render(fn, name))
