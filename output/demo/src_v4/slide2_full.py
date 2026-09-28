"""Slide 2 as one transparent overlay (13.333 x 7.5 in): title, headings, bullets, diagrams, footnote.
Insert on the template slide at position 0,0, size 13.33 x 7.5 in, after deleting the template's text boxes."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/tmp/demo4")
from PySide6.QtCore import QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtSvg import QSvgGenerator

app = QGuiApplication(sys.argv[:1])
import arch  # noqa: E402
import slide2  # noqa: E402
from arch import dtext  # noqa: E402

arch.TRANSPARENT = True
S = 300.0                          # px per inch
PT = S / 72.0                      # px per point
W, H = int(13.333 * S), int(7.5 * S)
BODY = QColor("#1F497D")
OUT = "/Users/ajith/sih/output/slides/slide2"
TITLE = "DRISHTI-3D: One Flight → Measured 3D"
INTRO = "DRISHTI-3D is an offline desktop app that turns one pass of drone video + GPS log into a georeferenced, measurable 3D model."
ROWS = [
    ("Detailed explanation of the proposed solution",
     ["Video + GPS log → georeferenced, measurable 3D model", "GPU stereo measures every view, fused in true 3D",
      "Offline desktop app with measurement tools"], slide2.flow),
    ("How it addresses the problem",
     ["Meets every target in the problem statement's output table", "Proven end to end on 2 real drone flights"], slide2.targets),
    ("Innovation and uniqueness of the solution",
     ["Our own reconstruction engine, not a wrapper", "Every surface tagged: measured or inferred"], slide2.unique),
]
FOOT = ("*0.29 m: median horizontal agreement with Spain's IGN orthophoto (flight01, 13 features) · "
        "13×: camera solve 368 s → 29 s vs our first solver · COLMAP 33 min sparse vs DRISHTI 9.5 min dense, same Mac, COLMAP without CUDA.")


def font(family, pt, bold=False):
    f = QFont(family); f.setPixelSize(int(round(pt * PT))); f.setBold(bold); return f


def wrap_lines(p, text, width):
    fm = QFontMetricsF(p.font()); n, cur = 1, ""
    for w_ in text.split():
        trial = (cur + " " + w_).strip()
        if fm.horizontalAdvance(trial) > width and cur:
            n += 1; cur = w_
        else:
            cur = trial
    return n, fm.height()


def draw(p):
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    I = lambda v: v * S  # noqa: E731
    p.setFont(font("Times New Roman", 32, True)); p.setPen(QColor("black"))
    dtext(p, QRectF(I(1.9), I(0.1), I(8.7), I(1.05)), Qt.AlignCenter, TITLE)
    p.setFont(font("Arial", 14, True)); p.setPen(QColor("white"))
    dtext(p, QRectF(I(0.36), I(0.28), I(1.37), I(0.88)), Qt.AlignCenter, "Robos.Inc")
    p.setFont(font("Arial", 16, True)); p.setPen(BODY)
    dtext(p, QRectF(I(0.45), I(1.16), I(12.4), I(0.36)), Qt.AlignLeft | Qt.AlignVCenter, "Proposed Solution (Describe your Idea/Solution/Prototype)")
    p.setFont(font("Arial", 14)); p.setPen(BODY)
    dtext(p, QRectF(I(0.45), I(1.5), I(12.45), I(0.3)), Qt.AlignLeft | Qt.AlignVCenter, INTRO)
    y0, rh, dx, dw = 1.84, 1.69, 6.8, 6.1
    for k, (head, bullets, fig) in enumerate(ROWS):
        y = I(y0 + k * rh)
        if k:
            p.setPen(QPen(QColor("#D5DEEA"), 5)); p.drawLine(I(0.45), y - I(0.04), I(12.9), y - I(0.04))
        p.setFont(font("Arial", 16, True)); p.setPen(BODY)
        dtext(p, QRectF(I(0.45), y + I(0.06), I(6.2), I(0.34)), Qt.AlignLeft | Qt.AlignVCenter, head)
        yy = y + I(0.46)
        p.setFont(font("Arial", 14)); p.setPen(BODY)
        for b in bullets:
            dtext(p, QRectF(I(0.5), yy, I(0.25), I(0.3)), Qt.AlignLeft | Qt.AlignTop, "•")
            r = QRectF(I(0.75), yy, I(5.8), I(0.62))
            dtext(p, r, arch._flag(Qt.AlignLeft) | arch._flag(Qt.AlignTop) | arch._flag(Qt.TextWordWrap), b)
            n, lh = wrap_lines(p, b, r.width())
            yy += lh * n + I(0.08)
            p.setFont(font("Arial", 14)); p.setPen(BODY)
        # the diagram, drawn as vectors at its slot (1890 px wide design -> dw inches)
        p.save(); p.translate(I(dx), y + I(0.02)); sc = I(dw) / 1890.0; p.scale(sc, sc); fig(p); p.restore()
    p.setFont(font("Arial", 9)); p.setPen(QColor("#6B7A90"))
    dtext(p, QRectF(I(0.45), I(6.5), I(6.1), I(0.44)), arch._flag(Qt.AlignLeft) | arch._flag(Qt.AlignBottom) | arch._flag(Qt.TextWordWrap), FOOT)


def main():
    gen = QSvgGenerator(); gen.setFileName(f"{OUT}/SLIDE2.svg"); gen.setSize(QSize(W, H)); gen.setViewBox(QRect(0, 0, W, H))
    gen.setResolution(int(S)); gen.setTitle("DRISHTI-3D slide 2")
    p = QPainter(gen); draw(p); p.end()
    img = QImage(W, H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); draw(p); p.end(); img.save(f"{OUT}/SLIDE2.png")
    print("ok", os.path.getsize(f"{OUT}/SLIDE2.svg") // 1024, "KB svg", os.path.getsize(f"{OUT}/SLIDE2.png") // 1024, "KB png")


main()
