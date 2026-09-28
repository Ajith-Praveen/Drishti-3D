"""Working-prototype panel as a high-resolution transparent PNG: green border, "Working prototype" heading,
the full app screenshot at native resolution, optional captions. Text is 14 pt when placed PW inches wide."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402,F401  (QGuiApplication, fonts)
from slidekit import I, Qt, QColor, QImage, QPainter, QPen, QRectF, text  # noqa: E402
import slide3_v4 as V4  # noqa: E402   rect(), tag()
from PySide6.QtGui import QPainterPath  # noqa: E402

GREEN = QColor("#2E9E5B")
INK = QColor("#0B1F44")
OUT = "/Users/ajith/sih/output/slides/slide3/assets"
SRC = QImage(f"{OUT}/working_prototype.png")       # full app window, 2498 x 1512, lossless
PW = 3.2                                           # panel width in inches
M = 0.02                                           # margin for the border stroke
CAPS = (("Real app, real flight", True), ("52 cameras solved", False), ("8 M-point 3D model", False))


def layout(caps):
    w = PW - 0.16
    h = w * SRC.height() / SRC.width()
    ph = 0.44 + h + (0.1 + len(caps) * 0.27 + 0.06 if caps else 0.08)
    return w, h, ph


def draw(p, caps):
    w, h, ph = layout(caps)
    x0, y0 = M, M
    V4.rect(p, x0, y0, PW, ph, GREEN, "#F1FAF4", 150, 5)
    V4.tag(p, x0, y0, GREEN, "monitor", "Working prototype")
    sx, sy = x0 + 0.08, y0 + 0.44
    clip = QPainterPath(); clip.addRoundedRect(QRectF(I(sx), I(sy), I(w), I(h)), I(0.05), I(0.05))
    p.save(); p.setClipPath(clip); p.drawImage(QRectF(I(sx), I(sy), I(w), I(h)), SRC); p.restore()
    p.setPen(QPen(QColor("#1B2A44"), 4)); p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(QRectF(I(sx), I(sy), I(w), I(h)), I(0.05), I(0.05))
    y = sy + h + 0.1
    for s, bold in caps:
        text(p, (sx, y, w, 0.26), s, 14, bold, INK, Qt.AlignHCenter | Qt.AlignVCenter)
        y += 0.27


def render(name, caps):
    w, h, ph = layout(caps)
    k = SRC.width() / I(w)                         # device px per painter px: screenshot drawn 1:1
    cw, ch = I(PW + 2 * M), I(ph + 2 * M)
    img = QImage(int(round(cw * k)), int(round(ch * k)), QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    p.scale(k, k); draw(p, caps); p.end()
    img.setDotsPerMeterX(int(round(k * 300 / 0.0254))); img.setDotsPerMeterY(int(round(k * 300 / 0.0254)))
    img.save(f"{OUT}/{name}.png")
    print(f"{name}.png  {img.width()} x {img.height()} px  ->  {PW + 2 * M:.2f} x {ph + 2 * M:.2f} in at 14 pt text")


if __name__ == "__main__":
    render("working_prototype_panel", CAPS)
    render("working_prototype_panel_no_captions", ())
