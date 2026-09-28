"""Slide 3 as the user's reference (vertical detailed tech stack + full architecture) plus a working-prototype column."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QImage, QPainter, text  # noqa: E402
import arch  # noqa: E402
import techstack_v as TV  # noqa: E402
import slide3_v4 as V4  # noqa: E402   (shot(), tag(), rect())
import numpy as np  # noqa: E402

INK = QColor("#0B1F44")
GREEN = QColor("#2E9E5B")
TECH_W, PROTO_W, GAP = 2.45, 2.2, 0.14


def arch_bbox():
    img = QImage(arch.W, arch.H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); arch.draw(p, title=False); p.end()
    q = img.convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, : q.width() * 4].reshape(q.height(), q.width(), 4)[..., 3]
    ys, xs = np.nonzero(a > 8)
    return xs.min() - 10, ys.min() - 10, xs.max() + 10, ys.max() + 10


BB = arch_bbox()


def prototype(p, x0, y0, x1, y1):
    V4.rect(p, x0, y0, x1 - x0, y1 - y0, GREEN, "#F1FAF4", 150, 5)
    V4.tag(p, x0, y0, GREEN, "monitor", "Working prototype")
    w = x1 - x0 - 0.16; h = w * 9 / 16
    top = y0 + 0.5
    V4.shot(p, "/tmp/demo3/feat/f0000.jpg", (0, 0, 1920, 1080), (x0 + 0.08, top, w, h), 0.05)
    y = top + h + 0.1
    for s, bold in (("Real app, real flight", True), ("52 cameras solved", False), ("8 M-point 3D model", False)):
        text(p, (x0 + 0.08, y, w, 0.26), s, 14, bold, INK, Qt.AlignHCenter | Qt.AlignVCenter)
        y += 0.27


def draw(p):
    K.chrome(p, "TECHNICAL APPROACH")
    cx0 = 0.45
    text(p, (cx0, 1.2, 3.35, 0.9), "Technologies to be used (e.g. programming languages, frameworks, hardware)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    tech_top = 2.12
    col_h = TECH_W * TV.H / TV.W
    sc = K.I(TECH_W) / TV.W
    p.save(); p.translate(I(cx0), I(tech_top)); p.scale(sc, sc); TV.draw(p); p.restore()
    ax0 = cx0 + TECH_W + GAP
    px0 = 12.9 - PROTO_W
    text(p, (3.95, 1.2, 12.9 - 3.95, 0.6), "Methodology and process for implementation (Flow Charts/Images/ working prototype)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    x0, y0, x1, y1 = BB
    aw = px0 - GAP - ax0
    sa = I(aw) / (x1 - x0)
    ah = (y1 - y0) * sa / K.S
    atop = 1.84 + max(0.0, (6.9 - 1.84 - ah) / 2)
    p.save(); p.translate(I(ax0), I(atop)); p.scale(sa, sa); p.translate(-x0, -y0); arch.draw(p, title=False); p.restore()
    ph = 0.5 + (PROTO_W - 0.16) * 9 / 16 + 0.1 + 3 * 0.27 + 0.12
    ptop = atop + (ah - ph) / 2
    prototype(p, px0, ptop, 12.9, ptop + ph)
    print(f"tech {TECH_W} in (to {tech_top + col_h:.2f}) · architecture {aw:.2f} x {ah:.2f} in, labels ~{30 * sa / K.PT:.1f} pt")


if __name__ == "__main__":
    K.export(draw, "SLIDE3")
