"""Slide 3 = the user's reference layout (vertical tech stack + full architecture, fonts enlarged)
with a small working-prototype screenshot under the tech stack."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QImage, QPainter, QPen, QRectF, text  # noqa: E402
import arch  # noqa: E402
import techstack_v as TV  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtGui import QPainterPath  # noqa: E402

GREEN = QColor("#2E9E5B")
CX0, COL_W = 0.45, 2.69
AX0 = CX0 + COL_W + 0.18


def arch_bbox():
    img = QImage(arch.W, arch.H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); arch.draw(p, title=False); p.end()
    q = img.convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, : q.width() * 4].reshape(q.height(), q.width(), 4)[..., 3]
    ys, xs = np.nonzero(a > 8)
    return xs.min() - 10, ys.min() - 10, xs.max() + 10, ys.max() + 10


BB = arch_bbox()


def tech_height():
    img = QImage(10, 10, QImage.Format_ARGB32_Premultiplied); p = QPainter(img); h = TV.draw(p); p.end()
    return h


def prototype(p, x, y, w):
    """Overall app screenshot (top bar, stage rail, 3D view, inspector) with a label on it."""
    src = QImage("/tmp/demo3/feat/f0000.jpg").copy(0, 0, 1920, 760)
    h = w * src.height() / src.width()
    img = src.scaled(int(I(w)), int(I(h)), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    clip = QPainterPath(); clip.addRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(0.05), I(0.05))
    p.save(); p.setClipPath(clip); p.drawImage(QRectF(I(x), I(y), I(w), I(h)), img); p.restore()
    p.setPen(QPen(GREEN, 6)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(0.05), I(0.05))
    tw, th = 2.02, 0.28
    tx, ty = x + w * 0.46 - tw / 2, y + h - th - 0.05
    p.setPen(Qt.NoPen); p.setBrush(GREEN); p.drawRoundedRect(QRectF(I(tx), I(ty), I(tw), I(th)), I(0.06), I(0.06))
    text(p, (tx, ty, tw, th), "Working prototype", 14, True, QColor("white"), Qt.AlignCenter)
    return h


def draw(p):
    K.chrome(p, "TECHNICAL APPROACH")
    text(p, (CX0, 1.2, 3.4, 0.86), "Technologies to be used (e.g. programming languages, frameworks, hardware)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    text(p, (3.95, 1.2, 12.9 - 3.95, 0.6), "Methodology and process for implementation (Flow Charts/Images/ working prototype)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    t_top = 2.08
    sc = I(COL_W) / TV.W
    p.save(); p.translate(I(CX0), I(t_top)); p.scale(sc, sc); TV.draw(p); p.restore()
    t_end = t_top + tech_height() * sc / K.S
    ph = prototype(p, CX0, t_end + 0.07, COL_W)
    x0, y0, x1, y1 = BB
    aw = 12.9 - AX0
    sa = I(aw) / (x1 - x0)
    ah = (y1 - y0) * sa / K.S
    atop = 1.84 + max(0.0, (6.9 - 1.84 - ah) / 2)
    p.save(); p.translate(I(AX0), I(atop)); p.scale(sa, sa); p.translate(-x0, -y0); arch.draw(p, title=False); p.restore()
    print(f"tech stack to {t_end:.2f} in (labels {46 * sc / K.PT:.1f} pt) · screenshot {COL_W:.2f} x {ph:.2f} in, ends {t_end + 0.07 + ph:.2f} · "
          f"architecture {aw:.2f} x {ah:.2f} in (labels {40 * sa / K.PT:.1f} pt, sub-labels {32 * sa / K.PT:.1f} pt)")


if __name__ == "__main__":
    K.export(draw, "SLIDE3")
