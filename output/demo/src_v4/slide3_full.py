"""Slide 3 overlay with the FULL architecture (the 14-flow AWS-style diagram) + a tech tile column."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QImage, QPainter, text  # noqa: E402
import arch  # noqa: E402
from arch import tile  # noqa: E402
import numpy as np  # noqa: E402

TECH = [("target", "ours", "Our BA solver"), ("planes", "ours", "Our GPU stereo"), ("python", "runtime", "Python 3.12"),
        ("flame", "runtime", "PyTorch"), ("chip", "runtime", "Metal · CUDA"), ("eye", "ext", "OpenCV"),
        ("graph", "ext", "LightGlue"), ("cube", "ext", "Open3D"), ("mask", "ext", "SegFormer"), ("window", "ext", "Qt 6 · VTK 9")]


def arch_bbox():
    img = QImage(arch.W, arch.H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); arch.draw(p, title=False); p.end()
    q = img.convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, : q.width() * 4].reshape(q.height(), q.width(), 4)[..., 3]
    ys, xs = np.nonzero(a > 8)
    return xs.min() - 10, ys.min() - 10, xs.max() + 10, ys.max() + 10


BB = arch_bbox()


def draw(p):
    import techstack_v as TV
    K.chrome(p, "TECHNICAL APPROACH")
    # left column: the detailed tech stack (vertical AWS-style groups)
    cx0, top, bottom = 0.45, 2.34, 6.9
    col_w = (bottom - top) * TV.W / TV.H
    text(p, (cx0, 1.2, col_w + 0.1, 1.1), "Technologies to be used (e.g. programming languages, frameworks, hardware)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    sc = I(col_w) / TV.W
    p.save(); p.translate(I(cx0), I(top)); p.scale(sc, sc); TV.draw(p); p.restore()
    # right: methodology = the full architecture
    ax0 = cx0 + col_w + 0.18
    text(p, (ax0, 1.2, 12.9 - ax0, 0.6), "Methodology and process for implementation (Flow Charts/Images/ working prototype)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    x0, y0, x1, y1 = BB
    aw = 12.9 - ax0
    sa = I(aw) / (x1 - x0)
    ah = (y1 - y0) * sa / K.S
    atop = 1.82 + max(0.0, (6.9 - 1.82 - ah) / 2)
    p.save(); p.translate(I(ax0), I(atop)); p.scale(sa, sa); p.translate(-x0, -y0); arch.draw(p, title=False); p.restore()
    print(f"tech column {col_w:.2f} in (labels ~{40 * sc / K.PT:.1f} pt) · architecture {aw:.2f} x {ah:.2f} in (labels ~{30 * sa / K.PT:.1f} pt)")


if __name__ == "__main__":
    K.export(draw, "SLIDE3")
