"""Each diagram as its own transparent, content-cropped PNG + SVG (vector text)."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/tmp/demo4")
import numpy as np
import slidekit  # noqa: F401  (QGuiApplication, TRANSPARENT)
import arch, techstack, techstack_v, slide2_aws
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgGenerator

arch.TRANSPARENT = True
ROOT = "/Users/ajith/sih/output/slides"


def hints(p):
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)


def export(drawer, size, path_noext, pad=24, dpi=300):
    W, H = size
    img = QImage(W, H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); hints(p); drawer(p); p.end()
    q = img.convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(q.constBits(), np.uint8).reshape(q.height(), q.bytesPerLine())[:, : q.width() * 4].reshape(q.height(), q.width(), 4)[..., 3]
    ys, xs = np.nonzero(a > 8)
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(W, xs.max() + pad), min(H, ys.max() + pad)
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    img.copy(x0, y0, x1 - x0, y1 - y0).save(path_noext + ".png")
    gen = QSvgGenerator(); gen.setFileName(path_noext + ".svg"); gen.setSize(QSize(x1 - x0, y1 - y0))
    gen.setViewBox(QRect(x0, y0, x1 - x0, y1 - y0)); gen.setResolution(dpi)
    p = QPainter(gen); hints(p); drawer(p); p.end()
    print(os.path.relpath(path_noext, ROOT), (x1 - x0, y1 - y0))


if __name__ == "__main__":
    export(lambda p: arch.draw(p, title=False), (arch.W, arch.H), f"{ROOT}/slide3/assets/architecture", dpi=400)
    export(lambda p: techstack_v.draw(p), (techstack_v.W, techstack_v.H), f"{ROOT}/slide3/assets/techstack_vertical", dpi=400)
    export(lambda p: techstack.draw(p), (techstack.W, techstack.H), f"{ROOT}/slide3/assets/techstack_wide", dpi=400)
    for k, name in ((1, "band1_how_it_works"), (2, "band2_meets_the_problem"), (3, "band3_what_is_ours")):
        export(lambda p, k=k: slide2_aws.draw(p, only=k), (slidekit.W, slidekit.H), f"{ROOT}/slide2/assets/{name}")
    # refresh the top-level copies too
    export(lambda p: arch.draw(p, title=False), (arch.W, arch.H), f"{ROOT}/DRISHTI-3D_architecture", dpi=400)
    export(lambda p: techstack.draw(p), (techstack.W, techstack.H), f"{ROOT}/DRISHTI-3D_techstack", dpi=400)
