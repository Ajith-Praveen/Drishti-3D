"""Transparent, content-cropped PNG + SVG of the architecture and tech-stack diagrams, for slides."""
import os, re, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/tmp/demo4")
import numpy as np
from PySide6.QtGui import QGuiApplication, QImage
app = QGuiApplication(sys.argv[:1])
import arch, techstack
OUT = "/Users/ajith/sih/output/slides"; os.makedirs(OUT, exist_ok=True)
arch.TRANSPARENT = True


def bbox(png, pad=28):
    img = QImage(png).convertToFormat(QImage.Format_RGBA8888)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4].reshape(img.height(), img.width(), 4)[..., 3]
    ys, xs = np.nonzero(a > 8)
    return max(0, xs.min() - pad), max(0, ys.min() - pad), min(img.width(), xs.max() + pad), min(img.height(), ys.max() + pad)


for name, kw in (("architecture", dict(drawer=None, title=False)), ("techstack", dict(drawer=techstack.draw))):
    png, svg = f"{OUT}/DRISHTI-3D_{name}.png", f"{OUT}/DRISHTI-3D_{name}.svg"
    arch.render(png, **kw); arch.render(svg, **kw)
    x0, y0, x1, y1 = bbox(png)
    QImage(png).copy(x0, y0, x1 - x0, y1 - y0).save(png)
    s = open(svg).read()
    s = re.sub(r'<svg width="[^"]*" height="[^"]*"', f'<svg width="{x1 - x0}px" height="{y1 - y0}px"', s, count=1)
    s = s.replace('viewBox="0 0 3840 2160"', f'viewBox="{x0} {y0} {x1 - x0} {y1 - y0}"', 1)
    open(svg, "w").write(s)
    print(name, (x1 - x0, y1 - y0), os.path.getsize(png) // 1024, "KB png", os.path.getsize(svg) // 1024, "KB svg")
