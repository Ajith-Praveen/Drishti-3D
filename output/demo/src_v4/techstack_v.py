"""Detailed tech stack, vertical (for a slide column): 6 AWS-style groups, 3 tiles per row, 31 technologies."""
import sys

sys.path.insert(0, "/tmp/demo4")
import arch  # noqa: E402
from arch import NAVY, ORANGE, QColor, QPainter, QRectF, Qt, dtext, group, qfont, tile  # noqa: E402

W, H = 1500, 2100
GREY = QColor("#6B7A90")
GROUPS = [
    ("Built by us", ORANGE, "#FFF7F0", "target", [("target", "ours", "Camera solver"), ("planes", "ours", "Depth engine"),
                                                   ("shield", "ours", "Depth check"), ("filter", "app", "Keyframe triage"),
                                                   ("clock", "app", "Clock sync"), ("globe", "app", "Georeferencing")]),
    ("Vision + AI", QColor("#2E73B8"), "#F2F7FC", "graph", [("eye", "ext", "OpenCV"), ("graph", "ext", "DISK + LightGlue"),
                                                            ("cube", "ext", "Open3D TSDF"), ("mask", "ext", "SegFormer"),
                                                            ("brain", "ext", "MapAnything"), ("grid", "ext", "xatlas UV")]),
    ("Desktop app", arch.BLUE, "#F3F8FE", "monitor", [("window", "ext", "Qt 6 · PySide6"), ("monitor", "ext", "VTK 9 viewer"),
                                                      ("package", "ext", "PyInstaller")]),
    ("Geo + media", arch.TEAL, "#F2FBF8", "globe", [("film", "ext", "PyAV · FFmpeg"), ("globe", "ext", "pyproj"), ("points", "ext", "laspy"),
                                                    ("pin", "ext", "tifffile"), ("files", "app", "glTF · FBX · OBJ"), ("pin", "app", "GeoJSON · KML")]),
    ("Compute", arch.SLATE, "#F5F7FA", "chip", [("python", "runtime", "Python 3.12"), ("grid", "runtime", "NumPy"),
                                                ("sigma", "runtime", "SciPy"), ("flame", "runtime", "PyTorch"),
                                                ("chip", "runtime", "Apple MPS"), ("chip", "runtime", "NVIDIA CUDA")]),
]
HEAD, ROW, GAP, TS = 80, 164, 22, 88


def draw(p, **_):
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    y = 4
    for title, col, fill, icon, tiles in GROUPS:
        rows = (len(tiles) + 2) // 3
        h = HEAD + rows * ROW + 10
        group(p, (4, y, W - 8, h), title, col, fill, icon, title_color=(ORANGE.darker(115) if title == "Built by us" else None),
              width=6 if title == "Built by us" else 4, hpx=54, hs=HEAD)
        for k, (g, cat, name) in enumerate(tiles):
            r, c = divmod(k, 3)
            cx, cy = 250 + c * 500, y + HEAD + 14 + r * ROW + TS / 2
            tile(p, g, cat, cx, cy, TS)
            p.setFont(qfont("IBMPlexSans-SemiBold", 46)); p.setPen(NAVY)
            dtext(p, QRectF(cx - 248, cy + TS / 2 + 8, 496, 58), Qt.AlignHCenter | Qt.AlignTop, name)
        y += h + GAP
    return y


if __name__ == "__main__":
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage
    app = QGuiApplication(sys.argv[:1])
    arch.TRANSPARENT = True
    img = QImage(10, 10, QImage.Format_ARGB32_Premultiplied); p = QPainter(img); print("height used", draw(p)); p.end()
