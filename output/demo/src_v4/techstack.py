"""DRISHTI-3D tech stack as an AWS-style layered diagram: icon tiles, few words, transparent for slides."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch  # noqa: E402
from arch import BLUE, INK, NAVY, ORANGE, PURPLE, SLATE, TEAL, QColor, QPainter, QRectF, Qt, dtext, group, qfont, tile  # noqa: E402

W, H = 3840, 2160
GREY = QColor("#6B7A90")

# band: (title, border, fill, header icon, [(glyph, category, name, sub)])
BANDS = [
    ("Desktop app", BLUE, "#F3F8FE", "monitor", [
        ("window", "ext", "Qt 6", "PySide6"), ("monitor", "ext", "VTK 9", "3D viewer"),
        ("ruler", "app", "Measure", "on the surface"), ("package", "ext", "PyInstaller", "native app")]),
    ("Built by us", ORANGE, "#FFF7F0", "target", [
        ("target", "ours", "Camera solver", "bundle adjustment"), ("planes", "ours", "Depth engine", "GPU multi-view stereo"),
        ("shield", "ours", "Depth check", "views must agree"), ("filter", "app", "Keyframe triage", None),
        ("clock", "app", "Clock sync", None), ("globe", "app", "Georeferencing", None)]),
    ("Vision + AI", QColor("#2E73B8"), "#F2F7FC", "graph", [
        ("eye", "ext", "OpenCV", None), ("graph", "ext", "DISK + LightGlue", "kornia"), ("cube", "ext", "Open3D", "TSDF fusion"),
        ("mask", "ext", "SegFormer", "transformers"), ("brain", "ext", "MapAnything", "fallback depth"), ("grid", "ext", "xatlas", "UV unwrap")]),
    ("Geo + media", TEAL, "#F2FBF8", "globe", [
        ("film", "ext", "PyAV · FFmpeg", None), ("globe", "ext", "pyproj", "WGS84 / UTM"), ("points", "ext", "laspy", "LAS"),
        ("pin", "ext", "tifffile", "GeoTIFF"), ("files", "app", "glTF · FBX · OBJ", "exporters"), ("pin", "app", "GeoJSON · KML", None)]),
    ("Compute", SLATE, "#F5F7FA", "chip", [
        ("python", "runtime", "Python 3.12", None), ("grid", "runtime", "NumPy", None), ("sigma", "runtime", "SciPy", None),
        ("flame", "runtime", "PyTorch", None), ("chip", "runtime", "Apple MPS", None), ("chip", "runtime", "NVIDIA CUDA", None)]),
]
SIDE = ("Build + quality", GREY, "#F6F7F9", "check", [
    ("bolt", "runtime", "uv", None), ("check", "runtime", "pytest", "739 tests"),
    ("branch", "runtime", "GitHub Actions", None), ("container", "runtime", "Docker", "CUDA image")])

X0, X1 = 60, 3180              # bands
SX0, SX1 = 3280, 3780          # side column
Y0, BH, GAP = 40, 380, 30
TS = 150                        # tile size


def draw(p, **_):
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    if not arch.TRANSPARENT:
        p.fillRect(QRectF(0, 0, W, H), QColor("#FFFFFF"))
    for k, (title, col, fill, icon, tiles) in enumerate(BANDS):
        y = Y0 + k * (BH + GAP)
        group(p, (X0, y, X1 - X0, BH), title, col, fill, icon, title_color=(ORANGE.darker(115) if title == "Built by us" else None),
              width=5 if title == "Built by us" else 4, hpx=42, hs=76)
        n = len(tiles); step = (X1 - X0 - 120) / 6
        x_start = X0 + 60 + (6 - n) * step / 2
        for i, (g, cat, name, sub) in enumerate(tiles):
            cx, cy = x_start + (i + 0.5) * step, y + 76 + 24 + TS / 2
            tile(p, g, cat, cx, cy, TS)
            p.setFont(qfont("IBMPlexSans-SemiBold", 34)); p.setPen(NAVY)
            dtext(p, QRectF(cx - 250, cy + TS / 2 + 14, 500, 44), Qt.AlignHCenter | Qt.AlignTop, name)
            if sub:
                p.setFont(qfont("IBMPlexSans-Regular", 28)); p.setPen(arch.SUB)
                dtext(p, QRectF(cx - 250, cy + TS / 2 + 58, 500, 40), Qt.AlignHCenter | Qt.AlignTop, sub)
    title, col, fill, icon, tiles = SIDE
    top, bottom = Y0, Y0 + 5 * BH + 4 * GAP
    group(p, (SX0, top, SX1 - SX0, bottom - top), title, col, fill, icon, hpx=42, hs=76)
    step = (bottom - top - 100) / len(tiles)
    for i, (g, cat, name, sub) in enumerate(tiles):
        cx, cy = (SX0 + SX1) / 2, top + 100 + (i + 0.5) * step - 40
        tile(p, g, cat, cx, cy, TS)
        p.setFont(qfont("IBMPlexSans-SemiBold", 34)); p.setPen(NAVY)
        dtext(p, QRectF(cx - 240, cy + TS / 2 + 14, 480, 44), Qt.AlignHCenter | Qt.AlignTop, name)
        if sub:
            p.setFont(qfont("IBMPlexSans-Regular", 28)); p.setPen(arch.SUB)
            dtext(p, QRectF(cx - 240, cy + TS / 2 + 58, 480, 40), Qt.AlignHCenter | Qt.AlignTop, sub)


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication(sys.argv[:1])
    arch.TRANSPARENT = "--slide" in sys.argv
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo4/techstack.png"
    arch.render(out, drawer=draw, size=(W, H))
    print("ok")
