"""Slide 3 at 14 pt everywhere: AWS-style tech cards (icon per item), architecture with numbered steps, prototype panel."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QPen, QPointF, QRectF, arrow, text  # noqa: E402
import arch  # noqa: E402
from arch import ORANGE, glyph, tile  # noqa: E402
import slide3_v4 as V4  # noqa: E402   rect(), tag(), shot()

INK = QColor("#0B1F44")
DARK = QColor("#243349")
GREEN = QColor("#2E9E5B")
TEAL, BLUE, SLATE, PURPLE = arch.TEAL, arch.BLUE, arch.SLATE, arch.PURPLE

CARDS = [
    ("Built by us", ORANGE, "target", [("target", "ours", "Camera solver"), ("planes", "ours", "Depth engine"),
                                       ("shield", "ours", "Depth check"), ("filter", "app", "Keyframes")]),
    ("Vision + AI", QColor("#2E73B8"), "graph", [("eye", "ext", "OpenCV"), ("graph", "ext", "LightGlue"),
                                                 ("cube", "ext", "Open3D fusion"), ("mask", "ext", "SegFormer")]),
    ("Desktop app", BLUE, "monitor", [("window", "ext", "Qt 6 · PySide"), ("monitor", "ext", "VTK 9 viewer"),
                                      ("ruler", "app", "Measure tools"), ("package", "ext", "PyInstaller")]),
    ("Geo + formats", TEAL, "globe", [("globe", "ext", "pyproj · UTM"), ("points", "ext", "laspy · LAS"),
                                      ("pin", "ext", "GeoTIFF"), ("files", "app", "glTF · FBX")]),
    ("Compute", SLATE, "chip", [("python", "runtime", "Python 3.12"), ("flame", "runtime", "PyTorch"),
                                ("sigma", "runtime", "NumPy · SciPy"), ("film", "runtime", "FFmpeg")]),
    ("Hardware", PURPLE, "video", [("video", "input", "Any drone"), ("chip", "runtime", "Apple Metal"),
                                   ("chip", "runtime", "NVIDIA CUDA"), ("app", "runtime", "No cloud")]),
    ("Quality", QColor("#4A5A73"), "check", [("check", "runtime", "pytest · 739"), ("branch", "runtime", "GitHub CI"),
                                             ("container", "runtime", "Docker"), ("bolt", "runtime", "uv builds")]),
]


def cards(p, y, h):
    n, gap = len(CARDS), 0.08
    w = (12.45 - gap * (n - 1)) / n
    for k, (title, col, icon, items) in enumerate(CARDS):
        x = 0.45 + k * (w + gap)
        V4.rect(p, x, y, w, h, col, "#FFFFFF", 0, 4)
        V4.tag(p, x, y, col, icon, title, col.darker(125) if title == "Built by us" else INK)
        for i, (g, cat, name) in enumerate(items):
            yy = y + 0.46 + i * 0.235
            tile(p, g, cat, I(x + 0.2), I(yy), I(0.2))
            text(p, (x + 0.35, yy - 0.12, w - 0.37, 0.24), name, 14, title == "Built by us",
                 ORANGE.darker(130) if title == "Built by us" else INK, Qt.AlignLeft | Qt.AlignVCenter)


# worker grid
C0, C1, NC = 1.7, 9.72, 8
STEP = (C1 - C0) / NC
COL = [C0 + STEP * (k + 0.5) for k in range(NC)]
R1, R2 = 4.8, 6.1
TS = 0.32


def badge(p, cx, cy, n):
    p.setPen(Qt.NoPen); p.setBrush(arch.NAVY); p.drawEllipse(QPointF(I(cx), I(cy)), I(0.11), I(0.11))
    text(p, (cx - 0.11, cy - 0.11, 0.22, 0.22), str(n), 12, True, QColor("white"), Qt.AlignCenter)


def step(p, n, col, row_y, g, cat, lines, core=False):
    cx = COL[col]
    tile(p, g, cat, I(cx), I(row_y), I(TS))
    badge(p, cx - TS / 2 - 0.02, row_y - TS / 2 - 0.02, n)
    for i, ln in enumerate(lines):
        text(p, (cx - STEP / 2 - 0.06, row_y + 0.18 + i * 0.215, STEP + 0.12, 0.23), ln, 14, True,
             ORANGE.darker(130) if core else INK, Qt.AlignHCenter | Qt.AlignVCenter)


def architecture(p):
    y0, y1 = 3.3, 6.88
    # inputs
    V4.rect(p, 0.45, y0 + 0.32, 0.95, y1 - y0 - 0.32, PURPLE, "#FAF7FF", 120, 4)
    V4.tag(p, 0.45, y0 + 0.32, PURPLE, "video", "Inputs")
    for yy, g, lab in ((4.45, "video", "Video"), (5.55, "gps", "GPS log")):
        tile(p, g, "input", I(0.925), I(yy), I(TS))
        text(p, (0.47, yy + 0.18, 0.91, 0.23), lab, 14, True, INK, Qt.AlignHCenter | Qt.AlignVCenter)
    # app boundary
    ax0, ax1 = 1.55, 9.88
    V4.rect(p, ax0, y0, ax1 - ax0, y1 - y0, arch.INK, width=6)
    V4.tag(p, ax0, y0, arch.INK, "app", "DRISHTI-3D desktop app  ·  offline, one laptop")
    # app UI band
    uy0, uy1 = y0 + 0.36, y0 + 0.9
    V4.rect(p, ax0 + 0.08, uy0, ax1 - ax0 - 0.16, uy1 - uy0, BLUE, "#F3F8FE", 150, 4)
    uy = (uy0 + uy1) / 2
    for x, g, cat, lab in ((1.86, "sliders", "app", "Controls"), (3.45, "chart", "app", "Live report"),
                           (5.25, "ruler", "app", "Measure"), (6.78, "pin", "app", "GIS export"), (COL[6], "monitor", "ext", "3D viewer")):
        tile(p, g, cat, I(x), I(uy), I(0.28))
        text(p, (x + 0.19, uy - 0.13, 1.4, 0.26), lab, 14, True, INK, Qt.AlignLeft | Qt.AlignVCenter)
    arrow(p, [(I(6.28), I(uy)), (I(6.6), I(uy))], DARK, 6, 20)                        # measure -> GIS export
    # pipeline worker band
    wy0, wy1 = y0 + 0.98, y1 - 0.05
    V4.rect(p, ax0 + 0.08, wy0, ax1 - ax0 - 0.16, wy1 - wy0, TEAL, "#F4FBF9", 150, 4)
    V4.tag(p, ax0 + 0.08, wy0, TEAL, "chip", "Pipeline worker")
    # our core box: row 2, columns 4-6, tag top-right so the arrow from Features stays clear
    cx0, cx1 = COL[3] - STEP / 2 + 0.03, COL[5] + STEP / 2 - 0.03
    cy0 = R2 - 0.5
    V4.rect(p, cx0, cy0, cx1 - cx0, wy1 - cy0 - 0.05, ORANGE, "#FFF3E8", 210, 5)
    V4.tag(p, cx0, cy0, ORANGE, "target", "Our core", ORANGE.darker(130), align_right_w=cx1 - cx0)
    # steps
    step(p, 1, 0, R1, "film", "ext", ["Decode"])
    step(p, 2, 1, R1, "filter", "app", ["Keyframes"])
    step(p, 3, 2, R1, "clock", "app", ["Clock", "sync"])
    step(p, 4, 3, R1, "graph", "ext", ["Features"])
    step(p, 5, 3, R2, "target", "ours", ["Camera", "solver"], True)
    step(p, 6, 4, R2, "planes", "ours", ["Depth", "engine"], True)
    step(p, 7, 5, R2, "shield", "ours", ["Depth", "check"], True)
    step(p, 8, 6, R2, "cube", "ext", ["3D", "fusion"])
    step(p, 9, 6, R1, "globe", "app", ["Georef"])
    step(p, 10, 7, R1, "files", "app", ["Export"])
    for a, b in ((0, 1), (1, 2), (2, 3), (6, 7)):
        arrow(p, [(I(COL[a] + TS / 2 + 0.05), I(R1)), (I(COL[b] - TS / 2 - 0.16), I(R1))], DARK, 6, 22)
    for a, b in ((3, 4), (4, 5), (5, 6)):
        arrow(p, [(I(COL[a] + TS / 2 + 0.05), I(R2)), (I(COL[b] - TS / 2 - 0.16), I(R2))], ORANGE.darker(115), 6, 22)
    arrow(p, [(I(COL[3]), I(R1 + 0.42)), (I(COL[3]), I(R2 - TS / 2 - 0.15))], DARK, 6, 22)          # features -> solver
    arrow(p, [(I(COL[6] + 0.12), I(R2 - TS / 2 - 0.03)), (I(COL[6] + 0.12), I(R1 + 0.42))], DARK, 6, 22)   # fusion -> georef
    arrow(p, [(I(COL[6]), I(R1 - TS / 2 - 0.15)), (I(COL[6]), I(uy + 0.18))], DARK, 6, 22)          # georef -> 3D viewer
    arrow(p, [(I(1.86), I(uy + 0.17)), (I(1.86), I(wy0 - 0.02))], DARK, 5, 20, Qt.DotLine)         # controls -> worker (Run)
    jx = ax0 - 0.08                                                                                  # inputs -> decode
    arrow(p, [(I(0.925 + TS / 2 + 0.03), I(4.45)), (I(jx), I(4.45)), (I(jx), I(R1))], DARK, 6, 1)
    arrow(p, [(I(0.925 + TS / 2 + 0.03), I(5.55)), (I(jx), I(5.55)), (I(jx), I(R1)), (I(COL[0] - TS / 2 - 0.16), I(R1))], DARK, 6, 22)
    # what our core does, in plain words (row 2, columns 1-3)
    tx = C0 + 0.04
    text(p, (tx, R2 - 0.52, 2.9, 0.26), "What our core does:", 14, True, ORANGE.darker(130), Qt.AlignLeft | Qt.AlignVCenter)
    for i, s in enumerate(("Solver: exact camera positions", "Engine: distance to every pixel", "Check: keep only agreed depth")):
        text(p, (tx, R2 - 0.26 + i * 0.235, 2.95, 0.25), s, 14, False, INK, Qt.AlignLeft | Qt.AlignVCenter)
    # 11: outputs under Export
    arrow(p, [(I(COL[7]), I(R1 + 0.42)), (I(COL[7]), I(R2 - 0.42))], DARK, 6, 20)
    for i, s in enumerate(("OBJ · PLY", "LAS · TIFF", "glTF · FBX")):
        text(p, (COL[7] - 0.58, R2 - 0.37 + i * 0.235, 1.16, 0.25), s, 14, True, INK, Qt.AlignHCenter | Qt.AlignVCenter)


def prototype(p):
    x0, y0, x1, y1 = 10.02, 3.3, 12.9, 6.88
    w = x1 - x0 - 0.16; h = w * 9 / 16
    ph = 0.44 + h + 0.1 + 3 * 0.27 + 0.1
    top = y0 + (y1 - y0 - ph) / 2
    V4.rect(p, x0, top, x1 - x0, ph, GREEN, "#F1FAF4", 150, 5)
    V4.tag(p, x0, top, GREEN, "monitor", "Working prototype")
    V4.shot(p, "/tmp/demo3/feat/f0000.jpg", (0, 0, 1920, 1080), (x0 + 0.08, top + 0.44, w, h), 0.05)
    y = top + 0.44 + h + 0.1
    for s, bold in (("Real app, real flight", True), ("52 cameras solved", False), ("8 M-point 3D model", False)):
        text(p, (x0 + 0.08, y, w, 0.26), s, 14, bold, INK, Qt.AlignHCenter | Qt.AlignVCenter)
        y += 0.27


def draw(p):
    K.chrome(p, "TECHNICAL APPROACH")
    text(p, (0.45, 1.16, 12.45, 0.34), "Technologies to be used (e.g. programming languages, frameworks, hardware)", 16, True, BODY)
    cards(p, 1.52, 1.4)
    text(p, (0.45, 2.94, 12.45, 0.34), "Methodology and process for implementation (Flow Charts/Images/ working prototype)", 16, True, BODY)
    architecture(p)
    prototype(p)


if __name__ == "__main__":
    K.export(draw, "SLIDE3")
