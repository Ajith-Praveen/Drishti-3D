"""Slide 3, readable: every label 14 pt, dark text. Detailed tech-stack cards, the full pipeline architecture
re-laid out at slide scale, and a working-prototype column with real app screenshots."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QPen, QPointF, QRectF, arrow, text  # noqa: E402
import arch  # noqa: E402
from arch import ORANGE, glyph, tile  # noqa: E402
from PySide6.QtGui import QImage, QPainterPath  # noqa: E402

INK = QColor("#0B1F44")          # labels: near-black navy, never grey
DARK = QColor("#1B2A44")
TEAL, BLUE, SLATE, PURPLE = arch.TEAL, arch.BLUE, arch.SLATE, arch.PURPLE
GREEN = QColor("#2E9E5B")

CARDS = [
    ("Built by us", ORANGE, "target", ["Camera solver", "Depth engine", "Depth check", "Keyframes"]),
    ("Vision + AI", QColor("#2E73B8"), "graph", ["OpenCV", "LightGlue", "Open3D fusion", "SegFormer"]),
    ("Desktop app", BLUE, "monitor", ["Qt 6 · PySide6", "VTK 9 viewer", "Measure tools", "PyInstaller"]),
    ("Geo + formats", TEAL, "globe", ["pyproj · UTM", "laspy · LAS", "tifffile · TIFF", "glTF · FBX"]),
    ("Compute", SLATE, "chip", ["Python 3.12", "PyTorch", "NumPy · SciPy", "PyAV · FFmpeg"]),
    ("Hardware", PURPLE, "video", ["Any drone", "Apple M-series", "NVIDIA CUDA", "Offline laptop"]),
    ("Quality", QColor("#4A5A73"), "check", ["pytest · 739", "GitHub Actions", "Docker · CUDA", "uv builds"]),
]


def rect(p, x, y, w, h, col, fill_hex=None, alpha=0, width=4, dash=False):
    c = QColor(fill_hex or "#FFFFFF"); c.setAlpha(alpha); p.setBrush(c)
    p.setPen(QPen(col, width, Qt.DashLine if dash else Qt.SolidLine)); p.drawRect(QRectF(I(x), I(y), I(w), I(h)))


def tag(p, x, y, col, icon, title, title_col=None, s=0.3, pt=14, align_right_w=None):
    if align_right_w:                      # tag in the top-right corner of a box of width align_right_w
        x = x + align_right_w - s
        p.setPen(Qt.NoPen); p.setBrush(col); p.drawRect(QRectF(I(x), I(y), I(s), I(s)))
        glyph(p, icon, I(x + s / 2), I(y + s / 2), I(s * 0.62))
        text(p, (x - 2.2, y, 2.15, s), title, pt, True, title_col or INK, Qt.AlignRight | Qt.AlignVCenter)
        return
    p.setPen(Qt.NoPen); p.setBrush(col); p.drawRect(QRectF(I(x), I(y), I(s), I(s)))
    glyph(p, icon, I(x + s / 2), I(y + s / 2), I(s * 0.62))
    text(p, (x + s + 0.07, y, 4.0, s), title, pt, True, title_col or INK, Qt.AlignLeft | Qt.AlignVCenter)


def cards(p, y, h):
    n, gap = len(CARDS), 0.08
    w = (12.45 - gap * (n - 1)) / n
    for k, (title, col, icon, items) in enumerate(CARDS):
        x = 0.45 + k * (w + gap)
        rect(p, x, y, w, h, col, "#FFFFFF", 0, 4)
        tag(p, x, y, col, icon, title, col.darker(125) if title == "Built by us" else INK)
        for i, it in enumerate(items):
            yy = y + 0.38 + i * 0.225
            p.setPen(Qt.NoPen); p.setBrush(col); p.drawRect(QRectF(I(x + 0.1), I(yy + 0.07), I(0.08), I(0.08)))
            text(p, (x + 0.24, yy, w - 0.28, 0.23), it, 14, title == "Built by us", ORANGE.darker(130) if title == "Built by us" else INK,
                 Qt.AlignLeft | Qt.AlignVCenter)


# worker columns
C0, C1, NC = 1.72, 9.14, 8
STEP = (C1 - C0) / NC
COL = [C0 + STEP * (k + 0.5) for k in range(NC)]
R1, R2 = 4.98, 6.08                      # row centres
TS = 0.34


def step(p, col, row_y, glyph_name, cat, lines, core=False):
    cx = COL[col]
    tile(p, glyph_name, cat, I(cx), I(row_y), I(TS))
    for i, ln in enumerate(lines):
        text(p, (cx - STEP / 2 - 0.05, row_y + 0.19 + i * 0.215, STEP + 0.1, 0.23), ln, 14, True,
             ORANGE.darker(130) if core else INK, Qt.AlignHCenter | Qt.AlignVCenter)


def architecture(p):
    y0, y1 = 3.32, 6.88
    # inputs
    rect(p, 0.45, y0 + 0.3, 0.98, y1 - y0 - 0.3, PURPLE, "#FAF7FF", 120, 4)
    tag(p, 0.45, y0 + 0.3, PURPLE, "video", "Inputs")
    for yy, g, lab in ((4.45, "video", "Video"), (5.55, "gps", "GPS log")):
        tile(p, g, "input", I(0.94), I(yy), I(TS)); text(p, (0.47, yy + 0.19, 0.94, 0.23), lab, 14, True, INK, Qt.AlignHCenter | Qt.AlignVCenter)
    # app boundary
    ax0, ax1 = 1.55, 9.3
    rect(p, ax0, y0, ax1 - ax0, y1 - y0, arch.INK, width=6)
    tag(p, ax0, y0, arch.INK, "app", "DRISHTI-3D desktop app  ·  offline, one laptop")
    # UI band
    uy0, uy1 = y0 + 0.4, y0 + 0.98
    rect(p, ax0 + 0.1, uy0, ax1 - ax0 - 0.2, uy1 - uy0, BLUE, "#F3F8FE", 150, 4)
    uy = (uy0 + uy1) / 2
    for x, g, cat, lab in ((1.86, "sliders", "app", "Controls"), (3.36, "chart", "app", "Live report"), (4.8, "ruler", "app", "Measure"),
                           (6.38, "pin", "app", "GIS export"), (COL[6], "monitor", "ext", "3D viewer")):
        tile(p, g, cat, I(x), I(uy), I(0.3)); text(p, (x + 0.2, uy - 0.13, 1.35, 0.26), lab, 14, True, INK, Qt.AlignLeft | Qt.AlignVCenter)
    arrow(p, [(I(5.82), I(uy)), (I(6.18), I(uy))], width=6, head=22)           # measure -> GIS export
    # worker band
    wy0, wy1 = y0 + 1.1, y1 - 0.06
    rect(p, ax0 + 0.1, wy0, ax1 - ax0 - 0.2, wy1 - wy0, TEAL, "#F4FBF9", 150, 4)
    tag(p, ax0 + 0.1, wy0, TEAL, "chip", "Pipeline worker")
    # our core box (row 2, columns 4-6), its tag top-right so the down-arrow into it stays clear
    cx0, cx1 = COL[3] - STEP / 2 + 0.03, COL[5] + STEP / 2 - 0.03
    rect(p, cx0, R2 - 0.42, cx1 - cx0, wy1 - (R2 - 0.42) - 0.05, ORANGE, "#FFF3E8", 210, 5)
    tag(p, cx0, R2 - 0.42, ORANGE, "target", "Built by us", ORANGE.darker(130), align_right_w=cx1 - cx0)
    # steps
    step(p, 0, R1, "film", "ext", ["Decode"])
    step(p, 1, R1, "filter", "app", ["Keyframes"])
    step(p, 2, R1, "clock", "app", ["Clock", "sync"])
    step(p, 3, R1, "graph", "ext", ["Features"])
    step(p, 6, R1, "globe", "app", ["Georef"])
    step(p, 7, R1, "files", "app", ["Export"])
    step(p, 3, R2, "target", "ours", ["Camera", "solver"], core=True)
    step(p, 4, R2, "planes", "ours", ["Depth", "engine"], core=True)
    step(p, 5, R2, "shield", "ours", ["Depth", "check"], core=True)
    step(p, 6, R2, "cube", "ext", ["3D", "fusion"])
    dark = QColor("#243349")
    for a, b in ((0, 1), (1, 2), (2, 3), (6, 7)):
        arrow(p, [(I(COL[a] + TS / 2 + 0.05), I(R1)), (I(COL[b] - TS / 2 - 0.05), I(R1))], dark, 6, 22)
    for a, b in ((3, 4), (4, 5), (5, 6)):
        arrow(p, [(I(COL[a] + TS / 2 + 0.05), I(R2)), (I(COL[b] - TS / 2 - 0.05), I(R2))], ORANGE.darker(115), 6, 22)
    arrow(p, [(I(COL[3]), I(R1 + 0.42)), (I(COL[3]), I(R2 - TS / 2 - 0.04))], dark, 6, 22)          # features -> solver
    arrow(p, [(I(COL[6]), I(R2 - TS / 2 - 0.04)), (I(COL[6]), I(R1 + 0.42))], dark, 6, 22)          # fusion -> georef
    arrow(p, [(I(COL[6]), I(R1 - TS / 2 - 0.04)), (I(COL[6]), I(uy + 0.19))], dark, 6, 22)          # georef -> viewer
    arrow(p, [(I(1.86), I(uy + 0.18)), (I(1.86), I(wy0 - 0.02))], dark, 5, 20, Qt.DotLine)          # controls -> worker
    # inputs -> decode
    jx = ax0 - 0.08
    arrow(p, [(I(0.94 + TS / 2 + 0.03), I(4.45)), (I(jx), I(4.45)), (I(jx), I(R1))], dark, 6, 1)
    arrow(p, [(I(0.94 + TS / 2 + 0.03), I(5.55)), (I(jx), I(5.55)), (I(jx), I(R1)), (I(COL[0] - TS / 2 - 0.05), I(R1))], dark, 6, 22)
    # what our core does (plain words), row 2 columns 1-3
    tx = C0 + 0.02
    text(p, (tx, R2 - 0.44, 2.7, 0.26), "What our core does:", 14, True, ORANGE.darker(130), Qt.AlignLeft | Qt.AlignVCenter)
    for i, s in enumerate(("Solver: exact camera poses", "Depth: distance per pixel", "Check: only agreed depth")):
        text(p, (tx, R2 - 0.18 + i * 0.23, 2.75, 0.24), s, 14, False, INK, Qt.AlignLeft | Qt.AlignVCenter)
    # outputs under Export (row 2, column 8)
    arrow(p, [(I(COL[7]), I(R1 + 0.44)), (I(COL[7]), I(R2 - 0.36))], dark, 6, 20)
    for i, s in enumerate(("OBJ · PLY", "LAS · TIFF", "glTF · FBX")):
        text(p, (COL[7] - 0.55, R2 - 0.32 + i * 0.23, 1.1, 0.24), s, 14, True, INK, Qt.AlignHCenter | Qt.AlignVCenter)


def shot(p, path, crop, rect_in, radius=0.08):
    img = QImage(path).copy(*crop)
    x, y, w, h = rect_in
    s = max(I(w) / img.width(), I(h) / img.height()); sw, sh = I(w) / s, I(h) / s
    src = QRectF((img.width() - sw) / 2, (img.height() - sh) / 2, sw, sh)
    img = img.copy(int(src.x()), int(src.y()), int(src.width()), int(src.height())).scaled(int(I(w)), int(I(h)), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    clip = QPainterPath(); clip.addRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(radius), I(radius))
    p.save(); p.setClipPath(clip); p.drawImage(QRectF(I(x), I(y), I(w), I(h)), img); p.restore()
    p.setPen(QPen(QColor("#1B2A44"), 4)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(radius), I(radius))


def prototype(p):
    x0, y0, x1, y1 = 9.45, 3.32, 12.9, 6.88
    rect(p, x0, y0, x1 - x0, y1 - y0, GREEN, "#F1FAF4", 150, 5)
    tag(p, x0, y0, GREEN, "monitor", "Working prototype (real app)")
    w = x1 - x0 - 0.2; h = w * 9 / 16
    top = y0 + 0.42 + (y1 - y0 - 0.42 - h - 0.62) / 2
    shot(p, "/tmp/demo3/feat/f0000.jpg", (0, 0, 1920, 1080), (x0 + 0.1, top, w, h), 0.06)
    text(p, (x0 + 0.1, top + h + 0.06, w, 0.26), "Real app, real 11-min flight", 14, True, INK, Qt.AlignHCenter | Qt.AlignVCenter)
    text(p, (x0 + 0.1, top + h + 0.32, w, 0.26), "52 cameras · 8 M points · done", 14, False, INK, Qt.AlignHCenter | Qt.AlignVCenter)


def draw(p):
    K.chrome(p, "TECHNICAL APPROACH")
    text(p, (0.45, 1.18, 12.45, 0.34), "Technologies to be used (e.g. programming languages, frameworks, hardware)", 16, True, BODY)
    cards(p, 1.56, 1.3)
    text(p, (0.45, 2.94, 12.45, 0.34), "Methodology and process for implementation (Flow Charts/Images/ working prototype)", 16, True, BODY)
    architecture(p)
    prototype(p)


if __name__ == "__main__":
    K.export(draw, "SLIDE3")
