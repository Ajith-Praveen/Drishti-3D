"""Slide 3 overlay: TECHNICAL APPROACH. Tech-stack strip + slide-sized architecture (14/16 pt)."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QPen, QPointF, QRectF, arrow, heading, text  # noqa: E402
import arch  # noqa: E402
from arch import ORANGE, TILE, glyph, tile  # noqa: E402

BLUE, TEAL, SLATE, PURPLE, INK = arch.BLUE, arch.TEAL, arch.SLATE, arch.PURPLE, arch.INK

STACK = [("Built by us", ORANGE, [("target", "ours", "Solver"), ("planes", "ours", "Stereo"), ("shield", "ours", "Checks")]),
         ("Vision + AI", BLUE, [("eye", "ext", "OpenCV"), ("graph", "ext", "LightGlue"), ("cube", "ext", "Open3D"), ("mask", "ext", "SegFormer")]),
         ("Desktop app", BLUE, [("window", "ext", "Qt 6"), ("monitor", "ext", "VTK 9")]),
         ("Runtime + GPU", SLATE, [("python", "runtime", "Python"), ("flame", "runtime", "PyTorch"), ("chip", "runtime", "Metal/CUDA")])]

STEPS = [("filter", "app", "Keyframes", "sharp only"), ("clock", "app", "Clock sync", "video↔GPS"),
         ("graph", "ext", "Features", "LightGlue"), ("target", "ours", "Cameras", "our solver"),
         ("planes", "ours", "Depth", "our GPU"), ("shield", "ours", "Check", "cross-view"),
         ("cube", "ext", "Fusion", "Open3D"), ("globe", "app", "Georef", "WGS84")]


def tech_strip(p, y_top):
    cell, gap, x = 1.0, 0.15, 0.45
    for title, col, tiles in STACK:
        w = cell * len(tiles)
        text(p, (x, y_top, w, 0.3), title, 14, True, col.darker(115) if col == ORANGE else col, Qt.AlignHCenter | Qt.AlignVCenter)
        p.setPen(QPen(col, 5)); p.drawLine(QPointF(I(x + 0.08), I(y_top + 0.36)), QPointF(I(x + w - 0.08), I(y_top + 0.36)))
        for k, (g, cat, name) in enumerate(tiles):
            cx = x + cell * (k + 0.5)
            tile(p, g, cat, I(cx), I(y_top + 0.72), I(0.5))
            text(p, (cx - 0.55, y_top + 1.02, 1.1, 0.3), name, 14, False, BODY, Qt.AlignHCenter | Qt.AlignTop)
        x += w + gap


def box(p, rect_in, title, col, fill, tag_icon=None, width=5, title_col=None, pt=14):
    x, y, w, h = rect_in
    c = QColor(fill[:7]) if isinstance(fill, str) else QColor(fill)
    if isinstance(fill, str) and len(fill) == 9:
        c.setAlpha(int(fill[7:], 16))
    p.setBrush(c); p.setPen(QPen(col, width)); p.drawRect(QRectF(I(x), I(y), I(w), I(h)))
    s = 0.3
    p.setPen(Qt.NoPen); p.setBrush(col); p.drawRect(QRectF(I(x), I(y), I(s), I(s)))
    if tag_icon:
        glyph(p, tag_icon, I(x + s / 2), I(y + s / 2), I(s * 0.62))
    text(p, (x + s + 0.08, y, w - s - 0.1, s), title, pt, True, title_col or arch.NAVY)


def architecture(p, y0, y1):
    # inputs
    box(p, (0.45, y0 + 0.45, 1.3, y1 - y0 - 0.45), "Inputs", PURPLE, "#FAF7FFB0", "video")
    tile(p, "video", "input", I(1.1), I(y0 + 1.25), I(0.5)); text(p, (0.5, y0 + 1.53, 1.2, 0.3), "Drone video", 14, False, BODY, Qt.AlignHCenter | Qt.AlignTop)
    tile(p, "gps", "input", I(1.1), I(y0 + 2.35), I(0.5)); text(p, (0.5, y0 + 2.63, 1.2, 0.3), "GPS log", 14, False, BODY, Qt.AlignHCenter | Qt.AlignTop)
    # app boundary
    ax0, ax1 = 1.95, 11.25
    box(p, (ax0, y0, ax1 - ax0, y1 - y0), "DRISHTI-3D desktop app  ·  offline, one laptop", INK, "#FFFFFF00", "app", width=6)
    # UI band
    uy0, uy1 = y0 + 0.38, y0 + 1.12
    box(p, (ax0 + 0.12, uy0, ax1 - ax0 - 0.24, uy1 - uy0), "App UI", BLUE, "#F3F8FEC0", "monitor")
    ui = [("sliders", "app", "Controls", 2.75), ("ruler", "app", "Measure", 5.0), ("pin", "app", "GeoJSON / KML", 6.75), ("monitor", "ext", "3D viewer", 9.38)]
    uy = uy0 + 0.5
    for g, cat, name, cx in ui:
        tile(p, g, cat, I(cx), I(uy), I(0.36))
        text(p, (cx + 0.25, uy - 0.15, 1.7, 0.3), name, 14, False, BODY, Qt.AlignLeft | Qt.AlignVCenter)
    arrow(p, [(I(6.1), I(uy)), (I(6.5), I(uy))])                      # measure -> GeoJSON/KML
    # worker band
    wy0, wy1 = y0 + 1.4, y1 - 0.08
    box(p, (ax0 + 0.12, wy0, ax1 - ax0 - 0.24, wy1 - wy0), "Pipeline worker", TEAL, "#F4FBF9C0", "chip")
    step_w = (ax1 - ax0 - 0.4) / len(STEPS)
    sx0 = ax0 + 0.2
    sy = wy0 + 0.9
    # our core group around steps 4-6
    cx0, cx1 = sx0 + step_w * 3 + 0.04, sx0 + step_w * 6 - 0.04
    box(p, (cx0, wy0 + 0.36, cx1 - cx0, wy1 - wy0 - 0.44), "Built by us", ORANGE, "#FFF3E8D0", "target", title_col=ORANGE.darker(125))
    centers = []
    for k, (g, cat, name, sub) in enumerate(STEPS):
        cx = sx0 + step_w * (k + 0.5); centers.append(cx)
        tile(p, g, cat, I(cx), I(sy + 0.08), I(0.5))
        # step number
        p.setPen(Qt.NoPen); p.setBrush(arch.NAVY); p.drawEllipse(QPointF(I(cx - 0.27), I(sy - 0.19)), I(0.1), I(0.1))
        text(p, (cx - 0.37, sy - 0.29, 0.2, 0.2), str(k + 1), 11, True, QColor("white"), Qt.AlignCenter)
        text(p, (cx - 0.58, sy + 0.38, 1.16, 0.3), name, 14, True, BODY, Qt.AlignHCenter | Qt.AlignTop)
        text(p, (cx - 0.6, sy + 0.64, 1.2, 0.3), sub, 14, False, K.MUTED if cat != "ours" else ORANGE.darker(125), Qt.AlignHCenter | Qt.AlignTop)
    for k in range(len(STEPS) - 1):
        col = ORANGE.darker(110) if 3 <= k < 5 else QColor("#3B4A63")
        arrow(p, [(I(centers[k] + 0.3), I(sy + 0.08)), (I(centers[k + 1] - 0.3), I(sy + 0.08))], col, 6, 26)
    # inputs -> step 1
    jx = ax0 - 0.12
    arrow(p, [(I(1.4), I(y0 + 1.25)), (I(jx), I(y0 + 1.25)), (I(jx), I(sy + 0.08))], width=6, head=1)
    arrow(p, [(I(1.4), I(y0 + 2.35)), (I(jx), I(y0 + 2.35)), (I(jx), I(sy + 0.08)), (I(centers[0] - 0.3), I(sy + 0.08))], width=6, head=26)
    # controls -> worker (dotted, "Run"); fusion -> live viewer ("live preview")
    arrow(p, [(I(2.75), I(uy + 0.2)), (I(2.75), I(wy0 - 0.02))], QColor("#3B4A63"), 5, 22, Qt.DotLine)
    text(p, (2.86, uy1 + 0.01, 1.2, 0.26), "Run", 14, True, K.MUTED, Qt.AlignLeft | Qt.AlignVCenter)
    fx = centers[6]
    arrow(p, [(I(fx), I(sy - 0.3)), (I(fx), I(uy + 0.21))], QColor("#2E73B8"), 5, 22, Qt.DashLine)
    text(p, (fx - 1.75, uy1 + 0.01, 1.65, 0.26), "live preview", 14, True, QColor("#2E73B8"), Qt.AlignRight | Qt.AlignVCenter)
    # georef -> outputs
    ox0 = 11.42
    arrow(p, [(I(centers[7] + 0.3), I(sy + 0.08)), (I(ox0 - 0.02), I(sy + 0.08))], width=6, head=26)
    box(p, (ox0, y0 + 0.45, 12.9 - ox0, y1 - y0 - 0.45), "Outputs", SLATE, "#F6F8FAC0", "files")
    tile(p, "cube", "app", I(12.16), I(y0 + 1.2), I(0.46))
    text(p, (11.45, y0 + 1.47, 1.42, 0.28), "OBJ PLY LAS", 14, False, BODY, Qt.AlignHCenter | Qt.AlignTop)
    text(p, (11.45, y0 + 1.72, 1.42, 0.28), "glTF FBX", 14, False, BODY, Qt.AlignHCenter | Qt.AlignTop)
    tile(p, "pin", "app", I(12.16), I(y0 + 2.45), I(0.46)); text(p, (11.45, y0 + 2.72, 1.42, 0.55), "GeoTIFF maps", 14, False, BODY, Qt.AlignHCenter | Qt.AlignTop, wrap=True)


def draw(p):
    K.chrome(p, "TECHNICAL APPROACH")
    heading(p, 1.16, "Technologies to be used (e.g. programming languages, frameworks, hardware)")
    tech_strip(p, 1.58)
    heading(p, 3.0, "Methodology and process for implementation (Flow Charts/Images/ working prototype)")
    architecture(p, 3.42, 6.88)


if __name__ == "__main__":
    K.export(draw, "SLIDE3")
