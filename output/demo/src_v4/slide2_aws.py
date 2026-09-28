"""Slide 2 overlay in the AWS style (grouped bands of icon tiles, like the tech-stack diagram). 14/16 pt."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QPen, QPointF, QRectF, arrow, text  # noqa: E402
import arch  # noqa: E402
from arch import ORANGE, glyph, tile  # noqa: E402
from PySide6.QtGui import QPainterPath  # noqa: E402

GREEN = QColor("#2E9E5B")
X0, X1 = 0.45, 12.9

FLOW = [("video", "input", "Fly once", "any drone, one pass"), ("gps", "input", "Load", "video + GPS log"),
        ("sliders", "app", "Run", "fully automatic"), ("cube", "ours", "Measured 3D", "mesh + point cloud"),
        ("ruler", "app", "Measure", "distance · area · volume"), ("files", "app", "Export", "6 standard formats")]
TARGETS = [("target", "Accuracy ≤ 1 m", "0.29 m*"), ("clock", "< 15 min per 10 min", "6.9 min for 11.4 min"),
           ("video", "Single pass", "1 flight"), ("shield", "Entire scene", "86 % measured"),
           ("files", "6 output formats", "all 6 exported"), ("monitor", "Viewer", "desktop 3D + tools")]
UNIQUE = [("target", "Own engine", "no COLMAP / ODM wrapper"), ("bolt", "Camera solver", "13× faster than our v1*"),
          ("planes", "GPU stereo", "dense 3D from every view"), ("chart", "vs COLMAP", "3.5× faster, and dense*"),
          ("shield", "Honest 3D", "measured vs inferred")]


def band(p, y, h, title, col, fill, icon, title_col=None):
    c = QColor(fill[:7]); c.setAlpha(int(fill[7:], 16) if len(fill) == 9 else 255)
    p.setBrush(c); p.setPen(QPen(col, 5)); p.drawRect(QRectF(I(X0), I(y), I(X1 - X0), I(h)))
    s = 0.34
    p.setPen(Qt.NoPen); p.setBrush(col); p.drawRect(QRectF(I(X0), I(y), I(s), I(s)))
    glyph(p, icon, I(X0 + s / 2), I(y + s / 2), I(s * 0.62))
    text(p, (X0 + s + 0.1, y, X1 - X0 - s - 0.2, s), title, 16, True, title_col or BODY)


def check(p, cx, cy, r=0.1):
    p.setPen(QPen(QColor("white"), 5)); p.setBrush(GREEN); p.drawEllipse(QPointF(I(cx), I(cy)), I(r), I(r))
    path = QPainterPath(QPointF(I(cx - r * 0.45), I(cy + r * 0.02)))
    path.lineTo(I(cx - r * 0.1), I(cy + r * 0.38)); path.lineTo(I(cx + r * 0.5), I(cy - r * 0.35))
    p.setPen(QPen(QColor("white"), 7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)); p.setBrush(Qt.NoBrush); p.drawPath(path)


def row(p, y, items, n_cols, tile_fn):
    step = (X1 - X0 - 0.3) / n_cols
    xs = [X0 + 0.15 + step * (k + 0.5) for k in range(n_cols)]
    for k, it in enumerate(items):
        tile_fn(p, xs[k], y, it, k)
    return xs


def draw(p, only=None):
    if only is None:
        K.chrome(p, "DRISHTI-3D: One Flight → Measured 3D")
    if only is None:
        text(p, (0.45, 1.16, 12.4, 0.36), "Proposed Solution (Describe your Idea/Solution/Prototype)", 16, True, BODY)
        text(p, (0.45, 1.5, 12.45, 0.3), "DRISHTI-3D is an offline desktop app that turns one pass of drone video + GPS log into a "
         "georeferenced, measurable 3D model.", 14, False, BODY)
    bh = 1.58
    ys = [1.86, 1.86 + bh + 0.08, 1.86 + 2 * (bh + 0.08)]

    # 1: what the operator does, as a numbered AWS-style flow
    if only not in (None, 1):
        return _rest(p, only, ys, bh)
    band(p, ys[0], bh, "Detailed explanation of the proposed solution", arch.TEAL, "#F4FBF9C8", "chip")
    ty = ys[0] + 0.72

    def flow_tile(p, cx, cy, it, k):
        g, cat, name, sub = it
        tile(p, g, cat, I(cx), I(cy), I(0.5))
        p.setPen(Qt.NoPen); p.setBrush(arch.NAVY); p.drawEllipse(QPointF(I(cx - 0.3), I(cy - 0.22)), I(0.105), I(0.105))
        text(p, (cx - 0.41, cy - 0.33, 0.22, 0.22), str(k + 1), 11, True, QColor("white"), Qt.AlignCenter)
        text(p, (cx - 1.0, cy + 0.3, 2.0, 0.28), name, 14, True, BODY, Qt.AlignHCenter | Qt.AlignTop)
        text(p, (cx - 1.1, cy + 0.56, 2.2, 0.28), sub, 14, False, ORANGE.darker(125) if cat == "ours" else QColor("#1B2A44"), Qt.AlignHCenter | Qt.AlignTop)
    xs = row(p, ty, FLOW, 6, flow_tile)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(p, [(I(a + 0.34), I(ty)), (I(b - 0.34), I(ty))], QColor("#3B4A63"), 6, 26)
    if only == 1:
        return
    _rest(p, only, ys, bh)


def _rest(p, only, ys, bh):

    # 2: how it meets the problem statement's output table
    if only in (None, 2):
        _band2(p, ys, bh)
    if only in (None, 3):
        _band3(p, ys, bh)


def _band2(p, ys, bh):
    band(p, ys[1], bh, "How it addresses the problem", GREEN, "#F1FAF4C8", "check")
    ty = ys[1] + 0.72

    def target_tile(p, cx, cy, it, k):
        g, name, got = it
        col = QColor(GREEN); p.setPen(Qt.NoPen)
        from PySide6.QtGui import QLinearGradient
        gr = QLinearGradient(I(cx - 0.25), I(cy - 0.25), I(cx + 0.25), I(cy + 0.25))
        gr.setColorAt(0, col.lighter(118)); gr.setColorAt(1, col.darker(108)); p.setBrush(gr)
        p.drawRoundedRect(QRectF(I(cx - 0.25), I(cy - 0.25), I(0.5), I(0.5)), I(0.08), I(0.08))
        glyph(p, g, I(cx), I(cy), I(0.36))
        check(p, cx + 0.25, cy - 0.25)
        text(p, (cx - 1.0, cy + 0.3, 2.0, 0.28), name, 14, True, BODY, Qt.AlignHCenter | Qt.AlignTop)
        text(p, (cx - 1.05, cy + 0.56, 2.1, 0.28), got, 14, True, GREEN.darker(130), Qt.AlignHCenter | Qt.AlignTop)
    row(p, ty, TARGETS, 6, target_tile)


def _band3(p, ys, bh):
    # 3: what is ours
    band(p, ys[2], bh, "Innovation and uniqueness of the solution", ORANGE, "#FFF6EEC8", "target", ORANGE.darker(125))
    ty = ys[2] + 0.72

    def unique_tile(p, cx, cy, it, k):
        g, name, sub = it
        tile(p, g, "ours", I(cx), I(cy), I(0.5))
        text(p, (cx - 1.2, cy + 0.3, 2.4, 0.28), name, 14, True, ORANGE.darker(130), Qt.AlignHCenter | Qt.AlignTop)
        text(p, (cx - 1.2, cy + 0.56, 2.4, 0.28), sub, 14, False, BODY, Qt.AlignHCenter | Qt.AlignTop)
    row(p, ty, UNIQUE, 5, unique_tile)

    text(p, (0.45, 6.75, 12.45, 0.2), "*0.29 m: median horizontal agreement with Spain's IGN orthophoto (flight01, 13 features)  ·  "
         "13×: 368 s → 29 s vs our first solver  ·  COLMAP 33 min sparse vs DRISHTI 9.5 min dense (same Mac, no CUDA)",
         9.5, False, QColor("#1B2A44"))


if __name__ == "__main__":
    K.export(draw, "SLIDE2")
