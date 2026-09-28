"""Slide 4 pieces from the validation runs: footage/time/accuracy table and the failure-aware panel.

Reads /Users/ajith/sih-validation/slide_data.json; writes transparent PNG + SVG like the other assets.
"""
import json
import sys

sys.path.insert(0, "/tmp/demo4")
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontMetricsF, QPen  # noqa: E402

import slidekit as K  # noqa: E402

INK = QColor("#10243E")
GRID = QColor("#9FB3C8")
HEAD = QColor("#DCE8F5")
OK = QColor("#1B7F3B")
WARN = QColor("#B45309")
D = json.load(open("/Users/ajith/sih-validation/slide_data.json"))


def fits(s, pt, bold, width_in, lines=1):
    m = QFontMetricsF(K.font("Arial", pt, bold))
    w = m.horizontalAdvance(s) / K.S
    if w > width_in * lines * 0.98:
        raise SystemExit(f"overflow ({w:.2f} > {width_in:.2f} in): {s!r}")


def cell(p, x, y, w, h, s, bold=False, color=INK, fill=None, align=Qt.AlignLeft | Qt.AlignVCenter, pt=14):
    if fill is not None:
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawRect(QRectF(K.I(x), K.I(y), K.I(w), K.I(h)))
    fits(s, pt, bold, w - 0.12)
    K.text(p, (x + 0.06, y, w - 0.12, h), s, pt, bold, color, align)


def table(p, x0, y0, cols, rows, title, note=None, row_h=0.34):
    K.text(p, (x0, y0, sum(w for _, w in cols), 0.34), title, 16, True, INK)
    y = y0 + 0.40
    x = x0
    for name, w in cols:
        cell(p, x, y, w, row_h, name, True, INK, HEAD)
        x += w
    y += row_h
    for r in rows:
        x = x0
        for (name, w), v in zip(cols, r):
            color, bold = INK, False
            if isinstance(v, (tuple, list)):
                v, color, bold = v[0], QColor(v[1]), True
            cell(p, x, y, w, row_h, v, bold, color)
            x += w
        p.setPen(QPen(GRID, 2))
        p.drawLine(K.I(x0), K.I(y + row_h), K.I(x), K.I(y + row_h))
        y += row_h
    if note:
        fits(note, 14, False, sum(w for _, w in cols))
        K.text(p, (x0, y + 0.04, sum(w for _, w in cols), 0.3), note, 14, False, INK)
    return y


def draw_table(p):
    t = D["table"]
    table(p, 0.45, 0.30, [(c["name"], c["w"]) for c in t["cols"]], t["rows"], t["title"], t.get("note"))


def draw_failure(p):
    f = D["failure"]
    x0, y0, w = 0.45, 0.30, f.get("w", 6.2)
    K.text(p, (x0, y0, w, 0.34), f["title"], 16, True, INK)
    y = y0 + 0.44
    for item in f["items"]:
        p.setPen(QPen(QColor(item.get("edge", "#B45309")), 5))
        p.setBrush(QColor(item.get("fill", "#FFF7ED")))
        p.drawRoundedRect(QRectF(K.I(x0), K.I(y), K.I(w), K.I(0.62)), 18, 18)
        fits(item["when"], 14, True, w - 0.3)
        fits(item["then"], 14, False, w - 0.3)
        K.text(p, (x0 + 0.15, y + 0.04, w - 0.3, 0.27), item["when"], 14, True, INK)
        K.text(p, (x0 + 0.15, y + 0.30, w - 0.3, 0.27), item["then"], 14, False, INK)
        y += 0.72


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "table"):
        K.chrome(None, "")
        K.export(draw_table, "s4_validation_table")
    if which in ("all", "failure"):
        K.chrome(None, "")
        K.export(draw_failure, "s4_failure_aware")
