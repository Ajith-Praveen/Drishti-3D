"""Slide 4 FEASIBILITY AND VIABILITY: screenshot strip, why-feasible tiles, proof charts, risk -> strategy diagram.
All text 14 pt (headings 16 pt), dark; each piece also exported on its own (transparent PNG + SVG)."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QImage, QPen, QPointF, QRectF, arrow, text  # noqa: E402
import arch  # noqa: E402
from arch import ORANGE, qfont, tile  # noqa: E402
import slide3_v4 as V4  # noqa: E402   rect(), tag()
from PySide6.QtGui import QFont, QFontMetricsF, QPainterPath  # noqa: E402

INK = QColor("#0B1F44")
DARK = QColor("#243349")
GREEN = QColor("#2E9E5B")
RED = QColor("#C0392B")
GREY = QColor("#8A96A8")
arch.TILE["risk"] = RED
arch.TILE["fix"] = GREEN
SCR = "/Users/ajith/sih/output/slides/slide4/assets/screens"

SHOTS = [("1_live_preview", "Live 3D preview", "watch it build"),
         ("2_confidence", "Confidence map", "green = measured"),
         ("3_uncertainty_2.5D", "Uncertainty ± m", "Terrain 2.5D mode"),
         ("4_elevation", "Elevation", "m above sea level"),
         ("5_measure_profile", "Measurements", "on the 3D surface"),
         ("6_ortho_dsm", "GeoTIFF maps", "ortho | elevation")]
FEASIBLE = [("video", "input", "Any drone", "GPS/RTK, no LiDAR"),
            ("app", "app", "One laptop", "offline, no cloud"),
            ("clock", "ours", "6.9 min", "for 11.4-min video"),
            ("check", "app", "2 real flights", "processed end to end"),
            ("package", "ext", "Open-source", "no licence fees"),
            ("shield", "runtime", "739 tests", "automated, passing")]
RISKS = [("gps", "GPS error", "no ground control"), ("eye", "Motion blur", "video compression"),
         ("user", "Moving objects", "cars, people"), ("mask", "Hidden surfaces", "under trees"),
         ("grid", "Scalability", "larger areas")]
FIXES = [("target", "RTK-aware solver", "GPS weighted by accuracy; RTK → cm"),
         ("filter", "Keyframe triage", "keeps only sharp, overlapping frames"),
         ("shield", "Depth check", "drops depth the other views reject"),
         ("check", "Confidence map", "unseen areas flagged, never faked"),
         ("chip", "Built to scale", "52 of 41k frames; Metal · CUDA · CPU")]

_warn = []


def fits(s, pt, bold, width_in, lines=1):
    f = QFont("Arial"); f.setPixelSize(int(round(pt * K.PT))); f.setBold(bold)
    fm = QFontMetricsF(f); n, cur = 1, ""
    for w_ in s.split():
        trial = (cur + " " + w_).strip()
        if fm.horizontalAdvance(trial) > I(width_in) and cur:
            n += 1; cur = w_
        else:
            cur = trial
    if n > lines or (lines == 1 and fm.horizontalAdvance(s) > I(width_in)):
        _warn.append(f"overflow: '{s}' needs {n} line(s) in {width_in:.2f} in (allowed {lines})")


def shot(p, path, x, y, w, h):
    img = QImage(path)
    img = img.scaled(int(I(w) * 2), int(I(h) * 2), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    sw, sh = img.width(), img.height(); tw, th = int(I(w) * 2), int(I(h) * 2)
    img = img.copy((sw - tw) // 2, (sh - th) // 2, tw, th)
    clip = QPainterPath(); clip.addRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(0.05), I(0.05))
    p.save(); p.setClipPath(clip); p.drawImage(QRectF(I(x), I(y), I(w), I(h)), img); p.restore()
    p.setPen(QPen(GREEN, 5)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(0.05), I(0.05))


def screens(p, y0=1.54, x0=0.45, x1=12.9, gap=0.15):
    n = len(SHOTS); w = (x1 - x0 - gap * (n - 1)) / n; h = w / 1.6
    for k, (name, title, sub) in enumerate(SHOTS):
        x = x0 + k * (w + gap)
        shot(p, f"{SCR}/{name}.png", x, y0, w, h)
        text(p, (x, y0 + h + 0.03, w, 0.26), title, 14, True, INK, Qt.AlignHCenter | Qt.AlignVCenter); fits(title, 14, True, w)
        text(p, (x, y0 + h + 0.27, w, 0.26), sub, 14, False, INK, Qt.AlignHCenter | Qt.AlignVCenter); fits(sub, 14, False, w)
    return y0 + h + 0.55


def feasible(p, x0=0.45, y0=3.36, w=7.55, h=1.42):
    V4.rect(p, x0, y0, w, h, GREEN, "#F1FAF4", 150, 5)
    V4.tag(p, x0, y0, GREEN, "check", "Why ours is more feasible")
    cw = (w - 0.2) / 3
    for k, (g, cat, title, sub) in enumerate(FEASIBLE):
        r, c = divmod(k, 3)
        x = x0 + 0.1 + c * cw; y = y0 + 0.42 + r * 0.46
        tile(p, g, cat, I(x + 0.2), I(y + 0.2), I(0.36))
        text(p, (x + 0.45, y - 0.02, cw - 0.5, 0.24), title, 14, True, INK); fits(title, 14, True, cw - 0.5)
        text(p, (x + 0.45, y + 0.2, cw - 0.5, 0.24), sub, 14, False, INK); fits(sub, 14, False, cw - 0.5)


def bar_chart(p, x, y, w, title, rows, note):
    text(p, (x, y, w, 0.25), title, 14, True, INK); fits(title, 14, True, w)
    lab_w, val_w = 0.9, 0.72
    bar_max = w - lab_w - val_w - 0.05
    vmax = max(v for _, v, _, _ in rows)
    for i, (lab, v, vtxt, col) in enumerate(rows):
        yy = y + 0.27 + i * 0.26
        text(p, (x, yy, lab_w, 0.24), lab, 14, True, INK); fits(lab, 14, True, lab_w)
        L = max(0.04, bar_max * v / vmax)
        p.setPen(Qt.NoPen); p.setBrush(col); p.drawRoundedRect(QRectF(I(x + lab_w), I(yy + 0.03), I(L), I(0.18)), I(0.03), I(0.03))
        text(p, (x + lab_w + L + 0.06, yy, val_w, 0.24), vtxt, 14, True, INK); fits(vtxt, 14, True, val_w)
    text(p, (x, y + 0.79, w, 0.24), note, 14, False, INK); fits(note, 14, False, w)


def proof(p, x0=8.12, y0=3.36, w=4.78, h=1.42):
    V4.rect(p, x0, y0, w, h, ORANGE, "#FFF6EE", 170, 5)
    V4.tag(p, x0, y0, ORANGE, "chart", "Proof in numbers", ORANGE.darker(130))
    cw = (w - 0.3) / 2
    bar_chart(p, x0 + 0.1, y0 + 0.34, cw, "vs COLMAP, same Mac",
              [("COLMAP", 33.0, "33 min", GREY), ("Ours", 9.5, "9.5 min", ORANGE)], "sparse vs our dense 3D")
    bar_chart(p, x0 + 0.2 + cw, y0 + 0.34, cw, "Our camera solver",
              [("Before", 368.0, "368 s", GREY), ("Now", 29.0, "29 s", ORANGE)], "13× faster")


def risks(p, y0=4.86):
    lab_w = 2.0
    ry = y0                        # risk row top
    fy = y0 + 0.96                 # strategy row top
    text(p, (0.45, ry - 0.02, lab_w, 0.85), "Potential challenges and risks", 16, True, BODY, Qt.AlignLeft | Qt.AlignTop, wrap=True)
    text(p, (0.45, fy + 0.04, lab_w, 1.0), "Strategies for overcoming these challenges", 16, True, BODY, Qt.AlignLeft | Qt.AlignTop, wrap=True)
    x0, x1 = 2.55, 12.9
    cw = (x1 - x0) / len(RISKS)
    for k, ((rg, rt, rs), (fg, ft, fs)) in enumerate(zip(RISKS, FIXES)):
        cx = x0 + cw * (k + 0.5)
        # risk card
        V4.rect(p, cx - cw / 2 + 0.05, ry, cw - 0.1, 0.8, RED, "#FDF1F0", 170, 4)
        tile(p, rg, "risk", I(cx), I(ry + 0.2), I(0.3))
        text(p, (cx - cw / 2 + 0.08, ry + 0.36, cw - 0.16, 0.2), rt, 14, True, RED.darker(135), Qt.AlignHCenter | Qt.AlignVCenter)
        text(p, (cx - cw / 2 + 0.08, ry + 0.555, cw - 0.16, 0.2), rs, 14, False, INK, Qt.AlignHCenter | Qt.AlignVCenter)
        fits(rt, 14, True, cw - 0.16); fits(rs, 14, False, cw - 0.16)
        # arrow
        arrow(p, [(I(cx), I(ry + 0.81)), (I(cx), I(fy - 0.01))], DARK, 7, 22)
        # strategy card
        V4.rect(p, cx - cw / 2 + 0.05, fy, cw - 0.1, 1.02, GREEN, "#F1FAF4", 170, 4)
        tile(p, fg, "fix", I(cx), I(fy + 0.2), I(0.3))
        text(p, (cx - cw / 2 + 0.08, fy + 0.36, cw - 0.16, 0.2), ft, 14, True, GREEN.darker(140), Qt.AlignHCenter | Qt.AlignVCenter)
        text(p, (cx - cw / 2 + 0.1, fy + 0.55, cw - 0.2, 0.46), fs, 14, False, INK, Qt.AlignHCenter | Qt.AlignTop, wrap=True)
        fits(ft, 14, True, cw - 0.16); fits(fs, 14, False, cw - 0.2, lines=2)


def draw(p):
    K.chrome(p, "FEASIBILITY AND VIABILITY")
    text(p, (0.45, 1.16, 8.0, 0.34), "Analysis of the feasibility of the idea", 16, True, BODY)
    V4.rect(p, 9.3, 1.19, 3.6, 0.28, GREEN, "#2E9E5B", 255, 1)
    text(p, (9.3, 1.19, 3.6, 0.28), "All screenshots: real app, real flight", 14, True, QColor("white"), Qt.AlignCenter)
    fits("All screenshots: real app, real flight", 14, True, 3.6)
    screens(p)
    feasible(p)
    proof(p)
    risks(p)


if __name__ == "__main__":
    K.export(draw, "SLIDE4")
    print("\n".join(sorted(set(_warn))) or "all text fits")
