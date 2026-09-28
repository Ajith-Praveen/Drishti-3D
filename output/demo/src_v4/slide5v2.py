"""Slide 5 v2 IMPACT AND BENEFITS, in the style of slides 2-4: AWS group of the problem statement's 8 applications,
red -> green "today vs DRISHTI-3D" cards, 4 big-number benefit tiles, and a mission timeline of real screenshots."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QImage, QPen, QPointF, QRectF, arrow, text  # noqa: E402
import arch  # noqa: E402
from arch import ORANGE, glyph, tile  # noqa: E402
import slide3_v4 as V4  # noqa: E402   rect(), tag()
from PySide6.QtGui import QFont, QFontMetricsF, QPainterPath  # noqa: E402

INK = QColor("#0B1F44")
DARK = QColor("#243349")
GREEN = QColor("#2E9E5B")
RED = QColor("#C0392B")
NAVY = QColor("#1F497D")
PURPLE = QColor("#8C4FFF")
TEAL2, AMBER = QColor("#0E8C7F"), QColor("#C2620A")
arch.TILE.update({"fix": GREEN, "risk": RED, "navy": NAVY, "teal2": TEAL2, "amber": AMBER, "grey": QColor("#7C8799")})
SCR = "/Users/ajith/sih/output/slides/slide4/assets/screens"
CLIP = "/tmp/demo3/clip/0100.jpg"

APPS = [("globe", "navy", "Border & strategic mapping"), ("bolt", "navy", "Disaster damage mapping"),
        ("grid", "navy", "Urban planning, smart cities"), ("eye", "navy", "Infrastructure inspection"),
        ("package", "navy", "Construction monitoring"), ("files", "navy", "Archaeological records"),
        ("cube", "navy", "Digital twin generation"), ("target", "navy", "Military recon & planning")]
PAIRS = [("video", "Multiple passes per mission", "check", "One pass, one flight"),
         ("grid", "Specialized flight planning", "gps", "No special flight plan"),
         ("user", "Ground control point crews", "shield", "No GCPs: GPS or RTK"),
         ("clock", "Significant post-processing", "bolt", "~7 min on one laptop")]
KPIS = [("user", "teal2", "SOCIAL", "Faster relief", "damage maps in minutes, no ground crews"),
        ("chart", "amber", "ECONOMIC", "₹0 licences", "open-source; one laptop, no cloud GPU"),
        ("globe", "fix", "ENVIRONMENTAL", "1 flight", "not repeat passes: less battery and fuel"),
        ("shield", "navy", "STRATEGIC", "100% offline", "imagery never leaves the device")]
STEPS = [(CLIP, "Fly one pass", "11.4-min video"), (f"{SCR}/1_live_preview.png", "Live preview", "from ~2 min"),
         (f"{SCR}/5_measure_profile.png", "Measured 3D", "done in 6.9 min"), (f"{SCR}/6_ortho_dsm.png", "GIS-ready", "GeoTIFF + LAS")]
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


def pic(p, path, x, y, w, h, border=GREEN, radius=0.05):
    img = QImage(path).scaled(int(I(w) * 2), int(I(h) * 2), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    tw, th = int(I(w) * 2), int(I(h) * 2)
    img = img.copy((img.width() - tw) // 2, (img.height() - th) // 2, tw, th)
    clip = QPainterPath(); clip.addRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(radius), I(radius))
    p.save(); p.setClipPath(clip); p.drawImage(QRectF(I(x), I(y), I(w), I(h)), img); p.restore()
    p.setPen(QPen(border, 5)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(I(x), I(y), I(w), I(h)), I(radius), I(radius))


def who(p, x0=0.45, y0=1.56, w=6.6, h=1.9):
    V4.rect(p, x0, y0, w, h, NAVY, "#F3F6FB", 170, 5)
    V4.tag(p, x0, y0, NAVY, "user", "Target users & applications")
    fits("Target users & applications", 14, True, w - 0.45)
    cw = (w - 0.2) / 2
    for k, (g, cat, label) in enumerate(APPS):
        c, r = divmod(k, 4)
        x = x0 + 0.1 + c * cw; y = y0 + 0.55 + r * 0.36
        tile(p, g, cat, I(x + 0.22), I(y), I(0.3))
        text(p, (x + 0.45, y - 0.14, cw - 0.5, 0.28), label, 14, True, INK); fits(label, 14, True, cw - 0.5)


def pill(p, x, y, w, h, fill, edge, g, cat, label, bold, col):
    r = QRectF(I(x), I(y), I(w), I(h))
    p.setPen(QPen(edge, 3)); p.setBrush(fill); p.drawRoundedRect(r, I(h / 2), I(h / 2))
    tile(p, g, cat, I(x + h / 2 + 0.02), I(y + h / 2), I(h - 0.07))
    text(p, (x + h + 0.12, y, w - h - 0.18, h), label, 14, bold, col); fits(label, 14, bold, w - h - 0.18)


def today_vs(p, x0=0.45, y0=3.56, w=6.6, h=1.66):
    lw, aw = 3.02, 0.36
    V4.rect(p, x0, y0, w, h, QColor("#44546F"), "#FFFFFF", 0, 5)
    V4.tag(p, x0, y0, RED, "eye", "Present gaps", RED.darker(135))
    V4.tag(p, x0 + 0.15 + lw + aw, y0, GREEN, "check", "With DRISHTI-3D", GREEN.darker(150))
    rw = w - 0.3 - lw - aw
    for k, (lg, lt, rg, rt) in enumerate(PAIRS):
        y = y0 + 0.44 + k * 0.3
        xl = x0 + 0.15
        pill(p, xl, y, lw, 0.26, QColor("#F1F3F6"), QColor("#C3CAD5"), lg, "grey", lt, False, INK)
        arrow(p, [(I(xl + lw + 0.05), I(y + 0.13)), (I(xl + lw + aw - 0.05), I(y + 0.13))], DARK, 6, 18)
        pill(p, xl + lw + aw, y, rw, 0.26, QColor("#EAF6EE"), QColor("#8FCBA3"), rg, "fix", rt, True, GREEN.darker(150))


def kpis(p, x0=7.25, x1=12.9, y0=1.82, rh=1.64, gap=0.1):
    cw = (x1 - x0 - gap) / 2
    for k, (g, cat, cat_label, big, desc) in enumerate(KPIS):
        r, c = divmod(k, 2)
        x, y = x0 + c * (cw + gap), y0 + r * (rh + gap)
        col = arch.TILE[cat]
        V4.rect(p, x, y, cw, rh, QColor("#C9D1DC"), "#FFFFFF", 255, 3)
        p.setPen(Qt.NoPen); p.setBrush(col); p.drawRect(QRectF(I(x), I(y), I(cw), I(0.07)))
        tile(p, g, cat, I(x + 0.36), I(y + 0.4), I(0.42))
        text(p, (x + 0.68, y + 0.25, cw - 0.75, 0.3), cat_label, 14, True, col)
        text(p, (x + 0.16, y + 0.66, cw - 0.3, 0.44), big, 24, True, NAVY); fits(big, 24, True, cw - 0.3)
        text(p, (x + 0.16, y + 1.1, cw - 0.3, 0.5), desc, 14, False, INK, Qt.AlignLeft | Qt.AlignTop, wrap=True)
        fits(desc, 14, False, cw - 0.3, lines=2)


def badge(p, x, y, n):
    p.setPen(QPen(QColor("white"), 4)); p.setBrush(NAVY); p.drawEllipse(QPointF(I(x), I(y)), I(0.13), I(0.13))
    text(p, (x - 0.13, y - 0.13, 0.26, 0.26), str(n), 12, True, QColor("white"), Qt.AlignCenter)


def timeline(p, x0=0.45, x1=12.9, y0=5.32, h=1.56):
    V4.rect(p, x0, y0, x1 - x0, h, NAVY, "#F5F7FB", 200, 5)
    V4.tag(p, x0, y0, NAVY, "clock", "Mission timeline")
    text(p, (x1 - 4.6, y0, 4.5, 0.3), "Take-off to measured 3D: about 20 min", 14, True, NAVY, Qt.AlignRight | Qt.AlignVCenter)
    fits("Take-off to measured 3D: about 20 min", 14, True, 4.5)
    tw = 1.2; th = tw / 1.6
    lw, aw = 1.45, 0.28
    step_w = tw + 0.1 + lw
    total = len(STEPS) * step_w + (len(STEPS) - 1) * aw
    sx = x0 + (x1 - x0 - total) / 2
    ty = y0 + 0.44
    for k, (path, title, sub) in enumerate(STEPS):
        x = sx + k * (step_w + aw)
        pic(p, path, x, ty, tw, th, border=NAVY)
        badge(p, x + 0.02, ty + 0.02, k + 1)
        text(p, (x + tw + 0.1, ty + 0.1, lw, 0.26), title, 14, True, INK); fits(title, 14, True, lw)
        text(p, (x + tw + 0.1, ty + 0.38, lw, 0.26), sub, 14, False, INK); fits(sub, 14, False, lw)
        if k < len(STEPS) - 1:
            ax = x + step_w
            arrow(p, [(I(ax + 0.04), I(ty + th / 2)), (I(ax + aw - 0.04), I(ty + th / 2))], NAVY, 8, 24)
    by, bh = y0 + h - 0.3, 0.2
    bx0, bx1 = sx, sx + total
    fl = step_w + aw / 2                     # flight = step 1; processing = steps 2-4 (live preview comes during processing)
    p.setPen(Qt.NoPen); p.setBrush(QColor("#44546F")); p.drawRoundedRect(QRectF(I(bx0), I(by), I(fl - 0.03), I(bh)), I(0.1), I(0.1))
    p.setBrush(ORANGE); p.drawRoundedRect(QRectF(I(bx0 + fl + 0.03), I(by), I(bx1 - bx0 - fl - 0.03), I(bh)), I(0.1), I(0.1))
    text(p, (bx0, by - 0.02, fl, bh + 0.04), "Flight  11.4 min", 14, True, QColor("white"), Qt.AlignCenter)
    text(p, (bx0 + fl, by - 0.02, bx1 - bx0 - fl, bh + 0.04), "Processing  6.9 min  ·  one laptop", 14, True, QColor("white"), Qt.AlignCenter)


def draw(p):
    K.chrome(p, "IMPACT AND BENEFITS")
    text(p, (0.45, 1.16, 6.6, 0.34), "Potential impact on the target audience", 16, True, BODY, Qt.AlignLeft | Qt.AlignTop)
    who(p)
    today_vs(p)
    text(p, (7.25, 1.16, 5.65, 0.62), "Benefits of the solution (social, economic, environmental, etc.)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    kpis(p)
    timeline(p)


if __name__ == "__main__":
    K.export(draw, "SLIDE5")
    print("\n".join(sorted(set(_warn))) or "all text fits")
