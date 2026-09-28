"""Slide 5 IMPACT AND BENEFITS: applications hub (the problem statement's 8), before -> after (its own background),
4 benefit cards, a real result. All text 14 pt (headings 16 pt), dark; each piece exported on its own."""
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
GREY = QColor("#6B7A90")
arch.TILE["fix"] = GREEN
arch.TILE["navy"] = QColor("#1F497D")
LOGO = "/tmp/demo/assets/logo.png"
RESULT = "/tmp/demo/assets/fly/0150.jpg"

APPS = [("globe", "input", "Border mapping", "strategic areas"), ("bolt", "ours", "Disaster relief", "damage mapping"),
        ("grid", "app", "Urban planning", "smart cities"), ("eye", "ext", "Infrastructure", "inspection"),
        ("package", "app", "Construction", "progress, volumes"), ("files", "runtime", "Archaeology", "site records"),
        ("cube", "ext", "Digital twins", "measured 3D"), ("target", "navy", "Military recon", "mission planning")]
BEFORE_AFTER = [("Multiple passes", "1 pass"), ("Planned flights", "Normal flight"),
                ("Heavy overlap", "Ordinary video"), ("Long processing", "~7 min, laptop")]
BENEFITS = [("user", "app", "Social", "Faster rescue and relief decisions; no crews on unsafe ground"),
            ("chart", "ours", "Economic", "One laptop, open-source: no licence fees, no cloud GPU, fewer flight hours"),
            ("globe", "fix", "Environmental", "One pass instead of repeat flights: less battery, fuel and airspace time"),
            ("shield", "navy", "Strategic", "Works offline: sensitive imagery never leaves the device")]
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


def hub(p, x0=0.45, x1=7.05, y0=1.8, pitch=0.75):
    """DRISHTI-3D in the centre, 4 applications on each side (labels on the outer side, so no spoke crosses text)."""
    cx = (x0 + x1) / 2
    lab_w = 1.6
    lx, rx = x0 + lab_w + 0.3, x1 - lab_w - 0.3          # tile centres, left / right column
    rows = [y0 + k * pitch for k in range(4)]
    cy = (rows[0] + rows[-1]) / 2
    card_w, card_h = 1.7, 1.25
    for k in range(4):
        for tx, edge in ((lx, cx - card_w / 2), (rx, cx + card_w / 2)):
            p.setPen(QPen(QColor("#9AA8BD"), 5, Qt.SolidLine, Qt.RoundCap))
            sx = tx + (0.24 if tx < cx else -0.24)
            p.drawLine(QPointF(I(sx), I(rows[k])), QPointF(I(edge), I(cy - 0.38 + 0.25 * k)))
    V4.rect(p, cx - card_w / 2, cy - card_h / 2, card_w, card_h, ORANGE, "#FFF3E8", 255, 6)
    p.drawImage(QRectF(I(cx - 0.27), I(cy - card_h / 2 + 0.08), I(0.54), I(0.54)), QImage(LOGO))
    text(p, (cx - card_w / 2, cy + 0.06, card_w, 0.28), "DRISHTI-3D", 16, True, ORANGE.darker(135), Qt.AlignHCenter | Qt.AlignVCenter)
    text(p, (cx - card_w / 2, cy + 0.32, card_w, 0.26), "one pass → 3D", 14, False, INK, Qt.AlignHCenter | Qt.AlignVCenter)
    fits("DRISHTI-3D", 16, True, card_w - 0.1); fits("one pass → 3D", 14, False, card_w - 0.1)
    for k, (g, cat, title, sub) in enumerate(APPS):
        side, r = divmod(k, 4)
        y = rows[r]
        if side == 0:
            tile(p, g, cat, I(lx), I(y), I(0.42))
            box, al = (x0, y - 0.25, lab_w + 0.05, 0.25), Qt.AlignRight | Qt.AlignVCenter
            box2 = (x0, y, lab_w + 0.05, 0.25)
        else:
            tile(p, g, cat, I(rx), I(y), I(0.42))
            box, al = (rx + 0.3, y - 0.25, lab_w, 0.25), Qt.AlignLeft | Qt.AlignVCenter
            box2 = (rx + 0.3, y, lab_w, 0.25)
        text(p, box, title, 14, True, INK, al); text(p, box2, sub, 14, False, INK, al)
        fits(title, 14, True, lab_w); fits(sub, 14, False, lab_w)


def awareness(p, x0=0.45, x1=7.05, y0=6.2):
    V4.rect(p, x0, y0, x1 - x0, 0.66, GREEN, "#F1FAF4", 170, 4)
    tile(p, "eye", "fix", I(x0 + 0.3), I(y0 + 0.33), I(0.38))
    t1 = "Near real-time situational awareness"
    t2 = "The 3D model builds live on screen: re-fly on site if coverage is poor"
    text(p, (x0 + 0.6, y0 + 0.07, x1 - x0 - 0.7, 0.26), t1, 14, True, INK)
    text(p, (x0 + 0.6, y0 + 0.33, x1 - x0 - 0.7, 0.26), t2, 14, False, INK)
    fits(t1, 14, True, x1 - x0 - 0.7); fits(t2, 14, False, x1 - x0 - 0.7)


def before_after(p, x0=0.45, x1=7.05, y0=4.62):
    text(p, (x0, y0, x1 - x0, 0.26), "Today (per the problem statement)  →  with DRISHTI-3D", 14, True, INK)
    fits("Today (per the problem statement)  →  with DRISHTI-3D", 14, True, x1 - x0)
    n = len(BEFORE_AFTER); cw = (x1 - x0) / n
    for k, (b, a) in enumerate(BEFORE_AFTER):
        x = x0 + k * cw
        pb = QRectF(I(x + 0.04), I(y0 + 0.33), I(cw - 0.08), I(0.36))
        p.setPen(QPen(GREY, 4)); p.setBrush(QColor("#EEF1F5")); p.drawRoundedRect(pb, I(0.18), I(0.18))
        text(p, (x + 0.04, y0 + 0.33, cw - 0.08, 0.36), b, 14, False, INK, Qt.AlignCenter)
        arrow(p, [(I(x + cw / 2), I(y0 + 0.71)), (I(x + cw / 2), I(y0 + 0.92))], DARK, 6, 18)
        pa = QRectF(I(x + 0.04), I(y0 + 0.94), I(cw - 0.08), I(0.36))
        p.setPen(Qt.NoPen); p.setBrush(GREEN); p.drawRoundedRect(pa, I(0.18), I(0.18))
        text(p, (x + 0.04, y0 + 0.94, cw - 0.08, 0.36), a, 14, True, QColor("white"), Qt.AlignCenter)
        fits(b, 14, False, cw - 0.14); fits(a, 14, True, cw - 0.14)


def benefits(p, x0=7.3, x1=12.9, y0=1.86, gap=0.12, ch=1.3):
    cw = (x1 - x0 - gap) / 2
    for k, (g, cat, title, body) in enumerate(BENEFITS):
        r, c = divmod(k, 2)
        x, y = x0 + c * (cw + gap), y0 + r * (ch + gap)
        col = arch.TILE[cat]
        V4.rect(p, x, y, cw, ch, col, "#FFFFFF", 0, 4)
        tile(p, g, cat, I(x + 0.3), I(y + 0.3), I(0.4))
        text(p, (x + 0.6, y + 0.14, cw - 0.7, 0.32), title, 16, True, col.darker(125) if cat != "navy" else INK)
        text(p, (x + 0.12, y + 0.56, cw - 0.24, 0.72), body, 14, False, INK, Qt.AlignLeft | Qt.AlignTop, wrap=True)
        fits(body, 14, False, cw - 0.24, lines=3)


def result(p, x0=7.3, x1=12.9, y0=4.62, y1=6.86):
    w, h = x1 - x0, y1 - y0
    src = QImage(RESULT)
    ch = int(src.width() * h / w); top = int((src.height() - ch) * 0.75)       # skip the empty sky
    img = src.copy(0, top, src.width(), ch).scaled(int(I(w) * 2), int(I(h) * 2), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    clip = QPainterPath(); clip.addRoundedRect(QRectF(I(x0), I(y0), I(w), I(h)), I(0.06), I(0.06))
    p.save(); p.setClipPath(clip); p.drawImage(QRectF(I(x0), I(y0), I(w), I(h)), img); p.restore()
    p.setPen(QPen(GREEN, 6)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(I(x0), I(y0), I(w), I(h)), I(0.06), I(0.06))
    cap = "Real result: 11-min flight → 8 M-point 3D model"
    tw = w - 0.4
    p.setPen(Qt.NoPen); p.setBrush(GREEN); p.drawRoundedRect(QRectF(I(x0 + 0.2), I(y1 - 0.44), I(tw), I(0.3)), I(0.06), I(0.06))
    text(p, (x0 + 0.2, y1 - 0.44, tw, 0.3), cap, 14, True, QColor("white"), Qt.AlignCenter)
    fits(cap, 14, True, tw - 0.1)


def draw(p):
    K.chrome(p, "IMPACT AND BENEFITS")
    text(p, (0.45, 1.16, 6.6, 0.34), "Potential impact on the target audience", 16, True, BODY)
    hub(p)
    before_after(p)
    awareness(p)
    text(p, (7.3, 1.16, 5.6, 0.66), "Benefits of the solution (social, economic, environmental, etc.)", 16, True, BODY,
         Qt.AlignLeft | Qt.AlignTop, wrap=True)
    benefits(p)
    result(p)


if __name__ == "__main__":
    K.export(draw, "SLIDE5")
    print("\n".join(sorted(set(_warn))) or "all text fits")
