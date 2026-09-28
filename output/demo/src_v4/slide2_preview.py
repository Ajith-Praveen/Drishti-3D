"""Full-slide mock-up of slide 2 on the SIH template layout (13.33 x 7.5 in at 144 px/in: 1 pt = 2 px)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/tmp/demo4")
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen

app = QGuiApplication(sys.argv[:1])
import arch  # noqa: E402
from arch import dtext  # noqa: E402

S = 144                           # px per inch
PT = 2                            # px per point
BODY = QColor("#1F497D")
OUT = "/Users/ajith/sih/output/slides/slide2"
TITLE = "DRISHTI-3D: One Flight → Measured 3D"
INTRO = "DRISHTI-3D is an offline desktop app that turns one pass of drone video + GPS log into a georeferenced, measurable 3D model."
ROWS = [
    ("Detailed explanation of the proposed solution",
     ["Video + GPS log → georeferenced, measurable 3D model",
      "GPU stereo measures every view, fused in true 3D",
      "Offline desktop app with measurement tools"], "s2_A_flow"),
    ("How it addresses the problem",
     ["Meets every target in the problem statement's output table",
      "Proven end to end on 2 real drone flights"], "s2_B_targets"),
    ("Innovation and uniqueness of the solution",
     ["Our own reconstruction engine, not a wrapper",
      "Every surface tagged: measured or inferred",
      "Live 3D preview: spot gaps, re-fly on site"], "s2_C_unique"),
]
FOOT = ("*0.29 m: median horizontal agreement with Spain's IGN orthophoto (flight01, 13 features) · "
        "COLMAP 33 min sparse vs DRISHTI 9.5 min dense, same Mac, COLMAP without CUDA.")


def font(family, pt, bold=False):
    f = QFont(family); f.setPixelSize(int(pt * PT)); f.setBold(bold); return f


def main():
    img = QImage(int(13.333 * S), int(7.5 * S), QImage.Format_ARGB32_Premultiplied); img.fill(QColor("white"))
    p = QPainter(img); p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
    # template chrome
    logo = QImage("/tmp/tpl/image2_s.png")
    p.drawImage(QRectF(10.7 * S, 0.0, 2.46 * S, 1.16 * S), logo)
    p.setPen(Qt.NoPen); p.setBrush(QColor("#8064A2")); p.drawEllipse(QRectF(0.36 * S, 0.28 * S, 1.37 * S, 0.88 * S))
    p.setFont(font("Arial", 14, True)); p.setPen(QColor("white"))
    dtext(p, QRectF(0.36 * S, 0.28 * S, 1.37 * S, 0.88 * S), Qt.AlignCenter, "Robos.Inc")
    p.setFont(font("Times New Roman", 32, True)); p.setPen(QColor("black"))
    dtext(p, QRectF(1.9 * S, 0.1 * S, 8.7 * S, 1.05 * S), Qt.AlignCenter, TITLE)
    p.setPen(Qt.NoPen); p.setBrush(QColor("#0070C0")); p.drawRect(QRectF(0, 6.95 * S, 13.333 * S, 0.55 * S))
    p.setFont(font("Arial", 12)); p.setPen(QColor("white"))
    dtext(p, QRectF(5.08 * S, 6.97 * S, 3.5 * S, 0.45 * S), Qt.AlignCenter, "@SIH Idea submission- Template")
    dtext(p, QRectF(12.4 * S, 6.97 * S, 0.6 * S, 0.45 * S), Qt.AlignCenter, "2")
    # the pointer heading
    p.setFont(font("Arial", 16, True)); p.setPen(BODY)
    dtext(p, QRectF(0.45 * S, 1.16 * S, 12.4 * S, 0.36 * S), Qt.AlignLeft | Qt.AlignVCenter, "Proposed Solution (Describe your Idea/Solution/Prototype)")
    p.setFont(font("Arial", 14)); p.setPen(BODY)
    dtext(p, QRectF(0.45 * S, 1.5 * S, 12.45 * S, 0.3 * S), Qt.AlignLeft | Qt.AlignVCenter, INTRO)
    # three rows: text left, diagram right
    y0, rh = 1.84, 1.69
    for k, (head, bullets, fig) in enumerate(ROWS):
        y = (y0 + k * rh) * S
        if k:
            p.setPen(QPen(QColor("#D5DEEA"), 2)); p.drawLine(0.45 * S, y - 0.04 * S, 12.9 * S, y - 0.04 * S)
        p.setFont(font("Arial", 16, True)); p.setPen(BODY)
        dtext(p, QRectF(0.45 * S, y + 0.06 * S, 6.2 * S, 0.34 * S), Qt.AlignLeft | Qt.AlignVCenter, head)
        yy = y + 0.46 * S
        p.setFont(font("Arial", 14))
        for b in bullets:
            dtext(p, QRectF(0.5 * S, yy, 0.25 * S, 0.3 * S), Qt.AlignLeft | Qt.AlignTop, "•")
            r = QRectF(0.75 * S, yy, 5.55 * S, 0.62 * S)
            dtext(p, r, arch._flag(Qt.AlignLeft) | arch._flag(Qt.AlignTop) | arch._flag(Qt.TextWordWrap), b)
            from PySide6.QtGui import QFontMetricsF
            fm = QFontMetricsF(p.font()); lines, cur = 1, ""
            for w_ in b.split():
                trial = (cur + " " + w_).strip()
                if fm.horizontalAdvance(trial) > r.width() and cur:
                    lines += 1; cur = w_
                else:
                    cur = trial
            yy += (fm.height() * lines) + 0.08 * S
        d = QImage(f"{OUT}/{fig}.png")
        p.drawImage(QRectF(6.8 * S, y + 0.02 * S, 6.1 * S, 6.1 * S * d.height() / d.width()), d)
    p.setFont(font("Arial", 9.5)); p.setPen(QColor("#1B2A44"))
    dtext(p, QRectF(0.45 * S, 6.5 * S, 6.1 * S, 0.44 * S), arch._flag(Qt.AlignLeft) | arch._flag(Qt.AlignBottom) | arch._flag(Qt.TextWordWrap), FOOT)
    p.end()
    img.save(f"{OUT}/s2_preview.png")
    print("ok")


main()
