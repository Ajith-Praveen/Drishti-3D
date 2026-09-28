"""Slide 6 RESEARCH AND REFERENCES: research-lineage timeline (1996 -> 2026), datasets, our evidence, project links.
Every paper / dataset title carries its verified link (DOI or arXiv, checked 2026-09-27). With NATIVE_LINKS the
image leaves the linked titles out and build_s6_pptx.py puts them back as real, clickable PowerPoint text.
Set GITHUB_URL / AWS_URL / DEMO_URL when known and re-run. 14 pt text, 16 pt headings."""
import sys

sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
from slidekit import BODY, I, Qt, QColor, QImage, QPen, QPointF, QRectF, text  # noqa: E402
import arch  # noqa: E402
from arch import ORANGE, tile  # noqa: E402
import slide3_v4 as V4  # noqa: E402   rect(), tag()
from PySide6.QtGui import QFont, QFontMetricsF  # noqa: E402

GITHUB_URL = None          # e.g. "github.com/<org>/drishti3d"
AWS_URL = None             # e.g. "drishti3d.<region>.amazonaws.com"
DEMO_URL = None            # e.g. a YouTube / Drive link to the 3-min demo video
PAPER_URLS = {             # verified: Crossref (DOIs) and the arXiv API
    "Plane sweep": "https://doi.org/10.1109/CVPR.1996.517097",
    "TSDF fusion": "https://doi.org/10.1145/237170.237269",
    "Bundle adjustment": "https://doi.org/10.1007/3-540-44480-7_21",
    "SGM stereo": "https://doi.org/10.1109/TPAMI.2007.1166",
    "BA in the large": "https://doi.org/10.1007/978-3-642-15552-9_3",
    "COLMAP SfM": "https://doi.org/10.1109/CVPR.2016.445",
    "Open3D": "https://arxiv.org/abs/1801.09847",
    "DISK features": "https://arxiv.org/abs/2006.13566",
    "SegFormer": "https://arxiv.org/abs/2105.15203",
    "LightGlue": "https://arxiv.org/abs/2306.13643",
    "MapAnything": "https://arxiv.org/abs/2509.13414",
}
DATA_URLS = {"PinPoint flight01": "https://doi.org/10.5281/zenodo.22671839", "IGN PNOA ortho + DEM": "https://pnoa.ign.es/"}
NATIVE_LINKS = False       # True: linked titles are left out of the image and recorded in LINK_TEXTS for the .pptx
LINK_TEXTS = []            # (x, y, w, h) in inches, text, bold, "RRGGBB", "ctr" | "l", url


def full_url(u):
    return None if not u else (u if u.startswith(("http://", "https://")) else "https://" + u)


def evidence_urls():
    repo = full_url(GITHUB_URL)
    return {"flight01 benchmark report": repo and repo.rstrip("/") + "/blob/main/evidence/flight01-benchmark.md",
            "739 automated tests": repo and repo.rstrip("/") + "/tree/main/tests",
            "3-min demo video": full_url(DEMO_URL)}

INK = QColor("#0B1F44")
NAVY = QColor("#1F497D")
PURPLE = QColor("#8C4FFF")
TEAL = arch.TEAL
GREEN = QColor("#2E9E5B")
GREY = QColor("#7C8799")
arch.TILE.update({"navy": NAVY, "learned": PURPLE, "grey": GREY, "fix": GREEN})
LOGO = "/tmp/demo/assets/logo.png"

# year, glyph, category, method, authors, venue
MILESTONES = [
    ("1996", "planes", "navy", "Plane sweep", "Collins", "CVPR"),
    ("1996", "cube", "navy", "TSDF fusion", "Curless & Levoy", "SIGGRAPH"),
    ("2000", "target", "navy", "Bundle adjustment", "Triggs et al.", "Vision Algorithms"),
    ("2008", "grid", "navy", "SGM stereo", "Hirschmüller", "IEEE TPAMI"),
    ("2010", "sigma", "navy", "BA in the large", "Agarwal et al.", "ECCV"),
    ("2016", "graph", "grey", "COLMAP SfM", "Schönberger, Frahm", "CVPR · baseline"),
    ("2018", "package", "navy", "Open3D", "Zhou, Park, Koltun", "arXiv"),
    ("2020", "eye", "learned", "DISK features", "Tyszkiewicz et al.", "NeurIPS"),
    ("2021", "mask", "learned", "SegFormer", "Xie et al.", "NeurIPS"),
    ("2023", "bolt", "learned", "LightGlue", "Lindenberger et al.", "ICCV"),
    ("2025", "brain", "learned", "MapAnything", "Keetha et al.", "arXiv"),
    ("2026", None, "ours", "DRISHTI-3D", "Robos.Inc", "this work"),
]
DATASETS = [("video", "navy", "PinPoint flight01", "100 surveyed ground points, Spain"),
            ("globe", "navy", "IGN PNOA ortho + DEM", "Spain national reference, CC-BY 4.0"),
            ("gps", "navy", "DJI_1001 flight", "11.4-min video + Airdata GPS log")]
EVIDENCE = [("chart", "ours", "flight01 benchmark report", "evidence/flight01-benchmark.md"),
            ("check", "ours", "739 automated tests", "pytest, all passing"),
            ("video", "ours", "3-min demo video", "real app, real flight")]
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


def width_in(s, pt, bold):
    f = QFont("Arial"); f.setPixelSize(int(round(pt * K.PT))); f.setBold(bold)
    return QFontMetricsF(f).horizontalAdvance(s) / K.S


LINK = QColor("#0563C1")


def link_text(p, rect, s_, bold, color, align, url):
    """Linked title: native PowerPoint hyperlink (NATIVE_LINKS) or drawn in link colour with an underline."""
    x, y, w, h = rect
    if url and NATIVE_LINKS:
        LINK_TEXTS.append({"rect": rect, "text": s_, "bold": bold, "rgb": color.name()[1:].upper(), "align": align, "url": url})
        return
    text(p, rect, s_, 14, bold, color, (Qt.AlignHCenter if align == "ctr" else Qt.AlignLeft) | Qt.AlignVCenter)
    if url:
        f = QFont("Arial"); f.setPixelSize(int(round(14 * K.PT))); f.setBold(bold); fm = QFontMetricsF(f)
        tw = fm.horizontalAdvance(s_)
        x0 = I(x) + (I(w) - tw) / 2 if align == "ctr" else I(x)
        uy = I(y) + (I(h) - fm.height()) / 2 + fm.ascent() + max(2.0, fm.underlinePos())
        p.setPen(QPen(color, max(3.0, fm.lineWidth()))); p.drawLine(QPointF(x0, uy), QPointF(x0 + tw, uy))


def timeline(p, x0=0.45, x1=12.9, y0=1.56, h=3.08):
    V4.rect(p, x0, y0, x1 - x0, h, NAVY, "#F5F7FB", 200, 5)
    V4.tag(p, x0, y0, NAVY, "graph", "Research foundations")
    # legend, right of the header
    items = [(NAVY, "Classic geometry"), (PURPLE, "Learned models"), (GREY, "Baseline"), (ORANGE, "Ours")]
    total = sum(0.22 + width_in(t, 14, True) + 0.22 for _, t in items)
    lx = x1 - 0.12 - total
    for col, t in items:
        p.setPen(Qt.NoPen); p.setBrush(col); p.drawRoundedRect(QRectF(I(lx), I(y0 + 0.09), I(0.14), I(0.14)), I(0.03), I(0.03))
        text(p, (lx + 0.2, y0, 2.0, 0.3), t, 14, True, INK)
        lx += 0.22 + width_in(t, 14, True) + 0.22
    n = len(MILESTONES)
    xa, xb = x0 + 0.88, x1 - 0.88
    step = (xb - xa) / (n - 1)
    bw = 2 * step - 0.06                       # text width: same-side neighbours are two steps apart
    axis = y0 + 1.68
    p.setPen(QPen(NAVY, 7, Qt.SolidLine, Qt.RoundCap)); p.drawLine(QPointF(I(x0 + 0.25), I(axis)), QPointF(I(x1 - 0.25), I(axis)))
    for k, (year, g, cat, method, authors, venue) in enumerate(MILESTONES):
        cx = xa + k * step
        above = k % 2 == 0
        ours = cat == "ours"
        col = ORANGE if ours else arch.TILE[cat]
        # connector + year pill on the axis
        p.setPen(QPen(col, 5))
        if above:
            p.drawLine(QPointF(I(cx), I(axis - 0.14)), QPointF(I(cx), I(axis - 0.24)))
        else:
            p.drawLine(QPointF(I(cx), I(axis + 0.14)), QPointF(I(cx), I(axis + 0.24)))
        pw = 0.66
        p.setPen(QPen(QColor("white"), 4)); p.setBrush(col)
        p.drawRoundedRect(QRectF(I(cx - pw / 2), I(axis - 0.15), I(pw), I(0.3)), I(0.15), I(0.15))
        text(p, (cx - pw / 2, axis - 0.15, pw, 0.3), year, 14, True, QColor("white"), Qt.AlignCenter)
        # tile + 3 lines, stacked away from the axis
        if above:
            ty = axis - 0.24 - 0.68 - 0.36          # tile top; text ends just above the connector
            ly = ty + 0.39
        else:
            ty = axis + 0.26
            ly = ty + 0.39
        if ours:
            p.drawImage(QRectF(I(cx - 0.19), I(ty), I(0.38), I(0.38)), QImage(LOGO))
        else:
            tile(p, g, cat, I(cx), I(ty + 0.18), I(0.36))
        url = full_url(GITHUB_URL) if ours else PAPER_URLS.get(method)
        tcol = ORANGE.darker(135) if ours else (LINK if url else INK)
        link_text(p, (cx - bw / 2, ly, bw, 0.23), method, True, tcol, "ctr", url)
        text(p, (cx - bw / 2, ly + 0.22, bw, 0.23), authors, 14, False, INK, Qt.AlignHCenter | Qt.AlignVCenter)
        text(p, (cx - bw / 2, ly + 0.44, bw, 0.23), venue, 14, False, INK, Qt.AlignHCenter | Qt.AlignVCenter)
        for s_, b_ in ((method, True), (authors, False), (venue, False)):
            fits(s_, 14, b_, bw)


def listing(p, x0, y0, w, h, col, glyph_name, title, rows, title_col=None, urls=None):
    V4.rect(p, x0, y0, w, h, col, "#FFFFFF", 0, 5)
    V4.tag(p, x0, y0, col, glyph_name, title, title_col)
    fits(title, 14, True, w - 0.45)
    for k, (g, cat, t1, t2) in enumerate(rows):
        y = y0 + 0.43 + k * 0.55
        tile(p, g, cat, I(x0 + 0.3), I(y + 0.24), I(0.36))
        url = (urls or {}).get(t1)
        link_text(p, (x0 + 0.58, y, w - 0.66, 0.24), t1, True, LINK if url else INK, "l", url); fits(t1, 14, True, w - 0.66)
        text(p, (x0 + 0.58, y + 0.23, w - 0.66, 0.24), t2, 14, False, INK); fits(t2, 14, False, w - 0.66)


def wrap_url(u, width):
    lines, cur = [], ""
    for ch in u:
        if cur and width_in(cur + ch, 14, False) > width:
            cut = max(cur.rfind(c) for c in "/.-?&=_")
            if cut > 0:
                lines.append(cur[:cut + 1]); cur = cur[cut + 1:] + ch
            else:
                lines.append(cur); cur = ch
        else:
            cur += ch
    return lines + [cur]


def link_card(p, x, y, w, h, g, title, url):
    V4.rect(p, x, y, w, h, QColor("#8FCBA3"), "#F1FAF4", 255, 3)
    tile(p, g, "fix", I(x + 0.32), I(y + h / 2), I(0.4))
    tw = w - 0.72
    text(p, (x + 0.64, y + 0.05, tw, 0.26), title, 14, True, INK); fits(title, 14, True, tw)
    if not url:
        text(p, (x + 0.64, y + 0.31, tw, 0.26), "link to be added", 14, False, NAVY)
        return
    import re
    shown = re.sub(r"^https?://", "", url).rstrip("/")
    lines = wrap_url(shown, tw)
    if len(lines) > 2:
        _warn.append(f"url needs {len(lines)} lines: {url}")
    for k, ln in enumerate(lines[:2]):
        link_text(p, (x + 0.64, y + 0.3 + k * 0.23, tw, 0.26), ln, False, LINK, "l", full_url(url))


def links(p, x0, y0, w, h):
    V4.rect(p, x0, y0, w, h, GREEN, "#FFFFFF", 0, 5)
    V4.tag(p, x0, y0, GREEN, "branch", "Project links", GREEN.darker(150))
    ch = (h - 0.42 - 0.06 - 0.06) / 2
    link_card(p, x0 + 0.08, y0 + 0.42, w - 0.16, ch, "branch", "GitHub repository", GITHUB_URL)
    link_card(p, x0 + 0.08, y0 + 0.42 + ch + 0.06, w - 0.16, ch, "app", "Live prototype (AWS)", AWS_URL)


def draw(p):
    LINK_TEXTS.clear()
    K.chrome(p, "RESEARCH AND REFERENCES")
    text(p, (0.45, 1.16, 12.45, 0.34), "Details / Links of the reference and research work", 16, True, BODY, Qt.AlignLeft | Qt.AlignTop)
    text(p, (6.9, 1.16, 6.0, 0.34), "Click a blue title to open the paper or dataset", 14, False, LINK, Qt.AlignRight | Qt.AlignTop)
    timeline(p)
    by, bh = 4.74, 2.14
    listing(p, 0.45, by, 4.1, bh, TEAL, "globe", "Datasets & ground truth", DATASETS, urls=DATA_URLS)
    listing(p, 4.65, by, 4.1, bh, ORANGE, "chart", "Our evidence", EVIDENCE, ORANGE.darker(135), urls=evidence_urls())
    links(p, 8.85, by, 4.05, bh)


if __name__ == "__main__":
    K.export(draw, "SLIDE6")
    print("\n".join(sorted(set(_warn))) or "all text fits")
