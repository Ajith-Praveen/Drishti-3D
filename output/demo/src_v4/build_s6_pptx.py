"""SLIDE6_with_links.pptx: the SIH template's slide 6 with the slide-6 graphic placed full-slide and every linked
title as real PowerPoint text carrying its hyperlink (clickable in slide show and in the exported PDF).
Template title and team oval stay native (oval text set to the team name)."""
import html
import os
import re
import sys
import zipfile
from xml.dom import minidom

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/tmp/demo4")
import slidekit as K  # noqa: E402
import slide6 as S  # noqa: E402
from PySide6.QtCore import QRect, QSize, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgGenerator  # noqa: E402

TEMPLATE = "/Users/ajith/Downloads/SIH2026-IDEA-Presentation-Format.pptx"
OUT = "/Users/ajith/sih/output/slides/slide6/SLIDE6_with_links.pptx"
TEAM = "Robos.Inc"
EMU = 914400
KEEP_SLIDE, KEEP_NOTES, KEEP_RID, KEEP_SLDID = 6, 5, "rId7", "296"
REL_IMG = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
REL_LINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"


def hints(p):
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)


def render_graphic():
    S.NATIVE_LINKS = True
    gen = QSvgGenerator(); buf_svg = "/tmp/demo4/_s6_native.svg"
    gen.setFileName(buf_svg); gen.setSize(QSize(K.W, K.H)); gen.setViewBox(QRect(0, 0, K.W, K.H)); gen.setResolution(300)
    p = QPainter(gen); hints(p); S.draw(p); p.end()
    img = QImage(K.W, K.H, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); hints(p); S.draw(p); p.end()
    img.save("/tmp/demo4/_s6_native.png")
    links = [dict(t) for t in S.LINK_TEXTS]
    S.NATIVE_LINKS = False
    return open("/tmp/demo4/_s6_native.png", "rb").read(), open(buf_svg, "rb").read(), links


def emu(v):
    return int(round(v * EMU))


def link_box(shape_id, rid, t):
    x, y, w, h = t["rect"]
    esc = html.escape(t["text"], quote=False)
    algn = "ctr" if t["align"] == "ctr" else "l"
    b = ' b="1"' if t["bold"] else ""
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Link: {html.escape(t["text"])}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
        '<p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr" rtlCol="0"><a:noAutofit/></a:bodyPr>'
        f'<a:lstStyle/><a:p><a:pPr algn="{algn}"/><a:r><a:rPr lang="en-US" sz="1400"{b} u="sng" dirty="0">'
        f'<a:solidFill><a:srgbClr val="{t["rgb"]}"/></a:solidFill><a:latin typeface="Arial"/><a:cs typeface="Arial"/>'
        f'<a:hlinkClick r:id="{rid}"><a:extLst><a:ext uri="{{A12FA001-AC4F-418D-AE19-62706E023703}}">'
        '<ahyp:hlinkClr xmlns:ahyp="http://schemas.microsoft.com/office/drawing/2018/hyperlinkcolor" val="tx"/>'
        f'</a:ext></a:extLst></a:hlinkClick></a:rPr><a:t>{esc}</a:t></a:r></a:p></p:txBody></p:sp>'
    )


def picture(shape_id, rid_png, rid_svg):
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="Slide 6 graphic" descr="Research foundations timeline, datasets, '
        'evidence and project links"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rid_png}"><a:extLst><a:ext uri="{{96DAC541-7B7A-43D3-8B79-37D633B846F1}}">'
        f'<asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" r:embed="{rid_svg}"/>'
        '</a:ext></a:extLst></a:blip><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        '<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def build():
    png, svg, links = render_graphic()
    src = zipfile.ZipFile(TEMPLATE)
    drop_slides = [i for i in range(1, 8) if i != KEEP_SLIDE]
    drop_notes = [i for i in range(1, 7) if i != KEEP_NOTES]
    drop = {f"ppt/slides/slide{i}.xml" for i in drop_slides} | {f"ppt/slides/_rels/slide{i}.xml.rels" for i in drop_slides}
    drop |= {f"ppt/notesSlides/notesSlide{i}.xml" for i in drop_notes} | {f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels" for i in drop_notes}
    parts = {}
    for n in src.namelist():
        if n not in drop:
            parts[n] = src.read(n)

    # the kept slide: drop the pointer text box, team name in the oval, add the graphic and the link texts
    sl = parts[f"ppt/slides/slide{KEEP_SLIDE}.xml"].decode("utf-8")
    sl, n_tb = re.subn(r'<p:sp><p:nvSpPr><p:cNvPr id="\d+" name="TextBox 8"/>.*?</p:sp>', "", sl, count=1, flags=re.S)
    sl, n_team = re.subn(r"<a:t>Your Team Name</a:t>", f"<a:t>{TEAM}</a:t>", sl, count=1)
    assert n_tb == 1 and n_team == 1, (n_tb, n_team)
    rels = parts[f"ppt/slides/_rels/slide{KEEP_SLIDE}.xml.rels"].decode("utf-8")
    used = [int(m) for m in re.findall(r'Id="rId(\d+)"', rels)]
    nxt = max(used) + 1
    new_rels, shapes = [], []
    rid_png, rid_svg = f"rId{nxt}", f"rId{nxt + 1}"; nxt += 2
    new_rels += [f'<Relationship Id="{rid_png}" Type="{REL_IMG}" Target="../media/drishti_slide6.png"/>',
                 f'<Relationship Id="{rid_svg}" Type="{REL_IMG}" Target="../media/drishti_slide6.svg"/>']
    shapes.append(picture(900, rid_png, rid_svg))
    for k, t in enumerate(links):
        rid = f"rId{nxt}"; nxt += 1
        new_rels.append(f'<Relationship Id="{rid}" Type="{REL_LINK}" Target="{html.escape(t["url"])}" TargetMode="External"/>')
        shapes.append(link_box(901 + k, rid, t))
    sl = sl.replace("</p:spTree>", "".join(shapes) + "</p:spTree>", 1)
    rels = rels.replace("</Relationships>", "".join(new_rels) + "</Relationships>", 1)
    parts[f"ppt/slides/slide{KEEP_SLIDE}.xml"] = sl.encode("utf-8")
    parts[f"ppt/slides/_rels/slide{KEEP_SLIDE}.xml.rels"] = rels.encode("utf-8")
    parts["ppt/media/drishti_slide6.png"] = png
    parts["ppt/media/drishti_slide6.svg"] = svg

    pres = parts["ppt/presentation.xml"].decode("utf-8")
    pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", f'<p:sldIdLst><p:sldId id="{KEEP_SLDID}" r:id="{KEEP_RID}"/></p:sldIdLst>', pres, flags=re.S)
    parts["ppt/presentation.xml"] = pres.encode("utf-8")
    prels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    for i in drop_slides:
        prels = re.sub(rf'<Relationship Id="rId\d+" Type="[^"]+/slide" Target="slides/slide{i}\.xml"/>', "", prels)
    parts["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")
    ct = parts["[Content_Types].xml"].decode("utf-8")
    for i in drop_slides:
        ct = re.sub(rf'<Override PartName="/ppt/slides/slide{i}\.xml"[^>]*/>', "", ct)
    for i in drop_notes:
        ct = re.sub(rf'<Override PartName="/ppt/notesSlides/notesSlide{i}\.xml"[^>]*/>', "", ct)
    if 'Extension="svg"' not in ct:
        ct = ct.replace("<Default Extension=\"png\"", '<Default Extension="svg" ContentType="image/svg+xml"/><Default Extension="png"', 1)
    parts["[Content_Types].xml"] = ct.encode("utf-8")
    app = parts["docProps/app.xml"].decode("utf-8")
    app = re.sub(r"<Slides>\d+</Slides>", "<Slides>1</Slides>", app); app = re.sub(r"<Notes>\d+</Notes>", "<Notes>1</Notes>", app)
    parts["docProps/app.xml"] = app.encode("utf-8")

    for n in ("[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", f"ppt/slides/slide{KEEP_SLIDE}.xml",
              f"ppt/slides/_rels/slide{KEEP_SLIDE}.xml.rels", "docProps/app.xml"):
        minidom.parseString(parts[n])                       # well-formed or raise
    leftovers = [n for n, d in parts.items() if n.endswith(".rels") and re.search(rb'Target="[^"]*slide(1|2|3|4|5|7)\.xml"', d)]
    assert not leftovers, leftovers

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", parts.pop("[Content_Types].xml"))
        for n, d in parts.items():
            z.writestr(n, d)
    print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KB), {len(links)} clickable links:")
    for t in links:
        print(f"  {t['text']:28s} {t['url']}")


if __name__ == "__main__":
    build()
