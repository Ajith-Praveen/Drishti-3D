"""Rebuild the Slide 3 technology-stack asset with official technology marks.

Logos are local copies of the Simple Icons brand marks downloaded into ``logos/``;
technologies without an official mark in that set remain labelled rather than being
misrepresented by a random icon.
"""
from __future__ import annotations

import base64
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

HERE = Path(__file__).parent
OUT_SVG = HERE / "techstack_vertical.svg"
OUT_PNG = HERE / "techstack_vertical.png"
LOGOS = {name: base64.b64encode((HERE / "logos" / f"{name}.svg").read_bytes()).decode() for name in
         ("opencv", "pytorch", "python", "numpy", "scipy", "qt", "nvidia", "apple", "ffmpeg", "open3d")}

GROUPS = [
    ("Built by us", "#F07700", ["Camera solver", "Depth engine", "Depth check", "Keyframe triage", "Clock sync", "Georeferencing"], [None] * 6),
    ("Vision + AI", "#2E73B8", ["OpenCV", "DISK + LightGlue", "Open3D TSDF", "SegFormer", "MapAnything", "xatlas UV"], ["opencv", None, "open3d", None, None, None]),
    ("Desktop application", "#2E73B8", ["Qt 6 · PySide6", "VTK 9 viewer", "PyInstaller"], ["qt", None, None]),
    ("Geo + media", "#00A98F", ["PyAV · FFmpeg", "pyproj", "laspy", "tifffile", "glTF · FBX · OBJ", "GeoJSON · KML"], ["ffmpeg", None, None, None, None, None]),
    ("Compute", "#627391", ["Python 3.12", "NumPy", "SciPy", "PyTorch", "Apple MPS", "NVIDIA CUDA"], ["python", "numpy", "scipy", "pytorch", "apple", "nvidia"]),
]


def data_uri(name: str) -> str:
    return "data:image/svg+xml;base64," + LOGOS[name]


def svg() -> str:
    y = 16; elements = []
    for title, colour, labels, icons in GROUPS:
        rows = (len(labels) + 2) // 3; h = 105 + rows * 165
        elements.append(f'<rect x="10" y="{y}" width="1480" height="{h}" rx="16" fill="#F8FBFD" stroke="{colour}" stroke-width="5"/>')
        elements.append(f'<rect x="10" y="{y}" width="1480" height="78" rx="16" fill="{colour}"/>')
        elements.append(f'<text x="52" y="{y + 53}" class="group">{title}</text>')
        for k, (label, icon) in enumerate(zip(labels, icons)):
            r, c = divmod(k, 3); cx, cy = 250 + c * 500, y + 126 + r * 165
            elements.append(f'<rect x="{cx - 62}" y="{cy - 62}" width="124" height="124" rx="22" fill="#FFFFFF" stroke="{colour}" stroke-width="3"/>')
            if icon:
                elements.append(f'<image x="{cx - 39}" y="{cy - 39}" width="78" height="78" href="{data_uri(icon)}" preserveAspectRatio="xMidYMid meet"/>')
            else:
                initials = "•" if label.startswith(("Camera", "Depth", "Keyframe", "Clock", "Geo")) else label.split()[0][:2].upper()
                elements.append(f'<text x="{cx}" y="{cy + 19}" class="initial" text-anchor="middle" fill="{colour}">{initials}</text>')
            elements.append(f'<text x="{cx}" y="{cy + 102}" class="label" text-anchor="middle">{label}</text>')
        y += h + 20
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="{y}" viewBox="0 0 1500 {y}">
<style>.group{{font:700 38px Arial;fill:#fff}} .label{{font:600 31px Arial;fill:#102A52}} .initial{{font:700 50px Arial}}</style>{''.join(elements)}</svg>'''


def main() -> None:
    OUT_SVG.write_text(svg())
    app = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(QByteArray(OUT_SVG.read_bytes()))
    image = QImage(QSize(1500, 2071), QImage.Format_ARGB32); image.fill(QColor("white"))
    painter = QPainter(image); renderer.render(painter, QRectF(0, 0, 1500, 2071)); painter.end()
    image.save(str(OUT_PNG))
    print(OUT_SVG, OUT_PNG)


if __name__ == "__main__":
    main()
