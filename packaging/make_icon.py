"""Render the app icon (logo on a dark tile) for every platform's bundle.

    python packaging/make_icon.py build/DRISHTI-3D.iconset        # macOS .iconset folder
    iconutil -c icns build/DRISHTI-3D.iconset -o packaging/DRISHTI-3D.icns
    python packaging/make_icon.py --ico packaging/DRISHTI-3D.ico   # Windows executable icon
    python packaging/make_icon.py --png packaging/DRISHTI-3D.png   # Linux desktop entry (512 px)

Uses the same ``icons.app_icon`` the running app shows in the dock/taskbar,
in the operator's saved theme, so the file-manager icon and the running
window's icon match.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _icon():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from drishti3d.app import icons, theme

    theme.load_preferences()
    return app, icons.app_icon()


def _pixmap(icon, px: int):
    from PySide6.QtCore import QSize

    pm = icon.pixmap(QSize(px, px))
    return pm if pm.width() == px else pm.scaled(px, px)


def write_iconset(out_dir: str) -> int:
    app, icon = _icon()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for base in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = base * scale
            name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
            _pixmap(icon, px).save(str(out / name), "PNG")
    del app
    print(f"  wrote {out}")
    return 0


def write_png(path: str, px: int = 512) -> int:
    app, icon = _icon()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _pixmap(icon, px).save(str(path), "PNG")
    del app
    print(f"  wrote {path}")
    return 0


def write_ico(path: str) -> int:
    """A multi-resolution .ico (16-256 px), the sizes Windows Explorer and the taskbar ask for."""
    import io

    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice

    app, icon = _icon()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    _pixmap(icon, 256).save(buf, "PNG")
    image = Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="ICO", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    del app
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--ico"]:
        sys.exit(write_ico(args[1] if len(args) > 1 else "packaging/DRISHTI-3D.ico"))
    if args[:1] == ["--png"]:
        sys.exit(write_png(args[1] if len(args) > 1 else "packaging/DRISHTI-3D.png"))
    sys.exit(write_iconset(args[0] if args else "build/DRISHTI-3D.iconset"))
