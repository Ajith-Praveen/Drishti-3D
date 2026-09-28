#!/usr/bin/env bash
# Build the DRISHTI-3D native application.
#
#   ./packaging/build_app.sh
#
# Produces dist/DRISHTI-3D.app (macOS) or dist/DRISHTI-3D/ (Windows/Linux).
# Run from the project root.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found. Run 'uv sync --extra gui --extra ml' first." >&2
    exit 1
fi

echo "==> checking build dependencies"
"$PYTHON" - <<'PY'
import sys
missing = []
for mod, hint in (("PyInstaller", "pyinstaller"), ("PySide6", "uv sync --extra gui"), ("vtk", "uv sync --extra gui")):
    try:
        __import__(mod)
    except ImportError:
        missing.append((mod, hint))
if missing:
    for mod, hint in missing:
        print(f"  missing {mod}  ->  {hint}", file=sys.stderr)
    sys.exit(1)
print("  ok")
PY

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "==> rendering app icon from drishti3d/app/assets/logo.svg"
    rm -rf build/DRISHTI-3D.iconset
    QT_QPA_PLATFORM=offscreen "$PYTHON" packaging/make_icon.py build/DRISHTI-3D.iconset
    iconutil -c icns build/DRISHTI-3D.iconset -o packaging/DRISHTI-3D.icns
fi

echo "==> building"
"$PYTHON" -m PyInstaller packaging/drishti3d.spec --noconfirm --distpath dist --workpath build/pyi

if [[ "$OSTYPE" == "darwin"* ]]; then
    APP="dist/DRISHTI-3D.app"
    # PyInstaller's own codesign pass fails when any bundled file carries an
    # extended attribute (it reports "resource fork, Finder information, or
    # similar detritus not allowed"). Stripping xattrs first and signing
    # ad-hoc afterwards is what makes the bundle launchable; without a
    # signature, macOS kills it on first run with no visible error.
    # Two wheels shipping different dylibs under one file name (pyproj's and
    # rasterio's libproj) collide on PyInstaller's single @rpath symlink;
    # point each package back at its own copy BEFORE signing.
    echo "==> resolving bundled dylib name collisions"
    "$PYTHON" packaging/fix_dylib_collisions.py "$APP"
    echo "==> stripping extended attributes"
    xattr -cr "$APP"
    echo "==> ad-hoc signing"
    codesign -s - --force --all-architectures --deep "$APP" 2>/dev/null || true
    codesign --verify --deep "$APP" && echo "  signature OK"
    # Validate the default pipeline without requiring optional learned-depth
    # assets. Explicitly bundled fallback weights must pass their own check.
    if [ "${DRISHTI3D_BUNDLE_ML:-auto}" != "0" ]; then
        echo "==> reconstruction runtime check (frozen app)"
        "$PYTHON" packaging/smoke_test.py "$APP/Contents/MacOS/DRISHTI-3D"
        if [ -n "${DRISHTI3D_BUILD_WEIGHTS_DIR:-}" ]; then
            "$APP/Contents/MacOS/DRISHTI-3D" runtime check --skip-hash
        fi
    fi
    echo
    echo "Built: $APP  ($(du -sh "$APP" | cut -f1))"
    echo "Run:   open $APP"
    echo "CLI:   $APP/Contents/MacOS/DRISHTI-3D run VIDEO --telemetry LOG --out DIR"
else
    echo
    echo "Built: dist/DRISHTI-3D/"
    echo "Run:   ./dist/DRISHTI-3D/DRISHTI-3D"
fi
