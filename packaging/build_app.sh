#!/usr/bin/env bash
# Build the DRISHTI-3D native application.
#
#   ./packaging/build_app.sh
#
# Produces dist/DRISHTI-3D.app (macOS) or dist/DRISHTI-3D/ (Windows/Linux).
# Run from the project root.

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON not found. Run 'uv sync --extra gui' first." >&2
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

echo "==> building"
"$PYTHON" -m PyInstaller packaging/drishti3d.spec --noconfirm --distpath dist --workpath build/pyi

if [[ "$OSTYPE" == "darwin"* ]]; then
    APP="dist/DRISHTI-3D.app"
    # PyInstaller's own codesign pass fails when any bundled file carries an
    # extended attribute (it reports "resource fork, Finder information, or
    # similar detritus not allowed"). Stripping xattrs first and signing
    # ad-hoc afterwards is what makes the bundle launchable; without a
    # signature, macOS kills it on first run with no visible error.
    echo "==> stripping extended attributes"
    xattr -cr "$APP"
    echo "==> ad-hoc signing"
    codesign -s - --force --all-architectures --deep "$APP" 2>/dev/null || true
    codesign --verify --deep "$APP" && echo "  signature OK"
    echo
    echo "Built: $APP  ($(du -sh "$APP" | cut -f1))"
    echo "Run:   open $APP"
else
    echo
    echo "Built: dist/DRISHTI-3D/"
    echo "Run:   ./dist/DRISHTI-3D/DRISHTI-3D"
fi
