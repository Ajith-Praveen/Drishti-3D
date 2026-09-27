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
    # Prove the bundle can reconstruct, not just open: the same offline
    # preflight every run performs (packages, pinned versions, backend,
    # weights). A slim build (DRISHTI3D_BUNDLE_ML=0) skips this.
    if [ "${DRISHTI3D_BUNDLE_ML:-auto}" != "0" ]; then
        echo "==> reconstruction runtime check (frozen app)"
        if "$APP/Contents/MacOS/DRISHTI-3D" runtime check --skip-hash; then
            echo "  runtime OK"
        else
            echo "error: the built app cannot reconstruct real footage (runtime check failed above)" >&2
            exit 1
        fi
        # The runtime check imports torch and the model, not the pipeline.
        # Import the libraries whose bundled copies have collided before
        # (pyproj and rasterio both ship a different libproj), in BOTH
        # orders, then every stage via `run --help`.
        echo "==> pipeline import check (frozen app)"
        if "$APP/Contents/MacOS/DRISHTI-3D" run --help > /dev/null \
            && "$APP/Contents/MacOS/DRISHTI-3D" selftest-imports pyproj rasterio \
            && "$APP/Contents/MacOS/DRISHTI-3D" selftest-imports rasterio pyproj \
            && "$APP/Contents/MacOS/DRISHTI-3D" selftest-imports kornia; then
            echo "  pipeline imports OK"
        else
            echo "error: the built app cannot import the pipeline (see the traceback above)" >&2
            exit 1
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
