#!/usr/bin/env bash
#
# Launch the DRISHTI-3D desktop app.
#
# Use this rather than `uv run drishti3d`, because of a macOS/uv
# interaction that otherwise stops the app starting at all.
#
# uv marks everything it writes inside .venv with the macOS UF_HIDDEN
# flag. Two separate consumers refuse to look at hidden files, and each
# produces a different, badly-misleading failure:
#
#   1. CPython >= 3.11.10 / 3.12.5 deliberately SKIPS hidden .pth files
#      (site.addpackage tests st_flags & UF_HIDDEN). This project is
#      installed editable, so its sys.path entry comes from
#      _editable_impl_drishti3d.pth -- one of the files uv hides. Symptom:
#          ModuleNotFoundError: No module named 'drishti3d'
#
#   2. Qt enumerates its plugin directories with QDir, which excludes
#      hidden entries by default, so PySide6/Qt/plugins/platforms is
#      invisible to it even though QLibraryInfo reports the correct path.
#      Symptom:
#          qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""
#      Note the empty path in that message: it is not a search-path
#      problem, and setting QT_PLUGIN_PATH does not help.
#
# Both come back on every `uv sync`, so clearing the flags by hand is not
# a fix. Clear them here, then exec the venv interpreter directly --
# going through `uv run` would re-sync and re-hide on the way in.
#
# Usage:
#   ./run.sh                                   # empty session
#   ./run.sh --video flight.mp4 --telemetry flight.csv
#   ./run.sh --log-level DEBUG

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv > /dev/null 2>&1; then
    echo "error: uv is not installed. See https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "==> syncing dependencies (gui + ml extras)"
uv sync --extra gui --extra ml

VENV_PYTHON=".venv/bin/python"
SITE_PACKAGES="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

if command -v chflags > /dev/null 2>&1; then
    echo "==> clearing macOS hidden flags in the virtualenv"
    # The .pth files at the top level, for CPython's sake...
    chflags nohidden "$SITE_PACKAGES"/*.pth 2> /dev/null || true
    # ...and the Qt plugin tree, for Qt's. Scoped to PySide6 rather than
    # recursing the whole venv, which would walk several gigabytes of
    # torch for no benefit.
    chflags -R nohidden "$SITE_PACKAGES/PySide6" 2> /dev/null || true
fi

if ! "$VENV_PYTHON" -c 'import drishti3d' 2> /dev/null; then
    echo "error: the drishti3d package is still not importable from $SITE_PACKAGES" >&2
    echo "       try: uv sync --reinstall-package drishti3d" >&2
    exit 1
fi

echo "==> starting DRISHTI-3D"
exec "$VENV_PYTHON" -m drishti3d.app.main "$@"
