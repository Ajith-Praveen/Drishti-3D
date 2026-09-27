#!/usr/bin/env bash
#
# Double-click in Finder to build DRISHTI-3D.app and open it.
#
# Produces dist/DRISHTI-3D.app (a normal Mac app you can drag to
# /Applications) and copies it to ~/Applications so it shows up in
# Launchpad and Spotlight. Takes a few minutes the first time.

set -euo pipefail
cd "$(dirname "$0")"

# A stale lock left by an interrupted git command blocks every later
# commit ("index.lock: File exists"). Only remove it if git isn't running.
if [ -f .git/index.lock ] && ! pgrep -x git > /dev/null; then
    rm -f .git/index.lock
fi

if ! command -v uv > /dev/null 2>&1; then
    echo "error: uv is not installed. See https://docs.astral.sh/uv/" >&2
    read -r -p "Press Return to close." _
    exit 1
fi

# The same extras as run.sh. `--extra gui` alone is an EXACT sync that
# uninstalls torch, MapAnything and kornia from .venv -- after which neither
# the built app nor the source checkout can reconstruct anything.
echo "==> syncing dependencies (gui + ml + texture + reference extras)"
uv sync --extra gui --extra ml --extra semantics --extra texture --extra reference

PY=".venv/bin/python"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# See run.sh: uv hides .pth files and the Qt plugin tree on macOS.
chflags nohidden "$SITE"/*.pth 2> /dev/null || true
chflags -R nohidden "$SITE/PySide6" 2> /dev/null || true

if ! "$PY" -c 'import PyInstaller' 2> /dev/null; then
    echo "==> installing PyInstaller into .venv"
    uv pip install --python "$PY" pyinstaller
fi

PYTHON="$PY" ./packaging/build_app.sh

APP="dist/DRISHTI-3D.app"
mkdir -p "$HOME/Applications"
rm -rf "$HOME/Applications/DRISHTI-3D.app"
cp -R "$APP" "$HOME/Applications/"
echo
echo "Installed: ~/Applications/DRISHTI-3D.app"
open "$HOME/Applications/DRISHTI-3D.app"
