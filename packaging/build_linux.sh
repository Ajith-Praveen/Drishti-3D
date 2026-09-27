#!/usr/bin/env bash
# Build DRISHTI-3D for Linux (x86_64 or aarch64), with NVIDIA CUDA acceleration.
#
#   ./packaging/build_linux.sh
#
# Produces dist/DRISHTI-3D/DRISHTI-3D (+ a .desktop entry and icon) and
# dist/DRISHTI-3D-linux-<arch>.tar.gz. Run from anywhere; needs uv
# (https://docs.astral.sh/uv/) and git.
#
# PyPI's Linux x86_64 torch wheels already carry CUDA (the nvidia-* runtime
# wheels come in through the lock), so no extra index is needed there; on
# a machine without an NVIDIA GPU the same build runs on the CPU.
#
# System libraries Qt/VTK need at run time (Debian/Ubuntu names):
#   libgl1 libegl1 libglib2.0-0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4
#   libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0 libxrender1 libxi6 libdbus-1-3 ffmpeg

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv > /dev/null 2>&1; then
    echo "error: uv is not installed. See https://docs.astral.sh/uv/" >&2
    exit 1
fi

PY=.venv/bin/python
echo "==> syncing the locked environment (gui + ml + semantics + reference)"
uv sync --extra gui --extra ml --extra semantics --extra reference
uv pip install --python "$PY" pyinstaller
"$PY" -c "import torch; print('  torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU available:', torch.cuda.is_available())"

echo "==> rendering the app icon"
QT_QPA_PLATFORM=offscreen "$PY" packaging/make_icon.py --png packaging/DRISHTI-3D.png

echo "==> building"
"$PY" -m PyInstaller packaging/drishti3d.spec --noconfirm --distpath dist --workpath build/pyi

APP=dist/DRISHTI-3D
cp packaging/DRISHTI-3D.png "$APP/DRISHTI-3D.png"
cat > "$APP/drishti3d.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=DRISHTI-3D
Comment=Single-pass drone video to a georeferenced 3D model
Exec=DRISHTI-3D
Icon=DRISHTI-3D
Categories=Graphics;Science;Geography;
Terminal=false
DESKTOP

echo "==> smoke test: bundled imports"
QT_QPA_PLATFORM=offscreen "$APP/DRISHTI-3D" runtime check --skip-hash || echo "  (runtime check reported problems; see above)"

ARCH=$(uname -m)
tar -C dist -czf "dist/DRISHTI-3D-linux-$ARCH.tar.gz" DRISHTI-3D
echo "Built: $APP/DRISHTI-3D  (archive: dist/DRISHTI-3D-linux-$ARCH.tar.gz)"
echo "CLI:   $APP/DRISHTI-3D run VIDEO --telemetry LOG --out DIR"
