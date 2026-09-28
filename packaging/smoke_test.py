"""Fail-closed smoke checks for a built executable or the Docker source entry.

python packaging/smoke_test.py dist/DRISHTI-3D/DRISHTI-3D
python packaging/smoke_test.py python packaging/entry.py

No model downloads, optional learned-depth weights, GPU or display required.
Each command gets a fresh process to catch native-library load-order bugs.
"""
from __future__ import annotations

import os
import subprocess
import sys

CHECKS = (
    ("run", "--help"),
    ("selftest-imports", "torch", "torchvision", "cv2", "open3d", "av"),
    ("selftest-imports", "pyproj", "rasterio"),
    ("selftest-imports", "rasterio", "pyproj"),
    ("selftest-imports", "kornia", "mapanything", "transformers.models.segformer"),
    ("selftest-imports", "drishti3d.pipeline.stages", "drishti3d.app.main_window", "PySide6", "vtkmodules.all"),
)


def main(command: list[str] | None = None) -> int:
    command = sys.argv[1:] if command is None else command
    if not command:
        raise SystemExit("usage: smoke_test.py EXECUTABLE [ENTRY_SCRIPT]")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HF_HUB_OFFLINE="1")
    for check in CHECKS:
        print("==> smoke:", " ".join(check), flush=True)
        # subprocess.run waits even for Windows GUI applications. In particular,
        # the executable must close all bundle files before ZIP creation starts.
        subprocess.run([*command, *check], env=env, check=True, timeout=180)
    print("All package smoke checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
