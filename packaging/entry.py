"""Frozen-application entry point.

Separate from ``drishti3d.app.main`` because a PyInstaller entry script has
to do three things a normal ``main`` must not:

1. Call ``multiprocessing.freeze_support()`` before anything else. Without
   it, any library that spawns a worker process re-executes the bundled
   binary from the top and forks the whole application again -- on Windows
   this presents as an endless cascade of windows, which is both baffling
   and hard to kill.
2. Survive being launched with no argv (double-clicking a ``.app`` passes
   none, and on macOS may pass a ``-psn_...`` process-serial-number
   argument that argparse would reject with a usage error and exit 2).
3. Show a GUI message box on a fatal import error rather than dying
   silently. A frozen app has no console: an unhandled traceback at
   startup is completely invisible to the user, who sees the icon bounce
   once and nothing else.

Two headless subcommands run from a terminal against the bundled binary
(``DRISHTI-3D.app/Contents/MacOS/DRISHTI-3D`` on macOS)::

    DRISHTI-3D runtime check [--skip-hash] [--smoke load|inference]
    DRISHTI-3D run VIDEO --telemetry LOG --out DIR [...]

``runtime`` is ``drishti3d.runtime`` (the offline model preflight) and
``run`` is the headless pipeline CLI. Both exist so a build can be proven
to reconstruct -- not merely to open -- and so an operator can script
batch runs on a machine that has only the app installed.
"""

from __future__ import annotations

import multiprocessing
import sys


def _strip_macos_psn(argv: list[str]) -> list[str]:
    """Drop the ``-psn_0_12345`` argument macOS adds when launching a bundle."""
    return [a for a in argv if not a.startswith("-psn_")]


def _run_headless(argv: list[str]) -> int:
    from drishti3d.pipeline.runner import main as run_main

    return run_main(argv)


def _runtime(argv: list[str]) -> int:
    from drishti3d.runtime import main as runtime_main

    return runtime_main(argv)


def _selftest_imports(argv: list[str]) -> int:
    """Import the named modules in order and exercise them lightly (build check for bundled libraries)."""
    import importlib

    for name in argv:
        module = importlib.import_module(name)
        if name == "pyproj":
            module.Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True).transform(-0.74, 41.77)
        elif name == "rasterio":
            module.crs.CRS.from_epsg(32630).to_wkt()
        elif name == "kornia":
            # TorchScript-compiled helpers need their .py source in the bundle.
            import torch
            from kornia.geometry.epipolar import sampson_epipolar_distance

            pts = torch.rand(1, 8, 2)
            sampson_epipolar_distance(pts, pts, torch.eye(3)[None])
    print("imports ok:", " ".join(argv))
    return 0


_COMMANDS = {"run": _run_headless, "runtime": _runtime, "selftest-imports": _selftest_imports}


def _use_bundled_torch_home() -> None:
    """Point torch hub at the bundled copy when the user's cache lacks MapAnything's DINOv2 code.

    Must run before torch is imported: TORCH_HOME is read when torch.hub
    first resolves its directory.
    """
    import os
    from pathlib import Path

    bundle = getattr(sys, "_MEIPASS", None)
    if not bundle or "TORCH_HOME" in os.environ:
        return
    bundled = Path(bundle) / "torch_home"
    user_hub = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torch" / "hub"
    if (bundled / "hub" / "facebookresearch_dinov2_main").is_dir() and not (
        user_hub / "facebookresearch_dinov2_main"
    ).is_dir():
        os.environ["TORCH_HOME"] = str(bundled)


def main() -> int:
    multiprocessing.freeze_support()
    _use_bundled_torch_home()

    argv = _strip_macos_psn(sys.argv[1:])
    if argv and argv[0] in _COMMANDS:
        return _COMMANDS[argv[0]](argv[1:])

    try:
        from drishti3d.app.main import main as app_main
    except Exception as exc:  # noqa: BLE001 -- last-resort startup reporting
        _report_fatal(f"DRISHTI-3D failed to start.\n\n{type(exc).__name__}: {exc}")
        return 1

    return app_main(argv)


def _report_fatal(message: str) -> None:
    """Best-effort GUI error, falling back to stderr if Qt itself is broken."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication([])
        QMessageBox.critical(None, "DRISHTI-3D", message)
        del app
    except Exception:  # noqa: BLE001
        print(message, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
