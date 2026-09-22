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
"""

from __future__ import annotations

import multiprocessing
import sys


def _strip_macos_psn(argv: list[str]) -> list[str]:
    """Drop the ``-psn_0_12345`` argument macOS adds when launching a bundle."""
    return [a for a in argv if not a.startswith("-psn_")]


def main() -> int:
    multiprocessing.freeze_support()

    try:
        from drishti3d.app.main import main as app_main
    except Exception as exc:  # noqa: BLE001 -- last-resort startup reporting
        _report_fatal(f"DRISHTI-3D failed to start.\n\n{type(exc).__name__}: {exc}")
        return 1

    return app_main(_strip_macos_psn(sys.argv[1:]))


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
