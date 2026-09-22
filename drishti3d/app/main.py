"""DRISHTI-3D desktop application entry point.

Launches the native PySide6 + VTK desktop shell. No web technology, no
network access — this process only touches the local filesystem.
"""

from __future__ import annotations

import argparse
import sys

from drishti3d.logging_setup import setup_logging


def version(package: str) -> str:
    """Installed version, or a placeholder when running from a source tree."""
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        return _version(package)
    except PackageNotFoundError:
        return "0.0.0+dev"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="drishti3d",
        description="DRISHTI-3D: drone video -> georeferenced 3D model (native desktop app).",
    )
    parser.add_argument("--video", type=str, default=None, help="Path to a drone video to preload.")
    parser.add_argument(
        "--telemetry", type=str, default=None, help="Path to a flight telemetry file to preload."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a pipeline config YAML.")
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(level=args.log_level)
    logger.info("Starting DRISHTI-3D")

    # Imported lazily so `--help` / arg-parsing errors don't pay the cost of
    # importing PySide6/VTK, and so this module stays importable in
    # environments without the `gui` extra installed.
    from PySide6.QtWidgets import QApplication

    from drishti3d.app.main_window import MainWindow
    from drishti3d.app.theme import apply_theme

    app = QApplication(sys.argv[:1] + (argv or []))

    # Identity, set before any window exists. Without this macOS shows
    # "Python" in the menu bar and a generic Python icon in the dock,
    # which is the first thing anyone notices about a packaged build.
    app.setApplicationName("DRISHTI-3D")
    app.setApplicationDisplayName("DRISHTI-3D")
    app.setOrganizationName("DRISHTI-3D")
    app.setOrganizationDomain("drishti3d.local")
    app.setApplicationVersion(version("drishti3d"))

    from drishti3d.app import icons

    app.setWindowIcon(icons.app_icon())
    apply_theme(app)

    window = MainWindow()

    if args.video:
        window.load_video(args.video)
    if args.telemetry:
        window.load_telemetry(args.telemetry)
    if args.config:
        window.apply_config_file(args.config)

    window.show()
    # After show(): the VTK interactor needs a realised window handle
    # before it can initialise its render loop.
    window.viewport.start()
    window.raise_()
    window.activateWindow()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
