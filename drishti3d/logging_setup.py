"""Logging configuration using rich for pretty console output."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the root logger with a RichHandler (and optional file handler).

    Parameters
    ----------
    level:
        Logging level, e.g. "DEBUG", "INFO", or a logging module constant.
    log_file:
        Optional path to also write plain-text logs to.

    Returns
    -------
    The configured root logger.
    """
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    ]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    return logging.getLogger("drishti3d")
