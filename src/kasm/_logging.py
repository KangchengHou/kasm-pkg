"""Logging configuration for kasm workflows."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("kasm")


def configure_logging(
    level: int | str = "INFO", *, filename: str | Path | None = None
) -> None:
    """Configure console logging, and optionally file logging, for ``kasm.*``.

    Only handlers previously installed by this function are replaced. This
    leaves application-owned handlers and the root logger untouched.
    """
    resolved_level = (
        logging._nameToLevel.get(level.upper()) if isinstance(level, str) else level
    )
    if not isinstance(resolved_level, int):
        raise ValueError(f"unknown logging level {level!r}")
    formatter = logging.Formatter(
        "%(asctime)s %(module)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in tuple(logger.handlers):
        if getattr(handler, "_kasm_managed", False):
            logger.removeHandler(handler)
            handler.close()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a"))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._kasm_managed = True
        logger.addHandler(handler)
    logger.setLevel(resolved_level)
    logger.propagate = False
