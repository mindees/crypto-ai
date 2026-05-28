"""Project-wide structured logging."""
from __future__ import annotations

import logging
import os
import sys
from logging import Logger
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)sZ %(levelname)s %(name)s :: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str = "mindees", level: str | int | None = None) -> Logger:
    """Return a configured logger. Idempotent — safe to call repeatedly."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()
    logger.setLevel(resolved_level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def add_file_handler(logger: Logger, path: str | Path, level: str | int = "DEBUG") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    logger.addHandler(fh)
