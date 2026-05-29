"""Logging configuration for aiida-relax-project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[str] = None
    propagate: bool = False


_DEFAULT_CONFIG = LoggingConfig()


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure logging for the package.

    Args:
        config: Logging configuration. Uses defaults if not provided.
    """
    cfg = config or _DEFAULT_CONFIG

    root_logger = logging.getLogger("aiida_relax_project")
    root_logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    root_logger.propagate = cfg.propagate

    if root_logger.handlers:
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(cfg.format, datefmt=cfg.date_format)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if cfg.file_path:
        file_path = Path(cfg.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name.

    Args:
        name: Logger name, typically __name__ of the module

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"aiida_relax_project.{name}")


@contextmanager
def log_section(logger: logging.Logger, title: str, level: int = logging.INFO):
    """Log a section with visual separator.

    Usage:
        with log_section(logger, "Running calculations"):
            # code here
    """
    separator = "=" * 60
    logger.log(level, separator)
    logger.log(level, title.center(60))
    logger.log(level, separator)
    try:
        yield
    finally:
        logger.log(level, "=" * 60)