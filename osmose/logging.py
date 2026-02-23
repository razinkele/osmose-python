"""Centralized logging configuration for OSMOSE."""

import logging
import sys


def setup_logging(
    name: str = "osmose",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger with console output.

    Args:
        name: Logger name (use dotted names for hierarchy, e.g. "osmose.runner").
        level: Logging level (default INFO).

    Returns:
        Configured logger instance. Repeated calls with the same name
        return the same logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
