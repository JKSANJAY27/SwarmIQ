"""
SwarmIQ — Logging utilities
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Configure and return a named logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the swarmiq namespace."""
    return logging.getLogger(f"swarmiq.{name}" if not name.startswith("swarmiq") else name)
