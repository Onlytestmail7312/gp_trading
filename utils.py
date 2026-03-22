import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

"""
Shared utility functions: logging, formatting, directory creation.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import OUTPUT_DIR


# ===================================================================
# DIRECTORY SETUP
# ===================================================================
def ensure_output_dirs() -> None:
    """Create all required output directories."""
    dirs = [
        OUTPUT_DIR,
        OUTPUT_DIR / "plots",
        OUTPUT_DIR / "reports",
        OUTPUT_DIR / "models",
        OUTPUT_DIR / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ===================================================================
# LOGGING
# ===================================================================
_loggers: dict = {}


def get_logger(name: str = "gp_system") -> logging.Logger:
    """Return a configured logger (console + file)."""
    global _loggers
    if name in _loggers:
        return _loggers[name]

    ensure_output_dirs()

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        _loggers[name] = logger
        return _loggers[name]

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(OUTPUT_DIR / "logs" / "gp_system.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    ))
    logger.addHandler(fh)

    _logger = logger
    return logger


# ===================================================================
# TIME FORMATTING
# ===================================================================
def fmt_seconds(seconds: float) -> str:
    """Format seconds into human-readable string."""
    s = max(int(seconds), 0)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d > 0:
        return f"{d}d {h:02d}h {m:02d}m"
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class Timer:
    """Simple context-manager timer."""

    def __init__(self, label: str = ""):
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            log = get_logger()
            log.info(f"{self.label} took {fmt_seconds(self.elapsed)}")


# ===================================================================
# BANNER
# ===================================================================
def print_banner(title: str) -> None:
    """Print a section banner."""
    log = get_logger()
    width = 70
    log.info("=" * width)
    log.info(f"  {title}")
    log.info("=" * width)


def print_table(headers: list, rows: list, col_widths: Optional[list] = None) -> None:
    """Print a simple aligned table to the logger."""
    log = get_logger()
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]

    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    log.info(header_line)
    log.info("-" * len(header_line))
    for row in rows:
        line = "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        log.info(line)