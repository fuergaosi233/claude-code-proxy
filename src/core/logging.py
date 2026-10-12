"""
Centralized logging configuration for the Claude-to-OpenAI proxy.

Provides a ``get_logger`` factory that returns a logger named after the
calling module, plus a few convenience utilities for model-state logging.
All logs are written to a single file (logs/proxy.log) that is overwritten
on each restart, with console output also enabled.
"""

import logging
import os
import sys
from typing import Any, Optional

from src.core.config import config

# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------
_LOG_LEVEL_RAW = config.log_level.split("#")[0].split()[0].upper()
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_LOG_LEVEL = _LOG_LEVEL_RAW if _LOG_LEVEL_RAW in _VALID_LEVELS else "INFO"

# ---------------------------------------------------------------------------
# Log file setup - single file, overwrite on restart
# ---------------------------------------------------------------------------
# Determine log file path (relative to project root)
_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_script_dir)
_log_dir = os.path.join(_project_root, "logs")
_log_file = os.path.join(_log_dir, "proxy.log")

# Ensure logs directory exists
os.makedirs(_log_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Root logger / format
# ---------------------------------------------------------------------------
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Get root logger and configure handlers
_root_logger = logging.getLogger()
_root_logger.setLevel(getattr(logging, _LOG_LEVEL))

# Clear any existing handlers
_root_logger.handlers = []

# Console handler (stream)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(getattr(logging, _LOG_LEVEL))
_console_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))
_root_logger.addHandler(_console_handler)

# File handler - overwrite mode (w = write, not append)
_file_handler = logging.FileHandler(_log_file, mode="w", encoding="utf-8")
_file_handler.setLevel(getattr(logging, _LOG_LEVEL))
_file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FMT))
_root_logger.addHandler(_file_handler)

# Silence noisy third-party loggers
for _noisy in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name* (convention: ``__name__`` from the calling module)."""
    return logging.getLogger(name)


# Pre-made logger for legacy imports (endpoints.py, etc.)
logger = get_logger("proxy")


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------
# Centralized redaction utilities for log-friendly summaries of values that
# may carry request/response message content (text, tool arguments, base64
# image payloads). Use these instead of logging raw chunks/exceptions, which
# can echo user content back into logs.

# How many characters of a value to keep when a short, non-sensitive prefix is
# useful for debugging without capturing message bodies.
_REDACT_PREFIX_LEN = 40


def redact_brief(value: Any, max_len: int = _REDACT_PREFIX_LEN) -> str:
    """Return a non-sensitive, length-bounded summary of ``value``.

    Returns ``type(value).__name__`` plus a truncated ``len`` for list/str/
    bytes values (the common shapes of chunk/exception payloads), rather than
    the content itself, so message text and base64 data never reach logs.
    """
    if value is None:
        return "None"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes(len={len(value)})"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def safe_error_summary(exc: BaseException) -> str:
    """Return a log-safe summary of an exception (type name + status only).

    ``str(exc)`` is avoided because OpenAI SDK error messages frequently embed
    the rejected request body or streamed content. Callers that need a
    user-facing message should resolve one via :func:`classify_openai_error`
    style helpers, not from this summary.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return f"{type(exc).__name__}(status={status})"
    return type(exc).__name__


# ---------------------------------------------------------------------------
# Model-in-use marker
# ---------------------------------------------------------------------------
_MODEL_MARKER = "🎯"


def log_active_models(log: Optional[logging.Logger] = None) -> None:
    """Emit an INFO-level marker showing the currently active model for each tier.

    Call this at startup and after any fallback/restore event so operators
    can always grep for ``🎯`` to see which models are in rotation.
    """
    if log is None:
        log = logging.getLogger("config")
    big = config.big_model
    middle = config.middle_model
    small = config.small_model

    big_idx = config._active_big_idx
    middle_idx = config._active_middle_idx
    small_idx = config._active_small_idx

    def _pos(idx: int, models: list) -> str:
        if idx == 0:
            return "primary"
        if idx < len(models):
            return f"fallback #{idx}"
        return "exhausted"

    big_pos = _pos(big_idx, config.big_models)
    middle_pos = _pos(middle_idx, config.middle_models)
    small_pos = _pos(small_idx, config.small_models)

    log.info(
        "%s ACTIVE MODELS ─ big: %s (%s) | middle: %s (%s) | small: %s (%s)",
        _MODEL_MARKER,
        big,
        big_pos,
        middle,
        middle_pos,
        small,
        small_pos,
    )


def log_model_switch(
    log: logging.Logger,
    tier: str,
    old_model: str,
    new_model: str,
    reason: str = "fallback",
) -> None:
    """Log a model switch with a consistent format and follow up with the active-models marker."""
    log.warning(
        "🔄 MODEL SWITCH [%s] %s → %s  (%s)",
        tier.upper(),
        old_model,
        new_model,
        reason,
    )
    log_active_models(log)
