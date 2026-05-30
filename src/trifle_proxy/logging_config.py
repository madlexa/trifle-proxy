"""Structured logging configuration via structlog."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

VALID_LEVELS = ("debug", "info", "warning", "error", "critical")

_configured = False


def _resolve_level(level: str) -> int:
    """Map a level name to a logging constant; default to INFO."""
    return getattr(logging, level.upper(), logging.INFO)


def configure_logging(level: str = "info", json_logs: bool = False) -> None:
    """Configure structlog for the application.

    Args:
        level: One of debug/info/warning/error/critical (case-insensitive).
        json_logs: Emit JSON lines instead of human-readable console output.
    """
    global _configured

    if level.lower() not in VALID_LEVELS:
        level = "info"

    log_level = _resolve_level(level)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Any
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger, configuring defaults if not done yet."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
