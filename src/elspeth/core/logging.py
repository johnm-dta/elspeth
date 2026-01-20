# src/elspeth/core/logging.py
"""Structured logging configuration for ELSPETH.

Uses structlog for structured logging that complements
OpenTelemetry spans for observability.
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    *,
    json_output: bool = False,
    level: str = "INFO",
) -> None:
    """Configure structlog for ELSPETH.

    Args:
        json_output: If True, output JSON. If False, human-readable.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        force=True,  # Allow reconfiguration
    )

    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        # JSON output for machine processing
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console output for humans
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        # IMPORTANT: Disable caching to allow reconfiguration in tests
        # Without this, tests that reconfigure logging will get stale loggers
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound logger for a module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Bound structlog logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
