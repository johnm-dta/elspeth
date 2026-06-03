"""Shared telemetry helpers for plugin lifecycle defaults."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Protocol

import structlog


class WarningLogger(Protocol):
    def warning(self, message: str, **kwargs: Any) -> None: ...


_logger = structlog.get_logger(__name__)


def warn_telemetry_before_start(event: Any, *, logger: WarningLogger = _logger) -> None:
    """Default telemetry callback before on_start() warns instead of silently dropping."""
    logger.warning(
        "telemetry_emit called before on_start() — event dropped",
        event_type=type(event).__name__,
    )


def make_warn_telemetry_before_start(logger: WarningLogger) -> Callable[[Any], None]:
    """Bind the shared pre-start telemetry warning to a module-specific logger."""
    return partial(warn_telemetry_before_start, logger=logger)
