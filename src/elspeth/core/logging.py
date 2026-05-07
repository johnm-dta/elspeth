"""Structured logging configuration for ELSPETH.

Uses structlog for structured logging that complements
OpenTelemetry spans for observability.

Architecture:
    This module configures BOTH structlog and stdlib logging to emit
    consistent output (JSON or console). It uses ProcessorFormatter
    to route stdlib log records through structlog's processor chain,
    ensuring that modules using logging.getLogger(__name__) produce
    the same output format as modules using structlog.get_logger().
"""

import io
import logging
import sys
from typing import Any

import structlog
from structlog.stdlib import ProcessorFormatter


class _LazyStdoutStream(io.TextIOBase):
    """Stream wrapper that delegates to the current sys.stdout at write time.

    StreamHandler(sys.stdout) captures a reference to stdout at construction.
    Under pytest (or any tool that swaps stdout), the captured reference goes
    stale after the swap's scope ends. This wrapper resolves sys.stdout lazily
    on every write, so it always targets the live stdout.
    """

    def write(self, s: str) -> int:
        return sys.stdout.write(s)

    def flush(self) -> None:
        sys.stdout.flush()

    @property
    def closed(self) -> bool:
        return sys.stdout.closed


# Third-party loggers that are excessively verbose at DEBUG level.
# These emit HTTP connection details, credential operations, etc. that
# are noise rather than signal. Silence them to WARNING even when
# ELSPETH runs in DEBUG mode.
_NOISY_LOGGERS: tuple[str, ...] = (
    # Azure SDK - emits HTTP request/response details for every call
    "azure",
    "azure.core",
    "azure.core.pipeline",
    "azure.core.pipeline.policies",
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.monitor",
    # urllib3 - connection pool management noise
    "urllib3",
    "urllib3.connectionpool",
    # OpenTelemetry SDK internals - span processing details
    "opentelemetry",
    "opentelemetry.sdk",
    "opentelemetry.exporter",
    # httpx/httpcore - async HTTP client internals
    "httpx",
    "httpcore",
)

# Track ELSPETH-owned handler ids for selective removal on reconfiguration.
# Using id-set instead of getattr() to comply with the defensive programming ban.
_elspeth_handler_ids: set[int] = set()


def _remove_internal_fields(
    logger: logging.Logger | None,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Remove internal structlog fields from output.

    ProcessorFormatter ALWAYS adds _record and _from_structlog when processing
    log records (see structlog.stdlib.ProcessorFormatter.format()). These are
    internal bookkeeping and should not appear in output.

    Using del instead of .pop(default) because these fields are guaranteed
    present by ProcessorFormatter's contract. KeyError would indicate a bug
    in our structlog integration.
    """
    del event_dict["_record"]
    del event_dict["_from_structlog"]
    return event_dict


def configure_logging(
    *,
    json_output: bool = False,
    level: str = "INFO",
) -> None:
    """Configure structlog and stdlib logging for ELSPETH.

    This configures BOTH structlog and stdlib logging to produce consistent
    output. Modules using logging.getLogger(__name__) will emit the same
    format (JSON or console) as modules using structlog.get_logger().

    Args:
        json_output: If True, output JSON. If False, human-readable.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = getattr(logging, level.upper())

    # Shared processors applied to ALL log records (structlog and stdlib)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    # Final processors for the formatter (run after shared_processors)
    if json_output:
        final_processors: list[Any] = [
            _remove_internal_fields,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        final_processors = [
            _remove_internal_fields,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog to route through stdlib logging
    # wrap_for_formatter prepares event_dict for ProcessorFormatter
    structlog.configure(
        processors=[*shared_processors, ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        # IMPORTANT: Disable caching to allow reconfiguration in tests
        # Without this, tests that reconfigure logging will get stale loggers
        cache_logger_on_first_use=False,
    )

    # Configure stdlib logging with ProcessorFormatter
    # This ensures stdlib loggers go through structlog's processor chain
    handler = logging.StreamHandler(_LazyStdoutStream())
    handler.setFormatter(
        ProcessorFormatter(
            processors=final_processors,
            # foreign_pre_chain: processors for stdlib-only records
            # (structlog records already went through shared_processors)
            foreign_pre_chain=shared_processors,
        )
    )

    # Remove only ELSPETH-owned handlers (preserve others like pytest caplog)
    root = logging.getLogger()
    removed_ids = {id(h) for h in root.handlers if id(h) in _elspeth_handler_ids}
    root.handlers = [h for h in root.handlers if id(h) not in _elspeth_handler_ids]
    _elspeth_handler_ids.difference_update(removed_ids)

    # Track our handler by id so we can find it later
    _elspeth_handler_ids.add(id(handler))
    root.addHandler(handler)
    root.setLevel(log_level)

    # Silence noisy third-party loggers that spam DEBUG output.
    # Never make noisy loggers less restrictive than the configured root level.
    # This prevents WARNING output when root is ERROR/CRITICAL.
    noisy_level = max(log_level, logging.WARNING)
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(noisy_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound logger for a module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Bound structlog logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
