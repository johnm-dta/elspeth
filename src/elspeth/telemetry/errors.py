"""Telemetry-specific exceptions.

These exceptions are for telemetry subsystem errors only.
They should NOT be raised for pipeline execution errors.

``TelemetryExporterError`` lives in :mod:`elspeth.contracts.errors` (L0) so the
engine layer can reference it without crossing the L2→L3 layer boundary. It is
re-exported here for callers within the telemetry subsystem; the class object
is identical, so ``isinstance``/``except`` identity is preserved.
"""

from elspeth.contracts.errors import TelemetryExporterError

# Exceptions that represent transport/IO failures — safe to swallow during telemetry export.
# Everything else is a programming error that must crash.
# Individual exporters may extend this with SDK-specific transport exceptions.
TELEMETRY_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,  # covers socket.error, BrokenPipeError, ConnectionResetError, etc.
)


__all__ = ["TELEMETRY_TRANSPORT_ERRORS", "TelemetryExporterError"]
