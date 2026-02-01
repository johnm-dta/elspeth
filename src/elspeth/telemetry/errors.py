# src/elspeth/telemetry/errors.py
"""Telemetry-specific exceptions.

These exceptions are for telemetry subsystem errors only.
They should NOT be raised for pipeline execution errors.
"""


class TelemetryExporterError(Exception):
    """Raised when an exporter encounters a configuration or initialization error.

    This is raised during exporter setup (configure/initialization), NOT during
    export operations. Export operations must not raise - they log errors instead.

    Attributes:
        exporter_name: Name of the exporter that failed
        message: Human-readable error description
    """

    def __init__(self, exporter_name: str, message: str) -> None:
        self.exporter_name = exporter_name
        self.message = message
        super().__init__(f"Exporter '{exporter_name}' failed: {message}")
