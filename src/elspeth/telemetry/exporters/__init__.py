# src/elspeth/telemetry/exporters/__init__.py
"""Built-in telemetry exporters.

This package provides exporters for shipping telemetry events to various
observability platforms. Exporters are discovered via pluggy hooks.

Available exporters:
- ConsoleExporter: Write events to stdout/stderr for testing and debugging
- OTLPExporter: Export to OTLP-compatible backends (Jaeger, Tempo, etc.)
- AzureMonitorExporter: Export to Azure Monitor / Application Insights
- DatadogExporter: Export to Datadog APM

Usage:
    from elspeth.telemetry.exporters import ConsoleExporter, OTLPExporter

Plugin registration:
    Exporters are registered via the elspeth_get_exporters hook.
    The BuiltinExportersPlugin in this module registers all built-in exporters.
"""

from elspeth.telemetry.exporters.azure_monitor import AzureMonitorExporter
from elspeth.telemetry.exporters.console import ConsoleExporter
from elspeth.telemetry.exporters.datadog import DatadogExporter
from elspeth.telemetry.exporters.otlp import OTLPExporter
from elspeth.telemetry.hookspecs import hookimpl


class BuiltinExportersPlugin:
    """Plugin that registers built-in telemetry exporters."""

    @hookimpl
    def elspeth_get_exporters(self) -> list[type]:
        """Return built-in exporter classes."""
        return [ConsoleExporter, OTLPExporter, AzureMonitorExporter, DatadogExporter]


__all__ = [
    "AzureMonitorExporter",
    "BuiltinExportersPlugin",
    "ConsoleExporter",
    "DatadogExporter",
    "OTLPExporter",
]
