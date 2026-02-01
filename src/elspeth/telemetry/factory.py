# src/elspeth/telemetry/factory.py
"""Factory functions for creating TelemetryManager from configuration.

This module provides the glue between configuration (RuntimeTelemetryConfig)
and the runtime TelemetryManager instance. It handles:
1. Looking up exporter classes by name
2. Instantiating and configuring exporters
3. Creating the TelemetryManager with configured exporters

Usage:
    from elspeth.contracts.config import RuntimeTelemetryConfig
    from elspeth.telemetry.factory import create_telemetry_manager

    config = RuntimeTelemetryConfig.from_settings(settings.telemetry)
    manager = create_telemetry_manager(config)
    # manager is ready to use with Orchestrator
"""

from __future__ import annotations

import structlog

from elspeth.contracts.config import RuntimeTelemetryConfig
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.exporters import (
    AzureMonitorExporter,
    ConsoleExporter,
    DatadogExporter,
    OTLPExporter,
)
from elspeth.telemetry.manager import TelemetryManager
from elspeth.telemetry.protocols import ExporterProtocol

logger = structlog.get_logger(__name__)

# Registry of built-in exporter classes by name
_EXPORTER_REGISTRY: dict[str, type[ExporterProtocol]] = {
    "console": ConsoleExporter,
    "otlp": OTLPExporter,
    "azure_monitor": AzureMonitorExporter,
    "datadog": DatadogExporter,
}


def create_telemetry_manager(config: RuntimeTelemetryConfig) -> TelemetryManager | None:
    """Create a TelemetryManager from runtime configuration.

    If telemetry is disabled in config, returns None. Otherwise instantiates
    all configured exporters and returns a TelemetryManager ready for use.

    Args:
        config: Runtime telemetry configuration from RuntimeTelemetryConfig.from_settings()

    Returns:
        TelemetryManager instance if telemetry is enabled, None otherwise

    Raises:
        TelemetryExporterError: If an unknown exporter name is configured
    """
    if not config.enabled:
        logger.debug("telemetry_disabled", reason="config.enabled=False")
        return None

    # Instantiate and configure exporters
    exporters: list[ExporterProtocol] = []
    for exporter_config in config.exporter_configs:
        # Look up exporter class - raises TelemetryExporterError if unknown
        try:
            exporter_class = _EXPORTER_REGISTRY[exporter_config.name]
        except KeyError:
            available = sorted(_EXPORTER_REGISTRY.keys())
            raise TelemetryExporterError(
                exporter_name=exporter_config.name,
                message=f"Unknown exporter. Available exporters: {available}",
            ) from None

        exporter = exporter_class()
        exporter.configure(exporter_config.options)
        exporters.append(exporter)
        logger.debug(
            "exporter_configured",
            exporter=exporter_config.name,
            options_keys=list(exporter_config.options.keys()),
        )

    if not exporters:
        logger.warning(
            "telemetry_enabled_no_exporters",
            message="Telemetry enabled but no exporters configured",
        )

    return TelemetryManager(config, exporters=exporters)
