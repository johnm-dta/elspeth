# src/elspeth/telemetry/hookspecs.py
"""pluggy hook specifications for telemetry exporters.

Exporters implement these hooks to register themselves with the framework.
The TelemetryManager calls these hooks during initialization to discover
available exporters.

Usage (implementing an exporter plugin):
    from elspeth.telemetry.hookspecs import hookimpl

    class MyExporterPlugin:
        @hookimpl
        def elspeth_get_exporters(self):
            return [MyExporter]
"""

from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from elspeth.telemetry.protocols import ExporterProtocol

# Use the same project name as the main plugin system
PROJECT_NAME = "elspeth"

# Hook specification marker
hookspec = pluggy.HookspecMarker(PROJECT_NAME)

# Hook implementation marker (for exporter plugins to use)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class ElspethTelemetrySpec:
    """Hook specifications for telemetry exporter plugins."""

    @hookspec
    def elspeth_get_exporters(self) -> list[type["ExporterProtocol"]]:  # type: ignore[empty-body]
        """Return telemetry exporter classes.

        Called during TelemetryManager initialization to discover
        available exporters. Exporters are then instantiated and
        configured based on pipeline settings.

        Returns:
            List of exporter classes (not instances) that implement
            ExporterProtocol
        """
