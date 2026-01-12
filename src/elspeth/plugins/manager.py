# src/elspeth/plugins/manager.py
"""Plugin manager for discovery, registration, and lifecycle.

Uses pluggy for hook-based plugin registration.
"""

from typing import Any

import pluggy

from elspeth.plugins.hookspecs import (
    PROJECT_NAME,
    ElspethSinkSpec,
    ElspethSourceSpec,
    ElspethTransformSpec,
)
from elspeth.plugins.protocols import (
    AggregationProtocol,
    CoalesceProtocol,
    GateProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)


class PluginManager:
    """Manages plugin discovery, registration, and lookup.

    Usage:
        manager = PluginManager()
        manager.register(MyPlugin())

        transforms = manager.get_transforms()
        my_transform = manager.get_transform_by_name("my_transform")
    """

    def __init__(self) -> None:
        self._pm = pluggy.PluginManager(PROJECT_NAME)

        # Register hookspecs
        self._pm.add_hookspecs(ElspethSourceSpec)
        self._pm.add_hookspecs(ElspethTransformSpec)
        self._pm.add_hookspecs(ElspethSinkSpec)

        # Caches
        self._sources: list[type[SourceProtocol]] = []
        self._transforms: list[type[TransformProtocol]] = []
        self._gates: list[type[GateProtocol]] = []
        self._aggregations: list[type[AggregationProtocol]] = []
        self._coalesces: list[type[CoalesceProtocol]] = []
        self._sinks: list[type[SinkProtocol]] = []

    def register(self, plugin: Any) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance implementing hook methods
        """
        self._pm.register(plugin)
        self._refresh_caches()

    def _refresh_caches(self) -> None:
        """Refresh plugin caches from hooks."""
        self._sources = []
        self._transforms = []
        self._gates = []
        self._aggregations = []
        self._coalesces = []
        self._sinks = []

        # Collect from all registered plugins
        for sources in self._pm.hook.elspeth_get_source():
            self._sources.extend(sources)

        for transforms in self._pm.hook.elspeth_get_transforms():
            self._transforms.extend(transforms)

        for gates in self._pm.hook.elspeth_get_gates():
            self._gates.extend(gates)

        for aggs in self._pm.hook.elspeth_get_aggregations():
            self._aggregations.extend(aggs)

        for coalesces in self._pm.hook.elspeth_get_coalesces():
            self._coalesces.extend(coalesces)

        for sinks in self._pm.hook.elspeth_get_sinks():
            self._sinks.extend(sinks)

    # === Getters ===

    def get_sources(self) -> list[type[SourceProtocol]]:
        """Get all registered source plugins."""
        return self._sources.copy()

    def get_transforms(self) -> list[type[TransformProtocol]]:
        """Get all registered transform plugins."""
        return self._transforms.copy()

    def get_gates(self) -> list[type[GateProtocol]]:
        """Get all registered gate plugins."""
        return self._gates.copy()

    def get_aggregations(self) -> list[type[AggregationProtocol]]:
        """Get all registered aggregation plugins."""
        return self._aggregations.copy()

    def get_coalesces(self) -> list[type[CoalesceProtocol]]:
        """Get all registered coalesce plugins."""
        return self._coalesces.copy()

    def get_sinks(self) -> list[type[SinkProtocol]]:
        """Get all registered sink plugins."""
        return self._sinks.copy()

    # === Lookup by name ===

    def get_source_by_name(self, name: str) -> type[SourceProtocol] | None:
        """Get source plugin by name."""
        for source in self._sources:
            if source.name == name:
                return source
        return None

    def get_transform_by_name(self, name: str) -> type[TransformProtocol] | None:
        """Get transform plugin by name."""
        for transform in self._transforms:
            if transform.name == name:
                return transform
        return None

    def get_gate_by_name(self, name: str) -> type[GateProtocol] | None:
        """Get gate plugin by name."""
        for gate in self._gates:
            if gate.name == name:
                return gate
        return None

    def get_aggregation_by_name(self, name: str) -> type[AggregationProtocol] | None:
        """Get aggregation plugin by name."""
        for agg in self._aggregations:
            if agg.name == name:
                return agg
        return None

    def get_coalesce_by_name(self, name: str) -> type[CoalesceProtocol] | None:
        """Get coalesce plugin by name."""
        for coalesce in self._coalesces:
            if coalesce.name == name:
                return coalesce
        return None

    def get_sink_by_name(self, name: str) -> type[SinkProtocol] | None:
        """Get sink plugin by name."""
        for sink in self._sinks:
            if sink.name == name:
                return sink
        return None
