# src/elspeth/plugins/manager.py
"""Plugin manager for discovery, registration, and lifecycle.

Uses pluggy for hook-based plugin registration.
"""

from dataclasses import dataclass
from typing import Any

import pluggy

from elspeth.plugins.enums import Determinism, NodeType
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


@dataclass(frozen=True)
class PluginSpec:
    """Registration record for a plugin.

    Captures metadata that Phase 3 stores in Landscape nodes table.
    Frozen for immutability - plugin specs shouldn't change after creation.
    """

    name: str
    node_type: NodeType
    version: str
    determinism: Determinism
    input_schema_hash: str | None = None
    output_schema_hash: str | None = None

    @classmethod
    def from_plugin(cls, plugin_cls: type, node_type: NodeType) -> "PluginSpec":
        """Create spec from plugin class.

        Args:
            plugin_cls: Plugin class to extract metadata from
            node_type: Type of node this plugin represents

        Returns:
            PluginSpec with extracted metadata
        """
        return cls(
            name=getattr(plugin_cls, "name", plugin_cls.__name__),
            node_type=node_type,
            version=getattr(plugin_cls, "plugin_version", "0.0.0"),
            determinism=getattr(plugin_cls, "determinism", Determinism.DETERMINISTIC),
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

        # Caches - map name to plugin class for duplicate detection
        self._sources: dict[str, type[SourceProtocol]] = {}
        self._transforms: dict[str, type[TransformProtocol]] = {}
        self._gates: dict[str, type[GateProtocol]] = {}
        self._aggregations: dict[str, type[AggregationProtocol]] = {}
        self._coalesces: dict[str, type[CoalesceProtocol]] = {}
        self._sinks: dict[str, type[SinkProtocol]] = {}

    def register(self, plugin: Any) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance implementing hook methods
        """
        self._pm.register(plugin)
        self._refresh_caches()

    def _refresh_caches(self) -> None:
        """Refresh plugin caches from hooks.

        Raises:
            ValueError: If a plugin with the same name and type is already registered
        """
        # Collect all plugins first, then check for duplicates
        new_sources: dict[str, type[SourceProtocol]] = {}
        new_transforms: dict[str, type[TransformProtocol]] = {}
        new_gates: dict[str, type[GateProtocol]] = {}
        new_aggregations: dict[str, type[AggregationProtocol]] = {}
        new_coalesces: dict[str, type[CoalesceProtocol]] = {}
        new_sinks: dict[str, type[SinkProtocol]] = {}

        # Collect from all registered plugins with duplicate detection
        for sources in self._pm.hook.elspeth_get_source():
            for cls in sources:
                name = getattr(cls, "name", cls.__name__)
                if name in new_sources:
                    raise ValueError(
                        f"Duplicate source plugin name: '{name}'. "
                        f"Already registered by {new_sources[name].__name__}"
                    )
                new_sources[name] = cls

        for transforms in self._pm.hook.elspeth_get_transforms():
            for cls in transforms:
                name = getattr(cls, "name", cls.__name__)
                if name in new_transforms:
                    raise ValueError(
                        f"Duplicate transform plugin name: '{name}'. "
                        f"Already registered by {new_transforms[name].__name__}"
                    )
                new_transforms[name] = cls

        for gates in self._pm.hook.elspeth_get_gates():
            for cls in gates:
                name = getattr(cls, "name", cls.__name__)
                if name in new_gates:
                    raise ValueError(
                        f"Duplicate gate plugin name: '{name}'. "
                        f"Already registered by {new_gates[name].__name__}"
                    )
                new_gates[name] = cls

        for aggs in self._pm.hook.elspeth_get_aggregations():
            for cls in aggs:
                name = getattr(cls, "name", cls.__name__)
                if name in new_aggregations:
                    raise ValueError(
                        f"Duplicate aggregation plugin name: '{name}'. "
                        f"Already registered by {new_aggregations[name].__name__}"
                    )
                new_aggregations[name] = cls

        for coalesces in self._pm.hook.elspeth_get_coalesces():
            for cls in coalesces:
                name = getattr(cls, "name", cls.__name__)
                if name in new_coalesces:
                    raise ValueError(
                        f"Duplicate coalesce plugin name: '{name}'. "
                        f"Already registered by {new_coalesces[name].__name__}"
                    )
                new_coalesces[name] = cls

        for sinks in self._pm.hook.elspeth_get_sinks():
            for cls in sinks:
                name = getattr(cls, "name", cls.__name__)
                if name in new_sinks:
                    raise ValueError(
                        f"Duplicate sink plugin name: '{name}'. "
                        f"Already registered by {new_sinks[name].__name__}"
                    )
                new_sinks[name] = cls

        # All validated, update caches
        self._sources = new_sources
        self._transforms = new_transforms
        self._gates = new_gates
        self._aggregations = new_aggregations
        self._coalesces = new_coalesces
        self._sinks = new_sinks

    # === Getters ===

    def get_sources(self) -> list[type[SourceProtocol]]:
        """Get all registered source plugins."""
        return list(self._sources.values())

    def get_transforms(self) -> list[type[TransformProtocol]]:
        """Get all registered transform plugins."""
        return list(self._transforms.values())

    def get_gates(self) -> list[type[GateProtocol]]:
        """Get all registered gate plugins."""
        return list(self._gates.values())

    def get_aggregations(self) -> list[type[AggregationProtocol]]:
        """Get all registered aggregation plugins."""
        return list(self._aggregations.values())

    def get_coalesces(self) -> list[type[CoalesceProtocol]]:
        """Get all registered coalesce plugins."""
        return list(self._coalesces.values())

    def get_sinks(self) -> list[type[SinkProtocol]]:
        """Get all registered sink plugins."""
        return list(self._sinks.values())

    # === Lookup by name ===

    def get_source_by_name(self, name: str) -> type[SourceProtocol] | None:
        """Get source plugin by name."""
        return self._sources.get(name)

    def get_transform_by_name(self, name: str) -> type[TransformProtocol] | None:
        """Get transform plugin by name."""
        return self._transforms.get(name)

    def get_gate_by_name(self, name: str) -> type[GateProtocol] | None:
        """Get gate plugin by name."""
        return self._gates.get(name)

    def get_aggregation_by_name(self, name: str) -> type[AggregationProtocol] | None:
        """Get aggregation plugin by name."""
        return self._aggregations.get(name)

    def get_coalesce_by_name(self, name: str) -> type[CoalesceProtocol] | None:
        """Get coalesce plugin by name."""
        return self._coalesces.get(name)

    def get_sink_by_name(self, name: str) -> type[SinkProtocol] | None:
        """Get sink plugin by name."""
        return self._sinks.get(name)
