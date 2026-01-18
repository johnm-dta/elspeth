# src/elspeth/plugins/manager.py
"""Plugin manager for discovery, registration, and lifecycle.

Uses pluggy for hook-based plugin registration.
"""

from dataclasses import dataclass
from typing import Any

import pluggy

from elspeth.contracts import Determinism, NodeType
from elspeth.core.canonical import stable_hash
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


def _schema_hash(schema_cls: Any) -> str | None:
    """Compute stable hash for a schema class.

    Hashes the schema's field names and types to detect compatibility changes.

    Args:
        schema_cls: A PluginSchema subclass, or None

    Returns:
        SHA-256 hex digest of field names/types, or None if no schema
    """
    if schema_cls is None:
        return None

    # Use Pydantic model_fields for accurate field introspection
    # This is legitimate type checking at a plugin boundary
    if not hasattr(schema_cls, "model_fields"):
        return None

    # Build deterministic representation
    fields_repr = {
        name: str(field.annotation) for name, field in schema_cls.model_fields.items()
    }
    return stable_hash(fields_repr)


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
        """Create spec from plugin class with schema hashes.

        Required attributes (will raise if missing):
        - name: str
        - plugin_version: str

        Optional attributes (have legitimate defaults):
        - determinism: defaults to DETERMINISTIC
        - input_schema: defaults to None
        - output_schema: defaults to None

        Args:
            plugin_cls: Plugin class to extract metadata from
            node_type: Type of node this plugin represents

        Returns:
            PluginSpec with extracted metadata

        Raises:
            ValueError: If plugin is missing required 'name' or 'plugin_version' attributes
        """
        # Required: name
        try:
            name = plugin_cls.name  # type: ignore[attr-defined]
        except AttributeError:
            raise ValueError(
                f"Plugin {plugin_cls.__name__} must define 'name' attribute. "
                f"Add: name = 'your_plugin_name' to the class."
            ) from None

        # Required: plugin_version
        try:
            version = plugin_cls.plugin_version  # type: ignore[attr-defined]
        except AttributeError:
            raise ValueError(
                f"Plugin {plugin_cls.__name__} must define 'plugin_version' attribute. "
                f"Add: plugin_version = '1.0.0' to the class."
            ) from None

        # All protocols define determinism with defaults in base classes.
        determinism = plugin_cls.determinism  # type: ignore[attr-defined]

        # Schemas vary by plugin type: sources have only output_schema,
        # sinks have only input_schema, transforms have both.
        input_schema = getattr(plugin_cls, "input_schema", None)
        output_schema = getattr(plugin_cls, "output_schema", None)

        return cls(
            name=name,
            node_type=node_type,
            version=version,
            determinism=determinism,
            input_schema_hash=_schema_hash(input_schema),
            output_schema_hash=_schema_hash(output_schema),
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

    def register_builtin_plugins(self) -> None:
        """Register all built-in plugin hook implementers.

        Call this once at startup to make built-in plugins discoverable.
        """
        from elspeth.plugins.sinks.hookimpl import builtin_sinks
        from elspeth.plugins.sources.hookimpl import builtin_sources
        from elspeth.plugins.transforms.hookimpl import builtin_transforms

        self.register(builtin_sources)
        self.register(builtin_transforms)
        self.register(builtin_sinks)

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
                name = cls.name
                if name in new_sources:
                    raise ValueError(
                        f"Duplicate source plugin name: '{name}'. "
                        f"Already registered by {new_sources[name].__name__}"
                    )
                new_sources[name] = cls

        for transforms in self._pm.hook.elspeth_get_transforms():
            for cls in transforms:
                name = cls.name
                if name in new_transforms:
                    raise ValueError(
                        f"Duplicate transform plugin name: '{name}'. "
                        f"Already registered by {new_transforms[name].__name__}"
                    )
                new_transforms[name] = cls

        for gates in self._pm.hook.elspeth_get_gates():
            for cls in gates:
                name = cls.name
                if name in new_gates:
                    raise ValueError(
                        f"Duplicate gate plugin name: '{name}'. "
                        f"Already registered by {new_gates[name].__name__}"
                    )
                new_gates[name] = cls

        for aggs in self._pm.hook.elspeth_get_aggregations():
            for cls in aggs:
                name = cls.name
                if name in new_aggregations:
                    raise ValueError(
                        f"Duplicate aggregation plugin name: '{name}'. "
                        f"Already registered by {new_aggregations[name].__name__}"
                    )
                new_aggregations[name] = cls

        for coalesces in self._pm.hook.elspeth_get_coalesces():
            for cls in coalesces:
                name = cls.name
                if name in new_coalesces:
                    raise ValueError(
                        f"Duplicate coalesce plugin name: '{name}'. "
                        f"Already registered by {new_coalesces[name].__name__}"
                    )
                new_coalesces[name] = cls

        for sinks in self._pm.hook.elspeth_get_sinks():
            for cls in sinks:
                name = cls.name
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
