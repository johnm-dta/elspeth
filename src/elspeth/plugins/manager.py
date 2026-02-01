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
    GateProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)
from elspeth.plugins.validation import PluginConfigValidator


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
        self._sinks: dict[str, type[SinkProtocol]] = {}

        # Config validator
        self._validator = PluginConfigValidator()

    def register_builtin_plugins(self) -> None:
        """Discover and register all built-in plugins.

        Scans plugin directories for classes inheriting from base classes
        and registers them via dynamically-generated hookimpls.

        Call this once at startup to make built-in plugins discoverable.

        NOTE: Gates are NOT registered here. Per docs/contracts/plugin-protocol.md,
        gates are config-driven system operations handled by the engine, not plugins.
        """
        from elspeth.plugins.discovery import create_dynamic_hookimpl, discover_all_plugins

        discovered = discover_all_plugins()

        # Register each plugin type via dynamic hookimpls
        # (gates excluded - they're system operations, not plugins)
        self.register(create_dynamic_hookimpl(discovered["sources"], "elspeth_get_source"))
        self.register(create_dynamic_hookimpl(discovered["transforms"], "elspeth_get_transforms"))
        self.register(create_dynamic_hookimpl(discovered["sinks"], "elspeth_get_sinks"))

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
        new_sinks: dict[str, type[SinkProtocol]] = {}

        # Collect from all registered plugins with duplicate detection
        for sources in self._pm.hook.elspeth_get_source():
            for cls in sources:
                name = cls.name
                if name in new_sources:
                    raise ValueError(f"Duplicate source plugin name: '{name}'. Already registered by {new_sources[name].__name__}")
                new_sources[name] = cls

        for transforms in self._pm.hook.elspeth_get_transforms():
            for cls in transforms:
                name = cls.name
                if name in new_transforms:
                    raise ValueError(f"Duplicate transform plugin name: '{name}'. Already registered by {new_transforms[name].__name__}")
                new_transforms[name] = cls

        for gates in self._pm.hook.elspeth_get_gates():
            for cls in gates:
                name = cls.name
                if name in new_gates:
                    raise ValueError(f"Duplicate gate plugin name: '{name}'. Already registered by {new_gates[name].__name__}")
                new_gates[name] = cls

        for sinks in self._pm.hook.elspeth_get_sinks():
            for cls in sinks:
                name = cls.name
                if name in new_sinks:
                    raise ValueError(f"Duplicate sink plugin name: '{name}'. Already registered by {new_sinks[name].__name__}")
                new_sinks[name] = cls

        # All validated, update caches
        self._sources = new_sources
        self._transforms = new_transforms
        self._gates = new_gates
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

    def get_sinks(self) -> list[type[SinkProtocol]]:
        """Get all registered sink plugins."""
        return list(self._sinks.values())

    # === Lookup by name ===

    def get_source_by_name(self, name: str) -> type[SourceProtocol]:
        """Get source plugin class by name.

        Args:
            name: Plugin name to look up

        Returns:
            Source plugin class

        Raises:
            ValueError: If plugin not found (configuration bug, should crash)
        """
        if name in self._sources:
            return self._sources[name]

        available = sorted(self._sources.keys())
        raise ValueError(f"Unknown source plugin: {name}. Available source plugins: {available}")

    def get_transform_by_name(self, name: str) -> type[TransformProtocol]:
        """Get transform plugin class by name.

        Args:
            name: Plugin name to look up

        Returns:
            Transform plugin class

        Raises:
            ValueError: If plugin not found (configuration bug, should crash)
        """
        if name in self._transforms:
            return self._transforms[name]

        available = sorted(self._transforms.keys())
        raise ValueError(f"Unknown transform plugin: {name}. Available transform plugins: {available}")

    def get_gate_by_name(self, name: str) -> type[GateProtocol]:
        """Get gate plugin class by name.

        Args:
            name: Plugin name to look up

        Returns:
            Gate plugin class

        Raises:
            ValueError: If plugin not found (configuration bug, should crash)
        """
        if name in self._gates:
            return self._gates[name]

        available = sorted(self._gates.keys())
        raise ValueError(f"Unknown gate plugin: {name}. Available gate plugins: {available}")

    def get_sink_by_name(self, name: str) -> type[SinkProtocol]:
        """Get sink plugin class by name.

        Args:
            name: Plugin name to look up

        Returns:
            Sink plugin class

        Raises:
            ValueError: If plugin not found (configuration bug, should crash)
        """
        if name in self._sinks:
            return self._sinks[name]

        available = sorted(self._sinks.keys())
        raise ValueError(f"Unknown sink plugin: {name}. Available sink plugins: {available}")

    # === Plugin creation with validation ===

    def create_source(self, source_type: str, config: dict[str, Any]) -> SourceProtocol:
        """Create source plugin instance with validated config.

        Args:
            source_type: Plugin type name (e.g., "csv", "json")
            config: Plugin configuration dict

        Returns:
            Instantiated source plugin

        Raises:
            ValueError: If config is invalid or plugin type not found
        """
        # Validate config first
        errors = self._validator.validate_source_config(source_type, config)
        if errors:
            # Format errors into readable message with field names
            error_lines = [f"  - {err.field}: {err.message}" for err in errors]
            error_msg = f"Invalid configuration for source '{source_type}':\n" + "\n".join(error_lines)
            raise ValueError(error_msg)

        # Get plugin class
        plugin_cls = self.get_source_by_name(source_type)

        # Instantiate with validated config
        return plugin_cls(config)

    def create_transform(self, transform_type: str, config: dict[str, Any]) -> TransformProtocol:
        """Create transform plugin instance with validated config.

        Args:
            transform_type: Plugin type name (e.g., "passthrough", "field_mapper")
            config: Plugin configuration dict

        Returns:
            Instantiated transform plugin

        Raises:
            ValueError: If config is invalid or plugin type not found
        """
        # Validate config first
        errors = self._validator.validate_transform_config(transform_type, config)
        if errors:
            # Format errors into readable message with field names
            error_lines = [f"  - {err.field}: {err.message}" for err in errors]
            error_msg = f"Invalid configuration for transform '{transform_type}':\n" + "\n".join(error_lines)
            raise ValueError(error_msg)

        # Get plugin class
        plugin_cls = self.get_transform_by_name(transform_type)

        # Instantiate with validated config
        return plugin_cls(config)

    def create_gate(self, gate_type: str, config: dict[str, Any]) -> GateProtocol:
        """Create gate plugin instance with validated config.

        Args:
            gate_type: Plugin type name
            config: Plugin configuration dict

        Returns:
            Instantiated gate plugin

        Raises:
            ValueError: If config is invalid or plugin type not found
        """
        # Validate config first
        errors = self._validator.validate_gate_config(gate_type, config)
        if errors:
            # Format errors into readable message with field names
            error_lines = [f"  - {err.field}: {err.message}" for err in errors]
            error_msg = f"Invalid configuration for gate '{gate_type}':\n" + "\n".join(error_lines)
            raise ValueError(error_msg)

        # Get plugin class
        plugin_cls = self.get_gate_by_name(gate_type)

        # Instantiate with validated config
        return plugin_cls(config)

    def create_sink(self, sink_type: str, config: dict[str, Any]) -> SinkProtocol:
        """Create sink plugin instance with validated config.

        Args:
            sink_type: Plugin type name (e.g., "csv", "json")
            config: Plugin configuration dict

        Returns:
            Instantiated sink plugin

        Raises:
            ValueError: If config is invalid or plugin type not found
        """
        # Validate config first
        errors = self._validator.validate_sink_config(sink_type, config)
        if errors:
            # Format errors into readable message with field names
            error_lines = [f"  - {err.field}: {err.message}" for err in errors]
            error_msg = f"Invalid configuration for sink '{sink_type}':\n" + "\n".join(error_lines)
            raise ValueError(error_msg)

        # Get plugin class
        plugin_cls = self.get_sink_by_name(sink_type)

        # Instantiate with validated config
        return plugin_cls(config)
