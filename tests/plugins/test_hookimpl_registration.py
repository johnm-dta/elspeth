"""Tests for plugin hook implementations.

Verifies that built-in plugins are discoverable via pluggy hooks
after calling manager.register_builtin_plugins().
"""

import pytest

from elspeth.plugins.manager import PluginManager


class TestBuiltinPluginDiscovery:
    """Verify built-in plugins are discoverable via hooks."""

    def test_builtin_sources_discoverable(self) -> None:
        """Built-in source plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        # CSVSource and JSONSource should be discoverable
        sources = manager.get_sources()
        source_names = [s.name for s in sources]

        assert "csv" in source_names
        assert "json" in source_names

    def test_builtin_transforms_discoverable(self) -> None:
        """Built-in transform plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        transforms = manager.get_transforms()
        transform_names = [t.name for t in transforms]

        assert "passthrough" in transform_names
        assert "field_mapper" in transform_names

    def test_builtin_gates_discoverable(self) -> None:
        """Built-in gate plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        gates = manager.get_gates()
        gate_names = [g.name for g in gates]

        assert "threshold_gate" in gate_names
        assert "field_match_gate" in gate_names
        assert "filter_gate" in gate_names

    def test_builtin_sinks_discoverable(self) -> None:
        """Built-in sink plugins are registered via hookimpl."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        sinks = manager.get_sinks()
        sink_names = [s.name for s in sinks]

        assert "csv" in sink_names
        assert "json" in sink_names
        assert "database" in sink_names

    def test_plugins_retrievable_by_name(self) -> None:
        """Built-in plugins can be retrieved by name after registration."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        # Sources
        assert manager.get_source_by_name("csv") is not None
        assert manager.get_source_by_name("json") is not None

        # Transforms
        assert manager.get_transform_by_name("passthrough") is not None
        assert manager.get_transform_by_name("field_mapper") is not None

        # Gates
        assert manager.get_gate_by_name("threshold_gate") is not None
        assert manager.get_gate_by_name("field_match_gate") is not None
        assert manager.get_gate_by_name("filter_gate") is not None

        # Sinks
        assert manager.get_sink_by_name("csv") is not None
        assert manager.get_sink_by_name("json") is not None
        assert manager.get_sink_by_name("database") is not None

    def test_register_builtin_plugins_idempotent(self) -> None:
        """Calling register_builtin_plugins twice raises duplicate error."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        # Second registration should raise because plugins are already registered
        # pluggy raises "Plugin name already registered" for duplicate registrations
        with pytest.raises(ValueError, match="already registered"):
            manager.register_builtin_plugins()
