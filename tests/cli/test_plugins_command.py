"""Tests for elspeth plugins commands."""

import pytest
from typer.testing import CliRunner

runner = CliRunner()


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_plugin_info_creation(self) -> None:
        """PluginInfo can be created with name and description."""
        from elspeth.cli import PluginInfo

        plugin = PluginInfo(name="csv", description="Load rows from CSV files")

        assert plugin.name == "csv"
        assert plugin.description == "Load rows from CSV files"

    def test_plugin_info_is_frozen(self) -> None:
        """PluginInfo instances are immutable."""
        from elspeth.cli import PluginInfo

        plugin = PluginInfo(name="csv", description="Load rows from CSV files")

        with pytest.raises(AttributeError):
            plugin.name = "json"  # type: ignore[misc]

    def test_plugin_info_equality(self) -> None:
        """PluginInfo instances with same values are equal."""
        from elspeth.cli import PluginInfo

        plugin1 = PluginInfo(name="csv", description="Load rows from CSV files")
        plugin2 = PluginInfo(name="csv", description="Load rows from CSV files")

        assert plugin1 == plugin2

    def test_plugin_info_inequality(self) -> None:
        """PluginInfo instances with different values are not equal."""
        from elspeth.cli import PluginInfo

        plugin1 = PluginInfo(name="csv", description="Load rows from CSV files")
        plugin2 = PluginInfo(name="json", description="Load rows from JSON files")

        assert plugin1 != plugin2

    def test_plugin_info_hashable(self) -> None:
        """PluginInfo instances can be used as dict keys or in sets."""
        from elspeth.cli import PluginInfo

        plugin1 = PluginInfo(name="csv", description="Load rows from CSV files")
        plugin2 = PluginInfo(name="csv", description="Load rows from CSV files")

        # Should be hashable and produce same hash for equal instances
        plugin_set = {plugin1, plugin2}
        assert len(plugin_set) == 1

    def test_build_plugin_registry_returns_plugin_info(self) -> None:
        """_build_plugin_registry returns PluginInfo instances."""
        from elspeth.cli import PluginInfo, _build_plugin_registry

        registry = _build_plugin_registry()

        for plugin_type, plugins in registry.items():
            for plugin in plugins:
                assert isinstance(plugin, PluginInfo), f"Plugin in {plugin_type} is not PluginInfo: {plugin}"
                assert isinstance(plugin.name, str)
                assert isinstance(plugin.description, str)

    def test_build_plugin_registry_includes_all_discovered_plugins(self) -> None:
        """_build_plugin_registry includes all plugins from PluginManager."""
        from elspeth.cli import _build_plugin_registry, _get_plugin_manager

        registry = _build_plugin_registry()
        manager = _get_plugin_manager()

        # All discovered sources should be in registry
        source_names = {p.name for p in registry["source"]}
        for source_cls in manager.get_sources():
            assert source_cls.name in source_names, f"Missing source: {source_cls.name}"

        # All discovered transforms should be in registry
        transform_names = {p.name for p in registry["transform"]}
        for transform_cls in manager.get_transforms():
            assert transform_cls.name in transform_names, f"Missing transform: {transform_cls.name}"

        # All discovered sinks should be in registry
        sink_names = {p.name for p in registry["sink"]}
        for sink_cls in manager.get_sinks():
            assert sink_cls.name in sink_names, f"Missing sink: {sink_cls.name}"


class TestPluginsListCommand:
    """Tests for plugins list command."""

    def test_plugins_list_shows_sources(self) -> None:
        """plugins list shows available sources."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "csv" in result.stdout.lower()
        assert "json" in result.stdout.lower()

    def test_plugins_list_shows_sinks(self) -> None:
        """plugins list shows available sinks."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "database" in result.stdout.lower()

    def test_plugins_list_has_sections(self) -> None:
        """plugins list organizes by type."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "source" in result.stdout.lower()
        assert "sink" in result.stdout.lower()

    def test_plugins_list_type_filter(self) -> None:
        """plugins list --type filters by plugin type."""
        from elspeth.cli import app

        # Filter to sources only
        result = runner.invoke(app, ["plugins", "list", "--type", "source"])
        assert result.exit_code == 0
        assert "csv" in result.stdout.lower()
        # Should not show sinks
        assert "database" not in result.stdout.lower()

    def test_plugins_list_invalid_type(self) -> None:
        """plugins list --type with invalid type shows error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "invalid"])
        assert result.exit_code != 0
