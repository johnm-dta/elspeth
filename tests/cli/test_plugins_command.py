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

    def test_plugin_registry_uses_plugin_info(self) -> None:
        """PLUGIN_REGISTRY entries are PluginInfo instances."""
        from elspeth.cli import PLUGIN_REGISTRY, PluginInfo

        for plugin_type, plugins in PLUGIN_REGISTRY.items():
            for plugin in plugins:
                assert isinstance(
                    plugin, PluginInfo
                ), f"Plugin in {plugin_type} is not PluginInfo: {plugin}"
                assert isinstance(plugin.name, str)
                assert isinstance(plugin.description, str)


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
