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

    def _get_section_content(self, output: str, section_header: str) -> str:
        """Extract content for a section by header.

        Args:
            output: Full CLI output
            section_header: Header like "SOURCES:" or "TRANSFORMS:"

        Returns:
            Content from header until next section or end
        """
        lines = output.split("\n")
        in_section = False
        section_lines = []
        for line in lines:
            if section_header in line.upper():
                in_section = True
                continue
            if in_section:
                # Check if we hit another section header
                if line.strip() and line.strip().endswith(":") and line.strip()[:-1].isupper():
                    break
                section_lines.append(line)
        return "\n".join(section_lines)

    def test_plugins_list_shows_sources_in_sources_section(self) -> None:
        """plugins list shows csv/json under SOURCES section specifically."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Parse sources section specifically
        sources_section = self._get_section_content(result.stdout, "SOURCES:")
        assert "csv" in sources_section.lower(), f"Expected 'csv' in SOURCES section, got: {sources_section}"
        assert "json" in sources_section.lower(), f"Expected 'json' in SOURCES section, got: {sources_section}"

    def test_plugins_list_shows_sinks_in_sinks_section(self) -> None:
        """plugins list shows database under SINKS section specifically."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Parse sinks section specifically
        sinks_section = self._get_section_content(result.stdout, "SINKS:")
        assert "database" in sinks_section.lower(), f"Expected 'database' in SINKS section, got: {sinks_section}"

    def test_plugins_list_shows_transforms_in_transforms_section(self) -> None:
        """plugins list shows transforms under TRANSFORMS section specifically."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Parse transforms section specifically
        transforms_section = self._get_section_content(result.stdout, "TRANSFORMS:")
        assert "passthrough" in transforms_section.lower(), f"Expected 'passthrough' in TRANSFORMS section, got: {transforms_section}"

    def test_plugins_list_has_all_sections(self) -> None:
        """plugins list organizes by type with section headers."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "SOURCES:" in output_upper, "Missing SOURCES: section header"
        assert "TRANSFORMS:" in output_upper, "Missing TRANSFORMS: section header"
        assert "SINKS:" in output_upper, "Missing SINKS: section header"

    def test_plugins_list_type_filter_source(self) -> None:
        """plugins list --type source shows only sources."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "source"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "SOURCES:" in output_upper, "Missing SOURCES: header"
        assert "SINKS:" not in output_upper, "Should not show SINKS: when filtering to source"
        assert "TRANSFORMS:" not in output_upper, "Should not show TRANSFORMS: when filtering to source"

        # csv should appear in sources section
        assert "csv" in result.stdout.lower()

    def test_plugins_list_type_filter_transform(self) -> None:
        """plugins list --type transform shows only transforms."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "transform"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "TRANSFORMS:" in output_upper, "Missing TRANSFORMS: header"
        assert "SOURCES:" not in output_upper, "Should not show SOURCES: when filtering to transform"
        assert "SINKS:" not in output_upper, "Should not show SINKS: when filtering to transform"

        # passthrough should appear in transforms section
        assert "passthrough" in result.stdout.lower()

    def test_plugins_list_type_filter_sink(self) -> None:
        """plugins list --type sink shows only sinks."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "sink"])
        assert result.exit_code == 0

        output_upper = result.stdout.upper()
        assert "SINKS:" in output_upper, "Missing SINKS: header"
        assert "SOURCES:" not in output_upper, "Should not show SOURCES: when filtering to sink"
        assert "TRANSFORMS:" not in output_upper, "Should not show TRANSFORMS: when filtering to sink"

    def test_plugins_list_invalid_type_error_message(self) -> None:
        """plugins list --type with invalid type shows specific error."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "invalid"])
        assert result.exit_code == 1, f"Expected exit code 1 for invalid type, got {result.exit_code}"
        # Should show the invalid type name
        output = result.stdout.lower() + (result.stderr or "").lower()
        assert "invalid" in output, f"Expected 'invalid' in error message, got: {output}"
        # Should mention valid types
        assert "valid types" in output or "source" in output, f"Expected mention of valid types, got: {output}"
