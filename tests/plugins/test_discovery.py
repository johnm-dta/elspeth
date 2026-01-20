"""Tests for dynamic plugin discovery."""

from pathlib import Path

from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform
from elspeth.plugins.discovery import discover_plugins_in_directory


class TestDiscoverPlugins:
    """Test plugin discovery from directories."""

    def test_discovers_csv_source(self) -> None:
        """Verify CSVSource is discovered in sources directory."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        names = [cls.name for cls in discovered]
        assert "csv" in names, f"Expected 'csv' in {names}"

    def test_discovers_passthrough_transform(self) -> None:
        """Verify PassThrough is discovered in transforms directory."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        transforms_dir = plugins_root / "transforms"

        discovered = discover_plugins_in_directory(transforms_dir, BaseTransform)

        names = [cls.name for cls in discovered]
        assert "passthrough" in names, f"Expected 'passthrough' in {names}"

    def test_discovers_csv_sink(self) -> None:
        """Verify CSVSink is discovered in sinks directory."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sinks_dir = plugins_root / "sinks"

        discovered = discover_plugins_in_directory(sinks_dir, BaseSink)

        names = [cls.name for cls in discovered]
        assert "csv" in names, f"Expected 'csv' in {names}"

    def test_excludes_non_plugin_files(self) -> None:
        """Verify __init__.py and base.py are not scanned for plugins."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        # Should not crash or include base classes
        for cls in discovered:
            assert hasattr(cls, "name"), f"{cls} has no name attribute"
            assert cls.name != "", f"{cls} has empty name"

    def test_skips_abstract_classes(self) -> None:
        """Verify abstract base classes are not included."""
        plugins_root = Path(__file__).parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        class_names = [cls.__name__ for cls in discovered]
        assert "BaseSource" not in class_names


class TestDiscoverAllPlugins:
    """Test discovery across all plugin directories."""

    def test_discover_all_sources(self) -> None:
        """Verify all sources are discovered including azure."""
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        source_names = [cls.name for cls in discovered["sources"]]
        assert "csv" in source_names
        assert "json" in source_names
        assert "null" in source_names
        # Azure blob source lives in plugins/azure/
        assert "azure_blob" in source_names

    def test_discover_all_transforms(self) -> None:
        """Verify all transforms are discovered including llm/ and transforms/azure/."""
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        transform_names = [cls.name for cls in discovered["transforms"]]
        assert "passthrough" in transform_names
        assert "field_mapper" in transform_names
        # LLM transforms live in plugins/llm/ - verify ALL are discovered
        assert "azure_llm" in transform_names, f"Missing azure_llm in {transform_names}"
        assert "openrouter_llm" in transform_names, f"Missing openrouter_llm in {transform_names}"
        assert "azure_batch_llm" in transform_names, f"Missing azure_batch_llm in {transform_names}"
        # Azure transforms live in plugins/transforms/azure/ (subdirectory!)
        assert "azure_content_safety" in transform_names, f"Missing azure_content_safety in {transform_names}"
        assert "azure_prompt_shield" in transform_names, f"Missing azure_prompt_shield in {transform_names}"

    def test_discover_all_sinks(self) -> None:
        """Verify all sinks are discovered including azure."""
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        sink_names = [cls.name for cls in discovered["sinks"]]
        assert "csv" in sink_names
        assert "json" in sink_names
        assert "database" in sink_names

    def test_no_duplicate_names_within_type(self) -> None:
        """Verify no duplicate plugin names within same type."""
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        for plugin_type, plugins in discovered.items():
            names = [cls.name for cls in plugins]
            assert len(names) == len(set(names)), f"Duplicate names in {plugin_type}: {names}"

    def test_discovery_matches_hookimpl_counts(self) -> None:
        """Verify discovery finds same plugins as static hookimpls.

        This is a MIGRATION TEST that compares discovery to the old static hookimpls.
        It will be CONVERTED to static count assertions in Task 9 after hookimpl
        files are deleted.

        IMPORTANT: This test imports from hookimpl files. When those files are
        deleted in Task 9, this test MUST be updated - see Task 9 Step 4.
        """
        from elspeth.plugins.discovery import discover_all_plugins
        from elspeth.plugins.sinks.hookimpl import builtin_sinks
        from elspeth.plugins.sources.hookimpl import builtin_sources
        from elspeth.plugins.transforms.hookimpl import builtin_transforms

        # Get counts from old static hookimpls
        old_source_count = len(builtin_sources.elspeth_get_source())
        old_transform_count = len(builtin_transforms.elspeth_get_transforms())
        old_sink_count = len(builtin_sinks.elspeth_get_sinks())

        # Get counts from new discovery
        discovered = discover_all_plugins()

        assert len(discovered["sources"]) == old_source_count, (
            f"Source count mismatch: discovery={len(discovered['sources'])}, hookimpl={old_source_count}"
        )
        assert len(discovered["transforms"]) == old_transform_count, (
            f"Transform count mismatch: discovery={len(discovered['transforms'])}, hookimpl={old_transform_count}"
        )
        assert len(discovered["sinks"]) == old_sink_count, (
            f"Sink count mismatch: discovery={len(discovered['sinks'])}, hookimpl={old_sink_count}"
        )


class TestGetPluginDescription:
    """Test docstring extraction for plugin descriptions."""

    def test_extracts_first_line_of_docstring(self) -> None:
        """Verify first docstring line is extracted."""
        from elspeth.plugins.discovery import get_plugin_description
        from elspeth.plugins.transforms.passthrough import PassThrough

        description = get_plugin_description(PassThrough)

        # PassThrough's class docstring starts with "Pass rows through unchanged."
        assert description == "Pass rows through unchanged."

    def test_handles_missing_docstring(self) -> None:
        """Verify fallback for classes without docstrings."""
        from elspeth.plugins.discovery import get_plugin_description

        class NoDocPlugin:
            name = "no_doc"

        description = get_plugin_description(NoDocPlugin)

        assert "no_doc" in description.lower()

    def test_strips_whitespace(self) -> None:
        """Verify whitespace is stripped from description."""
        from elspeth.plugins.discovery import get_plugin_description

        class WhitespaceDocPlugin:
            """Lots of whitespace here."""

            name = "whitespace"

        description = get_plugin_description(WhitespaceDocPlugin)

        assert description == "Lots of whitespace here."


class TestCreateDynamicHookimpl:
    """Test dynamic hookimpl generation for pluggy."""

    def test_creates_hookimpl_with_correct_method(self) -> None:
        """Verify hookimpl has correct method name."""
        from elspeth.plugins.discovery import create_dynamic_hookimpl

        class FakePlugin:
            name = "fake"

        hookimpl_obj = create_dynamic_hookimpl([FakePlugin], "elspeth_get_source")

        assert hasattr(hookimpl_obj, "elspeth_get_source")

    def test_hookimpl_returns_plugin_list(self) -> None:
        """Verify hookimpl method returns the plugin classes."""
        from elspeth.plugins.discovery import create_dynamic_hookimpl

        class FakePlugin1:
            name = "fake1"

        class FakePlugin2:
            name = "fake2"

        hookimpl_obj = create_dynamic_hookimpl([FakePlugin1, FakePlugin2], "elspeth_get_source")

        result = hookimpl_obj.elspeth_get_source()
        assert result == [FakePlugin1, FakePlugin2]

    def test_hookimpl_integrates_with_pluggy(self) -> None:
        """Verify dynamic hookimpl works with PluginManager."""
        from collections.abc import Iterator
        from typing import Any

        from elspeth.plugins.discovery import create_dynamic_hookimpl
        from elspeth.plugins.manager import PluginManager

        class TestSource:
            name = "test_dynamic"
            output_schema = None
            node_id = None
            determinism = "deterministic"
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[Any]:
                return iter([])

            def close(self) -> None:
                pass

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

        hookimpl_obj = create_dynamic_hookimpl([TestSource], "elspeth_get_source")

        manager = PluginManager()
        manager.register(hookimpl_obj)

        source = manager.get_source_by_name("test_dynamic")
        assert source is TestSource
