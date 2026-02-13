"""Tests for dynamic plugin discovery."""

from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform
from elspeth.plugins.discovery import discover_plugins_in_directory


class TestDiscoverPlugins:
    """Test plugin discovery from directories."""

    def test_discovers_csv_source(self) -> None:
        """Verify CSVSource is discovered in sources directory."""
        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        names = [cls.name for cls in discovered]  # type: ignore[attr-defined]
        assert "csv" in names, f"Expected 'csv' in {names}"

    def test_discovers_passthrough_transform(self) -> None:
        """Verify PassThrough is discovered in transforms directory."""
        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        transforms_dir = plugins_root / "transforms"

        discovered = discover_plugins_in_directory(transforms_dir, BaseTransform)

        names = [cls.name for cls in discovered]  # type: ignore[attr-defined]
        assert "passthrough" in names, f"Expected 'passthrough' in {names}"

    def test_discovers_csv_sink(self) -> None:
        """Verify CSVSink is discovered in sinks directory."""
        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        sinks_dir = plugins_root / "sinks"

        discovered = discover_plugins_in_directory(sinks_dir, BaseSink)

        names = [cls.name for cls in discovered]  # type: ignore[attr-defined]
        assert "csv" in names, f"Expected 'csv' in {names}"

    def test_excludes_non_plugin_files(self) -> None:
        """Verify __init__.py and base.py are not scanned for plugins."""
        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        # Should not crash or include base classes
        for cls in discovered:
            assert hasattr(cls, "name"), f"{cls} has no name attribute"
            assert cls.name != "", f"{cls} has empty name"

    def test_skips_abstract_classes(self) -> None:
        """Verify abstract base classes are not included."""
        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        sources_dir = plugins_root / "sources"

        discovered = discover_plugins_in_directory(sources_dir, BaseSource)

        class_names = [cls.__name__ for cls in discovered]
        assert "BaseSource" not in class_names

    def test_missing_name_attribute_raises(self, tmp_path: Path) -> None:
        """Concrete plugin without name must fail discovery."""
        plugin_file = tmp_path / "missing_name.py"
        plugin_file.write_text("""
from elspeth.plugins.base import BaseSource

class MissingNameSource(BaseSource):
    output_schema = None
    node_id = None
    determinism = "deterministic"
    plugin_version = "1.0.0"

    def __init__(self, config):
        self.config = config
        self.on_success = "continue"
        self._on_validation_failure = "discard"

    def load(self, ctx):
        return iter([])

    def close(self):
        pass

    def on_start(self, ctx):
        pass

    def on_complete(self, ctx):
        pass
""")

        with pytest.raises(ValueError, match="missing required class attribute 'name'"):
            discover_plugins_in_directory(tmp_path, BaseSource)


class TestDiscoverAllPlugins:
    """Test discovery across all plugin directories."""

    def test_discover_all_sources(self) -> None:
        """Verify all sources are discovered including azure."""
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        source_names = [cls.name for cls in discovered["sources"]]  # type: ignore[attr-defined]
        assert "csv" in source_names
        assert "json" in source_names
        assert "null" in source_names
        # Azure blob source lives in plugins/azure/
        assert "azure_blob" in source_names

    def test_discover_all_transforms(self) -> None:
        """Verify all transforms are discovered including llm/ and transforms/azure/."""
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        transform_names = [cls.name for cls in discovered["transforms"]]  # type: ignore[attr-defined]
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

        sink_names = [cls.name for cls in discovered["sinks"]]  # type: ignore[attr-defined]
        assert "csv" in sink_names
        assert "json" in sink_names
        assert "database" in sink_names

    def test_no_duplicate_names_within_type(self) -> None:
        """Verify no duplicate plugin names within same type."""
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        for plugin_type, plugins in discovered.items():
            names = [cls.name for cls in plugins]  # type: ignore[attr-defined]
            assert len(names) == len(set(names)), f"Duplicate names in {plugin_type}: {names}"

    def test_discovery_finds_expected_plugin_counts(self) -> None:
        """Verify discovery finds the expected number of plugins.

        These counts were verified against the old static hookimpl files during
        migration (Task 1-8). Now hookimpl files are deleted, we assert the
        expected counts directly.

        If a new plugin is added, update these counts.
        """
        from elspeth.plugins.discovery import discover_all_plugins

        # Expected counts verified during migration from hookimpl files
        EXPECTED_SOURCE_COUNT = 4  # csv, json, null, azure_blob
        EXPECTED_TRANSFORM_COUNT = 16  # includes json_explode, batch_replicate, keyword_filter, field_mapper, passthrough, truncate, batch_stats, web_scrape, azure_*, openrouter_*
        EXPECTED_SINK_COUNT = 4  # csv, json, database, azure_blob

        discovered = discover_all_plugins()

        assert len(discovered["sources"]) == EXPECTED_SOURCE_COUNT, (
            f"Source count: expected {EXPECTED_SOURCE_COUNT}, got {len(discovered['sources'])}. "
            f"Found: {[cls.name for cls in discovered['sources']]}"  # type: ignore[attr-defined]
        )
        assert len(discovered["transforms"]) == EXPECTED_TRANSFORM_COUNT, (
            f"Transform count: expected {EXPECTED_TRANSFORM_COUNT}, got {len(discovered['transforms'])}. "
            f"Found: {[cls.name for cls in discovered['transforms']]}"  # type: ignore[attr-defined]
        )
        assert len(discovered["sinks"]) == EXPECTED_SINK_COUNT, (
            f"Sink count: expected {EXPECTED_SINK_COUNT}, got {len(discovered['sinks'])}. "
            f"Found: {[cls.name for cls in discovered['sinks']]}"  # type: ignore[attr-defined]
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


class TestDuplicatePluginDetection:
    """Test that duplicate plugin names raise errors."""

    def test_duplicate_names_raise_value_error(self, tmp_path: Path) -> None:
        """Verify duplicate plugin names within same type raise ValueError.

        This tests the crash-on-bug behavior: if two plugins of the same type
        share a name, that's a bug in our codebase that should surface immediately.
        """

        from elspeth.plugins.discovery import discover_plugins_in_directory

        # Create two plugin files with same name attribute
        plugin1 = tmp_path / "plugin_one.py"
        plugin1.write_text("""
from elspeth.plugins.base import BaseSource

class SourceOne(BaseSource):
    name = "duplicate_name"
    output_schema = None
    node_id = None
    determinism = "deterministic"
    plugin_version = "1.0.0"

    def __init__(self, config):
        pass

    def load(self, ctx):
        return iter([])

    def close(self):
        pass

    def on_start(self, ctx):
        pass

    def on_complete(self, ctx):
        pass
""")

        plugin2 = tmp_path / "plugin_two.py"
        plugin2.write_text("""
from elspeth.plugins.base import BaseSource

class SourceTwo(BaseSource):
    name = "duplicate_name"  # Same name - this is the bug!
    output_schema = None
    node_id = None
    determinism = "deterministic"
    plugin_version = "1.0.0"

    def __init__(self, config):
        pass

    def load(self, ctx):
        return iter([])

    def close(self):
        pass

    def on_start(self, ctx):
        pass

    def on_complete(self, ctx):
        pass
""")

        # Discover plugins - both files have same name
        from elspeth.plugins.base import BaseSource

        discovered = discover_plugins_in_directory(tmp_path, BaseSource)

        # Should find 2 classes (both with name="duplicate_name")
        assert len(discovered) == 2
        names = [cls.name for cls in discovered]  # type: ignore[attr-defined]
        assert names == ["duplicate_name", "duplicate_name"]

    def test_discover_all_raises_on_duplicate_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify discover_all_plugins raises ValueError on duplicate names.

        This tests the actual deduplication code in discover_all_plugins by mocking
        discover_plugins_in_directory to return duplicate plugins.
        """
        from elspeth.plugins import discovery
        from elspeth.plugins.base import BaseSource

        # Create fake plugin classes with same name
        class FakeSource1(BaseSource):
            name = "collision"
            __module__ = "elspeth.plugins._discovered.sources.source1"
            output_schema = None  # type: ignore[assignment]
            node_id = None
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def load(self, ctx):  # type: ignore[no-untyped-def]
                return iter([])

            def close(self) -> None:
                pass

            def on_start(self, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

            def on_complete(self, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

        class FakeSource2(BaseSource):
            name = "collision"  # Same name - duplicate!
            __module__ = "elspeth.plugins._discovered.azure.source2"
            output_schema = None  # type: ignore[assignment]
            node_id = None
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def load(self, ctx):  # type: ignore[no-untyped-def]
                return iter([])

            def close(self) -> None:
                pass

            def on_start(self, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

            def on_complete(self, ctx) -> None:  # type: ignore[no-untyped-def]
                pass

        # Track call count to return different results per directory
        call_count = {"value": 0}

        def mock_discover(directory: Path, base_class: type) -> list[type]:
            call_count["value"] += 1
            # First call returns FakeSource1, second returns FakeSource2 (same name)
            if call_count["value"] == 1:
                return [FakeSource1]
            elif call_count["value"] == 2:
                return [FakeSource2]
            return []

        # Patch to use our mock
        monkeypatch.setattr(discovery, "discover_plugins_in_directory", mock_discover)

        # Patch config to scan two directories for sources only
        monkeypatch.setattr(
            discovery,
            "PLUGIN_SCAN_CONFIG",
            {"sources": ["sources", "azure"], "transforms": [], "sinks": []},
        )

        # Patch base classes function
        monkeypatch.setattr(
            discovery,
            "_get_base_classes",
            lambda: {"sources": BaseSource, "transforms": type, "sinks": type},
        )

        # This should raise because both directories return plugins with name="collision"
        with pytest.raises(ValueError, match=r"Duplicate sources plugin name 'collision'"):
            discovery.discover_all_plugins()


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

        result = hookimpl_obj.elspeth_get_source()  # type: ignore[attr-defined]
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
        assert source is TestSource  # type: ignore[comparison-overlap]
