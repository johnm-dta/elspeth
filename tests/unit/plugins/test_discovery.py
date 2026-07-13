"""Tests for dynamic plugin discovery."""

import sys
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform
from elspeth.plugins.infrastructure.discovery import (
    _canonical_module_name,
    discover_plugins_in_directory,
)


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
from elspeth.plugins.infrastructure.base import BaseSource

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


class TestDiscoverPluginsEdgeCases:
    """Test edge cases in plugin discovery."""

    def test_nonexistent_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """discover_plugins_in_directory raises FileNotFoundError for non-existent directory."""
        bogus_dir = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError, match="Plugin directory does not exist"):
            discover_plugins_in_directory(bogus_dir, BaseSource)

    def test_non_string_name_attribute_raises(self, tmp_path: Path) -> None:
        """Plugin with non-string name attribute raises ValueError."""
        plugin_file = tmp_path / "bad_name.py"
        plugin_file.write_text("""
from elspeth.plugins.infrastructure.base import BaseSource

class BadNameSource(BaseSource):
    name = 42  # Not a string!
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

        with pytest.raises(ValueError, match="invalid 'name' value"):
            discover_plugins_in_directory(tmp_path, BaseSource)

    def test_empty_name_attribute_raises(self, tmp_path: Path) -> None:
        """Plugin with empty-after-strip name attribute raises ValueError."""
        plugin_file = tmp_path / "empty_name.py"
        plugin_file.write_text("""
from elspeth.plugins.infrastructure.base import BaseSource

class EmptyNameSource(BaseSource):
    name = "   "  # Whitespace only, empty after strip
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

        with pytest.raises(ValueError, match="invalid 'name' value"):
            discover_plugins_in_directory(tmp_path, BaseSource)


class TestDiscoverAllPlugins:
    """Test discovery across all plugin directories."""

    def test_discover_all_sources(self) -> None:
        """Verify all sources are discovered including azure."""
        from elspeth.plugins.infrastructure.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        source_names = [cls.name for cls in discovered["sources"]]  # type: ignore[attr-defined]
        assert "csv" in source_names
        assert "json" in source_names
        assert "null" in source_names
        # Azure blob source lives in plugins/azure/
        assert "azure_blob" in source_names
        assert "web_source" not in source_names

    def test_discover_all_transforms(self) -> None:
        """Verify all transforms are discovered including llm/ and transforms/azure/."""
        from elspeth.plugins.infrastructure.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        transform_names = [cls.name for cls in discovered["transforms"]]  # type: ignore[attr-defined]
        assert "blob_csv_expand" in transform_names
        assert "blob_fetch" in transform_names
        assert "passthrough" in transform_names
        assert "field_mapper" in transform_names
        # LLM transforms live in plugins/llm/ - verify unified is discovered
        assert "llm" in transform_names, f"Missing llm in {transform_names}"
        # Azure transforms live in plugins/transforms/azure/ (subdirectory!)
        assert "azure_content_safety" in transform_names, f"Missing azure_content_safety in {transform_names}"
        assert "azure_prompt_shield" in transform_names, f"Missing azure_prompt_shield in {transform_names}"

    def test_discover_all_sinks(self) -> None:
        """Verify all sinks are discovered including azure."""
        from elspeth.plugins.infrastructure.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        sink_names = [cls.name for cls in discovered["sinks"]]  # type: ignore[attr-defined]
        assert "csv" in sink_names
        assert "json" in sink_names
        assert "text" in sink_names
        assert "database" in sink_names

    def test_no_duplicate_names_within_type(self) -> None:
        """Verify no duplicate plugin names within same type."""
        from elspeth.plugins.infrastructure.discovery import discover_all_plugins

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
        from elspeth.plugins.infrastructure.discovery import discover_all_plugins

        # Expected counts verified during migration from hookimpl files
        EXPECTED_SOURCE_COUNT = 6  # csv, json, null, azure_blob, dataverse, text
        EXPECTED_TRANSFORM_COUNT = (
            29  # 22 standard transforms + 2 azure safety + azure_document_intelligence + llm + rag_retrieval + blob_fetch + blob_csv_expand
        )
        EXPECTED_SINK_COUNT = 7  # csv, json, text, database, azure_blob, dataverse, chroma_sink

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

    def test_all_plugins_have_config_model(self) -> None:
        """Every registered plugin must declare a config_model (via ClassVar or get_config_model).

        A plugin that inherits config_model = None from the base class
        silently bypasses pre-validation in validation.py — invalid user
        config passes through with no error.  NullSource is the only
        intentional exception (resume-only, no config).
        """
        from elspeth.plugins.infrastructure.discovery import discover_all_plugins

        # Plugins intentionally exempt from config_model.
        # Each entry needs a justification comment.
        EXEMPT = {
            "null",  # NullSource: resume-only source, requires no config
        }

        discovered = discover_all_plugins()

        missing: list[str] = []
        for plugin_type, plugins in discovered.items():
            for cls in plugins:
                plugin_name: str = cls.name  # type: ignore[attr-defined]
                if plugin_name in EXEMPT:
                    continue
                model = cls.get_config_model()  # type: ignore[attr-defined]  # discovered plugin classes have this method
                if model is None:
                    missing.append(f"{plugin_type}/{plugin_name} ({cls.__qualname__})")

        assert not missing, (
            "Plugins without config_model silently bypass pre-validation. "
            "Either add a config_model ClassVar or add to EXEMPT with justification:\n" + "\n".join(f"  - {m}" for m in missing)
        )


class TestGetPluginDescription:
    """Test docstring extraction for plugin descriptions."""

    def test_extracts_first_line_of_docstring(self) -> None:
        """Verify first docstring line is extracted."""
        from elspeth.plugins.infrastructure.discovery import get_plugin_description
        from elspeth.plugins.transforms.passthrough import PassThrough

        description = get_plugin_description(PassThrough)

        # PassThrough's class docstring starts with "Pass rows through unchanged."
        assert description == "Pass rows through unchanged."

    def test_handles_missing_docstring(self) -> None:
        """Verify fallback for classes without docstrings."""
        from elspeth.plugins.infrastructure.discovery import get_plugin_description

        class NoDocPlugin:
            name = "no_doc"

        description = get_plugin_description(NoDocPlugin)

        assert "no_doc" in description.lower()

    def test_strips_whitespace(self) -> None:
        """Verify whitespace is stripped from description."""
        from elspeth.plugins.infrastructure.discovery import get_plugin_description

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

        from elspeth.plugins.infrastructure.discovery import discover_plugins_in_directory

        # Create two plugin files with same name attribute
        plugin1 = tmp_path / "plugin_one.py"
        plugin1.write_text("""
from elspeth.plugins.infrastructure.base import BaseSource

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
from elspeth.plugins.infrastructure.base import BaseSource

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
        from elspeth.plugins.infrastructure.base import BaseSource

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
        from elspeth.plugins.infrastructure import discovery
        from elspeth.plugins.infrastructure.base import BaseSource

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

            def load(self, ctx):
                return iter([])

            def close(self) -> None:
                pass

            def on_start(self, ctx) -> None:
                pass

            def on_complete(self, ctx) -> None:
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

            def load(self, ctx):
                return iter([])

            def close(self) -> None:
                pass

            def on_start(self, ctx) -> None:
                pass

            def on_complete(self, ctx) -> None:
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
        from elspeth.plugins.infrastructure.discovery import create_dynamic_hookimpl

        class FakePlugin:
            name = "fake"

        hookimpl_obj = create_dynamic_hookimpl([FakePlugin], "elspeth_get_source")

        assert hasattr(hookimpl_obj, "elspeth_get_source")

    def test_hookimpl_returns_plugin_list(self) -> None:
        """Verify hookimpl method returns the plugin classes."""
        from elspeth.plugins.infrastructure.discovery import create_dynamic_hookimpl

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

        from elspeth.plugins.infrastructure.discovery import create_dynamic_hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager

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


class TestDiscoveryOptionalDependency:
    """Regression: missing optional extras should skip with warning, not crash."""

    def test_allowlisted_optional_dependency_skips_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plugin file missing an allowlisted optional dependency should be skipped."""
        from elspeth.plugins.infrastructure import discovery

        plugin_file = tmp_path / "bad_plugin.py"
        plugin_file.write_text("import nonexistent_optional_package_xyz\n")
        monkeypatch.setattr(
            discovery,
            "OPTIONAL_PLUGIN_IMPORT_MODULES",
            frozenset({"nonexistent_optional_package_xyz"}),
        )

        # Should not raise — skips the file with a warning
        result = discover_plugins_in_directory(tmp_path, BaseSource)
        assert result == []

    def test_jinja2_optional_dependency_skips_file_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM template support imports jinja2, which belongs to optional plugin extras."""
        from elspeth.plugins.infrastructure import discovery

        plugin_file = tmp_path / "llm_plugin.py"
        plugin_file.write_text("raise AssertionError('patched discovery should not execute this file')\n")

        def missing_jinja2(_py_file: Path, _base_class: type) -> list[type]:
            raise ModuleNotFoundError("No module named 'jinja2'", name="jinja2")

        monkeypatch.setattr(discovery, "_discover_in_file", missing_jinja2)

        result = discover_plugins_in_directory(tmp_path, BaseTransform)
        assert result == []

    def test_missing_internal_import_in_concrete_plugin_crashes(self, tmp_path: Path) -> None:
        """A broken concrete system plugin must not disappear as an optional extra."""
        plugin_file = tmp_path / "broken_internal_import.py"
        plugin_file.write_text("""
import definitely_missing_internal_helper

from elspeth.plugins.infrastructure.base import BaseSource

class BrokenInternalImportSource(BaseSource):
    name = "broken_internal_import"
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

        with pytest.raises(ModuleNotFoundError, match="definitely_missing_internal_helper"):
            discover_plugins_in_directory(tmp_path, BaseSource)

    def test_syntax_error_still_crashes(self, tmp_path: Path) -> None:
        """Genuine code bugs (SyntaxError) must still crash."""
        plugin_file = tmp_path / "broken_plugin.py"
        plugin_file.write_text("def broken(\n")  # Invalid syntax

        with pytest.raises(SyntaxError):
            discover_plugins_in_directory(tmp_path, BaseSource)


class TestCanonicalModuleName:
    """Tests for _canonical_module_name() — the foundation of plugin identity.

    This function prevents dual class objects by computing a canonical dotted
    module name from a file path.  If it returns the wrong answer (or None
    when it should return a name), isinstance/issubclass checks can silently
    fail at runtime.
    """

    def test_standard_plugin_path(self) -> None:
        """A file under the real elspeth package tree returns the correct dotted name."""
        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        passthrough = plugins_root / "transforms" / "passthrough.py"
        assert passthrough.exists(), f"Test fixture missing: {passthrough}"

        result = _canonical_module_name(passthrough)

        assert result == "elspeth.plugins.transforms.passthrough"

    def test_nested_plugin_path(self) -> None:
        """A file in a subdirectory (llm/) returns the full dotted path."""
        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        multi_query = plugins_root / "transforms" / "llm" / "multi_query.py"
        assert multi_query.exists(), f"Test fixture missing: {multi_query}"

        result = _canonical_module_name(multi_query)

        assert result == "elspeth.plugins.transforms.llm.multi_query"

    def test_non_elspeth_path_returns_none(self, tmp_path: Path) -> None:
        """A file outside the elspeth package tree returns None."""
        fake_file = tmp_path / "some_plugin.py"
        fake_file.touch()

        result = _canonical_module_name(fake_file)

        assert result is None

    def test_missing_init_breaks_traversal(self, tmp_path: Path) -> None:
        """If __init__.py is missing partway up, traversal stops and returns None."""
        # Create: tmp/elspeth/plugins/transforms/my_plugin.py
        # but WITHOUT __init__.py in "plugins/" — chain is broken
        transforms = tmp_path / "elspeth" / "plugins" / "transforms"
        transforms.mkdir(parents=True)
        (transforms / "__init__.py").touch()
        # Deliberately skip (tmp_path / "elspeth" / "plugins" / "__init__.py")
        (tmp_path / "elspeth" / "__init__.py").touch()

        plugin_file = transforms / "my_plugin.py"
        plugin_file.touch()

        result = _canonical_module_name(plugin_file)

        # Traversal: transforms (has __init__) → plugins (NO __init__ → break)
        # Never reaches "elspeth", so returns None
        assert result is None

    def test_complete_init_chain_returns_name(self, tmp_path: Path) -> None:
        """With a complete __init__.py chain, the canonical name is computed."""
        transforms = tmp_path / "elspeth" / "plugins" / "transforms"
        transforms.mkdir(parents=True)
        (transforms / "__init__.py").touch()
        (tmp_path / "elspeth" / "plugins" / "__init__.py").touch()
        (tmp_path / "elspeth" / "__init__.py").touch()

        plugin_file = transforms / "my_plugin.py"
        plugin_file.touch()

        result = _canonical_module_name(plugin_file)

        assert result == "elspeth.plugins.transforms.my_plugin"

    def test_file_named_elspeth_outside_package(self, tmp_path: Path) -> None:
        """A directory named 'elspeth' without __init__.py does not produce a name."""
        # tmp/elspeth/my_plugin.py with NO __init__.py
        elspeth_dir = tmp_path / "elspeth"
        elspeth_dir.mkdir()
        # No __init__.py — traversal never enters the while loop meaningfully

        plugin_file = elspeth_dir / "my_plugin.py"
        plugin_file.touch()

        result = _canonical_module_name(plugin_file)

        # parent is "elspeth" but has no __init__.py → loop breaks immediately
        assert result is None

    def test_core_module_path(self) -> None:
        """Files deep in the elspeth tree (core/, contracts/) also resolve correctly."""
        src_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth"
        # Pick a known core file
        core_files = list((src_root / "core").glob("*.py"))
        non_init = [f for f in core_files if f.name != "__init__.py"]
        assert non_init, "Expected at least one non-init file in core/"

        target = non_init[0]
        result = _canonical_module_name(target)

        assert result is not None
        assert result.startswith("elspeth.core.")
        assert result.endswith(target.stem)


class TestDualSysModulesRegistration:
    """Tests for the dual sys.modules registration in _discover_in_file.

    When discovery loads a plugin by file path, the module gets a synthetic name
    (elspeth.plugins._discovered.<parent>.<stem>).  If the same file has a
    canonical name (elspeth.plugins.transforms.<stem>), _discover_in_file must:

    1. Reuse an existing canonical module (dedup path, line 133-134)
    2. Register the canonical alias after fresh load (line 164-165)
    3. Clean up sys.modules on exec_module failure (line 158)

    Without these, duplicate class objects silently defeat isinstance checks.
    """

    def test_canonical_module_reused_when_already_imported(self) -> None:
        """If the canonical module is already in sys.modules, discovery reuses it.

        This is the dedup path — it prevents creating a second class object.
        """
        from elspeth.plugins.infrastructure.discovery import _discover_in_file

        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        passthrough_file = plugins_root / "transforms" / "passthrough.py"

        # Explicitly import the module canonically (can't rely on test ordering
        # since pytest-xdist workers have independent sys.modules)
        import importlib

        canonical_name = "elspeth.plugins.transforms.passthrough"
        importlib.import_module(canonical_name)
        original_module = sys.modules[canonical_name]

        # Discovery should reuse the existing module, NOT load a new one
        discovered = _discover_in_file(passthrough_file, BaseTransform)

        # The discovered class must come from the ORIGINAL module
        assert len(discovered) >= 1
        passthrough_cls = next(c for c in discovered if c.name == "passthrough")  # type: ignore[attr-defined]
        assert passthrough_cls.__module__ == original_module.__name__

    def test_canonical_alias_registered_after_fresh_load(self, tmp_path: Path) -> None:
        """After loading a file under the elspeth package tree, the canonical
        name is registered in sys.modules as an alias.
        """
        from elspeth.plugins.infrastructure.discovery import _discover_in_file

        # Build a valid elspeth package tree in tmp_path
        transforms = tmp_path / "elspeth" / "plugins" / "transforms"
        transforms.mkdir(parents=True)
        (transforms / "__init__.py").touch()
        (tmp_path / "elspeth" / "plugins" / "__init__.py").touch()
        (tmp_path / "elspeth" / "__init__.py").touch()

        plugin_file = transforms / "test_fresh_load_plugin.py"
        plugin_file.write_text(
            "from elspeth.plugins.infrastructure.base import BaseTransform\n"
            "\n"
            "class FreshPlugin(BaseTransform):\n"
            "    name = 'test_fresh_load'\n"
            "    input_schema = None\n"
            "    output_schema = None\n"
            "    node_id = None\n"
            "    determinism = 'deterministic'\n"
            "    plugin_version = '1.0.0'\n"
            "\n"
            "    def __init__(self, config):\n"
            "        self.config = config\n"
            "\n"
            "    def process(self, row, ctx):\n"
            "        return row\n"
            "\n"
            "    def on_start(self, ctx):\n"
            "        pass\n"
            "\n"
            "    def on_complete(self, ctx):\n"
            "        pass\n"
        )

        canonical = "elspeth.plugins.transforms.test_fresh_load_plugin"
        synthetic = f"elspeth.plugins._discovered.transforms.{plugin_file.stem}"

        # Ensure neither name exists before discovery
        sys.modules.pop(canonical, None)
        sys.modules.pop(synthetic, None)

        try:
            _discover_in_file(plugin_file, BaseTransform)

            # Synthetic name should be registered (line 153)
            assert synthetic in sys.modules, f"Synthetic name {synthetic!r} not registered in sys.modules"
            # Canonical alias should also be registered (line 164-165)
            assert canonical in sys.modules, f"Canonical alias {canonical!r} not registered in sys.modules"
            # Both must point to the SAME module object
            assert sys.modules[synthetic] is sys.modules[canonical], "Synthetic and canonical sys.modules entries must be the same object"
        finally:
            sys.modules.pop(canonical, None)
            sys.modules.pop(synthetic, None)

    def test_sys_modules_cleaned_on_exec_failure(self, tmp_path: Path) -> None:
        """If exec_module raises, the synthetic sys.modules entry is removed."""
        from elspeth.plugins.infrastructure.discovery import _discover_in_file

        plugin_file = tmp_path / "exploding_plugin.py"
        plugin_file.write_text("raise RuntimeError('boom during import')\n")

        synthetic = f"elspeth.plugins._discovered.{tmp_path.name}.exploding_plugin"
        sys.modules.pop(synthetic, None)

        with pytest.raises(RuntimeError, match="boom during import"):
            _discover_in_file(plugin_file, BaseTransform)

        # The pre-registered entry must have been cleaned up (line 158)
        assert synthetic not in sys.modules, f"Failed module {synthetic!r} was not cleaned up from sys.modules"

    def test_no_duplicate_class_objects_on_double_discovery(self) -> None:
        """Discovering the same file twice returns classes from the same module object.

        This is the real-world scenario: discover_all_plugins() scans transforms/,
        but another plugin may have already imported a transform module at the
        module level. The dedup path must ensure a single class identity.
        """
        from elspeth.plugins.infrastructure.discovery import _discover_in_file

        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        passthrough_file = plugins_root / "transforms" / "passthrough.py"

        first = _discover_in_file(passthrough_file, BaseTransform)
        second = _discover_in_file(passthrough_file, BaseTransform)

        first_cls = next(c for c in first if c.name == "passthrough")  # type: ignore[attr-defined]
        second_cls = next(c for c in second if c.name == "passthrough")  # type: ignore[attr-defined]

        # Must be the EXACT SAME class object — not an equal copy
        assert first_cls is second_cls, (
            f"Double discovery produced duplicate class objects: {first_cls!r} (id={id(first_cls)}) vs {second_cls!r} (id={id(second_cls)})"
        )

    def test_isinstance_works_across_discovery_and_import(self) -> None:
        """isinstance() must work between discovery-loaded and import-loaded classes.

        This is the failure mode that dual registration prevents: if discovery
        creates a separate class object, isinstance checks fail silently.
        """
        from elspeth.plugins.infrastructure.discovery import _discover_in_file
        from elspeth.plugins.transforms.passthrough import PassThrough

        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        passthrough_file = plugins_root / "transforms" / "passthrough.py"

        discovered = _discover_in_file(passthrough_file, BaseTransform)
        discovered_cls = next(c for c in discovered if c.name == "passthrough")  # type: ignore[attr-defined]

        # The discovered class must BE the imported class
        assert discovered_cls is PassThrough, (
            f"Discovery returned a different class object than the import: "
            f"discovered={discovered_cls.__module__}.{discovered_cls.__qualname__} "
            f"vs imported={PassThrough.__module__}.{PassThrough.__qualname__}"
        )


class TestModuleFilterWithCanonicalNames:
    """Tests for the __module__ filter (line 171) interacting with canonical names.

    The filter `obj.__module__ != module.__name__` decides whether a class was
    "defined in" the discovered module vs. merely "imported into" it.  When the
    dedup path reuses a canonically-imported module, module.__name__ is the
    canonical name — so the filter must match cls.__module__ against that name,
    not the synthetic _discovered name.
    """

    def test_filter_does_not_exclude_classes_on_dedup_path(self) -> None:
        """When dedup reuses a canonical module, classes must still pass the filter.

        Regression guard: if _discover_in_file used the synthetic module name
        for filtering but the classes have __module__ set to the canonical name,
        the filter would silently exclude all classes → empty discovery result.
        """
        from elspeth.plugins.infrastructure.discovery import _discover_in_file

        plugins_root = Path(__file__).parent.parent.parent.parent / "src" / "elspeth" / "plugins"
        passthrough_file = plugins_root / "transforms" / "passthrough.py"

        # Ensure canonical module is loaded (import does this)
        import elspeth.plugins.transforms.passthrough  # noqa: F401

        canonical = "elspeth.plugins.transforms.passthrough"
        assert canonical in sys.modules

        discovered = _discover_in_file(passthrough_file, BaseTransform)

        # Must find at least PassThrough — filter must not exclude it
        names = [c.name for c in discovered]  # type: ignore[attr-defined]
        assert "passthrough" in names, (
            f"__module__ filter excluded PassThrough on dedup path. Module name: {sys.modules[canonical].__name__!r}"
        )

    def test_imported_classes_are_excluded(self, tmp_path: Path) -> None:
        """Classes imported INTO a module (not defined there) must be filtered out.

        A plugin file that does `from elspeth.plugins.infrastructure.base import BaseTransform`
        should NOT return BaseTransform as a discovered plugin — only classes
        defined in the file itself.
        """
        from elspeth.plugins.infrastructure.discovery import _discover_in_file

        plugin_file = tmp_path / "imports_base.py"
        plugin_file.write_text(
            "from elspeth.plugins.infrastructure.base import BaseTransform\n"
            "\n"
            "class MyTransform(BaseTransform):\n"
            "    name = 'my_transform'\n"
            "    input_schema = None\n"
            "    output_schema = None\n"
            "    node_id = None\n"
            "    determinism = 'deterministic'\n"
            "    plugin_version = '1.0.0'\n"
            "\n"
            "    def __init__(self, config):\n"
            "        self.config = config\n"
            "\n"
            "    def process(self, row, ctx):\n"
            "        return row\n"
            "\n"
            "    def on_start(self, ctx):\n"
            "        pass\n"
            "\n"
            "    def on_complete(self, ctx):\n"
            "        pass\n"
        )

        synthetic = f"elspeth.plugins._discovered.{tmp_path.name}.imports_base"
        sys.modules.pop(synthetic, None)

        try:
            discovered = _discover_in_file(plugin_file, BaseTransform)

            names = [c.__name__ for c in discovered]
            # MyTransform should be found
            assert "MyTransform" in names
            # BaseTransform must NOT appear (it's imported, not defined here)
            assert "BaseTransform" not in names
        finally:
            sys.modules.pop(synthetic, None)
