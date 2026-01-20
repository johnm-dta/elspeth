# Dynamic Plugin Discovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate manual plugin registration — drop a plugin file in the correct folder and it's automatically discovered.

**Architecture:** Add folder-scanning discovery that finds classes inheriting from base classes, extracts `name` attributes, and registers via dynamically-generated pluggy hookimpls. CLI uses PluginManager for all lookups.

**Tech Stack:** Python 3.12, pluggy, importlib, pytest

---

## Code Review Findings (2026-01-20)

Plan reviewed against codebase by pr-review-toolkit:code-reviewer. Issues fixed:

| Issue | Fix Applied |
|-------|-------------|
| ❌ `transforms/azure/` not in scan config | Added to `PLUGIN_SCAN_CONFIG["transforms"]["directories"]` |
| ⚠️ `EXCLUDED_FILES` incomplete | Added `aimd_throttle.py`, `capacity_errors.py`, `reorder_buffer.py`, client helpers |
| 💡 No regression test for plugin counts | Added `test_discovery_matches_hookimpl_counts()` |

### Second Review (2026-01-20)

Plan reviewed against codebase and `docs/contracts/plugin-protocol.md` + `ARCHITECTURE.md`:

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| 🔴 PassThrough docstring test wrong | Bug | Changed assertion to `"PassThrough transform plugin."` |
| 🔴 Regression test imports deleted hookimpl files | Bug | Added Step 4 in Task 9 to convert test to static counts |
| 🟡 Missing `pooled_executor.py` in EXCLUDED_FILES | Risk | Added to exclusion list |
| 🟡 Silent exception swallowing hides plugin bugs | Risk | Made exception handling more selective (ImportError only) |
| 🟡 Gates included as plugins but are config-driven per architecture | Design | Removed gates from PLUGIN_SCAN_CONFIG (gates are system operations, not plugins) |
| 🟢 Module name collision potential | Risk | Added parent directory to module name for uniqueness |
| 🟢 `azure_batch_llm` test coverage gap | Test | Added explicit assertions for all LLM transforms |

---

## Problem Summary

Adding a plugin currently requires edits in **4 separate locations**:
1. Plugin class file (`plugins/{type}/{name}.py`)
2. Hook implementation list (`plugins/{type}/hookimpl.py`)
3. CLI registries (`cli.py` - `TRANSFORM_PLUGINS` dict + if/elif chains)
4. CLI plugin listing (`cli.py` - `PLUGIN_REGISTRY` dict)

**After this change:** Only location 1 is needed.

---

## Scope Discipline

**DO NOT:**
- Add features not in this plan
- Add deprecation warnings or compatibility shims
- Keep old code "for reference"

**DO:**
- Follow TDD exactly as written
- Delete old registration code completely
- Test discovery with existing plugins

---

## Task 1: Create Discovery Module with Core Functions

**Files:**
- Create: `src/elspeth/plugins/discovery.py`
- Test: `tests/plugins/test_discovery.py`

**Step 1: Write the failing test for plugin discovery**

Create `tests/plugins/test_discovery.py`:

```python
"""Tests for dynamic plugin discovery."""

from pathlib import Path

import pytest

from elspeth.plugins.base import BaseSource, BaseSink, BaseTransform
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_discovery.py -v`

Expected: FAIL with "cannot import name 'discover_plugins_in_directory'"

**Step 3: Write minimal discovery implementation**

Create `src/elspeth/plugins/discovery.py`:

```python
"""Dynamic plugin discovery by folder scanning.

Scans plugin directories for classes that:
1. Inherit from a base class (BaseSource, BaseTransform, etc.)
2. Have a `name` class attribute
3. Are not abstract (no @abstractmethod methods without implementation)
"""

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Files that should never be scanned for plugins
EXCLUDED_FILES: frozenset[str] = frozenset({
    "__init__.py",
    "hookimpl.py",
    "base.py",
    "templates.py",
    "auth.py",
    "batch_errors.py",
    "sentinels.py",
    "schema_factory.py",
    "utils.py",
    "protocols.py",
    "results.py",
    "context.py",
    "config_base.py",
    "hookspecs.py",
    "manager.py",
    "discovery.py",
    # LLM helpers (not plugins)
    "aimd_throttle.py",
    "capacity_errors.py",
    "reorder_buffer.py",
    "pooled_executor.py",
    # Client helpers (not plugins)
    "http.py",
    "llm.py",
    "replayer.py",
    "verifier.py",
})


def discover_plugins_in_directory(
    directory: Path,
    base_class: type,
) -> list[type]:
    """Discover plugin classes in a directory.

    Scans all .py files in the directory (non-recursive) and finds classes
    that inherit from base_class and have a `name` attribute.

    Args:
        directory: Path to scan for plugin files
        base_class: Base class that plugins must inherit from

    Returns:
        List of discovered plugin classes
    """
    discovered: list[type] = []

    if not directory.exists():
        logger.warning("Plugin directory does not exist: %s", directory)
        return discovered

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name in EXCLUDED_FILES:
            continue

        try:
            plugins = _discover_in_file(py_file, base_class)
            discovered.extend(plugins)
        except ImportError as e:
            # Import errors are recoverable - missing optional dependencies
            logger.warning("Import error scanning %s for plugins: %s", py_file, e)
        except SyntaxError as e:
            # Syntax errors are bugs - log at error level but don't crash discovery
            logger.error("Syntax error in plugin file %s: %s", py_file, e)

    return discovered


def _discover_in_file(py_file: Path, base_class: type) -> list[type]:
    """Discover plugin classes in a single Python file.

    Args:
        py_file: Path to Python file
        base_class: Base class that plugins must inherit from

    Returns:
        List of plugin classes found in the file
    """
    # Load module from file path
    # Include parent directory in module name to avoid collisions
    # (e.g., transforms/base.py vs llm/base.py would both be "base" otherwise)
    parent_name = py_file.parent.name
    module_name = f"elspeth.plugins._discovered.{parent_name}.{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find classes that inherit from base_class
    discovered: list[type] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Must be defined in this module (not imported)
        if obj.__module__ != module.__name__:
            continue

        # Must inherit from base_class (but not BE base_class)
        if not issubclass(obj, base_class) or obj is base_class:
            continue

        # Must not be abstract
        if inspect.isabstract(obj):
            continue

        # Must have a `name` attribute
        if not hasattr(obj, "name"):
            logger.warning(
                "Class %s in %s inherits from %s but has no 'name' attribute - skipping",
                name,
                py_file,
                base_class.__name__,
            )
            continue

        # Must have a non-empty name
        if not obj.name:
            logger.warning(
                "Class %s in %s has empty 'name' attribute - skipping",
                name,
                py_file,
            )
            continue

        discovered.append(obj)

    return discovered
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_discovery.py -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/discovery.py tests/plugins/test_discovery.py
git commit -m "feat(plugins): add discover_plugins_in_directory function

Scans a directory for Python files and finds classes that:
- Inherit from a specified base class
- Have a 'name' class attribute
- Are not abstract

Excludes utility files like __init__.py, base.py, etc."
```

---

## Task 2: Add Multi-Directory Discovery Function

**Files:**
- Modify: `src/elspeth/plugins/discovery.py`
- Modify: `tests/plugins/test_discovery.py`

**Step 1: Write failing test for multi-directory discovery**

Add to `tests/plugins/test_discovery.py`:

```python
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
        from elspeth.plugins.sources.hookimpl import builtin_sources
        from elspeth.plugins.transforms.hookimpl import builtin_transforms
        from elspeth.plugins.sinks.hookimpl import builtin_sinks

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_discovery.py::TestDiscoverAllPlugins -v`

Expected: FAIL with "cannot import name 'discover_all_plugins'"

**Step 3: Implement discover_all_plugins**

Add to `src/elspeth/plugins/discovery.py`:

```python
from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform

# Configuration: which directories to scan for each plugin type
# NOTE: Non-recursive scanning - subdirectories must be listed explicitly
#
# IMPORTANT: Gates are NOT included here. Per docs/contracts/plugin-protocol.md,
# gates are "config-driven system operations" handled by the engine, NOT plugins.
# See "System Operations (NOT Plugins)" section in the plugin protocol.
PLUGIN_SCAN_CONFIG: dict[str, dict[str, Any]] = {
    "sources": {
        "base_class": BaseSource,
        "directories": ["sources", "azure"],
    },
    "transforms": {
        "base_class": BaseTransform,
        "directories": ["transforms", "transforms/azure", "llm"],  # transforms/azure has AzureContentSafety, AzurePromptShield
    },
    "sinks": {
        "base_class": BaseSink,
        "directories": ["sinks", "azure"],
    },
}


def discover_all_plugins() -> dict[str, list[type]]:
    """Discover all built-in plugins by scanning configured directories.

    Returns:
        Dict mapping plugin type to list of discovered plugin classes:
        {
            "sources": [CSVSource, JSONSource, ...],
            "transforms": [PassThrough, FieldMapper, ...],
            "gates": [...],
            "sinks": [CSVSink, JSONSink, ...],
        }
    """
    plugins_root = Path(__file__).parent
    result: dict[str, list[type]] = {}

    for plugin_type, config in PLUGIN_SCAN_CONFIG.items():
        base_class = config["base_class"]
        directories = config["directories"]

        all_discovered: list[type] = []
        seen_names: set[str] = set()

        for dir_name in directories:
            directory = plugins_root / dir_name
            discovered = discover_plugins_in_directory(directory, base_class)

            for cls in discovered:
                # Skip duplicates (same class discovered from different logic)
                if cls.name in seen_names:
                    continue
                seen_names.add(cls.name)
                all_discovered.append(cls)

        result[plugin_type] = all_discovered

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_discovery.py::TestDiscoverAllPlugins -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/discovery.py tests/plugins/test_discovery.py
git commit -m "feat(plugins): add discover_all_plugins for multi-directory scan

Scans configured directories for each plugin type:
- sources: sources/, azure/
- transforms: transforms/, llm/
- gates: transforms/
- sinks: sinks/, azure/

Deduplicates by plugin name within each type."
```

---

## Task 3: Add Description Extraction from Docstrings

**Files:**
- Modify: `src/elspeth/plugins/discovery.py`
- Modify: `tests/plugins/test_discovery.py`

**Step 1: Write failing test for description extraction**

Add to `tests/plugins/test_discovery.py`:

```python
class TestGetPluginDescription:
    """Test docstring extraction for plugin descriptions."""

    def test_extracts_first_line_of_docstring(self) -> None:
        """Verify first docstring line is extracted."""
        from elspeth.plugins.discovery import get_plugin_description
        from elspeth.plugins.transforms.passthrough import PassThrough

        description = get_plugin_description(PassThrough)

        # PassThrough's docstring starts with "PassThrough transform plugin."
        assert description == "PassThrough transform plugin."

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
            """   Lots of whitespace here.   """

            name = "whitespace"

        description = get_plugin_description(WhitespaceDocPlugin)

        assert description == "Lots of whitespace here."
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_discovery.py::TestGetPluginDescription -v`

Expected: FAIL with "cannot import name 'get_plugin_description'"

**Step 3: Implement get_plugin_description**

Add to `src/elspeth/plugins/discovery.py`:

```python
def get_plugin_description(plugin_cls: type) -> str:
    """Extract description from plugin class docstring.

    Returns the first non-empty line of the docstring, stripped of whitespace.
    If no docstring exists, returns a default message using the plugin name.

    Args:
        plugin_cls: Plugin class to extract description from

    Returns:
        Description string for display
    """
    if plugin_cls.__doc__:
        lines = plugin_cls.__doc__.strip().split("\n")
        for line in lines:
            cleaned = line.strip()
            if cleaned:
                return cleaned

    # Fallback to name-based description
    name = getattr(plugin_cls, "name", plugin_cls.__name__)
    return f"{name} plugin"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_discovery.py::TestGetPluginDescription -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/discovery.py tests/plugins/test_discovery.py
git commit -m "feat(plugins): add get_plugin_description for docstring extraction

Extracts first non-empty line of class docstring for CLI display.
Falls back to plugin name if no docstring present."
```

---

## Task 4: Add Dynamic Hookimpl Generation

**Files:**
- Modify: `src/elspeth/plugins/discovery.py`
- Modify: `tests/plugins/test_discovery.py`

**Step 1: Write failing test for hookimpl generation**

Add to `tests/plugins/test_discovery.py`:

```python
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
        from elspeth.plugins.discovery import create_dynamic_hookimpl
        from elspeth.plugins.manager import PluginManager

        class TestSource:
            name = "test_dynamic"
            output_schema = None
            node_id = None
            determinism = "deterministic"
            plugin_version = "1.0.0"

            def __init__(self, config: dict) -> None:
                pass

            def load(self, ctx):
                return iter([])

            def close(self) -> None:
                pass

            def on_start(self, ctx) -> None:
                pass

            def on_complete(self, ctx) -> None:
                pass

        hookimpl_obj = create_dynamic_hookimpl([TestSource], "elspeth_get_source")

        manager = PluginManager()
        manager.register(hookimpl_obj)

        source = manager.get_source_by_name("test_dynamic")
        assert source is TestSource
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_discovery.py::TestCreateDynamicHookimpl -v`

Expected: FAIL with "cannot import name 'create_dynamic_hookimpl'"

**Step 3: Implement create_dynamic_hookimpl**

Add to `src/elspeth/plugins/discovery.py`:

```python
from elspeth.plugins.hookspecs import hookimpl


def create_dynamic_hookimpl(
    plugin_classes: list[type],
    hook_method_name: str,
) -> object:
    """Create a pluggy hookimpl object for plugin registration.

    Dynamically generates a class with the appropriate hook method
    decorated with @hookimpl that returns the provided plugin classes.

    Args:
        plugin_classes: List of plugin classes to register
        hook_method_name: Name of the hook method (e.g., "elspeth_get_source")

    Returns:
        Object instance with the decorated hook method
    """

    class DynamicHookImpl:
        """Dynamically generated hook implementer."""

        pass

    # Create the hook method that returns the plugin classes
    def hook_method(self: Any) -> list[type]:
        return plugin_classes

    # Apply the hookimpl decorator
    decorated_method = hookimpl(hook_method)

    # Attach to the class
    setattr(DynamicHookImpl, hook_method_name, decorated_method)

    return DynamicHookImpl()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_discovery.py::TestCreateDynamicHookimpl -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/discovery.py tests/plugins/test_discovery.py
git commit -m "feat(plugins): add create_dynamic_hookimpl for pluggy integration

Generates hookimpl objects dynamically so discovery can integrate
with the existing pluggy-based PluginManager."
```

---

## Task 5: Update PluginManager to Use Discovery

**Files:**
- Modify: `src/elspeth/plugins/manager.py`
- Modify: `tests/plugins/test_manager.py`

**Step 1: Write failing test for discovery-based registration**

Add to `tests/plugins/test_manager.py`:

```python
class TestDiscoveryBasedRegistration:
    """Test PluginManager with automatic discovery."""

    def test_register_builtin_discovers_csv_source(self) -> None:
        """Verify register_builtin_plugins finds CSVSource via discovery."""
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        source = manager.get_source_by_name("csv")
        assert source is not None
        assert source.name == "csv"

    def test_register_builtin_discovers_all_transforms(self) -> None:
        """Verify register_builtin_plugins finds all transforms."""
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        transforms = manager.get_transforms()
        names = [t.name for t in transforms]

        assert "passthrough" in names
        assert "field_mapper" in names

    def test_register_builtin_discovers_all_sinks(self) -> None:
        """Verify register_builtin_plugins finds all sinks."""
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        sinks = manager.get_sinks()
        names = [s.name for s in sinks]

        assert "csv" in names
        assert "json" in names
```

**Step 2: Run test to verify current behavior (should pass with old implementation)**

Run: `pytest tests/plugins/test_manager.py::TestDiscoveryBasedRegistration -v`

Expected: PASS (current static hookimpl files work)

**Step 3: Update register_builtin_plugins to use discovery**

Modify `src/elspeth/plugins/manager.py`:

Replace the `register_builtin_plugins` method:

```python
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
```

**Step 4: Run test to verify it still passes**

Run: `pytest tests/plugins/test_manager.py::TestDiscoveryBasedRegistration -v`

Expected: All PASS

**Step 5: Run full manager tests**

Run: `pytest tests/plugins/test_manager.py -v`

Expected: All PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/manager.py tests/plugins/test_manager.py
git commit -m "feat(plugins): update PluginManager to use dynamic discovery

register_builtin_plugins() now:
1. Calls discover_all_plugins() to scan directories
2. Creates dynamic hookimpls for each plugin type
3. Registers with pluggy as before

This removes the need for static hookimpl.py files."
```

---

## Task 6: Update CLI to Use PluginManager

**Files:**
- Modify: `src/elspeth/cli.py`
- Test: `tests/cli/test_plugins_command.py`

**Step 1: Run existing CLI tests to establish baseline**

Run: `pytest tests/cli/test_plugins_command.py -v`

Expected: All PASS (establishes baseline)

**Step 2: Add cached PluginManager helper to CLI**

Add near the top of `src/elspeth/cli.py` (after imports):

```python
def _get_plugin_manager() -> "PluginManager":
    """Get initialized plugin manager (singleton).

    Returns:
        PluginManager with all built-in plugins registered
    """
    from elspeth.plugins.manager import PluginManager

    if not hasattr(_get_plugin_manager, "_instance"):
        manager = PluginManager()
        manager.register_builtin_plugins()
        _get_plugin_manager._instance = manager
    return _get_plugin_manager._instance
```

**Step 3: Replace TRANSFORM_PLUGINS dict in _execute_pipeline**

In `_execute_pipeline()`, find and delete the `TRANSFORM_PLUGINS` dict (around lines 246-258).

Replace the transform instantiation loop with:

```python
# Get transforms via PluginManager
manager = _get_plugin_manager()

transforms: list[BaseTransform] = []
for plugin_config in config.row_plugins:
    plugin_name = plugin_config.plugin
    plugin_options = dict(plugin_config.options)

    transform_cls = manager.get_transform_by_name(plugin_name)
    if transform_cls is None:
        available = [t.name for t in manager.get_transforms()]
        raise typer.BadParameter(
            f"Unknown transform plugin: {plugin_name}. Available: {available}"
        )
    transforms.append(transform_cls(plugin_options))
```

**Step 4: Replace source if/elif chain**

Replace the source instantiation code with:

```python
# Get source via PluginManager
source_plugin = config.datasource.plugin
source_options = dict(config.datasource.options)

source_cls = manager.get_source_by_name(source_plugin)
if source_cls is None:
    available = [s.name for s in manager.get_sources()]
    raise ValueError(f"Unknown source plugin: {source_plugin}. Available: {available}")
source = source_cls(source_options)
```

**Step 5: Replace sink if/elif chain**

Replace the sink instantiation code with:

```python
# Get sinks via PluginManager
sinks: dict[str, BaseSink] = {}
for sink_name, sink_settings in config.sinks.items():
    sink_plugin = sink_settings.plugin
    sink_options = dict(sink_settings.options)

    sink_cls = manager.get_sink_by_name(sink_plugin)
    if sink_cls is None:
        available = [s.name for s in manager.get_sinks()]
        raise ValueError(f"Unknown sink plugin: {sink_plugin}. Available: {available}")
    sinks[sink_name] = sink_cls(sink_options)
```

**Step 6: Run CLI tests**

Run: `pytest tests/cli/ -v`

Expected: All PASS

**Step 7: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "refactor(cli): use PluginManager for plugin instantiation

Replace hardcoded TRANSFORM_PLUGINS dict and if/elif chains with
PluginManager lookups. Error messages now show available plugins.

Part of dynamic plugin discovery migration."
```

---

## Task 7: Update CLI plugins list Command

**Files:**
- Modify: `src/elspeth/cli.py`

**Step 1: Replace PLUGIN_REGISTRY with dynamic generation**

Find and delete the static `PLUGIN_REGISTRY` dict (around lines 421-449).

Replace the `plugins_list` command implementation:

```python
@plugins_app.command("list")
def plugins_list(
    plugin_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Filter by plugin type"),
    ] = None,
) -> None:
    """List available plugins discovered from plugin directories."""
    from elspeth.plugins.discovery import get_plugin_description

    manager = _get_plugin_manager()

    # Build registry dynamically
    # NOTE: Gates are NOT included - they are config-driven system operations,
    # not plugins. See docs/contracts/plugin-protocol.md "System Operations".
    registry: dict[str, list[PluginInfo]] = {
        "source": [
            PluginInfo(name=cls.name, description=get_plugin_description(cls))
            for cls in manager.get_sources()
        ],
        "transform": [
            PluginInfo(name=cls.name, description=get_plugin_description(cls))
            for cls in manager.get_transforms()
        ],
        "sink": [
            PluginInfo(name=cls.name, description=get_plugin_description(cls))
            for cls in manager.get_sinks()
        ],
    }

    # Determine which types to show
    if plugin_type:
        if plugin_type not in registry:
            typer.echo(f"Unknown plugin type: {plugin_type}", err=True)
            typer.echo(f"Available types: {list(registry.keys())}", err=True)
            raise typer.Exit(1)
        types_to_show = [plugin_type]
    else:
        types_to_show = list(registry.keys())

    # Display plugins
    for ptype in types_to_show:
        plugins = registry[ptype]
        if plugins:
            typer.echo(f"\n{ptype.upper()}S:")
            for plugin in sorted(plugins, key=lambda p: p.name):
                typer.echo(f"  {plugin.name:20} - {plugin.description}")
```

**Step 2: Run CLI plugin list tests**

Run: `pytest tests/cli/test_plugins_command.py -v`

Expected: All PASS

**Step 3: Manually verify plugins list output**

Run: `.venv/bin/python -m elspeth plugins list`

Expected: Shows all discovered plugins with descriptions from docstrings

**Step 4: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "refactor(cli): generate plugin registry dynamically

plugins list command now uses PluginManager to discover plugins
and extracts descriptions from class docstrings.

Part of dynamic plugin discovery migration."
```

---

## Task 8: Update _build_resume_pipeline_config

**Files:**
- Modify: `src/elspeth/cli.py`

**Step 1: Find duplicate code in resume function**

The `_build_resume_pipeline_config()` function has duplicate `TRANSFORM_PLUGINS` dict and instantiation logic.

**Step 2: Replace with PluginManager lookups**

Update `_build_resume_pipeline_config()` to use the same pattern:

```python
def _build_resume_pipeline_config(
    config: ElspethSettings,
    checkpoint: CheckpointData,
) -> PipelineConfig:
    """Build pipeline config for resume from checkpoint."""
    manager = _get_plugin_manager()

    # Source - use NullSource for resume (data comes from checkpoint)
    source_cls = manager.get_source_by_name("null")
    if source_cls is None:
        raise ValueError("NullSource not found - required for resume")
    source = source_cls({})

    # Transforms
    transforms: list[BaseTransform] = []
    for plugin_config in config.row_plugins:
        transform_cls = manager.get_transform_by_name(plugin_config.plugin)
        if transform_cls is None:
            raise ValueError(f"Unknown transform: {plugin_config.plugin}")
        transforms.append(transform_cls(dict(plugin_config.options)))

    # ... rest of function uses same pattern for sinks
```

**Step 3: Run resume tests**

Run: `pytest tests/cli/ -v -k resume`

Expected: All PASS (or skip if no resume tests)

**Step 4: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "refactor(cli): use PluginManager in resume pipeline config

Eliminates duplicate TRANSFORM_PLUGINS dict in _build_resume_pipeline_config.
Now uses same _get_plugin_manager() helper as _execute_pipeline."
```

---

## Task 9: Delete Legacy hookimpl Files

**Files:**
- Delete: `src/elspeth/plugins/sources/hookimpl.py`
- Delete: `src/elspeth/plugins/transforms/hookimpl.py`
- Delete: `src/elspeth/plugins/sinks/hookimpl.py`

**Step 1: Run full test suite first**

Run: `pytest tests/ -v`

Expected: All PASS (confirms nothing depends on hookimpl files)

**Step 2: Delete the hookimpl files**

```bash
rm src/elspeth/plugins/sources/hookimpl.py
rm src/elspeth/plugins/transforms/hookimpl.py
rm src/elspeth/plugins/sinks/hookimpl.py
```

**Step 3: Update __init__.py files to remove hookimpl imports**

Check each `__init__.py` and remove any imports of `builtin_*` from hookimpl:

- `src/elspeth/plugins/sources/__init__.py`
- `src/elspeth/plugins/transforms/__init__.py`
- `src/elspeth/plugins/sinks/__init__.py`

**Step 4: Convert migration test to static count assertions**

The `test_discovery_matches_hookimpl_counts` test imports from hookimpl files which we just deleted.
Replace it with static count assertions in `tests/plugins/test_discovery.py`:

```python
    def test_discovery_plugin_counts(self) -> None:
        """Verify discovery finds expected number of plugins.

        These counts were verified during the migration from static hookimpl
        files to dynamic discovery. If this test fails, a plugin may have
        been inadvertently excluded from discovery.

        Counts as of 2026-01-20:
        - Sources: csv, json, null, azure_blob = 4
        - Transforms: passthrough, field_mapper, batch_stats, json_explode,
                      keyword_filter, batch_replicate, azure_content_safety,
                      azure_prompt_shield, azure_llm, openrouter_llm,
                      azure_batch_llm = 11
        - Sinks: csv, json, database, azure_blob = 4
        """
        from elspeth.plugins.discovery import discover_all_plugins

        discovered = discover_all_plugins()

        assert len(discovered["sources"]) == 4, (
            f"Expected 4 sources, found {len(discovered['sources'])}: "
            f"{[cls.name for cls in discovered['sources']]}"
        )
        assert len(discovered["transforms"]) == 11, (
            f"Expected 11 transforms, found {len(discovered['transforms'])}: "
            f"{[cls.name for cls in discovered['transforms']]}"
        )
        assert len(discovered["sinks"]) == 4, (
            f"Expected 4 sinks, found {len(discovered['sinks'])}: "
            f"{[cls.name for cls in discovered['sinks']]}"
        )
```

**Step 5: Run tests again**

Run: `pytest tests/ -v`

Expected: All PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "chore(plugins): delete legacy hookimpl registration files

These static registration files are no longer needed.
Dynamic discovery in PluginManager.register_builtin_plugins() replaces them.

Deleted:
- src/elspeth/plugins/sources/hookimpl.py
- src/elspeth/plugins/transforms/hookimpl.py
- src/elspeth/plugins/sinks/hookimpl.py

Also converted test_discovery_matches_hookimpl_counts to use static count
assertions since it can no longer import from the deleted hookimpl files."
```

---

## Task 10: Clean Up Unused Imports in CLI

**Files:**
- Modify: `src/elspeth/cli.py`

**Step 1: Remove direct plugin imports**

Remove any imports like:
```python
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.transforms.passthrough import PassThrough
# etc.
```

These are no longer needed since we use PluginManager.

**Step 2: Run linter**

Run: `.venv/bin/python -m ruff check src/elspeth/cli.py --fix`

**Step 3: Run type checker**

Run: `.venv/bin/python -m mypy src/elspeth/cli.py`

Expected: No errors

**Step 4: Run tests**

Run: `pytest tests/cli/ -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "chore(cli): remove unused direct plugin imports

Plugin classes are now accessed via PluginManager, not direct imports."
```

---

## Task 11: Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v`

Expected: All PASS

**Step 2: Run type checker on new code**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/discovery.py`

Expected: No errors

**Step 3: Verify discovery works end-to-end**

Run:
```bash
.venv/bin/python -c "
from elspeth.plugins.manager import PluginManager
m = PluginManager()
m.register_builtin_plugins()
print('Sources:', sorted([s.name for s in m.get_sources()]))
print('Transforms:', sorted([t.name for t in m.get_transforms()]))
print('Sinks:', sorted([s.name for s in m.get_sinks()]))
print()
print('Total plugins discovered:',
      len(m.get_sources()) + len(m.get_transforms()) + len(m.get_sinks()))
"
```

Expected: Lists all plugins discovered from directories

**Step 4: Verify CLI plugins list**

Run: `.venv/bin/python -m elspeth plugins list`

Expected: Shows all plugins with descriptions

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(plugins): complete dynamic plugin discovery migration

Adding a new plugin now requires only creating the plugin class file.
No manual registration in hookimpl.py or cli.py needed.

Changes:
- Added discovery.py with folder-scanning discovery
- Updated PluginManager to use discovery
- Updated CLI to use PluginManager for all plugin lookups
- Deleted legacy hookimpl.py files
- Plugin descriptions extracted from class docstrings

Before: 4 files to edit when adding a plugin
After: 1 file to create"
```

---

## Verification Checklist

Before considering this complete:

- [ ] `pytest tests/plugins/test_discovery.py -v` - All PASS
- [ ] `pytest tests/plugins/test_manager.py -v` - All PASS
- [ ] `pytest tests/cli/ -v` - All PASS
- [ ] `pytest tests/ -v` - Full suite PASS
- [ ] `.venv/bin/python -m mypy src/elspeth/plugins/discovery.py` - No errors
- [ ] `.venv/bin/python -m elspeth plugins list` - Shows all plugins (sources, transforms, sinks only - gates are NOT plugins)
- [ ] `hookimpl.py` files deleted from sources/, transforms/, sinks/
- [ ] No `TRANSFORM_PLUGINS` or `PLUGIN_REGISTRY` dicts in cli.py
- [ ] `test_discovery_plugin_counts` verifies 4 sources, 11 transforms, 4 sinks
