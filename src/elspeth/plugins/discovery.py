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

logger = logging.getLogger(__name__)

# Files that should never be scanned for plugins
EXCLUDED_FILES: frozenset[str] = frozenset(
    {
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
    }
)


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

        # Must have a `name` attribute with non-empty value
        # NOTE: This getattr is at a PLUGIN DISCOVERY TRUST BOUNDARY - we're scanning
        # arbitrary Python files and can't know at compile time which classes have
        # a `name` attribute. This is legitimate framework-level polymorphism.
        plugin_name = getattr(obj, "name", None)
        if not plugin_name:
            logger.warning(
                "Class %s in %s inherits from %s but has no/empty 'name' attribute - skipping",
                name,
                py_file,
                base_class.__name__,
            )
            continue

        discovered.append(obj)

    return discovered


# Import base classes for PLUGIN_SCAN_CONFIG
# These imports are deferred to avoid circular imports at module load time
def _get_base_classes() -> dict[str, type]:
    """Get base classes for plugin discovery (deferred import)."""
    from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform

    return {
        "sources": BaseSource,
        "transforms": BaseTransform,
        "sinks": BaseSink,
    }


# Configuration: which directories to scan for each plugin type
# NOTE: Non-recursive scanning - subdirectories must be listed explicitly
#
# IMPORTANT: Gates are NOT included here. Per docs/contracts/plugin-protocol.md,
# gates are "config-driven system operations" handled by the engine, NOT plugins.
# See "System Operations (NOT Plugins)" section in the plugin protocol.
PLUGIN_SCAN_CONFIG: dict[str, list[str]] = {
    "sources": ["sources", "azure"],
    "transforms": ["transforms", "transforms/azure", "llm"],
    "sinks": ["sinks", "azure"],
}


def discover_all_plugins() -> dict[str, list[type]]:
    """Discover all built-in plugins by scanning configured directories.

    Returns:
        Dict mapping plugin type to list of discovered plugin classes:
        {
            "sources": [CSVSource, JSONSource, ...],
            "transforms": [PassThrough, FieldMapper, ...],
            "sinks": [CSVSink, JSONSink, ...],
        }
    """
    plugins_root = Path(__file__).parent
    base_classes = _get_base_classes()
    result: dict[str, list[type]] = {}

    for plugin_type, directories in PLUGIN_SCAN_CONFIG.items():
        base_class = base_classes[plugin_type]

        all_discovered: list[type] = []
        seen_names: set[str] = set()

        for dir_name in directories:
            directory = plugins_root / dir_name
            discovered = discover_plugins_in_directory(directory, base_class)

            for cls in discovered:
                # Skip duplicates (same class discovered from different logic)
                # NOTE: cls.name is guaranteed to exist by _discover_in_file validation
                cls_name: str = cls.name  # type: ignore[attr-defined]
                if cls_name in seen_names:
                    continue
                seen_names.add(cls_name)
                all_discovered.append(cls)

        result[plugin_type] = all_discovered

    return result


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
