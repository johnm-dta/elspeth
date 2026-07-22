"""Dynamic plugin discovery by folder scanning.

Scans plugin directories for classes that:
1. Inherit from a base class (BaseSource, BaseTransform, etc.)
2. Have a `name` class attribute
3. Are not abstract (no @abstractmethod methods without implementation)
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

OPTIONAL_PLUGIN_IMPORT_MODULES: frozenset[str] = frozenset(
    {
        "bs4",
        "chromadb",
        "html2text",
        "jinja2",
    }
)

# Files that should never be scanned for plugins
EXCLUDED_FILES: frozenset[str] = frozenset(
    {
        "__init__.py",
        "hookimpl.py",
        "base.py",
        "templates.py",
        "auth.py",
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
        "multi_query.py",
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
        raise FileNotFoundError(
            f"Plugin directory does not exist: {directory}. "
            f"This indicates a broken installation — the plugins/ directory "
            f"structure is required."
        )

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name in EXCLUDED_FILES:
            continue

        # Plugin code is SYSTEM-OWNED, not user-provided.
        # Most import errors indicate bugs in our code — crash immediately.
        # ImportError from a missing optional extra (e.g., chromadb for [rag])
        # is expected when not all extras are installed — skip with a warning.
        try:
            plugins = _discover_in_file(py_file, base_class)
        except ModuleNotFoundError as exc:
            if not _is_optional_plugin_dependency_error(exc):
                raise
            logger.warning(
                "Skipping plugin file %s: optional dependency not installed (%s)",
                py_file.name,
                exc,
            )
            continue
        except ImportError:
            raise
        discovered.extend(plugins)

    return discovered


def _is_optional_plugin_dependency_error(exc: ModuleNotFoundError) -> bool:
    """Return whether a missing import is an expected optional plugin extra."""
    missing_name = exc.name
    if missing_name is None:
        return False
    missing_root = missing_name.split(".", 1)[0]
    return missing_root in OPTIONAL_PLUGIN_IMPORT_MODULES


def _canonical_module_name(py_file: Path) -> str | None:
    """Derive the canonical dotted module name for a plugin file.

    Walks up from *py_file* to find the ``elspeth`` package root (the
    directory whose ``__init__.py`` defines the ``elspeth`` package).
    Returns the dotted path (e.g. ``elspeth.plugins.transforms.llm.azure_batch``)
    or ``None`` if the file is not under the ``elspeth`` package tree.
    """
    parts: list[str] = [py_file.stem]
    current = py_file.parent
    while True:
        init = current / "__init__.py"
        if not init.exists():
            break
        parts.append(current.name)
        if current.name == "elspeth":
            parts.reverse()
            return ".".join(parts)
        current = current.parent
    return None


def _bind_canonical_module_to_parent(canonical_name: str, module: object) -> None:
    """Complete the package binding performed by ordinary imports.

    Inserting a dotted name directly into ``sys.modules`` does not expose the
    child on its parent package.  Dotted monkeypatch/import traversal expects
    both registrations to identify the same module object.
    """
    parent_name, separator, child_name = canonical_name.rpartition(".")
    if not separator:
        return
    parent = importlib.import_module(parent_name)
    if sys.modules.get(canonical_name) is not module:
        raise RuntimeError(f"Canonical module identity changed while binding {canonical_name!r}")
    setattr(parent, child_name, module)


def _discover_in_file(py_file: Path, base_class: type) -> list[type]:
    """Discover plugin classes in a single Python file.

    Args:
        py_file: Path to Python file
        base_class: Base class that plugins must inherit from

    Returns:
        List of plugin classes found in the file
    """
    # Compute the canonical module name (e.g. elspeth.plugins.transforms.llm.azure_batch).
    # If the module was already imported canonically (e.g. by another plugin's
    # module-level imports), reuse it to avoid dual class objects.
    canonical_name = _canonical_module_name(py_file)
    registered_name: str | None = None
    canonical_registered = False
    if canonical_name is not None and canonical_name in sys.modules:
        module = sys.modules[canonical_name]
    else:
        # Load module from file path
        # Include parent directory in module name to avoid collisions
        # (e.g., transforms/base.py vs llm/base.py would both be "base" otherwise)
        parent_name = py_file.parent.name
        module_name = f"elspeth.plugins._discovered.{parent_name}.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None:
            raise RuntimeError(
                f"Failed to create module spec for {py_file} (module_name={module_name!r}): spec_from_file_location() returned None"
            )
        if spec.loader is None:
            raise RuntimeError(f"Module spec for {py_file} (module_name={module_name!r}) has no loader")

        module = importlib.util.module_from_spec(spec)
        # Register module in sys.modules BEFORE exec_module
        # Required for Python 3.13+ where dataclass decorator looks up
        # cls.__module__ in sys.modules during field resolution.
        # Capture the registration key now: exec_module runs the plugin's
        # module-level code, which could reassign module.__name__, and we must
        # roll back the *exact* key we registered, not whatever __name__ became.
        registered_name = module.__name__
        sys.modules[registered_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Roll back only our own registration. Guard on identity (not mere
            # membership) so we never delete an entry another importer placed
            # under the same name during exec_module reentrancy, and use del
            # (not .pop(..., None)) so the rollback can never mask the original
            # import failure with a spurious KeyError on the re-raise.
            if registered_name in sys.modules and sys.modules[registered_name] is module:
                del sys.modules[registered_name]
            raise

        # Also register under the canonical name so that future standard
        # imports (e.g. `from elspeth.plugins.transforms.llm.transform import X`)
        # find the same module object rather than loading a duplicate.
        if canonical_name is not None and canonical_name not in sys.modules:
            sys.modules[canonical_name] = module
            canonical_registered = True

    if canonical_name is not None:
        try:
            _bind_canonical_module_to_parent(canonical_name, module)
        except Exception:
            if canonical_registered and sys.modules.get(canonical_name) is module:
                del sys.modules[canonical_name]
            if registered_name is not None and sys.modules.get(registered_name) is module:
                del sys.modules[registered_name]
            raise

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

        # Plugin code is system-owned; missing metadata is a bug, not recoverable input.
        try:
            plugin_name = obj.name  # type: ignore[attr-defined]
        except AttributeError as e:
            raise ValueError(
                f"Plugin class {name} in {py_file} inherits from {base_class.__name__} but is missing required class attribute 'name'."
            ) from e
        try:
            normalized_plugin_name = plugin_name.strip()
        except AttributeError as e:
            raise ValueError(
                f"Plugin class {name} in {py_file} inherits from {base_class.__name__} "
                f"but has invalid 'name' value {plugin_name!r}. Expected non-empty string."
            ) from e
        if not normalized_plugin_name:
            raise ValueError(
                f"Plugin class {name} in {py_file} inherits from {base_class.__name__} "
                f"but has invalid 'name' value {plugin_name!r}. Expected non-empty string."
            )

        discovered.append(obj)

    return discovered


# Import base classes for PLUGIN_SCAN_CONFIG
# These imports are deferred to avoid circular imports at module load time
def _get_base_classes() -> dict[str, type]:
    """Get base classes for plugin discovery (deferred import)."""
    from elspeth.plugins.infrastructure.base import BaseSink, BaseSource, BaseTransform

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
    "sources": ["sources"],
    "transforms": ["transforms", "transforms/aws", "transforms/azure", "transforms/llm", "transforms/rag"],
    "sinks": ["sinks"],
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
    plugins_root = Path(__file__).parent.parent  # Go up from infrastructure/ to plugins/
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
                # NOTE: cls.name is guaranteed to exist by _discover_in_file validation
                cls_name: str = cls.name  # type: ignore[attr-defined]  # cls is a BaseSource/BaseTransform/BaseSink subclass; .name validated by _discover_in_file

                # Duplicate plugin names are bugs - crash immediately to surface collision
                if cls_name in seen_names:
                    existing_cls = next(c for c in all_discovered if c.name == cls_name)  # type: ignore[attr-defined]  # same: plugin subclass with .name
                    raise ValueError(
                        f"Duplicate {plugin_type} plugin name '{cls_name}': "
                        f"found in both {existing_cls.__module__} and {cls.__module__}. "
                        f"Plugin names must be unique within each type."
                    )

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
    return f"{plugin_cls.name} plugin"  # type: ignore[attr-defined]  # validated by _discover_in_file


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
    from typing import Any

    from elspeth.plugins.infrastructure.hookspecs import hookimpl

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
