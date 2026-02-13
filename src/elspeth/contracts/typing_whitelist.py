# src/elspeth/contracts/typing_whitelist.py
"""Typing whitelist: locations where dict[str, Any] or Any is intentional.

This module documents every place in the codebase where soft typing
(dict[str, Any], Any parameters, etc.) is a deliberate design choice
rather than technical debt. If a location is not listed here, its use
of Any should be considered a bug to fix.

Categories:
1. DYNAMIC_SCHEMA - User opted out of type checking via schema mode
2. PLUGIN_CONFIG - Plugin-specific config validated by each plugin's Pydantic model
3. SERIALIZATION_BOUNDARIES - Must handle arbitrary Python objects
4. FRAMEWORK_INTEROP - Third-party library constraints

Each entry includes:
- location: file:symbol
- type_used: The soft type in use
- justification: Why full typing is impossible or counterproductive
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WhitelistEntry:
    """A single whitelisted soft-typing location."""

    location: str
    type_used: str
    justification: str


# === 1. DYNAMIC SCHEMA ===
# When schema mode is "observed" or "dynamic", the engine cannot know field
# names or types at construction time. The data shape is determined by the
# first row at runtime. These locations genuinely cannot be typed statically.

DYNAMIC_SCHEMA_LOCATIONS: tuple[WhitelistEntry, ...] = (
    WhitelistEntry(
        location="contracts/schema_contract.py:PipelineRow",
        type_used="dict[str, Any] (internal _data)",
        justification=(
            "PipelineRow wraps row data whose fields depend on the source schema. "
            "In dynamic/observed mode, fields are unknown until the first row arrives. "
            "The SchemaContract attached to PipelineRow carries the type metadata."
        ),
    ),
    WhitelistEntry(
        location="contracts/results.py:TransformResult.row",
        type_used="dict[str, Any] | PipelineRow | None",
        justification=(
            "Transform output row is a dict because transforms may add/remove fields. "
            "The contract on TransformResult describes the output schema separately."
        ),
    ),
    WhitelistEntry(
        location="contracts/results.py:SourceRow.row",
        type_used="dict[str, Any]",
        justification=(
            "Source rows are external data (Tier 3). The dict shape depends on the "
            "source file/API and is validated against the schema contract at the boundary."
        ),
    ),
)

# === 2. PLUGIN CONFIG ===
# Plugin configuration dicts vary per plugin type. Each plugin validates its
# own config through Pydantic models (PluginConfig hierarchy in config_base.py).
# The graph and engine treat these as opaque JSON-serializable blobs.

PLUGIN_CONFIG_LOCATIONS: tuple[WhitelistEntry, ...] = (
    WhitelistEntry(
        location="core/dag/models.py:NodeConfig",
        type_used="dict[str, Any] (TypeAlias)",
        justification=(
            "Node config varies by node type: source configs have path/delimiter, "
            "transform configs have model/temperature, coalesce configs have branches/policy. "
            "Only 'schema' key is accessed cross-type. The graph layer hashes configs for "
            "deterministic node IDs but does not interpret plugin-specific keys."
        ),
    ),
    WhitelistEntry(
        location="plugins/base.py:BaseTransform.__init__(config)",
        type_used="dict[str, Any]",
        justification=(
            "Plugin config is passed from YAML through Pydantic validation (per-plugin "
            "Settings model) before reaching the plugin. The base class stores it as "
            "dict[str, Any] because the keys are plugin-specific. Each plugin accesses "
            "its own validated keys directly."
        ),
    ),
    WhitelistEntry(
        location="core/config.py:TransformSettings.options",
        type_used="dict[str, Any]",
        justification=(
            "Plugin options in pipeline YAML are arbitrary per plugin. TransformSettings "
            "passes them to the plugin's Pydantic config model for validation. The Settings "
            "layer cannot type-check plugin-specific options without coupling to every plugin."
        ),
    ),
)

# === 3. SERIALIZATION BOUNDARIES ===
# Canonical JSON, payload storage, and audit export must serialize arbitrary
# Python objects. These boundaries handle pandas types, numpy scalars,
# datetimes, Decimals, etc. Full typing is impossible because the input
# domain is "any JSON-serializable Python value".

SERIALIZATION_BOUNDARY_LOCATIONS: tuple[WhitelistEntry, ...] = (
    WhitelistEntry(
        location="core/canonical.py:canonical_json(obj)",
        type_used="Any",
        justification=(
            "Canonical JSON must serialize any JSON-compatible Python object: "
            "dicts, lists, strings, numbers, booleans, None, plus pandas/numpy types. "
            "Phase 1 normalization handles the type zoo; Phase 2 (rfc8785) handles "
            "deterministic serialization."
        ),
    ),
    WhitelistEntry(
        location="core/landscape/recorder.py (JSON columns)",
        type_used="str (JSON) → dict[str, Any]",
        justification=(
            "Landscape stores context_before_json, context_after_json, success_reason_json, "
            "error_json as JSON text columns. These contain plugin-produced metadata whose "
            "shape varies per plugin and per operation type."
        ),
    ),
)

# === 4. FRAMEWORK INTEROP ===
# Third-party libraries impose their own typing constraints. These cannot
# be fixed without upstream changes.

FRAMEWORK_INTEROP_LOCATIONS: tuple[WhitelistEntry, ...] = (
    WhitelistEntry(
        location="plugins/hookspecs.py (all hookspec methods)",
        type_used="type: ignore[empty-body]",
        justification=(
            "pluggy hookspecs must have empty bodies with return type annotations. "
            "mypy reports empty-body because the method has no implementation; the body "
            "is provided by hookimpl implementations at runtime."
        ),
    ),
    WhitelistEntry(
        location="mcp/server.py, testing/chaosllm_mcp/server.py",
        type_used="type: ignore[misc, untyped-decorator]",
        justification=(
            "MCP SDK @server.list_tools() and @server.call_tool() decorators lack "
            "type stubs. The MCP Python SDK (mcp package) does not ship py.typed."
        ),
    ),
)
