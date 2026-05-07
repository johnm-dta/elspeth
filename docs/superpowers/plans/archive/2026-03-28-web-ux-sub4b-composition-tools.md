# Web UX Task-Plan 4B: Composition Tools

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement ToolResult, 6 discovery tools, 8 mutation tools with catalog validation, S2 path security, and state-reflecting returns
**Parent Plan:** `plans/2026-03-28-web-ux-sub4-composer.md`
**Spec:** `specs/2026-03-28-web-ux-sub4-composer-design.md`
**Depends On:** Task-Plan 4A (Data Models), Sub-Plan 3 (Catalog)
**Blocks:** Task-Plan 4D (Composer Service)

---

### File Map

| Action | File |
|--------|------|
| Create | `src/elspeth/web/composer/tools.py` |
| Create | `tests/unit/web/composer/test_tools.py` |

---

### Task 4: Composition Tools — ToolResult and Mutation Tools

**Files:**
- Create: `src/elspeth/web/composer/tools.py`
- Create: `tests/unit/web/composer/test_tools.py`

- [ ] **Step 1: Write ToolResult and mutation tool tests**

```python
# tests/unit/web/composer/test_tools.py
"""Tests for composition tools — discovery delegation and mutation + validation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginSchemaInfo,
    PluginSummary,
)
from elspeth.web.composer.tools import (
    ToolResult,
    execute_tool,
    get_expression_grammar,
    get_tool_definitions,
)


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None, nodes=(), edges=(), outputs=(),
        metadata=PipelineMetadata(), version=1,
    )


def _mock_catalog() -> MagicMock:
    """Mock CatalogService with real PluginSummary/PluginSchemaInfo instances.

    AC #16: Tests must use real PluginSummary and PluginSchemaInfo instances,
    not plain dicts. Mock return types must match the CatalogService protocol.
    """
    catalog = MagicMock()
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv", description="CSV file source",
            plugin_type="source", config_fields=[
                ConfigFieldSummary(name="path", type="string", required=True, description="File path", default=None),
            ],
        ),
        PluginSummary(
            name="json", description="JSON file source",
            plugin_type="source", config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(
            name="uppercase", description="Uppercase transform",
            plugin_type="transform", config_fields=[],
        ),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(
            name="csv", description="CSV file sink",
            plugin_type="sink", config_fields=[],
        ),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv", plugin_type="source", description="CSV file source",
        json_schema={"title": "CsvSourceConfig", "properties": {"path": {"type": "string"}}},
    )
    return catalog


class TestToolResult:
    def test_frozen(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=False, errors=("err",)),
            affected_nodes=("n1", "n2"),
        )
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    def test_affected_nodes_frozen(self) -> None:
        state = _empty_state()
        from elspeth.web.composer.state import ValidationSummary

        result = ToolResult(
            success=True,
            updated_state=state,
            validation=ValidationSummary(is_valid=True, errors=()),
            affected_nodes=("n1",),
        )
        assert isinstance(result.affected_nodes, tuple)


class TestSetSource:
    def test_sets_source(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/in.csv"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert result.updated_state.source is not None
        assert result.updated_state.source.plugin == "csv"
        assert result.updated_state.version == 2
        assert "source" in result.affected_nodes

    def test_unknown_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: foobar")
        result = execute_tool(
            "set_source",
            {
                "plugin": "foobar",
                "on_success": "t1",
                "options": {},
                "on_validation_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success is False
        assert result.updated_state.source is None  # unchanged
        assert result.updated_state.version == 1


class TestUpsertNode:
    def test_adds_new_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "t1",
                "node_type": "transform",
                "plugin": "uppercase",
                "input": "source_out",
                "on_success": "main",
                "options": {},
            },
            state,
            catalog,
        )
        assert result.success is True
        assert len(result.updated_state.nodes) == 1
        assert "t1" in result.affected_nodes

    def test_replaces_existing_node(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result1 = execute_tool(
            "upsert_node",
            {
                "id": "t1", "node_type": "transform", "plugin": "uppercase",
                "input": "in", "on_success": "out", "options": {},
            },
            state, catalog,
        )
        result2 = execute_tool(
            "upsert_node",
            {
                "id": "t1", "node_type": "transform", "plugin": "uppercase",
                "input": "new_in", "on_success": "out", "options": {"field": "x"},
            },
            result1.updated_state, catalog,
        )
        assert result2.success is True
        assert len(result2.updated_state.nodes) == 1
        assert result2.updated_state.nodes[0].input == "new_in"

    def test_gate_node_no_plugin_validation(self) -> None:
        """Gates don't have plugins — should not validate against catalog."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_node",
            {
                "id": "g1", "node_type": "gate", "plugin": None,
                "input": "in", "on_success": None, "options": {},
                "condition": "row['x'] > 0",
                "routes": {"high": "s1", "low": "s2"},
            },
            state, catalog,
        )
        assert result.success is True
        catalog.get_schema.assert_not_called()


class TestUpsertEdge:
    def test_adds_edge(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "upsert_edge",
            {
                "id": "e1", "from_node": "source", "to_node": "t1",
                "edge_type": "on_success", "label": None,
            },
            state, catalog,
        )
        assert result.success is True
        assert len(result.updated_state.edges) == 1
        assert "source" in result.affected_nodes
        assert "t1" in result.affected_nodes


class TestRemoveNode:
    def test_removes_node_and_edges(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        # Add a node and an edge to it
        r1 = execute_tool(
            "upsert_node",
            {
                "id": "t1", "node_type": "transform", "plugin": "uppercase",
                "input": "in", "on_success": "out", "options": {},
            },
            state, catalog,
        )
        r2 = execute_tool(
            "upsert_edge",
            {
                "id": "e1", "from_node": "source", "to_node": "t1",
                "edge_type": "on_success", "label": None,
            },
            r1.updated_state, catalog,
        )
        # Remove the node — edge should also be removed
        r3 = execute_tool("remove_node", {"id": "t1"}, r2.updated_state, catalog)
        assert r3.success is True
        assert len(r3.updated_state.nodes) == 0
        assert len(r3.updated_state.edges) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_node", {"id": "nope"}, state, catalog)
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged


class TestRemoveEdge:
    def test_removes_edge(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "upsert_edge",
            {
                "id": "e1", "from_node": "source", "to_node": "t1",
                "edge_type": "on_success", "label": None,
            },
            state, catalog,
        )
        r2 = execute_tool("remove_edge", {"id": "e1"}, r1.updated_state, catalog)
        assert r2.success is True
        assert len(r2.updated_state.edges) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_edge", {"id": "nope"}, state, catalog)
        assert result.success is False


class TestSetMetadata:
    def test_partial_update(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_metadata", {"patch": {"name": "My Pipeline"}}, state, catalog,
        )
        assert result.success is True
        assert result.updated_state.metadata.name == "My Pipeline"
        assert result.updated_state.metadata.description == ""  # preserved
        assert result.affected_nodes == ()  # metadata doesn't affect nodes


class TestSetOutput:
    def test_adds_output(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main",
                "plugin": "csv",
                "options": {"path": "/data/out.csv"},
                "on_write_failure": "discard",
            },
            state,
            catalog,
        )
        assert result.success is True
        assert len(result.updated_state.outputs) == 1
        assert result.updated_state.outputs[0].name == "main"
        assert result.updated_state.outputs[0].plugin == "csv"
        assert result.updated_state.version == 2
        assert "main" in result.affected_nodes

    def test_replaces_existing_output(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "set_output",
            {
                "sink_name": "main", "plugin": "csv",
                "options": {}, "on_write_failure": "discard",
            },
            state, catalog,
        )
        r2 = execute_tool(
            "set_output",
            {
                "sink_name": "main", "plugin": "csv",
                "options": {"path": "/new.csv"}, "on_write_failure": "quarantine",
            },
            r1.updated_state, catalog,
        )
        assert r2.success is True
        assert len(r2.updated_state.outputs) == 1
        assert r2.updated_state.outputs[0].on_write_failure == "quarantine"

    def test_unknown_sink_plugin_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        catalog.get_schema.side_effect = ValueError("Unknown plugin: foobar")
        result = execute_tool(
            "set_output",
            {
                "sink_name": "main", "plugin": "foobar",
                "options": {}, "on_write_failure": "discard",
            },
            state, catalog,
        )
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged


class TestRemoveOutput:
    def test_removes_output(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        r1 = execute_tool(
            "set_output",
            {
                "sink_name": "main", "plugin": "csv",
                "options": {}, "on_write_failure": "discard",
            },
            state, catalog,
        )
        r2 = execute_tool("remove_output", {"sink_name": "main"}, r1.updated_state, catalog)
        assert r2.success is True
        assert len(r2.updated_state.outputs) == 0

    def test_remove_nonexistent_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("remove_output", {"sink_name": "nope"}, state, catalog)
        assert result.success is False
        assert result.updated_state.version == 1  # unchanged


class TestSetSourcePathSecurity:
    """S2: Source path allowlist — paths must be under {data_dir}/uploads/."""

    def test_path_under_uploads_succeeds(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/uploads/input.csv"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is True

    def test_path_outside_uploads_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/etc/passwd"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False
        assert "path" in result.data["error"].lower()

    def test_traversal_attack_fails(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"path": "/data/uploads/../../etc/passwd"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_file_key_also_validated(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"file": "/tmp/evil.csv"},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is False

    def test_no_path_key_skips_validation(self) -> None:
        """Source options without path/file keys are not subject to S2."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv",
                "on_success": "t1",
                "options": {"connection_string": "postgres://..."},
                "on_validation_failure": "quarantine",
            },
            state,
            catalog,
            data_dir="/data",
        )
        assert result.success is True


class TestDiscoveryTools:
    def test_list_sources_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_sources", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_sources.assert_called_once()

    def test_list_transforms_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_transforms", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_transforms.assert_called_once()

    def test_list_sinks_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool("list_sinks", {}, _empty_state(), catalog)
        assert result.success is True
        catalog.list_sinks.assert_called_once()

    def test_get_plugin_schema_delegates(self) -> None:
        catalog = _mock_catalog()
        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "csv"},
            _empty_state(), catalog,
        )
        assert result.success is True
        catalog.get_schema.assert_called_once_with("source", "csv")

    def test_get_expression_grammar_is_static(self) -> None:
        grammar = get_expression_grammar()
        assert "row" in grammar
        assert isinstance(grammar, str)

    def test_get_current_state_returns_state(self) -> None:
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool("get_current_state", {}, state, catalog)
        assert result.success is True
        # State is unchanged
        assert result.updated_state.version == 1


class TestToolDefinitions:
    def test_has_fourteen_tools(self) -> None:
        """6 discovery + 8 mutation = 14 tools."""
        defs = get_tool_definitions()
        assert len(defs) == 14

    def test_all_have_json_schema(self) -> None:
        for defn in get_tool_definitions():
            assert "name" in defn
            assert "description" in defn
            assert "parameters" in defn


class TestToolResultValidation:
    def test_mutation_includes_validation(self) -> None:
        """Every mutation tool result includes validation summary."""
        state = _empty_state()
        catalog = _mock_catalog()
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv", "on_success": "t1",
                "options": {}, "on_validation_failure": "quarantine",
            },
            state, catalog,
        )
        assert result.validation is not None
        # Source is set but no sinks — should have validation error
        assert not result.validation.is_valid
        assert any("No sinks" in e for e in result.validation.errors)
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ToolResult and tool executor**

```python
# src/elspeth/web/composer/tools.py
"""Composition tools — discovery and mutation tools for the LLM composer.

Discovery tools delegate to CatalogService. Mutation tools modify
CompositionState and return ToolResult with validation.

Layer: L3 (application). Imports from L0 (contracts.freeze) and
L3 (web/composer/state).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.catalog.protocol import CatalogService

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)


CatalogServiceProtocol = CatalogService  # Re-export for local use


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the operation succeeded.
        updated_state: Full state after mutation (or original if success=False).
        validation: Stage 1 validation result for the updated state.
        affected_nodes: Node IDs changed or with changed edges.
        data: Optional data payload for discovery tools.
    """

    success: bool
    updated_state: CompositionState
    validation: ValidationSummary
    affected_nodes: tuple[str, ...]
    data: Any = None

    def __post_init__(self) -> None:
        freeze_fields(self, "affected_nodes")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for LLM tool response."""
        result: dict[str, Any] = {
            "success": self.success,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": list(self.validation.errors),
            },
            "affected_nodes": list(self.affected_nodes),
            "version": self.updated_state.version,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


# --- Expression Grammar (static) ---

_EXPRESSION_GRAMMAR = """\
Gate Expression Syntax Reference
=================================

Variables:
  row      - The current row as a dict. Access fields via row['field_name'].

Operators:
  ==, !=, <, >, <=, >=   Comparison
  and, or, not            Boolean logic
  in, not in              Membership test
  +, -, *, /, //, %       Arithmetic

Built-in functions:
  len(), str(), int(), float(), bool(), abs(), min(), max(), round()
  isinstance(), type()

Examples:
  row['confidence'] >= 0.85
  row['status'] == 'approved'
  row['category'] in ('A', 'B', 'C')
  len(row.get('tags', [])) > 0
  row['score'] > 0.5 and row['status'] != 'rejected'

Security:
  Expressions are validated by ExpressionParser. Attribute access, imports,
  function calls to non-builtins, and dunder access are forbidden.
"""


def get_expression_grammar() -> str:
    """Return the gate expression syntax reference."""
    return _EXPRESSION_GRAMMAR


# --- Tool Definitions for LLM ---

def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool definitions for the LLM.

    Returns 14 tools: 6 discovery, 8 mutation.
    """
    return [
        # Discovery tools
        {
            "name": "list_sources",
            "description": "List available source plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_transforms",
            "description": "List available transform plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_sinks",
            "description": "List available sink plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_plugin_schema",
            "description": "Get the full configuration schema for a plugin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_type": {
                        "type": "string",
                        "enum": ["source", "transform", "sink"],
                        "description": "Plugin type.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Plugin name (e.g. 'csv').",
                    },
                },
                "required": ["plugin_type", "name"],
            },
        },
        {
            "name": "get_expression_grammar",
            "description": "Get the gate expression syntax reference.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_current_state",
            "description": "Get the full current pipeline composition state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        # Mutation tools
        {
            "name": "set_source",
            "description": "Set or replace the pipeline source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin": {"type": "string", "description": "Source plugin name."},
                    "on_success": {"type": "string", "description": "Connection name for downstream."},
                    "options": {"type": "object", "description": "Plugin-specific config."},
                    "on_validation_failure": {
                        "type": "string",
                        "enum": ["discard", "quarantine"],
                        "description": "How to handle validation failures.",
                    },
                },
                "required": ["plugin", "on_success", "options", "on_validation_failure"],
            },
        },
        {
            "name": "upsert_node",
            "description": "Add or update a pipeline node (transform, gate, aggregation, coalesce).",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique node identifier."},
                    "node_type": {
                        "type": "string",
                        "enum": ["transform", "gate", "aggregation", "coalesce"],
                    },
                    "plugin": {"type": ["string", "null"], "description": "Plugin name. Null for gates/coalesces."},
                    "input": {"type": "string", "description": "Input connection name."},
                    "on_success": {"type": ["string", "null"], "description": "Output connection. Null for gates."},
                    "on_error": {"type": ["string", "null"], "description": "Error output connection."},
                    "options": {"type": "object", "description": "Plugin-specific config."},
                    "condition": {"type": ["string", "null"], "description": "Gate expression."},
                    "routes": {"type": ["object", "null"], "description": "Gate route mapping."},
                    "fork_to": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Fork destinations."},
                    "branches": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Coalesce branch inputs."},
                    "policy": {"type": ["string", "null"], "description": "Coalesce policy."},
                    "merge": {"type": ["string", "null"], "description": "Coalesce merge strategy."},
                },
                "required": ["id", "node_type", "input"],
            },
        },
        {
            "name": "upsert_edge",
            "description": "Add or update a connection between nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique edge identifier."},
                    "from_node": {"type": "string", "description": "Source node ID or 'source'."},
                    "to_node": {"type": "string", "description": "Destination node ID or sink name."},
                    "edge_type": {
                        "type": "string",
                        "enum": ["on_success", "on_error", "route_true", "route_false", "fork"],
                    },
                    "label": {"type": ["string", "null"], "description": "Display label."},
                },
                "required": ["id", "from_node", "to_node", "edge_type"],
            },
        },
        {
            "name": "remove_node",
            "description": "Remove a node and all its edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "remove_edge",
            "description": "Remove an edge by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Edge ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "set_metadata",
            "description": "Update pipeline metadata (name and description only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "object",
                        "description": "Partial metadata update. Only included fields are changed.",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "required": ["patch"],
            },
        },
        {
            "name": "set_output",
            "description": "Add or replace a pipeline output (sink).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {"type": "string", "description": "Sink name (connection point for edges/routes)."},
                    "plugin": {"type": "string", "description": "Sink plugin name (e.g. 'csv', 'json')."},
                    "options": {"type": "object", "description": "Plugin-specific config."},
                    "on_write_failure": {
                        "type": "string",
                        "enum": ["discard", "quarantine"],
                        "description": "How to handle write failures.",
                        "default": "discard",
                    },
                },
                "required": ["sink_name", "plugin", "options"],
            },
        },
        {
            "name": "remove_output",
            "description": "Remove a pipeline output (sink) by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {"type": "string", "description": "Sink name to remove."},
                },
                "required": ["sink_name"],
            },
        },
    ]


# --- State serialization ---

def _serialize_state(state: CompositionState) -> dict[str, Any]:
    """Serialize CompositionState to a JSON-compatible dict.

    Delegates to state.to_dict() which recursively unwraps frozen
    containers (MappingProxyType -> dict, tuple -> list).
    """
    return state.to_dict()


# --- Tool Executor ---

def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    """Execute a composition tool by name.

    Discovery tools return data without modifying state.
    Mutation tools return ToolResult with updated state and validation.
    Invalid tool names return a failure result with an error message.

    Args:
        data_dir: Base data directory for S2 path allowlist enforcement.
            When provided, source options containing ``path`` or ``file``
            keys are restricted to ``{data_dir}/uploads/``.
    """
    # Discovery tools
    if tool_name == "list_sources":
        return _discovery_result(state, catalog.list_sources())

    if tool_name == "list_transforms":
        return _discovery_result(state, catalog.list_transforms())

    if tool_name == "list_sinks":
        return _discovery_result(state, catalog.list_sinks())

    if tool_name == "get_plugin_schema":
        try:
            schema = catalog.get_schema(arguments["plugin_type"], arguments["name"])
            return _discovery_result(state, schema)
        except (ValueError, KeyError) as exc:
            return _failure_result(state, str(exc))

    if tool_name == "get_expression_grammar":
        return _discovery_result(state, get_expression_grammar())

    if tool_name == "get_current_state":
        serialized = _serialize_state(state)
        validation = state.validate()
        serialized["validation"] = {
            "is_valid": validation.is_valid,
            "errors": list(validation.errors),
        }
        return _discovery_result(state, serialized)

    # Mutation tools
    if tool_name == "set_source":
        return _execute_set_source(arguments, state, catalog, data_dir)

    if tool_name == "upsert_node":
        return _execute_upsert_node(arguments, state, catalog)

    if tool_name == "upsert_edge":
        return _execute_upsert_edge(arguments, state)

    if tool_name == "remove_node":
        return _execute_remove_node(arguments, state)

    if tool_name == "remove_edge":
        return _execute_remove_edge(arguments, state)

    if tool_name == "set_metadata":
        return _execute_set_metadata(arguments, state)

    if tool_name == "set_output":
        return _execute_set_output(arguments, state, catalog)

    if tool_name == "remove_output":
        return _execute_remove_output(arguments, state)

    return _failure_result(state, f"Unknown tool: {tool_name}")


def _discovery_result(state: CompositionState, data: Any) -> ToolResult:
    """Build a ToolResult for a discovery (read-only) tool."""
    validation = state.validate()
    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=data,
    )


def _failure_result(
    state: CompositionState,
    error_msg: str,
) -> ToolResult:
    """Build a ToolResult for a failed mutation."""
    validation = state.validate()
    return ToolResult(
        success=False,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data={"error": error_msg},
    )


def _mutation_result(
    new_state: CompositionState,
    affected: tuple[str, ...],
) -> ToolResult:
    """Build a ToolResult for a successful mutation."""
    validation = new_state.validate()
    return ToolResult(
        success=True,
        updated_state=new_state,
        validation=validation,
        affected_nodes=affected,
    )


def _validate_source_path(
    options: dict[str, Any],
    data_dir: str | None,
) -> str | None:
    """S2: Validate that path/file options are under {data_dir}/uploads/.

    Returns an error message if validation fails, None if OK.
    Uses Path.resolve() + is_relative_to() to defeat ../ traversal.
    """
    if data_dir is None:
        return None

    uploads_dir = Path(data_dir).resolve() / "uploads"

    for key in ("path", "file"):
        if key in options:
            resolved = Path(options[key]).resolve()
            if not resolved.is_relative_to(uploads_dir):
                return (
                    f"Path violation (S2): '{options[key]}' is outside the "
                    f"allowed directory '{uploads_dir}'. Source file paths "
                    f"must be under {{data_dir}}/uploads/."
                )
    return None


def _execute_set_source(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    """Set or replace the pipeline source."""
    plugin = args["plugin"]
    # Validate plugin exists in catalog
    try:
        catalog.get_schema("source", plugin)
    except (ValueError, KeyError) as exc:
        return _failure_result(state, f"Unknown source plugin '{plugin}': {exc}")

    # S2: Validate source path allowlist
    options = args.get("options", {})
    path_error = _validate_source_path(options, data_dir)
    if path_error is not None:
        return _failure_result(state, path_error)

    source = SourceSpec(
        plugin=plugin,
        on_success=args["on_success"],
        options=options,
        on_validation_failure=args.get("on_validation_failure", "quarantine"),
    )
    new_state = state.with_source(source)
    return _mutation_result(new_state, ("source",))


def _execute_upsert_node(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
) -> ToolResult:
    """Add or update a pipeline node."""
    node_type = args["node_type"]
    plugin = args.get("plugin")

    # Validate plugin for types that require one
    if node_type in ("transform", "aggregation") and plugin is not None:
        try:
            catalog.get_schema("transform", plugin)
        except (ValueError, KeyError) as exc:
            return _failure_result(
                state, f"Unknown {node_type} plugin '{plugin}': {exc}"
            )

    fork_to = args.get("fork_to")
    if fork_to is not None:
        fork_to = tuple(fork_to)

    branches = args.get("branches")
    if branches is not None:
        branches = tuple(branches)

    node = NodeSpec(
        id=args["id"],
        node_type=node_type,
        plugin=plugin,
        input=args["input"],
        on_success=args.get("on_success"),
        on_error=args.get("on_error"),
        options=args.get("options", {}),
        condition=args.get("condition"),
        routes=args.get("routes"),
        fork_to=fork_to,
        branches=branches,
        policy=args.get("policy"),
        merge=args.get("merge"),
    )

    node_id = args["id"]
    new_state = state.with_node(node)

    # Affected: the node itself plus nodes with edges referencing it
    affected = {node_id}
    for edge in new_state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_upsert_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Add or update an edge."""
    edge = EdgeSpec(
        id=args["id"],
        from_node=args["from_node"],
        to_node=args["to_node"],
        edge_type=args["edge_type"],
        label=args.get("label"),
    )
    new_state = state.with_edge(edge)
    return _mutation_result(new_state, (args["from_node"], args["to_node"]))


def _execute_remove_node(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove a node and its edges."""
    node_id = args["id"]

    # Collect affected nodes before removal (edges that reference this node)
    affected = {node_id}
    for edge in state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    new_state = state.without_node(node_id)
    if new_state is None:
        return _failure_result(state, f"Node '{node_id}' not found.")

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_remove_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove an edge."""
    edge_id = args["id"]

    # Find the edge to get affected nodes
    edge = next((e for e in state.edges if e.id == edge_id), None)
    if edge is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    affected = (edge.from_node, edge.to_node)
    new_state = state.without_edge(edge_id)
    if new_state is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    return _mutation_result(new_state, affected)


def _execute_set_metadata(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Update pipeline metadata."""
    patch = args.get("patch", args)
    # If the LLM passes fields directly instead of under "patch"
    if "patch" in args and isinstance(args["patch"], dict):
        patch = args["patch"]

    new_state = state.with_metadata(patch)
    return _mutation_result(new_state, ())


def _execute_set_output(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
) -> ToolResult:
    """Add or replace a pipeline output (sink)."""
    plugin = args["plugin"]
    # Validate plugin exists in catalog
    try:
        catalog.get_schema("sink", plugin)
    except (ValueError, KeyError) as exc:
        return _failure_result(state, f"Unknown sink plugin '{plugin}': {exc}")

    output = OutputSpec(
        name=args["sink_name"],
        plugin=plugin,
        options=args.get("options", {}),
        on_write_failure=args.get("on_write_failure", "discard"),
    )
    new_state = state.with_output(output)
    return _mutation_result(new_state, (args["sink_name"],))


def _execute_remove_output(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove a pipeline output (sink) by name."""
    sink_name = args["sink_name"]
    new_state = state.without_output(sink_name)
    if new_state is None:
        return _failure_result(state, f"Output '{sink_name}' not found.")
    return _mutation_result(new_state, (sink_name,))
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/tools.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "feat(web/composer): add composition tools — 6 discovery, 8 mutation, ToolResult, S2 path security"
```

---

## Self-Review Checklist

After completing all steps, verify:

- [ ] `ToolResult` is frozen with `affected_nodes` deep-frozen via `freeze_fields()`. `pytest tests/unit/web/composer/test_tools.py`
- [ ] All 14 tools (6 discovery, 8 mutation) work. Mutations return `ToolResult` with validation. Invalid input returns `success=False`, not exceptions.
- [ ] Discovery tools delegate to `CatalogService` — `list_sources`, `list_transforms`, `list_sinks`, `get_plugin_schema`, `get_expression_grammar`, `get_current_state`.
- [ ] Mutation tools validate against catalog — `set_source` and `upsert_node` (for transform/aggregation types) call `catalog.get_schema()` before mutating state.
- [ ] Gate nodes skip plugin validation (`catalog.get_schema` not called).
- [ ] `remove_node` cascades edge removal. `remove_edge` finds affected nodes before removal.
- [ ] Nonexistent node/edge removal returns `success=False` with unchanged state.
- [ ] Every mutation result includes `ValidationSummary` from `state.validate()`.
- [ ] `ToolResult.to_dict()` serializes to LLM-friendly dict with `success`, `validation`, `affected_nodes`, `version`, and optional `data`.
- [ ] `get_tool_definitions()` returns exactly 14 tool definitions, each with `name`, `description`, and `parameters` (JSON Schema).
- [ ] `set_output` validates sink plugin against catalog and delegates to `state.with_output()`. `remove_output` delegates to `state.without_output()` and returns `success=False` for nonexistent outputs.
- [ ] S2 path allowlist enforced in `_execute_set_source`: options containing `path` or `file` keys validated via `Path.resolve()` + `is_relative_to()`. Traversal attacks (`../`) defeated. Returns `ToolResult(success=False)` on violation.
- [ ] All mock catalogs in tests use real `PluginSummary` and `PluginSchemaInfo` instances, not plain dicts (AC #16).
- [ ] mypy passes on `src/elspeth/web/composer/tools.py`.

```bash
# Test suite
.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v

# Type checking
.venv/bin/python -m mypy src/elspeth/web/composer/tools.py

# Freeze guard CI check
.venv/bin/python scripts/cicd/enforce_freeze_guards.py
```

---

## Review Amendments

### Amendment 1: `set_output` and `remove_output` mutation tools

**Date:** 2026-03-28
**Reason:** The CompositionState (Sub-4a) defines `with_output()` and
`without_output()` methods, but the original plan had no LLM tools exposing
them. Without these tools, the LLM cannot add sinks to a pipeline, and Stage 1
validation will always fail with "No sinks configured."

**Changes:**
- Added `set_output` tool definition (parameters: `sink_name`, `plugin`,
  `options`, `on_write_failure`) to the MUTATION_TOOLS list in
  `get_tool_definitions()`.
- Added `remove_output` tool definition (parameter: `sink_name`).
- Added `_execute_set_output(args, state, catalog)` -- validates sink plugin
  against catalog via `get_schema("sink", plugin)`, then delegates to
  `state.with_output(OutputSpec(...))`.
- Added `_execute_remove_output(args, state)` -- delegates to
  `state.without_output(sink_name)`, returns `success=False` if output not found.
- Added dispatch entries in `execute_tool()` for both new tools.
- Added `TestSetOutput` and `TestRemoveOutput` test classes.
- Updated tool count from 12 to 14 (6 discovery + 8 mutation).

### Amendment 2: S2 source path allowlist in `set_source`

**Date:** 2026-03-28
**Reason:** Security rule S2 (R4 expert panel review) requires that source
plugin options containing `path` or `file` keys are restricted to paths under
`{WebSettings.data_dir}/uploads/`. This prevents prompt injection attacks that
trick the LLM into configuring a source that reads arbitrary server-side files.
Documented in `docs/superpowers/meta/web-ux-program.md` and
`docs/superpowers/specs/2026-03-28-web-ux-seam-contracts.md` (Seam H).

**Changes:**
- Added `_validate_source_path(options, data_dir)` helper that checks `path`
  and `file` keys against `{data_dir}/uploads/` using `Path.resolve()` +
  `is_relative_to()` to defeat `../` traversal attacks.
- Updated `_execute_set_source` to accept `data_dir` parameter and call
  `_validate_source_path()` before constructing the SourceSpec. Returns
  `ToolResult(success=False)` with a descriptive error on violation.
- Updated `execute_tool()` signature to accept optional `data_dir` parameter,
  passed through to `_execute_set_source`.
- Added `pathlib.Path` import.
- Added `TestSetSourcePathSecurity` test class with 5 cases: path under
  uploads (success), path outside uploads (fail), traversal attack (fail),
  `file` key validation (fail), and no path/file key (skip validation).

### Amendment 3: Replace local `CatalogServiceProtocol` with import (2026-03-29)

**Date:** 2026-03-29
**Reason:** Go/no-go review identified a duplicate protocol definition. The local
`CatalogServiceProtocol` in `tools.py` used `list[Any]` return types, losing the
type safety of the real `CatalogService` protocol in `web.catalog.protocol` (which
returns `list[PluginSummary]` and `PluginSchemaInfo`). Both are L3, so importing
across web subpackages is a valid same-layer dependency, not a layer violation.

**Changes:**
- Removed the local `CatalogServiceProtocol(Protocol)` class from `tools.py`.
- Added `from elspeth.web.catalog.protocol import CatalogService` import.
- Added `CatalogServiceProtocol = CatalogService` re-export alias (preserves
  existing references in `prompts.py` and `service.py` without renaming).
- Removed unused `Protocol` import from `typing`.

### Amendment 4: `ToolResult.data` field — spec deviation note (2026-03-29)

**Date:** 2026-03-29
**Reason:** Go/no-go review identified that the spec defines `ToolResult` with 4
fields (`success`, `updated_state`, `validation`, `affected_nodes`) but the plan
adds `data: Any = None` for discovery tool payloads. This is a necessary extension
— discovery tools need to return their results (plugin lists, schemas, state
serialization) and `ToolResult` is the unified return type for all tools.

**Spec update needed:** Add `data` field to the ToolResult definition in
`specs/2026-03-28-web-ux-sub4-composer-design.md`. Description: "Optional data
payload for discovery tools. None for mutation tools."

---

## Round 5 Review Findings

**No blocking issues.**

### Warnings (implement during execution)

**W-4B-1: `upsert_node` with unknown transform plugin untested.**
`test_unknown_plugin_fails` covers `set_source`. `test_unknown_sink_plugin_fails`
covers `set_output`. But `upsert_node` with a nonexistent transform plugin name
has no corresponding test, leaving a gap in the "LLM hallucinates a plugin name"
scenario for the most common mutation tool. Add
`test_upsert_node_unknown_transform_plugin_fails` that calls `upsert_node` with
`plugin="nonexistent_xyz"` and asserts `success=False`.

**W-4B-2: `file` key traversal via S2 not tested.**
The `TestSetSourcePathSecurity` class tests `path` key traversal but not `file`
key traversal via a path that starts correctly then escapes (e.g.,
`{data_dir}/uploads/../../etc/passwd`). Add
`test_file_key_traversal_via_uploads_prefix_fails` that uses a path starting
with the uploads directory but using `../` to escape.

**W-4B-3: `_serialize_state()` is a trivial wrapper.**
The function is a one-liner wrapper for `state.to_dict()` that exists only for a
docstring comment about frozen container handling. Either inline it (it has
exactly one call site) or give it a substantive contract. As written it is an
abstraction with no content.

### Dependency correction (W7)

The plan header previously said "Blocks: Task-Plan 4C (YAML Generator)" but 4C
only imports from `elspeth.web.composer.state` (4A output), not from `tools.py`
(4B output). **4B does not block 4C.** 4B and 4C can run in parallel after 4A
completes. Header updated to "Blocks: Task-Plan 4D (Composer Service)" since 4D
needs the tool executor from 4B.
