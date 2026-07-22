"""Dispatcher boundary regressions for LLM-supplied tool calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import _dispatch as dispatch_module
from elspeth.web.composer.tools._common import ToolContext, ToolResult
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV file source", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV file source",
        json_schema={"title": "CsvSourceConfig", "properties": {"path": {"type": "string"}}},
        knob_schema={"fields": []},
    )
    return catalog


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    **kwargs: Any,
) -> ToolResult:
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return dispatch_module.execute_tool(
        tool_name,
        arguments,
        state,
        PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
        **kwargs,
    )


def test_execute_tool_rejects_extra_top_level_arguments_before_handler() -> None:
    result = execute_tool(
        "get_pipeline_state",
        {"component": "source", "leaked_value": "sk-test-secret"},
        _empty_state(),
        _catalog(),
        tool_arguments_hash="0" * 64,
    )

    assert result.success is False
    assert "Invalid arguments for tool 'get_pipeline_state'" in result.data["error"]
    assert "unsupported" in result.data["error"]
    assert "sk-test-secret" not in result.data["error"]


def test_execute_tool_rejects_wrong_argument_types_before_handler() -> None:
    result = execute_tool(
        "get_pipeline_state",
        {"component": {"secret": "sk-test-secret"}},
        _empty_state(),
        _catalog(),
        tool_arguments_hash="0" * 64,
    )

    assert result.success is False
    assert "Invalid arguments for tool 'get_pipeline_state'" in result.data["error"]
    assert "type" in result.data["error"]
    assert "sk-test-secret" not in result.data["error"]


def test_source_path_arguments_require_data_dir_for_s2_confinement() -> None:
    result = execute_tool(
        "set_source",
        {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": "outside.csv", "schema": {"mode": "observed"}},
            "on_validation_failure": "error",
        },
        _empty_state(),
        _catalog(),
        tool_arguments_hash="0" * 64,
    )

    assert result.success is False
    assert "Path violation (S2)" in result.data["error"]
    assert "data_dir" in result.data["error"]


def test_secret_tool_missing_context_fails_before_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(
        arguments: dict[str, Any],
        state: CompositionState,
        context: ToolContext,
    ) -> ToolResult:
        del arguments, state, context
        raise AssertionError("secret handler should not run without secret context")

    monkeypatch.setattr(dispatch_module, "_SECRET_DISCOVERY_TOOLS", {"list_secret_refs": _explode})

    result = execute_tool("list_secret_refs", {}, _empty_state(), _catalog())

    assert result.success is False
    assert "Secret tools require secret service context" in result.data["error"]
