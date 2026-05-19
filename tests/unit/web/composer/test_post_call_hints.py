"""Tests for the postscript-hint envelope on ToolResult.

Covers ``_attach_post_call_hints`` (the resolver helper used by the
five mutation handle wrappers) and ``ToolResult.to_dict``'s
emit-only-when-non-empty rule.

Phase 1 / Step 3 of composer-jit-hints — see
``contracts/plugin_assistance.py`` for the discipline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
    ValidationSummary,
)
from elspeth.web.composer.tools import ToolResult, _attach_post_call_hints


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _success_result(state: CompositionState | None = None) -> ToolResult:
    s = state if state is not None else _empty_state()
    return ToolResult(
        success=True,
        updated_state=s,
        validation=ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=()),
        affected_nodes=("source",),
    )


def _failed_result(state: CompositionState | None = None) -> ToolResult:
    s = state if state is not None else _empty_state()
    return ToolResult(
        success=False,
        updated_state=s,
        validation=ValidationSummary(
            is_valid=False,
            errors=(),
            warnings=(),
            suggestions=(),
        ),
        affected_nodes=(),
    )


def _catalog_returning(hints: tuple[str, ...]) -> MagicMock:
    catalog = MagicMock()
    catalog.post_call_hints.return_value = hints
    return catalog


def test_attach_post_call_hints_populates_field_when_plugin_returns_hints() -> None:
    catalog = _catalog_returning(("first hint", "second hint"))
    result = _attach_post_call_hints(
        _success_result(),
        catalog,
        plugin_type="source",
        tool_name="set_source",
        plugin_name="csv",
        config_snapshot={"path": "/tmp/data.csv"},
    )
    assert result.post_call_hints == ("first hint", "second hint")
    catalog.post_call_hints.assert_called_once_with(
        plugin_type="source",
        plugin_name="csv",
        tool_name="set_source",
        config_snapshot={"path": "/tmp/data.csv"},
    )


def test_attach_post_call_hints_no_change_when_hints_empty() -> None:
    """Empty hints from the catalog leave the result unchanged (no field added)."""
    catalog = _catalog_returning(())
    original = _success_result()
    result = _attach_post_call_hints(
        original,
        catalog,
        plugin_type="source",
        tool_name="set_source",
        plugin_name="csv",
        config_snapshot={},
    )
    assert result is original
    assert result.post_call_hints == ()


def test_attach_post_call_hints_skipped_on_failure() -> None:
    """Failed mutations skip the catalog call entirely (no coaching on errors)."""
    catalog = _catalog_returning(("hint",))
    result = _attach_post_call_hints(
        _failed_result(),
        catalog,
        plugin_type="source",
        tool_name="set_source",
        plugin_name="csv",
        config_snapshot={},
    )
    assert result.post_call_hints == ()
    catalog.post_call_hints.assert_not_called()


def test_attach_post_call_hints_skipped_when_plugin_name_none() -> None:
    """Gate/coalesce nodes (plugin=None) skip the catalog call."""
    catalog = _catalog_returning(("hint",))
    result = _attach_post_call_hints(
        _success_result(),
        catalog,
        plugin_type="transform",
        tool_name="upsert_node",
        plugin_name=None,
        config_snapshot={"condition": "row['x'] > 0"},
    )
    assert result.post_call_hints == ()
    catalog.post_call_hints.assert_not_called()


def test_tool_result_to_dict_omits_post_call_hints_when_empty() -> None:
    """The envelope MUST NOT include the field when empty — preserves the existing wire format."""
    result = _success_result()
    payload: dict[str, Any] = result.to_dict()
    assert "post_call_hints" not in payload


def test_tool_result_to_dict_emits_post_call_hints_when_populated() -> None:
    """A populated tuple appears as a JSON list under the post_call_hints key."""
    result = ToolResult(
        success=True,
        updated_state=_empty_state(),
        validation=ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=()),
        affected_nodes=("source",),
        post_call_hints=("did you call inspect_source?",),
    )
    payload: dict[str, Any] = result.to_dict()
    assert payload["post_call_hints"] == ["did you call inspect_source?"]
