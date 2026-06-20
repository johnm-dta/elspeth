"""Integration coverage for the ``post_call_hints`` envelope.

Done-bar item 3 for composer-jit-hints Phase 1 (see
``.claude/plans/composer-llm-drifting-hollerith.md``): the
``post_call_hints`` envelope is present on ``set_source`` /
``upsert_node`` / ``patch_*_options`` responses for the 6 plugins that
override ``get_post_call_hints``, and absent otherwise.

The unit-level test (``tests/unit/web/composer/test_post_call_hints.py``)
drives ``_attach_post_call_hints`` with a MagicMock catalog — it
exercises the wiring but not the catalog→plugin dispatch. This
integration test drives the same wiring through the *real* catalog
(``CatalogServiceImpl`` with the production ``PluginManager``) against
the real plugin classes, then asserts the envelope on the serialised
response. It pins the boundary an LLM caller actually sees over MCP.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
    ValidationSummary,
)
from elspeth.web.composer.tools import ToolResult, _attach_post_call_hints

# (plugin_type, plugin_name, tool_name, triggering_config_snapshot, expected_substring)
# Each tuple is a *concrete* config that the plugin's get_post_call_hints
# override is documented to react to. The expected_substring is asserted
# against the serialised hint so a regression in the override body fails
# the test rather than silently dropping the hint.
HINTED_PLUGIN_CASES: tuple[tuple[str, str, str, Mapping[str, object], str], ...] = (
    (
        "source",
        "csv",
        "set_source",
        {"schema": {"mode": "fixed"}},
        "schema.mode: 'fixed'",
    ),
    (
        "source",
        "json",
        "set_source",
        {"schema": {"mode": "fixed"}},
        "schema.mode: 'fixed'",
    ),
    (
        # The subjective-term prompt hint was moved out of get_post_call_hints
        # into the interpretation-review flow (commit 33ae4f52b). The live LLM
        # post-call hint now flags manually-declared token-usage/model-ID fields
        # that the engine appends automatically.
        "transform",
        "llm",
        "upsert_node",
        {"response_field": "rating", "output_schema": {"fields": ["rating_usage:int"]}},
        "appended automatically",
    ),
    (
        "transform",
        "web_scrape",
        "upsert_node",
        {"format": "text", "text_separator": " "},
        "text_separator",
    ),
    (
        "sink",
        "json",
        "patch_output_options",
        {"format": "json"},
        "array mode",
    ),
    (
        "sink",
        "database",
        "patch_output_options",
        # The database sink's only write-behaviour knob is if_exists (append|replace);
        # the former write_mode/upsert surface was hint-rot and was removed.
        {"if_exists": "replace"},
        "recreates the target table",
    ),
)


@pytest.fixture(scope="module")
def catalog() -> CatalogServiceImpl:
    return CatalogServiceImpl(get_shared_plugin_manager())


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _success_result() -> ToolResult:
    return ToolResult(
        success=True,
        updated_state=_empty_state(),
        validation=ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=()),
        affected_nodes=("anchor",),
    )


@pytest.mark.parametrize(
    ("plugin_type", "plugin_name", "tool_name", "config_snapshot", "expected_substring"),
    HINTED_PLUGIN_CASES,
)
def test_post_call_hints_envelope_populated_for_hinted_plugins(
    catalog: CatalogServiceImpl,
    plugin_type: str,
    plugin_name: str,
    tool_name: str,
    config_snapshot: Mapping[str, object],
    expected_substring: str,
) -> None:
    """For each of the 6 priority plugins, a successful mutation attaches a non-empty post_call_hints envelope."""
    result = _attach_post_call_hints(
        _success_result(),
        catalog,
        plugin_type=plugin_type,  # type: ignore[arg-type]
        tool_name=tool_name,
        plugin_name=plugin_name,
        config_snapshot=config_snapshot,
    )
    assert result.post_call_hints, (
        f"{plugin_type}/{plugin_name} produced an empty post_call_hints envelope for "
        f"tool {tool_name!r} with triggering config {dict(config_snapshot)!r}. "
        "The plugin's get_post_call_hints override is not reachable from the real "
        "catalog dispatch."
    )
    payload: dict[str, Any] = result.to_dict()
    assert "post_call_hints" in payload, (
        f"to_dict() omitted post_call_hints despite a populated tuple for {plugin_type}/{plugin_name} — emission rule broken."
    )
    serialised: list[str] = payload["post_call_hints"]
    assert any(expected_substring in hint for hint in serialised), (
        f"{plugin_type}/{plugin_name} hint set {serialised!r} does not contain "
        f"the expected substring {expected_substring!r}; the override's hint text "
        "may have drifted from the integration contract."
    )


def test_post_call_hints_envelope_absent_when_plugin_has_no_override(
    catalog: CatalogServiceImpl,
) -> None:
    """A plugin without get_post_call_hints override leaves the envelope unset (no field on the wire)."""
    result = _attach_post_call_hints(
        _success_result(),
        catalog,
        plugin_type="source",
        tool_name="set_source",
        plugin_name="azure_blob",
        config_snapshot={},
    )
    assert result.post_call_hints == ()
    payload: dict[str, Any] = result.to_dict()
    assert "post_call_hints" not in payload, (
        "Catalog returned hints for a plugin without override — check that the "
        "BaseSource default of get_post_call_hints is returning () and the "
        "emission rule in ToolResult.to_dict is omitting empty tuples."
    )


def test_post_call_hints_envelope_absent_when_config_does_not_trigger(
    catalog: CatalogServiceImpl,
) -> None:
    """A hinted plugin returns no hint for a config snapshot that doesn't trigger its branches."""
    # csv source overrides get_post_call_hints, but only emits a hint
    # when schema.mode == 'fixed'. A bare config carries no trigger.
    result = _attach_post_call_hints(
        _success_result(),
        catalog,
        plugin_type="source",
        tool_name="set_source",
        plugin_name="csv",
        config_snapshot={"path": "/tmp/sample.csv"},
    )
    assert result.post_call_hints == ()
    payload: dict[str, Any] = result.to_dict()
    assert "post_call_hints" not in payload
