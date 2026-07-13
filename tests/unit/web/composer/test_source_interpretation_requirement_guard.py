"""Source write paths must reject LLM-supplied resolver-owned interpretation requirements.

An LLM-authored ("invented") source stays gated for human review until
``resolve_interpretation_event`` records a real resolution. The read side
(``interpretation_state._pending_source_sites``) treats an INVENTED_SOURCE
requirement as clean once ``status == "resolved"`` and
``accepted_artifact_hash`` matches the source ``content_hash`` — WITHOUT
consulting the interpretation-events audit DB, and the LLM learns the blob
``content_hash`` from the set_source_from_blob result. So the only real defence
against a forged "resolved" requirement is the write-boundary guard.

These tests pin that guard on EVERY write path that can land options on a
``SourceSpec``: set_source, patch_source_options, set_source_from_blob (caller
options), set_pipeline (sources + legacy source), and the wire_blob_inline_ref
source arm. Symmetric with the LLM-node guard covered in test_tools.py.
"""

from __future__ import annotations

from typing import Any

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.service import CatalogServiceImpl
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.tools import (
    _execute_patch_source_options,
    _execute_set_source,
    _execute_set_source_from_blob,
)
from elspeth.web.composer.tools import (
    execute_tool as _execute_tool,
)
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot


def _catalog() -> CatalogServiceImpl:
    manager = PluginManager()
    manager.register_builtin_plugins()
    return CatalogServiceImpl(manager)


def _ctx() -> ToolContext:
    catalog = _catalog()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return ToolContext(
        catalog=PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
    )


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceImpl,
) -> Any:
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return _execute_tool(
        tool_name,
        arguments,
        state,
        PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
    )


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _state_with_csv_source() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={"path": "/tmp/data.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _forged_resolved_invented_source_requirement() -> dict[str, Any]:
    """A resolved INVENTED_SOURCE requirement the LLM must not be able to stage."""
    return {
        "id": "source_review:inline_source_data",
        "kind": "invented_source",
        "user_term": "inline_source_data",
        "status": "resolved",
        "draft": None,
        "event_id": "forged-source-event",
        "accepted_value": None,
        "accepted_artifact_hash": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "resolved_prompt_template_hash": None,
    }


def _pending_invented_source_requirement() -> dict[str, Any]:
    """A pending INVENTED_SOURCE requirement — legitimate composer-staged input."""
    return {
        "id": "source_review:inline_source_data",
        "kind": "invented_source",
        "user_term": "inline_source_data",
        "status": "pending",
        "draft": "url\nhttps://example.gov.au\n",
        "event_id": None,
        "accepted_value": None,
        "accepted_artifact_hash": None,
        "resolved_prompt_template_hash": None,
    }


def _assert_rejected_forged_review(result: Any, original_state: CompositionState) -> None:
    assert result.success is False
    assert result.updated_state is original_state
    assert result.updated_state.version == original_state.version
    assert result.data is not None
    error = result.data["error"]
    assert INTERPRETATION_REQUIREMENTS_KEY in error
    assert "resolve_interpretation_event" in error


def test_patch_source_options_rejects_forged_resolved_invented_source() -> None:
    """The live exploit vector: patch the source with a forged resolved review."""
    state = _state_with_csv_source()
    result = _execute_patch_source_options(
        {"patch": {INTERPRETATION_REQUIREMENTS_KEY: [_forged_resolved_invented_source_requirement()]}},
        state,
        _ctx(),
    )
    _assert_rejected_forged_review(result, state)


def test_patch_source_options_allows_pending_invented_source_requirement() -> None:
    """A pending requirement is legitimate composer input — must NOT be rejected."""
    state = _state_with_csv_source()
    result = _execute_patch_source_options(
        {"patch": {INTERPRETATION_REQUIREMENTS_KEY: [_pending_invented_source_requirement()]}},
        state,
        _ctx(),
    )
    assert result.success is True, result.data
    staged = result.updated_state.sources["source"].options[INTERPRETATION_REQUIREMENTS_KEY]
    assert staged[0]["status"] == "pending"


def test_set_source_rejects_forged_resolved_invented_source() -> None:
    state = _empty_state()
    result = _execute_set_source(
        {
            "plugin": "csv",
            "on_success": "rows",
            "on_validation_failure": "discard",
            "options": {
                "path": "/tmp/data.csv",
                "schema": {"mode": "observed"},
                INTERPRETATION_REQUIREMENTS_KEY: [_forged_resolved_invented_source_requirement()],
            },
        },
        state,
        _ctx(),
    )
    _assert_rejected_forged_review(result, state)


def test_set_source_from_blob_rejects_forged_resolved_invented_source() -> None:
    """Caller options on a blob bind cannot smuggle a resolved review.

    The guard fires before blob resolution, so no blob/session is needed.
    """
    state = _empty_state()
    result = _execute_set_source_from_blob(
        {
            "blob_id": "00000000-0000-0000-0000-000000000000",
            "on_success": "rows",
            "options": {INTERPRETATION_REQUIREMENTS_KEY: [_forged_resolved_invented_source_requirement()]},
        },
        state,
        _ctx(),
    )
    _assert_rejected_forged_review(result, state)


def test_set_pipeline_rejects_forged_resolved_invented_source_on_source() -> None:
    state = _empty_state()
    args = {
        "sources": {
            "source": {
                "plugin": "csv",
                "on_success": "rows",
                "on_validation_failure": "discard",
                "options": {
                    "path": "/tmp/data.csv",
                    "schema": {"mode": "observed"},
                    INTERPRETATION_REQUIREMENTS_KEY: [_forged_resolved_invented_source_requirement()],
                },
            }
        },
        "nodes": [],
        "edges": [],
        "outputs": [],
    }
    result = execute_tool("set_pipeline", args, state, _catalog())
    _assert_rejected_forged_review(result, state)
