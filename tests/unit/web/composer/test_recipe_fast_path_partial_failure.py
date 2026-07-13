"""Recipe fast-path partial-failure handling (audit doctrine).

``_try_apply_freeform_recipe_intent`` performs two server-side tool
executions (``create_blob`` then ``apply_pipeline_recipe``) before the
normal LLM path is consulted. Under the audit doctrine every side effect
is either audited or undone — silence is the defect:

- (a) If ``create_blob`` succeeds but ``apply_pipeline_recipe`` fails, the
  created blob must not silently persist as an orphan: the service
  best-effort deletes it and emits a structured warning log + telemetry
  counter (explicitly saying whether the deletion worked), then falls
  through to the normal LLM path.
- (b) If the synthetic audit envelope for a tool that DID run successfully
  cannot be canonicalized, the service must fail loudly
  (``AuditIntegrityError``) rather than silently under-record the action
  — mirroring the crash-on-anomaly contract documented on
  ``finish_success`` (composer/audit.py) for first-party payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import structlog
from sqlalchemy import Engine

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.recipe_intent_routing import (
    FreeformRecipeIntentMatch,
    InlineRecipeBlob,
)
from elspeth.web.composer.service import _RECIPE_FAST_PATH_ORPHAN_BLOB_COUNTER, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import ToolResult, execute_tool
from elspeth.web.config import WebSettings

_SERVICE_MODULE = "elspeth.web.composer.service"


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = [
        PluginSummary(name="passthrough", description="Passthrough", plugin_type="transform", config_fields=[]),
        PluginSummary(name="truncate", description="Truncate", plugin_type="transform", config_fields=[]),
    ]
    catalog.list_sinks.return_value = [
        PluginSummary(name="json", description="JSON", plugin_type="sink", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"type": "object", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def _make_settings() -> WebSettings:
    return WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_advisor_max_calls_per_compose=4,
        composer_advisor_timeout_seconds=60.0,
        shareable_link_signing_key=b"\x00" * 32,
    )


def _make_service() -> ComposerServiceImpl:
    return ComposerServiceImpl.for_trained_operator(
        catalog=_mock_catalog(),
        settings=_make_settings(),
        session_engine=MagicMock(spec=Engine),
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


def _match(*, content: str = "a,b\n1,2", slots: Mapping[str, object] | None = None) -> FreeformRecipeIntentMatch:
    return FreeformRecipeIntentMatch(
        recipe_name="fork-coalesce-truncate-jsonl",
        inline_blob=InlineRecipeBlob(
            filename="inline-fork-coalesce.csv",
            mime_type="text/csv",
            content=content,
        ),
        slots=dict(slots) if slots is not None else {"output_path": "out.jsonl"},
    )


def _tool_result(*, success: bool, data: Any = None, version: int = 2) -> MagicMock:
    result = MagicMock(spec=ToolResult)
    result.success = success
    result.data = data
    result.updated_state = MagicMock(spec=CompositionState)
    result.updated_state.version = version
    result.to_dict.return_value = {"success": success}
    return result


async def _run_fast_path(
    service: ComposerServiceImpl,
    *,
    match: FreeformRecipeIntentMatch,
    execute_tool_mock: MagicMock,
    recorder: BufferingRecorder | None = None,
) -> tuple[Any, BufferingRecorder]:
    """Drive ``_try_apply_freeform_recipe_intent`` and return ``(result, recorder)``.

    ``recorder`` defaults to a fresh :class:`BufferingRecorder` when the
    caller does not need to inspect it across a fall-through (``None``
    result); tests asserting on the orphan-and-cleanup audit trail (G4) pass
    their own and read it back afterwards.
    """
    recorder = recorder if recorder is not None else BufferingRecorder()
    with (
        patch(f"{_SERVICE_MODULE}.match_freeform_recipe_intent", return_value=match),
        patch(f"{_SERVICE_MODULE}.execute_tool", execute_tool_mock),
    ):
        plugin_snapshot, policy_catalog = service._plugin_policy_context("u1")
        result = await service._try_apply_freeform_recipe_intent(
            message="build the fork-coalesce pipeline",
            state=_empty_state(),
            session_id=str(uuid4()),
            user_id="u1",
            progress=None,
            user_message_id=None,
            recorder=recorder,
            plugin_snapshot=plugin_snapshot,
            policy_catalog=policy_catalog,
        )
    return result, recorder


def _calls_by_tool(execute_tool_mock: MagicMock, tool_name: str) -> list[Any]:
    return [c for c in execute_tool_mock.call_args_list if c.args[0] == tool_name]


@pytest.mark.asyncio
async def test_apply_failure_deletes_orphan_blob_and_falls_through() -> None:
    """(a) Blob created + recipe apply failed => blob deleted, warn+count, None."""
    blob_id = str(uuid4())

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        if tool_name == "create_blob":
            return _tool_result(success=True, data={"blob_id": blob_id})
        if tool_name == "apply_pipeline_recipe":
            return _tool_result(success=False)
        if tool_name == "delete_blob":
            return _tool_result(success=True, data={"blob_id": blob_id, "deleted": True})
        raise AssertionError(f"unexpected tool {tool_name}")

    execute_tool_mock = MagicMock(spec=execute_tool, side_effect=_execute)
    service = _make_service()
    counter = MagicMock(spec=_RECIPE_FAST_PATH_ORPHAN_BLOB_COUNTER)
    slog_mock = MagicMock(spec=structlog.stdlib.BoundLogger)
    with (
        patch(f"{_SERVICE_MODULE}._RECIPE_FAST_PATH_ORPHAN_BLOB_COUNTER", counter),
        patch(f"{_SERVICE_MODULE}.slog", slog_mock),
    ):
        result, recorder = await _run_fast_path(service, match=_match(), execute_tool_mock=execute_tool_mock)

    # Falls through to the normal LLM path with clean state.
    assert result is None
    # The orphan blob was deleted (best effort) with the created blob_id.
    delete_calls = _calls_by_tool(execute_tool_mock, "delete_blob")
    assert len(delete_calls) == 1
    assert delete_calls[0].args[1] == {"blob_id": blob_id}
    # Structured warning log names the blob and says the deletion worked.
    assert slog_mock.warning.called
    warn_kwargs = slog_mock.warning.call_args.kwargs
    assert warn_kwargs["blob_id"] == blob_id
    assert warn_kwargs["blob_deleted"] is True
    # Telemetry counter incremented.
    assert counter.add.called
    # G4: the orphan create_blob + cleanup delete_blob pair lands in the
    # composer-audit invocation trail (not just slog/counter).
    invocations = recorder.invocations
    assert [inv.tool_name for inv in invocations] == ["create_blob", "delete_blob"]
    assert all(inv.status is ComposerToolStatus.SUCCESS for inv in invocations)


@pytest.mark.asyncio
async def test_apply_failure_with_delete_failure_is_logged_explicitly() -> None:
    """(a) When the cleanup deletion itself fails, the log/metric must say so."""
    blob_id = str(uuid4())

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        if tool_name == "create_blob":
            return _tool_result(success=True, data={"blob_id": blob_id})
        if tool_name == "apply_pipeline_recipe":
            return _tool_result(success=False)
        if tool_name == "delete_blob":
            raise RuntimeError("db unavailable")
        raise AssertionError(f"unexpected tool {tool_name}")

    execute_tool_mock = MagicMock(spec=execute_tool, side_effect=_execute)
    service = _make_service()
    counter = MagicMock(spec=_RECIPE_FAST_PATH_ORPHAN_BLOB_COUNTER)
    slog_mock = MagicMock(spec=structlog.stdlib.BoundLogger)
    with (
        patch(f"{_SERVICE_MODULE}._RECIPE_FAST_PATH_ORPHAN_BLOB_COUNTER", counter),
        patch(f"{_SERVICE_MODULE}.slog", slog_mock),
    ):
        result, recorder = await _run_fast_path(service, match=_match(), execute_tool_mock=execute_tool_mock)

    # The cleanup failure must not mask the fall-through.
    assert result is None
    assert slog_mock.warning.called
    warn_kwargs = slog_mock.warning.call_args.kwargs
    assert warn_kwargs["blob_id"] == blob_id
    assert warn_kwargs["blob_deleted"] is False
    assert warn_kwargs["delete_error"] == "RuntimeError"
    assert counter.add.called
    # G4: create_blob's own audit row still lands, and the failed cleanup
    # attempt is recorded as PLUGIN_CRASH rather than silently dropped.
    invocations = recorder.invocations
    assert [inv.tool_name for inv in invocations] == ["create_blob", "delete_blob"]
    assert invocations[0].status is ComposerToolStatus.SUCCESS
    assert invocations[1].status is ComposerToolStatus.PLUGIN_CRASH
    assert invocations[1].error_class == "RuntimeError"


@pytest.mark.asyncio
async def test_unrecordable_successful_apply_raises_audit_integrity_error() -> None:
    """(b) Canonicalization failure while recording a SUCCESSFUL application
    fails loudly (AuditIntegrityError) instead of silently under-recording,
    after best-effort cleanup of the fast-path blob."""
    blob_id = str(uuid4())

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        if tool_name == "create_blob":
            return _tool_result(success=True, data={"blob_id": blob_id})
        if tool_name == "apply_pipeline_recipe":
            return _tool_result(success=True, version=3)
        if tool_name == "delete_blob":
            return _tool_result(success=True, data={"blob_id": blob_id, "deleted": True})
        raise AssertionError(f"unexpected tool {tool_name}")

    execute_tool_mock = MagicMock(spec=execute_tool, side_effect=_execute)
    service = _make_service()
    # NaN in the recipe slots makes the synthetic apply_pipeline_recipe audit
    # envelope non-canonicalizable while the create_blob envelope stays fine.
    match = _match(slots={"output_path": "out.jsonl", "max_chars": float("nan")})
    slog_mock = MagicMock(spec=structlog.stdlib.BoundLogger)
    with patch(f"{_SERVICE_MODULE}.slog", slog_mock), pytest.raises(AuditIntegrityError):
        await _run_fast_path(service, match=match, execute_tool_mock=execute_tool_mock)

    # Best-effort cleanup ran before the loud failure.
    assert len(_calls_by_tool(execute_tool_mock, "delete_blob")) == 1


@pytest.mark.asyncio
async def test_unrecordable_create_blob_raises_audit_integrity_error() -> None:
    """(b) The same loud contract holds for the create_blob envelope."""
    blob_id = str(uuid4())

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        if tool_name == "create_blob":
            return _tool_result(success=True, data={"blob_id": blob_id})
        if tool_name == "delete_blob":
            return _tool_result(success=True, data={"blob_id": blob_id, "deleted": True})
        raise AssertionError(f"unexpected tool {tool_name}")

    execute_tool_mock = MagicMock(spec=execute_tool, side_effect=_execute)
    service = _make_service()
    # A non-canonicalizable inline blob payload (NaN) poisons the synthetic
    # create_blob audit envelope; the blob "creation" itself is mocked.
    match = FreeformRecipeIntentMatch(
        recipe_name="fork-coalesce-truncate-jsonl",
        inline_blob=InlineRecipeBlob(
            filename="inline-fork-coalesce.csv",
            mime_type="text/csv",
            content=float("nan"),  # type: ignore[arg-type]
        ),
        slots={"output_path": "out.jsonl"},
    )
    slog_mock = MagicMock(spec=structlog.stdlib.BoundLogger)
    with patch(f"{_SERVICE_MODULE}.slog", slog_mock), pytest.raises(AuditIntegrityError):
        await _run_fast_path(service, match=match, execute_tool_mock=execute_tool_mock)

    # apply_pipeline_recipe never ran (the envelope failed first), and the
    # orphaned blob was best-effort cleaned up.
    assert _calls_by_tool(execute_tool_mock, "apply_pipeline_recipe") == []
    assert len(_calls_by_tool(execute_tool_mock, "delete_blob")) == 1


@pytest.mark.asyncio
async def test_full_success_records_both_synthetic_invocations() -> None:
    """Happy path unchanged: both synthetic audit rows recorded, no cleanup."""
    blob_id = str(uuid4())

    def _execute(tool_name: str, *args: Any, **kwargs: Any) -> Any:
        if tool_name == "create_blob":
            return _tool_result(success=True, data={"blob_id": blob_id})
        if tool_name == "apply_pipeline_recipe":
            return _tool_result(success=True, version=3)
        raise AssertionError(f"unexpected tool {tool_name}")

    execute_tool_mock = MagicMock(spec=execute_tool, side_effect=_execute)
    service = _make_service()
    result, recorder = await _run_fast_path(service, match=_match(), execute_tool_mock=execute_tool_mock)

    assert result is not None
    assert len(result.tool_invocations) == 2
    assert result.tool_invocations == recorder.invocations
    assert _calls_by_tool(execute_tool_mock, "delete_blob") == []
