"""Real-loop coverage for explicit-approval ``set_pipeline`` prevalidation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, func, insert, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.blobs.protocol import BlobPendingProposalError
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.anti_anchor import AntiAnchorTracker
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.protocol import ComposerPluginCrashError
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata, SourceSpec, ValidationEntry, ValidationSummary
from elspeth.web.composer.tools import ToolResult
from elspeth.web.composer.tools.sessions import build_set_pipeline_candidate as real_build_set_pipeline_candidate
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.validation import ProfileAwareValidationResult
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    blobs_table,
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from tests.unit.web.composer.conftest import (
    _fake_llm_response,
    _make_settings,
    build_test_sessions_service,
)


@dataclass(frozen=True, slots=True)
class _Harness:
    engine: Engine
    sessions: SessionServiceImpl
    service: ComposerServiceImpl
    session_id: str
    user_message_id: str


class _ScriptedLLM:
    def __init__(self, *responses: Any) -> None:
        self._responses = list(responses)
        self.message_snapshots: list[list[dict[str, Any]]] = []

    async def __call__(self, messages: list[dict[str, Any]], _tools: Any) -> Any:
        self.message_snapshots.append(deepcopy(messages))
        if not self._responses:
            return _fake_llm_response(content="Done.")
        return self._responses.pop(0)


class _FinalRejectingCatalog(PolicyCatalogView):
    """Request catalog that rejects one otherwise-valid complete candidate."""

    def validate_composition_state(self, state: CompositionState) -> ProfileAwareValidationResult:
        raw = state.validate()
        if raw.is_valid and state.metadata.name == "final-profile-reject":
            raw = ValidationSummary(
                is_valid=False,
                errors=(
                    ValidationEntry(
                        component="policy:final",
                        message="The request-scoped profile rejects this complete composition.",
                        severity="high",
                        error_code="profile_complete_state_rejected",
                    ),
                ),
                warnings=raw.warnings,
                suggestions=raw.suggestions,
                edge_contracts=raw.edge_contracts,
                semantic_contracts=raw.semantic_contracts,
            )
        return ProfileAwareValidationResult(
            authored_state=state,
            executable_state=state,
            policy_findings=(),
            validation=raw,
        )


class _PreproposalBaseSignal(BaseException):
    """Test-only shutdown-style signal outside the ``Exception`` hierarchy."""


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _harness(tmp_path: Path) -> _Harness:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    sessions = build_test_sessions_service(engine=engine, data_dir=tmp_path)
    session_id = str(uuid4())
    user_message_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="proposal-prevalidation-user",
                auth_provider_type="local",
                title="Proposal prevalidation",
                trust_mode="explicit_approve",
                density_default="high",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content="Build a reviewed pipeline.",
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    with patch.object(
        ComposerServiceImpl,
        "_compute_availability",
        return_value=ComposerAvailability(available=True, model="test-model", provider="test"),
    ):
        service = ComposerServiceImpl.for_trained_operator(
            catalog=create_catalog_service(),
            settings=_make_settings(tmp_path),
            sessions_service=sessions,
            session_engine=engine,
        )
    return _Harness(
        engine=engine,
        sessions=sessions,
        service=service,
        session_id=session_id,
        user_message_id=user_message_id,
    )


def _valid_pipeline_args(tmp_path: Path, *, metadata_name: str = "proposal-valid") -> dict[str, Any]:
    source_path = tmp_path / "blobs" / "input.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("value\n1\n", encoding="utf-8")
    return {
        "source": {
            "plugin": "csv",
            "on_success": "main",
            "options": {"path": str(source_path), "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "sink_name": "main",
                "plugin": "json",
                "options": {
                    "path": str(tmp_path / "outputs" / "result.jsonl"),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": {"name": metadata_name},
    }


def _tool_turn(call_id: str, tool_name: str, arguments: dict[str, Any]) -> Any:
    return _fake_llm_response(
        tool_calls=(
            {
                "id": call_id,
                "name": tool_name,
                "arguments": arguments,
            },
        )
    )


def _inline_pipeline_args(tmp_path: Path) -> dict[str, Any]:
    args = _valid_pipeline_args(tmp_path, metadata_name="inline-proposal")
    args["source"] = {
        "plugin": "csv",
        "on_success": "main",
        "options": {"schema": {"mode": "observed"}},
        "on_validation_failure": "discard",
        "inline_blob": {
            "filename": "candidate.csv",
            "mime_type": "text/csv",
            "content": "value\n42\n",
            "description": "Prepared only for a pending proposal",
        },
    }
    return args


def _count_rows(engine: Engine, table: Any) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def _persisted_tool_content(harness: _Harness, tool_call_id: str) -> str:
    with harness.engine.connect() as conn:
        return str(
            conn.execute(
                select(chat_messages_table.c.content)
                .where(chat_messages_table.c.session_id == harness.session_id)
                .where(chat_messages_table.c.role == "tool")
                .where(chat_messages_table.c.tool_call_id == tool_call_id)
            ).scalar_one()
        )


@pytest.mark.asyncio
async def test_semantic_rejection_reaches_next_model_turn_then_repair_creates_one_proposal(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    invalid = _valid_pipeline_args(tmp_path, metadata_name="invalid-plugin")
    invalid["source"]["plugin"] = "not_installed"
    repaired = _valid_pipeline_args(tmp_path, metadata_name="repaired")
    llm = _ScriptedLLM(
        _tool_turn("call_invalid", "set_pipeline", invalid),
        _tool_turn("call_repaired", "set_pipeline", repaired),
        _fake_llm_response(content="The repaired proposal is pending approval."),
    )

    with patch.object(harness.service, "_call_llm", new=llm):
        result = await harness.service.compose(
            "Build a reviewed pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == 1
    assert proposals[0].tool_call_id == "call_repaired"
    assert result.state is state
    assert _count_rows(harness.engine, composition_proposals_table) == 1
    assert _count_rows(harness.engine, composition_states_table) == 0
    assert _count_rows(harness.engine, blobs_table) == 0

    proposal_outcome = harness.service._phase3_last_tool_outcomes[0]
    assert isinstance(proposal_outcome.response, ToolResult)
    assert proposal_outcome.post_version == state.version

    second_call_tools = [message for message in llm.message_snapshots[1] if message.get("role") == "tool"]
    assert second_call_tools
    invalid_feedback = next(message for message in second_call_tools if message["tool_call_id"] == "call_invalid")
    invalid_payload = json.loads(invalid_feedback["content"])
    assert invalid_payload["success"] is False
    assert invalid_payload["version"] == state.version
    assert invalid_payload["data"]["status"] == "PREVALIDATION_REJECTED"
    assert invalid_payload["data"]["applied"] is False
    assert invalid_payload["validation"]["errors"][0]["component"] == "rejected_mutation"
    assert "not_installed" in invalid_feedback["content"]
    with harness.engine.connect() as conn:
        persisted_tool_content = tuple(
            conn.execute(
                select(chat_messages_table.c.content)
                .where(chat_messages_table.c.session_id == harness.session_id)
                .where(chat_messages_table.c.role == "tool")
            ).scalars()
        )
    assert any("not_installed" in content for content in persisted_tool_content)

    assert len(result.tool_invocations) == 2
    assert result.tool_invocations[0].version_after == state.version
    assert result.tool_invocations[1].version_after == state.version


@pytest.mark.asyncio
async def test_final_profile_rejection_is_unapplied_audited_and_repairable(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    invalid = _valid_pipeline_args(tmp_path, metadata_name="final-profile-reject")
    repaired = _valid_pipeline_args(tmp_path, metadata_name="profile-repaired")
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(harness.service._catalog)
    catalog = _FinalRejectingCatalog.for_trained_operator(harness.service._catalog, snapshot)
    failures: list[tuple[str, str]] = []
    message_snapshots: list[list[dict[str, Any]]] = []
    responses = [
        _tool_turn("call_final_invalid", "set_pipeline", invalid),
        _tool_turn("call_final_repaired", "set_pipeline", repaired),
        _fake_llm_response(content="The repaired proposal is pending approval."),
    ]
    files_before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()))
    original_record_failure = AntiAnchorTracker.record_failure

    def _record_failure(tracker: AntiAnchorTracker, tool_name: str, arguments_hash: str) -> None:
        failures.append((tool_name, arguments_hash))
        original_record_failure(tracker, tool_name, arguments_hash)

    async def _llm(messages: list[dict[str, Any]], _tools: Any) -> Any:
        message_snapshots.append(deepcopy(messages))
        if len(message_snapshots) == 2:
            assert _count_rows(harness.engine, composition_proposals_table) == 0
            assert _count_rows(harness.engine, composition_states_table) == 0
            assert _count_rows(harness.engine, blobs_table) == 0
            assert builder.call_count == 1
        return responses.pop(0)

    with (
        patch.object(harness.service, "_plugin_policy_context", return_value=(snapshot, catalog)),
        patch.object(harness.service, "_call_llm", new=_llm),
        patch.object(AntiAnchorTracker, "record_failure", new=_record_failure),
        patch(
            "elspeth.web.composer.tool_batch.build_set_pipeline_candidate",
            wraps=real_build_set_pipeline_candidate,
        ) as builder,
    ):
        result = await harness.service.compose(
            "Build a reviewed pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == 1
    assert proposals[0].tool_call_id == "call_final_repaired"
    assert result.state is state
    assert builder.call_count == 2
    assert [tool_name for tool_name, _hash in failures] == ["set_pipeline"]

    invalid_feedback = next(message for message in message_snapshots[1] if message.get("tool_call_id") == "call_final_invalid")
    invalid_payload = json.loads(invalid_feedback["content"])
    assert invalid_payload["success"] is True
    assert invalid_payload["validation"]["is_valid"] is False
    assert invalid_payload["validation"]["errors"][0]["error_code"] == "profile_complete_state_rejected"
    assert invalid_payload["data"] == {
        "status": "PREVALIDATION_REJECTED",
        "applied": False,
        "applied_version": state.version,
        "candidate_version": state.version + 1,
        "message": (
            "The candidate pipeline failed prevalidation, was not applied, and was not submitted for approval. "
            "Repair the reported validation errors and retry."
        ),
    }
    assert invalid_payload["version"] == state.version + 1
    assert tuple(invocation.version_after for invocation in result.tool_invocations) == (state.version, state.version)
    files_after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()))
    assert files_after == files_before


@pytest.mark.asyncio
async def test_inline_candidate_materializes_one_custody_safe_proposal_without_raw_audit_surfaces(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _harness(tmp_path)
    existing_path = tmp_path / "existing.csv"
    existing_path.write_text("value\n1\n", encoding="utf-8")
    state = CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="main",
            options={"path": str(existing_path), "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(name="existing non-empty pipeline"),
        version=4,
    )
    args = _inline_pipeline_args(tmp_path)
    raw_content = "private-inline-value-7f45\n42\n"
    args["source"]["inline_blob"]["content"] = raw_content
    llm = _ScriptedLLM(
        _tool_turn("call_inline", "set_pipeline", args),
        _fake_llm_response(content="The inline pipeline proposal is pending approval."),
    )

    with (
        patch.object(harness.service, "_call_llm", new=llm),
        patch(
            "elspeth.web.composer.tool_batch.build_set_pipeline_candidate",
            wraps=real_build_set_pipeline_candidate,
        ) as builder,
    ):
        result = await harness.service.compose(
            "Build a reviewed pipeline with generated CSV content.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == 1
    assert proposals[0].tool_call_id == "call_inline"
    assert proposals[0].status == "pending"
    safe_arguments = deep_thaw(proposals[0].arguments_json)
    safe_redacted_arguments = deep_thaw(proposals[0].arguments_redacted_json)
    assert "inline_blob" not in safe_arguments["source"]
    assert "inline_blob" not in safe_redacted_arguments["source"]
    assert UUID(safe_arguments["source"]["blob_id"]).version == 5
    assert safe_redacted_arguments["source"]["blob_id"] == safe_arguments["source"]["blob_id"]
    assert proposals[0].tool_arguments_hash == stable_hash(safe_arguments)
    assert builder.call_count == 2
    assert result.state is state
    assert _count_rows(harness.engine, blobs_table) == 1
    assert _count_rows(harness.engine, composition_states_table) == 0
    with harness.engine.connect() as conn:
        quota_bytes = conn.execute(
            select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == harness.session_id)
        ).scalar_one()
        blob_row = conn.execute(select(blobs_table)).one()
        event_payloads = tuple(conn.execute(select(proposal_events_table.c.payload)).scalars())
        persisted_messages = tuple(
            conn.execute(
                select(chat_messages_table.c.content, chat_messages_table.c.raw_content, chat_messages_table.c.tool_calls).where(
                    chat_messages_table.c.session_id == harness.session_id
                )
            )
        )
    assert int(quota_bytes) == len(raw_content.encode("utf-8"))
    assert blob_row.id == safe_arguments["source"]["blob_id"]
    assert blob_row.creating_arguments_hash == stable_hash(safe_arguments)
    assert raw_content not in json.dumps(event_payloads)
    assert raw_content not in json.dumps(tuple(tuple(row) for row in persisted_messages))
    assert raw_content not in json.dumps(safe_arguments)
    assert len(result.tool_invocations) == 1
    assert result.tool_invocations[0].version_after == state.version
    assert result.tool_invocations[0].arguments_canonical == canonical_json(safe_arguments)
    assert result.tool_invocations[0].arguments_hash == stable_hash(safe_arguments)
    api_facing_result = {
        "message": result.message,
        "tool_invocations": [
            {
                "arguments_canonical": invocation.arguments_canonical,
                "result_canonical": invocation.result_canonical,
                "error_message": invocation.error_message,
            }
            for invocation in result.tool_invocations
        ],
    }
    assert raw_content not in json.dumps(api_facing_result)
    assert "[redacted inline content held for custody]" not in json.dumps(api_facing_result)
    assert raw_content not in caplog.text
    assert "[redacted inline content held for custody]" not in caplog.text
    assert len(llm.message_snapshots) == 2
    assert raw_content not in json.dumps(llm.message_snapshots[1])
    assert result.llm_calls[1].messages_hash == stable_hash(llm.message_snapshots[1])

    blob_service = BlobServiceImpl(harness.engine, tmp_path)
    blob_id = UUID(safe_arguments["source"]["blob_id"])
    with pytest.raises(BlobPendingProposalError):
        await blob_service.delete_blob(blob_id)
    rejected = await harness.sessions.reject_composition_proposal(
        session_id=UUID(harness.session_id),
        proposal_id=proposals[0].id,
        actor="composer-parity-test",
    )
    assert rejected.status == "rejected"
    assert Path(blob_row.storage_path).exists()
    await blob_service.delete_blob(blob_id)
    assert not Path(blob_row.storage_path).exists()


@pytest.mark.asyncio
async def test_inline_proposal_gap_retry_reuses_one_custody_blob_and_quota_charge(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    arguments = _inline_pipeline_args(tmp_path)
    raw_content = "proposal-gap-private-value-2f16\n42\n"
    arguments["source"]["inline_blob"]["content"] = raw_content
    captured_safe_arguments: list[dict[str, Any]] = []

    async def _interrupt_before_proposal(**kwargs: Any) -> Any:
        captured_safe_arguments.append(deepcopy(kwargs["arguments_json"]))
        raise RuntimeError("simulated interruption before proposal creation")

    first_llm = _ScriptedLLM(_tool_turn("call_gap", "set_pipeline", arguments))
    with (
        patch.object(harness.service, "_call_llm", new=first_llm),
        patch.object(harness.sessions, "create_composition_proposal", new=_interrupt_before_proposal),
        pytest.raises(RuntimeError, match="simulated interruption before proposal creation"),
    ):
        await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    assert len(captured_safe_arguments) == 1
    assert "inline_blob" not in captured_safe_arguments[0]["source"]
    assert raw_content not in json.dumps(captured_safe_arguments)
    assert _count_rows(harness.engine, blobs_table) == 1
    assert _count_rows(harness.engine, composition_proposals_table) == 0
    with harness.engine.connect() as conn:
        first_blob = conn.execute(select(blobs_table)).one()
        quota_after_interruption = conn.execute(
            select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == harness.session_id)
        ).scalar_one()
    assert int(quota_after_interruption) == len(raw_content.encode("utf-8"))
    assert Path(first_blob.storage_path).read_bytes() == raw_content.encode("utf-8")

    retry_llm = _ScriptedLLM(
        _tool_turn("call_gap_retry", "set_pipeline", arguments),
        _fake_llm_response(content="The retried inline proposal is pending approval."),
    )
    with patch.object(harness.service, "_call_llm", new=retry_llm):
        await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == 1
    retry_arguments = deep_thaw(proposals[0].arguments_json)
    assert retry_arguments["source"]["blob_id"] == first_blob.id
    assert "inline_blob" not in retry_arguments["source"]
    assert raw_content not in json.dumps(retry_arguments)
    assert _count_rows(harness.engine, blobs_table) == 1
    with harness.engine.connect() as conn:
        retried_blob = conn.execute(select(blobs_table)).one()
        quota_after_retry = conn.execute(
            select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == harness.session_id)
        ).scalar_one()
    assert retried_blob.id == first_blob.id
    assert retried_blob.creating_arguments_hash == stable_hash(retry_arguments)
    assert int(quota_after_retry) == len(raw_content.encode("utf-8"))
    assert Path(retried_blob.storage_path).read_bytes() == raw_content.encode("utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_argument", "rejected_fragment"),
    [
        ("disallowed_mime", "mime_type", "application/octet-stream"),
        ("unsanitizable_filename", "filename", ".."),
    ],
)
async def test_inline_candidate_argument_error_is_audited_once_and_repairable(
    tmp_path: Path,
    case: str,
    expected_argument: str,
    rejected_fragment: str,
) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    invalid = _inline_pipeline_args(tmp_path)
    inline_blob = invalid["source"]["inline_blob"]
    if case == "disallowed_mime":
        inline_blob["mime_type"] = rejected_fragment
    elif case == "unsanitizable_filename":
        inline_blob["filename"] = rejected_fragment
    else:  # pragma: no cover - parametrization is a closed local table
        raise AssertionError(f"unknown case: {case}")

    repaired = _inline_pipeline_args(tmp_path)
    responses = [
        _tool_turn(f"call_{case}", "set_pipeline", invalid),
        _tool_turn(f"call_{case}_repaired", "set_pipeline", repaired),
        _fake_llm_response(content="The repaired inline proposal is pending approval."),
    ]
    message_snapshots: list[list[dict[str, Any]]] = []
    invalid_turn_outcomes: list[Any] = []

    async def _llm(messages: list[dict[str, Any]], _tools: Any) -> Any:
        message_snapshots.append(deepcopy(messages))
        if len(message_snapshots) == 2:
            invalid_turn_outcomes.extend(harness.service._phase3_last_tool_outcomes)
            assert builder.call_count == 1
            assert _count_rows(harness.engine, composition_proposals_table) == 0
            assert _count_rows(harness.engine, blobs_table) == 0
            assert _count_rows(harness.engine, composition_states_table) == 0
        return responses.pop(0)

    with (
        patch.object(harness.service, "_call_llm", new=_llm),
        patch(
            "elspeth.web.composer.tool_batch.build_set_pipeline_candidate",
            wraps=real_build_set_pipeline_candidate,
        ) as builder,
    ):
        result = await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == 1
    assert proposals[0].tool_call_id == f"call_{case}_repaired"
    assert result.state is state
    assert builder.call_count == 3
    assert _count_rows(harness.engine, blobs_table) == 1
    assert _count_rows(harness.engine, composition_states_table) == 0

    arg_error_invocations = [invocation for invocation in result.tool_invocations if invocation.status is ComposerToolStatus.ARG_ERROR]
    assert len(arg_error_invocations) == 1
    arg_error_invocation = arg_error_invocations[0]
    assert arg_error_invocation.tool_call_id == f"call_{case}"
    assert arg_error_invocation.error_class == "ToolArgumentError"
    assert arg_error_invocation.version_before == state.version
    assert arg_error_invocation.version_after is None
    inline_content = invalid["source"]["inline_blob"]["content"]
    assert inline_content not in arg_error_invocation.arguments_canonical
    assert inline_content not in json.dumps(message_snapshots[1])

    arg_error_outcomes = [outcome for outcome in invalid_turn_outcomes if outcome.error_class == "ToolArgumentError"]
    assert len(arg_error_outcomes) == 1
    assert arg_error_outcomes[0].pre_version == state.version
    assert arg_error_outcomes[0].post_version == state.version
    assert arg_error_outcomes[0].response is None

    feedback = next(message for message in message_snapshots[1] if message.get("tool_call_id") == f"call_{case}")
    feedback_payload = json.loads(feedback["content"])
    assert f"'{expected_argument}' must be" in feedback_payload["error"]
    assert rejected_fragment not in feedback["content"]
    assert "caused by" not in feedback["content"]

    persisted_feedback = _persisted_tool_content(harness, f"call_{case}")
    assert json.loads(persisted_feedback)["error_class"] == "ToolArgumentError"
    assert rejected_fragment not in persisted_feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_shape", ["scalar", "list", "unknown", "named"])
async def test_malformed_inline_blob_never_survives_full_compose_surfaces(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    malformed_shape: str,
) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    invalid = _inline_pipeline_args(tmp_path)
    private_value = f"private malformed {malformed_shape} value 71b2"
    if malformed_shape == "scalar":
        invalid["source"]["inline_blob"] = private_value
    elif malformed_shape == "list":
        invalid["source"]["inline_blob"] = [private_value, {"nested": private_value}]
    elif malformed_shape == "unknown":
        invalid["source"]["inline_blob"]["unknown"] = {"payload": private_value}
    elif malformed_shape == "named":
        invalid["sources"] = {"named": {"inline_blob": {"unknown": private_value}}}
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(malformed_shape)

    repaired = _inline_pipeline_args(tmp_path)
    llm = _ScriptedLLM(
        _tool_turn(f"call_malformed_{malformed_shape}", "set_pipeline", invalid),
        _tool_turn(f"call_repaired_{malformed_shape}", "set_pipeline", repaired),
        _fake_llm_response(content="The repaired inline proposal is pending approval."),
    )
    with patch.object(harness.service, "_call_llm", new=llm):
        result = await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == 1
    assert private_value not in json.dumps(llm.message_snapshots)
    assert private_value not in json.dumps([deep_thaw(proposal.arguments_json) for proposal in proposals])
    assert private_value not in json.dumps([invocation.arguments_canonical for invocation in result.tool_invocations])
    assert private_value not in caplog.text
    with harness.engine.connect() as conn:
        persisted = tuple(
            conn.execute(
                select(chat_messages_table.c.content, chat_messages_table.c.raw_content, chat_messages_table.c.tool_calls).where(
                    chat_messages_table.c.session_id == harness.session_id
                )
            )
        )
    assert private_value not in json.dumps(tuple(tuple(row) for row in persisted))


@pytest.mark.asyncio
async def test_surrogate_inline_content_fails_closed_at_canonicalization_and_is_repairable(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    invalid = _inline_pipeline_args(tmp_path)
    rejected_content = "bad\udc80private"
    invalid["source"]["inline_blob"]["content"] = rejected_content
    repaired = _inline_pipeline_args(tmp_path)
    responses = [
        _tool_turn("call_surrogate", "set_pipeline", invalid),
        _tool_turn("call_surrogate_repaired", "set_pipeline", repaired),
        _fake_llm_response(content="The repaired inline proposal is pending approval."),
    ]
    message_snapshots: list[list[dict[str, Any]]] = []
    invalid_turn_outcomes: list[Any] = []

    async def _llm(messages: list[dict[str, Any]], _tools: Any) -> Any:
        message_snapshots.append(deepcopy(messages))
        if len(message_snapshots) == 2:
            invalid_turn_outcomes.extend(harness.service._phase3_last_tool_outcomes)
            assert builder.call_count == 0
            assert _count_rows(harness.engine, composition_proposals_table) == 0
            assert _count_rows(harness.engine, blobs_table) == 0
            assert _count_rows(harness.engine, composition_states_table) == 0
        return responses.pop(0)

    with (
        patch.object(harness.service, "_call_llm", new=_llm),
        patch(
            "elspeth.web.composer.tool_batch.build_set_pipeline_candidate",
            wraps=real_build_set_pipeline_candidate,
        ) as builder,
    ):
        result = await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == 1
    assert proposals[0].tool_call_id == "call_surrogate_repaired"
    assert result.state is state
    assert builder.call_count == 2
    assert _count_rows(harness.engine, blobs_table) == 1
    assert _count_rows(harness.engine, composition_states_table) == 0

    arg_error_invocations = [invocation for invocation in result.tool_invocations if invocation.status is ComposerToolStatus.ARG_ERROR]
    assert len(arg_error_invocations) == 1
    arg_error_invocation = arg_error_invocations[0]
    assert arg_error_invocation.tool_call_id == "call_surrogate"
    assert arg_error_invocation.error_class == "ToolArgumentError"
    assert arg_error_invocation.version_before == state.version
    assert arg_error_invocation.version_after is None
    assert rejected_content not in arg_error_invocation.arguments_canonical
    assert rejected_content not in json.dumps(message_snapshots[1])

    arg_error_outcomes = [outcome for outcome in invalid_turn_outcomes if outcome.error_class == "ToolArgumentError"]
    assert len(arg_error_outcomes) == 1
    assert arg_error_outcomes[0].pre_version == state.version
    assert arg_error_outcomes[0].post_version == state.version
    assert arg_error_outcomes[0].response is None

    feedback = next(message for message in message_snapshots[1] if message.get("tool_call_id") == "call_surrogate")
    feedback_payload = json.loads(feedback["content"])
    assert "object conforming to SetPipelineArgumentsModel" in feedback_payload["error"]
    assert rejected_content not in feedback["content"]

    persisted_feedback = _persisted_tool_content(harness, "call_surrogate")
    assert json.loads(persisted_feedback)["error_class"] == "ToolArgumentError"
    assert rejected_content not in persisted_feedback


@pytest.mark.asyncio
async def test_unexpected_candidate_finalizer_exception_uses_plugin_crash_audit_and_wrapper(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    args = _inline_pipeline_args(tmp_path)
    llm = _ScriptedLLM(_tool_turn("call_finalizer_crash", "set_pipeline", args))
    unexpected = RuntimeError("candidate finalizer internal failure with private detail")

    with (
        patch.object(harness.service, "_call_llm", new=llm),
        patch(
            "elspeth.web.composer.tool_batch.build_set_pipeline_candidate",
            wraps=real_build_set_pipeline_candidate,
        ) as builder,
        patch("elspeth.web.composer.tool_batch.finalize_tool_result", side_effect=unexpected) as finalizer,
        pytest.raises(ComposerPluginCrashError) as exc_info,
    ):
        await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    assert exc_info.value.original_exc is unexpected
    assert exc_info.value.__cause__ is unexpected
    assert exc_info.value.partial_state is None
    assert builder.call_count == 1
    assert finalizer.call_count == 1
    assert len(llm.message_snapshots) == 1
    assert await harness.sessions.list_composition_proposals(UUID(harness.session_id)) == []
    assert _count_rows(harness.engine, blobs_table) == 0
    assert _count_rows(harness.engine, composition_states_table) == 0

    outcomes = harness.service._phase3_last_tool_outcomes
    assert len(outcomes) == 1
    assert outcomes[0].error_class == "RuntimeError"
    assert outcomes[0].error_message == "RuntimeError"
    assert outcomes[0].pre_version == state.version
    assert outcomes[0].post_version == state.version

    assert exc_info.value.failed_turn is not None
    assert exc_info.value.failed_turn.tool_calls_attempted == 1
    persisted_feedback = _persisted_tool_content(harness, "call_finalizer_crash")
    assert json.loads(persisted_feedback) == {
        "error_class": "RuntimeError",
        "error_message": "RuntimeError",
    }
    assert str(unexpected) not in persisted_feedback


@pytest.mark.asyncio
async def test_preproposal_base_exception_is_audited_once_and_propagated_unchanged(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    args = _inline_pipeline_args(tmp_path)
    llm = _ScriptedLLM(_tool_turn("call_finalizer_base_signal", "set_pipeline", args))
    signal = _PreproposalBaseSignal("shutdown-style private detail")
    recorded: list[Any] = []
    original_record = BufferingRecorder.record

    def _record(recorder: BufferingRecorder, invocation: Any) -> None:
        recorded.append(invocation)
        original_record(recorder, invocation)

    with (
        patch.object(harness.service, "_call_llm", new=llm),
        patch(
            "elspeth.web.composer.tool_batch.build_set_pipeline_candidate",
            wraps=real_build_set_pipeline_candidate,
        ) as builder,
        patch("elspeth.web.composer.tool_batch.finalize_tool_result", side_effect=signal) as finalizer,
        patch.object(BufferingRecorder, "record", new=_record),
        pytest.raises(_PreproposalBaseSignal) as exc_info,
    ):
        await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    assert exc_info.value is signal
    assert not isinstance(exc_info.value, ComposerPluginCrashError)
    assert builder.call_count == 1
    assert finalizer.call_count == 1
    assert len(llm.message_snapshots) == 1
    assert await harness.sessions.list_composition_proposals(UUID(harness.session_id)) == []
    assert _count_rows(harness.engine, blobs_table) == 0
    assert _count_rows(harness.engine, composition_states_table) == 0

    assert len(recorded) == 1
    invocation = recorded[0]
    assert invocation.tool_call_id == "call_finalizer_base_signal"
    assert invocation.status is ComposerToolStatus.PLUGIN_CRASH
    assert invocation.error_class == "_PreproposalBaseSignal"
    assert invocation.error_message == "_PreproposalBaseSignal"
    assert invocation.version_before == state.version
    assert invocation.version_after is None
    assert invocation.result_canonical is None


@pytest.mark.asyncio
async def test_candidate_prior_validation_runtime_error_uses_plugin_crash_audit_and_wrapper(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    response = _tool_turn("call_prior_runtime_crash", "set_pipeline", _inline_pipeline_args(tmp_path))
    failure = RuntimeError("candidate-prior validation private detail")
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(harness.service._catalog)
    catalog = PolicyCatalogView.for_trained_operator(harness.service._catalog, snapshot)
    original_validate = catalog.validate_composition_state
    armed = False
    llm_calls = 0
    recorded: list[Any] = []
    original_record = BufferingRecorder.record

    def _validate(candidate_state: CompositionState) -> Any:
        if armed:
            raise failure
        return original_validate(candidate_state)

    async def _llm(messages: list[dict[str, Any]], _tools: Any) -> Any:
        nonlocal armed, llm_calls
        assert messages
        llm_calls += 1
        armed = True
        return response

    def _record(recorder: BufferingRecorder, invocation: Any) -> None:
        recorded.append(invocation)
        original_record(recorder, invocation)

    with (
        patch.object(harness.service, "_plugin_policy_context", return_value=(snapshot, catalog)),
        patch.object(harness.service, "_call_llm", new=_llm),
        patch.object(catalog, "validate_composition_state", side_effect=_validate) as validation,
        patch("elspeth.web.composer.tool_batch.build_set_pipeline_candidate") as builder,
        patch("elspeth.web.composer.tool_batch.finalize_tool_result") as finalizer,
        patch.object(BufferingRecorder, "record", new=_record),
        pytest.raises(ComposerPluginCrashError) as exc_info,
    ):
        await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    assert llm_calls == 1
    assert validation.call_count >= 1
    assert builder.call_count == 0
    assert finalizer.call_count == 0
    assert exc_info.value.original_exc is failure
    assert exc_info.value.__cause__ is failure
    assert exc_info.value.partial_state is None
    assert await harness.sessions.list_composition_proposals(UUID(harness.session_id)) == []
    assert _count_rows(harness.engine, blobs_table) == 0
    assert _count_rows(harness.engine, composition_states_table) == 0

    assert len(recorded) == 1
    invocation = recorded[0]
    assert invocation.tool_call_id == "call_prior_runtime_crash"
    assert invocation.status is ComposerToolStatus.PLUGIN_CRASH
    assert invocation.error_class == "RuntimeError"
    assert invocation.error_message == "RuntimeError"
    assert invocation.version_before == state.version
    assert invocation.version_after is None


@pytest.mark.asyncio
async def test_candidate_prior_validation_base_exception_is_audited_once_and_propagated_unchanged(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    response = _tool_turn("call_prior_base_signal", "set_pipeline", _inline_pipeline_args(tmp_path))
    signal = _PreproposalBaseSignal("candidate-prior shutdown-style private detail")
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(harness.service._catalog)
    catalog = PolicyCatalogView.for_trained_operator(harness.service._catalog, snapshot)
    original_validate = catalog.validate_composition_state
    armed = False
    llm_calls = 0
    recorded: list[Any] = []
    original_record = BufferingRecorder.record

    def _validate(candidate_state: CompositionState) -> Any:
        if armed:
            raise signal
        return original_validate(candidate_state)

    async def _llm(messages: list[dict[str, Any]], _tools: Any) -> Any:
        nonlocal armed, llm_calls
        assert messages
        llm_calls += 1
        armed = True
        return response

    def _record(recorder: BufferingRecorder, invocation: Any) -> None:
        recorded.append(invocation)
        original_record(recorder, invocation)

    with (
        patch.object(harness.service, "_plugin_policy_context", return_value=(snapshot, catalog)),
        patch.object(harness.service, "_call_llm", new=_llm),
        patch.object(catalog, "validate_composition_state", side_effect=_validate) as validation,
        patch("elspeth.web.composer.tool_batch.build_set_pipeline_candidate") as builder,
        patch("elspeth.web.composer.tool_batch.finalize_tool_result") as finalizer,
        patch.object(BufferingRecorder, "record", new=_record),
        pytest.raises(_PreproposalBaseSignal) as exc_info,
    ):
        await harness.service.compose(
            "Build a reviewed inline pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    assert llm_calls == 1
    assert validation.call_count >= 1
    assert builder.call_count == 0
    assert finalizer.call_count == 0
    assert exc_info.value is signal
    assert not isinstance(exc_info.value, ComposerPluginCrashError)
    assert await harness.sessions.list_composition_proposals(UUID(harness.session_id)) == []
    assert _count_rows(harness.engine, blobs_table) == 0
    assert _count_rows(harness.engine, composition_states_table) == 0

    assert len(recorded) == 1
    invocation = recorded[0]
    assert invocation.tool_call_id == "call_prior_base_signal"
    assert invocation.status is ComposerToolStatus.PLUGIN_CRASH
    assert invocation.error_class == "_PreproposalBaseSignal"
    assert invocation.error_message == "_PreproposalBaseSignal"
    assert invocation.version_before == state.version
    assert invocation.version_after is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected_proposals", "expected_error_class"),
    [
        ({"patch": {"name": "reviewed"}}, 1, None),
        ({"patch": "not-an-object"}, 1, None),
    ],
    ids=("valid", "invalid"),
)
async def test_non_pipeline_explicit_approval_behavior_is_unchanged(
    tmp_path: Path,
    arguments: dict[str, Any],
    expected_proposals: int,
    expected_error_class: str | None,
) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    llm = _ScriptedLLM(
        _tool_turn("call_metadata", "set_metadata", arguments),
        _fake_llm_response(content="Done."),
    )

    with (
        patch.object(harness.service, "_call_llm", new=llm),
        patch("elspeth.web.composer.tool_batch.build_set_pipeline_candidate") as builder,
    ):
        result = await harness.service.compose(
            "Update metadata under explicit approval.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    proposals = await harness.sessions.list_composition_proposals(UUID(harness.session_id))
    assert len(proposals) == expected_proposals
    assert builder.call_count == 0
    assert result.state is state
    outcome = harness.service._phase3_last_tool_outcomes[0]
    assert outcome.error_class == expected_error_class
    assert outcome.post_version == state.version


@pytest.mark.asyncio
async def test_auto_commit_set_pipeline_uses_candidate_builder_once(tmp_path: Path) -> None:
    harness = _harness(tmp_path)
    state = _empty_state()
    await harness.sessions.update_composer_preferences(
        UUID(harness.session_id),
        trust_mode="auto_commit",
        density_default="high",
        actor="user:proposal-prevalidation-user",
    )
    llm = _ScriptedLLM(
        _tool_turn("call_auto", "set_pipeline", _valid_pipeline_args(tmp_path, metadata_name="auto")),
        _fake_llm_response(content="Applied."),
    )

    with (
        patch.object(harness.service, "_call_llm", new=llm),
        patch(
            "elspeth.web.composer.tools.sessions.build_set_pipeline_candidate",
            wraps=real_build_set_pipeline_candidate,
        ) as builder,
    ):
        result = await harness.service.compose(
            "Build and apply a pipeline.",
            [],
            state,
            session_id=harness.session_id,
            user_id="proposal-prevalidation-user",
            user_message_id=harness.user_message_id,
        )

    assert builder.call_count == 1
    assert result.state.version == state.version + 1
    assert await harness.sessions.list_composition_proposals(UUID(harness.session_id)) == []
