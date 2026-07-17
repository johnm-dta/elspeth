from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder, begin_dispatch, finish_plugin_crash, finish_success
from elspeth.web.composer.pipeline_commit import (
    PipelineCommitConfig,
    PipelineCommitError,
    PipelineCommitMismatchError,
    PipelineDispatchAuditBinding,
    prepare_pipeline_proposal_commit,
)
from elspeth.web.composer.pipeline_planner import PipelinePlanResult
from elspeth.web.composer.pipeline_proposal import AbsentBase, PipelineProposal, PlannerSurface, PresentBase
from elspeth.web.composer.redaction import redact_tool_call_arguments
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import CompositionStateData, StaleComposeStateError
from elspeth.web.sessions.routes._helpers import _persist_tool_invocations
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@pytest.fixture
def service() -> SessionServiceImpl:
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    return SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))


def _insert_session(service: SessionServiceImpl, session_id: UUID) -> None:
    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=str(session_id),
                user_id="alice",
                auth_provider_type="local",
                title="Pipeline lifecycle",
                trust_mode="explicit_approve",
                density_default="high",
                created_at=now,
                updated_at=now,
            )
        )


def _pipeline() -> dict[str, object]:
    return {"sources": {}, "nodes": [], "edges": [], "outputs": []}


def _state_data(*, composer_meta: dict[str, object] | None = None) -> CompositionStateData:
    return CompositionStateData(
        sources={},
        nodes=(),
        edges=(),
        outputs=(),
        metadata_={"name": "Untitled Pipeline", "description": ""},
        is_valid=True,
        validation_errors=(),
        composer_meta=composer_meta,
    )


def _state_content_hash(state: CompositionStateData) -> str:
    return stable_hash(
        {
            "sources": state.sources,
            "nodes": state.nodes,
            "edges": state.edges,
            "outputs": state.outputs,
            "metadata": state.metadata_,
        }
    )


def _plan(base: AbsentBase | PresentBase | None = None) -> PipelinePlanResult:
    if base is None:
        base = AbsentBase()
    return PipelinePlanResult(
        proposal=PipelineProposal.create(
            pipeline=_pipeline(),
            base=base,
            reviewed_facts={},
            surface=PlannerSurface.FREEFORM,
            repair_count=0,
            skill_hash=stable_hash("skill"),
        ),
        tool_call_id="planner-terminal-call",
        custody_result="not_required",
    )


def _runnable_pipeline(tmp_path) -> dict[str, object]:
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": str(tmp_path / "blobs" / "input.csv"), "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "sink_name": "rows",
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
    }


def _runnable_plan(tmp_path) -> PipelinePlanResult:
    return PipelinePlanResult(
        proposal=PipelineProposal.create(
            pipeline=_runnable_pipeline(tmp_path),
            base=AbsentBase(),
            reviewed_facts={},
            surface=PlannerSurface.FREEFORM,
            repair_count=0,
            skill_hash=stable_hash("skill"),
        ),
        tool_call_id="planner-terminal-call",
        custody_result="not_required",
    )


def _redacted_pipeline(pipeline: dict[str, object]) -> dict[str, object]:
    return redact_tool_call_arguments("set_pipeline", pipeline, telemetry=NoopRedactionTelemetry())


async def _create(service: SessionServiceImpl, session_id: UUID, plan: PipelinePlanResult):
    return await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=plan,
        summary="Replace the pipeline.",
        rationale="Requested by the operator.",
        affects=("graph", "validation"),
        arguments_redacted_json=_redacted_pipeline(_pipeline()),
        actor="composer-web:user:alice",
        composer_model_identifier="planner-model",
        composer_model_version="planner-model-v1",
        composer_provider="provider",
    )


async def _persist_dispatch(
    service: SessionServiceImpl,
    session_id: UUID,
    *,
    tool_call_id: str = "planner-terminal-call",
    result_hash_suffix: str = "",
) -> PipelineDispatchAuditBinding:
    audit = begin_dispatch(
        tool_call_id,
        "set_pipeline",
        _pipeline(),
        version_before=0,
        actor="user:alice",
    )
    invocation = finish_success(
        audit,
        result_payload={
            "success": True,
            "content_hash": _state_content_hash(_state_data()) + result_hash_suffix,
            "pipeline_content_hash_schema": "composer.pipeline-dispatch-result.v1",
            "pipeline_content_hash": _state_content_hash(_state_data()),
        },
        version_after=1,
    )
    bindings = await _persist_tool_invocations(
        service,
        session_id,
        (invocation,),
        None,
        plugin_crash_pending=False,
    )
    assert len(bindings) == 1
    return bindings[0]


async def _persist_failed_dispatch(
    service: SessionServiceImpl,
    session_id: UUID,
    *,
    tool_call_id: str = "planner-terminal-call",
) -> None:
    audit = begin_dispatch(
        tool_call_id,
        "set_pipeline",
        _pipeline(),
        version_before=0,
        actor="user:alice",
    )
    await _persist_tool_invocations(
        service,
        session_id,
        (finish_plugin_crash(audit, exc=RuntimeError("redacted")),),
        None,
        plugin_crash_pending=False,
    )


def _settlement_kwargs(session_id: UUID, proposal_id: UUID, plan: PipelinePlanResult, binding: PipelineDispatchAuditBinding):
    return {
        "session_id": session_id,
        "proposal_id": proposal_id,
        "draft_hash": plan.proposal.draft_hash,
        "reviewed_facts": {},
        "state": _state_data(),
        "candidate_content_hash": _state_content_hash(_state_data()),
        "executor_content_hash": _state_content_hash(_state_data()),
        "final_composer_metadata": None,
        "dispatch": binding,
        "actor": "user:alice",
    }


def _latest_audit_envelope(service: SessionServiceImpl) -> dict[str, object]:
    with service._engine.begin() as conn:
        row = conn.execute(
            select(chat_messages_table.c.tool_calls)
            .where(chat_messages_table.c.role == "audit")
            .order_by(chat_messages_table.c.created_at.desc())
            .limit(1)
        ).one()
    envelope = deep_thaw(row.tool_calls[0])
    assert type(envelope) is dict
    return envelope


async def _persist_cloned_audit_envelope(
    service: SessionServiceImpl,
    session_id: UUID,
    envelope: dict[str, object],
) -> None:
    await service.add_message(
        session_id,
        role="audit",
        content="set_pipeline: success",
        tool_calls=[envelope],
        writer_principal="compose_loop",
    )


@pytest.mark.asyncio
async def test_atomic_pipeline_settlement_inserts_state_terminal_event_and_row_together(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    final_meta = {"surface": "freeform", "draft_hash": plan.proposal.draft_hash}

    settled = await service.settle_pipeline_composition_proposal(
        session_id=session_id,
        proposal_id=row.id,
        draft_hash=plan.proposal.draft_hash,
        reviewed_facts={},
        state=_state_data(),
        candidate_content_hash=_state_content_hash(_state_data()),
        executor_content_hash=_state_content_hash(_state_data()),
        final_composer_metadata=final_meta,
        dispatch=binding,
        actor="user:alice",
    )

    assert settled.proposal.status == "committed"
    assert settled.proposal.committed_state_id == settled.state.id
    assert settled.state.composer_meta == final_meta
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == ["proposal.created", "proposal.accepted"]
    terminal = events[-1].payload
    assert set(terminal) == {
        "schema",
        "tool_call_id",
        "tool_name",
        "status",
        "outcome",
        "draft_hash",
        "committed_state_id",
        "committed_state_content_hash",
        "final_composer_metadata_hash",
        "dispatch",
    }
    assert terminal["schema"] == "pipeline_proposal_accepted.v1"
    assert terminal["dispatch"] == binding.to_dict()


@pytest.mark.asyncio
async def test_atomic_pipeline_settlement_exact_retry_returns_same_state_and_event(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    kwargs = {
        "session_id": session_id,
        "proposal_id": row.id,
        "draft_hash": plan.proposal.draft_hash,
        "reviewed_facts": {},
        "state": _state_data(),
        "candidate_content_hash": _state_content_hash(_state_data()),
        "executor_content_hash": _state_content_hash(_state_data()),
        "final_composer_metadata": None,
        "dispatch": binding,
        "actor": "user:alice",
    }

    first = await service.settle_pipeline_composition_proposal(**kwargs)
    second = await service.settle_pipeline_composition_proposal(**kwargs)

    assert second == first
    with service._engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(composition_states_table)).scalar_one() == 1
        assert (
            conn.execute(
                select(func.count()).select_from(proposal_events_table).where(proposal_events_table.c.event_type == "proposal.accepted")
            ).scalar_one()
            == 1
        )


@pytest.mark.asyncio
async def test_atomic_pipeline_settlement_exact_retry_rejects_tampered_terminal_event_pointer(
    service: SessionServiceImpl,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    kwargs = {
        "session_id": session_id,
        "proposal_id": row.id,
        "draft_hash": plan.proposal.draft_hash,
        "reviewed_facts": {},
        "state": _state_data(),
        "candidate_content_hash": _state_content_hash(_state_data()),
        "executor_content_hash": _state_content_hash(_state_data()),
        "final_composer_metadata": None,
        "dispatch": binding,
        "actor": "user:alice",
    }
    await service.settle_pipeline_composition_proposal(**kwargs)
    with service._engine.begin() as conn:
        conn.execute(
            update(composition_proposals_table).where(composition_proposals_table.c.id == str(row.id)).values(audit_event_id=str(uuid4()))
        )

    with pytest.raises(AuditIntegrityError, match="terminal event"):
        await service.settle_pipeline_composition_proposal(**kwargs)


@pytest.mark.asyncio
async def test_absent_base_conflicts_when_first_state_appears_before_settlement(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    await service.save_composition_state(session_id, _state_data(), provenance="session_seed")

    with pytest.raises(StaleComposeStateError):
        await service.settle_pipeline_composition_proposal(
            session_id=session_id,
            proposal_id=row.id,
            draft_hash=plan.proposal.draft_hash,
            reviewed_facts={},
            state=_state_data(),
            candidate_content_hash=_state_content_hash(_state_data()),
            executor_content_hash=_state_content_hash(_state_data()),
            final_composer_metadata=None,
            dispatch=binding,
            actor="user:alice",
        )
    assert (await service.list_composition_proposals(session_id))[0].status == "pending"


@pytest.mark.asyncio
async def test_present_base_conflicts_on_same_content_new_state_id(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    base_record = await service.save_composition_state(session_id, _state_data(), provenance="session_seed")
    base = PresentBase(state_id=base_record.id, composition_content_hash=_state_content_hash(_state_data()))
    plan = _plan(base)
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    await service.save_composition_state(session_id, _state_data(), provenance="session_seed")

    with pytest.raises(StaleComposeStateError):
        await service.settle_pipeline_composition_proposal(
            session_id=session_id,
            proposal_id=row.id,
            draft_hash=plan.proposal.draft_hash,
            reviewed_facts={},
            state=_state_data(),
            candidate_content_hash=_state_content_hash(_state_data()),
            executor_content_hash=_state_content_hash(_state_data()),
            final_composer_metadata=None,
            dispatch=binding,
            actor="user:alice",
        )


@pytest.mark.asyncio
async def test_settlement_rolls_back_state_event_and_status_when_interrupted_after_insert(
    service: SessionServiceImpl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    original = service._insert_composition_state

    def interrupt_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("settlement interruption")

    monkeypatch.setattr(service, "_insert_composition_state", interrupt_after_insert)
    with pytest.raises(RuntimeError, match="settlement interruption"):
        await service.settle_pipeline_composition_proposal(
            session_id=session_id,
            proposal_id=row.id,
            draft_hash=plan.proposal.draft_hash,
            reviewed_facts={},
            state=_state_data(),
            candidate_content_hash=_state_content_hash(_state_data()),
            executor_content_hash=_state_content_hash(_state_data()),
            final_composer_metadata=None,
            dispatch=binding,
            actor="user:alice",
        )

    with service._engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(composition_states_table)).scalar_one() == 0
        assert (
            conn.execute(
                select(func.count()).select_from(proposal_events_table).where(proposal_events_table.c.event_type == "proposal.accepted")
            ).scalar_one()
            == 0
        )
        assert conn.execute(select(composition_proposals_table.c.status)).scalar_one() == "pending"


@pytest.mark.asyncio
async def test_settlement_rejects_missing_or_tampered_durable_dispatch_audit(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    with service._engine.begin() as conn:
        audit_row = conn.execute(select(chat_messages_table).where(chat_messages_table.c.role == "audit")).one()
        envelope = list(audit_row.tool_calls)
        envelope[0]["invocation"]["arguments_hash"] = "0" * 64
        conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == audit_row.id).values(tool_calls=envelope))

    with pytest.raises(AuditIntegrityError, match="dispatch audit"):
        await service.settle_pipeline_composition_proposal(
            session_id=session_id,
            proposal_id=row.id,
            draft_hash=plan.proposal.draft_hash,
            reviewed_facts={},
            state=_state_data(),
            candidate_content_hash=_state_content_hash(_state_data()),
            executor_content_hash=_state_content_hash(_state_data()),
            final_composer_metadata=None,
            dispatch=binding,
            actor="user:alice",
        )


@pytest.mark.asyncio
async def test_settlement_rejects_unrelated_successful_dispatch_call_id(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id, tool_call_id="unrelated-call")

    with pytest.raises(AuditIntegrityError, match="tool call"):
        await service.settle_pipeline_composition_proposal(
            session_id=session_id,
            proposal_id=row.id,
            draft_hash=plan.proposal.draft_hash,
            reviewed_facts={},
            state=_state_data(),
            candidate_content_hash=_state_content_hash(_state_data()),
            executor_content_hash=_state_content_hash(_state_data()),
            final_composer_metadata=None,
            dispatch=binding,
            actor="user:alice",
        )


@pytest.mark.asyncio
async def test_settlement_rejects_concurrent_successes_for_same_proposal_call_id(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    state_hash = _state_content_hash(_state_data())
    first_audit = begin_dispatch(plan.tool_call_id, "set_pipeline", _pipeline(), version_before=0, actor="user:alice")
    second_audit = begin_dispatch(plan.tool_call_id, "set_pipeline", _pipeline(), version_before=0, actor="user:alice")
    first = finish_success(
        first_audit,
        result_payload={
            "success": True,
            "attempt": 1,
            "pipeline_content_hash_schema": "composer.pipeline-dispatch-result.v1",
            "pipeline_content_hash": state_hash,
        },
        version_after=1,
    )
    second = finish_success(
        second_audit,
        result_payload={
            "success": True,
            "attempt": 2,
            "pipeline_content_hash_schema": "composer.pipeline-dispatch-result.v1",
            "pipeline_content_hash": state_hash,
        },
        version_after=1,
    )
    persisted = await asyncio.gather(
        _persist_tool_invocations(service, session_id, (first,), None, plugin_crash_pending=False),
        _persist_tool_invocations(service, session_id, (second,), None, plugin_crash_pending=False),
    )
    binding = persisted[0][0]

    with pytest.raises(AuditIntegrityError, match=r"duplicate|exactly one|one durable|does not match"):
        await service.settle_pipeline_composition_proposal(**_settlement_kwargs(session_id, row.id, plan, binding))


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["wrong_tool", "unknown_status", "arguments_hash", "result_canonical"])
async def test_settlement_rejects_malformed_or_mismatched_success_for_same_call_id(
    service: SessionServiceImpl,
    corruption: str,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    malformed = _latest_audit_envelope(service)
    invocation = malformed["invocation"]
    assert type(invocation) is dict
    if corruption == "wrong_tool":
        invocation["tool_name"] = "inspect_pipeline"
    elif corruption == "unknown_status":
        invocation["status"] = "not-a-status"
    elif corruption == "arguments_hash":
        invocation["arguments_hash"] = "0" * 64
    else:
        invocation["result_canonical"] = "{"
    await _persist_cloned_audit_envelope(service, session_id, malformed)

    with pytest.raises(AuditIntegrityError):
        await service.settle_pipeline_composition_proposal(**_settlement_kwargs(session_id, row.id, plan, binding))


@pytest.mark.asyncio
async def test_failed_dispatch_then_one_success_remains_recoverable(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    await _persist_failed_dispatch(service, session_id)
    binding = await _persist_dispatch(service, session_id)

    settled = await service.settle_pipeline_composition_proposal(**_settlement_kwargs(session_id, row.id, plan, binding))

    assert settled.proposal.status == "committed"


@pytest.mark.asyncio
async def test_committed_retry_rejects_duplicate_successful_dispatch_evidence(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    kwargs = _settlement_kwargs(session_id, row.id, plan, binding)
    await service.settle_pipeline_composition_proposal(**kwargs)
    await _persist_cloned_audit_envelope(service, session_id, _latest_audit_envelope(service))

    with pytest.raises(AuditIntegrityError, match=r"duplicated|duplicate|one durable"):
        await service.settle_pipeline_composition_proposal(**kwargs)


@pytest.mark.asyncio
async def test_rejection_rejects_duplicate_successful_dispatch_evidence(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)
    await _persist_cloned_audit_envelope(service, session_id, _latest_audit_envelope(service))

    with pytest.raises(AuditIntegrityError, match=r"exactly one|duplicate"):
        await service.reject_pipeline_composition_proposal(
            session_id=session_id,
            proposal_id=row.id,
            draft_hash=plan.proposal.draft_hash,
            reviewed_facts={},
            reason="candidate_executor_mismatch",
            dispatch=binding,
            actor="system:pipeline-commit",
        )


@pytest.mark.asyncio
async def test_pipeline_rejection_uses_closed_versioned_reason_and_is_exactly_idempotent(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)

    first = await service.reject_pipeline_composition_proposal(
        session_id=session_id,
        proposal_id=row.id,
        draft_hash=plan.proposal.draft_hash,
        reviewed_facts={},
        reason="operator_rejected",
        dispatch=None,
        actor="user:alice",
    )
    second = await service.reject_pipeline_composition_proposal(
        session_id=session_id,
        proposal_id=row.id,
        draft_hash=plan.proposal.draft_hash,
        reviewed_facts={},
        reason="operator_rejected",
        dispatch=None,
        actor="user:alice",
    )

    assert first == second
    assert first.status == "rejected"
    events = await service.list_proposal_events(session_id)
    assert events[-1].payload == {
        "schema": "pipeline_proposal_rejected.v1",
        "tool_call_id": plan.tool_call_id,
        "tool_name": "set_pipeline",
        "status": "rejected",
        "outcome": "rejected",
        "reason_code": "operator_rejected",
        "draft_hash": plan.proposal.draft_hash,
        "dispatch": None,
    }
    with pytest.raises((TypeError, ValueError, AuditIntegrityError)):
        await service.reject_pipeline_composition_proposal(
            session_id=session_id,
            proposal_id=row.id,
            draft_hash=plan.proposal.draft_hash,
            reviewed_facts={},
            reason="operator said no because RAW FREE TEXT",
            dispatch=None,
            actor="user:alice",
        )


@pytest.mark.asyncio
async def test_pipeline_rejection_exact_retry_rejects_tampered_terminal_event_pointer(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    kwargs = {
        "session_id": session_id,
        "proposal_id": row.id,
        "draft_hash": plan.proposal.draft_hash,
        "reviewed_facts": {},
        "reason": "operator_rejected",
        "dispatch": None,
        "actor": "user:alice",
    }
    await service.reject_pipeline_composition_proposal(**kwargs)
    with service._engine.begin() as conn:
        conn.execute(
            update(composition_proposals_table).where(composition_proposals_table.c.id == str(row.id)).values(audit_event_id=str(uuid4()))
        )

    with pytest.raises(AuditIntegrityError, match="terminal binding"):
        await service.reject_pipeline_composition_proposal(**kwargs)


async def _reload_canonical_proposal(
    service: SessionServiceImpl,
    session_id: UUID,
    proposal_id: UUID,
    accessor: str,
) -> None:
    if accessor == "get":
        await service.get_authoritative_composition_proposal(
            session_id=session_id,
            proposal_id=proposal_id,
            reviewed_facts={},
        )
    else:
        await service.list_composition_proposals(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("accessor", ["get", "list"])
async def test_pending_canonical_reload_rejects_terminal_event(
    service: SessionServiceImpl,
    accessor: str,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    with service._engine.begin() as conn:
        conn.execute(
            insert(proposal_events_table).values(
                id=str(uuid4()),
                session_id=str(session_id),
                proposal_id=str(row.id),
                event_type="proposal.rejected",
                actor="tamper",
                payload={"status": "rejected"},
                created_at=datetime.now(UTC),
            )
        )

    with pytest.raises(AuditIntegrityError, match="pending"):
        await _reload_canonical_proposal(service, session_id, row.id, accessor)


@pytest.mark.asyncio
@pytest.mark.parametrize("accessor", ["get", "list"])
async def test_pending_canonical_reload_rejects_non_creation_event_pointer(
    service: SessionServiceImpl,
    accessor: str,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    with service._engine.begin() as conn:
        conn.execute(
            update(composition_proposals_table).where(composition_proposals_table.c.id == str(row.id)).values(audit_event_id=str(uuid4()))
        )

    with pytest.raises(AuditIntegrityError, match="pending"):
        await _reload_canonical_proposal(service, session_id, row.id, accessor)


@pytest.mark.asyncio
@pytest.mark.parametrize("accessor", ["get", "list"])
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "tampered", "pointer"])
async def test_rejected_canonical_reload_fails_closed_on_terminal_corruption(
    service: SessionServiceImpl,
    accessor: str,
    corruption: str,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    await service.reject_pipeline_composition_proposal(
        session_id=session_id,
        proposal_id=row.id,
        draft_hash=plan.proposal.draft_hash,
        reviewed_facts={},
        reason="operator_rejected",
        dispatch=None,
        actor="user:alice",
    )
    with service._engine.begin() as conn:
        terminal = conn.execute(
            select(proposal_events_table)
            .where(proposal_events_table.c.proposal_id == str(row.id))
            .where(proposal_events_table.c.event_type == "proposal.rejected")
        ).one()
        if corruption == "missing":
            conn.execute(delete(proposal_events_table).where(proposal_events_table.c.id == terminal.id))
        elif corruption == "duplicate":
            conn.execute(
                insert(proposal_events_table).values(
                    id=str(uuid4()),
                    session_id=terminal.session_id,
                    proposal_id=terminal.proposal_id,
                    event_type=terminal.event_type,
                    actor=terminal.actor,
                    payload=deep_thaw(terminal.payload),
                    created_at=datetime.now(UTC),
                )
            )
        elif corruption == "tampered":
            payload = deep_thaw(terminal.payload)
            assert type(payload) is dict
            payload["draft_hash"] = "0" * 64
            conn.execute(update(proposal_events_table).where(proposal_events_table.c.id == terminal.id).values(payload=payload))
        else:
            conn.execute(
                update(composition_proposals_table)
                .where(composition_proposals_table.c.id == str(row.id))
                .values(audit_event_id=str(uuid4()))
            )

    with pytest.raises(AuditIntegrityError, match="rejected"):
        await _reload_canonical_proposal(service, session_id, row.id, accessor)


@pytest.mark.asyncio
async def test_pipeline_candidate_executor_mismatch_rejection_binds_dispatch_and_publishes_no_state(service: SessionServiceImpl) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _plan()
    row = await _create(service, session_id, plan)
    binding = await _persist_dispatch(service, session_id)

    rejected = await service.reject_pipeline_composition_proposal(
        session_id=session_id,
        proposal_id=row.id,
        draft_hash=plan.proposal.draft_hash,
        reviewed_facts={},
        reason="candidate_executor_mismatch",
        dispatch=binding,
        actor="system:pipeline-commit",
    )

    assert rejected.status == "rejected"
    with service._engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(composition_states_table)).scalar_one() == 0
    terminal = (await service.list_proposal_events(session_id))[-1].payload
    assert terminal["outcome"] == "failed"
    assert terminal["reason_code"] == "candidate_executor_mismatch"
    assert terminal["dispatch"] == binding.to_dict()


def test_dispatch_binding_is_closed_and_success_only() -> None:
    binding = PipelineDispatchAuditBinding(
        tool_call_id="call",
        tool_name="set_pipeline",
        status=ComposerToolStatus.SUCCESS,
        arguments_hash="a" * 64,
        result_hash="b" * 64,
    )
    assert set(binding.to_dict()) == {"tool_call_id", "tool_name", "status", "arguments_hash", "result_hash"}
    with pytest.raises((TypeError, ValueError, AuditIntegrityError)):
        PipelineDispatchAuditBinding(
            tool_call_id="call",
            tool_name="set_pipeline",
            status=ComposerToolStatus.PLUGIN_CRASH,
            arguments_hash="a" * 64,
            result_hash="b" * 64,
        )


@pytest.mark.asyncio
async def test_prepare_pipeline_commit_revalidates_and_audits_exact_arguments_without_settlement(
    service: SessionServiceImpl,
    tmp_path,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _runnable_plan(tmp_path)
    row = await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=plan,
        summary="Replace the pipeline.",
        rationale="Requested by the operator.",
        affects=("graph", "validation"),
        arguments_redacted_json=_redacted_pipeline(_runnable_pipeline(tmp_path)),
        actor="composer-web:user:alice",
        composer_model_identifier="planner-model",
        composer_model_version="planner-model-v1",
        composer_provider="provider",
    )
    authority = await service.get_authoritative_pipeline_proposal(
        session_id=session_id,
        proposal_id=row.id,
        reviewed_facts={},
    )
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    policy = PolicyCatalogView.for_trained_operator(catalog, snapshot)
    recorder = BufferingRecorder()
    current = CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)

    prepared = await prepare_pipeline_proposal_commit(
        authority=authority,
        current_state=current,
        current_state_id=None,
        policy_catalog=policy,
        plugin_snapshot=snapshot,
        config=PipelineCommitConfig(
            data_dir=str(tmp_path),
            session_engine=service._engine,
            secret_service=None,
            user_id="alice",
            user_message_content=None,
            max_blob_storage_per_session_bytes=1_000_000,
            runtime_preflight=None,
            timeout_seconds=5.0,
        ),
        recorder=recorder,
        actor="user:alice",
    )

    assert prepared.dispatch.tool_call_id == plan.tool_call_id
    assert prepared.dispatch.arguments_hash == stable_hash(plan.proposal.pipeline)
    assert prepared.candidate_content_hash == prepared.executor_content_hash
    assert len(recorder.invocations) == 1
    assert await service.get_current_state(session_id) is None
    assert (await service.list_composition_proposals(session_id))[0].status == "pending"


@pytest.mark.asyncio
async def test_prepare_pipeline_commit_runs_blocking_policy_validation_off_event_loop(
    service: SessionServiceImpl,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _runnable_plan(tmp_path)
    row = await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=plan,
        summary="Replace the pipeline.",
        rationale="Requested by the operator.",
        affects=("graph",),
        arguments_redacted_json=_redacted_pipeline(_runnable_pipeline(tmp_path)),
        actor="composer-web:user:alice",
        composer_model_identifier="planner-model",
        composer_model_version="planner-model-v1",
        composer_provider="provider",
    )
    authority = await service.get_authoritative_pipeline_proposal(
        session_id=session_id,
        proposal_id=row.id,
        reviewed_facts={},
    )
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    policy = PolicyCatalogView.for_trained_operator(catalog, snapshot)
    original_validate = policy.validate_composition_state
    started = threading.Event()
    release = threading.Event()
    validation_thread_ids: list[int] = []
    event_loop_thread_id = threading.get_ident()

    def blocking_validate(state: CompositionState):
        validation_thread_ids.append(threading.get_ident())
        started.set()
        release.wait(timeout=0.4)
        return original_validate(state)

    monkeypatch.setattr(policy, "validate_composition_state", blocking_validate)
    task = asyncio.create_task(
        prepare_pipeline_proposal_commit(
            authority=authority,
            current_state=CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1),
            current_state_id=None,
            policy_catalog=policy,
            plugin_snapshot=snapshot,
            config=PipelineCommitConfig(
                data_dir=str(tmp_path),
                session_engine=service._engine,
                secret_service=None,
                user_id="alice",
                user_message_content=None,
                max_blob_storage_per_session_bytes=1_000_000,
                runtime_preflight=None,
                timeout_seconds=2.0,
            ),
            recorder=BufferingRecorder(),
            actor="user:alice",
        )
    )
    for _ in range(50):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()
    assert not task.done()
    assert validation_thread_ids == [validation_thread_ids[0]]
    assert validation_thread_ids[0] != event_loop_thread_id
    release.set()

    await task


@pytest.mark.asyncio
async def test_prepare_pipeline_commit_uses_one_total_timeout_budget(
    service: SessionServiceImpl,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _runnable_plan(tmp_path)
    row = await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=plan,
        summary="Replace the pipeline.",
        rationale="Requested by the operator.",
        affects=("graph",),
        arguments_redacted_json=_redacted_pipeline(_runnable_pipeline(tmp_path)),
        actor="composer-web:user:alice",
        composer_model_identifier="planner-model",
        composer_model_version="planner-model-v1",
        composer_provider="provider",
    )
    authority = await service.get_authoritative_pipeline_proposal(
        session_id=session_id,
        proposal_id=row.id,
        reviewed_facts={},
    )
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    policy = PolicyCatalogView.for_trained_operator(catalog, snapshot)
    original_validate = policy.validate_composition_state

    def slow_validate(state: CompositionState):
        time.sleep(0.12)
        return original_validate(state)

    from elspeth.web.composer.tools.sessions import build_set_pipeline_candidate as original_candidate

    def slow_candidate(*args, **kwargs):
        time.sleep(0.12)
        return original_candidate(*args, **kwargs)

    monkeypatch.setattr(policy, "validate_composition_state", slow_validate)
    monkeypatch.setattr("elspeth.web.composer.pipeline_commit.build_set_pipeline_candidate", slow_candidate)
    with pytest.raises(PipelineCommitError, match="timed out") as exc_info:
        await prepare_pipeline_proposal_commit(
            authority=authority,
            current_state=CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1),
            current_state_id=None,
            policy_catalog=policy,
            plugin_snapshot=snapshot,
            config=PipelineCommitConfig(
                data_dir=str(tmp_path),
                session_engine=service._engine,
                secret_service=None,
                user_id="alice",
                user_message_content=None,
                max_blob_storage_per_session_bytes=1_000_000,
                runtime_preflight=None,
                timeout_seconds=0.2,
            ),
            recorder=BufferingRecorder(),
            actor="user:alice",
        )

    assert exc_info.value.code == "TIMEOUT"


@pytest.mark.asyncio
async def test_prepare_pipeline_commit_detects_candidate_executor_mismatch_after_audited_dispatch(
    service: SessionServiceImpl,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = uuid4()
    _insert_session(service, session_id)
    plan = _runnable_plan(tmp_path)
    row = await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=plan,
        summary="Replace the pipeline.",
        rationale="Requested by the operator.",
        affects=("graph",),
        arguments_redacted_json=_redacted_pipeline(_runnable_pipeline(tmp_path)),
        actor="composer-web:user:alice",
        composer_model_identifier="planner-model",
        composer_model_version="planner-model-v1",
        composer_provider="provider",
    )
    authority = await service.get_authoritative_pipeline_proposal(
        session_id=session_id,
        proposal_id=row.id,
        reviewed_facts={},
    )
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    policy = PolicyCatalogView.for_trained_operator(catalog, snapshot)
    recorder = BufferingRecorder()
    current = CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)

    import elspeth.web.composer.pipeline_commit as commit_module

    original_execute = commit_module.execute_tool

    def mismatching_execute(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        return replace(
            result,
            updated_state=CompositionState(
                source=None,
                nodes=(),
                edges=(),
                outputs=(),
                metadata=PipelineMetadata(name="mismatch"),
                version=result.updated_state.version,
            ),
        )

    monkeypatch.setattr(commit_module, "execute_tool", mismatching_execute)
    with pytest.raises(PipelineCommitMismatchError):
        await prepare_pipeline_proposal_commit(
            authority=authority,
            current_state=current,
            current_state_id=None,
            policy_catalog=policy,
            plugin_snapshot=snapshot,
            config=PipelineCommitConfig(
                data_dir=str(tmp_path),
                session_engine=service._engine,
                secret_service=None,
                user_id="alice",
                user_message_content=None,
                max_blob_storage_per_session_bytes=1_000_000,
                runtime_preflight=None,
                timeout_seconds=5.0,
            ),
            recorder=recorder,
            actor="user:alice",
        )

    assert len(recorder.invocations) == 1
    assert await service.get_current_state(session_id) is None
