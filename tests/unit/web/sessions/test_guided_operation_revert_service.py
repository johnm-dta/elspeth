"""Atomic state-revert settlement under a guided-operation fence."""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import Callable, Coroutine
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import delete, event, func, select, update
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.audit import begin_dispatch, finish_success
from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import (
    ComponentTarget,
    GuidedProposalRef,
    GuidedSession,
    TerminalKind,
    TerminalState,
    TurnRecord,
)
from elspeth.web.composer.pipeline_commit import PipelineDispatchAuditBinding
from elspeth.web.composer.pipeline_planner import PipelinePlanResult
from elspeth.web.composer.pipeline_proposal import PipelineProposal, PlannerSurface, PresentBase, composition_content_hash
from elspeth.web.composer.redaction import redact_tool_call_arguments
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    guided_operations_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    CompositionStateRecord,
    GuidedCompositionStateResult,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationFence,
    GuidedOperationFenceLostError,
    GuidedOperationTakenOver,
    GuidedStartStateConverged,
    GuidedStartStateSeeded,
)
from elspeth.web.sessions.routes._helpers import _persist_tool_invocations
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@pytest.fixture
def engine():
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    return engine


@pytest.fixture
def service(engine):
    return SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))


@pytest.fixture
def file_engine(tmp_path: Path):
    engine = create_session_engine(f"sqlite:///{tmp_path / 'guided-start.db'}")
    initialize_session_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(params=("sqlite", "postgres"))
def durable_engine(request: pytest.FixtureRequest, tmp_path: Path):
    """Production-shaped SQLite plus opt-in PostgreSQL real-lock races."""

    if request.param == "postgres":
        url = os.environ.get("ELSPETH_TEST_POSTGRES_URL")
        if url is None:
            pytest.skip("ELSPETH_TEST_POSTGRES_URL is required for the PostgreSQL accept/revert race")
        race_engine = create_session_engine(url)
    else:
        race_engine = create_session_engine(f"sqlite:///{tmp_path / 'revert-races.db'}")
    initialize_session_schema(race_engine)
    try:
        yield race_engine
    finally:
        race_engine.dispose()


def _service_for(engine: Any) -> SessionServiceImpl:
    return SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test.revert-race"))


async def _service_lock_contention(
    first_service: SessionServiceImpl,
    second_service: SessionServiceImpl,
    session_id: UUID,
    first: Callable[[], Coroutine[Any, Any, Any]],
    second: Callable[[], Coroutine[Any, Any, Any]],
) -> tuple[Any, Any]:
    """Pause the winner while it holds the production session lock."""

    original_first_lock = first_service._session_write_lock
    original_second_begin = second_service._session_process_locked_begin
    original_second_lock = second_service._session_write_lock
    held = threading.Barrier(2)
    release = threading.Barrier(2)
    contender_waiting = threading.Event()
    contender_acquired = threading.Event()
    paused = False

    @contextlib.contextmanager
    def controlled_first_lock(conn: Any, locked_session_id: str):
        nonlocal paused
        with original_first_lock(conn, locked_session_id):
            if locked_session_id == str(session_id) and not paused:
                paused = True
                held.wait(timeout=5)
                release.wait(timeout=5)
            yield

    @contextlib.contextmanager
    def observed_second_begin(locked_session_id: str):
        if locked_session_id == str(session_id):
            contender_waiting.set()
        with original_second_begin(locked_session_id) as conn:
            yield conn

    @contextlib.contextmanager
    def observed_second_lock(conn: Any, locked_session_id: str):
        with original_second_lock(conn, locked_session_id):
            if locked_session_id == str(session_id):
                contender_acquired.set()
            yield

    with (
        patch.object(first_service, "_session_write_lock", new=controlled_first_lock),
        patch.object(second_service, "_session_process_locked_begin", new=observed_second_begin),
        patch.object(second_service, "_session_write_lock", new=observed_second_lock),
    ):
        first_task: asyncio.Task[Any] = asyncio.create_task(first())
        await asyncio.to_thread(held.wait, 5)
        second_task: asyncio.Task[Any] = asyncio.create_task(second())
        assert await asyncio.to_thread(contender_waiting.wait, 5)
        was_blocked = not contender_acquired.is_set()
        await asyncio.to_thread(release.wait, 5)
        results = tuple(await asyncio.gather(first_task, second_task, return_exceptions=True))
        assert was_blocked
        return results  # type: ignore[return-value]


async def _claim(
    service: SessionServiceImpl,
    session_id: UUID,
    operation_id="00000000-0000-4000-8000-000000000001",
    *,
    kind="state_revert",
    request_hash="a" * 64,
) -> GuidedOperationFence:
    outcome = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind=kind,
        request_hash=request_hash,
        actor="route",
        lease_seconds=60,
    )
    assert isinstance(outcome, GuidedOperationClaimed)
    return outcome.fence


async def _attach_pending_guided_pipeline_proposal(
    service: SessionServiceImpl,
    *,
    session_id: UUID,
    state: CompositionStateRecord,
    guided: GuidedSession,
    proposal_base_state: CompositionStateRecord | None = None,
    surface: PlannerSurface = PlannerSurface.GUIDED_STAGED,
):
    base_state = proposal_base_state or state
    reviewed_facts = guided_private_reviewed_facts(guided)
    proposal = PipelineProposal.create(
        pipeline={"sources": {}, "nodes": [], "edges": [], "outputs": []},
        base=PresentBase(
            state_id=base_state.id,
            composition_content_hash=composition_content_hash(state_from_record(base_state)),
        ),
        reviewed_facts=reviewed_facts,
        surface=surface,
        repair_count=0,
        skill_hash=stable_hash("guided-revert-test-skill"),
        covered_deferred_intent_ids=(),
        supersedes_draft_hash=None,
    )
    plan = PipelinePlanResult(
        proposal=proposal,
        tool_call_id=f"guided-revert-{uuid4()}",
        custody_result="not_required",
        model_identifier="test-model",
        model_version="test-model-v1",
        provider="test",
    )
    row = await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=plan,
        summary="Stage the guided topology.",
        rationale="Exercise revert integrity.",
        affects=("graph",),
        arguments_redacted_json=redact_tool_call_arguments(
            "set_pipeline",
            deep_thaw(proposal.pipeline),
            telemetry=NoopRedactionTelemetry(),
        ),
        actor="composer-web:user:alice",
        composer_model_identifier="test-model",
        composer_model_version="test-model-v1",
        composer_provider="test",
    )
    active = replace(
        guided,
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(
            *guided.history,
            TurnRecord(
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.PROPOSE_PIPELINE,
                payload_hash="d" * 64,
                response_hash=None,
                emitter="server",
            ),
        ),
        advisor_checkpoint_passes_used=2,
        advisor_signoff_escape_offered=True,
        transition_consumed=True,
        active_proposal=GuidedProposalRef(
            proposal_id=row.id,
            draft_hash=proposal.draft_hash,
            base=proposal.base,
            reviewed_anchor_hash=proposal.reviewed_anchor_hash,
            covered_deferred_intent_ids=proposal.covered_deferred_intent_ids,
            creation_event_schema="pipeline_proposal_created.v1",
        ),
        active_edit_target=ComponentTarget(kind="node", stable_id=str(uuid4())),
    )
    with service._engine.begin() as conn:
        conn.execute(
            update(composition_states_table)
            .where(composition_states_table.c.id == str(state.id))
            .values(
                composer_meta={
                    "_version": 1,
                    "data": {
                        "guided_session": active.to_dict(),
                        "guided_operation_replay": {"stale": "must be removed"},
                    },
                }
            )
        )
    refreshed = await service.get_state_in_session(state.id, session_id)
    assert refreshed is not None
    return row, refreshed


async def _assert_revert_integrity_failure_is_atomic(
    service: SessionServiceImpl,
    engine: Any,
    *,
    session_id: UUID,
    target_state_id: UUID,
    fence: GuidedOperationFence,
    proposal_id: UUID | None,
    match: str,
    expected_proposal_status: str = "pending",
    expected_rejection_count: int = 0,
) -> None:
    with engine.begin() as conn:
        state_count = conn.execute(
            select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session_id))
        ).scalar_one()
        message_count = conn.execute(
            select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session_id))
        ).scalar_one()

    with pytest.raises(AuditIntegrityError, match=match):
        await service.revert_state_for_guided_operation(
            fence,
            state_id=target_state_id,
            actor="route",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        )

    with engine.begin() as conn:
        assert (
            conn.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session_id))
            ).scalar_one()
            == state_count
        )
        assert (
            conn.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session_id))
            ).scalar_one()
            == message_count
        )
        operation = conn.execute(
            select(guided_operations_table.c.status, guided_operations_table.c.result_state_id)
            .where(guided_operations_table.c.session_id == str(session_id))
            .where(guided_operations_table.c.operation_id == fence.operation_id)
        ).one()
        if proposal_id is not None:
            proposal = conn.execute(
                select(composition_proposals_table.c.status).where(composition_proposals_table.c.id == str(proposal_id))
            ).one()
            terminal_count = conn.execute(
                select(func.count())
                .select_from(proposal_events_table)
                .where(proposal_events_table.c.proposal_id == str(proposal_id))
                .where(proposal_events_table.c.event_type == "proposal.rejected")
            ).scalar_one()
            assert proposal.status == expected_proposal_status
            assert terminal_count == expected_rejection_count
    assert operation.status == "in_progress"
    assert operation.result_state_id is None


def _race_state_content_hash(state: CompositionStateData) -> str:
    return stable_hash(
        {
            "sources": state.sources,
            "nodes": state.nodes,
            "edges": state.edges,
            "outputs": state.outputs,
            "metadata": state.metadata_,
        }
    )


async def _prepare_accept_revert_race(
    accept_service: SessionServiceImpl,
    revert_service: SessionServiceImpl,
):
    session = await accept_service.create_session("alice", "Accept/revert race", "local")
    target = await accept_service.save_composition_state(
        session.id,
        CompositionStateData(
            sources={"target": {"type": "csv"}},
            metadata_={"name": "Revert target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    accepted_state = CompositionStateData(
        sources={},
        nodes=(),
        edges=(),
        outputs=(),
        metadata_={"name": "Accepted proposal", "description": ""},
        is_valid=True,
    )
    current = await accept_service.save_composition_state(
        session.id,
        accepted_state,
        provenance="session_seed",
    )
    pipeline: dict[str, object] = {"sources": {}, "nodes": [], "edges": [], "outputs": []}
    proposal = PipelineProposal.create(
        pipeline=pipeline,
        base=PresentBase(
            state_id=current.id,
            composition_content_hash=composition_content_hash(state_from_record(current)),
        ),
        reviewed_facts={},
        surface=PlannerSurface.FREEFORM,
        repair_count=0,
        skill_hash=stable_hash("accept-revert-race-skill"),
        covered_deferred_intent_ids=(),
        supersedes_draft_hash=None,
    )
    plan = PipelinePlanResult(
        proposal=proposal,
        tool_call_id=f"accept-revert-{uuid4()}",
        custody_result="not_required",
        model_identifier="test-model",
        model_version="test-model-v1",
        provider="test",
    )
    proposal_row = await accept_service.create_pipeline_composition_proposal(
        session_id=session.id,
        plan=plan,
        summary="Accept or revert.",
        rationale="Exercise the shared lock.",
        affects=("graph",),
        arguments_redacted_json=redact_tool_call_arguments(
            "set_pipeline",
            pipeline,
            telemetry=NoopRedactionTelemetry(),
        ),
        actor="composer-web:user:alice",
        composer_model_identifier="test-model",
        composer_model_version="test-model-v1",
        composer_provider="test",
    )
    content_hash = _race_state_content_hash(accepted_state)
    invocation = finish_success(
        begin_dispatch(
            plan.tool_call_id,
            "set_pipeline",
            pipeline,
            version_before=current.version,
            actor="user:alice",
        ),
        result_payload={
            "success": True,
            "content_hash": content_hash,
            "pipeline_content_hash_schema": "composer.pipeline-dispatch-result.v1",
            "pipeline_content_hash": content_hash,
        },
        version_after=current.version + 1,
    )
    bindings = await _persist_tool_invocations(
        accept_service,
        session.id,
        (invocation,),
        None,
        plugin_crash_pending=False,
    )
    assert len(bindings) == 1
    dispatch: PipelineDispatchAuditBinding = bindings[0]
    fence = await _claim(
        revert_service,
        session.id,
        operation_id=str(uuid4()),
        request_hash="f" * 64,
    )

    async def accept():
        return await accept_service.settle_pipeline_composition_proposal(
            session_id=session.id,
            proposal_id=proposal_row.id,
            draft_hash=proposal.draft_hash,
            reviewed_facts={},
            state=accepted_state,
            candidate_content_hash=content_hash,
            executor_content_hash=content_hash,
            final_composer_metadata=None,
            dispatch=dispatch,
            actor="user:alice",
        )

    async def revert():
        return await revert_service.revert_state_for_guided_operation(
            fence,
            state_id=target.id,
            actor="composer_route",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id), "version": state.version}),
        )

    return session, target, proposal_row, fence, accept, revert


@pytest.mark.asyncio
async def test_revert_state_and_system_message_settle_in_fence_transaction(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    first = await service.save_composition_state(
        session.id,
        CompositionStateData(sources={"source": {"type": "csv"}}, is_valid=True),
        provenance="session_seed",
    )
    await service.save_composition_state(
        session.id,
        CompositionStateData(sources={"source": {"type": "api"}}, is_valid=True),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    reverted = await service.revert_state_for_guided_operation(
        fence,
        state_id=first.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id), "version": state.version}),
    )

    assert reverted.version == 3
    messages = await service.get_messages(session.id, limit=None)
    assert messages[-1].content == "Pipeline reverted to version 1."
    outcome = await service.get_guided_operation(
        session_id=session.id,
        operation_id=fence.operation_id,
        kind="state_revert",
        request_hash="a" * 64,
    )
    assert outcome == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=reverted.id),
        response_hash=stable_hash({"state_id": str(reverted.id), "version": 3}),
    )


@pytest.mark.asyncio
async def test_revert_older_guided_proposal_checkpoint_rejects_pending_and_scrubs_to_topology(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Guided target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": None},
            metadata_={"name": "Current freeform", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    reverted = await service.revert_state_for_guided_operation(
        fence,
        state_id=target.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id), "version": state.version}),
    )

    with engine.begin() as conn:
        proposal_row = conn.execute(select(composition_proposals_table).where(composition_proposals_table.c.id == str(proposal.id))).one()
        terminal_events = conn.execute(
            select(proposal_events_table)
            .where(proposal_events_table.c.proposal_id == str(proposal.id))
            .where(proposal_events_table.c.event_type == "proposal.rejected")
        ).fetchall()
    assert proposal_row.status == "rejected"
    assert len(terminal_events) == 1
    terminal_payload = deep_thaw(terminal_events[0].payload)
    assert terminal_payload["reason_code"] == "superseded"
    assert terminal_payload["outcome"] == "superseded"

    assert reverted.composer_meta is not None
    assert "guided_operation_replay" not in reverted.composer_meta
    restored = state_from_record(reverted).guided_session
    assert restored is not None
    assert restored.step is GuidedStep.STEP_3_TRANSFORMS
    assert restored.history == ()
    assert restored.advisor_checkpoint_passes_used == 0
    assert restored.advisor_signoff_escape_offered is False
    assert restored.terminal is None
    assert restored.transition_consumed is False
    assert restored.active_proposal is None
    assert restored.active_edit_target is None


@pytest.mark.asyncio
async def test_revert_terminal_target_reference_rolls_back_every_surface(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    guided = GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS)
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": guided.to_dict()},
            metadata_={"name": "Guided target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=guided,
    )
    bound = state_from_record(target).guided_session
    assert bound is not None and bound.active_proposal is not None
    await service.reject_pipeline_composition_proposal(
        session_id=session.id,
        proposal_id=proposal.id,
        draft_hash=bound.active_proposal.draft_hash,
        reviewed_facts=guided_private_reviewed_facts(bound),
        reason="superseded",
        dispatch=None,
        actor="test",
    )
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": None},
            metadata_={"name": "Current freeform", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    await _assert_revert_integrity_failure_is_atomic(
        service,
        engine,
        session_id=session.id,
        target_state_id=target.id,
        fence=fence,
        proposal_id=proposal.id,
        match="terminal",
        expected_proposal_status="rejected",
        expected_rejection_count=1,
    )


@pytest.mark.asyncio
async def test_revert_freeform_target_rejects_current_guided_pending_proposal(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={
                "guided_session": None,
                "guided_operation_replay": {"stale": "must be removed from freeform too"},
            },
            metadata_={"name": "Freeform target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    current = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Guided current", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, _ = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=current,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    fence = await _claim(service, session.id)

    reverted = await service.revert_state_for_guided_operation(
        fence,
        state_id=target.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    with engine.begin() as conn:
        status = conn.execute(
            select(composition_proposals_table.c.status).where(composition_proposals_table.c.id == str(proposal.id))
        ).scalar_one()
    assert status == "rejected"
    assert deep_thaw(reverted.composer_meta) == {"guided_session": None}
    assert state_from_record(reverted).guided_session is None


@pytest.mark.asyncio
async def test_revert_early_guided_checkpoint_preserves_stage_and_unanswered_turn_while_scrubbing_replay(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    guided = GuidedSession(
        step=GuidedStep.STEP_1_SOURCE,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_1_SOURCE,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash="b" * 64,
                response_hash=None,
                emitter="server",
            ),
        ),
    )
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={
                "guided_session": guided.to_dict(),
                "guided_operation_replay": {"stale": "must be removed"},
            },
            metadata_={"name": "Early guided", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": None},
            metadata_={"name": "Current", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    reverted = await service.revert_state_for_guided_operation(
        fence,
        state_id=target.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    assert reverted.composer_meta is not None
    assert "guided_operation_replay" not in reverted.composer_meta
    restored = state_from_record(reverted).guided_session
    assert restored is not None
    assert restored.step is GuidedStep.STEP_1_SOURCE
    assert restored.history == guided.history


@pytest.mark.asyncio
async def test_revert_completed_guided_checkpoint_preserves_terminal_step_and_state(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    guided = GuidedSession(
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash="c" * 64,
                response_hash="d" * 64,
                emitter="server",
            ),
        ),
        advisor_checkpoint_passes_used=2,
        advisor_signoff_escape_offered=True,
        terminal=TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="pipeline: {}"),
        transition_consumed=True,
    )
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={
                "guided_session": guided.to_dict(),
                "guided_operation_replay": {"stale": "must be removed"},
            },
            metadata_={"name": "Completed guided", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": None},
            metadata_={"name": "Current", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    reverted = await service.revert_state_for_guided_operation(
        fence,
        state_id=target.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    assert reverted.composer_meta is not None
    assert "guided_operation_replay" not in reverted.composer_meta
    restored = state_from_record(reverted).guided_session
    assert restored == guided


@pytest.mark.asyncio
async def test_revert_step3_non_proposal_unanswered_turn_remains_valid(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    guided = GuidedSession(
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash="e" * 64,
                response_hash=None,
                emitter="server",
            ),
        ),
    )
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": guided.to_dict()},
            metadata_={"name": "Guided", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    reverted = await service.revert_state_for_guided_operation(
        fence,
        state_id=target.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    restored = state_from_record(reverted).guided_session
    assert restored is not None
    assert restored.step is GuidedStep.STEP_3_TRANSFORMS
    assert restored.history == ()


@pytest.mark.asyncio
async def test_revert_rejects_distinct_current_and_target_pending_refs_once_each(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Guided target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    target_proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    current = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Guided current", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    current_proposal, _ = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=current,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    fence = await _claim(service, session.id)

    await service.revert_state_for_guided_operation(
        fence,
        state_id=target.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    proposal_ids = {str(target_proposal.id), str(current_proposal.id)}
    with engine.begin() as conn:
        statuses = conn.execute(
            select(composition_proposals_table.c.id, composition_proposals_table.c.status).where(
                composition_proposals_table.c.id.in_(proposal_ids)
            )
        ).fetchall()
        terminal_counts = conn.execute(
            select(proposal_events_table.c.proposal_id, func.count())
            .where(proposal_events_table.c.proposal_id.in_(proposal_ids))
            .where(proposal_events_table.c.event_type == "proposal.rejected")
            .group_by(proposal_events_table.c.proposal_id)
        ).fetchall()
    assert {row.id: row.status for row in statuses} == dict.fromkeys(proposal_ids, "rejected")
    assert {row.proposal_id: row[1] for row in terminal_counts} == dict.fromkeys(proposal_ids, 1)


@pytest.mark.asyncio
async def test_revert_reference_mismatch_rolls_back_proposals_state_message_and_operation(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Guided target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    tampered_meta = deep_thaw(target.composer_meta)
    tampered_meta["guided_session"]["active_proposal"]["draft_hash"] = "e" * 64
    with engine.begin() as conn:
        conn.execute(
            update(composition_states_table)
            .where(composition_states_table.c.id == str(target.id))
            .values(composer_meta={"_version": 1, "data": tampered_meta})
        )
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": None},
            metadata_={"name": "Current", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    with pytest.raises(AuditIntegrityError, match="reference differs"):
        await service.revert_state_for_guided_operation(
            fence,
            state_id=target.id,
            actor="route",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        )

    with engine.begin() as conn:
        assert (
            conn.execute(
                select(composition_proposals_table.c.status).where(composition_proposals_table.c.id == str(proposal.id))
            ).scalar_one()
            == "pending"
        )
        assert (
            conn.execute(
                select(func.count())
                .select_from(proposal_events_table)
                .where(proposal_events_table.c.proposal_id == str(proposal.id))
                .where(proposal_events_table.c.event_type == "proposal.rejected")
            ).scalar_one()
            == 0
        )
        assert (
            conn.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session.id))
            ).scalar_one()
            == 2
        )
        assert (
            conn.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session.id))
            ).scalar_one()
            == 0
        )
        operation_status = conn.execute(
            select(guided_operations_table.c.status)
            .where(guided_operations_table.c.session_id == str(session.id))
            .where(guided_operations_table.c.operation_id == fence.operation_id)
        ).scalar_one()
    assert operation_status == "in_progress"


@pytest.mark.asyncio
async def test_revert_guided_ref_wrong_checkpoint_base_rolls_back_every_surface(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    current = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Current", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
        proposal_base_state=current,
    )
    fence = await _claim(service, session.id)

    await _assert_revert_integrity_failure_is_atomic(
        service,
        engine,
        session_id=session.id,
        target_state_id=target.id,
        fence=fence,
        proposal_id=proposal.id,
        match="checkpoint base",
    )


@pytest.mark.asyncio
async def test_revert_guided_ref_wrong_checkpoint_content_rolls_back_every_surface(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Original", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    with engine.begin() as conn:
        conn.execute(
            update(composition_states_table)
            .where(composition_states_table.c.id == str(target.id))
            .values(metadata_={"_version": 1, "data": {"name": "Tampered", "description": ""}})
        )
    fence = await _claim(service, session.id)

    await _assert_revert_integrity_failure_is_atomic(
        service,
        engine,
        session_id=session.id,
        target_state_id=target.id,
        fence=fence,
        proposal_id=proposal.id,
        match="checkpoint content",
    )


@pytest.mark.asyncio
async def test_revert_guided_ref_wrong_surface_rolls_back_every_surface(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    guided = GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS)
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": guided.to_dict()},
            metadata_={"name": "Guided", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=guided,
    )
    active_guided = state_from_record(target).guided_session
    assert active_guided is not None and active_guided.active_proposal is not None
    active = active_guided.active_proposal
    wrong_surface = PipelineProposal.create(
        pipeline={"sources": {}, "nodes": [], "edges": [], "outputs": []},
        base=active.base,
        reviewed_facts=guided_private_reviewed_facts(active_guided),
        surface=PlannerSurface.FREEFORM,
        repair_count=0,
        skill_hash=stable_hash("guided-revert-test-skill"),
        covered_deferred_intent_ids=active.covered_deferred_intent_ids,
        supersedes_draft_hash=active.supersedes_draft_hash,
    )
    tampered_meta = deep_thaw(target.composer_meta)
    tampered_meta["guided_session"]["active_proposal"]["draft_hash"] = wrong_surface.draft_hash
    with engine.begin() as conn:
        creation_payload = deep_thaw(
            conn.execute(
                select(proposal_events_table.c.payload).where(proposal_events_table.c.id == str(proposal.audit_event_id))
            ).scalar_one()
        )
        creation_payload["surface"] = PlannerSurface.FREEFORM.value
        creation_payload["draft_hash"] = wrong_surface.draft_hash
        conn.execute(
            update(proposal_events_table).where(proposal_events_table.c.id == str(proposal.audit_event_id)).values(payload=creation_payload)
        )
        conn.execute(
            update(composition_states_table)
            .where(composition_states_table.c.id == str(target.id))
            .values(composer_meta={"_version": 1, "data": tampered_meta})
        )
    fence = await _claim(service, session.id)

    await _assert_revert_integrity_failure_is_atomic(
        service,
        engine,
        session_id=session.id,
        target_state_id=target.id,
        fence=fence,
        proposal_id=proposal.id,
        match="surface",
    )


@pytest.mark.asyncio
async def test_revert_guided_ref_without_trailing_proposal_turn_rolls_back_every_surface(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Guided", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    tampered_meta = deep_thaw(target.composer_meta)
    tampered_meta["guided_session"]["history"] = []
    with engine.begin() as conn:
        conn.execute(
            update(composition_states_table)
            .where(composition_states_table.c.id == str(target.id))
            .values(composer_meta={"_version": 1, "data": tampered_meta})
        )
    fence = await _claim(service, session.id)

    await _assert_revert_integrity_failure_is_atomic(
        service,
        engine,
        session_id=session.id,
        target_state_id=target.id,
        fence=fence,
        proposal_id=proposal.id,
        match="schema-9 authority is malformed",
    )


@pytest.mark.asyncio
async def test_revert_guided_ref_with_multiple_unanswered_turns_rolls_back_every_surface(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    guided = GuidedSession(
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash="b" * 64,
                response_hash="c" * 64,
                emitter="server",
            ),
        ),
    )
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": guided.to_dict()},
            metadata_={"name": "Guided", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=guided,
    )
    tampered_meta = deep_thaw(target.composer_meta)
    tampered_meta["guided_session"]["history"][0]["response_hash"] = None
    with engine.begin() as conn:
        conn.execute(
            update(composition_states_table)
            .where(composition_states_table.c.id == str(target.id))
            .values(composer_meta={"_version": 1, "data": tampered_meta})
        )
    fence = await _claim(service, session.id)

    await _assert_revert_integrity_failure_is_atomic(
        service,
        engine,
        session_id=session.id,
        target_state_id=target.id,
        fence=fence,
        proposal_id=proposal.id,
        match="schema-9 authority is malformed",
    )


@pytest.mark.asyncio
async def test_revert_trailing_proposal_turn_without_ref_rolls_back_every_surface(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    guided = GuidedSession(
        step=GuidedStep.STEP_3_TRANSFORMS,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_3_TRANSFORMS,
                turn_type=TurnType.PROPOSE_PIPELINE,
                payload_hash="a" * 64,
                response_hash=None,
                emitter="server",
            ),
        ),
    )
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": guided.to_dict()},
            metadata_={"name": "Guided", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    await _assert_revert_integrity_failure_is_atomic(
        service,
        engine,
        session_id=session.id,
        target_state_id=target.id,
        fence=fence,
        proposal_id=None,
        match="history coupling",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault_point",
    ("proposal_event", "proposal_update", "state_insert", "message_insert", "operation_complete"),
)
async def test_revert_fault_rolls_back_every_settlement_surface(service, engine, fault_point: str) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    target = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS).to_dict()},
            metadata_={"name": "Guided target", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    proposal, target = await _attach_pending_guided_pipeline_proposal(
        service,
        session_id=session.id,
        state=target,
        guided=GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS),
    )
    await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": None},
            metadata_={"name": "Current", "description": ""},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    def inject(_conn, _cursor, statement, _parameters, _context, _executemany):
        normalized = " ".join(statement.lower().split())
        should_fail = (
            (fault_point == "proposal_event" and normalized.startswith("insert into proposal_events"))
            or (fault_point == "proposal_update" and normalized.startswith("update composition_proposals"))
            or (fault_point == "state_insert" and normalized.startswith("insert into composition_states"))
            or (fault_point == "message_insert" and normalized.startswith("insert into chat_messages"))
            or (fault_point == "operation_complete" and normalized.startswith("update guided_operations") and "status" in normalized)
        )
        if should_fail:
            raise RuntimeError(f"injected {fault_point}")

    event.listen(engine, "before_cursor_execute", inject)
    try:
        with pytest.raises(RuntimeError, match=fault_point):
            await service.revert_state_for_guided_operation(
                fence,
                state_id=target.id,
                actor="route",
                response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
            )
    finally:
        event.remove(engine, "before_cursor_execute", inject)

    with engine.begin() as conn:
        assert (
            conn.execute(
                select(composition_proposals_table.c.status).where(composition_proposals_table.c.id == str(proposal.id))
            ).scalar_one()
            == "pending"
        )
        assert (
            conn.execute(
                select(func.count())
                .select_from(proposal_events_table)
                .where(proposal_events_table.c.proposal_id == str(proposal.id))
                .where(proposal_events_table.c.event_type == "proposal.rejected")
            ).scalar_one()
            == 0
        )
        assert (
            conn.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session.id))
            ).scalar_one()
            == 2
        )
        assert (
            conn.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session.id))
            ).scalar_one()
            == 0
        )
        operation = conn.execute(
            select(guided_operations_table.c.status, guided_operations_table.c.result_state_id)
            .where(guided_operations_table.c.session_id == str(session.id))
            .where(guided_operations_table.c.operation_id == fence.operation_id)
        ).one()
    assert operation.status == "in_progress"
    assert operation.result_state_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ("accept", "revert"))
async def test_accept_vs_revert_serializes_on_real_session_lock(durable_engine, winner: str) -> None:
    accept_service = _service_for(durable_engine)
    revert_service = _service_for(durable_engine)
    session, target, proposal, fence, accept, revert = await _prepare_accept_revert_race(accept_service, revert_service)
    try:
        if winner == "accept":
            accept_result, revert_result = await _service_lock_contention(
                accept_service,
                revert_service,
                session.id,
                accept,
                revert,
            )
            assert not isinstance(accept_result, BaseException)
            assert not isinstance(revert_result, BaseException)
            expected_status = "committed"
            expected_versions = [1, 2, 3, 4]
        else:
            revert_result, accept_result = await _service_lock_contention(
                revert_service,
                accept_service,
                session.id,
                revert,
                accept,
            )
            assert not isinstance(revert_result, BaseException)
            assert isinstance(accept_result, ValueError)
            expected_status = "rejected"
            expected_versions = [1, 2, 3]

        with durable_engine.begin() as conn:
            proposal_row = conn.execute(
                select(composition_proposals_table).where(composition_proposals_table.c.id == str(proposal.id))
            ).one()
            terminal_events = conn.execute(
                select(proposal_events_table.c.event_type)
                .where(proposal_events_table.c.proposal_id == str(proposal.id))
                .where(proposal_events_table.c.event_type.in_(("proposal.accepted", "proposal.rejected")))
            ).fetchall()
        assert proposal_row.status == expected_status
        assert [event.event_type for event in terminal_events] == ["proposal.accepted" if winner == "accept" else "proposal.rejected"]
        versions = await accept_service.get_state_versions(session.id)
        assert [state.version for state in versions] == expected_versions
        current = versions[-1]
        assert current.derived_from_state_id == target.id
        assert current.sources == target.sources
        if winner == "accept":
            assert proposal_row.committed_state_id == str(versions[-2].id)
        else:
            assert proposal_row.committed_state_id is None
        messages = await accept_service.get_messages(session.id, limit=None)
        assert [message.content for message in messages if message.role == "system"] == ["Pipeline reverted to version 1."]
        operation = await accept_service.get_guided_operation(
            session_id=session.id,
            operation_id=fence.operation_id,
            kind="state_revert",
            request_hash="f" * 64,
        )
        assert isinstance(operation, GuidedOperationCompleted)
    finally:
        with durable_engine.begin() as conn:
            conn.execute(delete(sessions_table).where(sessions_table.c.id == str(session.id)))


@pytest.mark.asyncio
async def test_stale_revert_fence_writes_nothing(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    first = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
    stale_fence = await _claim(service, session.id)
    with engine.begin() as conn:
        conn.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == str(session.id))
            .where(guided_operations_table.c.operation_id == stale_fence.operation_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    takeover = await service.reserve_guided_operation(
        session_id=session.id,
        operation_id=stale_fence.operation_id,
        kind="state_revert",
        request_hash="a" * 64,
        actor="route-b",
        lease_seconds=60,
    )
    assert isinstance(takeover, GuidedOperationTakenOver)

    with pytest.raises(GuidedOperationFenceLostError):
        await service.revert_state_for_guided_operation(
            stale_fence,
            state_id=first.id,
            actor="route-a",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        )

    with engine.connect() as conn:
        assert (
            conn.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session.id))
            ).scalar_one()
            == 1
        )
        assert (
            conn.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session.id))
            ).scalar_one()
            == 0
        )


@pytest.mark.asyncio
async def test_guided_state_save_system_message_and_settlement_are_atomic(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    fence = await _claim(service, session.id, kind="guided_convert")
    state_data = CompositionStateData(composer_meta={"guided_session": {"schema_version": 9}}, is_valid=True)

    saved = await service.save_state_for_guided_operation(
        fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        state=state_data,
        provenance="session_seed",
        actor="route",
        system_message="Switched to guided mode.",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    current = await service.get_current_state(session.id)
    assert current is not None and current.id == saved.id
    assert [message.content for message in await service.get_messages(session.id, limit=None)] == ["Switched to guided mode."]
    outcome = await service.get_guided_operation(
        session_id=session.id,
        operation_id=fence.operation_id,
        kind="guided_convert",
        request_hash="a" * 64,
    )
    assert outcome == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=saved.id),
        response_hash=stable_hash({"state_id": str(saved.id)}),
    )


@pytest.mark.asyncio
async def test_existing_guided_state_can_settle_without_new_state_version(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    existing = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
    fence = await _claim(service, session.id, kind="guided_start")

    settled = await service.complete_existing_state_guided_operation(
        fence,
        state_id=existing.id,
        expected_current_state_id=existing.id,
        expected_current_state_version=existing.version,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    assert settled == existing
    assert [state.id for state in await service.get_state_versions(session.id)] == [existing.id]


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_persists_and_settles_empty_session(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    fence = await _claim(service, session.id, kind="guided_start")
    state_data = CompositionStateData(
        composer_meta={"guided_session": {"schema_version": 9}},
        is_valid=True,
    )

    outcome = await service.seed_or_complete_guided_start_operation(
        fence,
        state=state_data,
        provenance="session_seed",
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    assert isinstance(outcome, GuidedStartStateSeeded)
    assert await service.get_current_state(session.id) == outcome.state
    operation = await service.get_guided_operation(
        session_id=session.id,
        operation_id=fence.operation_id,
        kind="guided_start",
        request_hash="a" * 64,
    )
    assert operation == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=outcome.state.id),
        response_hash=stable_hash({"state_id": str(outcome.state.id)}),
    )


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_converges_exact_existing_guided_head(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    existing = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": {"schema_version": 9}},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id, kind="guided_start")

    def guided_response_hash(state):
        assert state.composer_meta is not None
        assert "guided_session" in state.composer_meta
        return stable_hash({"state_id": str(state.id)})

    outcome = await service.seed_or_complete_guided_start_operation(
        fence,
        state=CompositionStateData(is_valid=True),
        provenance="session_seed",
        actor="route",
        response_hash_factory=guided_response_hash,
    )

    assert outcome == GuidedStartStateConverged(state=existing)
    assert [state.id for state in await service.get_state_versions(session.id)] == [existing.id]


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_does_not_treat_generic_integrity_error_as_convergence(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    freeform = await service.save_composition_state(
        session.id,
        CompositionStateData(is_valid=True),
        provenance="post_compose",
    )
    fence = await _claim(service, session.id, kind="guided_start")

    def reject_freeform(_state):
        raise AuditIntegrityError("current head is not a valid guided checkpoint")

    with pytest.raises(AuditIntegrityError, match="not a valid guided checkpoint"):
        await service.seed_or_complete_guided_start_operation(
            fence,
            state=CompositionStateData(
                composer_meta={"guided_session": {"schema_version": 9}},
                is_valid=True,
            ),
            provenance="session_seed",
            actor="route",
            response_hash_factory=reject_freeform,
        )

    assert [state.id for state in await service.get_state_versions(session.id)] == [freeform.id]


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_serializes_two_sqlite_services(file_engine) -> None:
    service_a = SessionServiceImpl(file_engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test.a"))
    service_b = SessionServiceImpl(file_engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test.b"))
    session = await service_a.create_session("alice", "Pipeline", "local")
    fence_a = await _claim(
        service_a,
        session.id,
        operation_id="00000000-0000-4000-8000-000000000011",
        kind="guided_start",
        request_hash="b" * 64,
    )
    fence_b = await _claim(
        service_b,
        session.id,
        operation_id="00000000-0000-4000-8000-000000000012",
        kind="guided_start",
        request_hash="c" * 64,
    )
    state_data = CompositionStateData(
        composer_meta={"guided_session": {"schema_version": 9}},
        is_valid=True,
    )

    outcomes = await asyncio.gather(
        service_a.seed_or_complete_guided_start_operation(
            fence_a,
            state=state_data,
            provenance="session_seed",
            actor="route-a",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        ),
        service_b.seed_or_complete_guided_start_operation(
            fence_b,
            state=state_data,
            provenance="session_seed",
            actor="route-b",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        ),
    )

    assert {type(outcome) for outcome in outcomes} == {
        GuidedStartStateSeeded,
        GuidedStartStateConverged,
    }
    assert outcomes[0].state.id == outcomes[1].state.id
    assert len(await service_a.get_state_versions(session.id)) == 1
