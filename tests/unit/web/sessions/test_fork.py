"""Tests for session fork — service-level fork_session and route-level fork endpoint."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import fork_blob_id
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    blobs_table,
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    guided_operations_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    GuidedForkSettlementCommand,
    GuidedOperationClaimed,
    GuidedOperationTakenOver,
    InvalidForkTargetError,
)
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.routes.guided_operations import guided_response_hash
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.schemas import ForkSessionResponse
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

_FORK_SOURCE_ID = "11111111-1111-4111-8111-111111111111"
_FORK_OUTPUT_ID = "22222222-2222-4222-8222-222222222222"
_FORK_INTENT_ID = "33333333-3333-4333-8333-333333333333"
_FORK_HASH_A = "a" * 64
_FORK_HASH_B = "b" * 64


def _guided_fork_checkpoint(
    *,
    step: str,
    root_message_id: uuid.UUID,
    deferred_message_id: uuid.UUID,
    deferred_message_content: str,
    current_turn: str,
) -> Any:
    """Build a schema-9 checkpoint containing the fork-sensitive fields."""
    from elspeth.core.canonical import stable_hash
    from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
    from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
    from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
    from elspeth.web.composer.guided.state_machine import (
        ComponentTarget,
        DeferredStageIntent,
        GuidedProposalRef,
        GuidedSession,
        TurnRecord,
        guided_reviewed_anchor_hash,
    )
    from elspeth.web.composer.pipeline_proposal import AbsentBase

    guided_step = GuidedStep(step)
    turn_type = TurnType(current_turn)
    reviewed_sources = {
        _FORK_SOURCE_ID: SourceResolved(
            name="orders",
            plugin="csv",
            options={"path": "/data/orders.csv"},
            observed_columns=("id", "total"),
            sample_rows=({"id": 1, "total": 3},),
            on_validation_failure="discard",
        )
    }
    reviewed_outputs = {
        _FORK_OUTPUT_ID: SinkOutputResolved(
            name="archive",
            plugin="json",
            options={"path": "/data/archive.jsonl"},
            required_fields=("id", "total"),
            schema_mode="fixed",
            on_write_failure="discard",
        )
    }
    deferred = DeferredStageIntent.create(
        intent_id=_FORK_INTENT_ID,
        receiving_stage="source",
        target_stage="topology",
        catalog_kind="transform",
        catalog_name="rename",
        redacted_summary="Rename total before writing the archive.",
        originating_message_id=str(deferred_message_id),
        message_content_hash=stable_hash(deferred_message_content),
        constraints=(),
    )
    active_proposal = None
    if (guided_step, turn_type) in {
        (GuidedStep.STEP_3_TRANSFORMS, TurnType.PROPOSE_PIPELINE),
        (GuidedStep.STEP_4_WIRE, TurnType.CONFIRM_WIRING),
    }:
        active_proposal = GuidedProposalRef(
            proposal_id=uuid.UUID("44444444-4444-4444-8444-444444444444"),
            draft_hash=_FORK_HASH_A,
            base=AbsentBase(),
            reviewed_anchor_hash=guided_reviewed_anchor_hash(
                source_order=(_FORK_SOURCE_ID,),
                reviewed_sources=reviewed_sources,
                output_order=(_FORK_OUTPUT_ID,),
                reviewed_outputs=reviewed_outputs,
            ),
            covered_deferred_intent_ids=(),
            creation_event_schema="pipeline_proposal_created.v1",
        )
    return GuidedSession(
        step=guided_step,
        history=(
            TurnRecord(
                step=guided_step,
                turn_type=turn_type,
                payload_hash=_FORK_HASH_A,
                response_hash=_FORK_HASH_B,
                emitter="server",
                summary="Answered occurrence stays in fork history.",
            ),
            TurnRecord(
                step=guided_step,
                turn_type=turn_type,
                payload_hash=_FORK_HASH_B,
                response_hash=None,
                emitter="server",
                summary="Unanswered authority must not cross the fork.",
            ),
        ),
        profile=TUTORIAL_PROFILE,
        transition_consumed=True,
        source_order=(_FORK_SOURCE_ID,),
        reviewed_sources=reviewed_sources,
        output_order=(_FORK_OUTPUT_ID,),
        reviewed_outputs=reviewed_outputs,
        deferred_intents=(deferred,),
        active_proposal=active_proposal,
        active_edit_target=ComponentTarget(kind="source", stable_id=_FORK_SOURCE_ID),
        root_intent_message_id=str(root_message_id),
    )


async def _attach_pending_fork_proposal(
    service: SessionServiceImpl,
    *,
    session_id: uuid.UUID,
    state,
    guided,
    proposal_base_state=None,
):
    """Persist one canonical pending proposal and bind it to ``state``."""
    from dataclasses import replace

    from elspeth.contracts.freeze import deep_thaw
    from elspeth.contracts.hashing import stable_hash
    from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
    from elspeth.web.composer.guided.state_machine import GuidedProposalRef
    from elspeth.web.composer.pipeline_planner import PipelinePlanResult
    from elspeth.web.composer.pipeline_proposal import PipelineProposal, PlannerSurface, PresentBase, composition_content_hash
    from elspeth.web.composer.redaction import redact_tool_call_arguments
    from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
    from elspeth.web.sessions.converters import state_from_record

    base_state = proposal_base_state or state
    proposal = PipelineProposal.create(
        pipeline={"sources": {}, "nodes": [], "edges": [], "outputs": []},
        base=PresentBase(
            state_id=base_state.id,
            composition_content_hash=composition_content_hash(state_from_record(base_state)),
        ),
        reviewed_facts=guided_private_reviewed_facts(guided),
        surface=PlannerSurface.TUTORIAL_PROFILE,
        repair_count=0,
        skill_hash=stable_hash("guided-fork-test-skill"),
        covered_deferred_intent_ids=(),
        supersedes_draft_hash=None,
    )
    plan = PipelinePlanResult(
        proposal=proposal,
        tool_call_id=f"guided-fork-{uuid.uuid4()}",
        custody_result="not_required",
        model_identifier="test-model",
        model_version="test-model-v1",
        provider="test",
    )
    row = await service.create_pipeline_composition_proposal(
        session_id=session_id,
        plan=plan,
        summary="Stage fork proposal.",
        rationale="Exercise fork proposal integrity.",
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
        active_proposal=GuidedProposalRef(
            proposal_id=row.id,
            draft_hash=proposal.draft_hash,
            base=proposal.base,
            reviewed_anchor_hash=proposal.reviewed_anchor_hash,
            covered_deferred_intent_ids=proposal.covered_deferred_intent_ids,
            creation_event_schema="pipeline_proposal_created.v1",
        ),
    )
    with service._engine.begin() as conn:
        conn.execute(
            update(composition_states_table)
            .where(composition_states_table.c.id == str(state.id))
            .values(composer_meta={"_version": 1, "data": {"guided_session": active.to_dict()}})
        )
    refreshed = await service.get_state_in_session(state.id, session_id)
    return row, refreshed, active


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


@pytest.fixture
def service(engine):
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )


async def _fork_session(
    service: SessionServiceImpl,
    *,
    source_session_id: uuid.UUID,
    fork_message_id: uuid.UUID,
    new_message_content: str,
    user_id: str,
    auth_provider_type: str,
):
    """Exercise the hard-cut staged service through atomic settlement."""
    parent = await service.get_session(source_session_id)
    assert parent.user_id == user_id
    assert parent.auth_provider_type == auth_provider_type
    reserved = await service.reserve_guided_operation(
        session_id=source_session_id,
        operation_id=str(uuid.uuid4()),
        kind="session_fork",
        request_hash="a" * 64,
        actor="composer_route",
        lease_seconds=300,
    )
    assert type(reserved) in {GuidedOperationClaimed, GuidedOperationTakenOver}
    staged = await service.fork_session(
        reserved.fence,
        fork_message_id=fork_message_id,
        new_message_content=new_message_content,
    )
    active = await service.settle_guided_fork_operation(
        GuidedForkSettlementCommand(
            fence=reserved.fence,
            child_session_id=staged.session.id,
            expected_current_state_id=staged.state.id if staged.state is not None else None,
            edited_message_id=staged.messages[-1].id,
            rewritten_state_id=None,
            rewritten_state=None,
            response_hash="b" * 64,
            actor="composer_route",
        )
    )
    return active, list(staged.messages), staged.state


async def _assert_fork_integrity_failure_is_atomic(
    service: SessionServiceImpl,
    *,
    source_session_id: uuid.UUID,
    fork_message_id: uuid.UUID,
    match: str,
) -> None:
    from elspeth.contracts.errors import AuditIntegrityError

    with service._engine.begin() as conn:
        session_count = conn.execute(select(func.count()).select_from(sessions_table)).scalar_one()
        state_count = conn.execute(select(func.count()).select_from(composition_states_table)).scalar_one()

    with pytest.raises(AuditIntegrityError, match=match):
        await _fork_session(
            service,
            source_session_id=source_session_id,
            fork_message_id=fork_message_id,
            new_message_content="retry",
            user_id="alice",
            auth_provider_type="local",
        )

    with service._engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(sessions_table)).scalar_one() == session_count
        assert conn.execute(select(func.count()).select_from(composition_states_table)).scalar_one() == state_count


async def _canonical_guided_fork_source(service: SessionServiceImpl):
    from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

    session = await service.create_session("alice", "Guided", "local")
    root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
    guided = _guided_fork_checkpoint(
        step=GuidedStep.STEP_3_TRANSFORMS.value,
        root_message_id=root.id,
        deferred_message_id=root.id,
        deferred_message_content=root.content,
        current_turn=TurnType.PROPOSE_PIPELINE.value,
    )
    state = await service.save_composition_state(
        session.id,
        CompositionStateData(
            sources={},
            nodes=[],
            edges=[],
            outputs=[],
            is_valid=True,
            metadata_={"name": "Guided", "description": ""},
            composer_meta={"guided_session": guided.to_dict()},
        ),
        provenance="session_seed",
    )
    proposal, state, guided = await _attach_pending_fork_proposal(
        service,
        session_id=session.id,
        state=state,
        guided=guided,
    )
    fork_message = await service.add_message(
        session.id,
        "user",
        "fork",
        composition_state_id=state.id,
        writer_principal="route_user_message",
    )
    return session, proposal, state, guided, fork_message


class TestForkSession:
    """Tests for SessionServiceImpl.fork_session."""

    @pytest.mark.asyncio
    async def test_fork_creates_new_session_with_provenance(self, service) -> None:
        """Forked session has forked_from fields set."""
        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "Hi there", writer_principal="compose_loop")
        msg2 = await service.add_message(session.id, "user", "Do something", writer_principal="route_user_message")

        new_session, _messages, _state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=msg2.id,
            new_message_content="Do something else",
            user_id="alice",
            auth_provider_type="local",
        )

        assert new_session.forked_from_session_id == session.id
        assert new_session.forked_from_message_id == msg2.id
        assert new_session.user_id == "alice"
        assert "(fork)" in new_session.title

    @pytest.mark.asyncio
    async def test_fork_copies_messages_before_fork_point(self, service) -> None:
        """Only messages before the fork message are copied."""
        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "First", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "Response 1", writer_principal="compose_loop")
        fork_msg = await service.add_message(session.id, "user", "Second", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "Response 2", writer_principal="compose_loop")

        _, messages, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Second (edited)",
            user_id="alice",
            auth_provider_type="local",
        )

        # Messages: First, Response 1, system fork msg, edited user msg
        assert len(messages) == 4
        assert messages[0].content == "First"
        assert messages[0].role == "user"
        assert messages[1].content == "Response 1"
        assert messages[1].role == "assistant"
        assert messages[2].role == "system"
        assert "forked" in messages[2].content.lower()
        assert messages[3].content == "Second (edited)"
        assert messages[3].role == "user"

    @pytest.mark.asyncio
    async def test_fork_copies_composition_state_at_fork_point(self, service) -> None:
        """Fork copies the pre-send state from the forked message, not latest."""
        session = await service.create_session("alice", "Original", "local")

        # Save initial state
        state_v1 = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"plugin": "csv", "options": {"path": "data.csv"}},
                is_valid=True,
            ),
            provenance="session_seed",
        )

        # User message records pre-send state = v1
        fork_msg = await service.add_message(
            session.id,
            "user",
            "Build a pipeline",
            composition_state_id=state_v1.id,
            writer_principal="route_user_message",
        )

        # Assistant responds and mutates state to v2
        state_v2 = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"plugin": "json", "options": {"path": "data.json"}},
                nodes=[{"id": "n1", "plugin": "llm"}],
                is_valid=True,
            ),
            provenance="session_seed",
        )
        await service.add_message(
            session.id,
            "assistant",
            "Done!",
            composition_state_id=state_v2.id,
            writer_principal="compose_loop",
        )

        # Fork from the user message — should get state v1, not v2
        _, _, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Build a different pipeline",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        assert copied_state.sources == state_v1.sources
        # v2 had nodes; v1 did not
        assert copied_state.nodes is None

    @pytest.mark.asyncio
    async def test_fork_preserves_named_sources_at_fork_point(self, service) -> None:
        session = await service.create_session("alice", "Original", "local")
        sources = {
            "orders": {"plugin": "csv", "on_success": "orders_rows", "on_validation_failure": "discard", "options": {"path": "orders.csv"}},
            "refunds": {
                "plugin": "csv",
                "on_success": "refunds_rows",
                "on_validation_failure": "discard",
                "options": {"path": "refunds.csv"},
            },
        }
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(sources=sources, is_valid=True),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "Build this",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        _, _, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Build that",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        assert copied_state.sources == sources

    @pytest.mark.asyncio
    async def test_fork_raises_audit_integrity_error_for_cross_session_fork_message_state(
        self,
        service,
        engine,
    ) -> None:
        """Corrupted cross-session message provenance must fail loudly.

        fork_session() reads composition_state_id indirectly from a message in the
        source session, so it must use the session-scoped guard rather than the
        raw get_state() helper.
        """
        from elspeth.contracts.errors import AuditIntegrityError

        session_a = await service.create_session("alice", "Session A", "local")
        session_b = await service.create_session("alice", "Session B", "local")

        state_in_a = await service.save_composition_state(
            session_a.id,
            CompositionStateData(
                source={"plugin": "csv", "options": {"path": "a.csv"}},
                is_valid=True,
            ),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(session_b.id, "user", "Fork me", writer_principal="route_user_message")

        raw = engine.raw_connection()
        try:
            cursor = raw.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=OFF")
                cursor.execute(
                    "UPDATE chat_messages SET composition_state_id = ? WHERE id = ?",
                    (str(state_in_a.id), str(fork_msg.id)),
                )
                raw.commit()
                cursor.execute("PRAGMA foreign_keys=ON")
                raw.commit()
            finally:
                cursor.close()
        finally:
            raw.close()

        with pytest.raises(AuditIntegrityError, match="Tier 1 audit anomaly"):
            await _fork_session(
                service,
                source_session_id=session_b.id,
                fork_message_id=fork_msg.id,
                new_message_content="Fork me differently",
                user_id="alice",
                auth_provider_type="local",
            )

        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_fork_preserves_original_session(self, service) -> None:
        """Original session is unchanged after fork."""
        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        msg2 = await service.add_message(session.id, "user", "World", writer_principal="route_user_message")

        original_messages_before = await service.get_messages(session.id)

        await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=msg2.id,
            new_message_content="Universe",
            user_id="alice",
            auth_provider_type="local",
        )

        original_messages_after = await service.get_messages(session.id)
        assert len(original_messages_after) == len(original_messages_before)
        original_session = await service.get_session(session.id)
        assert original_session.title == "Original"

    @pytest.mark.asyncio
    async def test_fork_from_nonexistent_message_raises(self, service) -> None:
        """Fork fails if message doesn't exist in session."""
        session = await service.create_session("alice", "Test", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")

        with pytest.raises(ValueError, match="not found"):
            await _fork_session(
                service,
                source_session_id=session.id,
                fork_message_id=uuid.uuid4(),
                new_message_content="Hi",
                user_id="alice",
                auth_provider_type="local",
            )

    @pytest.mark.asyncio
    async def test_fork_from_assistant_message_raises(self, service) -> None:
        """Fork fails if target message is not a user message."""
        session = await service.create_session("alice", "Test", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        assistant_msg = await service.add_message(session.id, "assistant", "Hi", writer_principal="compose_loop")

        with pytest.raises(InvalidForkTargetError):
            await _fork_session(
                service,
                source_session_id=session.id,
                fork_message_id=assistant_msg.id,
                new_message_content="Hi",
                user_id="alice",
                auth_provider_type="local",
            )

    @pytest.mark.asyncio
    async def test_fork_from_first_message(self, service) -> None:
        """Forking from the first message copies no prior history."""
        session = await service.create_session("alice", "Test", "local")
        first_msg = await service.add_message(session.id, "user", "First", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "Response", writer_principal="compose_loop")

        _, messages, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=first_msg.id,
            new_message_content="First (edited)",
            user_id="alice",
            auth_provider_type="local",
        )

        # Only: system fork msg + edited user msg (no prior messages to copy)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].content == "First (edited)"

    @pytest.mark.asyncio
    async def test_fork_without_composition_state(self, service) -> None:
        """Fork works even when no composition state exists."""
        session = await service.create_session("alice", "Test", "local")
        msg = await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")

        new_session, _messages, state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=msg.id,
            new_message_content="Hello edited",
            user_id="alice",
            auth_provider_type="local",
        )

        assert state is None
        assert new_session.forked_from_session_id == session.id

    @pytest.mark.asyncio
    async def test_fork_new_messages_have_new_ids(self, service) -> None:
        """Copied messages get new IDs, not the originals."""
        session = await service.create_session("alice", "Test", "local")
        original_msg = await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        fork_msg = await service.add_message(session.id, "user", "World", writer_principal="route_user_message")

        _, messages, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Universe",
            user_id="alice",
            auth_provider_type="local",
        )

        copied_ids = {m.id for m in messages}
        assert original_msg.id not in copied_ids

    @pytest.mark.asyncio
    async def test_fork_preserves_assistant_raw_content_for_copied_history(self, service) -> None:
        """Fork copies raw model provenance for historical assistant messages."""
        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Build it", writer_principal="route_user_message")
        await service.add_message(
            session.id,
            "assistant",
            "I cannot mark this pipeline complete yet because runtime preflight failed: bad config.",
            raw_content="The pipeline is complete and valid.",
            writer_principal="compose_loop",
        )
        fork_msg = await service.add_message(session.id, "user", "Try again", writer_principal="route_user_message")

        _, messages, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Try a different way",
            user_id="alice",
            auth_provider_type="local",
        )

        copied_assistant = next(message for message in messages if message.role == "assistant")
        assert copied_assistant.content.startswith("I cannot mark this pipeline complete")
        assert copied_assistant.raw_content == "The pipeline is complete and valid."
        assert all(message.raw_content is None for message in messages if message.role in {"system", "user"})

    # ── §14.6 fork sweep regressions ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_fork_session_preserves_copied_writer_principal(self, service) -> None:
        """Copied rows must keep the source row's stored ``writer_principal``;
        fork-time inserts (system notice + new edited user) use ``session_fork``."""
        from sqlalchemy import select

        from elspeth.web.sessions import models

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Build it", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "OK", writer_principal="compose_loop")
        fork_msg = await service.add_message(session.id, "user", "Try again", writer_principal="route_user_message")

        new_session, _new_messages, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Different approach",
            user_id="alice",
            auth_provider_type="local",
        )

        with service._engine.begin() as conn:
            rows = conn.execute(
                select(
                    models.chat_messages_table.c.role,
                    models.chat_messages_table.c.writer_principal,
                    models.chat_messages_table.c.sequence_no,
                )
                .where(models.chat_messages_table.c.session_id == str(new_session.id))
                .order_by(models.chat_messages_table.c.sequence_no)
            ).fetchall()

        # Copied: user (route_user_message), assistant (compose_loop).
        # Synthetic: system (session_fork), user (session_fork).
        assert [(r.role, r.writer_principal) for r in rows] == [
            ("user", "route_user_message"),
            ("assistant", "compose_loop"),
            ("system", "session_fork"),
            ("user", "session_fork"),
            ("audit", "session_fork"),
        ]

    @pytest.mark.asyncio
    async def test_fork_session_assigns_contiguous_sequence_no(self, service) -> None:
        """``sequence_no`` for the new session must be ``[1, 2, ..., N+2]``
        with no gaps — N copied rows plus 2 fork-time inserts (system + user)."""
        from sqlalchemy import select

        from elspeth.web.sessions import models

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "1", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "2", writer_principal="compose_loop")
        await service.add_message(session.id, "user", "3", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "4", writer_principal="compose_loop")
        fork_msg = await service.add_message(session.id, "user", "5", writer_principal="route_user_message")

        new_session, _, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="edit",
            user_id="alice",
            auth_provider_type="local",
        )

        with service._engine.begin() as conn:
            seqs = (
                conn.execute(
                    select(models.chat_messages_table.c.sequence_no)
                    .where(models.chat_messages_table.c.session_id == str(new_session.id))
                    .order_by(models.chat_messages_table.c.sequence_no)
                )
                .scalars()
                .all()
            )

        # 4 copied + system notice + edited user + frozen plan audit row.
        assert seqs == [1, 2, 3, 4, 5, 6, 7]

    @pytest.mark.asyncio
    async def test_fork_session_preserves_tool_call_id_and_parent(self, service) -> None:
        """Tool-row fork: ``tool_call_id`` is carried verbatim; ``parent_assistant_id``
        is REWRITTEN from the source assistant id to the COPIED assistant id."""
        from sqlalchemy import select

        from elspeth.web.sessions import models

        session = await service.create_session("alice", "Original", "local")
        user_msg = await service.add_message(session.id, "user", "go", writer_principal="route_user_message")  # noqa: F841
        assistant_msg = await service.add_message(session.id, "assistant", "ok", writer_principal="compose_loop")
        await service.add_message(
            session.id,
            "tool",
            '{"ok":true}',
            writer_principal="compose_loop",
            tool_call_id="call_abc",
            parent_assistant_id=assistant_msg.id,
        )
        fork_msg = await service.add_message(session.id, "user", "again", writer_principal="route_user_message")

        new_session, _, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="retry",
            user_id="alice",
            auth_provider_type="local",
        )

        with service._engine.begin() as conn:
            assistant_row = conn.execute(
                select(models.chat_messages_table.c.id)
                .where(models.chat_messages_table.c.session_id == str(new_session.id))
                .where(models.chat_messages_table.c.role == "assistant")
            ).scalar_one()
            tool_row = conn.execute(
                select(
                    models.chat_messages_table.c.tool_call_id,
                    models.chat_messages_table.c.parent_assistant_id,
                )
                .where(models.chat_messages_table.c.session_id == str(new_session.id))
                .where(models.chat_messages_table.c.role == "tool")
            ).first()

        assert tool_row is not None
        assert tool_row.tool_call_id == "call_abc"
        # parent_assistant_id was rewritten to point at the COPIED assistant,
        # not the source assistant id.
        assert tool_row.parent_assistant_id == assistant_row
        assert tool_row.parent_assistant_id != str(assistant_msg.id)

    @pytest.mark.asyncio
    async def test_fork_session_rejects_tool_with_out_of_slice_parent(self, service) -> None:
        """If the slice ``[:fork_idx]`` excludes the assistant message a tool
        row depends on, fork must crash with the precise named error rather
        than letting the FK fire generically.

        The natural production flow can't easily produce this state (the
        compose loop always writes the assistant before its tool rows, and
        fork_idx is a user message), so we synthesize the slice by injecting
        a synthetic tool row into ``get_messages``' return whose
        ``parent_assistant_id`` is a UUID outside the slice. This exercises
        the offensive-programming guard directly, parallel to
        ``_assert_state_in_session``.
        """
        from uuid import uuid4

        from elspeth.web.sessions.protocol import ChatMessageRecord

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "first", writer_principal="route_user_message")
        fork_msg = await service.add_message(session.id, "user", "edit", writer_principal="route_user_message")

        original_get_messages = service.get_messages

        async def patched_get_messages(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            real = await original_get_messages(*args, **kwargs)
            synthetic_tool = ChatMessageRecord(
                id=uuid4(),
                session_id=session.id,
                role="tool",
                content="{}",
                created_at=real[0].created_at,
                writer_principal="compose_loop",
                tool_call_id="call_orphan",
                parent_assistant_id=uuid4(),
            )
            # Place the synthetic tool BEFORE fork_msg so it falls inside
            # the [:fork_idx] slice.
            return [real[0], synthetic_tool, real[1]]

        service.get_messages = patched_get_messages  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="fork slice excludes parent assistant"):
                await _fork_session(
                    service,
                    source_session_id=session.id,
                    fork_message_id=fork_msg.id,
                    new_message_content="retry",
                    user_id="alice",
                    auth_provider_type="local",
                )
        finally:
            service.get_messages = original_get_messages  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_fork_session_preserves_admin_tool_writer_principal_on_copied_rows(self, service) -> None:
        """Source rows with non-default ``writer_principal`` (e.g. ``admin_tool``)
        must be copied verbatim — fork-time provenance fabrication via role-
        keyed defaults is forbidden."""
        from sqlalchemy import select

        from elspeth.web.sessions import models

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "admin annotation", writer_principal="admin_tool")
        fork_msg = await service.add_message(session.id, "user", "fork here", writer_principal="route_user_message")

        new_session, _, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="ok",
            user_id="alice",
            auth_provider_type="local",
        )

        with service._engine.begin() as conn:
            principals = (
                conn.execute(
                    select(models.chat_messages_table.c.writer_principal)
                    .where(models.chat_messages_table.c.session_id == str(new_session.id))
                    .where(models.chat_messages_table.c.content == "admin annotation")
                )
                .scalars()
                .all()
            )
        assert principals == ["admin_tool"]

    @pytest.mark.asyncio
    async def test_fork_session_excludes_audit_rows_from_response_but_preserves_in_db(self, service) -> None:
        """Plan §2909: copied ``role="audit"`` rows live in the DB for audit
        fidelity, but must be excluded from the fork response payload."""
        from sqlalchemy import select

        from elspeth.web.sessions import models

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "go", writer_principal="route_user_message")
        await service.add_message(
            session.id,
            "audit",
            '{"_kind":"llm_call_audit","status":"ok"}',
            writer_principal="compose_loop",
        )
        fork_msg = await service.add_message(session.id, "user", "again", writer_principal="route_user_message")

        new_session, new_messages, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="retry",
            user_id="alice",
            auth_provider_type="local",
        )

        # Response payload — no audit row should be visible.
        assert all(m.role != "audit" for m in new_messages)

        # DB — audit row must still be persisted in the new session.
        with service._engine.begin() as conn:
            audit_rows = (
                conn.execute(
                    select(models.chat_messages_table.c.id)
                    .where(models.chat_messages_table.c.session_id == str(new_session.id))
                    .where(models.chat_messages_table.c.role == "audit")
                )
                .scalars()
                .all()
            )
        # Historical audit evidence plus this operation's strict blob plan.
        assert len(audit_rows) == 2

    @pytest.mark.asyncio
    async def test_fork_remaps_all_guided_message_references_and_preserves_reviewed_facts(self, service) -> None:
        """Schema-8 custody survives without parent chat or proposal authority."""
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.profile import EMPTY_PROFILE
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
        from elspeth.web.composer.guided.state_machine import GuidedSession

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root intent", writer_principal="route_user_message")
        deferred_origin = await service.add_message(
            session.id,
            "user",
            "deferred detail",
            writer_principal="route_user_message",
        )
        source_guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=root.id,
            deferred_message_id=deferred_origin.id,
            deferred_message_content=deferred_origin.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={},
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={"name": "Guided", "description": ""},
                is_valid=True,
                composer_meta={"guided_session": source_guided.to_dict()},
            ),
            provenance="session_seed",
        )
        _proposal, state, source_guided = await _attach_pending_fork_proposal(
            service,
            session_id=session.id,
            state=state,
            guided=source_guided,
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork here",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        _child, child_messages, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="changed root request",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        copied_meta = dict(deep_thaw(copied_state.composer_meta))
        copied_guided = GuidedSession.from_dict(copied_meta["guided_session"])
        copied_root = next(message for message in child_messages if message.content == "root intent")
        copied_deferred_origin = next(message for message in child_messages if message.content == "deferred detail")

        assert copied_guided.profile == EMPTY_PROFILE
        assert copied_guided.step is GuidedStep.STEP_3_TRANSFORMS
        assert copied_guided.reviewed_sources == source_guided.reviewed_sources
        assert copied_guided.reviewed_outputs == source_guided.reviewed_outputs
        assert copied_guided.deferred_intents[0].redacted_summary == source_guided.deferred_intents[0].redacted_summary
        assert copied_guided.root_intent_message_id == str(copied_root.id)
        assert copied_guided.root_intent_message_id != str(root.id)
        assert copied_guided.deferred_intents[0].originating_message_id == str(copied_deferred_origin.id)
        assert copied_guided.deferred_intents[0].originating_message_id != str(deferred_origin.id)
        assert copied_guided.active_proposal is None
        assert copied_guided.active_edit_target is None
        assert copied_guided.transition_consumed is False
        assert len(copied_guided.history) == 1
        assert copied_guided.history[0].response_hash == _FORK_HASH_B
        persisted_state = await service.get_current_state(copied_state.session_id)
        assert persisted_state is not None
        persisted_meta = dict(deep_thaw(persisted_state.composer_meta))
        assert GuidedSession.from_dict(persisted_meta["guided_session"]) == copied_guided

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("step", "turn_type"),
        [
            ("step_3_transforms", "propose_pipeline"),
            ("step_4_wire", "confirm_wiring"),
        ],
    )
    async def test_fork_rewinds_proposal_and_wire_stages_without_fabricating_answers(
        self,
        service,
        step: str,
        turn_type: str,
    ) -> None:
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.state_machine import GuidedSession

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        guided = _guided_fork_checkpoint(
            step=step,
            root_message_id=root.id,
            deferred_message_id=root.id,
            deferred_message_content=root.content,
            current_turn=turn_type,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                is_valid=True,
                metadata_={"name": "Guided", "description": ""},
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )
        _proposal, state, guided = await _attach_pending_fork_proposal(
            service,
            session_id=session.id,
            state=state,
            guided=guided,
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        _, _, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="retry",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        copied_meta = dict(deep_thaw(copied_state.composer_meta))
        copied_guided = GuidedSession.from_dict(copied_meta["guided_session"])
        assert copied_guided.step is GuidedStep.STEP_3_TRANSFORMS
        assert copied_guided.active_proposal is None
        assert copied_guided.active_edit_target is None
        assert copied_guided.transition_consumed is False
        assert len(copied_guided.history) == 1
        assert copied_guided.history[0].response_hash == _FORK_HASH_B
        assert copied_guided.history[0].summary == "Answered occurrence stays in fork history."

    @pytest.mark.asyncio
    async def test_fork_rejects_missing_active_proposal_authority_before_creating_child(self, service) -> None:
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=root.id,
            deferred_message_id=root.id,
            deferred_message_content=root.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                is_valid=True,
                metadata_={"name": "Guided", "description": ""},
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        with pytest.raises(AuditIntegrityError, match="missing or cross-session"):
            await _fork_session(
                service,
                source_session_id=session.id,
                fork_message_id=fork_msg.id,
                new_message_content="retry",
                user_id="alice",
                auth_provider_type="local",
            )

        sessions = await service.list_sessions("alice", "local")
        assert [record.id for record in sessions] == [session.id]

    @pytest.mark.asyncio
    async def test_fork_rejects_cross_session_active_proposal_before_creating_child(self, service) -> None:
        from dataclasses import replace

        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

        session = await service.create_session("alice", "Source", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=root.id,
            deferred_message_id=root.id,
            deferred_message_content=root.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={},
                nodes=[],
                edges=[],
                outputs=[],
                is_valid=True,
                metadata_={"name": "Source", "description": ""},
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )
        _foreign_session, _proposal, _foreign_state, foreign_guided, _foreign_fork = await _canonical_guided_fork_source(service)
        assert foreign_guided.active_proposal is not None
        cross_session_guided = replace(guided, active_proposal=foreign_guided.active_proposal)
        with service._engine.begin() as conn:
            conn.execute(
                update(composition_states_table)
                .where(composition_states_table.c.id == str(state.id))
                .values(composer_meta={"_version": 1, "data": {"guided_session": cross_session_guided.to_dict()}})
            )
        fork_message = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        await _assert_fork_integrity_failure_is_atomic(
            service,
            source_session_id=session.id,
            fork_message_id=fork_message.id,
            match="missing or cross-session",
        )

    @pytest.mark.asyncio
    async def test_fork_rejects_ambiguous_creation_authority_before_creating_child(self, service) -> None:
        session, proposal, _state, _guided, fork_message = await _canonical_guided_fork_source(service)
        with service._engine.begin() as conn:
            creation = conn.execute(
                select(proposal_events_table)
                .where(proposal_events_table.c.proposal_id == str(proposal.id))
                .where(proposal_events_table.c.event_type == "proposal.created")
            ).one()
            conn.execute(
                insert(proposal_events_table).values(
                    id=str(uuid.uuid4()),
                    session_id=str(session.id),
                    proposal_id=str(proposal.id),
                    event_type=creation.event_type,
                    actor=creation.actor,
                    payload=creation.payload,
                    created_at=creation.created_at,
                )
            )

        await _assert_fork_integrity_failure_is_atomic(
            service,
            source_session_id=session.id,
            fork_message_id=fork_message.id,
            match="exactly one creation event",
        )

    @pytest.mark.asyncio
    async def test_fork_rejects_terminal_active_proposal_before_creating_child(self, service) -> None:
        from elspeth.web.composer.guided.planning import guided_private_reviewed_facts

        session, proposal, _state, guided, fork_message = await _canonical_guided_fork_source(service)
        assert guided.active_proposal is not None
        await service.reject_pipeline_composition_proposal(
            session_id=session.id,
            proposal_id=proposal.id,
            draft_hash=guided.active_proposal.draft_hash,
            reviewed_facts=guided_private_reviewed_facts(guided),
            reason="superseded",
            dispatch=None,
            actor="test",
        )

        await _assert_fork_integrity_failure_is_atomic(
            service,
            source_session_id=session.id,
            fork_message_id=fork_message.id,
            match="terminal pipeline proposal",
        )
        with service._engine.begin() as conn:
            status = conn.execute(
                select(composition_proposals_table.c.status).where(composition_proposals_table.c.id == str(proposal.id))
            ).scalar_one()
        assert status == "rejected"

    @pytest.mark.asyncio
    async def test_fork_rejects_active_proposal_with_wrong_checkpoint_base(self, service) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=root.id,
            deferred_message_id=root.id,
            deferred_message_content=root.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        target = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={},
                nodes=[],
                edges=[],
                outputs=[],
                is_valid=True,
                metadata_={"name": "Target", "description": ""},
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )
        other = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={},
                nodes=[],
                edges=[],
                outputs=[],
                is_valid=True,
                metadata_={"name": "Other", "description": ""},
                composer_meta={"guided_session": None},
            ),
            provenance="session_seed",
        )
        _proposal, target, _guided = await _attach_pending_fork_proposal(
            service,
            session_id=session.id,
            state=target,
            guided=guided,
            proposal_base_state=other,
        )
        fork_message = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=target.id,
            writer_principal="route_user_message",
        )

        await _assert_fork_integrity_failure_is_atomic(
            service,
            source_session_id=session.id,
            fork_message_id=fork_message.id,
            match="checkpoint base",
        )

    @pytest.mark.asyncio
    async def test_fork_rejects_active_proposal_with_changed_checkpoint_content(self, service) -> None:
        session, _proposal, state, _guided, fork_message = await _canonical_guided_fork_source(service)
        with service._engine.begin() as conn:
            conn.execute(
                update(composition_states_table)
                .where(composition_states_table.c.id == str(state.id))
                .values(metadata_={"_version": 1, "data": {"name": "Tampered", "description": ""}})
            )

        await _assert_fork_integrity_failure_is_atomic(
            service,
            source_session_id=session.id,
            fork_message_id=fork_message.id,
            match="checkpoint content binding",
        )

    @pytest.mark.asyncio
    async def test_fork_topology_rewind_clears_terminal_and_advisor_state(self, service) -> None:
        from dataclasses import replace

        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
        from elspeth.web.composer.guided.state_machine import GuidedSession, TerminalKind, TerminalState

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        completed = _guided_fork_checkpoint(
            step=GuidedStep.STEP_4_WIRE.value,
            root_message_id=root.id,
            deferred_message_id=root.id,
            deferred_message_content=root.content,
            current_turn=TurnType.CONFIRM_WIRING.value,
        )
        guided = replace(
            completed,
            history=(*completed.history[:-1], replace(completed.history[-1], response_hash="c" * 64)),
            advisor_checkpoint_passes_used=3,
            advisor_signoff_escape_offered=True,
            terminal=TerminalState(
                kind=TerminalKind.COMPLETED,
                reason=None,
                pipeline_yaml="nodes: []",
            ),
            active_proposal=None,
            active_edit_target=None,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": guided.to_dict()}),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        _, _, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="retry",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        copied_meta = dict(deep_thaw(copied_state.composer_meta))
        copied_guided = GuidedSession.from_dict(copied_meta["guided_session"])
        assert copied_guided.terminal is None
        assert copied_guided.advisor_checkpoint_passes_used == 0
        assert copied_guided.advisor_signoff_escape_offered is False

    @pytest.mark.asyncio
    async def test_fork_topology_rewind_removes_trailing_step3_edit_turn(self, service) -> None:
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
        from elspeth.web.composer.guided.state_machine import GuidedSession

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=root.id,
            deferred_message_id=root.id,
            deferred_message_content=root.content,
            current_turn=TurnType.SCHEMA_FORM.value,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": guided.to_dict()}),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        _, _, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="retry",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        copied_meta = dict(deep_thaw(copied_state.composer_meta))
        copied_guided = GuidedSession.from_dict(copied_meta["guided_session"])
        assert len(copied_guided.history) == 1
        assert copied_guided.history[0].response_hash == _FORK_HASH_B

    @pytest.mark.asyncio
    @pytest.mark.parametrize("malformation", ["non_trailing", "multiple"])
    async def test_fork_rejects_malformed_unanswered_history_before_creating_child(
        self,
        service,
        malformation: str,
    ) -> None:
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=root.id,
            deferred_message_id=root.id,
            deferred_message_content=root.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        answered, unanswered = guided.history
        malformed_history = (unanswered, answered) if malformation == "non_trailing" else (unanswered, unanswered)
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": guided.to_dict()}),
            provenance="session_seed",
        )
        malformed_meta = deep_thaw(state.composer_meta)
        malformed_meta["guided_session"]["history"] = [record.to_dict() for record in malformed_history]
        with service._engine.begin() as conn:
            conn.execute(
                update(composition_states_table)
                .where(composition_states_table.c.id == str(state.id))
                .values(composer_meta={"_version": 1, "data": malformed_meta})
            )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        with pytest.raises(AuditIntegrityError, match="schema-9 authority is malformed"):
            await _fork_session(
                service,
                source_session_id=session.id,
                fork_message_id=fork_msg.id,
                new_message_content="retry",
                user_id="alice",
                auth_provider_type="local",
            )

        sessions = await service.list_sessions("alice", "local")
        assert [record.id for record in sessions] == [session.id]

    @pytest.mark.asyncio
    async def test_fork_rejects_out_of_slice_guided_message_reference_before_creating_child(self, service) -> None:
        """A child checkpoint can never point back to the excluded fork row."""
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

        session = await service.create_session("alice", "Guided", "local")
        fork_msg = await service.add_message(session.id, "user", "fork", writer_principal="route_user_message")
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=fork_msg.id,
            deferred_message_id=fork_msg.id,
            deferred_message_content=fork_msg.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": guided.to_dict()}),
            provenance="session_seed",
        )
        with service._engine.begin() as conn:
            conn.execute(
                update(chat_messages_table).where(chat_messages_table.c.id == str(fork_msg.id)).values(composition_state_id=str(state.id))
            )

        with pytest.raises(AuditIntegrityError, match="outside copied slice"):
            await _fork_session(
                service,
                source_session_id=session.id,
                fork_message_id=fork_msg.id,
                new_message_content="retry",
                user_id="alice",
                auth_provider_type="local",
            )

        sessions = await service.list_sessions("alice", "local")
        assert [record.id for record in sessions] == [session.id]

    @pytest.mark.asyncio
    async def test_fork_rejects_non_user_guided_lineage_before_creating_child(self, service) -> None:
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

        session = await service.create_session("alice", "Guided", "local")
        assistant = await service.add_message(
            session.id,
            "assistant",
            "assistant-authored lineage",
            writer_principal="compose_loop",
        )
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=assistant.id,
            deferred_message_id=assistant.id,
            deferred_message_content=assistant.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": guided.to_dict()}),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        with pytest.raises(AuditIntegrityError, match="must identify user messages"):
            await _fork_session(
                service,
                source_session_id=session.id,
                fork_message_id=fork_msg.id,
                new_message_content="retry",
                user_id="alice",
                auth_provider_type="local",
            )

        sessions = await service.list_sessions("alice", "local")
        assert [record.id for record in sessions] == [session.id]

    @pytest.mark.asyncio
    async def test_fork_rejects_deferred_content_hash_mismatch_before_creating_child(self, service) -> None:
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType

        session = await service.create_session("alice", "Guided", "local")
        root = await service.add_message(session.id, "user", "root", writer_principal="route_user_message")
        deferred_origin = await service.add_message(
            session.id,
            "user",
            "deferred detail",
            writer_principal="route_user_message",
        )
        guided = _guided_fork_checkpoint(
            step=GuidedStep.STEP_3_TRANSFORMS.value,
            root_message_id=root.id,
            deferred_message_id=deferred_origin.id,
            deferred_message_content=deferred_origin.content,
            current_turn=TurnType.PROPOSE_PIPELINE.value,
        )
        guided_meta = guided.to_dict()
        guided_meta["deferred_intents"][0]["message_content_hash"] = "0" * 64
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": guided_meta}),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        with pytest.raises(AuditIntegrityError, match="content hash mismatch"):
            await _fork_session(
                service,
                source_session_id=session.id,
                fork_message_id=fork_msg.id,
                new_message_content="retry",
                user_id="alice",
                auth_provider_type="local",
            )

        sessions = await service.list_sessions("alice", "local")
        assert [record.id for record in sessions] == [session.id]

    @pytest.mark.asyncio
    async def test_fork_and_archive_parent_session_with_durable_history(self, service) -> None:
        """Archiving a fork parent with durable history soft-archives the parent."""
        session = await service.create_session("alice", "Original", "local")
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"plugin": "csv", "options": {"path": "data.csv"}},
                is_valid=True,
            ),
            provenance="session_seed",
        )
        await service.add_message(session.id, "user", "Hello", composition_state_id=state.id, writer_principal="route_user_message")
        msg = await service.add_message(session.id, "user", "World", composition_state_id=state.id, writer_principal="route_user_message")

        await service.create_run(session.id, state.id)

        child_session, _, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=msg.id,
            new_message_content="Universe",
            user_id="alice",
            auth_provider_type="local",
        )

        await service.archive_session(session.id)

        archived_session = await service.get_session(session.id)
        assert archived_session.archived_at is not None

        child = await service.get_session(child_session.id)
        assert child.forked_from_session_id == session.id

    @pytest.mark.asyncio
    async def test_completed_outgoing_fork_retains_archived_parent_history(self, service) -> None:
        """A completed outgoing fork is durable parent history."""
        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        msg = await service.add_message(session.id, "user", "World", writer_principal="route_user_message")

        child_session, _, _ = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=msg.id,
            new_message_content="Universe",
            user_id="alice",
            auth_provider_type="local",
        )

        await service.archive_session(session.id)

        archived_parent = await service.get_session(session.id)
        assert archived_parent.archived_at is not None

        child = await service.get_session(child_session.id)
        assert child.forked_from_session_id == session.id

    @pytest.mark.asyncio
    async def test_fork_strips_tutorial_profile_from_guided_session(self, service) -> None:
        """Forking a tutorial-profile guided session yields the EMPTY profile.

        Critical case (finding 10, rev 4 — CORRECTED). The canonical tutorial
        source MATERIALISES (set_pipeline from ``source.inline_blob``) to a real
        ``json`` source whose ``options`` carry ``blob_ref``
        (``composer/tools/sessions.py:425``), so the route-layer blob-rewrite save
        DOES fire (``rewritten=True``). This fixture uses that real shape on
        purpose: it proves the strip survives EVEN on the path that re-saves the
        state — because the blob-rewrite re-save preserves ``composer_meta``
        verbatim (``sessions/routes/sessions.py:479-480``) and never strips the
        profile. The strip therefore lives in ``fork_session`` (both the :5150
        persist copy and the :5227 return copy) and is independent of
        ``rewritten``. (The earlier "no blob_ref => rewritten=False" framing was a
        false premise — see the spec's two-objects ``blob_ref`` note in §5/B4.)
        """
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
        from elspeth.web.composer.guided.state_machine import GuidedSession

        session = await service.create_session("alice", "Tutorial", "local")
        tutorial_guided = GuidedSession.initial(profile=TUTORIAL_PROFILE)
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                # Materialised canonical URL source (sessions.py:420-427): a real
                # ``json`` plugin with ``blob_ref`` in options => rewritten=True.
                # The blob-rewrite save fires but preserves composer_meta verbatim,
                # so the profile strip must still come from fork_session.
                sources={
                    "urls": {
                        "plugin": "json",
                        "options": {
                            "path": "composer_blobs/canonical-url-list.json",
                            "blob_ref": "a1b2c3d4-0000-0000-0000-000000000099",
                        },
                    }
                },
                is_valid=True,
                composer_meta={"guided_session": tutorial_guided.to_dict()},
            ),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "Build this",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        _, _, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Build something else",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        # Returned record (the :5227 copy) carries the EMPTY profile. The record
        # freezes composer_meta (CompositionStateRecord.__post_init__), so thaw
        # before GuidedSession.from_dict — the canonical read in converters.py:67.
        forked_meta = dict(deep_thaw(copied_state.composer_meta))
        forked_guided = GuidedSession.from_dict(forked_meta["guided_session"])
        assert forked_guided.profile == EMPTY_PROFILE
        # And it is PERSISTED that way (the :5150 copy) — re-read from the DB.
        persisted = await service.get_current_state(copied_state.session_id)
        persisted_meta = dict(deep_thaw(persisted.composer_meta))
        persisted_guided = GuidedSession.from_dict(persisted_meta["guided_session"])
        assert persisted_guided.profile == EMPTY_PROFILE

    @pytest.mark.asyncio
    async def test_fork_without_guided_session_passes_meta_through(self, service) -> None:
        """An ordinary (non-guided) fork is unaffected by the profile strip."""
        session = await service.create_session("alice", "Plain", "local")
        state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={"s": {"plugin": "csv", "options": {"path": "x.csv"}}},
                is_valid=True,
                composer_meta={"repair_turns_used": 2},
            ),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id, "user", "Build", composition_state_id=state.id, writer_principal="route_user_message"
        )
        _, _, copied_state = await _fork_session(
            service,
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Build other",
            user_id="alice",
            auth_provider_type="local",
        )
        assert copied_state is not None
        # composer_meta passes through verbatim (no guided_session key to strip).
        assert copied_state.composer_meta == {"repair_turns_used": 2}


# ── Route-level tests ───────────────────────────────────────────────────


def _make_fork_app(
    tmp_path: Path,
    user_id: str = "alice",
) -> tuple[FastAPI, SessionServiceImpl, BlobServiceImpl]:
    """Create a test app with session + blob services for fork testing."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    blob_service = BlobServiceImpl(engine, tmp_path)

    app = FastAPI()

    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user():
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    app.state.session_service = session_service
    app.state.blob_service = blob_service
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.composer_service = None

    from elspeth.web.middleware.rate_limit import ComposerRateLimiter

    app.state.rate_limiter = ComposerRateLimiter(limit=100)

    router = create_session_router()
    app.include_router(router)
    app.include_router(create_blobs_router())

    return app, session_service, blob_service


def _read_composition_state_provenances(service: SessionServiceImpl, session_id: str) -> list[str]:
    from sqlalchemy import text

    with service._engine.connect() as conn:
        rows = conn.execute(
            text("SELECT provenance FROM composition_states WHERE session_id = :sid ORDER BY version"),
            {"sid": session_id},
        ).fetchall()
    return [row.provenance for row in rows]


class TestForkEndpoint:
    """Route-level tests for POST /api/sessions/{id}/fork."""

    @pytest.mark.asyncio
    async def test_partial_copy_stale_worker_takeover_completes_without_stale_cleanup(self, tmp_path) -> None:
        """A worker fenced after one copy joins the takeover winner without compensating it."""

        app, service, blob_service = _make_fork_app(tmp_path)
        parent = await service.create_session("alice", "Parent", "local")
        source_blobs = [
            await blob_service.create_blob(parent.id, f"source-{index}.csv", f"v\n{index}\n".encode(), "text/csv") for index in range(2)
        ]
        message = await service.add_message(
            parent.id,
            "user",
            "fork",
            writer_principal="route_user_message",
        )
        operation_id = str(uuid.uuid4())
        body = {
            "operation_id": operation_id,
            "from_message_id": str(message.id),
            "new_message_content": "edited",
        }
        client = TestClient(app, raise_server_exceptions=False)
        original_copy = blob_service.copy_blobs_for_fork
        original_cleanup = blob_service.cleanup_blobs_for_fork
        partial_copied = threading.Barrier(2)
        resume_stale = threading.Barrier(2)
        call_guard = threading.Lock()
        copy_calls = 0
        cleanup_calls = 0

        async def controlled_copy(*args: Any, **kwargs: Any):
            nonlocal copy_calls
            with call_guard:
                copy_calls += 1
                invocation = copy_calls
            checkpoint = kwargs["checkpoint"]
            checkpoint_count = 0

            async def controlled_checkpoint() -> None:
                nonlocal checkpoint_count
                await checkpoint()
                checkpoint_count += 1
                if invocation == 1 and checkpoint_count == 3:
                    await asyncio.to_thread(partial_copied.wait, 5)
                    await asyncio.to_thread(resume_stale.wait, 5)

            return await original_copy(*args, **{**kwargs, "checkpoint": controlled_checkpoint})

        async def observed_cleanup(*args: Any, **kwargs: Any):
            nonlocal cleanup_calls
            cleanup_calls += 1
            return await original_cleanup(*args, **kwargs)

        with (
            patch.object(blob_service, "copy_blobs_for_fork", new=controlled_copy),
            patch.object(blob_service, "cleanup_blobs_for_fork", new=observed_cleanup),
        ):
            stale_task = asyncio.create_task(
                asyncio.to_thread(
                    client.post,
                    f"/api/sessions/{parent.id}/fork",
                    json=body,
                )
            )
            await asyncio.to_thread(partial_copied.wait, 5)
            with service._engine.connect() as conn:
                operation = conn.execute(
                    select(guided_operations_table).where(
                        guided_operations_table.c.session_id == str(parent.id),
                        guided_operations_table.c.operation_id == operation_id,
                    )
                ).one()
                child_id = uuid.UUID(operation.result_session_id)
                assert (
                    conn.execute(
                        select(func.count()).select_from(blobs_table).where(blobs_table.c.session_id == str(child_id))
                    ).scalar_one()
                    == 1
                )
            with service._engine.begin() as conn:
                conn.execute(
                    update(guided_operations_table)
                    .where(
                        guided_operations_table.c.session_id == str(parent.id),
                        guided_operations_table.c.operation_id == operation_id,
                    )
                    .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
                )

            winner_response = await asyncio.to_thread(
                client.post,
                f"/api/sessions/{parent.id}/fork",
                json=body,
            )
            await asyncio.to_thread(resume_stale.wait, 5)
            stale_response = await stale_task

        assert winner_response.status_code == stale_response.status_code == 201
        assert winner_response.content == stale_response.content
        assert winner_response.json() == {"session_id": str(child_id)}
        assert copy_calls == 2
        assert cleanup_calls == 0
        child = await service.get_session(child_id)
        assert child.archived_at is None
        child_blob_ids = {item.id for item in await blob_service.list_blobs(child_id, limit=None)}
        assert child_blob_ids == {fork_blob_id(target_session_id=child_id, source_blob_id=source_blob.id) for source_blob in source_blobs}
        with service._engine.connect() as conn:
            assert (
                conn.execute(
                    select(func.count()).select_from(sessions_table).where(sessions_table.c.forked_from_session_id == str(parent.id))
                ).scalar_one()
                == 1
            )
            operation = conn.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == str(parent.id),
                    guided_operations_table.c.operation_id == operation_id,
                )
            ).one()
            assert operation.status == "completed"
            assert operation.attempt == 2
            assert operation.result_session_id == str(child_id)

    @pytest.mark.asyncio
    async def test_successful_fork_lost_response_replays_exact_locator_and_rejects_hash_tamper(self, tmp_path) -> None:
        """A committed response is byte-stable, copy-once, and hash-verified on replay."""

        app, service, blob_service = _make_fork_app(tmp_path)
        parent = await service.create_session("alice", "Parent", "local")
        await blob_service.create_blob(parent.id, "source.csv", b"a,b\n1,2\n", "text/csv")
        message = await service.add_message(
            parent.id,
            "user",
            "fork",
            writer_principal="route_user_message",
        )
        operation_id = str(uuid.uuid4())
        body = {
            "operation_id": operation_id,
            "from_message_id": str(message.id),
            "new_message_content": "edited",
        }
        original_copy = blob_service.copy_blobs_for_fork
        copy_calls = 0

        async def observed_copy(*args: Any, **kwargs: Any):
            nonlocal copy_calls
            copy_calls += 1
            return await original_copy(*args, **kwargs)

        client = TestClient(app, raise_server_exceptions=False)
        with patch.object(blob_service, "copy_blobs_for_fork", new=observed_copy):
            first = client.post(f"/api/sessions/{parent.id}/fork", json=body)
            replay = client.post(f"/api/sessions/{parent.id}/fork", json=body)

        assert first.status_code == replay.status_code == 201
        assert first.content == replay.content
        child_id = uuid.UUID(first.json()["session_id"])
        assert copy_calls == 1
        expected_hash = guided_response_hash(ForkSessionResponse(session_id=child_id))
        with service._engine.connect() as conn:
            assert (
                conn.execute(
                    select(func.count()).select_from(sessions_table).where(sessions_table.c.forked_from_session_id == str(parent.id))
                ).scalar_one()
                == 1
            )
            operation = conn.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == str(parent.id),
                    guided_operations_table.c.operation_id == operation_id,
                )
            ).one()
            assert operation.response_hash == expected_hash

        with service._engine.begin() as conn:
            conn.execute(text("DROP TRIGGER trg_guided_operations_terminal_immutable"))
            conn.execute(
                update(guided_operations_table)
                .where(
                    guided_operations_table.c.session_id == str(parent.id),
                    guided_operations_table.c.operation_id == operation_id,
                )
                .values(response_hash="f" * 64)
            )
        tampered = client.post(f"/api/sessions/{parent.id}/fork", json=body)
        assert tampered.status_code == 500

    @pytest.mark.asyncio
    async def test_partial_copy_failure_cleans_exact_cohort_once_and_replays_terminal_failure(self, tmp_path) -> None:
        """The fail-CAS winner cleans copied blobs once while retaining strict evidence."""

        app, service, blob_service = _make_fork_app(tmp_path)
        parent = await service.create_session("alice", "Parent", "local")
        source_blobs = [
            await blob_service.create_blob(parent.id, f"source-{index}.csv", f"v\n{index}\n".encode(), "text/csv") for index in range(2)
        ]
        message = await service.add_message(
            parent.id,
            "user",
            "fork",
            writer_principal="route_user_message",
        )
        operation_id = str(uuid.uuid4())
        body = {
            "operation_id": operation_id,
            "from_message_id": str(message.id),
            "new_message_content": "edited",
        }
        original_copy = blob_service.copy_blobs_for_fork
        original_cleanup = blob_service.cleanup_blobs_for_fork
        copy_calls = 0
        cleanup_calls = 0
        cleanup_results: list[Any] = []

        async def fail_after_first_copy(*args: Any, **kwargs: Any):
            nonlocal copy_calls
            copy_calls += 1
            checkpoint = kwargs["checkpoint"]
            checkpoint_count = 0

            async def failing_checkpoint() -> None:
                nonlocal checkpoint_count
                await checkpoint()
                checkpoint_count += 1
                if checkpoint_count == 3:
                    raise RuntimeError("injected fault after first durable blob copy")

            return await original_copy(*args, **{**kwargs, "checkpoint": failing_checkpoint})

        async def observed_cleanup(*args: Any, **kwargs: Any):
            nonlocal cleanup_calls
            cleanup_calls += 1
            result = await original_cleanup(*args, **kwargs)
            cleanup_results.append(result)
            return result

        client = TestClient(app, raise_server_exceptions=False)
        with (
            patch.object(blob_service, "copy_blobs_for_fork", new=fail_after_first_copy),
            patch.object(blob_service, "cleanup_blobs_for_fork", new=observed_cleanup),
        ):
            first = client.post(f"/api/sessions/{parent.id}/fork", json=body)
            replay = client.post(f"/api/sessions/{parent.id}/fork", json=body)

        assert first.status_code == replay.status_code == 500
        assert first.content == replay.content
        assert first.json()["detail"]["failure_code"] == "operation_failed"
        assert copy_calls == cleanup_calls == 1
        assert len(cleanup_results) == 1
        with service._engine.connect() as conn:
            operation = conn.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == str(parent.id),
                    guided_operations_table.c.operation_id == operation_id,
                )
            ).one()
            assert operation.status == "failed"
            assert operation.failure_code == "operation_failed"
            assert operation.result_session_id is None
            child_id = uuid.UUID(
                conn.execute(select(sessions_table.c.id).where(sessions_table.c.forked_from_session_id == str(parent.id))).scalar_one()
            )
            child = conn.execute(select(sessions_table).where(sessions_table.c.id == str(child_id))).one()
            assert child.archived_at is not None
            assert (
                conn.execute(select(func.count()).select_from(blobs_table).where(blobs_table.c.session_id == str(child_id))).scalar_one()
                == 0
            )
            plan_rows = conn.execute(
                select(chat_messages_table.c.content).where(
                    chat_messages_table.c.session_id == str(child_id),
                    chat_messages_table.c.role == "audit",
                    chat_messages_table.c.writer_principal == "session_fork",
                )
            ).all()
        assert len(plan_rows) == 1
        plan = json.loads(plan_rows[0].content)
        assert plan["schema"] == "session-fork-blob-plan.v1"
        assert plan["source_session_id"] == str(parent.id)
        assert plan["child_session_id"] == str(child_id)
        assert plan["operation_id"] == operation_id
        assert {entry["source_blob_id"] for entry in plan["source_blobs"]} == {str(blob.id) for blob in source_blobs}
        assert set(cleanup_results[0].deleted_ids) == {
            fork_blob_id(target_session_id=child_id, source_blob_id=uuid.UUID(plan["source_blobs"][0]["source_blob_id"]))
        }
        assert cleanup_results[0].errors == ()

    @pytest.mark.asyncio
    async def test_staged_archived_child_is_404_through_session_and_blob_public_gates(self, tmp_path) -> None:
        from fastapi import HTTPException

        from elspeth.web.sessions.ownership import verify_session_ownership

        app, service, blob_service = _make_fork_app(tmp_path)
        parent = await service.create_session("alice", "Parent", "local")
        fork_message = await service.add_message(
            parent.id,
            "user",
            "fork",
            writer_principal="route_user_message",
        )
        claimed = await service.reserve_guided_operation(
            session_id=parent.id,
            operation_id=str(uuid.uuid4()),
            kind="session_fork",
            request_hash="a" * 64,
            actor="test",
            lease_seconds=300,
        )
        assert isinstance(claimed, GuidedOperationClaimed)
        staged = await service.fork_session(
            claimed.fence,
            fork_message_id=fork_message.id,
            new_message_content="edited",
        )
        staged_blob = await blob_service.create_blob(
            staged.session.id,
            "staged.csv",
            b"private staged bytes",
            "text/csv",
        )
        client = TestClient(app)

        listed = client.get("/api/sessions?include_archived=true")
        assert listed.status_code == 200
        assert str(staged.session.id) not in {item["id"] for item in listed.json()}
        assert client.get(f"/api/sessions/{staged.session.id}").status_code == 404
        assert client.get(f"/api/sessions/{staged.session.id}/messages").status_code == 404
        assert client.get(f"/api/sessions/{staged.session.id}/blobs").status_code == 404
        assert client.get(f"/api/sessions/{staged.session.id}/blobs/{staged_blob.id}").status_code == 404
        assert client.delete(f"/api/sessions/{staged.session.id}/blobs/{staged_blob.id}").status_code == 404
        with pytest.raises(HTTPException) as shared_gate:
            await verify_session_ownership(
                staged.session.id,
                UserIdentity(user_id="alice", username="alice"),
                type("RequestStub", (), {"app": app})(),
            )
        assert shared_gate.value.status_code == 404

    @pytest.mark.asyncio
    async def test_pending_inspection_review_fork_rewrites_custody_and_commits_from_child_blob(self, tmp_path) -> None:
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
        from elspeth.web.composer.guided.stage_transitions import (
            AnsweredTurn,
            InspectionResponse,
            transition_source_inspection_review,
        )
        from elspeth.web.composer.guided.state_machine import GuidedSession, SourceIntent, TurnRecord
        from elspeth.web.composer.source_inspection import SourceInspectionFacts

        app, service, blob_service = _make_fork_app(tmp_path)
        parent = await service.create_session("alice", "Parent", "local")
        root = await service.add_message(parent.id, "user", "root", writer_principal="route_user_message")
        parent_blob = await blob_service.create_blob(parent.id, "orders.csv", b"id,name\n1,Ada\n", "text/csv")
        stable_id = str(uuid.uuid4())
        guided = GuidedSession(
            step=GuidedStep.STEP_1_SOURCE,
            history=(
                TurnRecord(
                    step=GuidedStep.STEP_1_SOURCE,
                    turn_type=TurnType.INSPECT_AND_CONFIRM,
                    payload_hash="a" * 64,
                    response_hash=None,
                    emitter="server",
                ),
            ),
            source_order=(stable_id,),
            pending_source_intents={
                stable_id: SourceIntent(
                    name="orders",
                    phase="inspection_review",
                    plugin="csv",
                    options={
                        "path": f"blob:{parent_blob.id}",
                        "blob_ref": str(parent_blob.id),
                        "on_validation_failure": "discard",
                    },
                    inspection_facts=SourceInspectionFacts(
                        source_kind="csv",
                        redacted_identity={
                            "filename": "orders.csv",
                            "mime_type": "text/csv",
                            "blob_id": str(parent_blob.id),
                        },
                        byte_range_inspected=(0, parent_blob.size_bytes),
                        sample_row_count=1,
                        observed_headers=("id", "name"),
                        inferred_types={"id": "int", "name": "str"},
                        url_candidates=(),
                        warnings=(),
                    ),
                    observed_columns=("id", "name"),
                    sample_rows=({"id": 1, "name": "Ada"},),
                )
            },
            root_intent_message_id=str(root.id),
        )
        state = await service.save_composition_state(
            parent.id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": guided.to_dict()}),
            provenance="session_seed",
        )
        fork_message = await service.add_message(
            parent.id,
            "user",
            "fork",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )
        response = TestClient(app).post(
            f"/api/sessions/{parent.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(fork_message.id),
                "new_message_content": "edited",
            },
        )
        assert response.status_code == 201
        child_id = uuid.UUID(response.json()["session_id"])
        child_state = await service.get_current_state(child_id)
        assert child_state is not None
        child_guided = GuidedSession.from_dict(deep_thaw(child_state.composer_meta["guided_session"]))
        child_intent = child_guided.pending_source_intents[stable_id]
        child_blob_id = child_intent.inspection_facts.redacted_identity["blob_id"]
        assert child_blob_id != str(parent_blob.id)
        assert child_intent.options["blob_ref"] == child_blob_id
        assert child_intent.options["path"] == f"blob:{child_blob_id}"
        assert child_intent.sample_rows == ({"id": 1, "name": "Ada"},)
        child_blob = await blob_service.get_blob(uuid.UUID(child_blob_id))
        assert child_blob.session_id == child_id
        assert await blob_service.read_blob_content(child_blob.id) == b"id,name\n1,Ada\n"

        committed = transition_source_inspection_review(
            child_guided,
            target_id=stable_id,
            turn=AnsweredTurn(history_index=0),
            response=InspectionResponse(columns=("id", "name")),
        )
        assert stable_id not in committed.pending_source_intents
        assert committed.reviewed_sources[stable_id].options["blob_ref"] == child_blob_id
        assert str(parent_blob.id) not in str(committed.to_dict())
        committed_state = await service.save_composition_state(
            child_id,
            CompositionStateData(is_valid=True, composer_meta={"guided_session": committed.to_dict()}),
            provenance="post_compose",
        )
        reloaded = await service.get_state_in_session(committed_state.id, child_id)
        reloaded_guided = GuidedSession.from_dict(deep_thaw(reloaded.composer_meta["guided_session"]))
        assert stable_id not in reloaded_guided.pending_source_intents
        assert reloaded_guided.reviewed_sources[stable_id].options["blob_ref"] == child_blob_id
        assert str(parent_blob.id) not in str(reloaded_guided.to_dict())

    @pytest.mark.asyncio
    async def test_fork_endpoint_creates_session(self, tmp_path) -> None:
        app, service, _ = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        msg = await service.add_message(session.id, "user", "Hello world", writer_principal="route_user_message")

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Hello universe",
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert set(body) == {"session_id"}
        child_id = uuid.UUID(body["session_id"])
        child = await service.get_session(child_id)
        assert child.forked_from_session_id == session.id
        assert child.forked_from_message_id == msg.id
        assert "(fork)" in child.title

        # New session should have system + edited user messages
        msgs = await service.get_messages(child_id, limit=None)
        assert any(message.role == "system" for message in msgs)
        assert any(message.content == "Hello universe" for message in msgs)

    @pytest.mark.asyncio
    async def test_fork_blob_rewrite_persists_session_fork_provenance(self, tmp_path) -> None:
        """Fork-time blob remapping is still part of the fork writer path."""
        app, service, blob_service = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        blob = await blob_service.create_blob(
            session.id,
            "data.csv",
            b"a,b\n1,2",
            "text/csv",
        )
        source_state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={
                    "my_csv": {
                        "plugin": "csv",
                        "options": {
                            "blob_ref": str(blob.id),
                            "path": blob.storage_path,
                        },
                    }
                },
                is_valid=True,
            ),
            provenance="session_seed",
        )
        msg = await service.add_message(
            session.id,
            "user",
            "Process this",
            composition_state_id=source_state.id,
            writer_principal="route_user_message",
        )

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 201
        fork_session_id = response.json()["session_id"]
        assert _read_composition_state_provenances(service, fork_session_id) == ["session_fork"]

    @pytest.mark.asyncio
    async def test_fork_endpoint_idor_protection(self, tmp_path) -> None:
        """Fork endpoint returns 404 for sessions not owned by the user."""
        app, service, _ = _make_fork_app(tmp_path, user_id="alice")
        client = TestClient(app)

        # Create a session as "bob" directly in the service (bypassing auth)
        bob_session = await service.create_session("bob", "Bob's Session", "local")
        msg = await service.add_message(bob_session.id, "user", "Hello", writer_principal="route_user_message")

        # Alice tries to fork Bob's session
        response = client.post(
            f"/api/sessions/{bob_session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Hi",
            },
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_fork_endpoint_nonexistent_message(self, tmp_path) -> None:
        app, service, _ = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Test", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(uuid.uuid4()),
                "new_message_content": "Hi",
            },
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_fork_preserves_original_messages(self, tmp_path) -> None:
        """Original session is unchanged after fork via endpoint."""
        app, service, _ = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "First", writer_principal="route_user_message")
        msg2 = await service.add_message(session.id, "user", "Second", writer_principal="route_user_message")

        # Get message count before fork
        msgs_before = await service.get_messages(session.id)

        client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg2.id),
                "new_message_content": "Second edited",
            },
        )

        # Verify original unchanged
        msgs_after = await service.get_messages(session.id)
        assert len(msgs_after) == len(msgs_before)

    @pytest.mark.asyncio
    async def test_fork_copies_blobs(self, tmp_path) -> None:
        """Blobs from source session are copied to forked session."""
        app, service, blob_service = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        await blob_service.create_blob(
            session.id,
            "data.csv",
            b"a,b,c\n1,2,3",
            "text/csv",
        )
        msg = await service.add_message(session.id, "user", "Process this", writer_principal="route_user_message")

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 201
        new_session_id = uuid.UUID(response.json()["session_id"])

        # Verify blob was copied to new session
        new_blobs = await blob_service.list_blobs(new_session_id)
        assert len(new_blobs) == 1
        assert new_blobs[0].filename == "data.csv"
        assert new_blobs[0].session_id == new_session_id

        # Verify content matches
        content = await blob_service.read_blob_content(new_blobs[0].id)
        assert content == b"a,b,c\n1,2,3"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source_key", ["path", "file"])
    async def test_fork_rewrites_blob_backed_source_paths_to_copied_blob(
        self,
        tmp_path,
        source_key: str,
    ) -> None:
        """Forked composition state must point at the copied blob, not the source session.

        The blob subsystem accepts both ``path`` and ``file`` as blob-backed
        source references, so fork rewriting must remap both shapes.
        """
        app, service, blob_service = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        original_blob = await blob_service.create_blob(
            session.id,
            "data.csv",
            b"a,b,c\n1,2,3",
            "text/csv",
        )
        source_state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "options": {
                        source_key: original_blob.storage_path,
                    },
                },
                is_valid=True,
            ),
            provenance="session_seed",
        )
        msg = await service.add_message(
            session.id,
            "user",
            "Process this",
            composition_state_id=source_state.id,
            writer_principal="route_user_message",
        )

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 201
        new_session_id = uuid.UUID(response.json()["session_id"])

        copied_blobs = await blob_service.list_blobs(new_session_id)
        assert len(copied_blobs) == 1
        copied_blob = copied_blobs[0]

        copied_state = await service.get_current_state(new_session_id)
        assert copied_state is not None
        options = copied_state.sources["source"]["options"]
        assert options["blob_ref"] == str(copied_blob.id)
        assert options[source_key] == copied_blob.storage_path
        assert options[source_key] != original_blob.storage_path

    @pytest.mark.asyncio
    async def test_fork_rewrites_inline_content_markers_to_copied_blobs(self, tmp_path) -> None:
        """Forked inline_content refs must point at copied blobs in the target session."""
        app, service, blob_service = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        original_blob = await blob_service.create_blob(
            session.id,
            "prompt.txt",
            b"Classify this row.",
            "text/plain",
        )
        marker = {
            "blob_ref": str(original_blob.id),
            "mode": "inline_content",
            "sha256": original_blob.content_hash,
            "encoding": "utf-16",
        }
        source_state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "options": {
                        "path": original_blob.storage_path,
                    },
                },
                nodes=[
                    {
                        "id": "classify",
                        "node_type": "transform",
                        "plugin": "llm",
                        "options": {"prompt_template": marker},
                    }
                ],
                outputs=[
                    {
                        "name": "results",
                        "plugin": "json",
                        "options": {"header": marker},
                    }
                ],
                is_valid=True,
            ),
            provenance="session_seed",
        )
        msg = await service.add_message(
            session.id,
            "user",
            "Process this",
            composition_state_id=source_state.id,
            writer_principal="route_user_message",
        )

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 201
        new_session_id = uuid.UUID(response.json()["session_id"])
        copied_blob = (await blob_service.list_blobs(new_session_id))[0]

        copied_state = await service.get_current_state(new_session_id)
        assert copied_state is not None
        assert copied_state.nodes is not None
        assert copied_state.outputs is not None
        copied_node_marker = copied_state.nodes[0]["options"]["prompt_template"]
        copied_output_marker = copied_state.outputs[0]["options"]["header"]

        assert copied_node_marker == {
            "blob_ref": str(copied_blob.id),
            "mode": "inline_content",
            "sha256": original_blob.content_hash,
            "encoding": "utf-16",
        }
        assert copied_output_marker == copied_node_marker
        state_blob_refs = repr((copied_state.source, copied_state.nodes, copied_state.outputs))
        assert str(original_blob.id) not in state_blob_refs

    @pytest.mark.asyncio
    async def test_fork_inline_content_marker_without_copied_blob_fails_closed(self, tmp_path) -> None:
        """Inline-content refs must be audited even when no source blobs are copied."""
        app, service, _blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")
        missing_blob_id = uuid.uuid4()
        source_state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                nodes=[
                    {
                        "id": "classify",
                        "node_type": "transform",
                        "plugin": "llm",
                        "options": {
                            "prompt_template": {
                                "blob_ref": str(missing_blob_id),
                                "mode": "inline_content",
                                "sha256": "a" * 64,
                            }
                        },
                    }
                ],
                is_valid=True,
            ),
            provenance="session_seed",
        )
        msg = await service.add_message(
            session.id,
            "user",
            "Process this",
            composition_state_id=source_state.id,
            writer_principal="route_user_message",
        )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 500

        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_fork_preserves_original_messages_status_check(self, tmp_path) -> None:
        """Fork endpoint returns 201 and original session is unchanged."""
        app, service, _ = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "First", writer_principal="route_user_message")
        msg2 = await service.add_message(session.id, "user", "Second", writer_principal="route_user_message")

        msgs_before = await service.get_messages(session.id)

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg2.id),
                "new_message_content": "Second edited",
            },
        )

        assert response.status_code == 201
        msgs_after = await service.get_messages(session.id)
        assert len(msgs_after) == len(msgs_before)

    @pytest.mark.asyncio
    async def test_fork_from_assistant_message_returns_422(self, tmp_path) -> None:
        """Forking from an assistant message returns 422, not 404."""
        app, service, _ = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Test", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        assistant_msg = await service.add_message(session.id, "assistant", "Hi", writer_principal="compose_loop")

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(assistant_msg.id),
                "new_message_content": "Hi",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_fork_blob_quota_exceeded_returns_413(self, tmp_path) -> None:
        """Fork returns and replays the same closed, safe 413 quota failure."""
        # Create blob service with very small quota
        from sqlalchemy.pool import StaticPool

        engine = create_session_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        initialize_session_schema(engine)
        session_service = SessionServiceImpl(
            engine,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test"),
        )
        # Source blob service has generous quota; we'll swap to a tight one for the fork
        blob_service = BlobServiceImpl(engine, tmp_path, max_storage_per_session=500)

        app = FastAPI()

        identity = UserIdentity(user_id="alice", username="alice")

        async def mock_user():
            return identity

        app.dependency_overrides[get_current_user] = mock_user
        app.state.session_service = session_service
        app.state.blob_service = blob_service
        app.state.settings = WebSettings(
            data_dir=tmp_path,
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=b"\x00" * 32,
        )
        app.state.composer_service = None

        from elspeth.web.middleware.rate_limit import ComposerRateLimiter

        app.state.rate_limiter = ComposerRateLimiter(limit=100)
        router = create_session_router()
        app.include_router(router)

        client = TestClient(app)

        # Create source session with blobs using the generous-quota service
        session = await session_service.create_session("alice", "Original", "local")
        await blob_service.create_blob(
            session.id,
            "big.csv",
            b"x" * 200,
            "text/csv",
        )

        # Now swap the blob service on the app to one with a tiny quota (50 bytes)
        # so the fork's copy will exceed the target session quota
        tight_blob_service = BlobServiceImpl(engine, tmp_path, max_storage_per_session=50)
        app.state.blob_service = tight_blob_service
        msg = await session_service.add_message(session.id, "user", "Go", writer_principal="route_user_message")

        operation_id = str(uuid.uuid4())
        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": operation_id,
                "from_message_id": str(msg.id),
                "new_message_content": "Go edited",
            },
        )
        replay = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": operation_id,
                "from_message_id": str(msg.id),
                "new_message_content": "Go edited",
            },
        )

        assert response.status_code == replay.status_code == 413
        assert (
            response.json()
            == replay.json()
            == {
                "detail": {
                    "error_type": "guided_operation_terminal_failure",
                    "failure_code": "quota_exceeded",
                    "detail": "The operation exceeded the session storage quota.",
                }
            }
        )

        # The staged child remains hidden as integrity evidence.
        sessions = await session_service.list_sessions("alice", "local")
        assert len(sessions) == 1  # Only the original remains

    @pytest.mark.asyncio
    async def test_fork_with_non_uuid_blob_ref_raises_audit_integrity_error_and_archives(self, tmp_path) -> None:
        """Tier 1 anomaly: non-UUID blob_ref in composition_states.source must crash.

        composition_states.source is our own data (Tier 1).  blob_ref is
        written by composer/tools.py as a UUID string; a malformed value at
        fork time indicates a write-path bug, DB corruption, or tampering
        and must crash per CLAUDE.md's Tier 1 trust model.  Silently skipping
        the remap would leave the forked session's blob_ref pointing at the
        source session's blob (cross-session reference, audit-contradictory).

        The fork coordinator must fail the operation, retain its already
        archived child and frozen plan as evidence, and compensate only any
        copied child blobs. The partial child must remain hidden from users.
        """
        app, service, blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")

        # Tier 1 anomaly: persist a non-UUID blob_ref (simulates corrupt
        # or tampered source data).
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "options": {"blob_ref": "not-a-valid-uuid", "path": "/data/x.csv"},
                },
                is_valid=True,
            ),
            provenance="session_seed",
        )

        current_state = await service.get_current_state(session.id)
        assert current_state is not None
        msg = await service.add_message(
            session.id,
            "user",
            "Hello",
            composition_state_id=current_state.id,
            writer_principal="route_user_message",
        )

        # Create a blob so blob_map is non-empty (triggers the rewrite path).
        await blob_service.create_blob(
            session.id,
            "data.csv",
            b"a,b\n1,2",
            "text/csv",
        )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Hello edited",
            },
        )
        assert response.status_code == 500

        # The failed archived child is retained as evidence but stays hidden;
        # only the original remains visible to the owner.
        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_fork_with_non_string_blob_ref_raises_audit_integrity_error_and_archives(self, tmp_path) -> None:
        """Tier 1 anomaly: blob_ref must be the composer-written UUID string."""
        app, service, blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")

        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "options": {"blob_ref": 123, "path": "/data/x.csv"},
                },
                is_valid=True,
            ),
            provenance="session_seed",
        )

        current_state = await service.get_current_state(session.id)
        assert current_state is not None
        msg = await service.add_message(
            session.id,
            "user",
            "Hello",
            composition_state_id=current_state.id,
            writer_principal="route_user_message",
        )

        await blob_service.create_blob(
            session.id,
            "data.csv",
            b"a,b\n1,2",
            "text/csv",
        )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Hello edited",
            },
        )
        assert response.status_code == 500

        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_fork_non_quota_blob_error_archives_session(self, tmp_path) -> None:
        """Non-quota blob failures during fork must archive the new session.

        copy_blobs_for_fork can fail for reasons other than quota (missing
        blob row, filesystem error, DB disconnect). The fork route must fail
        the operation, retain its hidden archived child and frozen plan, and
        compensate only copied child blobs.
        """
        app, service, blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")
        await blob_service.create_blob(session.id, "data.csv", b"a,b\n1,2", "text/csv")
        msg = await service.add_message(session.id, "user", "Go", writer_principal="route_user_message")

        # Use raise_server_exceptions=False so the 500 is returned as an
        # HTTP response rather than propagated as a Python exception.
        client = TestClient(app, raise_server_exceptions=False)

        async def fail_copy_blobs_for_fork(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("disk I/O error")

        with patch.object(
            blob_service,
            "copy_blobs_for_fork",
            new=fail_copy_blobs_for_fork,
        ):
            response = client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "operation_id": str(uuid.uuid4()),
                    "from_message_id": str(msg.id),
                    "new_message_content": "Go edited",
                },
            )

        assert response.status_code == 500

        # The retained failed child is hidden from the public list.
        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1  # Only the original remains

    @pytest.mark.asyncio
    async def test_fork_state_rewrite_failure_archives_session(self, tmp_path) -> None:
        """Failure during state rewrite after blob copy must archive the fork.

        If settlement fails after staging and blob copy, the copied blobs must
        be compensated while the hidden archived child and frozen plan remain
        as failure evidence.
        """
        app, service, blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")

        # Save a state with a blob_ref so the rewrite path is triggered
        blob = await blob_service.create_blob(
            session.id,
            "data.csv",
            b"a,b\n1,2",
            "text/csv",
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={
                    "plugin": "csv",
                    "options": {"blob_ref": str(blob.id), "path": blob.storage_path},
                },
                is_valid=True,
            ),
            provenance="session_seed",
        )

        current_state = await service.get_current_state(session.id)
        assert current_state is not None
        msg = await service.add_message(
            session.id,
            "user",
            "Go",
            composition_state_id=current_state.id,
            writer_principal="route_user_message",
        )

        # Use raise_server_exceptions=False so the 500 is returned as an
        # HTTP response rather than propagated as a Python exception.
        client = TestClient(app, raise_server_exceptions=False)

        async def fail_settle_guided_fork_operation(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DB write failed")

        with patch.object(
            service,
            "settle_guided_fork_operation",
            new=fail_settle_guided_fork_operation,
        ):
            response = client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "operation_id": str(uuid.uuid4()),
                    "from_message_id": str(msg.id),
                    "new_message_content": "Go edited",
                },
            )

        assert response.status_code == 500

        # The retained failed child is hidden from the public list.
        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1  # Only the original remains

    @pytest.mark.asyncio
    async def test_fork_cleanup_failure_preserves_primary_exception_with_note(self, tmp_path) -> None:
        """Fork-blob compensation failure must not mask the real cause.

        If copy_blobs_for_fork raises and cleanup_blobs_for_fork also fails,
        the operator must still see the original blob-copy failure as the
        headline. A RecoveryFailed[...] note identifies copied blob residue
        on the retained archived evidence child for manual cleanup.

        Without this guarantee, a rare cleanup failure would replace the
        true root cause in tracebacks, sending operators down the wrong
        investigation path.
        """
        app, service, blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")
        await blob_service.create_blob(session.id, "data.csv", b"a,b\n1,2", "text/csv")
        msg = await service.add_message(session.id, "user", "Go", writer_principal="route_user_message")

        primary = RuntimeError("disk I/O error during blob copy")
        cleanup = OSError("permission denied removing blob dir")

        # Default raise_server_exceptions=True propagates the exact
        # exception object so __notes__ is inspectable.
        client = TestClient(app)

        async def fail_copy_blobs_for_fork(*args: Any, **kwargs: Any) -> None:
            raise primary

        async def fail_cleanup_blobs_for_fork(*args: Any, **kwargs: Any) -> None:
            raise cleanup

        with (
            patch.object(
                blob_service,
                "copy_blobs_for_fork",
                new=fail_copy_blobs_for_fork,
            ),
            patch.object(
                blob_service,
                "cleanup_blobs_for_fork",
                new=fail_cleanup_blobs_for_fork,
            ),
        ):
            response = client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "operation_id": str(uuid.uuid4()),
                    "from_message_id": str(msg.id),
                    "new_message_content": "Go edited",
                },
            )

        assert response.status_code == 500

        # RecoveryFailed[...] note identifies residual copied-blob custody.
        notes = getattr(primary, "__notes__", [])
        assert any("RecoveryFailed[OSError]" in note for note in notes), f"expected RecoveryFailed[OSError] note, got: {notes!r}"
        assert any("permission denied removing blob dir" in note for note in notes)
        assert any("fork blob cleanup failed" in note.lower() for note in notes)

    @pytest.mark.asyncio
    async def test_fork_top_level_blob_ref_without_copied_blob_fails_closed(self, tmp_path) -> None:
        """sources[].options.blob_ref must be guarded even when blob_map is empty.

        When copy_blobs_for_fork returns {} (because the referenced blob was
        deleted from the source session before the fork), the per-source loop
        was gated on ``blob_map`` being non-empty.  That made the inner guard
        at ``if old_uuid not in blob_map: raise AuditIntegrityError(...)``
        unreachable in exactly the case it protects: a source whose options
        carry a stale blob_ref with no corresponding copied blob.

        The fork must fail-closed with AuditIntegrityError rather than
        silently carrying the stale cross-session blob_ref into the forked
        session. The failed archived child and plan remain hidden evidence,
        and the visible session list stays unchanged.
        """
        app, service, _blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")

        # Persist a composition state whose source options carry a blob_ref
        # for a blob that does NOT exist (simulating a deleted blob).
        # No actual blob is created → copy_blobs_for_fork returns {}.
        missing_blob_id = uuid.uuid4()
        source_state = await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources={
                    "my_csv": {
                        "plugin": "csv",
                        "options": {
                            "blob_ref": str(missing_blob_id),
                            "path": "/data/deleted.csv",
                        },
                    }
                },
                is_valid=True,
            ),
            provenance="session_seed",
        )
        msg = await service.add_message(
            session.id,
            "user",
            "Process this",
            composition_state_id=source_state.id,
            writer_principal="route_user_message",
        )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "operation_id": str(uuid.uuid4()),
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )
        assert response.status_code == 500

        # The partial fork session must have been archived so the list
        # length is unchanged.
        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1
