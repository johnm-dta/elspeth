from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import structlog
from sqlalchemy import insert, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from elspeth.web.blobs.protocol import BlobNotFoundError, BlobPendingProposalError
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    composition_proposals_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


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


def _insert_session(conn, session_id: str) -> None:
    conn.execute(
        insert(sessions_table).values(
            id=session_id,
            user_id="alice",
            auth_provider_type="local",
            title="Composer UX",
            trust_mode="explicit_approve",
            density_default="high",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )


def test_session_preferences_columns_exist(engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("sessions")}
    assert {"trust_mode", "density_default", "interpretation_review_disabled"} <= columns


def test_proposal_tables_exist(engine) -> None:
    table_names = set(inspect(engine).get_table_names())
    assert "composition_proposals" in table_names
    assert "proposal_events" in table_names


def test_composition_proposal_status_is_closed(engine) -> None:
    session_id = str(uuid4())
    with engine.begin() as conn:
        _insert_session(conn, session_id)
        with pytest.raises(IntegrityError, match="ck_composition_proposals_status"):
            conn.execute(
                insert(composition_proposals_table).values(
                    id=str(uuid4()),
                    session_id=session_id,
                    tool_call_id="call_1",
                    tool_name="set_pipeline",
                    status="half_done",
                    summary="Add a pipeline",
                    rationale="Requested by the user",
                    affects=["graph", "validation"],
                    arguments_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
                    arguments_redacted_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
                    base_state_id=None,
                    committed_state_id=None,
                    audit_event_id=None,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )


def test_proposal_event_type_is_closed(engine) -> None:
    session_id = str(uuid4())
    with engine.begin() as conn:
        _insert_session(conn, session_id)
        with pytest.raises(IntegrityError, match="ck_proposal_events_type"):
            conn.execute(
                insert(proposal_events_table).values(
                    id=str(uuid4()),
                    session_id=session_id,
                    proposal_id=None,
                    event_type="proposal.maybe",
                    actor="user:alice",
                    payload={"status": "unknown"},
                    created_at=datetime.now(UTC),
                )
            )


def test_default_session_preferences_are_inserted_by_database(engine) -> None:
    session_id = str(uuid4())
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="alice",
                auth_provider_type="local",
                title="Defaults",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        row = conn.execute(
            select(
                sessions_table.c.trust_mode,
                sessions_table.c.density_default,
            ).where(sessions_table.c.id == session_id)
        ).one()

    # Default trust_mode is auto_commit (commit c4e2f69cd reverted the
    # explicit_approve default — see the comment block on
    # sessions_table.trust_mode in models.py for the rationale).
    assert row.trust_mode == "auto_commit"
    assert row.density_default == "high"


@pytest.mark.asyncio
async def test_get_composer_preferences_returns_defaults(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    prefs = await service.get_composer_preferences(session_id)

    assert str(prefs.session_id) == str(session_id)
    assert prefs.trust_mode == "explicit_approve"
    assert prefs.density_default == "high"
    assert prefs.interpretation_review_disabled is False


@pytest.mark.asyncio
async def test_update_trust_mode_writes_audit_event_before_return(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    transition = await service.update_composer_preferences(
        session_id,
        trust_mode="auto_commit",
        density_default="medium",
        actor="user:alice",
    )

    # B2 (Phase 8a-2): service returns a transition wrapper exposing
    # both prior and current state. ``current`` is the post-write
    # record; ``prior`` reflects the row state observed inside the same
    # transaction *before* the UPDATE — here the row's seed values from
    # ``_insert_session`` (trust_mode='explicit_approve',
    # density_default='high').
    assert transition.current.trust_mode == "auto_commit"
    assert transition.current.density_default == "medium"
    assert transition.current.interpretation_review_disabled is False
    assert transition.prior.trust_mode == "explicit_approve"
    assert transition.prior.density_default == "high"
    assert transition.prior.interpretation_review_disabled is False
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == ["trust_mode.changed"]
    # B1 (Phase 8a-1): payload now carries ``prior_trust_mode`` alongside
    # the existing keys. The strict-equality assertion is preserved —
    # the post-extension payload is the new Tier-1 schema.
    assert events[0].payload == {
        "trust_mode": "auto_commit",
        "prior_trust_mode": "explicit_approve",
        "density_default": "medium",
    }


@pytest.mark.asyncio
async def test_update_trust_mode_no_op_returns_prior_equal_current(service) -> None:
    """B2 contract: a write that does not change the field still returns
    a (prior, current) pair where ``prior == current`` for the unchanged
    fields. Verifies the prior-load runs unconditionally inside the
    transaction rather than being skipped on no-op writes."""
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    # The session seed has trust_mode='explicit_approve'; PATCH back
    # the same value.
    transition = await service.update_composer_preferences(
        session_id,
        trust_mode="explicit_approve",
        density_default="high",
        actor="user:alice",
    )

    assert transition.prior.trust_mode == transition.current.trust_mode == "explicit_approve"
    assert transition.prior.density_default == transition.current.density_default == "high"
    events = await service.list_proposal_events(session_id)
    # The audit row is still written — ``trust_mode.changed`` records
    # the PATCH event, not a transition delta; the recorded
    # ``prior_trust_mode`` makes the no-op explicit downstream.
    assert events[0].payload == {
        "trust_mode": "explicit_approve",
        "prior_trust_mode": "explicit_approve",
        "density_default": "high",
    }


@pytest.mark.asyncio
async def test_create_composition_proposal_writes_created_event(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_set_pipeline",
        tool_name="set_pipeline",
        summary="Replace the pipeline with one source and one sink.",
        rationale="Requested by the user.",
        affects=("graph", "validation"),
        arguments_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        arguments_redacted_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        base_state_id=None,
        actor="composer-web:user-alice",
    )

    assert proposal.status == "pending"
    assert proposal.affects == ("graph", "validation")
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == ["proposal.created"]
    assert str(events[0].proposal_id) == str(proposal.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["missing", "cross_session", "pending"])
async def test_create_composition_proposal_rejects_unusable_blob_reference(engine, service, tmp_path, failure_kind) -> None:
    session_id = uuid4()
    other_session_id = uuid4()
    with engine.begin() as conn:
        _insert_session(conn, str(session_id))
        _insert_session(conn, str(other_session_id))

    blob_id = uuid4()
    if failure_kind != "missing":
        owner = other_session_id if failure_kind == "cross_session" else session_id
        blob_service = BlobServiceImpl(engine, tmp_path)
        if failure_kind == "pending":
            record = await blob_service.create_pending_blob(
                session_id=owner,
                filename="proposal.csv",
                mime_type="text/csv",
                created_by="assistant",
            )
        else:
            record = await blob_service.create_blob(
                session_id=owner,
                filename="proposal.csv",
                content=b"value\n1\n",
                mime_type="text/csv",
                created_by="assistant",
            )
        blob_id = record.id

    with pytest.raises(ValueError, match="blob"):
        await service.create_composition_proposal(
            session_id=session_id,
            tool_call_id=f"call_{failure_kind}",
            tool_name="set_source_from_blob",
            summary="Use the blob as source.",
            rationale="Requested by the user.",
            affects=("source",),
            arguments_json={"blob_id": str(blob_id)},
            arguments_redacted_json={"blob_id": str(blob_id)},
            base_state_id=None,
            actor="composer-web:user-alice",
        )

    assert await service.list_composition_proposals(session_id) == []


@pytest.mark.asyncio
async def test_create_composition_proposal_accepts_owned_ready_blob(engine, service, tmp_path) -> None:
    session_id = uuid4()
    with engine.begin() as conn:
        _insert_session(conn, str(session_id))
    blob_service = BlobServiceImpl(engine, tmp_path)
    record = await blob_service.create_blob(
        session_id=session_id,
        filename="proposal.csv",
        content=b"value\n1\n",
        mime_type="text/csv",
        created_by="assistant",
    )

    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_ready_blob",
        tool_name="set_source_from_blob",
        summary="Use the blob as source.",
        rationale="Requested by the user.",
        affects=("source",),
        arguments_json={"blob_id": str(record.id)},
        arguments_redacted_json={"blob_id": str(record.id)},
        base_state_id=None,
        actor="composer-web:user-alice",
    )

    assert proposal.status == "pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["proposal", "delete"])
async def test_proposal_blob_validation_and_delete_share_one_serial_order(tmp_path, monkeypatch, winner) -> None:
    """A deterministic barrier proves the lock closes both race outcomes."""
    engine = create_session_engine(f"sqlite:///{tmp_path / 'race.db'}")
    initialize_session_schema(engine)
    session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    blob_service = BlobServiceImpl(engine, tmp_path)
    session_id = uuid4()
    with engine.begin() as conn:
        _insert_session(conn, str(session_id))
    blob = await blob_service.create_blob(
        session_id=session_id,
        filename="race.csv",
        content=b"value\n1\n",
        mime_type="text/csv",
        created_by="assistant",
    )

    entered = threading.Event()
    release = threading.Event()

    async def create_proposal():
        return await session_service.create_composition_proposal(
            session_id=session_id,
            tool_call_id=f"call_{winner}",
            tool_name="set_source_from_blob",
            summary="Use the blob as source.",
            rationale="Requested by the user.",
            affects=("source",),
            arguments_json={"blob_id": str(blob.id)},
            arguments_redacted_json={"blob_id": str(blob.id)},
            base_state_id=None,
            actor="composer-web:user-alice",
        )

    if winner == "proposal":
        from elspeth.web.sessions import service as service_module

        original_validate = service_module.validate_proposal_blob_references

        def blocked_validate(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("proposal race barrier timed out")
            return original_validate(*args, **kwargs)

        monkeypatch.setattr(service_module, "validate_proposal_blob_references", blocked_validate)
        proposal_task = asyncio.create_task(create_proposal())
        assert await asyncio.to_thread(entered.wait, 5)
        delete_task = asyncio.create_task(blob_service.delete_blob(blob.id))
        await asyncio.sleep(0)
        release.set()

        proposal = await proposal_task
        with pytest.raises(BlobPendingProposalError):
            await delete_task
        assert proposal.status == "pending"
        assert await blob_service.get_blob(blob.id) == blob
        return

    from elspeth.web.blobs import service as blob_service_module

    original_pending = blob_service_module.pending_proposal_reference_id

    def blocked_pending(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("delete race barrier timed out")
        return original_pending(*args, **kwargs)

    monkeypatch.setattr(blob_service_module, "pending_proposal_reference_id", blocked_pending)
    delete_task = asyncio.create_task(blob_service.delete_blob(blob.id))
    assert await asyncio.to_thread(entered.wait, 5)
    proposal_task = asyncio.create_task(create_proposal())
    await asyncio.sleep(0)
    release.set()

    await delete_task
    with pytest.raises(ValueError, match="does not exist"):
        await proposal_task
    with pytest.raises(BlobNotFoundError):
        await blob_service.get_blob(blob.id)
    assert await session_service.list_composition_proposals(session_id) == []


@pytest.mark.asyncio
async def test_reject_composition_proposal_is_forward_only(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_set_pipeline",
        tool_name="set_pipeline",
        summary="Replace the pipeline.",
        rationale="Requested by the user.",
        affects=("graph",),
        arguments_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        arguments_redacted_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        base_state_id=None,
        actor="composer-web:user-alice",
    )

    rejected = await service.reject_composition_proposal(
        session_id=session_id,
        proposal_id=proposal.id,
        actor="user:alice",
    )

    assert rejected.status == "rejected"
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == [
        "proposal.created",
        "proposal.rejected",
    ]


@pytest.mark.asyncio
async def test_accept_composition_proposal_requires_pending_status(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_set_pipeline",
        tool_name="set_pipeline",
        summary="Replace the pipeline.",
        rationale="Requested by the user.",
        affects=("graph",),
        arguments_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        arguments_redacted_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        base_state_id=None,
        actor="composer-web:user-alice",
    )
    await service.reject_composition_proposal(
        session_id=session_id,
        proposal_id=proposal.id,
        actor="user:alice",
    )

    with pytest.raises(ValueError, match="pending"):
        await service.mark_composition_proposal_committed(
            session_id=session_id,
            proposal_id=proposal.id,
            committed_state_id=uuid4(),
            actor="user:alice",
        )


@pytest.mark.asyncio
async def test_mark_composition_proposal_committed_writes_forward_event(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
    state_record = await service.save_composition_state(
        session_id,
        CompositionStateData(is_valid=True),
        provenance="tool_call",
    )
    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_set_pipeline",
        tool_name="set_pipeline",
        summary="Replace the pipeline.",
        rationale="Requested by the user.",
        affects=("graph",),
        arguments_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        arguments_redacted_json={"sources": {"primary": {"plugin": "csv", "options": {}}}},
        base_state_id=None,
        actor="composer-web:user-alice",
    )

    committed = await service.mark_composition_proposal_committed(
        session_id=session_id,
        proposal_id=proposal.id,
        committed_state_id=state_record.id,
        actor="user:alice",
    )

    assert committed.status == "committed"
    assert committed.committed_state_id == state_record.id
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == [
        "proposal.created",
        "proposal.accepted",
    ]
    assert events[-1].payload == {"committed_state_id": str(state_record.id)}
