"""Fenced session-fork staging, takeover, settlement, and archive exclusion."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import delete, event, func, insert, select, update
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.blobs.protocol import BlobForkWriteFence, BlobInProgressForkError, fork_blob_id
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, chat_messages_table, guided_operations_table, sessions_table
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    GuidedForkSettlementCommand,
    GuidedOperationActive,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationFailed,
    GuidedOperationTakenOver,
    GuidedSessionResult,
    SessionGuidedOperationInProgressError,
    SessionNotFoundError,
)
from elspeth.web.sessions.routes.sessions import _rewrite_fork_state_blob_custody
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl, _fork_blob_plan_from_content
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@pytest.fixture()
def engine():
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    return engine


@pytest.fixture()
def service(engine) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )


@pytest.fixture(params=("sqlite", "postgres"))
def durable_engine(request: pytest.FixtureRequest, tmp_path: Path):
    """Exercise lock races against production-shaped file SQLite and opt-in PG."""

    if request.param == "postgres":
        url = os.environ.get("ELSPETH_TEST_POSTGRES_URL")
        if url is None:
            pytest.skip("ELSPETH_TEST_POSTGRES_URL is required for the PostgreSQL fork race matrix")
        race_engine = create_session_engine(url)
    else:
        race_engine = create_session_engine(f"sqlite:///{tmp_path / 'fork-races.db'}")
    initialize_session_schema(race_engine)
    try:
        yield race_engine
    finally:
        race_engine.dispose()


def _service_for(engine: Any) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.fork-race"),
    )


async def _service_lock_contention(
    first_service: SessionServiceImpl,
    second_service: SessionServiceImpl,
    session_id: UUID,
    first: Callable[[], Awaitable[Any]],
    second: Callable[[], Awaitable[Any]],
) -> tuple[Any, Any]:
    """Pause the winner while it holds the real session lock, then contend."""

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
        first_task = asyncio.create_task(first())
        await asyncio.to_thread(held.wait, 5)
        second_task = asyncio.create_task(second())
        assert await asyncio.to_thread(contender_waiting.wait, 5)
        was_blocked = not contender_acquired.is_set()
        await asyncio.to_thread(release.wait, 5)
        results = tuple(await asyncio.gather(first_task, second_task, return_exceptions=True))
        assert was_blocked
        return results  # type: ignore[return-value]


async def _blob_delete_first_contention(
    reserve_service: SessionServiceImpl,
    session_id: UUID,
    delete_first: Callable[[], Awaitable[Any]],
    reserve_second: Callable[[], Awaitable[Any]],
) -> tuple[Any, Any]:
    """Hold blob deletion's source lock while fork reservation contends on it."""

    from elspeth.web.blobs import service as blob_service_module

    original_blob_lock = blob_service_module.locked_session_transaction
    original_service_begin = reserve_service._session_process_locked_begin
    original_service_lock = reserve_service._session_write_lock
    held = threading.Barrier(2)
    release = threading.Barrier(2)
    reserve_waiting = threading.Event()
    reserve_acquired = threading.Event()

    @contextlib.contextmanager
    def controlled_blob_lock(engine: Any, locked_session_id: str):
        with original_blob_lock(engine, locked_session_id) as conn:
            if locked_session_id == str(session_id):
                held.wait(timeout=5)
                release.wait(timeout=5)
            yield conn

    @contextlib.contextmanager
    def observed_service_begin(locked_session_id: str):
        if locked_session_id == str(session_id):
            reserve_waiting.set()
        with original_service_begin(locked_session_id) as conn:
            yield conn

    @contextlib.contextmanager
    def observed_service_lock(conn: Any, locked_session_id: str):
        with original_service_lock(conn, locked_session_id):
            if locked_session_id == str(session_id):
                reserve_acquired.set()
            yield

    with (
        patch.object(blob_service_module, "locked_session_transaction", new=controlled_blob_lock),
        patch.object(reserve_service, "_session_process_locked_begin", new=observed_service_begin),
        patch.object(reserve_service, "_session_write_lock", new=observed_service_lock),
    ):
        delete_task = asyncio.create_task(delete_first())
        await asyncio.to_thread(held.wait, 5)
        reserve_task = asyncio.create_task(reserve_second())
        assert await asyncio.to_thread(reserve_waiting.wait, 5)
        was_blocked = not reserve_acquired.is_set()
        await asyncio.to_thread(release.wait, 5)
        results = tuple(await asyncio.gather(delete_task, reserve_task, return_exceptions=True))
        assert was_blocked
        return results  # type: ignore[return-value]


async def _fork_first_blob_contention(
    fork_service: SessionServiceImpl,
    session_id: UUID,
    fork_first: Callable[[], Awaitable[Any]],
    delete_second: Callable[[], Awaitable[Any]],
) -> tuple[Any, Any]:
    """Hold fork reservation's source lock while blob deletion contends on it."""

    from elspeth.web.blobs import service as blob_service_module

    original_blob_lock = blob_service_module.locked_session_transaction
    original_service_lock = fork_service._session_write_lock
    held = threading.Barrier(2)
    release = threading.Barrier(2)
    delete_waiting = threading.Event()
    delete_acquired = threading.Event()
    paused = False

    @contextlib.contextmanager
    def controlled_service_lock(conn: Any, locked_session_id: str):
        nonlocal paused
        with original_service_lock(conn, locked_session_id):
            if locked_session_id == str(session_id) and not paused:
                paused = True
                held.wait(timeout=5)
                release.wait(timeout=5)
            yield

    @contextlib.contextmanager
    def observed_blob_lock(engine: Any, locked_session_id: str):
        if locked_session_id == str(session_id):
            delete_waiting.set()
        with original_blob_lock(engine, locked_session_id) as conn:
            if locked_session_id == str(session_id):
                delete_acquired.set()
            yield conn

    with (
        patch.object(fork_service, "_session_write_lock", new=controlled_service_lock),
        patch.object(blob_service_module, "locked_session_transaction", new=observed_blob_lock),
    ):
        fork_task = asyncio.create_task(fork_first())
        await asyncio.to_thread(held.wait, 5)
        delete_task = asyncio.create_task(delete_second())
        assert await asyncio.to_thread(delete_waiting.wait, 5)
        was_blocked = not delete_acquired.is_set()
        await asyncio.to_thread(release.wait, 5)
        results = tuple(await asyncio.gather(fork_task, delete_task, return_exceptions=True))
        assert was_blocked
        return results  # type: ignore[return-value]


def _cleanup_race_user(engine: Any, user_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(delete(sessions_table).where(sessions_table.c.user_id == user_id))


async def _claim_fork(service: SessionServiceImpl, parent_id: UUID, *, operation_id: str | None = None):
    claimed = await service.reserve_guided_operation(
        session_id=parent_id,
        operation_id=operation_id or str(uuid4()),
        kind="session_fork",
        request_hash="a" * 64,
        actor="composer_route",
        lease_seconds=300,
    )
    assert type(claimed) in {GuidedOperationClaimed, GuidedOperationTakenOver}
    return claimed.fence


async def _parent_with_fork_message(service: SessionServiceImpl):
    parent = await service.create_session("alice", "Parent", "local")
    await service.add_message(parent.id, "user", "root", writer_principal="route_user_message")
    fork_message = await service.add_message(
        parent.id,
        "user",
        "fork here",
        writer_principal="route_user_message",
    )
    return parent, fork_message


def _insert_blob_row(
    engine,
    *,
    blob_id: UUID,
    session_id: UUID,
    content_hash: str = "c" * 64,
    size_bytes: int = 3,
    status: str = "ready",
    storage_path: str | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(blobs_table).values(
                id=str(blob_id),
                session_id=str(session_id),
                filename=f"{blob_id}.bin",
                mime_type="application/octet-stream",
                size_bytes=size_bytes,
                content_hash=content_hash,
                storage_path=storage_path or f"/tmp/{blob_id}.bin",
                created_at=datetime.now(UTC),
                created_by="user",
                source_description=None,
                status=status,
                creation_modality="verbatim",
            )
        )


@pytest.mark.parametrize("noncanonical", ["upper", "braced"])
def test_frozen_blob_plan_rejects_noncanonical_uuid_spellings(noncanonical: str) -> None:
    source_session_id = uuid4()
    child_session_id = uuid4()
    source_blob_id = uuid4()
    target_blob_id = fork_blob_id(target_session_id=child_session_id, source_blob_id=source_blob_id)

    def spelling(value: UUID) -> str:
        return str(value).upper() if noncanonical == "upper" else f"{{{value}}}"

    content = json.dumps(
        {
            "schema": "session-fork-blob-plan.v1",
            "source_session_id": str(source_session_id),
            "child_session_id": str(child_session_id),
            "operation_id": "fork-operation",
            "source_blobs": [
                {
                    "source_blob_id": spelling(source_blob_id),
                    "target_blob_id": spelling(target_blob_id),
                    "content_hash": "a" * 64,
                    "size_bytes": 1,
                }
            ],
        }
    )

    with pytest.raises(AuditIntegrityError, match="non-canonical blob id"):
        _fork_blob_plan_from_content(
            content,
            expected_source_session_id=source_session_id,
            expected_child_session_id=child_session_id,
            expected_operation_id="fork-operation",
        )


@pytest.mark.asyncio
async def test_fork_stages_one_hidden_bound_child_and_takeover_reuses_it(service, engine) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    operation_id = str(uuid4())
    first_fence = await _claim_fork(service, parent.id, operation_id=operation_id)

    first = await service.fork_session(
        first_fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )

    assert first.session.archived_at is not None
    assert first.state is None
    assert first.messages[-1].content == "edited"
    assert [session.id for session in await service.list_sessions("alice", "local", include_archived=True)] == [parent.id]
    with engine.begin() as conn:
        conn.execute(
            update(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(parent.id),
                guided_operations_table.c.operation_id == operation_id,
            )
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    takeover_fence = await _claim_fork(service, parent.id, operation_id=operation_id)

    second = await service.fork_session(
        takeover_fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )

    assert second.session.id == first.session.id
    assert [message.id for message in second.messages] == [message.id for message in first.messages]
    assert second.state == first.state
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(sessions_table)).scalar_one() == 2


@pytest.mark.asyncio
async def test_takeover_fails_closed_when_bound_child_lineage_drifted(service, engine) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(
        fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )
    with engine.begin() as conn:
        conn.execute(
            update(sessions_table).where(sessions_table.c.id == str(staged.session.id)).values(forked_from_session_id=str(uuid4()))
        )

    with pytest.raises(AuditIntegrityError, match="bound child"):
        await service.fork_session(
            fence,
            fork_message_id=fork_message.id,
            new_message_content="edited",
        )


@pytest.mark.asyncio
async def test_settlement_rewrites_state_activates_child_and_completes_locator_atomically(service) -> None:
    parent = await service.create_session("alice", "Parent", "local")
    original_state = await service.save_composition_state(
        parent.id,
        CompositionStateData(
            sources={"orders": {"plugin": "csv", "options": {"blob_ref": str(uuid4())}}},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fork_message = await service.add_message(
        parent.id,
        "user",
        "fork here",
        composition_state_id=original_state.id,
        writer_principal="route_user_message",
    )
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(
        fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )
    assert staged.state is not None
    rewritten_blob_id = uuid4()
    response_hash = "b" * 64

    settled = await service.settle_guided_fork_operation(
        GuidedForkSettlementCommand(
            fence=fence,
            child_session_id=staged.session.id,
            expected_current_state_id=staged.state.id,
            edited_message_id=staged.messages[-1].id,
            rewritten_state_id=uuid4(),
            rewritten_state=CompositionStateData(
                sources={"orders": {"plugin": "csv", "options": {"blob_ref": str(rewritten_blob_id)}}},
                is_valid=True,
            ),
            response_hash=response_hash,
            actor="composer_route",
        )
    )

    assert settled.id == staged.session.id
    assert settled.archived_at is None
    current_state = await service.get_current_state(staged.session.id)
    assert current_state is not None
    assert current_state.version == 1
    assert current_state.sources["orders"]["options"]["blob_ref"] == str(rewritten_blob_id)
    child_messages = await service.get_messages(staged.session.id, limit=None)
    edited_message = next(message for message in child_messages if message.id == staged.messages[-1].id)
    assert edited_message.composition_state_id == current_state.id
    operation = await service.get_guided_operation(
        session_id=parent.id,
        operation_id=fence.operation_id,
        kind="session_fork",
        request_hash="a" * 64,
    )
    assert operation == GuidedOperationCompleted(
        result=GuidedSessionResult(session_id=staged.session.id),
        response_hash=response_hash,
    )


@pytest.mark.asyncio
async def test_settlement_rejects_missing_retained_frozen_blob_plan(service, engine) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(fence, fork_message_id=fork_message.id, new_message_content="edited")
    with engine.begin() as conn:
        conn.execute(
            update(chat_messages_table)
            .where(
                chat_messages_table.c.session_id == str(staged.session.id),
                chat_messages_table.c.role == "audit",
                chat_messages_table.c.writer_principal == "session_fork",
            )
            .values(writer_principal="route_system_message")
        )

    with pytest.raises(AuditIntegrityError, match="exactly one retained frozen blob plan"):
        await service.settle_guided_fork_operation(
            GuidedForkSettlementCommand(
                fence=fence,
                child_session_id=staged.session.id,
                expected_current_state_id=None,
                edited_message_id=staged.messages[-1].id,
                rewritten_state_id=None,
                rewritten_state=None,
                response_hash="b" * 64,
                actor="composer_route",
            )
        )

    assert (await service.get_session(staged.session.id)).archived_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["missing", "extra", "pending", "hash", "size"])
async def test_settlement_requires_exact_ready_child_blob_cohort(service, engine, drift: str) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    parent_blob_id = uuid4()
    _insert_blob_row(engine, blob_id=parent_blob_id, session_id=parent.id)
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(fence, fork_message_id=fork_message.id, new_message_content="edited")
    assert len(staged.blob_plan) == 1
    expected = staged.blob_plan[0]
    if drift != "missing":
        _insert_blob_row(
            engine,
            blob_id=expected.target_blob_id,
            session_id=staged.session.id,
            content_hash="d" * 64 if drift == "hash" else expected.content_hash,
            size_bytes=expected.size_bytes + 1 if drift == "size" else expected.size_bytes,
            status="pending" if drift == "pending" else "ready",
        )
    if drift == "extra":
        _insert_blob_row(engine, blob_id=uuid4(), session_id=staged.session.id)

    with pytest.raises(AuditIntegrityError, match="child blob"):
        await service.settle_guided_fork_operation(
            GuidedForkSettlementCommand(
                fence=fence,
                child_session_id=staged.session.id,
                expected_current_state_id=None,
                edited_message_id=staged.messages[-1].id,
                rewritten_state_id=None,
                rewritten_state=None,
                response_hash="b" * 64,
                actor="composer_route",
            )
        )

    assert (await service.get_session(staged.session.id)).archived_at is not None
    operation = await service.get_guided_operation(
        session_id=parent.id,
        operation_id=fence.operation_id,
        kind="session_fork",
        request_hash="a" * 64,
    )
    assert isinstance(operation, GuidedOperationActive)


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_kind", ["id", "sentinel", "storage_path"])
async def test_settlement_rejects_rewritten_state_with_parent_blob_custody(
    service,
    engine,
    reference_kind: str,
) -> None:
    parent = await service.create_session("alice", "Parent", "local")
    parent_blob_id = uuid4()
    parent_storage_path = f"/tmp/{parent_blob_id}.bin"
    _insert_blob_row(
        engine,
        blob_id=parent_blob_id,
        session_id=parent.id,
        storage_path=parent_storage_path,
    )
    original_state = await service.save_composition_state(
        parent.id,
        CompositionStateData(sources={"orders": {"plugin": "csv", "options": {"path": "old.csv"}}}),
        provenance="session_seed",
    )
    fork_message = await service.add_message(
        parent.id,
        "user",
        "fork here",
        composition_state_id=original_state.id,
        writer_principal="route_user_message",
    )
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(fence, fork_message_id=fork_message.id, new_message_content="edited")
    assert staged.state is not None and len(staged.blob_plan) == 1
    expected = staged.blob_plan[0]
    _insert_blob_row(
        engine,
        blob_id=expected.target_blob_id,
        session_id=staged.session.id,
        content_hash=expected.content_hash,
        size_bytes=expected.size_bytes,
    )
    stale_reference = {
        "id": str(parent_blob_id),
        "sentinel": f"blob:{parent_blob_id}",
        "storage_path": parent_storage_path,
    }[reference_kind]

    with pytest.raises(AuditIntegrityError, match="retains parent blob custody"):
        await service.settle_guided_fork_operation(
            GuidedForkSettlementCommand(
                fence=fence,
                child_session_id=staged.session.id,
                expected_current_state_id=staged.state.id,
                edited_message_id=staged.messages[-1].id,
                rewritten_state_id=uuid4(),
                rewritten_state=CompositionStateData(sources={"orders": {"plugin": "csv", "options": {"blob_ref": stale_reference}}}),
                response_hash="b" * 64,
                actor="composer_route",
            )
        )

    assert (await service.get_session(staged.session.id)).archived_at is not None


@pytest.mark.asyncio
async def test_settlement_rejects_parent_blob_reference_excluded_from_ready_plan(service, engine) -> None:
    parent = await service.create_session("alice", "Parent", "local")
    pending_parent_blob_id = uuid4()
    _insert_blob_row(
        engine,
        blob_id=pending_parent_blob_id,
        session_id=parent.id,
        status="pending",
    )
    original_state = await service.save_composition_state(
        parent.id,
        CompositionStateData(
            sources={
                "orders": {
                    "plugin": "csv",
                    "options": {"blob_ref": str(pending_parent_blob_id)},
                }
            }
        ),
        provenance="session_seed",
    )
    fork_message = await service.add_message(
        parent.id,
        "user",
        "fork here",
        composition_state_id=original_state.id,
        writer_principal="route_user_message",
    )
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(fence, fork_message_id=fork_message.id, new_message_content="edited")
    assert staged.state is not None and staged.blob_plan == ()

    with pytest.raises(AuditIntegrityError, match="retains parent blob custody"):
        await service.settle_guided_fork_operation(
            GuidedForkSettlementCommand(
                fence=fence,
                child_session_id=staged.session.id,
                expected_current_state_id=staged.state.id,
                edited_message_id=staged.messages[-1].id,
                rewritten_state_id=uuid4(),
                rewritten_state=CompositionStateData(
                    sources={
                        "orders": {
                            "plugin": "csv",
                            "options": {"blob_ref": str(pending_parent_blob_id)},
                        }
                    }
                ),
                response_hash="b" * 64,
                actor="composer_route",
            )
        )

    assert (await service.get_session(staged.session.id)).archived_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("fault_point", ["state_replace", "message_repoint", "activation", "operation_complete"])
async def test_settlement_fault_rolls_back_every_surface_and_child_remains_takeover_safe(
    service,
    engine,
    fault_point: str,
) -> None:
    parent = await service.create_session("alice", "Parent", "local")
    state = await service.save_composition_state(
        parent.id,
        CompositionStateData(sources={"source": {"plugin": "csv", "options": {"path": "old.csv"}}}, is_valid=True),
        provenance="session_seed",
    )
    message = await service.add_message(
        parent.id,
        "user",
        "fork",
        composition_state_id=state.id,
        writer_principal="route_user_message",
    )
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(fence, fork_message_id=message.id, new_message_content="edited")
    assert staged.state is not None
    chat_updates = 0

    def inject(_conn, _cursor, statement, _parameters, _context, _executemany):
        nonlocal chat_updates
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("update chat_messages"):
            chat_updates += 1
        should_fail = (
            (fault_point == "state_replace" and normalized.startswith("delete from composition_states"))
            # Settlement rewrites the edited message across two chat_messages
            # UPDATEs (detach-to-NULL -> delete staged state -> insert
            # replacement -> repoint), ordered so the replacement reclaims
            # composition-state version 1. The detach is the 1st ``update
            # chat_messages`` statement and the repoint is the 2nd, so the
            # repoint fault probe targets ``chat_updates == 2``.
            or (fault_point == "message_repoint" and normalized.startswith("update chat_messages") and chat_updates == 2)
            or (fault_point == "activation" and normalized.startswith("update sessions") and "archived_at" in normalized)
            or (fault_point == "operation_complete" and normalized.startswith("update guided_operations") and "status" in normalized)
        )
        if should_fail:
            raise RuntimeError(f"injected {fault_point}")

    event.listen(engine, "before_cursor_execute", inject)
    try:
        with pytest.raises(RuntimeError, match=fault_point):
            await service.settle_guided_fork_operation(
                GuidedForkSettlementCommand(
                    fence=fence,
                    child_session_id=staged.session.id,
                    expected_current_state_id=staged.state.id,
                    edited_message_id=staged.messages[-1].id,
                    rewritten_state_id=uuid4(),
                    rewritten_state=CompositionStateData(
                        sources={"source": {"plugin": "csv", "options": {"path": "new.csv"}}},
                        is_valid=True,
                    ),
                    response_hash="b" * 64,
                    actor="composer_route",
                )
            )
    finally:
        event.remove(engine, "before_cursor_execute", inject)

    retained = await service.get_session(staged.session.id)
    assert retained.archived_at is not None
    retained_state = await service.get_current_state(staged.session.id)
    assert retained_state is not None and retained_state.id == staged.state.id
    retained_messages = await service.get_messages(staged.session.id, limit=None)
    retained_edited = next(item for item in retained_messages if item.id == staged.messages[-1].id)
    assert retained_edited.composition_state_id == staged.state.id
    operation = await service.get_guided_operation(
        session_id=parent.id,
        operation_id=fence.operation_id,
        kind="session_fork",
        request_hash="a" * 64,
    )
    assert isinstance(operation, GuidedOperationActive)
    resumed = await service.fork_session(fence, fork_message_id=message.id, new_message_content="edited")
    assert resumed.session.id == staged.session.id


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("ELSPETH_TEST_POSTGRES_URL"),
    reason="ELSPETH_TEST_POSTGRES_URL is required for the PostgreSQL settlement rollback matrix",
)
async def test_postgres_settlement_fault_matrix() -> None:
    postgres_engine = create_session_engine(os.environ["ELSPETH_TEST_POSTGRES_URL"])
    initialize_session_schema(postgres_engine)
    with postgres_engine.connect() as conn:
        before_ids = {row.id for row in conn.execute(select(sessions_table.c.id)).all()}
    postgres_service = SessionServiceImpl(
        postgres_engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.postgres-fork-settlement"),
    )
    try:
        for fault_point in ("state_replace", "message_repoint", "activation", "operation_complete"):
            await test_settlement_fault_rolls_back_every_surface_and_child_remains_takeover_safe(
                postgres_service,
                postgres_engine,
                fault_point,
            )
    finally:
        with postgres_engine.begin() as conn:
            conn.execute(delete(sessions_table).where(sessions_table.c.id.not_in(before_ids)))
        postgres_engine.dispose()


@pytest.mark.asyncio
async def test_failed_binding_clear_retains_hidden_archived_child_and_plan(service, engine) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(
        fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )
    assert staged.session.id not in {session.id for session in await service.list_sessions("alice", "local", include_archived=True)}

    await service.fail_guided_operation(fence, failure_code="operation_failed", actor="composer_route")

    listed = await service.list_sessions("alice", "local", include_archived=True)
    assert staged.session.id not in {session.id for session in listed}
    assert staged.session.id not in {session.id for session in await service.list_sessions("alice", "local")}
    retained = await service.get_session(staged.session.id)
    assert retained.archived_at is not None
    with engine.connect() as conn:
        audit_rows = conn.execute(
            select(chat_messages_table.c.content).where(
                chat_messages_table.c.session_id == str(staged.session.id),
                chat_messages_table.c.role == "audit",
                chat_messages_table.c.writer_principal == "session_fork",
            )
        ).all()
    assert len(audit_rows) == 1
    assert '"schema":"session-fork-blob-plan.v1"' in audit_rows[0].content


@pytest.mark.asyncio
async def test_archiving_completed_fork_child_soft_archives_due_to_result_binding(service) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    fence = await _claim_fork(service, parent.id)
    staged = await service.fork_session(
        fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )
    await service.settle_guided_fork_operation(
        GuidedForkSettlementCommand(
            fence=fence,
            child_session_id=staged.session.id,
            expected_current_state_id=None,
            edited_message_id=staged.messages[-1].id,
            rewritten_state_id=None,
            rewritten_state=None,
            response_hash="b" * 64,
            actor="composer_route",
        )
    )

    await service.archive_session(staged.session.id)

    retained = await service.get_session(staged.session.id)
    assert retained.archived_at is not None
    assert staged.session.id in {session.id for session in await service.list_sessions("alice", "local", include_archived=True)}


@pytest.mark.asyncio
async def test_archiving_completed_fork_parent_preserves_child_and_terminal_replay(service) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    operation_id = str(uuid4())
    fence = await _claim_fork(service, parent.id, operation_id=operation_id)
    staged = await service.fork_session(
        fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )
    await service.settle_guided_fork_operation(
        GuidedForkSettlementCommand(
            fence=fence,
            child_session_id=staged.session.id,
            expected_current_state_id=None,
            edited_message_id=staged.messages[-1].id,
            rewritten_state_id=None,
            rewritten_state=None,
            response_hash="b" * 64,
            actor="composer_route",
        )
    )

    await service.archive_session(parent.id)
    replay = await service.reserve_guided_operation(
        session_id=parent.id,
        operation_id=operation_id,
        kind="session_fork",
        request_hash="a" * 64,
        actor="composer_route",
        lease_seconds=300,
    )

    assert replay == GuidedOperationCompleted(
        result=GuidedSessionResult(session_id=staged.session.id),
        response_hash="b" * 64,
    )
    assert (await service.get_session(parent.id)).archived_at is not None
    assert staged.session.id in {session.id for session in await service.list_sessions("alice", "local")}


@pytest.mark.asyncio
async def test_archiving_failed_fork_parent_preserves_failed_operation_and_child_evidence(service) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    operation_id = str(uuid4())
    fence = await _claim_fork(service, parent.id, operation_id=operation_id)
    staged = await service.fork_session(
        fence,
        fork_message_id=fork_message.id,
        new_message_content="edited",
    )
    await service.fail_guided_operation(fence, failure_code="operation_failed", actor="composer_route")

    await service.archive_session(parent.id)
    replay = await service.reserve_guided_operation(
        session_id=parent.id,
        operation_id=operation_id,
        kind="session_fork",
        request_hash="a" * 64,
        actor="composer_route",
        lease_seconds=300,
    )

    assert replay == GuidedOperationFailed(failure_code="operation_failed")
    assert (await service.get_session(parent.id)).archived_at is not None
    assert (await service.get_session(staged.session.id)).archived_at is not None


@pytest.mark.asyncio
async def test_archive_parent_rejects_in_progress_fork_under_database_guard(service) -> None:
    parent, _fork_message = await _parent_with_fork_message(service)
    await _claim_fork(service, parent.id)

    with pytest.raises(SessionGuidedOperationInProgressError):
        await service.archive_session(parent.id)

    assert (await service.get_session(parent.id)).id == parent.id


@pytest.mark.asyncio
async def test_reservation_rejects_parent_already_removed_without_operation_or_child(service, engine) -> None:
    parent, _fork_message = await _parent_with_fork_message(service)
    await service.archive_session(parent.id)

    with pytest.raises(SessionNotFoundError):
        await _claim_fork(service, parent.id)

    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(guided_operations_table)).scalar_one() == 0
        assert conn.execute(select(func.count()).select_from(sessions_table)).scalar_one() == 0


@pytest.mark.asyncio
async def test_fork_of_fork_preserves_historical_plan_but_selects_current_binding(service) -> None:
    parent, fork_message = await _parent_with_fork_message(service)
    first_fence = await _claim_fork(service, parent.id)
    first = await service.fork_session(
        first_fence,
        fork_message_id=fork_message.id,
        new_message_content="first edit",
    )
    await service.settle_guided_fork_operation(
        GuidedForkSettlementCommand(
            fence=first_fence,
            child_session_id=first.session.id,
            expected_current_state_id=None,
            edited_message_id=first.messages[-1].id,
            rewritten_state_id=None,
            rewritten_state=None,
            response_hash="b" * 64,
            actor="composer_route",
        )
    )
    second_fork_message = await service.add_message(
        first.session.id,
        "user",
        "fork child",
        writer_principal="route_user_message",
    )
    second_fence = await _claim_fork(service, first.session.id)

    second = await service.fork_session(
        second_fence,
        fork_message_id=second_fork_message.id,
        new_message_content="second edit",
    )

    assert second.session.forked_from_session_id == first.session.id
    assert second.session.archived_at is not None
    assert second.messages[-1].content == "second edit"


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ("archive", "stage"))
async def test_parent_archive_and_fork_staging_serialize_under_lock_contention(
    durable_engine,
    winner: str,
) -> None:
    """The parent lock admits either archive or one hidden staged child, never both."""

    race_service = _service_for(durable_engine)
    other_service = _service_for(durable_engine)
    user_id = f"fork-archive-race-{uuid4()}"
    parent = await race_service.create_session(user_id, "Parent", "local")
    message = await race_service.add_message(
        parent.id,
        "user",
        "fork here",
        writer_principal="route_user_message",
    )
    operation_id = str(uuid4())

    async def archive(target_service: SessionServiceImpl) -> None:
        await target_service.archive_session(parent.id)

    async def reserve_and_stage(target_service: SessionServiceImpl) -> Any:
        fence = await _claim_fork(target_service, parent.id, operation_id=operation_id)
        return await target_service.fork_session(
            fence,
            fork_message_id=message.id,
            new_message_content="edited",
        )

    try:
        if winner == "archive":
            archive_result, stage_result = await _service_lock_contention(
                race_service,
                other_service,
                parent.id,
                lambda: archive(race_service),
                lambda: reserve_and_stage(other_service),
            )
            assert archive_result is None
            assert isinstance(stage_result, SessionNotFoundError)
            with durable_engine.connect() as conn:
                assert (
                    conn.execute(
                        select(func.count())
                        .select_from(guided_operations_table)
                        .where(guided_operations_table.c.session_id == str(parent.id))
                    ).scalar_one()
                    == 0
                )
                assert (
                    conn.execute(select(func.count()).select_from(sessions_table).where(sessions_table.c.user_id == user_id)).scalar_one()
                    == 0
                )
        else:
            staged, archive_error = await _service_lock_contention(
                race_service,
                other_service,
                parent.id,
                lambda: reserve_and_stage(race_service),
                lambda: archive(other_service),
            )
            assert not isinstance(staged, BaseException)
            assert isinstance(archive_error, SessionGuidedOperationInProgressError)
            assert staged.session.archived_at is not None
            assert (await race_service.get_session(parent.id)).archived_at is None
            with durable_engine.connect() as conn:
                operation = conn.execute(
                    select(guided_operations_table).where(
                        guided_operations_table.c.session_id == str(parent.id),
                        guided_operations_table.c.operation_id == operation_id,
                    )
                ).one()
                assert operation.result_session_id == str(staged.session.id)
                assert (
                    conn.execute(select(func.count()).select_from(sessions_table).where(sessions_table.c.user_id == user_id)).scalar_one()
                    == 2
                )
    finally:
        _cleanup_race_user(durable_engine, user_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ("delete", "copy"))
async def test_source_blob_delete_and_planned_copy_serialize_under_lock_contention(
    durable_engine,
    tmp_path: Path,
    winner: str,
) -> None:
    """Deletion wins before planning, or the frozen fork blocks it through settlement."""

    race_service = _service_for(durable_engine)
    blob_service = BlobServiceImpl(durable_engine, tmp_path / f"blob-race-{uuid4()}")
    user_id = f"fork-blob-race-{uuid4()}"
    parent = await race_service.create_session(user_id, "Parent", "local")
    source_blob = await blob_service.create_blob(parent.id, "source.csv", b"a,b\n1,2\n", "text/csv")
    state = await race_service.save_composition_state(
        parent.id,
        CompositionStateData(
            sources={
                "source": {
                    "plugin": "csv",
                    "options": {"blob_ref": str(source_blob.id), "path": source_blob.storage_path},
                }
            }
        ),
        provenance="session_seed",
    )
    message = await race_service.add_message(
        parent.id,
        "user",
        "fork here",
        composition_state_id=state.id,
        writer_principal="route_user_message",
    )
    operation_id = str(uuid4())

    async def reserve_stage_copy(target_service: SessionServiceImpl) -> tuple[Any, dict[UUID, Any], Any]:
        fence = await _claim_fork(target_service, parent.id, operation_id=operation_id)
        staged = await target_service.fork_session(
            fence,
            fork_message_id=message.id,
            new_message_content="edited",
        )

        async def checkpoint() -> None:
            return None

        copied = await blob_service.copy_blobs_for_fork(
            parent.id,
            staged.session.id,
            staged.blob_plan,
            BlobForkWriteFence(
                source_session_id=parent.id,
                target_session_id=staged.session.id,
                operation_id=fence.operation_id,
                lease_token=fence.lease_token,
                attempt=fence.attempt,
            ),
            checkpoint=checkpoint,
        )
        return staged, copied, fence

    try:
        if winner == "delete":
            deleted, staged_result = await _blob_delete_first_contention(
                race_service,
                parent.id,
                lambda: blob_service.delete_blob(source_blob.id),
                lambda: reserve_stage_copy(race_service),
            )
            assert deleted is None
            assert not isinstance(staged_result, BaseException)
            staged, copied, fence = staged_result
            assert staged.blob_plan == ()
            assert copied == {}
            with pytest.raises(AuditIntegrityError, match="absent from the frozen fork plan"):
                _rewrite_fork_state_blob_custody(
                    staged.state,
                    copied,
                    {},
                    child_session_id=staged.session.id,
                )
            await race_service.fail_guided_operation(
                fence,
                failure_code="integrity_error",
                actor="composer_route",
            )
            await blob_service.cleanup_blobs_for_fork(parent.id, staged.session.id, operation_id)
            assert [item.id for item in await race_service.list_sessions(user_id, "local")] == [parent.id]
            assert (await race_service.get_session(staged.session.id)).archived_at is not None
        else:
            (staged, copied, fence), delete_error = await _fork_first_blob_contention(
                race_service,
                parent.id,
                lambda: reserve_stage_copy(race_service),
                lambda: blob_service.delete_blob(source_blob.id),
            )
            assert isinstance(delete_error, BlobInProgressForkError)
            assert len(staged.blob_plan) == len(copied) == 1
            rewritten = _rewrite_fork_state_blob_custody(
                staged.state,
                copied,
                {source_blob.storage_path: copied[source_blob.id]},
                child_session_id=staged.session.id,
            )
            response_hash = "b" * 64
            settled = await race_service.settle_guided_fork_operation(
                GuidedForkSettlementCommand(
                    fence=fence,
                    child_session_id=staged.session.id,
                    expected_current_state_id=staged.state.id,
                    edited_message_id=staged.messages[-1].id,
                    rewritten_state_id=uuid4(),
                    rewritten_state=rewritten,
                    response_hash=response_hash,
                    actor="composer_route",
                )
            )
            assert settled.archived_at is None
            await blob_service.delete_blob(source_blob.id)
            assert await blob_service.list_blobs(parent.id, limit=None) == []
    finally:
        _cleanup_race_user(durable_engine, user_id)


@pytest.mark.asyncio
async def test_current_fence_and_concurrent_takeover_reuse_one_hidden_child(durable_engine) -> None:
    """Repeated workers race on one operation but can never publish a second child."""

    race_service = _service_for(durable_engine)
    user_id = f"fork-takeover-race-{uuid4()}"
    parent = await race_service.create_session(user_id, "Parent", "local")
    message = await race_service.add_message(
        parent.id,
        "user",
        "fork here",
        writer_principal="route_user_message",
    )
    operation_id = str(uuid4())
    fence = await _claim_fork(race_service, parent.id, operation_id=operation_id)
    initial = await race_service.fork_session(
        fence,
        fork_message_id=message.id,
        new_message_content="edited",
    )

    async def repeated_stage() -> Any:
        return await race_service.fork_session(
            fence,
            fork_message_id=message.id,
            new_message_content="edited",
        )

    current_barrier = asyncio.Barrier(2)

    async def current_worker() -> Any:
        await current_barrier.wait()
        return await repeated_stage()

    try:
        current_results = await asyncio.gather(current_worker(), current_worker())
        assert {item.session.id for item in current_results} == {initial.session.id}

        with durable_engine.begin() as conn:
            conn.execute(
                update(guided_operations_table)
                .where(
                    guided_operations_table.c.session_id == str(parent.id),
                    guided_operations_table.c.operation_id == operation_id,
                )
                .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )

        takeover_barrier = asyncio.Barrier(2)

        async def takeover_worker() -> Any:
            await takeover_barrier.wait()
            return await race_service.reserve_guided_operation(
                session_id=parent.id,
                operation_id=operation_id,
                kind="session_fork",
                request_hash="a" * 64,
                actor="composer_route",
                lease_seconds=300,
            )

        outcomes = await asyncio.gather(takeover_worker(), takeover_worker())
        takeover = next(item for item in outcomes if isinstance(item, GuidedOperationTakenOver))
        assert sum(isinstance(item, GuidedOperationTakenOver) for item in outcomes) == 1
        assert sum(isinstance(item, GuidedOperationActive) for item in outcomes) == 1
        resumed = await race_service.fork_session(
            takeover.fence,
            fork_message_id=message.id,
            new_message_content="edited",
        )
        assert resumed.session.id == initial.session.id
        with durable_engine.connect() as conn:
            assert (
                conn.execute(select(func.count()).select_from(sessions_table).where(sessions_table.c.user_id == user_id)).scalar_one() == 2
            )
            operation = conn.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == str(parent.id),
                    guided_operations_table.c.operation_id == operation_id,
                )
            ).one()
            assert operation.result_session_id == str(initial.session.id)
    finally:
        _cleanup_race_user(durable_engine, user_id)
