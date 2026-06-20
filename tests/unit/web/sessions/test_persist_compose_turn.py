"""Unit tests for SessionServiceImpl persistence helpers (spec §5.7.1).

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import structlog
from sqlalchemy import text

from elspeth.web.sessions._persist_payload import StatePayload
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web.conftest import _make_session


@pytest.fixture
def service(engine, tmp_path) -> SessionServiceImpl:
    """Use the shared in-memory SQLite ``engine`` fixture from
    ``tests/unit/web/conftest.py`` (which already wires
    ``create_session_engine`` + ``StaticPool`` + schema bootstrap).

    The explicit return annotation lets mypy propagate
    ``SessionServiceImpl`` through every test that consumes this
    fixture — without it, the fixture's untyped parameters poison the
    return type to ``Any`` and helper-method calls return ``Any``.
    """
    return SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )


def test_advisory_lock_sqlite_is_noop(service):
    """SQLite does not support pg_advisory_xact_lock; the Postgres-only
    helper itself remains a no-op. Same-session SQLite serialization is
    verified through _session_write_lock tests below."""
    with service._engine.begin() as conn:
        # No raise expected.
        service._acquire_session_advisory_lock(conn, "session_1")


def test_session_write_lock_sqlite_is_reentrant(service):
    """SQLite branch uses a process-wide per-session RLock so nested
    helper calls inside one transaction cannot deadlock.

    The nested ``with`` statements are intentional and load-bearing:
    flattening them with ``with A, B:`` would acquire both contexts
    sequentially before yielding, which is NOT what reentrancy testing
    asks. The test must enter the outer lock, then attempt to enter the
    same lock again from within — that nested acquisition is what proves
    the RLock semantics. ``# noqa: SIM117`` is correct here.
    """
    with service._engine.begin() as conn:  # noqa: SIM117
        with service._session_write_lock(conn, "session_1"):
            with service._session_write_lock(conn, "session_1"):
                pass


def test_session_write_lock_sqlite_commit_removes_rollback_listener(
    service: SessionServiceImpl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit release must remove the sibling rollback callback immediately."""

    from elspeth.web.sessions import service as service_module

    removed: list[str] = []
    real_remove = service_module.event.remove

    def spy_remove(target, identifier, fn):
        removed.append(str(identifier))
        return real_remove(target, identifier, fn)

    monkeypatch.setattr(service_module.event, "remove", spy_remove)

    with service._engine.begin() as conn, service._session_write_lock(conn, "session_listener_commit"):
        pass

    assert "rollback" in removed


def test_session_write_lock_sqlite_rollback_removes_commit_listener(
    service: SessionServiceImpl,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback release must remove the sibling commit callback immediately."""

    from elspeth.web.sessions import service as service_module

    removed: list[str] = []
    real_remove = service_module.event.remove

    def spy_remove(target, identifier, fn):
        removed.append(str(identifier))
        return real_remove(target, identifier, fn)

    monkeypatch.setattr(service_module.event, "remove", spy_remove)

    with (
        pytest.raises(RuntimeError, match="force rollback"),
        service._engine.begin() as conn,
        service._session_write_lock(conn, "session_listener_rollback"),
    ):
        raise RuntimeError("force rollback")

    assert "commit" in removed


def test_reserve_sequence_range_starts_at_one_for_empty_session(service):
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with service._session_write_lock(conn, "s1"):
            base = service._reserve_sequence_range(conn, "s1", count=3)
        assert base == 1


def test_reserve_sequence_range_continues_after_existing(service):
    from sqlalchemy import insert

    from elspeth.web.sessions import models

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s2")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="m1",
                session_id="s2",
                role="user",
                content="hi",
                sequence_no=5,
                writer_principal="route_user_message",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        with service._session_write_lock(conn, "s2"):
            base = service._reserve_sequence_range(conn, "s2", count=2)
        assert base == 6


def test_reserve_sequence_range_requires_session_write_lock(service):
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_no_lock")
        with pytest.raises(RuntimeError, match="_session_write_lock"):
            service._reserve_sequence_range(conn, "s_no_lock", count=1)


@pytest.mark.timeout(5)
def test_session_write_lock_serializes_sqlite_same_session_sequence_allocation(service: SessionServiceImpl) -> None:
    """B3 fixture coverage: successive same-session allocations under the
    write lock must not reuse a sequence_no.

    This uses the shared StaticPool in-memory engine, which multiplexes a
    SINGLE DBAPI connection across every checkout. Since 92612cf91 the
    session engine takes manual pysqlite begin control and issues an
    explicit ``BEGIN IMMEDIATE`` for write-intent transactions; two
    concurrent ``engine.begin()`` calls over StaticPool's one shared
    connection would therefore (correctly) raise "cannot start a
    transaction within a transaction" — StaticPool cannot model two
    genuinely independent concurrent writers. The representative
    concurrent-writer race proof lives in
    ``test_file_backed_sqlite_lock_serializes_independent_connections``,
    which uses file-backed SQLite with independently checked-out
    connections. This test keeps the StaticPool variant honest about what
    it can actually exercise: that the per-session write lock + sequence
    allocator hand out monotonic, non-colliding sequence numbers across
    successive transactions on the same connection."""
    from sqlalchemy import insert, select

    from elspeth.web.sessions import models

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_sqlite_lock")

    def _writer(index: int) -> int:
        with service._engine.begin() as conn:  # noqa: SIM117
            with service._session_write_lock(conn, "s_sqlite_lock"):
                seq = service._reserve_sequence_range(conn, "s_sqlite_lock", count=1)
                conn.execute(
                    insert(models.chat_messages_table).values(
                        id=f"m{index}",
                        session_id="s_sqlite_lock",
                        role="user",
                        content=f"message {index}",
                        sequence_no=seq,
                        writer_principal="route_user_message",
                        created_at=datetime(2026, 4, 30, tzinfo=UTC),
                    )
                )
                return seq

    seqs = sorted(_writer(index) for index in (1, 2))

    assert seqs == [1, 2]
    with service._engine.begin() as conn:
        persisted = (
            conn.execute(
                select(models.chat_messages_table.c.sequence_no)
                .where(models.chat_messages_table.c.session_id == "s_sqlite_lock")
                .order_by(models.chat_messages_table.c.sequence_no)
            )
            .scalars()
            .all()
        )
    assert persisted == [1, 2]


def test_file_backed_sqlite_sequence_allocator_smoke(tmp_path):
    """Staging uses file-backed SQLite, not only StaticPool in-memory SQLite.
    This smoke proves the same helper path works against a temporary file DB."""
    from elspeth.web.sessions.engine import create_session_engine
    from elspeth.web.sessions.schema import initialize_session_schema

    db_path = tmp_path / "sessions.db"
    engine = create_session_engine(f"sqlite:///{db_path}")
    initialize_session_schema(engine)
    service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_file")
        with service._session_write_lock(conn, "s_file"):
            assert service._reserve_sequence_range(conn, "s_file", count=1) == 1


@pytest.mark.timeout(5)
def test_file_backed_sqlite_lock_serializes_independent_connections(tmp_path):
    """Staging uses file-backed SQLite with independently checked-out
    connections. This is the representative race proof; the StaticPool
    in-memory test above is only fixture/reentrancy coverage."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, wait

    from sqlalchemy import insert, select

    from elspeth.web.sessions import models
    from elspeth.web.sessions.engine import create_session_engine
    from elspeth.web.sessions.schema import initialize_session_schema

    db_path = tmp_path / "sessions.db"
    engine = create_session_engine(f"sqlite:///{db_path}")
    initialize_session_schema(engine)
    service: SessionServiceImpl = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    with engine.begin() as conn:
        _make_session(conn, session_id="s_file_concurrent")

    barrier = threading.Barrier(2)

    def _writer(index: int) -> int:
        barrier.wait()
        with engine.begin() as conn:  # noqa: SIM117
            with service._session_write_lock(conn, "s_file_concurrent"):
                seq = service._reserve_sequence_range(conn, "s_file_concurrent", count=1)
                time.sleep(0.01)
                conn.execute(
                    insert(models.chat_messages_table).values(
                        id=f"m_file_{index}",
                        session_id="s_file_concurrent",
                        role="user",
                        content=f"message {index}",
                        sequence_no=seq,
                        writer_principal="route_user_message",
                        created_at=datetime(2026, 4, 30, tzinfo=UTC),
                    )
                )
                return seq

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(_writer, index) for index in (1, 2)]
        done, not_done = wait(futures, timeout=2.0)
        assert not not_done, (
            "File-backed SQLite sequence-allocation workers did not finish within 2s; likely deadlock or lock-order regression"
        )
        seqs = sorted(future.result(timeout=0) for future in done)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert seqs == [1, 2]
    with engine.begin() as conn:
        persisted = (
            conn.execute(
                select(models.chat_messages_table.c.sequence_no)
                .where(models.chat_messages_table.c.session_id == "s_file_concurrent")
                .order_by(models.chat_messages_table.c.sequence_no)
            )
            .scalars()
            .all()
        )
    assert persisted == [1, 2]


def test_insert_chat_message_returns_id_and_persists_row(service):
    """Happy-path: helper returns a non-empty UUID string, persists exactly one row,
    and the persisted row carries the supplied role and sequence_no.
    """
    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3")
        with service._session_write_lock(conn, "s3"):
            msg_id = service._insert_chat_message(
                conn,
                session_id="s3",
                role="assistant",
                content="hello",
                raw_content=None,
                tool_calls=None,
                sequence_no=1,
                writer_principal="compose_loop",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=now,
            )
        assert isinstance(msg_id, str) and len(msg_id) > 0
        rows = conn.execute(text("SELECT id, role, sequence_no, raw_content FROM chat_messages WHERE session_id='s3'")).fetchall()
        assert len(rows) == 1
        assert rows[0].id == msg_id
        assert rows[0].role == "assistant"
        assert rows[0].sequence_no == 1
        assert rows[0].raw_content is None


def test_insert_chat_message_persists_raw_content(service):
    """``raw_content`` is the audit-attribution column for assistant
    messages whose visible content was modified by runtime preflight
    interception. The helper MUST persist it when supplied; if it does
    not, every assistant message after Phase 1 silently loses raw
    attribution data (regression vs the pre-rev-4 ``add_message``)."""
    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3_raw")
        with service._session_write_lock(conn, "s3_raw"):
            service._insert_chat_message(
                conn,
                session_id="s3_raw",
                role="assistant",
                content="redacted output",
                raw_content="original LLM output before preflight redaction",
                tool_calls=None,
                sequence_no=1,
                writer_principal="compose_loop",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=now,
            )
        row = conn.execute(text("SELECT content, raw_content FROM chat_messages WHERE session_id='s3_raw'")).first()
        assert row.content == "redacted output"
        assert row.raw_content == "original LLM output before preflight redaction"


def test_insert_chat_message_requires_session_write_lock(service):
    """The helper is the actual chat-row writer, so the lock precondition
    must be mechanical instead of docstring-only. A future caller that
    skips ``_reserve_sequence_range`` must still crash before writing an
    arbitrary caller-supplied sequence number.
    """
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3_no_lock")
        with pytest.raises(RuntimeError, match="_session_write_lock"):
            service._insert_chat_message(
                conn,
                session_id="s3_no_lock",
                role="assistant",
                content="hello",
                raw_content=None,
                tool_calls=None,
                sequence_no=1,
                writer_principal="compose_loop",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=datetime.now(UTC),
            )


def test_insert_chat_message_rejects_tool_parent_that_is_not_assistant(service):
    """The DB FK proves same-session parent existence only. The service
    writer must mechanically enforce that tool parents are assistant rows.
    """
    from sqlalchemy import insert

    from elspeth.web.sessions import models

    now = datetime.now(UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s3_parent_role")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="u_parent",
                session_id="s3_parent_role",
                role="user",
                content="not an assistant",
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
            )
        )
        with (
            service._session_write_lock(conn, "s3_parent_role"),
            pytest.raises(RuntimeError, match=r"parent_assistant_id.*assistant"),
        ):
            service._insert_chat_message(
                conn,
                session_id="s3_parent_role",
                role="tool",
                content="{}",
                raw_content=None,
                tool_calls=None,
                sequence_no=2,
                writer_principal="compose_loop",
                composition_state_id=None,
                tool_call_id="tc_1",
                parent_assistant_id="u_parent",
                created_at=now,
            )


# --------------------------------------------------------------------------
# Task 10 RED tests: ``_insert_composition_state`` helper.
#
# Plan §2298-2502. The helper does not yet exist; these tests are expected
# to fail with AttributeError until Task 10 GREEN lands the implementation.
# --------------------------------------------------------------------------


def test_insert_composition_state_returns_id(service):
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s4")
        # B1/B3: the helper has a documented precondition — caller MUST
        # be inside the session write-lock context for the same
        # session_id within the same transaction.
        with service._session_write_lock(conn, "s4"):
            state_id = service._insert_composition_state(
                conn,
                session_id="s4",
                # B1: no ``version=``. The helper allocates it.
                # Phase 1B: state + lineage bundled into ``StatePayload``.
                payload=StatePayload(
                    data=CompositionStateData(
                        source={"kind": "tool_response"},
                        nodes=[],
                        edges=[],
                        outputs=[],
                        metadata_={},
                        is_valid=True,
                        validation_errors=None,
                    ),
                    derived_from_state_id=None,
                ),
                provenance="tool_call",
            )
        assert isinstance(state_id, str)
        rows = conn.execute(
            text("SELECT id, version, provenance, is_valid, derived_from_state_id FROM composition_states WHERE session_id='s4'")
        ).fetchall()
        assert len(rows) == 1
        # First state for this session: helper allocates COALESCE(MAX,0)+1 = 1.
        assert rows[0].version == 1
        assert rows[0].provenance == "tool_call"
        assert rows[0].is_valid == 1  # SQLite Boolean → INTEGER
        assert rows[0].derived_from_state_id is None


def test_insert_composition_state_allocates_contiguous_versions(service):
    """B1 (Phase 1 plan-review synthesis): under the held advisory lock,
    repeated calls to ``_insert_composition_state`` for the same session
    allocate contiguous versions starting at 1. The test runs serially
    within a single transaction. The concurrent state-version allocator
    proof for current 1A helpers is the SQLite race regression
    immediately below; Task 16 owns only the later
    ``persist_compose_turn`` stale-state intent check."""
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s4_seq")
        ids: list[str] = []
        with service._session_write_lock(conn, "s4_seq"):
            for _ in range(3):
                ids.append(
                    service._insert_composition_state(
                        conn,
                        session_id="s4_seq",
                        payload=StatePayload(
                            data=CompositionStateData(),
                            derived_from_state_id=None,
                        ),
                        provenance="session_seed",
                    )
                )
        rows = conn.execute(text("SELECT id, version FROM composition_states WHERE session_id='s4_seq' ORDER BY version")).fetchall()
    assert [r.version for r in rows] == [1, 2, 3]
    assert [r.id for r in rows] == ids


@pytest.mark.timeout(5)
def test_file_backed_sqlite_lock_serializes_same_session_state_version_allocation(tmp_path):
    """Current 1A state writers also use SELECT MAX(version)+1.

    Use file-backed SQLite here, not the shared in-memory StaticPool fixture:
    StaticPool hands both worker threads the same DB-API connection, so
    concurrent ``engine.begin()`` blocks can interfere before the per-session
    lock is reached. Staging and production use independently checked-out
    file-backed connections, matching this proof.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, wait

    from sqlalchemy import select

    from elspeth.web.sessions import models
    from elspeth.web.sessions.engine import create_session_engine
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.schema import initialize_session_schema

    db_path = tmp_path / "sessions.db"
    engine = create_session_engine(f"sqlite:///{db_path}")
    initialize_session_schema(engine)
    service: SessionServiceImpl = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )
    barrier = threading.Barrier(2)
    with engine.begin() as conn:
        _make_session(conn, session_id="s4_state_lock")

    def _writer(index: int) -> int:
        barrier.wait()
        with engine.begin() as conn:  # noqa: SIM117
            with service._session_write_lock(conn, "s4_state_lock"):
                state_id = service._insert_composition_state(
                    conn,
                    session_id="s4_state_lock",
                    payload=StatePayload(
                        data=CompositionStateData(metadata_={"index": index}),
                        derived_from_state_id=None,
                    ),
                    provenance="session_seed",
                )
                time.sleep(0.01)
                row = conn.execute(
                    select(models.composition_states_table.c.version).where(models.composition_states_table.c.id == state_id)
                ).scalar_one()
                return int(row)

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(_writer, index) for index in (1, 2)]
        done, not_done = wait(futures, timeout=2.0)
        assert not not_done, "SQLite same-session state-version workers did not finish within 2s; likely deadlock or lock-order regression"
        versions = sorted(future.result(timeout=0) for future in done)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    assert versions == [1, 2]


def test_insert_composition_state_versions_are_per_session(service):
    """B1 (Phase 1 plan-review synthesis): the
    ``SELECT COALESCE(MAX(version), 0) + 1`` allocation MUST filter by
    ``session_id``. ``uq_composition_state_version`` is a per-session
    constraint (see Task 1's CREATE TABLE); a global MAX would produce
    versions that are unique cluster-wide but not contiguous within a
    session, breaking the per-session monotonic-version contract every
    other read path assumes (cf. Task 11's ``assert states[0].version
    == 1`` after a fresh session)."""
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_ver_a")
        _make_session(conn, session_id="s_ver_b")
        with service._session_write_lock(conn, "s_ver_a"):
            for _ in range(5):
                service._insert_composition_state(
                    conn,
                    session_id="s_ver_a",
                    payload=StatePayload(
                        data=CompositionStateData(),
                        derived_from_state_id=None,
                    ),
                    provenance="session_seed",
                )

    # New transaction; new session. Allocation should restart at 1
    # (per-session), NOT continue at 6 (global).
    with service._engine.begin() as conn:
        with service._session_write_lock(conn, "s_ver_b"):
            service._insert_composition_state(
                conn,
                session_id="s_ver_b",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )
        row = conn.execute(text("SELECT version FROM composition_states WHERE session_id='s_ver_b'")).first()
    assert row.version == 1, (
        f"Per-session version allocation broken: s_ver_b got "
        f"version={row.version}, expected 1. The COALESCE(MAX(version)) "
        "query is missing its WHERE session_id filter."
    )


def test_insert_composition_state_requires_session_write_lock(service):
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s4_no_lock")
        with pytest.raises(RuntimeError, match="_session_write_lock"):
            service._insert_composition_state(
                conn,
                session_id="s4_no_lock",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )


def test_insert_composition_state_rejects_unknown_provenance(service):
    from sqlalchemy.exc import IntegrityError

    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s5")
        # Precondition contract: session write-lock first (see B1/B3 test above).
        with (
            service._session_write_lock(conn, "s5"),
            pytest.raises(IntegrityError, match="ck_composition_states_provenance"),
        ):
            service._insert_composition_state(
                conn,
                session_id="s5",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="rogue_value",
            )


# ----------------------------------------------------------------------------
# Task 14: ``add_message`` rev-4 preservation regression suite (plan §2949).
#
# Six tests pinning behaviours that pre-rev-4 ``add_message`` already
# exhibited and MUST survive the rewrite. Each exists because skipping
# the preservation would silently regress audit integrity.
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_message_preserves_assert_state_in_session_guard(service):
    """Rev-4 must NOT silently drop the cross-session ``composition_state_id``
    guard that pre-rev-4 ``add_message`` enforced via
    ``_assert_state_in_session``. Closes synthesised review finding SA-9."""
    from uuid import uuid4

    from elspeth.web.sessions.protocol import CompositionStateData

    sid_a = uuid4()
    sid_b = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid_a))
        _make_session(conn, session_id=str(sid_b))

    state_a = await service.save_composition_state(sid_a, CompositionStateData(), provenance="session_seed")

    with pytest.raises(RuntimeError, match="cross-session reference"):
        await service.add_message(
            sid_b,
            "user",
            "should fail",
            composition_state_id=state_a.id,
            writer_principal="route_user_message",
        )


@pytest.mark.asyncio
async def test_add_message_preserves_updated_at_write(service):
    """Rev-4 must continue to bump ``sessions.updated_at`` on every insert.
    The list-sessions UI/API path orders by this column; dropping the
    write silently misorders the home screen."""
    import asyncio
    from uuid import uuid4

    from sqlalchemy import select

    from elspeth.web.sessions import models

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))
        before = conn.execute(select(models.sessions_table.c.updated_at).where(models.sessions_table.c.id == str(sid))).scalar_one()

    await asyncio.sleep(0.001)

    await service.add_message(sid, "user", "hi", writer_principal="route_user_message")

    with service._engine.begin() as conn:
        after = conn.execute(select(models.sessions_table.c.updated_at).where(models.sessions_table.c.id == str(sid))).scalar_one()
    assert after > before, "sessions.updated_at must advance after add_message"


@pytest.mark.asyncio
async def test_add_message_preserves_raw_content(service):
    """Rev-4 must persist ``raw_content`` when supplied. The compose/recompose
    routes pass ``raw_content=result.raw_assistant_content`` for assistant
    messages whose visible content was rewritten by preflight redaction;
    dropping the value would silently lose audit attribution."""
    from uuid import uuid4

    from sqlalchemy import select

    from elspeth.web.sessions import models

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))

    record = await service.add_message(
        sid,
        "assistant",
        "redacted",
        raw_content="original LLM output",
        writer_principal="compose_loop",
    )
    assert record.raw_content == "original LLM output"

    with service._engine.begin() as conn:
        row = conn.execute(
            select(models.chat_messages_table.c.raw_content).where(models.chat_messages_table.c.id == str(record.id))
        ).scalar_one()
    assert row == "original LLM output"


@pytest.mark.asyncio
async def test_add_message_returns_chat_message_record(service):
    """Return type MUST stay ``ChatMessageRecord``. Callers consume
    ``.id``, ``.created_at``, ``.raw_content`` — a ``str`` return would
    break every one."""
    from uuid import uuid4

    from elspeth.web.sessions.protocol import ChatMessageRecord

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))

    result = await service.add_message(sid, "user", "hi", writer_principal="route_user_message")
    assert isinstance(result, ChatMessageRecord)
    assert result.session_id == sid
    assert result.role == "user"
    assert result.content == "hi"
    assert result.writer_principal == "route_user_message"
    assert result.created_at is not None


@pytest.mark.asyncio
async def test_add_message_requires_writer_principal(service):
    """Rev-4 breaking change: ``writer_principal`` is required keyword-only."""
    from uuid import uuid4

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))
    with pytest.raises(TypeError, match="writer_principal"):
        await service.add_message(sid, "user", "hi")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_get_messages_orders_same_timestamp_rows_by_sequence_no(service):
    """Rev-4 (B2): the canonical ordering key is ``sequence_no``. Write two
    or more rows whose ``created_at`` values are identical but whose
    ``sequence_no`` values are intentionally non-chronological through the
    1A writer/helper path; ``get_messages`` must follow ``sequence_no``,
    not ``created_at`` or insertion accident."""
    from uuid import UUID, uuid4

    sid = uuid4()
    same_ts = datetime(2026, 5, 6, tzinfo=UTC)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))
        with service._session_write_lock(conn, str(sid)):
            # Reserve sequence range, then insert in sequence_no order [3, 1, 2]
            # by issuing three single-row helper calls. Each call passes the
            # explicit sequence_no the test wants; the helper does not allocate.
            base = service._reserve_sequence_range(conn, str(sid), count=3)
            id_third = service._insert_chat_message(
                conn,
                session_id=str(sid),
                role="user",
                content="third",
                raw_content=None,
                tool_calls=None,
                sequence_no=base + 2,
                writer_principal="route_user_message",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=same_ts,
            )
            id_first = service._insert_chat_message(
                conn,
                session_id=str(sid),
                role="user",
                content="first",
                raw_content=None,
                tool_calls=None,
                sequence_no=base,
                writer_principal="route_user_message",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=same_ts,
            )
            id_second = service._insert_chat_message(
                conn,
                session_id=str(sid),
                role="user",
                content="second",
                raw_content=None,
                tool_calls=None,
                sequence_no=base + 1,
                writer_principal="route_user_message",
                composition_state_id=None,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=same_ts,
            )

    messages = await service.get_messages(sid, limit=None)
    assert [m.content for m in messages] == ["first", "second", "third"]
    assert [m.id for m in messages] == [UUID(id_first), UUID(id_second), UUID(id_third)]


@pytest.mark.asyncio
async def test_add_message_rejects_unknown_writer_principal(service):
    """The CHECK constraint backs the type system: an unknown
    ``writer_principal`` value MUST raise ``IntegrityError`` at write
    time. The schema is the load-bearing enforcement; the Python type
    is ``str`` for forward-compatibility with future enum extensions."""
    from uuid import uuid4

    from sqlalchemy.exc import IntegrityError

    sid = uuid4()
    with service._engine.begin() as conn:
        _make_session(conn, session_id=str(sid))
    with pytest.raises(IntegrityError, match="ck_chat_messages_writer_principal"):
        await service.add_message(sid, "user", "hi", writer_principal="rogue_writer")


# ---------------------------------------------------------------------------
# Task 11 tests: persist_compose_turn happy path + transcript validation +
# commit-wins async contract.
# ---------------------------------------------------------------------------


def test_persist_compose_turn_happy_path(service):
    from elspeth.web.sessions._persist_payload import (
        RedactedToolRow,
        StatePayload,
    )
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s6")

    outcome = service.persist_compose_turn(
        session_id="s6",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_1", "function": {"name": "set_source"}},),
        redacted_tool_rows=(
            RedactedToolRow(
                tool_call_id="tc_1",
                content='{"ok": true}',
                # B1 (Phase 1 plan-review synthesis): no ``version=``.
                # ``_insert_composition_state`` allocates it under the
                # held session write lock; the assertion below pins the
                # allocated value to 1 (first state in this session).
                composition_state_payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # On the success path, AuditOutcome carries the new
    # assistant_id and unwind_audit_failed=False. The old
    # tier1_violation field was removed in Stage 4 of the plan
    # revision (Tier-1 failures now raise AuditIntegrityError
    # directly -- see Task 13).
    assert outcome.unwind_audit_failed is False
    assert outcome.assistant_id is not None

    with service._engine.begin() as conn:
        rows = conn.execute(
            text("SELECT role, sequence_no, tool_call_id FROM chat_messages WHERE session_id='s6' ORDER BY sequence_no")
        ).fetchall()
        assert [r.role for r in rows] == ["assistant", "tool"]
        assert rows[0].sequence_no == 1
        assert rows[1].sequence_no == 2
        assert rows[1].tool_call_id == "tc_1"

        states = conn.execute(text("SELECT version, provenance FROM composition_states WHERE session_id='s6'")).fetchall()
        assert len(states) == 1
        assert states[0].version == 1
        assert states[0].provenance == "tool_call"


def test_persist_compose_turn_zero_tool_rows(service):
    """W10a (Phase 1 plan-review synthesis): a turn with
    ``redacted_tool_rows=()`` and ``redacted_assistant_tool_calls=()``
    is a valid and reachable shape -- the assistant produced text but
    chose not to call any tools. Spec §5.2 explicitly allows this. The
    primitive MUST commit cleanly: the assistant row is persisted, no
    tool rows are inserted, and no ``composition_states`` rows are
    created (because the empty tool-row tuple has no
    ``composition_state_payload`` to write).

    The zero-row case is not exercised by ``happy_path`` (which always
    includes one ``RedactedToolRow``), so without this regression the
    next caller migrating an assistant-only call site (Phase 3) would
    discover an off-by-one or empty-tuple bug at integration time
    rather than at the primitive's own unit boundary.
    """
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_zero")
    outcome = service.persist_compose_turn(
        session_id="s_zero",
        assistant_content="text only",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False
    with service._engine.begin() as conn:
        roles = [r.role for r in conn.execute(text("SELECT role FROM chat_messages WHERE session_id='s_zero'")).fetchall()]
        assert roles == ["assistant"]
        states = conn.execute(text("SELECT id FROM composition_states WHERE session_id='s_zero'")).fetchall()
        assert states == []


def test_persist_compose_turn_persists_raw_content(service):
    """B2 (Phase 1 plan-review synthesis): ``persist_compose_turn`` must
    plumb the optional ``raw_content`` argument to the assistant row
    verbatim. ``raw_content`` is the audit-attribution column that
    captures the original LLM output BEFORE preflight redaction
    rewrote ``content``. Routes 2151 and 2601 in
    ``src/elspeth/web/sessions/routes.py`` already pass
    ``raw_content=result.raw_assistant_content`` to ``add_message``;
    Phase 3 migrates those call sites to ``persist_compose_turn``, so
    the primitive must accept and persist the column today.
    """
    from elspeth.web.sessions._persist_payload import RedactedToolRow

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s6_raw")

    outcome = service.persist_compose_turn(
        session_id="s6_raw",
        assistant_content="ok (redacted)",
        raw_content="original LLM output before preflight redaction",
        redacted_assistant_tool_calls=({"id": "tc_1", "function": {"name": "f"}},),
        redacted_tool_rows=(RedactedToolRow(tool_call_id="tc_1", content="{}", composition_state_payload=None),),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False

    with service._engine.begin() as conn:
        rows = conn.execute(
            text("SELECT role, content, raw_content FROM chat_messages WHERE session_id='s6_raw' ORDER BY sequence_no")
        ).fetchall()
        # Assistant row carries both visible content (post-redaction)
        # and raw_content (pre-redaction); tool row has raw_content=None.
        assert rows[0].role == "assistant"
        assert rows[0].content == "ok (redacted)"
        assert rows[0].raw_content == "original LLM output before preflight redaction"
        assert rows[1].role == "tool"
        assert rows[1].raw_content is None


def test_persist_compose_turn_rejects_cross_session_parent_state(service):
    """B5: when ``parent_composition_state_id`` belongs to a DIFFERENT
    session, the call MUST raise ``RuntimeError`` with the precise
    diagnostic produced by ``_assert_state_in_session`` -- not a generic
    FK error.
    """
    from elspeth.web.sessions._persist_payload import StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_A")
        _make_session(conn, session_id="s_B")
        with service._session_write_lock(conn, "s_A"):
            state_a_id = service._insert_composition_state(
                conn,
                session_id="s_A",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    with pytest.raises(
        RuntimeError,
        match=r"persist_compose_turn: composition_state_id=.*belongs to session "
        r"'s_A', not 's_B'.*cross-session reference is a contract violation",
    ):
        service.persist_compose_turn(
            session_id="s_B",
            assistant_content="should fail",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=state_a_id,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    with service._engine.begin() as conn:
        b_count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id='s_B'")).scalar()
        assert b_count == 0, f"persist_compose_turn rolled back incorrectly; s_B has {b_count} chat rows after a guard-rejected call"


def test_persist_compose_turn_accepts_valid_same_session_parent_state(service):
    """B5 happy path: same-session parent state -- guard passes silently
    and the assistant row is correctly stamped with that
    ``composition_state_id``."""
    from elspeth.web.sessions._persist_payload import StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_C")
        with service._session_write_lock(conn, "s_C"):
            state_c_id = service._insert_composition_state(
                conn,
                session_id="s_C",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    outcome = service.persist_compose_turn(
        session_id="s_C",
        assistant_content="ok",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=state_c_id,
        expected_current_state_id=state_c_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    assert outcome.unwind_audit_failed is False
    assert outcome.assistant_id is not None

    with service._engine.begin() as conn:
        assistant_row = conn.execute(
            text("SELECT composition_state_id FROM chat_messages WHERE session_id='s_C' AND role='assistant'")
        ).fetchone()
        assert assistant_row is not None
        assert assistant_row.composition_state_id == state_c_id


def test_persist_compose_turn_rejects_stale_expected_current_state(service):
    """A compose turn may not persist if the session's current state
    changed while the LLM call was in flight."""
    from elspeth.web.sessions._persist_payload import StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData, StaleComposeStateError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_stale")
        with service._session_write_lock(conn, "s_stale"):
            stale_state_id = service._insert_composition_state(
                conn,
                session_id="s_stale",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )
            current_state_id = service._insert_composition_state(
                conn,
                session_id="s_stale",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=stale_state_id,
                ),
                provenance="session_seed",
            )

    with pytest.raises(
        StaleComposeStateError,
        match=r"current composition state changed.*expected=.*actual=",
    ):
        service.persist_compose_turn(
            session_id="s_stale",
            assistant_content="stale",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=stale_state_id,
            expected_current_state_id=stale_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    with service._engine.begin() as conn:
        rows = conn.execute(text("SELECT role FROM chat_messages WHERE session_id='s_stale'")).fetchall()
        latest = conn.execute(
            text("SELECT id FROM composition_states WHERE session_id='s_stale' ORDER BY version DESC LIMIT 1")
        ).scalar_one()
    assert rows == []
    assert latest == current_state_id


def test_persist_compose_turn_accepts_matching_expected_current_state(service):
    from elspeth.web.sessions._persist_payload import StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_current_ok")
        with service._session_write_lock(conn, "s_current_ok"):
            current_state_id = service._insert_composition_state(
                conn,
                session_id="s_current_ok",
                payload=StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    outcome = service.persist_compose_turn(
        session_id="s_current_ok",
        assistant_content="ok",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=None,
        expected_current_state_id=current_state_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False


@pytest.mark.asyncio
async def test_persist_compose_turn_refuses_async_invocation(service):
    """SA-7 / M1: calling the sync method from inside async raises
    RuntimeError. Production callers use ``await
    service.persist_compose_turn_async(...)`` which dispatches to a
    worker thread."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_async_guard")

    with pytest.raises(RuntimeError, match="must be dispatched via"):
        service.persist_compose_turn(
            session_id="s_async_guard",
            assistant_content="",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )


@pytest.mark.asyncio
async def test_persist_compose_turn_async_protocol_dispatch_succeeds_from_async(service):
    """Companion: the protocol-public async dispatcher runs the sync
    primitive in a worker thread (no running loop in that thread), so
    the guard passes."""
    from elspeth.web.sessions._persist_payload import RedactedToolRow

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_run_sync")

    outcome = await service.persist_compose_turn_async(
        session_id="s_run_sync",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_run_sync", "function": {"name": "f"}},),
        redacted_tool_rows=(RedactedToolRow("tc_run_sync", "{}", None),),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False


def test_persist_compose_turn_rejects_missing_tool_row(service):
    """Q-F1 missing axis."""
    from elspeth.web.sessions.protocol import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_missing")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"missing=\['tc_X'\].*extra=\[\]",
    ):
        service.persist_compose_turn(
            session_id="s_missing",
            assistant_content="ok",
            redacted_assistant_tool_calls=({"id": "tc_X", "function": {"name": "f"}},),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id='s_missing'")).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_extra_tool_row(service):
    """Q-F1 extra axis."""
    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.protocol import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_extra")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"missing=\[\].*extra=\['tc_Y'\]",
    ):
        service.persist_compose_turn(
            session_id="s_extra",
            assistant_content="ok",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(RedactedToolRow("tc_Y", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id='s_extra'")).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_mismatched_tool_call_ids(service):
    """Q-F1: both ``missing`` and ``extra`` axes fire simultaneously."""
    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.protocol import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_mismatch")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"missing=\['tc_A'\].*extra=\['tc_B'\]",
    ):
        service.persist_compose_turn(
            session_id="s_mismatch",
            assistant_content="ok",
            redacted_assistant_tool_calls=({"id": "tc_A", "function": {"name": "f"}},),
            redacted_tool_rows=(RedactedToolRow("tc_B", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id='s_mismatch'")).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_duplicate_tool_call_id_in_assistant(service):
    """Q-F1: duplicate in assistant tool_calls."""
    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.protocol import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_dup_assist")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"duplicates_in_assistant=\['tc_D'\]",
    ):
        service.persist_compose_turn(
            session_id="s_dup_assist",
            assistant_content="ok",
            redacted_assistant_tool_calls=(
                {"id": "tc_D", "function": {"name": "f"}},
                {"id": "tc_D", "function": {"name": "g"}},
            ),
            redacted_tool_rows=(RedactedToolRow("tc_D", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id='s_dup_assist'")).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_duplicate_tool_call_id_in_rows(service):
    """Q-F1: duplicate in tool rows."""
    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.protocol import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_dup_rows")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"duplicates_in_rows=\['tc_E'\]",
    ):
        service.persist_compose_turn(
            session_id="s_dup_rows",
            assistant_content="ok",
            redacted_assistant_tool_calls=({"id": "tc_E", "function": {"name": "f"}},),
            redacted_tool_rows=(
                RedactedToolRow("tc_E", "{}", None),
                RedactedToolRow("tc_E", "{}", None),
            ),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id='s_dup_rows'")).scalar()
        assert count == 0


@pytest.mark.asyncio
async def test_persist_compose_turn_async_caller_cancellation_commits_anyway(service):
    """Q-F2 commit-wins contract: caller cancellation does NOT roll
    back the worker. The post-cancel DB state must contain the
    persisted rows, and the integrity counter MUST NOT have moved on
    a benign cancel-and-retry pattern.

    Deterministic-gate variant: the spec's original sleep(50ms) trigger
    is a timing race against the worker's commit speed. On in-memory
    SQLite the inserts complete in well under 50ms, so by the time
    cancel arrives the awaiter has already returned cleanly and no
    ``CancelledError`` ever fires. To pin the contract under test --
    *cancel-while-work-is-in-flight commits anyway* -- we monkeypatch
    the sync primitive with a wrapper that blocks on a
    ``threading.Event`` until the test explicitly releases it. That
    holds the worker open long enough to receive the cancel; the
    contract under test is unchanged.
    """
    import asyncio
    import threading

    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_cancel")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    real_persist = service.persist_compose_turn
    release = threading.Event()
    worker_started = threading.Event()
    worker_finished = threading.Event()
    worker_errors: list[BaseException] = []

    def gated_persist(*args, **kwargs):
        try:
            worker_started.set()
            if not release.wait(timeout=10.0):
                pytest.fail("test never released the gated worker within 10s")
            return real_persist(*args, **kwargs)
        except BaseException as exc:
            worker_errors.append(exc)
            raise
        finally:
            worker_finished.set()

    # SessionServiceImpl is a plain class with no __slots__; bound-method
    # rebinding via attribute assignment is the standard test-time
    # monkey-patch. The mypy ignore is required because mypy treats
    # bound methods as immutable on classes that declare them.
    service.persist_compose_turn = gated_persist  # type: ignore[method-assign]

    async def _do_persist() -> None:
        await service.persist_compose_turn_async(
            session_id="s_cancel",
            assistant_content="commit-wins",
            redacted_assistant_tool_calls=({"id": "tc_c1", "function": {"name": "f"}},),
            redacted_tool_rows=(RedactedToolRow("tc_c1", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    inner = asyncio.create_task(_do_persist())

    # Wait until the worker thread has actually entered the gate. This
    # replaces the spec's racy 50ms sleep with a deterministic
    # rendezvous on in-memory SQLite (where the entire
    # persist_compose_turn body would otherwise complete in well under
    # 50ms, leaving cancel() to land on an already-done task).
    for _ in range(200):
        if worker_started.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("worker thread never reached the gate within 2s")

    inner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inner

    # Now release the worker so it can commit. The worker is shielded;
    # the cancel above only affected the awaiter.
    release.set()

    # Wait until the shielded worker has finished. The async bridge has
    # no public completion handle after caller cancellation, so the test
    # records the monkeypatched worker's terminal state directly.
    for _ in range(1000):
        if worker_finished.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("shielded worker did not finish within 10s; commit-wins contract is not honoured by the current _run_sync bridge.")

    if worker_errors:
        raise AssertionError("shielded worker failed after caller cancellation") from worker_errors[0]

    with service._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id='s_cancel'")).scalar()
    assert count == 2  # assistant + tool

    # Counter MUST NOT have moved -- there was no IntegrityError, no
    # benign "fabricated Tier-1" event from the cancel path.
    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting, (
        "Q-F2 regression: caller cancellation produced a "
        "tool_row_integrity_violation_total increment. Under the "
        "commit-wins contract, a clean cancel must not fabricate "
        "Tier-1 alerts (SLO threshold = 0)."
    )


def test_persist_compose_turn_integrity_error_propagates(service):
    """Duplicate tool_call_id within one session triggers IntegrityError;
    counter increments; helper re-raises (no recovery — spec §4.5)."""
    from sqlalchemy.exc import IntegrityError

    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s7")

    # First turn: creates tool_call_id='dup'
    service.persist_compose_turn(
        session_id="s7",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "dup", "function": {"name": "x"}},),
        redacted_tool_rows=(RedactedToolRow("dup", "{}", None),),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)
    with pytest.raises(
        IntegrityError,
        match=(
            r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
            r"|uq_chat_messages_tool_call_id)"
        ),
    ):
        service.persist_compose_turn(
            session_id="s7",
            assistant_content="",
            redacted_assistant_tool_calls=({"id": "dup", "function": {"name": "x"}},),
            redacted_tool_rows=(RedactedToolRow("dup", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting + 1


# Spec §4.5 enumerates multiple constraint sources that all flow
# through the same IntegrityError handler in persist_compose_turn.
# The test above covers the partial-unique-tool_call_id source; the
# parametrised test below covers the other reachable source via
# persist_compose_turn's parameter surface.
#
# Sources that are not reachable through public parameters are NOT in
# this matrix because they cannot occur via the entry point:
#
# - role enum violation — persist_compose_turn hardcodes
#   'assistant'/'tool'.
# - **uq_composition_state_version — closed by B1.** StatePayload no
#   longer carries caller-supplied version; _insert_composition_state
#   allocates under _session_write_lock. Constraint is structurally
#   unreachable. The replacement test
#   ``test_persist_compose_turn_state_versions_do_not_collide`` below
#   pins the post-B1 contract.
#
# nonexistent_parent_composition_state is deliberately NOT in this
# matrix. Task 11's _assert_state_in_session guard rejects it with
# RuntimeError before any INSERT — counting that as IntegrityError
# would double-count a caller contract violation as a Tier-1 DB
# integrity event. Separate RuntimeError regression below pins this.


@pytest.mark.parametrize(
    "scenario_name, setup_kwargs, expected_match",
    [
        pytest.param(
            "unknown_writer_principal",
            {"writer_principal": "rogue_caller"},
            r"ck_chat_messages_writer_principal",
            id="ck_chat_messages_writer_principal",
        ),
    ],
)
def test_persist_compose_turn_integrity_error_matrix(
    service,
    scenario_name,
    setup_kwargs,
    expected_match,
):
    """Each scenario triggers a distinct §4.5 source via
    persist_compose_turn's parameter surface; all flow through the
    same handler. Asserts both the counter increments AND the
    constraint name appears in the raised exception message."""
    from sqlalchemy.exc import IntegrityError

    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id=f"s_{scenario_name}")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    base_kwargs = {
        "session_id": f"s_{scenario_name}",
        "assistant_content": "",
        "redacted_assistant_tool_calls": ({"id": f"{scenario_name}_tc", "function": {"name": "f"}},),
        "redacted_tool_rows": (RedactedToolRow(f"{scenario_name}_tc", "{}", None),),
        "parent_composition_state_id": None,
        "expected_current_state_id": None,
        "writer_principal": "compose_loop",
        "plugin_crash_pending": False,
    }
    base_kwargs.update(setup_kwargs)

    with pytest.raises(IntegrityError, match=expected_match):
        service.persist_compose_turn(**base_kwargs)

    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting + 1, (
        f"counter must increment for {scenario_name}"
    )


def test_persist_compose_turn_rejects_missing_parent_state_before_insert(service):
    """A nonexistent parent composition state is a caller contract error,
    not an IntegrityError-source matrix case.

    Task 11's _assert_state_in_session guard rejects the missing state
    before the assistant row INSERT. The audit-integrity counter must
    not move because no DB constraint fired and no Tier-1 audit
    corruption was observed."""
    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_missing_parent")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    with pytest.raises(
        RuntimeError,
        match=(
            r"persist_compose_turn: composition_state_id='doesnotexist' "
            r"does not exist"
        ),
    ):
        service.persist_compose_turn(
            session_id="s_missing_parent",
            assistant_content="",
            redacted_assistant_tool_calls=({"id": "missing_parent_tc", "function": {"name": "f"}},),
            redacted_tool_rows=(RedactedToolRow("missing_parent_tc", "{}", None),),
            parent_composition_state_id="doesnotexist",
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting


def test_persist_compose_turn_state_versions_do_not_collide(service):
    """B1 contract pin: serial successful persists allocate contiguous
    versions and never increment the integrity counter for the
    version-collision constraint.

    Pre-B1 a draft of this test asserted the counter SHOULD increment
    when StatePayload(version=1) was supplied twice. **That codified
    the fabrication vector B1 closes** — every IntegrityError increment
    on uq_composition_state_version was structurally a contention loss
    masquerading as a Tier-1 audit-integrity violation.

    Post-B1 StatePayload has no version field; _insert_composition_state
    allocates versions under _session_write_lock. Two successive turns
    get [1, 2] (contiguous), counter MUST stay at starting."""
    from sqlalchemy import text

    from elspeth.web.sessions._persist_payload import RedactedToolRow, StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_ver")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    service.persist_compose_turn(
        session_id="s_ver",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_v1", "function": {"name": "f"}},),
        redacted_tool_rows=(
            RedactedToolRow(
                "tc_v1",
                "{}",
                StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    with service._engine.begin() as conn:
        first_state_id = conn.execute(
            text("SELECT id FROM composition_states WHERE session_id='s_ver' ORDER BY version DESC LIMIT 1")
        ).scalar_one()

    # Second turn — pre-B1 this would have collided on
    # uq_composition_state_version because the test supplied
    # version=1 on both turns. Post-B1 the helper allocates
    # version=2 (COALESCE(MAX,0)+1 = 2), so the call succeeds.
    service.persist_compose_turn(
        session_id="s_ver",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_v2", "function": {"name": "f"}},),
        redacted_tool_rows=(
            RedactedToolRow(
                "tc_v2",
                "{}",
                StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=first_state_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting, (
        "B1 regression: tool_row_integrity_violation_total incremented "
        "on serial state-version allocation. SLO threshold for this "
        "counter is 0; any increment here is a fabricated Tier-1 alert "
        "and evidence-tampering-class harm under the audit doctrine."
    )

    with service._engine.begin() as conn:
        versions = [
            r.version for r in conn.execute(text("SELECT version FROM composition_states WHERE session_id='s_ver' ORDER BY version"))
        ]
    assert versions == [1, 2], f"B1 regression: per-session version allocation broken; got {versions}"
