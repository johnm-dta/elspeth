"""Unit tests for SessionServiceImpl persistence helpers (spec §5.7.1).

Uses the shared ``engine`` fixture and ``_make_session`` helper from
``tests/unit/web/conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import structlog
from sqlalchemy import text

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
    """B3 regression: two same-session SQLite writers must not both read
    the same MAX(sequence_no). This test uses the real StaticPool
    in-memory SQLite engine from the shared fixture and two worker
    threads. The sleep happens inside the session write lock to widen
    the race window; without the process-wide per-session lock both
    workers can reserve sequence_no=1 and one insert fails."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, wait

    from sqlalchemy import insert, select

    from elspeth.web.sessions import models

    barrier = threading.Barrier(2)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_sqlite_lock")

    def _writer(index: int) -> int:
        barrier.wait()
        with service._engine.begin() as conn:  # noqa: SIM117
            with service._session_write_lock(conn, "s_sqlite_lock"):
                seq = service._reserve_sequence_range(conn, "s_sqlite_lock", count=1)
                time.sleep(0.01)
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

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(_writer, index) for index in (1, 2)]
        done, not_done = wait(futures, timeout=2.0)
        assert not not_done, (
            "SQLite same-session sequence-allocation workers did not finish within 2s; likely deadlock or lock-order regression"
        )
        seqs = sorted(future.result(timeout=0) for future in done)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

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
                state=CompositionStateData(
                    source={"kind": "tool_response"},
                    nodes=[],
                    edges=[],
                    outputs=[],
                    metadata_={},
                    is_valid=True,
                    validation_errors=None,
                ),
                derived_from_state_id=None,
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
                        state=CompositionStateData(),
                        derived_from_state_id=None,
                        provenance="session_seed",
                    )
                )
        rows = conn.execute(text("SELECT id, version FROM composition_states WHERE session_id='s4_seq' ORDER BY version")).fetchall()
    assert [r.version for r in rows] == [1, 2, 3]
    assert [r.id for r in rows] == ids


@pytest.mark.timeout(5)
def test_session_write_lock_serializes_sqlite_same_session_state_version_allocation(service):
    """Current 1A state writers also use SELECT MAX(version)+1.
    Two same-session SQLite writers must not both reserve version 1."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, wait

    from sqlalchemy import select

    from elspeth.web.sessions import models
    from elspeth.web.sessions.protocol import CompositionStateData

    barrier = threading.Barrier(2)
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s4_state_lock")

    def _writer(index: int) -> int:
        barrier.wait()
        with service._engine.begin() as conn:  # noqa: SIM117
            with service._session_write_lock(conn, "s4_state_lock"):
                state_id = service._insert_composition_state(
                    conn,
                    session_id="s4_state_lock",
                    state=CompositionStateData(metadata_={"index": index}),
                    derived_from_state_id=None,
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
                    state=CompositionStateData(),
                    derived_from_state_id=None,
                    provenance="session_seed",
                )

    # New transaction; new session. Allocation should restart at 1
    # (per-session), NOT continue at 6 (global).
    with service._engine.begin() as conn:
        with service._session_write_lock(conn, "s_ver_b"):
            service._insert_composition_state(
                conn,
                session_id="s_ver_b",
                state=CompositionStateData(),
                derived_from_state_id=None,
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
                state=CompositionStateData(),
                derived_from_state_id=None,
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
                state=CompositionStateData(),
                derived_from_state_id=None,
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

    state_a = await service.save_composition_state(sid_a, CompositionStateData())

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
