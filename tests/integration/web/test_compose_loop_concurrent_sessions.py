"""CL-PP-11 (spec §8.2): concurrent writes against two sessions that share
a PostgreSQL connection pool. Verifies (a) per-session sequence_no values
are independently monotonic, (b) advisory-lock collisions serialise but
do not deadlock, (c) the two-argument advisory-lock form
(``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
hashtext(session_id))``) partitions ELSPETH's
session-write namespace away from any other application or
ELSPETH subsystem on the same Postgres cluster (B3 from the Phase 1
plan-review synthesis). ``hashtext`` returns the required int4 key
directly; do not replace it with ``hashtextextended(... )::int``
because PostgreSQL integer casts are range-checked. Within the
``ELSPETH_SESSIONS_LOCK_CLASSID`` namespace birthday collisions become
probable around ~65k *concurrent* sessions — collisions cause benign
extra serialisation only, never duplicate rows (the unique index
``ix_chat_messages_session_sequence`` is the correctness guarantee)."""

from __future__ import annotations

import threading
import uuid

import pytest
import structlog
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer

from elspeth.web.sessions._persist_payload import RedactedToolRow, StatePayload
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

# Integration-suite shared session-insert helper.
from .conftest import _make_session

pytestmark = [
    pytest.mark.testcontainer,
]


@pytest.fixture(scope="module")
def pg_engine():
    """Spin up a Postgres testcontainer for the duration of this test
    module. ``create_session_engine`` is a no-op for non-SQLite
    dialects (FK enforcement is the database's responsibility on
    Postgres) — using it here keeps the engine-construction pattern
    uniform with the rest of the suite. ``initialize_session_schema``
    is the same path production uses; we explicitly avoid
    ``metadata.create_all`` here because the production schema
    bootstrap is the only path tested in CI."""
    with PostgresContainer("postgres:16-alpine") as pg:
        engine = create_session_engine(pg.get_connection_url())
        initialize_session_schema(engine)
        yield engine


@pytest.fixture
def service(pg_engine, tmp_path):
    return SessionServiceImpl(
        pg_engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


import traceback  # noqa: E402  # plan body deliberately places this import after the fixtures (see plan §Task 16 Step 3 line 208); preserved verbatim


def _worker(
    service,
    session_id: str,
    writer_id: str,
    n: int,
    errors: list,
    tracebacks: list,
) -> None:
    """Concurrent-write worker. Captures the traceback alongside any
    exception so test failures are diagnosable rather than just
    showing the exception class. Closes synthesised review nit P-N-2."""
    try:
        for i in range(n):
            tool_call_id = f"{session_id}_{writer_id}_tc_{i}"
            service.persist_compose_turn(
                session_id=session_id,
                assistant_content=f"turn {i}",
                redacted_assistant_tool_calls=({"id": tool_call_id, "function": {"name": "f"}},),
                redacted_tool_rows=(
                    RedactedToolRow(
                        tool_call_id,
                        "{}",
                        None,
                    ),
                ),
                parent_composition_state_id=None,
                expected_current_state_id=None,
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )
    except Exception as exc:
        errors.append(exc)
        tracebacks.append(traceback.format_exc())


def test_concurrent_DIFFERENT_sessions_do_not_deadlock(service):
    """Concurrent writes to two DIFFERENT sessions must not interfere.
    Different ``session_id`` values map to different
    ``hashtext(...)`` slots within the
    ``ELSPETH_SESSIONS_LOCK_CLASSID`` namespace, so the advisory locks
    do NOT contend (modulo ~65k-concurrent-session birthday-collision
    probability — see module docstring) — this test verifies the
    *non-contending* path: two sessions can write concurrently with
    both reaching their expected row counts.

    For the advisory-lock CONTENTION path see
    ``test_concurrent_SAME_session_serialises_via_advisory_lock``
    below — that is the test the spec's CL-PP-11 description was
    actually targeting."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_a")
        _make_session(conn, session_id="s_b")

    errors: list[Exception] = []
    tracebacks: list[str] = []

    threads = [
        threading.Thread(target=_worker, args=(service, "s_a", "writer_a", 5, errors, tracebacks)),
        threading.Thread(target=_worker, args=(service, "s_b", "writer_b", 5, errors, tracebacks)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"errors: {errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)

    # Both sessions reached 10 messages each (1 assistant + 1 tool per turn × 5).  # noqa: RUF003
    with service._engine.begin() as conn:
        for sid in ("s_a", "s_b"):
            count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id = :sid"), {"sid": sid}).scalar()
            assert count == 10, f"session {sid} count: {count}"
            seqs = [
                row.sequence_no
                for row in conn.execute(
                    text("SELECT sequence_no FROM chat_messages WHERE session_id = :sid ORDER BY sequence_no"), {"sid": sid}
                )
            ]
            # Strictly monotonic; gaps allowed but not duplicates.
            assert all(b > a for a, b in zip(seqs, seqs[1:], strict=False))  # noqa: RUF007  # plan body uses zip(seqs, seqs[1:]); itertools.pairwise would be a style change beyond mechanical translation


def test_concurrent_SAME_session_serialises_via_advisory_lock(service):
    """Concurrent writes to the SAME session must serialise via the
    session-scoped PostgreSQL advisory lock
    (``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
    hashtext(session_id))`` — the two-argument form
    introduced by B3 to partition ELSPETH's namespace from other
    cluster users).

    Without serialisation, two writers would race on
    ``_reserve_sequence_range``'s ``SELECT MAX(sequence_no) ...``
    and produce duplicate ``sequence_no`` values, hitting the
    ``ix_chat_messages_session_sequence`` unique index and raising
    ``IntegrityError``. With the advisory lock, the second writer
    blocks until the first commits, sees the updated max, and
    allocates the next contiguous range — no duplicates, no
    deadlock.

    Closes synthesised review finding F-02 / Q-F-02 (CL-PP-11
    coverage gap)."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_shared")

    errors: list[Exception] = []
    tracebacks: list[str] = []

    threads = [
        threading.Thread(
            target=_worker,
            args=(service, "s_shared", "writer_A", 10, errors, tracebacks),
            name="writer_A",
        ),
        threading.Thread(
            target=_worker,
            args=(service, "s_shared", "writer_B", 10, errors, tracebacks),
            name="writer_B",
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"advisory-lock serialisation failed; errors:\n{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)

    # Both writers must have completed; total rows = 2 × 10 turns ×  # noqa: RUF003
    # 2 rows/turn (one assistant + one tool) = 40.
    with service._engine.begin() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chat_messages WHERE session_id = 's_shared'")).scalar()
        assert count == 40, f"expected 40 rows, got {count}"

        # sequence_no values must be unique across both writers.
        # Duplicates would manifest as IntegrityError above; the
        # explicit check here documents the expected post-condition
        # so a future regression that disables the unique index
        # (instead of the lock) still fails the test.
        seqs = [
            row.sequence_no
            for row in conn.execute(text("SELECT sequence_no FROM chat_messages WHERE session_id = 's_shared' ORDER BY sequence_no"))
        ]
        assert len(seqs) == len(set(seqs)), f"duplicate sequence_no values: {[s for s in seqs if seqs.count(s) > 1]}"
        # Strict monotonicity: 1..40 with no gaps under successful
        # serial allocation. (Gaps are allowed in general — see spec
        # §4.1.1 — but happen only on rollback paths, none of which
        # this test exercises.)
        assert seqs == list(range(min(seqs), min(seqs) + 40))


def test_advisory_lock_actually_acquired_on_postgres(service):
    """Positive assertion against ``pg_locks``: when
    ``_acquire_session_advisory_lock`` is called inside an open
    transaction on PostgreSQL, a row appears in ``pg_locks`` with
    ``locktype='advisory'``, ``granted=true``,
    ``classid=ELSPETH_SESSIONS_LOCK_CLASSID``, and
    ``objid=hashtext('lock_probe')``. Without this,
    a typo in the dialect guard (e.g. checking ``"postgres"``
    instead of ``"postgresql"``) would silently no-op the lock
    on PG and the contention test above would still pass for
    the wrong reason — the SELECT MAX races would simply not
    overlap in test time.

    Filtering on the classid is load-bearing for B3 (Phase 1
    plan-review synthesis): if a future second classid lands in the
    same cluster, an unfiltered ``classid=*`` assertion would pass
    against the wrong subsystem's lock and silently green-light a
    regression in the sessions-DB lock path. The classid is the
    namespace-partitioning guarantee the two-argument advisory-lock
    form was introduced to provide.

    Closes synthesised review finding F-03 / Q-F-03."""
    from elspeth.contracts.advisory_locks import ELSPETH_SESSIONS_LOCK_CLASSID

    with service._engine.begin() as conn:
        _make_session(conn, session_id="lock_probe")
        # Acquire the advisory lock; do NOT release manually —
        # ``pg_advisory_xact_lock`` releases at transaction end.
        service._acquire_session_advisory_lock(conn, "lock_probe")
        # Inspect pg_locks while still inside the transaction.
        # Filter on classid AND objid so this test fails loudly if a
        # future change accidentally drops the classid (collapsing
        # back to single-argument form) or hashes the wrong session_id.
        #
        # pg_locks.objid is the Postgres ``oid`` type (unsigned 32-bit),
        # while hashtext() returns int4 (signed 32-bit). The bit
        # patterns are identical, but psycopg2 honours the type system
        # literally and Python sees them as different ints (e.g. the
        # same 0x8F4F38FB is 2_403_958_491 read unsigned vs
        # -1_891_008_805 read signed). Cast objid to int4 in the SELECT
        # projection so both sides of the comparison live in int4
        # space — keeps the type conversion in the layer that owns the
        # semantics, and preserves the post-fetch assertion structure
        # below (which distinguishes "classid filter found nothing"
        # from "objid mismatch — wrong value being hashed").
        row = conn.execute(
            text(
                "SELECT classid, objid::int4 AS objid FROM pg_locks WHERE locktype = 'advisory' AND granted = true AND classid = :classid"
            ),
            {"classid": ELSPETH_SESSIONS_LOCK_CLASSID},
        ).first()
        assert row is not None, (
            "advisory lock not visible in pg_locks under "
            f"classid={ELSPETH_SESSIONS_LOCK_CLASSID} — dialect guard "
            "in _acquire_session_advisory_lock may have skipped the "
            "Postgres branch, or the two-argument form was reverted "
            "to single-argument (regressing B3)."
        )
        # Verify objid matches hashtext(session_id).
        # Computing the expected value via the database guarantees we
        # exercise the same hash function the helper used.
        expected_objid = conn.execute(text("SELECT hashtext(:sid)"), {"sid": "lock_probe"}).scalar()
        assert row.objid == expected_objid, (
            f"advisory-lock objid mismatch: got {row.objid}, "
            f"expected {expected_objid} — _acquire_session_advisory_lock "
            "may be hashing a different value than session_id."
        )


def test_cross_allocator_save_state_serialises_with_persist_turn(service):
    """B3 (Phase 1 plan-review synthesis): the dual-allocator race.

    Pre-B3, ``persist_compose_turn`` allocated ``composition_states.version``
    under _session_write_lock, but ``save_composition_state``
    relied on a 3-attempt retry loop with NO lock. A concurrent
    interleave produced a real ``IntegrityError`` on the locked path,
    which the locked path's exception handler classified as a Tier-1
    audit-integrity violation — fabricating a Tier-1 violation from a
    benign race. (Fabricated Tier-1 violations are evidence-tampering
    under the audit-grade doctrine.)

    Post-B3, ``save_composition_state`` (and ``set_active_state``)
    also enter ``_session_write_lock`` before their SELECT MAX. The
    persist side also has the stale-current-state guard introduced in
    Task 11: if a save wins after the persist worker reads the current
    state but before it acquires the write lock, ``persist_compose_turn``
    rejects with ``StaleComposeStateError``. That is the correct
    compose-vs-revert disposition and must NOT be counted as an audit
    integrity failure. Successful writers must still leave sequential,
    contiguous version numbers in ``composition_states`` with NO
    ``IntegrityError`` raised on either path.

    This test would have failed pre-B3 by either crashing one path
    with ``IntegrityError`` or, worse, by silently incrementing the
    ``tool_row_integrity_violation_total`` counter (the false-positive
    Tier-1 alert) on the persist path.
    """
    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.service import StaleComposeStateError
    from elspeth.web.sessions.telemetry import observed_value

    session_uuid = uuid.uuid4()
    sid = str(session_uuid)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)

    errors: list[Exception] = []
    stale_rejections: list[StaleComposeStateError] = []
    tracebacks: list[str] = []
    persist_count = 5
    save_count = 5
    starting_counter = observed_value(service._telemetry.tool_row_integrity_violation_total)

    def _persist_worker():
        # B2 (Phase 1 plan-review synthesis): pre-B2 this loop passed
        # ``assistant_id=str(uuid.uuid4())`` and
        # ``assistant_raw_content=None`` to ``persist_compose_turn`` —
        # neither kwarg is declared on the signature.
        # ``assistant_id`` is helper-generated (the test never read it
        # back); ``assistant_raw_content`` was a typo for the new
        # ``raw_content`` parameter, which defaults to ``None`` and so
        # is omitted entirely (this test exercises the
        # cross-allocator race, not the redaction-attribution path).
        try:
            for i in range(persist_count):
                with service._engine.begin() as conn:
                    expected_current_state_id = conn.execute(
                        text("SELECT id FROM composition_states WHERE session_id = :sid ORDER BY version DESC LIMIT 1"), {"sid": sid}
                    ).scalar_one_or_none()
                tool_call_id = f"tc_persist_{i}"
                service.persist_compose_turn(
                    session_id=sid,
                    assistant_content="ok",
                    redacted_assistant_tool_calls=({"id": tool_call_id, "function": {"name": "f"}},),
                    redacted_tool_rows=(
                        # B1: no ``version=``; helper allocates under lock.
                        RedactedToolRow(tool_call_id, "{}", StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
                    ),
                    parent_composition_state_id=None,
                    expected_current_state_id=expected_current_state_id,
                    writer_principal="compose_loop",
                    plugin_crash_pending=False,
                )
        except StaleComposeStateError as e:
            stale_rejections.append(e)
        except Exception as e:
            import traceback

            errors.append(e)
            tracebacks.append(traceback.format_exc())

    def _save_worker():
        import asyncio

        try:
            for _ in range(save_count):
                asyncio.run(
                    service.save_composition_state(
                        session_uuid,  # ``save_composition_state`` takes UUID
                        CompositionStateData(),
                        provenance="session_seed",
                    )
                )
        except Exception as e:
            import traceback

            errors.append(e)
            tracebacks.append(traceback.format_exc())

    threads = [
        threading.Thread(target=_persist_worker, name="persist"),
        threading.Thread(target=_save_worker, name="save"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"cross-allocator race not serialised; errors:\n{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting_counter

    # Sanity: every allocated version is unique and contiguous from 1.
    with service._engine.begin() as conn:
        versions = sorted(
            row.version
            for row in conn.execute(text("SELECT version FROM composition_states WHERE session_id = :sid ORDER BY version"), {"sid": sid})
        )
    # Successful writers only: stale compose attempts roll back before
    # inserting state rows. Whatever succeeded must still be contiguous
    # from 1 with no duplicate/gap evidence of allocator races.
    assert versions == list(range(1, len(versions) + 1)), f"cross-allocator versions not contiguous: {versions}"
    assert len(versions) + len(stale_rejections) >= save_count


def test_concurrent_persist_compose_turn_same_state_stale_rejects_without_integrity_alert(service):
    """B1 plus stale-state guard: two concurrent state-changing compose
    persists based on the same current state must not fabricate a Tier-1
    integrity alert.

    Pre-B1, both writers could try to insert the same caller-supplied
    ``composition_states.version`` and one would hit
    ``uq_composition_state_version``. Post-B1 the helper allocates the
    version under the lock, but Task 11's stale-current-state guard adds
    the stronger user-intent rule: after the first writer creates the new
    current state, the second writer's ``expected_current_state_id=None``
    is stale and should be rejected with ``StaleComposeStateError`` before
    it inserts any audit rows. The integrity counter must not move.

    Closes B1 and the compose-vs-revert stale-state race from the Phase 1
    plan-review synthesis."""
    from elspeth.web.sessions._persist_payload import RedactedToolRow
    from elspeth.web.sessions.service import StaleComposeStateError
    from elspeth.web.sessions.telemetry import observed_value

    sid = "s_b1_concurrent"
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)

    starting_counter = observed_value(service._telemetry.tool_row_integrity_violation_total)

    errors: list[Exception] = []
    stale_rejections: list[StaleComposeStateError] = []
    tracebacks: list[str] = []
    barrier = threading.Barrier(2)

    def _stateful_worker(writer_id: str) -> None:
        try:
            barrier.wait(timeout=10)
            tool_call_id = f"{sid}_{writer_id}_tc_0"
            service.persist_compose_turn(
                session_id=sid,
                assistant_content="ok",
                redacted_assistant_tool_calls=({"id": tool_call_id, "function": {"name": "f"}},),
                redacted_tool_rows=(
                    RedactedToolRow(
                        tool_call_id,
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
        except StaleComposeStateError as exc:
            stale_rejections.append(exc)
        except Exception as exc:
            errors.append(exc)
            tracebacks.append(traceback.format_exc())

    threads = [
        threading.Thread(
            target=_stateful_worker,
            args=("b1_writer_A",),
            name="b1_writer_A",
        ),
        threading.Thread(
            target=_stateful_worker,
            args=("b1_writer_B",),
            name="b1_writer_B",
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # B1/stale-guard: NO IntegrityError on either path. The loser of the
    # same-state race is a stale compose, not an audit-integrity event.
    assert not errors, f"B1 regression: dual-allocator race not closed. errors:\n{errors}\ntracebacks:\n" + "\n---\n".join(tracebacks)
    assert len(stale_rejections) == 1

    # B1: the integrity-violation counter MUST NOT have moved. Any
    # increment here is a fabricated Tier-1 alert (SLO threshold = 0).
    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting_counter, (
        "B1 regression: tool_row_integrity_violation_total incremented "
        "during concurrent persist_compose_turn. SLO threshold for this "
        "counter is 0; any increment is a fabricated Tier-1 alert and "
        "evidence-tampering-class harm under ELSPETH's audit doctrine."
    )

    # Exactly one writer commits; the stale writer rolls back before
    # inserting any state or chat rows. The one committed state receives
    # version 1.
    with service._engine.begin() as conn:
        versions = [
            row.version
            for row in conn.execute(text("SELECT version FROM composition_states WHERE session_id = :sid ORDER BY version"), {"sid": sid})
        ]
    assert versions == [1]
