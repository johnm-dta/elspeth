"""Spec §4.1.2 / §1.4 NFR: state-ahead-of-audit is impossible at the
schema level. After any persist_compose_turn call, the SQL predicate
below must return zero rows."""

from __future__ import annotations

import pytest
import structlog
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions._persist_payload import RedactedToolRow, StatePayload
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

# ``_make_session`` lives in ``tests/integration/web/conftest.py`` — a
# duplicate of the unit-test conftest helper. Importing the helper
# here keeps the per-test session-insert site uniform with the rest
# of the suite.
from .conftest import _make_session


@pytest.fixture
def service(tmp_path):
    """Service with an in-memory SQLite engine. The test runs
    end-to-end against the real production code paths
    (``create_session_engine`` + ``initialize_session_schema``);
    integration here means "exercises persist_compose_turn against
    a real SQLite engine," not "uses Docker" — see the conftest
    docstring for why integration and unit conftests are separate."""
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return SessionServiceImpl(
        eng,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


_BACKWARD_PREDICATE = """
SELECT cs.id
  FROM composition_states cs
  LEFT JOIN chat_messages cm
    ON cm.composition_state_id = cs.id AND cm.role = 'tool'
 WHERE cs.provenance = 'tool_call' AND cs.version > 0
   AND cm.id IS NULL
"""


@pytest.fixture
def populated_audit_db(service):
    """Initialized audit DB with successful and rolled-back compose-turn writes."""

    from sqlalchemy.exc import IntegrityError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="populated_audit")
    service.persist_compose_turn(
        session_id="populated_audit",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_populated_a", "function": {"name": "f"}},),
        redacted_tool_rows=(
            RedactedToolRow(
                "tc_populated_a",
                '{"r": 1}',
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
            text("SELECT id FROM composition_states WHERE session_id='populated_audit' ORDER BY version DESC LIMIT 1")
        ).scalar_one()
    with pytest.raises(
        IntegrityError,
        match=(
            r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
            r"|uq_chat_messages_tool_call_id)"
        ),
    ):
        service.persist_compose_turn(
            session_id="populated_audit",
            assistant_content="duplicate",
            redacted_assistant_tool_calls=({"id": "tc_populated_a", "function": {"name": "f"}},),
            redacted_tool_rows=(
                RedactedToolRow(
                    "tc_populated_a",
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
    return service._engine


def test_no_state_row_without_tool_row(populated_audit_db):
    """The INV-AUDIT-AHEAD backward-direction post-condition is a SQL predicate."""

    with populated_audit_db.connect() as conn:
        orphans = conn.execute(text(_BACKWARD_PREDICATE)).fetchall()
    assert orphans == []


def test_backward_direction_holds_after_successful_persist(service):
    with service._engine.begin() as conn:
        _make_session(conn, session_id="b1")
    service.persist_compose_turn(
        session_id="b1",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_a", "function": {"name": "f"}},),
        redacted_tool_rows=(
            RedactedToolRow(
                "tc_a",
                '{"r": 1}',
                # B1 (Phase 1 plan-review synthesis): no ``version=``;
                # ``_insert_composition_state`` allocates under the lock.
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
        violations = conn.execute(text(_BACKWARD_PREDICATE)).fetchall()
    assert violations == [], f"backward-direction violation rows: {violations}"


def test_backward_direction_holds_after_integrity_error_rollback(service):
    """After a failed persist_compose_turn (rolled back transaction), no
    composition_states row should be visible."""
    from sqlalchemy.exc import IntegrityError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="b2")
    # First successful turn.
    service.persist_compose_turn(
        session_id="b2",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_x", "function": {"name": "f"}},),
        redacted_tool_rows=(
            RedactedToolRow(
                "tc_x",
                "{}",
                # B1: no ``version=``; helper allocates under the lock.
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
            text("SELECT id FROM composition_states WHERE session_id='b2' ORDER BY version DESC LIMIT 1")
        ).scalar_one()
    # Second turn deliberately reuses tc_x to trigger the partial
    # unique index ``uq_chat_messages_tool_call_id`` (added in Task 2).
    with pytest.raises(
        IntegrityError,
        match=(
            r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
            r"|uq_chat_messages_tool_call_id)"
        ),
    ):
        service.persist_compose_turn(
            session_id="b2",
            assistant_content="",
            redacted_assistant_tool_calls=({"id": "tc_x", "function": {"name": "f"}},),
            redacted_tool_rows=(
                RedactedToolRow(
                    "tc_x",
                    "{}",
                    # B1: no ``version=``.
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
    with service._engine.begin() as conn:
        violations = conn.execute(text(_BACKWARD_PREDICATE)).fetchall()
        assert violations == []
        # And exactly one tool_call provenance row from the first (successful) turn.
        state_count = conn.execute(
            text("SELECT COUNT(*) AS c FROM composition_states WHERE session_id='b2' AND provenance='tool_call'")
        ).scalar()
        assert state_count == 1


def test_get_messages_orders_assistant_before_tool_rows_within_one_turn(service):
    """B2 (Phase 1 plan-review synthesis): a single ``persist_compose_turn``
    stamps every row in the turn with one shared ``created_at`` = ``now``;
    on fast SQLite the rows share a microsecond, so ``get_messages``'s
    pre-B2 ``ORDER BY created_at`` returned them nondeterministically.
    Post-B2 ``get_messages`` orders by ``sequence_no`` (allocated under
    the session write lock = monotonic and unique), so the
    intra-turn order is the order in which the writer appended rows
    (assistant first, tool rows in plan order). This test would have
    failed on the pre-B2 codebase.
    """
    from uuid import UUID

    # B2 (Phase 1 plan-review synthesis): pre-B2 this test bound
    # ``sid="ord1"`` and ``sid_uuid=UUID("00000000-...-001")``, then
    # inserted the session under ``sid`` and queried under
    # ``sid_uuid`` — two different sessions. The fix derives one
    # canonical UUID and uses its string form for ``_make_session`` /
    # ``persist_compose_turn`` and the UUID form for ``get_messages``.
    sid_uuid = UUID("00000000-0000-0000-0000-000000000001")
    sid = str(sid_uuid)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)
    # B2: ``assistant_id`` and ``assistant_raw_content`` were stale
    # kwargs from a prior plan draft that ``persist_compose_turn`` does
    # not declare. ``assistant_id`` is helper-generated (the test never
    # referenced the supplied value); ``assistant_raw_content`` was a
    # typo for the new ``raw_content`` parameter and is left at its
    # default ``None`` here because the test's narrative does not
    # exercise the redaction-attribution path.
    service.persist_compose_turn(
        session_id=sid,
        assistant_content="ok",
        redacted_assistant_tool_calls=(
            {"id": "tc_a", "function": {"name": "f"}},
            {"id": "tc_b", "function": {"name": "g"}},
            {"id": "tc_c", "function": {"name": "h"}},
        ),
        # B1 (Phase 1 plan-review synthesis): no ``version=`` kwargs.
        # ``_insert_composition_state`` allocates per-session contiguous
        # versions (1, 2, 3) under _session_write_lock.
        redacted_tool_rows=(
            RedactedToolRow(
                "tc_a",
                "{}",
                StatePayload(data=CompositionStateData(), derived_from_state_id=None),
            ),
            RedactedToolRow(
                "tc_b",
                "{}",
                StatePayload(data=CompositionStateData(), derived_from_state_id=None),
            ),
            RedactedToolRow(
                "tc_c",
                "{}",
                StatePayload(data=CompositionStateData(), derived_from_state_id=None),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # ``get_messages`` is async (returns ChatMessageRecord objects); the
    # post-B2 ORDER BY sequence_no clause guarantees a stable order.
    import asyncio

    msgs = asyncio.run(service.get_messages(sid_uuid))
    roles = [m.role for m in msgs]
    # Exactly four rows from this turn: 1 assistant + 3 tool. No fork
    # rows, no system messages — _make_session left the chat empty.
    assert roles == ["assistant", "tool", "tool", "tool"], (
        f"intra-turn ordering broken: expected assistant before all tool rows, "
        f"got {roles!r} — see plan §14.7 (B2 fix). The pre-B2 ORDER BY created_at "
        f"would produce a nondeterministic permutation of these four roles."
    )
    # And the tool_call_id sequence is preserved (a→b→c, the order
    # ``redacted_tool_rows`` was supplied in).
    tool_ids = [m.tool_call_id for m in msgs if m.role == "tool"]
    assert tool_ids == ["tc_a", "tc_b", "tc_c"], f"intra-tool ordering broken: {tool_ids!r}"
