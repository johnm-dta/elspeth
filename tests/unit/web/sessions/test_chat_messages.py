"""Schema tests for the rev-4 chat_messages columns and CHECK constraints.

These tests run against an in-memory SQLite database to exercise the actual
database engine, not just SQLAlchemy metadata declarations. Schema-only
introspection would pass against any declared schema regardless of whether
the database enforces the declarations (closes spec QA F-8).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions import models
from tests.unit.web.conftest import _make_session


def test_tool_call_id_column_exists(engine):
    cols = {c.name for c in models.chat_messages_table.columns}
    assert "tool_call_id" in cols
    assert "parent_assistant_id" in cols
    assert "sequence_no" in cols
    assert "writer_principal" in cols


def test_role_tool_requires_tool_call_id(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_tool_call_id_role"):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="m1",
                    session_id="s1",
                    role="tool",
                    content="{}",
                    sequence_no=1,
                    writer_principal="compose_loop",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                    tool_call_id=None,  # rejected by CHECK
                    parent_assistant_id=None,
                )
            )


def test_role_assistant_rejects_tool_call_id(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_tool_call_id_role"):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="m1",
                    session_id="s1",
                    role="assistant",
                    content="hi",
                    sequence_no=1,
                    writer_principal="compose_loop",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                    tool_call_id="should_not_be_set",
                )
            )


def test_writer_principal_check_rejects_unknown(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_writer_principal"):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="m1",
                    session_id="s1",
                    role="user",
                    content="hi",
                    sequence_no=1,
                    writer_principal="rogue_writer",  # not in CHECK enum
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )


def test_audit_role_allows_unparented_internal_breadcrumb(engine):
    """Internal composer audit breadcrumbs that have no assistant parent
    are stored as role='audit', not role='tool'. This is the
    schema-compatible shape for existing _persist_llm_calls and
    failure-path _persist_tool_invocations rows until Phase 3 wires
    successful tool responses through persist_compose_turn."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="audit1",
                session_id="s1",
                role="audit",
                content='{"_kind": "llm_call_audit"}',
                sequence_no=1,
                writer_principal="compose_loop",
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )


# The ``ck_chat_messages_parent_role`` CHECK is a BICONDITIONAL on
# ``(role = 'tool') = (parent_assistant_id IS NOT NULL)``. The four
# tests below exercise all four cells of the truth table — closing
# synthesised review finding M7 / Q-F-05. Without these, a
# constraint that fired on one direction but not the other would
# pass the broader column-existence and tool-id tests above.


def test_parent_role_tool_with_parent_id_set_is_accepted(engine):
    """role='tool' AND parent_assistant_id IS NOT NULL — biconditional
    holds in the affirmative; insert succeeds. (Tool rows must also
    have tool_call_id and a parent assistant message in the session;
    the test sets both up.)"""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="a1",
                session_id="s1",
                role="assistant",
                content="hi",
                sequence_no=1,
                writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        conn.execute(
            insert(models.chat_messages_table).values(
                id="t1",
                session_id="s1",
                role="tool",
                content="{}",
                sequence_no=2,
                writer_principal="compose_loop",
                tool_call_id="tc_1",
                parent_assistant_id="a1",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )


def test_parent_role_tool_without_parent_id_rejected(engine):
    """role='tool' AND parent_assistant_id IS NULL — biconditional
    violated; CHECK rejects. tool_call_id is set so this test
    isolates the parent_role CHECK from the tool_call_id_role CHECK."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        with pytest.raises(IntegrityError, match="ck_chat_messages_parent_role"):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="t1",
                    session_id="s1",
                    role="tool",
                    content="{}",
                    sequence_no=1,
                    writer_principal="compose_loop",
                    tool_call_id="tc_1",
                    parent_assistant_id=None,  # rejected by parent_role CHECK
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )


def test_parent_role_non_tool_with_parent_id_rejected(engine):
    """role='user' AND parent_assistant_id IS NOT NULL — biconditional
    violated; CHECK rejects. tool_call_id is None (correct for user)
    so this test isolates the parent_role CHECK."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="a1",
                session_id="s1",
                role="assistant",
                content="hi",
                sequence_no=1,
                writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        with pytest.raises(IntegrityError, match="ck_chat_messages_parent_role"):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="u1",
                    session_id="s1",
                    role="user",
                    content="hello",
                    sequence_no=2,
                    writer_principal="route_user_message",
                    tool_call_id=None,
                    parent_assistant_id="a1",  # rejected: non-tool roles MUST have NULL parent
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )


def test_parent_role_non_tool_without_parent_id_is_accepted(engine):
    """role='user' AND parent_assistant_id IS NULL — biconditional
    holds in the negative; insert succeeds. The fourth and final
    cell of the truth table."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="u1",
                session_id="s1",
                role="user",
                content="hello",
                sequence_no=1,
                writer_principal="route_user_message",
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )


def test_session_sequence_no_unique(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="m1",
                session_id="s1",
                role="user",
                content="hi",
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        # SQLite reports unique-index violations with the column names
        # rather than the index name, so anchor the match on the
        # column-list rather than ``ix_chat_messages_session_sequence``.
        with pytest.raises(IntegrityError, match=r"UNIQUE.*chat_messages.*sequence_no"):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="m2",
                    session_id="s1",
                    role="assistant",
                    content="hi",
                    sequence_no=1,  # duplicate
                    writer_principal="compose_loop",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )


def test_direct_delete_assistant_row_is_blocked_and_session_cascade_purges_tools(engine):
    """Direct transcript-row deletes are blocked; session archival cascades.

    This is a Tier-1 referential-integrity invariant on the audit DB. The
    column-existence and CHECK-constraint tests above only inspect schema
    metadata; a future migration that drops the session-scoped cascade
    path would pass those tests silently and leave transcript rows behind
    after archival. This test binds the invariant to behaviour: insert
    one assistant + two child tool rows, prove direct assistant deletion
    is rejected, then delete the owning session and assert all rows are
    gone.

    FK enforcement on SQLite requires ``PRAGMA foreign_keys=ON`` for
    every connection in the pool. The ``engine`` fixture builds the
    engine via ``create_session_engine`` (see
    ``src/elspeth/web/sessions/engine.py``), which registers a
    ``connect`` event listener that issues the PRAGMA on every DBAPI
    connection and asserts on first connect that the PRAGMA took
    effect. Combined with the conftest's ``StaticPool`` (so worker
    threads see the same in-memory database), CASCADE genuinely fires
    here and cannot silently no-op the way bare ``create_engine``
    would.
    """
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Parent assistant row (no tool_call_id, no parent_assistant_id —
        # consistent with the biconditional CHECK constraints).
        conn.execute(
            insert(models.chat_messages_table).values(
                id="a1",
                session_id="s1",
                role="assistant",
                content="hi",
                sequence_no=1,
                writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        # Two child tool rows pointing back at a1; distinct tool_call_ids
        # so the partial unique index (Task 2) is not implicated here.
        conn.execute(
            insert(models.chat_messages_table).values(
                id="t1",
                session_id="s1",
                role="tool",
                content="{}",
                sequence_no=2,
                writer_principal="compose_loop",
                tool_call_id="tc_1",
                parent_assistant_id="a1",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        conn.execute(
            insert(models.chat_messages_table).values(
                id="t2",
                session_id="s1",
                role="tool",
                content="{}",
                sequence_no=3,
                writer_principal="compose_loop",
                tool_call_id="tc_2",
                parent_assistant_id="a1",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )

        # Sanity: all three rows are present before the delete.
        pre_rows = (
            conn.execute(select(models.chat_messages_table.c.id).where(models.chat_messages_table.c.session_id == "s1")).scalars().all()
        )
        assert sorted(pre_rows) == ["a1", "t1", "t2"]

        # Direct transcript-row deletion is not a supported cleanup path.
        with pytest.raises(IntegrityError, match="append-only"):
            conn.execute(delete(models.chat_messages_table).where(models.chat_messages_table.c.id == "a1"))

        blocked_rows = (
            conn.execute(select(models.chat_messages_table.c.id).where(models.chat_messages_table.c.session_id == "s1")).scalars().all()
        )
        assert sorted(blocked_rows) == ["a1", "t1", "t2"]

        # Whole-session deletion is the bounded lifecycle purge path; it
        # must still cascade through the self-referential assistant/tool FK.
        conn.execute(delete(models.sessions_table).where(models.sessions_table.c.id == "s1"))

        post_rows = (
            conn.execute(select(models.chat_messages_table.c.id).where(models.chat_messages_table.c.session_id == "s1")).scalars().all()
        )
        assert post_rows == [], (
            "Expected ON DELETE CASCADE from sessions to remove assistant "
            "and child tool rows during archival; found "
            f"orphaned rows: {post_rows!r}. If this test fails the audit "
            "DB has lost orphan-prevention — investigate the FK definition "
            "in models.chat_messages_table and the engine's foreign_keys "
            "PRAGMA before re-enabling."
        )


def test_tool_row_rejects_cross_session_parent_assistant(engine):
    """A tool row's parent assistant must belong to the same session.

    The old one-column FK on ``parent_assistant_id`` only proved that
    the assistant row existed somewhere in ``chat_messages``. It allowed
    a tool row in session B to point at an assistant row in session A,
    which would splice two transcripts together in the audit DB. The
    composite FK
    ``(parent_assistant_id, session_id) -> (chat_messages.id, chat_messages.session_id)``
    is the mechanical invariant.
    """
    with engine.begin() as conn:
        _make_session(conn, session_id="s_parent")
        _make_session(conn, session_id="s_child")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="a_parent",
                session_id="s_parent",
                role="assistant",
                content="hi",
                sequence_no=1,
                writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        with pytest.raises(
            IntegrityError,
            match=r"FOREIGN KEY|fk_chat_messages_parent_assistant_session",
        ):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="t_child",
                    session_id="s_child",
                    role="tool",
                    content="{}",
                    sequence_no=1,
                    writer_principal="compose_loop",
                    tool_call_id="tc_cross",
                    parent_assistant_id="a_parent",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )


# Task 2: partial unique index ``uq_chat_messages_tool_call_id`` on
# ``(session_id, tool_call_id) WHERE role='tool'``. The three tests
# below pin three independent failure modes a partial unique index can
# exhibit: (1) duplicate within session must collide, (2) duplicate
# across sessions must NOT collide (index is session-scoped), (3) NULL
# ``tool_call_id`` rows must NOT collide on each other (the partial
# predicate excludes role!='tool' rows entirely). Without all three
# the index could be silently broken in any direction.


def test_tool_call_id_unique_within_session(engine):
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        # Insert assistant first to satisfy parent_assistant_id FK
        conn.execute(
            insert(models.chat_messages_table).values(
                id="a1",
                session_id="s1",
                role="assistant",
                content="",
                sequence_no=1,
                writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        conn.execute(
            insert(models.chat_messages_table).values(
                id="t1",
                session_id="s1",
                role="tool",
                content="{}",
                sequence_no=2,
                writer_principal="compose_loop",
                tool_call_id="dup_id",
                parent_assistant_id="a1",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        # SQLite reports the column list rather than the partial-index
        # name; PostgreSQL may include the named index. Match both so
        # the regression fails for the intended uniqueness contract.
        with pytest.raises(
            IntegrityError,
            match=(
                r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
                r"|uq_chat_messages_tool_call_id)"
            ),
        ):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id="t2",
                    session_id="s1",
                    role="tool",
                    content="{}",
                    sequence_no=3,
                    writer_principal="compose_loop",
                    tool_call_id="dup_id",  # duplicate
                    parent_assistant_id="a1",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )


def test_tool_call_id_may_repeat_across_sessions(engine):
    """The unique index scope is (session_id, tool_call_id), not
    tool_call_id globally. Two sessions may receive the same provider
    tool_call_id without colliding."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        _make_session(conn, session_id="s2")
        for sid, assistant_id, tool_id, seq in (
            ("s1", "a1", "t1", 1),
            ("s2", "a2", "t2", 1),
        ):
            conn.execute(
                insert(models.chat_messages_table).values(
                    id=assistant_id,
                    session_id=sid,
                    role="assistant",
                    content="",
                    sequence_no=seq,
                    writer_principal="compose_loop",
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )
            conn.execute(
                insert(models.chat_messages_table).values(
                    id=tool_id,
                    session_id=sid,
                    role="tool",
                    content="{}",
                    sequence_no=seq + 1,
                    writer_principal="compose_loop",
                    tool_call_id="same_provider_id",
                    parent_assistant_id=assistant_id,
                    created_at=datetime(2026, 4, 30, tzinfo=UTC),
                )
            )


def test_tool_call_id_unique_only_within_role_tool(engine):
    """The partial unique index excludes role!='tool' rows so user/assistant
    rows with NULL tool_call_id do not all collide on NULL."""
    with engine.begin() as conn:
        _make_session(conn, session_id="s1")
        conn.execute(
            insert(models.chat_messages_table).values(
                id="u1",
                session_id="s1",
                role="user",
                content="hi",
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
        # Should not collide despite both having NULL tool_call_id.
        conn.execute(
            insert(models.chat_messages_table).values(
                id="a1",
                session_id="s1",
                role="assistant",
                content="hi",
                sequence_no=2,
                writer_principal="compose_loop",
                created_at=datetime(2026, 4, 30, tzinfo=UTC),
            )
        )
