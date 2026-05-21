"""Tests for session fork — service-level fork_session and route-level fork endpoint."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    InvalidForkTargetError,
)
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


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


class TestForkSession:
    """Tests for SessionServiceImpl.fork_session."""

    @pytest.mark.asyncio
    async def test_fork_creates_new_session_with_provenance(self, service) -> None:
        """Forked session has forked_from fields set."""
        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "Hi there", writer_principal="compose_loop")
        msg2 = await service.add_message(session.id, "user", "Do something", writer_principal="route_user_message")

        new_session, _messages, _state = await service.fork_session(
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

        _, messages, _ = await service.fork_session(
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
        _, _, copied_state = await service.fork_session(
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Build a different pipeline",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        assert copied_state.source == state_v1.source
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
            CompositionStateData(source=sources["orders"], sources=sources, is_valid=True),
            provenance="session_seed",
        )
        fork_msg = await service.add_message(
            session.id,
            "user",
            "Build this",
            composition_state_id=state.id,
            writer_principal="route_user_message",
        )

        _, _, copied_state = await service.fork_session(
            source_session_id=session.id,
            fork_message_id=fork_msg.id,
            new_message_content="Build that",
            user_id="alice",
            auth_provider_type="local",
        )

        assert copied_state is not None
        assert copied_state.source == sources["orders"]
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
            await service.fork_session(
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

        await service.fork_session(
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
            await service.fork_session(
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
            await service.fork_session(
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

        _, messages, _ = await service.fork_session(
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

        new_session, _messages, state = await service.fork_session(
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

        _, messages, _ = await service.fork_session(
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

        _, messages, _ = await service.fork_session(
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

        new_session, _new_messages, _ = await service.fork_session(
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

        new_session, _, _ = await service.fork_session(
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

        # 4 copied + system fork notice + new user message = 6 rows.
        assert seqs == [1, 2, 3, 4, 5, 6]

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

        new_session, _, _ = await service.fork_session(
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
                await service.fork_session(
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

        new_session, _, _ = await service.fork_session(
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

        new_session, new_messages, _ = await service.fork_session(
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
        assert len(audit_rows) == 1

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

        child_session, _, _ = await service.fork_session(
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
    async def test_fork_and_delete_parent_session_no_durable_history(self, service) -> None:
        """Archiving a fork parent without durable history physically deletes the parent."""
        from elspeth.web.sessions.protocol import SessionNotFoundError

        session = await service.create_session("alice", "Original", "local")
        await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        msg = await service.add_message(session.id, "user", "World", writer_principal="route_user_message")

        child_session, _, _ = await service.fork_session(
            source_session_id=session.id,
            fork_message_id=msg.id,
            new_message_content="Universe",
            user_id="alice",
            auth_provider_type="local",
        )

        await service.archive_session(session.id)

        with pytest.raises(SessionNotFoundError):
            await service.get_session(session.id)

        child = await service.get_session(child_session.id)
        assert child.forked_from_session_id == session.id


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

    return app, session_service, blob_service


class TestForkEndpoint:
    """Route-level tests for POST /api/sessions/{id}/fork."""

    @pytest.mark.asyncio
    async def test_fork_endpoint_creates_session(self, tmp_path) -> None:
        app, service, _ = _make_fork_app(tmp_path)
        client = TestClient(app)

        session = await service.create_session("alice", "Original", "local")
        msg = await service.add_message(session.id, "user", "Hello world", writer_principal="route_user_message")

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "from_message_id": str(msg.id),
                "new_message_content": "Hello universe",
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["session"]["forked_from_session_id"] == str(session.id)
        assert body["session"]["forked_from_message_id"] == str(msg.id)
        assert "(fork)" in body["session"]["title"]

        # New session should have system + edited user messages
        msgs = body["messages"]
        assert any(m["role"] == "system" for m in msgs)
        assert any(m["content"] == "Hello universe" for m in msgs)

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
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 201
        new_session_id = uuid.UUID(response.json()["session"]["id"])

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
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 201
        new_session_id = uuid.UUID(response.json()["session"]["id"])

        copied_blobs = await blob_service.list_blobs(new_session_id)
        assert len(copied_blobs) == 1
        copied_blob = copied_blobs[0]

        copied_state = await service.get_current_state(new_session_id)
        assert copied_state is not None
        assert copied_state.source is not None
        options = copied_state.source["options"]
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
                "from_message_id": str(msg.id),
                "new_message_content": "Process that instead",
            },
        )

        assert response.status_code == 201
        new_session_id = uuid.UUID(response.json()["session"]["id"])
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

        from elspeth.contracts.errors import AuditIntegrityError

        client = TestClient(app)
        with pytest.raises(AuditIntegrityError) as exc_info:
            client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "from_message_id": str(msg.id),
                    "new_message_content": "Process that instead",
                },
            )

        message = str(exc_info.value)
        assert "Tier 1" in message
        assert str(missing_blob_id) in message
        assert "source blob was not copied" in message

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
                "from_message_id": str(assistant_msg.id),
                "new_message_content": "Hi",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_fork_blob_quota_exceeded_returns_413(self, tmp_path) -> None:
        """Fork returns 413 and cleans up when blob quota is exceeded."""
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

        response = client.post(
            f"/api/sessions/{session.id}/fork",
            json={
                "from_message_id": str(msg.id),
                "new_message_content": "Go edited",
            },
        )

        assert response.status_code == 413

        # The partially created session should have been cleaned up
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

        The fork-rollback machinery in ``fork_from_message`` must archive
        the partially-created fork session so no orphan artifacts remain.
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

        # raise_server_exceptions=True (default) lets us inspect the actual
        # exception object so we can verify diagnostic content and cause chain.
        from elspeth.contracts.errors import AuditIntegrityError

        client = TestClient(app)
        with pytest.raises(AuditIntegrityError) as exc_info:
            client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "from_message_id": str(msg.id),
                    "new_message_content": "Hello edited",
                },
            )

        # Diagnostic names the tier, the offending value, and both state ids
        # so operators can locate the corrupted record.
        message = str(exc_info.value)
        assert "Tier 1" in message
        assert "blob_ref" in message
        assert "not-a-valid-uuid" in message

        # Cause chain preserves the original ValueError for forensics.
        assert isinstance(exc_info.value.__cause__, ValueError)

        # Fork-rollback archived the partially-created fork session — only
        # the original session remains visible to the owner.
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

        from elspeth.contracts.errors import AuditIntegrityError

        client = TestClient(app)
        with pytest.raises(AuditIntegrityError) as exc_info:
            client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "from_message_id": str(msg.id),
                    "new_message_content": "Hello edited",
                },
            )

        message = str(exc_info.value)
        assert "Tier 1" in message
        assert "blob_ref" in message
        assert "int" in message
        assert "UUID string" in message
        assert exc_info.value.__cause__ is None

        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_fork_non_quota_blob_error_archives_session(self, tmp_path) -> None:
        """Non-quota blob failures during fork must archive the new session.

        copy_blobs_for_fork can fail for reasons other than quota (missing
        blob row, filesystem error, DB disconnect).  The fork route must
        compensate by archiving the partially-created session.
        """
        app, service, blob_service = _make_fork_app(tmp_path)

        session = await service.create_session("alice", "Original", "local")
        await blob_service.create_blob(session.id, "data.csv", b"a,b\n1,2", "text/csv")
        msg = await service.add_message(session.id, "user", "Go", writer_principal="route_user_message")

        # Use raise_server_exceptions=False so the 500 is returned as an
        # HTTP response rather than propagated as a Python exception.
        client = TestClient(app, raise_server_exceptions=False)
        with patch.object(
            blob_service,
            "copy_blobs_for_fork",
            new=AsyncMock(side_effect=RuntimeError("disk I/O error")),
        ):
            response = client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "from_message_id": str(msg.id),
                    "new_message_content": "Go edited",
                },
            )

        assert response.status_code == 500

        # The fork session must have been cleaned up
        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1  # Only the original remains

    @pytest.mark.asyncio
    async def test_fork_state_rewrite_failure_archives_session(self, tmp_path) -> None:
        """Failure during state rewrite after blob copy must archive the fork.

        If save_composition_state fails after fork_session and blob copy
        have both committed, the fork session (and copied blobs) must be
        cleaned up so users don't see an orphaned half-initialised fork.
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
        with patch.object(
            service,
            "save_composition_state",
            new=AsyncMock(side_effect=RuntimeError("DB write failed")),
        ):
            response = client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "from_message_id": str(msg.id),
                    "new_message_content": "Go edited",
                },
            )

        assert response.status_code == 500

        # The fork session must have been cleaned up
        sessions = await service.list_sessions("alice", "local")
        assert len(sessions) == 1  # Only the original remains

    @pytest.mark.asyncio
    async def test_fork_cleanup_failure_preserves_primary_exception_with_note(self, tmp_path) -> None:
        """archive_session failure during rollback must not mask the real cause.

        If copy_blobs_for_fork raises and the compensating archive_session
        ALSO fails (e.g. shutil.rmtree on a locked blob dir), the operator
        must still see the original blob-copy failure as the headline.  A
        RecoveryFailed[...] note flags that the fork session row is now an
        orphan that needs manual cleanup.

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

        with (
            patch.object(
                blob_service,
                "copy_blobs_for_fork",
                new=AsyncMock(side_effect=primary),
            ),
            patch.object(
                service,
                "archive_session",
                new=AsyncMock(side_effect=cleanup),
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            client.post(
                f"/api/sessions/{session.id}/fork",
                json={
                    "from_message_id": str(msg.id),
                    "new_message_content": "Go edited",
                },
            )

        # Identity check: the propagated exception is the original primary,
        # not a re-wrap and not the cleanup OSError.
        assert exc_info.value is primary

        # RecoveryFailed[...] note attached for orphan-session visibility.
        notes = getattr(primary, "__notes__", [])
        assert any("RecoveryFailed[OSError]" in note for note in notes), f"expected RecoveryFailed[OSError] note, got: {notes!r}"
        assert any("permission denied removing blob dir" in note for note in notes)
        # Note must identify the orphan session id so operators can clean up.
        assert any("manual cleanup" in note.lower() for note in notes)
