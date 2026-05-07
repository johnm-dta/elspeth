"""Round-trip tests for timezone-aware DateTime columns in session DB.

Verifies that DateTime(timezone=True) columns preserve timezone info
through SQLite storage and retrieval via SessionService.

SQLite stores timestamps as text and strips tzinfo on read.
SessionServiceImpl._ensure_utc() restores UTC on all datetime reads.
"""

from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine migrated to head."""
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


@pytest.fixture
def service(engine):
    """Create a SessionServiceImpl backed by the in-memory engine."""
    return SessionServiceImpl(engine)


class TestDatetimeTimezoneRoundTrip:
    """DateTime(timezone=True) columns must preserve tzinfo through SQLite."""

    @pytest.mark.asyncio
    async def test_session_created_at_preserves_timezone(self, service) -> None:
        """created_at on a freshly-created session must be timezone-aware."""
        session = await service.create_session("alice", "TZ Test", "local")
        fetched = await service.get_session(session.id)

        assert fetched.created_at is not None
        assert fetched.created_at.tzinfo is not None, "created_at lost timezone info after round-trip through SQLite"

    @pytest.mark.asyncio
    async def test_session_updated_at_preserves_timezone(self, service) -> None:
        """updated_at on a freshly-created session must be timezone-aware."""
        session = await service.create_session("alice", "TZ Test", "local")
        fetched = await service.get_session(session.id)

        assert fetched.updated_at is not None
        assert fetched.updated_at.tzinfo is not None, "updated_at lost timezone info after round-trip through SQLite"

    @pytest.mark.asyncio
    async def test_run_started_at_preserves_timezone(self, service) -> None:
        """started_at on a run record must be timezone-aware after retrieval."""
        session = await service.create_session("alice", "Run TZ Test", "local")
        state = await service.save_composition_state(
            session_id=session.id,
            state=CompositionStateData(source=None, nodes=[], edges=[], outputs=[], metadata_=None, is_valid=False),
        )
        run = await service.create_run(
            session_id=session.id,
            state_id=state.id,
        )
        fetched = await service.get_run(run.id)

        assert fetched.started_at is not None
        assert fetched.started_at.tzinfo is not None, "run.started_at lost timezone info after round-trip through SQLite"

    @pytest.mark.asyncio
    async def test_message_created_at_preserves_timezone(self, service) -> None:
        """created_at on a chat message must be timezone-aware after retrieval."""
        session = await service.create_session("alice", "Msg TZ Test", "local")
        await service.add_message(session.id, "user", "hello")

        messages = await service.get_messages(session.id)
        assert len(messages) >= 1
        msg = messages[0]

        assert msg.created_at is not None
        assert msg.created_at.tzinfo is not None, "message.created_at lost timezone info after round-trip through SQLite"
