"""Shared helpers for Phase 1 web integration tests.

Duplicates ``_make_session`` from
``tests/unit/web/conftest.py``; if either copy changes the
other must be updated to match. Engine fixtures are NOT shared because
integration tests own their engine fixture and call ``_make_session``
against whatever connection they have.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Connection, event, insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions import models
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


def _make_session(
    conn: Connection,
    *,
    session_id: str,
    user_id: str = "test_user",
    auth_provider_type: str = "local",
    title: str = "test session",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Insert a session row with every NOT NULL column populated."""
    now = created_at or datetime.now(UTC)
    conn.execute(
        insert(models.sessions_table).values(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=updated_at or now,
        )
    )


@pytest.fixture
def composer_test_client(tmp_path: Path) -> TestClient:
    """FastAPI ``TestClient`` configured for generic compose-route tests."""

    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    session_service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.phase3.integration"),
    )
    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = session_service
    app.state.session_engine = engine
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
    )
    app.state.composer_service = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.execution_service = None
    app.state.composer_progress_registry = None
    app.state.scoped_secret_resolver = None
    app.include_router(create_session_router())
    client = TestClient(app)
    client.app.state.phase3_engine = engine
    client.app.state.phase3_sessions_service = session_service
    return client


@pytest.fixture
def session_with_pending_compose_request(composer_test_client: TestClient) -> dict[str, str]:
    """Session with owner and user message ready for a compose request."""

    session_id = str(uuid4())
    user_message_id = str(uuid4())
    now = datetime.now(UTC)
    engine = composer_test_client.app.state.phase3_engine
    with engine.begin() as conn:
        _make_session(conn, session_id=session_id, user_id="alice")
        conn.execute(
            insert(models.chat_messages_table).values(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content="Build a pipeline",
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return {"session_id": session_id, "user_message_id": user_message_id}


@pytest.fixture
def session_with_composer_state(session_with_pending_compose_request: dict[str, str]) -> dict[str, str]:
    """Session fixture used by failed-turn route tests."""

    return session_with_pending_compose_request


@pytest.fixture
def inject_commit_OperationalError() -> object:
    """Integration-scope one-shot SQLAlchemy COMMIT failure hook."""

    def _install(engine: Engine) -> None:
        def _raise(_conn: object) -> None:
            event.remove(engine, "commit", _raise)
            raise OperationalError("COMMIT", {}, RuntimeError("phase3 commit failure"))

        event.listen(engine, "commit", _raise)

    return _install
