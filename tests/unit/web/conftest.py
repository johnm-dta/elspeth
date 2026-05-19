"""Shared fixtures and helpers for Phase 1 web unit tests.

Hoisted from ``tests/unit/web/sessions/conftest.py`` to the parent
``tests/unit/web/`` package so both the sessions suite
(``tests/unit/web/sessions/test_*.py``) and the composer suite see the
same fixtures. pytest auto-loads parent-directory conftests for any
test under that directory tree, so no per-subdirectory shim is needed.

Provides:
- ``engine`` — an in-memory SQLite engine with FK enforcement applied
  via ``create_session_engine``'s connect-event listener (so EVERY
  pool checkout enforces FKs, not just the first), backed by
  ``StaticPool`` so worker threads dispatched via ``_run_sync`` see
  the same in-memory database. Schema is bootstrapped via
  ``initialize_session_schema`` (the same path production uses).
- ``_make_session`` — non-fixture helper that inserts a row into
  ``sessions_table`` with every NOT NULL column populated. Test code
  imports it explicitly via the absolute path
  ``from tests.unit.web.conftest import _make_session`` (matches the
  codebase convention for cross-package shared helpers — see
  ``tests/fixtures/`` and ``tests/helpers/`` import sites).

Why these live in conftest rather than each test file inlining them:
the four NOT NULL columns on ``sessions_table`` (``user_id``,
``auth_provider_type``, ``title``, ``updated_at``) and the
``StaticPool`` requirement are easy to forget. Centralising both
makes the fixture banned-pattern violations (plain ``create_engine``;
minimum-columns inserts) literally absent from the test code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import Connection, insert
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions import models
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import AuditAccessLogWriteError
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
    """Insert a session row with every NOT NULL column populated.

    The defaults are sufficient for tests that do not care about
    user/auth-provider fields. Tests that exercise auth-scoped
    behaviour should pass explicit ``user_id`` and
    ``auth_provider_type`` values.
    """
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
def test_client(tmp_path: Path) -> TestClient:
    """Sync ASGI test client with app state exposing ``sessions_service``."""

    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    service = SessionServiceImpl(
        eng,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.phase3.web"),
    )
    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    async def audit_access_log_write_error_handler(_request, _exc):
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "audit_access_log_write_failed",
                "detail": "Audit-grade transcript access could not be recorded; no audit-grade data returned.",
            },
        )

    app.dependency_overrides[get_current_user] = mock_user
    app.add_exception_handler(AuditAccessLogWriteError, audit_access_log_write_error_handler)
    app.state.session_service = service
    # Phase 8 Task 2: route-level telemetry emits read
    # ``request.app.state.sessions_telemetry``. The service already
    # holds the same container as ``_telemetry``; mirror it on
    # ``app.state`` so route-level tests can observe via either name
    # (matches production wiring in ``web/app.py:579``).
    app.state.sessions_telemetry = service._telemetry
    app.state.session_engine = eng
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.composer_service = None
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.execution_service = None
    app.state.composer_progress_registry = None
    app.state.scoped_secret_resolver = None
    app.include_router(create_session_router())
    client = TestClient(app)
    client.app.state.phase3_engine = eng
    client.app.state.phase3_sessions_service = service
    return client


@pytest.fixture
def inject_non_compose_loop_AuditIntegrityError() -> object:
    """Return a route-visible raiser for non-compose-loop audit failures."""

    def _raise() -> None:
        raise AuditIntegrityError("phase3 non-compose audit failure")

    return _raise


@pytest.fixture
def inject_audit_access_log_write_failure(monkeypatch: pytest.MonkeyPatch) -> object:
    """Force the future audit-grade access-log writer to fail at the boundary."""

    def _install(service: SessionServiceImpl) -> None:
        def _raise(*_args: object, **_kwargs: object) -> None:
            service._telemetry.audit_access_log_write_failed_total.add(1)
            raise AuditAccessLogWriteError("phase3 audit access log write failure")

        monkeypatch.setattr(service, "record_audit_grade_view", _raise, raising=False)

    return _install


@pytest.fixture
def session_with_user_assistant_tool_rows(engine):
    """Create user, assistant, and tool rows with monotonic ``sequence_no``."""

    session_id = str(uuid4())
    assistant_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        _make_session(conn, session_id=session_id, user_id="alice")
        conn.execute(
            insert(models.chat_messages_table),
            [
                {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "role": "user",
                    "content": "Build it",
                    "raw_content": None,
                    "tool_calls": None,
                    "tool_call_id": None,
                    "sequence_no": 1,
                    "writer_principal": "route_user_message",
                    "created_at": now,
                    "composition_state_id": None,
                    "parent_assistant_id": None,
                },
                {
                    "id": assistant_id,
                    "session_id": session_id,
                    "role": "assistant",
                    "content": "Calling a tool",
                    "raw_content": None,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_pipeline_state"}}],
                    "tool_call_id": None,
                    "sequence_no": 2,
                    "writer_principal": "compose_loop",
                    "created_at": now,
                    "composition_state_id": None,
                    "parent_assistant_id": None,
                },
                {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "role": "tool",
                    "content": "{}",
                    "raw_content": None,
                    "tool_calls": None,
                    "tool_call_id": "call_1",
                    "sequence_no": 3,
                    "writer_principal": "compose_loop",
                    "created_at": now,
                    "composition_state_id": None,
                    "parent_assistant_id": assistant_id,
                },
            ],
        )
    return {"session_id": session_id, "assistant_id": assistant_id}
