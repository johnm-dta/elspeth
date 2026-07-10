"""Fixtures for HTTP-layer guided endpoint integration tests.

Establishes SyncASGITestClient-based fixtures for testing the
/api/sessions/{id}/guided/* endpoint routes. Replicates the pattern from
tests/unit/web/sessions/test_fork.py: in-memory SQLite, mock auth, minimal
FastAPI app with session router mounted.

Scope: Fixtures live in this file and are available to all tests in
tests/integration/web/composer/guided/*.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from fastapi import FastAPI
from sqlalchemy.pool import StaticPool

from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


@pytest.fixture
def composer_test_client(tmp_path: Path) -> Iterator[TestClient]:
    """Yields a TestClient wrapping a minimal FastAPI app with session router.

    App state includes:
    - session_service (SessionServiceImpl with in-memory SQLite)
    - blob_service (BlobServiceImpl)
    - auth override: all requests authenticated as "alice"
    - rate_limiter (ComposerRateLimiter)
    - settings (WebSettings with test defaults)
    - audit_recorder (BufferingRecorder for test inspection)

    The app has create_session_router() mounted so POST /api/sessions routes
    are available for fixture smoke tests and guided endpoint tests.

    Usage:
        def test_something(composer_test_client):
            resp = composer_test_client.post("/api/sessions", json={"title": "x"})
            assert resp.status_code == 201
    """
    # Create in-memory session database with schema initialized
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)

    # Session and blob services
    session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided.conftest"),
    )
    blob_service = BlobServiceImpl(engine, tmp_path)

    # FastAPI app
    app = FastAPI()

    # Mock auth: all requests authenticated as "alice"
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    # App state: minimal set required by session router
    app.state.session_service = session_service
    app.state.session_engine = engine  # for guided step-2.5 recipe application
    app.state.blob_service = blob_service
    app.state.payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    app.state.scoped_secret_resolver = None
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
    )
    app.state.composer_service = None  # Not used in session router
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.catalog_service = create_catalog_service()

    # Audit recorder for test inspection (Phase 3 Task 3.4 will wire this)
    app.state.composer_recorder = BufferingRecorder()

    # post_guided_chat (elspeth-a8eeebb3aa) publishes composer-progress
    # snapshots the same way the freeform send_message route does — mirrors
    # the create_app() production wiring at app.py:930. Without this the
    # guided/chat route's unconditional _get_composer_progress_registry(request)
    # call AttributeErrors against this hand-rolled minimal app.
    app.state.composer_progress_registry = ComposerProgressRegistry()

    # Mount session router (sessions + guided endpoints)
    router = create_session_router()
    app.include_router(router)

    # Mount blobs router so tests can upload blobs via /api/sessions/{id}/blobs/inline
    blobs_router = create_blobs_router()
    app.include_router(blobs_router)

    # Wrap in TestClient and yield
    client = TestClient(app)
    yield client


@pytest.fixture
def audit_recorder(composer_test_client: TestClient) -> BufferingRecorder:
    """Yields the BufferingRecorder from the app state.

    Lets test code inspect emitted ComposerToolInvocation records during
    a request. The recorder is accessible via the app's state.

    Usage:
        def test_something(composer_test_client, audit_recorder):
            # Make a request that triggers composer operations
            resp = composer_test_client.post(...)
            # Inspect recorded invocations
            invocations = audit_recorder.invocations
            assert len(invocations) > 0
            assert invocations[0].tool_name == "..."
    """
    # Extract recorder from the test client's app state
    app = composer_test_client.app
    recorder: BufferingRecorder = app.state.composer_recorder
    return recorder
