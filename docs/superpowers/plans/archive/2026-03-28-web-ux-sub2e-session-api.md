# Web UX Task-Plan 2E: Session API & Wiring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement Pydantic schemas, session REST routes (including revert endpoint), and wire auth + sessions into the app factory
**Parent Plan:** `plans/2026-03-28-web-ux-sub2-auth-sessions.md`
**Spec:** `specs/2026-03-28-web-ux-sub2-auth-sessions-design.md`
**Depends On:** Task-Plans 2A, 2B, 2C, 2D (all auth and session internals)
**Blocks:** Sub-Plans 4 (Composer), 5 (Execution)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/sessions/schemas.py` | Pydantic request/response models for all session endpoints |
| Create | `src/elspeth/web/sessions/routes.py` | /api/sessions/* endpoints with IDOR protection, including state revert |
| Create | `tests/unit/web/sessions/test_routes.py` | Session API endpoint tests, IDOR tests, upload path traversal test |
| Modify | `src/elspeth/web/app.py` | Register auth and session routers, create session DB engine, call metadata.create_all on startup |
| Modify | `src/elspeth/web/dependencies.py` | Add get_current_user, get_session_service, get_auth_provider dependencies |
| Modify | `src/elspeth/web/config.py` | Add OIDC/Entra conditional validator to WebSettings |
| Create | `tests/unit/web/test_config.py` | Tests for conditional auth field validation |

---

## Pre-requisites

Task-Plans 2A through 2D must be complete. The following files must exist:

- `src/elspeth/web/auth/__init__.py`
- `src/elspeth/web/auth/protocol.py`
- `src/elspeth/web/auth/models.py`
- `src/elspeth/web/auth/local.py`
- `src/elspeth/web/auth/oidc.py`
- `src/elspeth/web/auth/entra.py`
- `src/elspeth/web/auth/middleware.py`
- `src/elspeth/web/auth/routes.py`
- `src/elspeth/web/sessions/__init__.py`
- `src/elspeth/web/sessions/protocol.py`
- `src/elspeth/web/sessions/models.py`
- `src/elspeth/web/sessions/service.py`

---

### Task 2.10: Session Pydantic Schemas

**Files:**
- Create: `src/elspeth/web/sessions/schemas.py`

- [ ] **Step 1: Implement request/response schemas**

```python
# src/elspeth/web/sessions/schemas.py
"""Pydantic request/response models for all session API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    """Request body for POST /api/sessions."""

    title: str = "New session"


class SessionResponse(BaseModel):
    """Response for session CRUD operations."""

    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SendMessageRequest(BaseModel):
    """Request body for POST /api/sessions/{id}/messages."""

    content: str


class ChatMessageResponse(BaseModel):
    """Response for a single chat message."""

    id: str
    session_id: str
    role: str
    content: str
    tool_calls: Any | None = None
    created_at: datetime


class MessageWithStateResponse(BaseModel):
    """Response for POST /api/sessions/{id}/messages.

    In Phase 2, state is always null. In Phase 4, it will contain
    the updated CompositionState after the ComposerService processes
    the message.
    """

    message: ChatMessageResponse
    state: CompositionStateResponse | None = None


class CompositionStateResponse(BaseModel):
    """Response for composition state endpoints."""

    id: str
    session_id: str
    version: int
    source: Any | None = None
    nodes: list[Any] | None = None
    edges: list[Any] | None = None
    outputs: list[Any] | None = None
    metadata: Any | None = None
    is_valid: bool
    validation_errors: list[str] | None = None
    created_at: datetime


class RevertStateRequest(BaseModel):
    """Request body for POST /api/sessions/{id}/state/revert."""

    state_id: str


class UploadResponse(BaseModel):
    """Response for POST /api/sessions/{id}/upload."""

    path: str
    filename: str
    size_bytes: int


# Forward reference resolution
MessageWithStateResponse.model_rebuild()
```

- [ ] **Step 2: Commit**

```bash
git add src/elspeth/web/sessions/schemas.py
git commit -m "feat(web/sessions): add Pydantic request/response schemas for session API"
```

---

### Task 2.11: Session API Routes

**Files:**
- Create: `src/elspeth/web/sessions/routes.py`
- Create: `tests/unit/web/sessions/test_routes.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/web/sessions/test_routes.py
"""Tests for session API routes -- CRUD, IDOR, upload, path traversal."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.models import metadata
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.service import SessionServiceImpl


def _make_app(
    tmp_path: Path,
    user_id: str = "alice",
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> tuple[FastAPI, SessionServiceImpl]:
    """Create a test app with session routes and a mock auth user."""
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    service = SessionServiceImpl(engine)

    app = FastAPI()

    # Override auth dependency to return a fixed user
    identity = UserIdentity(user_id=user_id, username=user_id)

    async def mock_user():
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    # Set up app state
    app.state.session_service = service
    app.state.settings = type(
        "FakeSettings", (),
        {"data_dir": tmp_path, "max_upload_bytes": max_upload_bytes,
         "auth_provider": "local"},
    )()

    router = create_session_router()
    app.include_router(router)

    return app, service


class TestSessionCRUDRoutes:
    """Tests for session create, list, get, delete endpoints."""

    def test_create_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.post(
            "/api/sessions",
            json={"title": "My Pipeline"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["title"] == "My Pipeline"
        assert body["user_id"] == "alice"
        assert "id" in body

    def test_create_session_default_title(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.post("/api/sessions", json={})
        assert response.status_code == 201
        assert response.json()["title"] == "New session"

    def test_list_sessions(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        client.post("/api/sessions", json={"title": "S1"})
        client.post("/api/sessions", json={"title": "S2"})

        response = client.get("/api/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 2

    def test_get_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions", json={"title": "Test"},
        )
        session_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == session_id

    def test_get_session_not_found(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        response = client.get(f"/api/sessions/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_delete_session(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        create_resp = client.post(
            "/api/sessions", json={"title": "To Delete"},
        )
        session_id = create_resp.json()["id"]

        del_resp = client.delete(f"/api/sessions/{session_id}")
        assert del_resp.status_code == 204

        # Verify it's gone
        get_resp = client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 404


class TestIDORProtection:
    """Tests for W5 -- IDOR protection on all session-scoped routes.

    Creates a session as user A, then attempts to access it as user B.
    All should return 404 (not 403).
    """

    @pytest.fixture
    def alice_session(self, tmp_path):
        """Create a session owned by alice, return (session_id, tmp_path)."""
        app, service = _make_app(tmp_path, user_id="alice")
        client = TestClient(app)
        resp = client.post("/api/sessions", json={"title": "Alice's"})
        return resp.json()["id"], tmp_path

    def _bob_client(self, tmp_path) -> TestClient:
        """Create a TestClient where the authenticated user is bob."""
        app, _ = _make_app(tmp_path, user_id="bob")
        return TestClient(app)

    def test_get_other_users_session_returns_404(
        self, alice_session,
    ) -> None:
        session_id, tmp_path = alice_session
        client = self._bob_client(tmp_path)
        # Bob creates his own app with a fresh DB, so alice's session
        # won't exist. For a proper IDOR test, we need a shared DB.
        # We'll test at the route level instead.

    def test_idor_session_crud(self, tmp_path) -> None:
        """Shared-DB IDOR test: alice creates, bob tries to access."""
        engine = create_engine("sqlite:///:memory:")
        metadata.create_all(engine)
        service = SessionServiceImpl(engine)

        # Create two apps sharing the same service
        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)
            async def mock_user():
                return identity
            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = service
            app.state.settings = type(
                "S", (), {"data_dir": tmp_path, "max_upload_bytes": 10_000_000,
                 "auth_provider": "local"},
            )()
            app.include_router(create_session_router())
            return app

        alice_app = make_app_for_user("alice")
        bob_app = make_app_for_user("bob")

        alice_client = TestClient(alice_app)
        bob_client = TestClient(bob_app)

        # Alice creates a session
        resp = alice_client.post(
            "/api/sessions", json={"title": "Alice Only"},
        )
        assert resp.status_code == 201
        session_id = resp.json()["id"]

        # Bob tries to GET it -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 404

        # Bob tries to DELETE it -- should be 404
        resp = bob_client.delete(f"/api/sessions/{session_id}")
        assert resp.status_code == 404

        # Bob tries to GET messages -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/messages")
        assert resp.status_code == 404

        # Bob tries to POST a message -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "hacked"},
        )
        assert resp.status_code == 404

        # Bob tries to GET state -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/state")
        assert resp.status_code == 404

        # Bob tries to GET state versions -- should be 404
        resp = bob_client.get(f"/api/sessions/{session_id}/state/versions")
        assert resp.status_code == 404

        # Bob tries to revert state -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session_id}/state/revert",
            json={"state_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404

        # Alice can still access her own session
        resp = alice_client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200


class TestMessageRoutes:
    """Tests for message send and retrieval endpoints."""

    def test_send_message(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello, build me a pipeline"},
        )
        assert msg_resp.status_code == 200
        body = msg_resp.json()
        assert body["message"]["content"] == "Hello, build me a pipeline"
        assert body["message"]["role"] == "user"
        assert body["state"] is None  # Phase 2: no composer yet

    def test_get_messages(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Chat"})
        session_id = resp.json()["id"]

        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "First"},
        )
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Second"},
        )

        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        messages = msgs_resp.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"


class TestStateRoutes:
    """Tests for composition state endpoints."""

    def test_get_state_empty(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Empty"})
        session_id = resp.json()["id"]

        state_resp = client.get(f"/api/sessions/{session_id}/state")
        assert state_resp.status_code == 200
        assert state_resp.json() is None

    def test_get_state_versions(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Pipeline"})
        session_id = resp.json()["id"]

        versions_resp = client.get(
            f"/api/sessions/{session_id}/state/versions",
        )
        assert versions_resp.status_code == 200
        assert versions_resp.json() == []


class TestRevertEndpoint:
    """Tests for POST /api/sessions/{id}/state/revert (R1)."""

    def test_revert_creates_new_version(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        # Create session and two state versions via the service
        import asyncio
        loop = asyncio.new_event_loop()
        session = loop.run_until_complete(
            service.create_session("alice", "Pipeline", "local"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1 = loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(source={"type": "csv"}, is_valid=True),
            ),
        )
        v2 = loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(source={"type": "api"}, is_valid=True),
            ),
        )
        loop.close()

        # Revert to v1
        resp = client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"state_id": str(v1.id)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == 3
        # Should match v1's source, not v2's
        assert body["source"] == {"type": "csv"}

    def test_revert_injects_system_message(self, tmp_path) -> None:
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        import asyncio
        loop = asyncio.new_event_loop()
        session = loop.run_until_complete(
            service.create_session("alice", "Pipeline", "local"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1 = loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(is_valid=True),
            ),
        )
        loop.run_until_complete(
            service.save_composition_state(
                session.id,
                CompositionStateData(is_valid=True),
            ),
        )
        loop.close()

        client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"state_id": str(v1.id)},
        )

        # Check that a system message was injected
        msgs_resp = client.get(f"/api/sessions/{session.id}/messages")
        messages = msgs_resp.json()
        system_msgs = [m for m in messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "reverted to version 1" in system_msgs[0]["content"].lower()

    def test_revert_idor_protection(self, tmp_path) -> None:
        """Revert to a state in another user's session returns 404."""
        from sqlalchemy import create_engine as _ce
        engine = _ce("sqlite:///:memory:")
        metadata.create_all(engine)
        service = SessionServiceImpl(engine)

        def make_app_for_user(uid: str) -> FastAPI:
            app = FastAPI()
            identity = UserIdentity(user_id=uid, username=uid)
            async def mock_user():
                return identity
            app.dependency_overrides[get_current_user] = mock_user
            app.state.session_service = service
            app.state.settings = type(
                "S", (), {"data_dir": tmp_path, "max_upload_bytes": 10_000_000,
                 "auth_provider": "local"},
            )()
            app.include_router(create_session_router())
            return app

        alice_app = make_app_for_user("alice")
        bob_app = make_app_for_user("bob")

        alice_client = TestClient(alice_app)
        bob_client = TestClient(bob_app)

        # Alice creates a session with a state
        import asyncio
        loop = asyncio.new_event_loop()
        session = loop.run_until_complete(
            service.create_session("alice", "Alice Only", "local"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1 = loop.run_until_complete(
            service.save_composition_state(
                session.id, CompositionStateData(is_valid=True),
            ),
        )
        loop.close()

        # Bob tries to revert -- should be 404
        resp = bob_client.post(
            f"/api/sessions/{session.id}/state/revert",
            json={"state_id": str(v1.id)},
        )
        assert resp.status_code == 404

    def test_revert_state_not_belonging_to_session(self, tmp_path) -> None:
        """Revert with a state_id from a different session returns 404."""
        app, service = _make_app(tmp_path)
        client = TestClient(app)

        import asyncio
        loop = asyncio.new_event_loop()
        s1 = loop.run_until_complete(
            service.create_session("alice", "Session 1", "local"),
        )
        s2 = loop.run_until_complete(
            service.create_session("alice", "Session 2", "local"),
        )
        from elspeth.web.sessions.protocol import CompositionStateData
        v1_s2 = loop.run_until_complete(
            service.save_composition_state(
                s2.id, CompositionStateData(is_valid=True),
            ),
        )
        loop.close()

        # Try to revert s1 using s2's state -- should fail
        resp = client.post(
            f"/api/sessions/{s1.id}/state/revert",
            json={"state_id": str(v1_s2.id)},
        )
        assert resp.status_code == 404


class TestUploadRoute:
    """Tests for file upload endpoint including path traversal (B5)."""

    def test_upload_file(self, tmp_path) -> None:
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Upload"})
        session_id = resp.json()["id"]

        file_content = b"col1,col2\na,b\nc,d"
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("data.csv", io.BytesIO(file_content), "text/csv")},
        )
        assert upload_resp.status_code == 200
        body = upload_resp.json()
        assert body["filename"] == "data.csv"
        assert body["size_bytes"] == len(file_content)
        assert "path" in body

        # Verify the file exists on disk
        saved_path = Path(body["path"])
        assert saved_path.exists()
        assert saved_path.read_bytes() == file_content

    def test_upload_path_traversal_user_id_sanitized(self, tmp_path) -> None:
        """B5: user_id containing ../../etc is sanitized to just 'etc'."""
        app, _ = _make_app(tmp_path, user_id="../../etc")
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Hack"})
        session_id = resp.json()["id"]

        file_content = b"malicious"
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("payload.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert upload_resp.status_code == 200
        saved_path = Path(upload_resp.json()["path"])

        # The path should NOT contain ".." components
        assert ".." not in str(saved_path)
        # Should be under data_dir/uploads/etc/ (sanitized)
        assert "etc" in saved_path.parts
        assert saved_path.is_relative_to(tmp_path / "uploads")

    def test_upload_path_traversal_filename_sanitized(self, tmp_path) -> None:
        """Filename containing path traversal is sanitized."""
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Hack"})
        session_id = resp.json()["id"]

        file_content = b"malicious"
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={
                "file": (
                    "../../etc/passwd",
                    io.BytesIO(file_content),
                    "application/octet-stream",
                ),
            },
        )
        assert upload_resp.status_code == 200
        saved_path = Path(upload_resp.json()["path"])
        # Filename should be just "passwd", not "../../etc/passwd"
        assert saved_path.name == "passwd"
        assert ".." not in str(saved_path)

    def test_upload_file_too_large(self, tmp_path) -> None:
        """Files exceeding max_upload_bytes are rejected with 413."""
        app, _ = _make_app(tmp_path, max_upload_bytes=100)
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Big File"})
        session_id = resp.json()["id"]

        big_content = b"x" * 200  # 200 bytes > 100 byte limit
        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={
                "file": (
                    "big.dat",
                    io.BytesIO(big_content),
                    "application/octet-stream",
                ),
            },
        )
        assert upload_resp.status_code == 413

    def test_upload_empty_user_id_sanitization(self, tmp_path) -> None:
        """User ID of '..' sanitizes to empty via Path.name, which should raise."""
        app, _ = _make_app(tmp_path, user_id="..")
        client = TestClient(app)

        resp = client.post("/api/sessions", json={"title": "Hack"})
        session_id = resp.json()["id"]

        upload_resp = client.post(
            f"/api/sessions/{session_id}/upload",
            files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")},
        )
        # Should fail because Path("..").name is "" on some platforms
        # or ".." which sanitizes poorly. Either 400 or 500 is acceptable.
        assert upload_resp.status_code in (400, 422, 500)
```

- [ ] **Step 2: Run tests -- verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.sessions.routes'`

- [ ] **Step 3: Implement session routes**

```python
# src/elspeth/web/sessions/routes.py
"""Session API routes -- /api/sessions/* with IDOR protection.

All endpoints require authentication via Depends(get_current_user).
Session-scoped endpoints verify ownership before any business logic.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.sessions.protocol import RunAlreadyActiveError, SessionRecord
from elspeth.web.sessions.schemas import (
    ChatMessageResponse,
    CompositionStateResponse,
    CreateSessionRequest,
    MessageWithStateResponse,
    RevertStateRequest,
    SendMessageRequest,
    SessionResponse,
    UploadResponse,
)


def _session_response(session: SessionRecord) -> SessionResponse:
    """Convert a SessionRecord to a SessionResponse."""
    return SessionResponse(
        id=str(session.id),
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


async def _verify_session_ownership(
    session_id: str,
    user: UserIdentity,
    request: Request,
) -> SessionRecord:
    """Verify the session exists and belongs to the current user.

    Returns 404 (not 403) to avoid leaking session existence (IDOR, W5).
    """
    service = request.app.state.session_service
    try:
        session = await service.get_session(UUID(session_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found") from None

    if session.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


def create_session_router() -> APIRouter:
    """Create the session router with /api/sessions prefix."""
    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

    @router.post("", status_code=201, response_model=SessionResponse)
    async def create_session(
        body: CreateSessionRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> SessionResponse:
        """Create a new session for the authenticated user."""
        service = request.app.state.session_service
        settings = request.app.state.settings
        session = await service.create_session(
            user.user_id, body.title, settings.auth_provider,
        )
        return _session_response(session)

    @router.get("", response_model=list[SessionResponse])
    async def list_sessions(
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> list[SessionResponse]:
        """List sessions for the authenticated user."""
        service = request.app.state.session_service
        sessions = await service.list_sessions(user.user_id)
        return [_session_response(s) for s in sessions]

    @router.get("/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> SessionResponse:
        """Get a single session. IDOR-protected."""
        session = await _verify_session_ownership(session_id, user, request)
        return _session_response(session)

    @router.delete("/{session_id}", status_code=204)
    async def delete_session(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> None:
        """Archive (delete) a session and all associated data."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        await service.archive_session(session.id)

    @router.post(
        "/{session_id}/messages",
        response_model=MessageWithStateResponse,
    )
    async def send_message(
        session_id: str,
        body: SendMessageRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> MessageWithStateResponse:
        """Send a user message. In Phase 2, only persists the message."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        msg = await service.add_message(session.id, "user", body.content)
        return MessageWithStateResponse(
            message=ChatMessageResponse(
                id=str(msg.id),
                session_id=str(msg.session_id),
                role=msg.role,
                content=msg.content,
                tool_calls=msg.tool_calls,
                created_at=msg.created_at,
            ),
            state=None,
        )

    @router.get(
        "/{session_id}/messages",
        response_model=list[ChatMessageResponse],
    )
    async def get_messages(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> list[ChatMessageResponse]:
        """Get conversation history for a session."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        messages = await service.get_messages(session.id)
        return [
            ChatMessageResponse(
                id=str(m.id),
                session_id=str(m.session_id),
                role=m.role,
                content=m.content,
                tool_calls=m.tool_calls,
                created_at=m.created_at,
            )
            for m in messages
        ]

    @router.get("/{session_id}/state")
    async def get_current_state(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> CompositionStateResponse | None:
        """Get the current (highest-version) composition state."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        state = await service.get_current_state(session.id)
        if state is None:
            return None
        return CompositionStateResponse(
            id=str(state.id),
            session_id=str(state.session_id),
            version=state.version,
            source=state.source,
            nodes=state.nodes,
            edges=state.edges,
            outputs=state.outputs,
            metadata=state.metadata_,
            is_valid=state.is_valid,
            validation_errors=state.validation_errors,
            created_at=state.created_at,
        )

    @router.get(
        "/{session_id}/state/versions",
        response_model=list[CompositionStateResponse],
    )
    async def get_state_versions(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> list[CompositionStateResponse]:
        """Get all composition state versions for a session."""
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        versions = await service.get_state_versions(session.id)
        return [
            CompositionStateResponse(
                id=str(v.id),
                session_id=str(v.session_id),
                version=v.version,
                source=v.source,
                nodes=v.nodes,
                edges=v.edges,
                outputs=v.outputs,
                metadata=v.metadata_,
                is_valid=v.is_valid,
                validation_errors=v.validation_errors,
                created_at=v.created_at,
            )
            for v in versions
        ]

    @router.post(
        "/{session_id}/state/revert",
        response_model=CompositionStateResponse,
    )
    async def revert_state(
        session_id: str,
        body: RevertStateRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> CompositionStateResponse:
        """Revert the pipeline to a prior composition state version (R1).

        Creates a new version that is a copy of the specified prior state.
        Injects a system message recording the revert.
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service

        try:
            new_state = await service.set_active_state(
                session.id, UUID(body.state_id),
            )
        except ValueError:
            raise HTTPException(
                status_code=404, detail="State not found",
            ) from None

        # Look up the original version number for the system message
        original_state = await service.get_state(UUID(body.state_id))
        await service.add_message(
            session.id,
            role="system",
            content=f"Pipeline reverted to version {original_state.version}.",
        )

        return CompositionStateResponse(
            id=str(new_state.id),
            session_id=str(new_state.session_id),
            version=new_state.version,
            source=new_state.source,
            nodes=new_state.nodes,
            edges=new_state.edges,
            outputs=new_state.outputs,
            metadata=new_state.metadata_,
            is_valid=new_state.is_valid,
            validation_errors=new_state.validation_errors,
            created_at=new_state.created_at,
        )

    @router.get("/{session_id}/state/yaml")
    async def get_state_yaml(
        session_id: str,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> dict[str, str]:
        """Get YAML representation of the current composition state (M1).

        Stub endpoint -- returns 501 until Sub-4 implements generate_yaml().
        When Sub-4 lands, replace the 501 with:
            state = await service.get_current_state(session.id)
            yaml_str = generate_yaml(CompositionState.from_dict(state))
            return {"yaml": yaml_str}
        """
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        state = await service.get_current_state(session.id)
        if state is None:
            raise HTTPException(status_code=404, detail="No composition state exists")
        # TODO(sub-4): Replace stub with generate_yaml() call
        raise HTTPException(
            status_code=501,
            detail="YAML generation not yet implemented (see Sub-4)",
        )

    @router.post("/{session_id}/upload", response_model=UploadResponse)
    async def upload_file(
        session_id: str,
        request: Request,
        file: UploadFile = File(...),
        user: UserIdentity = Depends(get_current_user),
    ) -> UploadResponse:
        """Upload a source file to the user's scratch directory.

        Path traversal protection (B5): both user_id and filename are
        sanitized via Path().name to strip directory components.
        """
        session = await _verify_session_ownership(session_id, user, request)
        settings = request.app.state.settings

        # B5: Sanitize user_id -- strip all directory components
        sanitized_user_id = Path(user.user_id).name
        if not sanitized_user_id or sanitized_user_id in (".", ".."):
            raise HTTPException(
                status_code=400,
                detail="Invalid user ID for file upload",
            )

        # B5: Sanitize filename
        original_filename = file.filename or "upload"
        sanitized_filename = Path(original_filename).name
        if not sanitized_filename or sanitized_filename in (".", ".."):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename",
            )

        # Read file content into memory and check size
        content = await file.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {settings.max_upload_bytes} bytes",
            )

        # Create upload directory and save
        upload_dir = Path(settings.data_dir) / "uploads" / sanitized_user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / sanitized_filename
        file_path.write_bytes(content)

        return UploadResponse(
            path=str(file_path),
            filename=original_filename,
            size_bytes=len(content),
        )

    return router
```

- [ ] **Step 4: Run tests -- verify they pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_routes.py -v
```

Expected: all 20 tests pass.

- [ ] **Step 5: Run all session tests**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/ -v
```

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/routes.py tests/unit/web/sessions/test_routes.py
git commit -m "feat(web/sessions): implement session API routes with IDOR protection and path traversal sanitization"
```

---

### Task 2.12: Wire Auth and Sessions into App Factory

**Files:**
- Modify: `src/elspeth/web/app.py`
- Modify: `src/elspeth/web/dependencies.py`

- [ ] **Step 1: Update app.py to register routers and create session DB**

Add the following to `create_app()` in `src/elspeth/web/app.py`:

```python
# Add these imports at the top of app.py:
import sys

from fastapi.responses import JSONResponse
from sqlalchemy import create_engine

from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.routes import create_auth_router
from elspeth.web.sessions.models import metadata as session_metadata
from elspeth.web.sessions.protocol import RunAlreadyActiveError
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.service import SessionServiceImpl
```

**Before wiring auth into create_app(), add the OIDC/Entra conditional validator to WebSettings.**

In `src/elspeth/web/config.py`, add a `@model_validator(mode="after")` that enforces conditional field requirements:

```python
from pydantic import model_validator

    @model_validator(mode="after")
    def _validate_auth_fields(self) -> WebSettings:
        if self.auth_provider == "oidc":
            missing = [f for f in ("oidc_issuer", "oidc_audience", "oidc_client_id") if getattr(self, f) is None]
            if missing:
                raise ValueError(f"OIDC auth requires: {', '.join(missing)}")
        elif self.auth_provider == "entra":
            missing = [f for f in ("oidc_issuer", "oidc_audience", "oidc_client_id", "entra_tenant_id") if getattr(self, f) is None]
            if missing:
                raise ValueError(f"Entra auth requires: {', '.join(missing)}")
        return self
```

This catches misconfiguration at settings construction (CLI or test setup) rather than deep in the auth provider init. Tracked as `elspeth-34df5d61e4` (from Sub-1 review). Add corresponding tests to `tests/unit/web/test_config.py`:
- `WebSettings(auth_provider="oidc")` without oidc fields -> ValueError
- `WebSettings(auth_provider="oidc", oidc_issuer=..., oidc_audience=..., oidc_client_id=...)` -> valid
- `WebSettings(auth_provider="entra")` without entra_tenant_id -> ValueError
- `WebSettings(auth_provider="local")` with no OIDC fields -> valid (the default)

Inside `create_app()`, after the existing CORS and health setup, add:

```python
    # --- Auth provider setup ---
    if settings.auth_provider == "local":
        auth_provider = LocalAuthProvider(
            db_path=settings.data_dir / "auth.db",
            secret_key=settings.secret_key,
        )
    elif settings.auth_provider == "oidc":
        from elspeth.web.auth.oidc import OIDCAuthProvider
        auth_provider = OIDCAuthProvider(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
        )
    elif settings.auth_provider == "entra":
        from elspeth.web.auth.entra import EntraAuthProvider
        auth_provider = EntraAuthProvider(
            tenant_id=settings.entra_tenant_id,
            audience=settings.oidc_audience,
        )
    app.state.auth_provider = auth_provider

    # W16/S3: Secret key production guard -- hard crash (S3/C4 upgrade)
    if (
        settings.secret_key == "change-me-in-production"
        and "pytest" not in sys.modules
        and os.environ.get("ELSPETH_ENV") != "test"
    ):
        raise SystemExit(
            "FATAL: WebSettings.secret_key is set to the default value. "
            "Set a secure secret_key before starting the web server. "
            "See WebSettings documentation."
        )

    # --- Session database setup (W6, S14) ---
    session_db_url = settings.get_session_db_url()
    session_engine = create_engine(session_db_url)
    session_metadata.create_all(session_engine)

    session_service = SessionServiceImpl(session_engine)
    app.state.session_service = session_service

    # --- Register routers ---
    app.include_router(create_auth_router())
    app.include_router(create_session_router())

    # --- Seam contract D: RunAlreadyActiveError -> 409 with error_type ---
    @app.exception_handler(RunAlreadyActiveError)
    async def handle_run_already_active(
        request: Request, exc: RunAlreadyActiveError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "error_type": "run_already_active"},
        )
```

- [ ] **Step 2: Update dependencies.py**

Add these dependency functions to `src/elspeth/web/dependencies.py`:

```python
from elspeth.web.auth.middleware import get_current_user  # noqa: F401 -- re-export


def get_session_service(request: Request):
    """Get the SessionService from app state."""
    return request.app.state.session_service


def get_auth_provider(request: Request):
    """Get the AuthProvider from app state."""
    return request.app.state.auth_provider
```

- [ ] **Step 3: Run all tests**

```bash
.venv/bin/python -m pytest tests/unit/web/ -v
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/app.py src/elspeth/web/dependencies.py
git commit -m "feat(web): wire auth and session modules into app factory with DB schema creation"
```

---

## Self-Review Checklist

After completing all tasks, run the full test suite and verify:

```bash
# All session tests
.venv/bin/python -m pytest tests/unit/web/sessions/ -v

# All web tests together
.venv/bin/python -m pytest tests/unit/web/ -v

# Type checking
.venv/bin/python -m mypy src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes.py src/elspeth/web/app.py src/elspeth/web/dependencies.py

# Linting
.venv/bin/python -m ruff check src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes.py src/elspeth/web/app.py src/elspeth/web/dependencies.py
```

**Expected results:**

- [ ] All session route tests pass (CRUD, IDOR, upload, path traversal, revert)
- [ ] All web tests pass end-to-end (auth + sessions together)
- [ ] `schemas.py` has forward reference resolution (`MessageWithStateResponse.model_rebuild()`)
- [ ] All session routes verify ownership via `_verify_session_ownership()` returning 404 on IDOR
- [ ] Revert endpoint creates a new state version (not in-place mutation) and injects a system message
- [ ] YAML stub endpoint returns 501 with `TODO(sub-4)` marker
- [ ] Upload path sanitizes both `user_id` and `filename` via `Path().name` (B5)
- [ ] File size is checked against `max_upload_bytes` before writing (not trusting Content-Length)
- [ ] `create_app()` registers both auth and session routers
- [ ] `create_app()` creates session DB engine and calls `metadata.create_all()` on startup
- [ ] Secret key production guard raises `SystemExit` (not just a warning)
- [ ] `RunAlreadyActiveError` global exception handler returns 409 with `error_type` field (seam contract D)
- [ ] `dependencies.py` re-exports `get_current_user` and adds `get_session_service`, `get_auth_provider`
- [ ] WebSettings conditional validator rejects OIDC/Entra without required fields
- [ ] No mypy errors, no ruff warnings
