"""Tests for WebSocket /ws/runs/{run_id} endpoint.

Verifies authentication close codes (4001), IDOR protection close codes
(4004), and the pre-accept vs post-accept distinction: auth failures close
BEFORE accept (client never sees a successful connection), while IDOR
failures close AFTER accept (connection established, then terminated).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from elspeth.web.auth.models import AuthenticationError
from elspeth.web.execution.schemas import CompletedData, ProgressData, RunEvent, RunStatusResponse

# ── Helpers ───────────────────────────────────────────────────────────

_TEST_USER_ID = "ws-test-user"


def _create_ws_test_app(
    auth_provider: MagicMock | None = None,
    execution_service: MagicMock | None = None,
    broadcaster: MagicMock | None = None,
) -> FastAPI:
    """Create a minimal app with only the WebSocket route wired.

    Unlike the REST test app, WebSocket auth uses query-parameter tokens
    and calls auth_provider.authenticate() directly (no get_current_user
    dependency override).
    """
    from elspeth.web.execution.routes import create_execution_router

    app = FastAPI()
    app.state.auth_provider = auth_provider or MagicMock()
    app.state.execution_service = execution_service or MagicMock()
    app.state.broadcaster = broadcaster or MagicMock()

    app.include_router(create_execution_router())
    return app


def _make_broadcaster() -> MagicMock:
    """Create a mock broadcaster with subscribe/unsubscribe."""
    broadcaster = MagicMock()
    broadcaster.subscribe.return_value = MagicMock()
    return broadcaster


# ── Auth Close Code Tests (4001 — before accept) ─────────────────────


class TestWebSocketAuth:
    """Close code 4001 on authentication failure."""

    def test_missing_token_closes_4001(self) -> None:
        """No ?token= query parameter → 4001 before accept."""
        app = _create_ws_test_app()
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect("/ws/runs/some-run-id"):
            pass  # Should not reach here
        assert exc_info.value.code == 4001
        assert "Missing" in (exc_info.value.reason or "")

    def test_invalid_token_closes_4001(self) -> None:
        """Invalid JWT → auth_provider.authenticate() raises → 4001."""
        auth = MagicMock()
        auth.authenticate = AsyncMock(side_effect=AuthenticationError("bad token"))
        app = _create_ws_test_app(auth_provider=auth)
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect("/ws/runs/some-run-id?token=bad-jwt"):
            pass
        assert exc_info.value.code == 4001
        assert "Invalid" in (exc_info.value.reason or "")


# ── IDOR Close Code Tests (4004 — after accept) ─────────────────────


class TestWebSocketIDOR:
    """Close code 4004 on ownership verification failure."""

    def test_wrong_user_closes_4004(self) -> None:
        """Authenticated user does not own the run's session → 4004."""
        from elspeth.web.auth.models import UserIdentity

        auth = MagicMock()
        user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")
        auth.authenticate = AsyncMock(return_value=user)

        svc = MagicMock()
        svc.verify_run_ownership = AsyncMock(return_value=False)

        broadcaster = _make_broadcaster()
        app = _create_ws_test_app(
            auth_provider=auth,
            execution_service=svc,
            broadcaster=broadcaster,
        )
        client = TestClient(app)
        with client.websocket_connect("/ws/runs/some-run-id?token=valid") as ws, pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_json()
        assert exc_info.value.code == 4004
        assert "not found" in (exc_info.value.reason or "").lower()

    def test_nonexistent_run_closes_4004(self) -> None:
        """Run ID not found → verify_run_ownership raises ValueError → 4004."""
        from elspeth.web.auth.models import UserIdentity

        auth = MagicMock()
        user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")
        auth.authenticate = AsyncMock(return_value=user)

        svc = MagicMock()
        svc.verify_run_ownership = AsyncMock(side_effect=ValueError("Run not found"))

        broadcaster = _make_broadcaster()
        app = _create_ws_test_app(
            auth_provider=auth,
            execution_service=svc,
            broadcaster=broadcaster,
        )
        client = TestClient(app)
        with client.websocket_connect("/ws/runs/nonexistent?token=valid") as ws, pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_json()
        assert exc_info.value.code == 4004
        assert "not found" in (exc_info.value.reason or "").lower()


class TestWebSocketTimeoutRecovery:
    """Timeout path must probe authoritative status, not send ad-hoc payloads."""

    @staticmethod
    def _make_authed_app(execution_service: MagicMock) -> FastAPI:
        from elspeth.web.auth.models import UserIdentity

        auth = MagicMock()
        auth.authenticate = AsyncMock(return_value=UserIdentity(user_id=_TEST_USER_ID, username="testuser"))
        broadcaster = _make_broadcaster()
        return _create_ws_test_app(
            auth_provider=auth,
            execution_service=execution_service,
            broadcaster=broadcaster,
        )

    def test_timeout_with_still_running_status_does_not_emit_heartbeat_payload(self) -> None:
        """After an idle timeout, the next payload must still be a real RunEvent."""
        run_id = uuid4()
        svc = MagicMock()
        svc.verify_run_ownership = AsyncMock(return_value=True)
        svc.get_status = AsyncMock(
            side_effect=[
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=datetime.now(tz=UTC),
                    finished_at=None,
                    rows_processed=1,
                    rows_succeeded=1,
                    rows_failed=0,
                    rows_routed_success=0,
                    rows_routed_failure=0,
                    rows_quarantined=0,
                    error=None,
                    landscape_run_id=None,
                ),
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=datetime.now(tz=UTC),
                    finished_at=None,
                    rows_processed=1,
                    rows_succeeded=1,
                    rows_failed=0,
                    rows_routed_success=0,
                    rows_routed_failure=0,
                    rows_quarantined=0,
                    error=None,
                    landscape_run_id=None,
                ),
            ]
        )
        app = self._make_authed_app(svc)
        # elspeth-5069612f3c — exercise the routed split on the streaming
        # progress path so a regression that drops the new wire fields back
        # to the pre-fix two-field shape would fail this assertion. Without
        # this, only terminal events would carry the split and an in-flight
        # silent-zero regression would slip through.
        queued_event = RunEvent(
            run_id=str(run_id),
            timestamp=datetime.now(tz=UTC),
            event_type="progress",
            data=ProgressData(
                rows_processed=2,
                rows_failed=0,
                rows_routed_success=3,
                rows_routed_failure=1,
            ),
        )

        with (
            patch(
                "elspeth.web.execution.routes.asyncio.wait_for",
                new=AsyncMock(side_effect=[TimeoutError(), queued_event, WebSocketDisconnect(code=1000)]),
            ),
            TestClient(app) as client,
            client.websocket_connect(f"/ws/runs/{run_id}?token=valid") as ws,
        ):
            payload = ws.receive_json()

        assert payload["event_type"] == "progress"
        assert payload["data"]["rows_processed"] == 2
        assert payload["data"]["rows_routed_success"] == 3
        assert payload["data"]["rows_routed_failure"] == 1
        assert "type" not in payload

    def test_timeout_synthesizes_terminal_event_when_status_turned_completed(self) -> None:
        """Missed terminal broadcasts must be recovered from authoritative status."""
        run_id = uuid4()
        svc = MagicMock()
        svc.verify_run_ownership = AsyncMock(return_value=True)
        svc.get_status = AsyncMock(
            side_effect=[
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=datetime.now(tz=UTC),
                    finished_at=None,
                    rows_processed=1,
                    rows_succeeded=1,
                    rows_failed=0,
                    rows_routed_success=0,
                    rows_routed_failure=0,
                    rows_quarantined=0,
                    error=None,
                    landscape_run_id=None,
                ),
                RunStatusResponse(
                    run_id=str(run_id),
                    status="completed",
                    started_at=datetime.now(tz=UTC),
                    finished_at=datetime.now(tz=UTC),
                    rows_processed=1,
                    rows_succeeded=1,
                    rows_failed=0,
                    rows_routed_success=0,
                    rows_routed_failure=0,
                    rows_quarantined=0,
                    error=None,
                    landscape_run_id="land-1",
                ),
            ]
        )
        app = self._make_authed_app(svc)
        queued_event = RunEvent(
            run_id=str(run_id),
            timestamp=datetime.now(tz=UTC),
            event_type="completed",
            data=CompletedData(
                status="completed",
                rows_processed=1,
                rows_succeeded=1,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                landscape_run_id="land-1",
            ),
        )

        with (
            patch(
                "elspeth.web.execution.routes.asyncio.wait_for",
                new=AsyncMock(side_effect=[TimeoutError(), queued_event]),
            ),
            TestClient(app) as client,
            client.websocket_connect(f"/ws/runs/{run_id}?token=valid") as ws,
        ):
            payload = ws.receive_json()
            assert payload["event_type"] == "completed"
            assert payload["data"]["landscape_run_id"] == "land-1"
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_json()

        assert exc_info.value.code == 1000
