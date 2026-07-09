"""Tests for WebSocket /ws/runs/{run_id} endpoint.

Verifies authentication close codes (4001), IDOR protection close codes
(4004), and the pre-accept vs post-accept distinction: auth failures close
BEFORE accept (client never sees a successful connection), while IDOR
failures close AFTER accept (connection established, then terminated).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.routing import APIWebSocketRoute
from starlette.websockets import WebSocketDisconnect

from elspeth.web.auth.models import UserIdentity
from elspeth.web.execution.schemas import (
    ProgressData,
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
    RunEvent,
    RunStatusResponse,
)
from elspeth.web.execution.websocket_ticket import WebSocketTicketStore
from elspeth.web.sessions.protocol import RunEventRecord

# ── Helpers ───────────────────────────────────────────────────────────

_TEST_USER_ID = "ws-test-user"


class FakeWebSocket:
    """Minimal WebSocket double for direct route-handler tests."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.accepted = False
        self.sent_json: list[dict[str, Any]] = []
        self.close_code: int | None = None
        self.close_reason: str | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.close_code = code
        self.close_reason = reason

    async def send_json(self, data: dict[str, Any]) -> None:
        self.sent_json.append(data)


@dataclass(slots=True)
class FakeRunRecord:
    id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    state_id: UUID = field(default_factory=uuid4)
    status: str = "running"
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None
    rows_processed: int = 0
    rows_succeeded: int = 0
    rows_failed: int = 0
    rows_routed_success: int = 0
    rows_routed_failure: int = 0
    rows_quarantined: int = 0
    error: str | None = None
    landscape_run_id: str | None = None
    pipeline_yaml: str | None = None


@dataclass(slots=True)
class FakeSettings:
    auth_provider: str = "test"
    data_dir: str = "/tmp/elspeth-test-data"


class FakeAuthProvider:
    def __init__(self, user: UserIdentity | None = None) -> None:
        self.user = user or UserIdentity(user_id=_TEST_USER_ID, username="testuser")
        self.authenticate_tokens: list[str] = []

    async def authenticate(self, token: str) -> UserIdentity:
        self.authenticate_tokens.append(token)
        return self.user

    def assert_not_authenticated(self) -> None:
        assert self.authenticate_tokens == []


class FakeExecutionService:
    def __init__(
        self,
        *,
        ownership: bool | Exception = True,
        statuses: list[RunStatusResponse] | None = None,
    ) -> None:
        self.ownership = ownership
        self.statuses = list(statuses or [])
        self.verify_run_ownership_calls: list[tuple[UserIdentity, str]] = []
        self.get_status_calls: list[tuple[UUID, RunAccounting | None, FakeRunRecord]] = []

    async def verify_run_ownership(self, user: UserIdentity, run_id: str) -> bool:
        self.verify_run_ownership_calls.append((user, run_id))
        if isinstance(self.ownership, Exception):
            raise self.ownership
        return self.ownership

    async def get_status(
        self,
        run_id: UUID,
        *,
        accounting: RunAccounting | None = None,
        run_record: FakeRunRecord,
    ) -> RunStatusResponse:
        self.get_status_calls.append((run_id, accounting, run_record))
        if not self.statuses:
            raise AssertionError(f"No fake run status configured for {run_id}")
        return self.statuses.pop(0)


class FakeSessionService:
    def __init__(
        self,
        *,
        run_record: FakeRunRecord | None = None,
        run_events: list[RunEventRecord] | None = None,
    ) -> None:
        self.run_record = run_record or FakeRunRecord()
        self.run_events = list(run_events or [])
        self.get_run_calls: list[UUID] = []
        self.list_run_events_calls: list[UUID] = []

    async def get_run(self, run_id: UUID) -> FakeRunRecord:
        self.get_run_calls.append(run_id)
        return self.run_record

    async def list_run_events(self, run_id: UUID) -> list[RunEventRecord]:
        self.list_run_events_calls.append(run_id)
        return list(self.run_events)


class FakeSubscriberQueue:
    def get(self) -> object:
        return self


class FakeBroadcaster:
    def __init__(self) -> None:
        self.queue = FakeSubscriberQueue()
        self.subscribe_calls: list[str] = []
        self.unsubscribe_calls: list[tuple[str, FakeSubscriberQueue]] = []

    def subscribe(self, run_id: str) -> FakeSubscriberQueue:
        self.subscribe_calls.append(run_id)
        return self.queue

    def unsubscribe(self, run_id: str, queue: FakeSubscriberQueue) -> None:
        self.unsubscribe_calls.append((run_id, queue))


def _websocket_endpoint(app: FastAPI) -> Callable[..., Awaitable[None]]:
    for route in app.routes:
        if isinstance(route, APIWebSocketRoute) and route.path == "/ws/runs/{run_id}":
            return cast(Callable[..., Awaitable[None]], route.endpoint)
    raise AssertionError("WebSocket route not found")


async def _call_websocket(
    app: FastAPI,
    run_id: str,
    *,
    ticket: str | None = None,
    token: str | None = None,
) -> FakeWebSocket:
    websocket = FakeWebSocket(app)
    await _websocket_endpoint(app)(websocket, run_id, ticket=ticket, token=token)
    return websocket


def _issue_ws_ticket(
    app: FastAPI,
    run_id: str,
    *,
    user: UserIdentity | None = None,
) -> str:
    identity = user or UserIdentity(user_id=_TEST_USER_ID, username="testuser")
    ticket = app.state.websocket_ticket_store.issue(run_id=run_id, user=identity)
    return ticket.ticket


def _create_ws_test_app(
    auth_provider: FakeAuthProvider | None = None,
    execution_service: FakeExecutionService | None = None,
    broadcaster: FakeBroadcaster | None = None,
) -> FastAPI:
    """Create a minimal app with only the WebSocket route wired.

    Unlike the REST test app, WebSocket auth uses query-parameter tokens
    and calls auth_provider.authenticate() directly (no get_current_user
    dependency override).
    """
    from elspeth.web.execution.routes import create_execution_router

    app = FastAPI()
    app.state.auth_provider = auth_provider or FakeAuthProvider()
    app.state.execution_service = execution_service or FakeExecutionService()
    app.state.broadcaster = broadcaster or FakeBroadcaster()
    app.state.session_service = FakeSessionService()
    app.state.settings = FakeSettings()
    app.state.websocket_ticket_store = WebSocketTicketStore()

    app.include_router(create_execution_router())
    return app


def _accounting(
    *,
    source_rows: int = 1,
    succeeded: int = 1,
    failed: int = 0,
    structural: int = 0,
    pending: int = 0,
    closure: Literal["closed", "open", "unknown"] = "closed",
) -> RunAccounting:
    terminal = succeeded + failed + structural
    return RunAccounting(
        source=RunAccountingSource(rows_processed=source_rows),
        tokens=RunAccountingTokens(
            emitted=terminal + pending,
            terminal=terminal,
            succeeded=succeeded,
            failed=failed,
            structural=structural,
            pending=pending,
        ),
        routing=RunAccountingRouting(routed_success=0, routed_failure=0, quarantined=0, discarded=0),
        integrity=RunAccountingIntegrity(
            closure=closure,
            missing_terminal_outcomes=0,
            duplicate_terminal_outcomes=0,
        ),
    )


def _make_broadcaster() -> FakeBroadcaster:
    """Create a broadcaster fake with subscribe/unsubscribe call records."""
    return FakeBroadcaster()


# ── Auth Close Code Tests (4001 — before accept) ─────────────────────


class TestWebSocketAuth:
    """Close code 4001 on authentication failure."""

    @pytest.mark.asyncio
    async def test_missing_ticket_closes_4001(self) -> None:
        """No ?ticket= query parameter → 4001 before accept."""
        app = _create_ws_test_app()
        websocket = await _call_websocket(app, "some-run-id")
        assert websocket.accepted is False
        assert websocket.close_code == 4001
        assert "Missing" in (websocket.close_reason or "")

    @pytest.mark.asyncio
    async def test_invalid_ticket_closes_4001(self) -> None:
        """Unknown opaque ticket → 4001 without attempting JWT auth."""
        auth = FakeAuthProvider()
        app = _create_ws_test_app(auth_provider=auth)
        websocket = await _call_websocket(app, "some-run-id", ticket="not-a-real-ticket")
        assert websocket.accepted is False
        assert websocket.close_code == 4001
        assert "ticket" in (websocket.close_reason or "").lower()
        auth.assert_not_authenticated()

    @pytest.mark.asyncio
    async def test_legacy_jwt_query_param_is_rejected_before_authentication(self) -> None:
        """Session JWTs must not be accepted in WebSocket query parameters."""
        auth = FakeAuthProvider(UserIdentity(user_id=_TEST_USER_ID, username="testuser"))
        app = _create_ws_test_app(auth_provider=auth)

        websocket = await _call_websocket(app, "some-run-id", token="full-session-jwt")

        assert websocket.accepted is False
        assert websocket.close_code == 4001
        assert "ticket" in (websocket.close_reason or "").lower()
        auth.assert_not_authenticated()

    def test_ticket_store_expires_and_consumes_tickets_once(self) -> None:
        """Opaque tickets are bound, short-lived, and single-use."""
        now = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
        store = WebSocketTicketStore(ttl_seconds=30)
        user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")
        issued = store.issue(run_id="run-1", user=user, now=now)

        assert store.consume(ticket=issued.ticket, run_id="wrong-run", now=now) is None
        assert store.consume(ticket=issued.ticket, run_id="run-1", now=now) is None

        issued = store.issue(run_id="run-1", user=user, now=now)
        assert store.consume(ticket=issued.ticket, run_id="run-1", now=now + timedelta(seconds=31)) is None

        issued = store.issue(run_id="run-1", user=user, now=now)
        assert store.consume(ticket=issued.ticket, run_id="run-1", now=now) == user
        assert store.consume(ticket=issued.ticket, run_id="run-1", now=now) is None


# ── IDOR Close Code Tests (4004 — after accept) ─────────────────────


class TestWebSocketIDOR:
    """Close code 4004 on ownership verification failure."""

    @pytest.mark.asyncio
    async def test_wrong_user_closes_4004(self) -> None:
        """Authenticated user does not own the run's session → 4004."""
        user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")
        auth = FakeAuthProvider(user)
        svc = FakeExecutionService(ownership=False)

        broadcaster = _make_broadcaster()
        app = _create_ws_test_app(
            auth_provider=auth,
            execution_service=svc,
            broadcaster=broadcaster,
        )
        ticket = _issue_ws_ticket(app, "some-run-id", user=user)
        websocket = await _call_websocket(app, "some-run-id", ticket=ticket)
        assert websocket.accepted is True
        assert websocket.close_code == 4004
        assert "not found" in (websocket.close_reason or "").lower()

    @pytest.mark.asyncio
    async def test_nonexistent_run_closes_4004(self) -> None:
        """Run ID not found → verify_run_ownership raises ValueError → 4004."""
        user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")
        auth = FakeAuthProvider(user)
        svc = FakeExecutionService(ownership=ValueError("Run not found"))

        broadcaster = _make_broadcaster()
        app = _create_ws_test_app(
            auth_provider=auth,
            execution_service=svc,
            broadcaster=broadcaster,
        )
        ticket = _issue_ws_ticket(app, "nonexistent", user=user)
        websocket = await _call_websocket(app, "nonexistent", ticket=ticket)
        assert websocket.accepted is True
        assert websocket.close_code == 4004
        assert "not found" in (websocket.close_reason or "").lower()

    @pytest.mark.asyncio
    async def test_dangling_session_fk_closes_1011_not_4004(self) -> None:
        """Existing run with a missing parent session → Tier-1 sessions-DB
        corruption → 1011 internal-error close, NOT a benign 4004.

        Regression: ``RunSessionIntegrityError`` (a plain Exception, NOT a
        ValueError subclass) must not be laundered through the Tier-3
        not-found path into a benign 4004. The handler logs to the operator
        channel and closes 1011, mirroring the seed-snapshot integrity path.
        """
        from elspeth.web.execution.errors import RunSessionIntegrityError

        user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")
        auth = FakeAuthProvider(user)
        svc = FakeExecutionService(ownership=RunSessionIntegrityError(run_id="run-1", session_id="missing-session"))

        broadcaster = _make_broadcaster()
        app = _create_ws_test_app(
            auth_provider=auth,
            execution_service=svc,
            broadcaster=broadcaster,
        )
        ticket = _issue_ws_ticket(app, "run-1", user=user)
        websocket = await _call_websocket(app, "run-1", ticket=ticket)
        assert websocket.accepted is True
        assert websocket.close_code == 1011
        assert "not found" not in (websocket.close_reason or "").lower()


class TestWebSocketTimeoutRecovery:
    """Timeout path must probe authoritative status, not send ad-hoc payloads."""

    @staticmethod
    def _make_authed_app(execution_service: FakeExecutionService) -> FastAPI:
        auth = FakeAuthProvider(UserIdentity(user_id=_TEST_USER_ID, username="testuser"))
        broadcaster = _make_broadcaster()
        return _create_ws_test_app(
            auth_provider=auth,
            execution_service=execution_service,
            broadcaster=broadcaster,
        )

    @pytest.mark.asyncio
    async def test_timeout_with_still_running_status_does_not_emit_heartbeat_payload(self) -> None:
        """After an idle timeout, the next payload must still be a real RunEvent."""
        run_id = uuid4()
        svc = FakeExecutionService(
            statuses=[
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=datetime.now(tz=UTC),
                    finished_at=None,
                    error=None,
                    landscape_run_id=None,
                ),
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=datetime.now(tz=UTC),
                    finished_at=None,
                    error=None,
                    landscape_run_id=None,
                ),
            ],
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
                source_rows_processed=2,
                tokens_succeeded=0,
                tokens_failed=0,
                tokens_quarantined=0,
                tokens_routed_success=3,
                tokens_routed_failure=1,
            ),
        )

        with patch(
            "elspeth.web.execution.routes.asyncio.wait_for",
            new=AsyncMock(
                spec=asyncio.wait_for,
                side_effect=[TimeoutError(), queued_event, WebSocketDisconnect(code=1000)],
            ),
        ):
            websocket = await _call_websocket(app, str(run_id), ticket=_issue_ws_ticket(app, str(run_id)))
        payload = websocket.sent_json[0]

        assert payload["event_type"] == "progress"
        assert payload["data"]["source_rows_processed"] == 2
        assert payload["data"]["tokens_routed_success"] == 3
        assert payload["data"]["tokens_routed_failure"] == 1
        assert "type" not in payload

    @pytest.mark.asyncio
    async def test_replays_persisted_run_events_before_waiting_for_live_queue(self) -> None:
        run_id = uuid4()
        svc = FakeExecutionService(
            statuses=[
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=datetime.now(tz=UTC),
                    finished_at=None,
                    error=None,
                    landscape_run_id=None,
                )
            ],
        )
        app = self._make_authed_app(svc)
        app.state.session_service.run_events = [
            RunEventRecord(
                id=uuid4(),
                run_id=run_id,
                sequence=1,
                timestamp=datetime.now(tz=UTC),
                event_type="error",
                data={"message": "row failed", "node_id": None, "row_id": None},
            )
        ]

        with patch(
            "elspeth.web.execution.routes.asyncio.wait_for",
            new=AsyncMock(spec=asyncio.wait_for, side_effect=WebSocketDisconnect(code=1000)),
        ):
            websocket = await _call_websocket(app, str(run_id), ticket=_issue_ws_ticket(app, str(run_id)))

        assert websocket.sent_json == [
            {
                "run_id": str(run_id),
                "timestamp": websocket.sent_json[0]["timestamp"],
                "event_type": "error",
                "data": {"message": "row failed", "node_id": None, "row_id": None},
            }
        ]
        assert app.state.session_service.list_run_events_calls == [run_id]

    @pytest.mark.asyncio
    async def test_replay_sends_later_persisted_events_before_closing_on_terminal(self) -> None:
        run_id = uuid4()
        timestamp = datetime.now(tz=UTC)
        svc = FakeExecutionService(
            statuses=[
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=timestamp,
                    finished_at=None,
                    error=None,
                    landscape_run_id=None,
                )
            ],
        )
        app = self._make_authed_app(svc)
        app.state.session_service.run_events = [
            RunEventRecord(
                id=uuid4(),
                run_id=run_id,
                sequence=1,
                timestamp=timestamp,
                event_type="progress",
                data={
                    "source_rows_processed": 1,
                    "tokens_succeeded": 0,
                    "tokens_failed": 0,
                    "tokens_quarantined": 0,
                    "tokens_routed_success": 0,
                    "tokens_routed_failure": 0,
                },
            ),
            RunEventRecord(
                id=uuid4(),
                run_id=run_id,
                sequence=2,
                timestamp=timestamp,
                event_type="failed",
                data={"status": "failed", "detail": "pipeline crashed", "node_id": None},
            ),
            RunEventRecord(
                id=uuid4(),
                run_id=run_id,
                sequence=3,
                timestamp=timestamp,
                event_type="error",
                data={"message": "late row error", "node_id": None, "row_id": None},
            ),
        ]

        websocket = await _call_websocket(app, str(run_id), ticket=_issue_ws_ticket(app, str(run_id)))

        assert [payload["event_type"] for payload in websocket.sent_json] == ["progress", "failed", "error"]
        assert websocket.close_code == 1000

    @pytest.mark.asyncio
    async def test_reconnect_skips_live_event_already_delivered_by_replay(self) -> None:
        run_id = uuid4()
        timestamp = datetime.now(tz=UTC)
        svc = FakeExecutionService(
            statuses=[
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=timestamp,
                    finished_at=None,
                    error=None,
                    landscape_run_id=None,
                )
            ],
        )
        app = self._make_authed_app(svc)
        replay_record = RunEventRecord(
            id=uuid4(),
            run_id=run_id,
            sequence=1,
            timestamp=timestamp,
            event_type="error",
            data={"message": "row failed", "node_id": None, "row_id": None},
        )
        app.state.session_service.run_events = [replay_record]
        duplicate_live = RunEvent(
            run_id=str(run_id),
            timestamp=timestamp,
            event_type="error",
            data={"message": "row failed", "node_id": None, "row_id": None},
        ).with_event_sequence(1)
        later_live = RunEvent(
            run_id=str(run_id),
            timestamp=timestamp,
            event_type="progress",
            data=ProgressData(
                source_rows_processed=2,
                tokens_succeeded=0,
                tokens_failed=0,
                tokens_quarantined=0,
                tokens_routed_success=0,
                tokens_routed_failure=0,
            ),
        ).with_event_sequence(2)

        with patch(
            "elspeth.web.execution.routes.asyncio.wait_for",
            new=AsyncMock(
                spec=asyncio.wait_for,
                side_effect=[duplicate_live, later_live, WebSocketDisconnect(code=1000)],
            ),
        ):
            websocket = await _call_websocket(app, str(run_id), ticket=_issue_ws_ticket(app, str(run_id)))

        assert [payload["event_type"] for payload in websocket.sent_json] == ["error", "progress"]
        assert websocket.sent_json[0]["data"]["message"] == "row failed"
        assert websocket.sent_json[1]["data"]["source_rows_processed"] == 2

    @pytest.mark.asyncio
    async def test_timeout_synthesizes_terminal_event_when_status_turned_completed(self) -> None:
        """Missed terminal broadcasts must be recovered from authoritative status."""
        run_id = uuid4()
        svc = FakeExecutionService(
            statuses=[
                RunStatusResponse(
                    run_id=str(run_id),
                    status="running",
                    started_at=datetime.now(tz=UTC),
                    finished_at=None,
                    error=None,
                    landscape_run_id=None,
                ),
                RunStatusResponse(
                    run_id=str(run_id),
                    status="completed",
                    started_at=datetime.now(tz=UTC),
                    finished_at=datetime.now(tz=UTC),
                    accounting=_accounting(source_rows=1, succeeded=1),
                    error=None,
                    landscape_run_id="land-1",
                ),
            ],
        )
        app = self._make_authed_app(svc)
        with patch(
            "elspeth.web.execution.routes.asyncio.wait_for",
            new=AsyncMock(spec=asyncio.wait_for, side_effect=[TimeoutError()]),
        ):
            websocket = await _call_websocket(app, str(run_id), ticket=_issue_ws_ticket(app, str(run_id)))
        payload = websocket.sent_json[0]
        assert payload["event_type"] == "completed"
        assert payload["data"]["landscape_run_id"] == "land-1"
        assert websocket.close_code == 1000
