"""Tests for tutorial telemetry helpers and routes."""

from __future__ import annotations

from fastapi import FastAPI

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.composer import tutorial_telemetry as tutorial_telemetry_module
from elspeth.web.composer.tutorial_abandon_routes import create_tutorial_abandon_router
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


class _RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object]]] = []

    def add(self, amount: int, *, attributes: dict[str, object]) -> None:
        self.calls.append((amount, dict(attributes)))


def test_record_tutorial_completed_rejects_unknown_path() -> None:
    try:
        tutorial_telemetry_module.record_tutorial_completed_path("made_up")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "completion_path" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_abandon_route_increments_counter(monkeypatch) -> None:
    counter = _RecordingCounter()
    monkeypatch.setattr(tutorial_telemetry_module, "_TUTORIAL_ABANDON_COUNTER", counter)
    app = FastAPI()
    identity = UserIdentity(user_id="alice", username="alice")

    async def _mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = _mock_user
    app.include_router(create_tutorial_abandon_router())
    client = TestClient(app)

    response = client.post("/api/tutorial/abandon")

    assert response.status_code == 204
    assert counter.calls == [(1, {})]
