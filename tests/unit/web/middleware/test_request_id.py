"""Tests for the request-id middleware.

The middleware assigns a correlation id to every inbound HTTP request so
that exception handlers, structured log events, and the client-facing
response share one id.  Operators can pair an error report to its slog
event with a single lookup, closing the support-ticket triage round trip.

Security boundary: the middleware MUST reject or regenerate a
caller-supplied ``X-Request-ID`` that looks suspicious (too long, wrong
shape) — otherwise an attacker can spoof the log-correlation field to
frame requests they did not make.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI, Request

from elspeth.web.middleware.request_id import (
    MAX_REQUEST_ID_LENGTH,
    REQUEST_ID_HEADER,
    RequestIdMiddleware,
)
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/_echo")
    async def echo(request: Request) -> dict[str, str]:
        rid = request.state.request_id
        return {"request_id": rid}

    return app


def _make_boom_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/_boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    return app


class TestRequestIdAssignment:
    """Every request must carry a request_id on request.state and in the response header."""

    def test_missing_header_generates_uuid(self) -> None:
        """No inbound X-Request-ID → middleware generates a UUID4."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/_echo")

        assert resp.status_code == 200
        rid = resp.json()["request_id"]
        # UUID4 validation: parsing must succeed and the version must be 4.
        parsed = uuid.UUID(rid)
        assert parsed.version == 4

        # Same id echoed on the response header for client correlation.
        assert resp.headers[REQUEST_ID_HEADER] == rid

    def test_inbound_header_echoed(self) -> None:
        """A safe, short inbound X-Request-ID must pass through unchanged."""
        app = _make_app()
        client = TestClient(app)
        supplied = "req-abc-123"
        resp = client.get("/_echo", headers={REQUEST_ID_HEADER: supplied})

        assert resp.json()["request_id"] == supplied
        assert resp.headers[REQUEST_ID_HEADER] == supplied

    def test_each_request_gets_unique_id_when_header_missing(self) -> None:
        """Without a supplied header, ids must not repeat across requests."""
        app = _make_app()
        client = TestClient(app)
        ids = {client.get("/_echo").json()["request_id"] for _ in range(10)}
        assert len(ids) == 10

    def test_unhandled_500_response_still_echoes_request_id(self) -> None:
        """Generic 500 responses still need the correlation header."""
        app = _make_boom_app()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/_boom")

        assert resp.status_code == 500
        rid = resp.headers[REQUEST_ID_HEADER]
        assert uuid.UUID(rid).version == 4


class TestRequestIdHardening:
    """Security: middleware must not let a caller poison the log correlation."""

    def test_overly_long_header_is_replaced(self) -> None:
        """An oversized X-Request-ID is discarded — middleware generates a fresh UUID.

        Attacker scenario: a multi-megabyte request_id could bloat every
        slog event it taints, or be used to smuggle data through a log
        pipeline.  The middleware caps length and regenerates on violation.
        """
        app = _make_app()
        client = TestClient(app)
        oversized = "A" * (MAX_REQUEST_ID_LENGTH + 1)
        resp = client.get("/_echo", headers={REQUEST_ID_HEADER: oversized})

        rid = resp.json()["request_id"]
        assert rid != oversized
        # Regenerated value is a UUID4.
        assert uuid.UUID(rid).version == 4

    def test_control_characters_rejected(self) -> None:
        """Newlines / CRs would enable log-injection — must be regenerated."""
        app = _make_app()
        client = TestClient(app)
        # httpx rejects raw \n in header values at send time, so exercise
        # the filter via characters that reach the middleware but must
        # still be stripped: tab + vertical bar smuggle attempt.
        suspicious = "rid\tINJECT|payload"
        resp = client.get("/_echo", headers={REQUEST_ID_HEADER: suspicious})

        rid = resp.json()["request_id"]
        assert rid != suspicious
        assert uuid.UUID(rid).version == 4

    def test_empty_header_treated_as_absent(self) -> None:
        """An empty X-Request-ID is indistinguishable from absent — regenerate."""
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/_echo", headers={REQUEST_ID_HEADER: ""})

        rid = resp.json()["request_id"]
        assert rid != ""
        assert uuid.UUID(rid).version == 4


class TestMaxLengthConstant:
    """The length cap is deliberate — fix at a small, documented value."""

    def test_max_length_is_reasonable(self) -> None:
        # A UUID4 with four hyphens is 36 chars; "req-<UUID>" prefixes
        # are under 64.  The cap must leave room for both conventions
        # while blocking attacker payloads measured in kilobytes.
        assert 32 <= MAX_REQUEST_ID_LENGTH <= 128


@pytest.fixture(scope="module")
def hypothesis_available() -> bool:
    try:
        import hypothesis  # noqa: F401

        return True
    except ImportError:
        return False
