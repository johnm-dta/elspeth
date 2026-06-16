"""Tests for the FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import sys
import weakref
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretBytes, ValidationError
from sqlalchemy.exc import CompileError, OperationalError
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse
from structlog.testing import capture_logs

import elspeth.web.app as app_module
from elspeth.contracts import RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.web.app import (
    _JSON_COLLECTION_FIELDS,
    _BodySizeLimitMiddleware,
    _periodic_orphan_cleanup,
    _settings_from_env,
    create_app,
    lifespan,
)
from elspeth.web.auth.audit import AuthAuditRecorder
from elspeth.web.composer.boot_probe import ComposerBootConfigError
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import get_settings
from elspeth.web.sessions.protocol import CompositionStateData, RunRecord
from elspeth.web.sessions.telemetry import _FakeCounter, build_sessions_telemetry, observed_value
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _settings(tmp_path: Path, **overrides) -> WebSettings:
    """Create WebSettings with data_dir pointed at a temp directory."""
    defaults = {
        "data_dir": tmp_path,
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": SecretBytes(b"\x00" * 32),
    }
    defaults.update(overrides)
    return WebSettings(**defaults)  # type: ignore[arg-type]


class _StaticJsonResponse:
    """Minimal httpx-like response stub for lifespan discovery tests."""

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _StaticAsyncClient:
    """Minimal async context manager stub for patched httpx.AsyncClient."""

    def __init__(self, responses: list[_StaticJsonResponse]) -> None:
        self._responses = iter(responses)

    async def __aenter__(self) -> _StaticAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def get(self, url: str) -> _StaticJsonResponse:
        del url
        return next(self._responses)

    async def request(self, method: str, url: str) -> _StaticJsonResponse:
        request = httpx.Request(method, url)
        raise httpx.ConnectError("catalog probe disabled in test", request=request)


class _EnteringAsyncClientRaises:
    """Async client stub whose context enter fails before a request exists."""

    async def __aenter__(self) -> _EnteringAsyncClientRaises:
        raise OSError("catalog client context failed")

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class TestCreateApp:
    """Tests for create_app()."""

    def test_returns_fastapi_instance_with_correct_title(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        assert app.title == "ELSPETH Web"

    def test_returns_fastapi_instance_with_correct_version(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        assert app.version == "0.1.0"

    def test_default_settings_when_none_passed(self, tmp_path) -> None:
        # create_app(None) uses WebSettings() which defaults data_dir to "data".
        # That won't exist in tests, so we pass explicit settings instead.
        app = create_app(_settings(tmp_path))
        assert app.state.settings.port == 8451

    def test_settings_stored_on_app_state(self, tmp_path) -> None:
        settings = _settings(tmp_path, port=9999)
        app = create_app(settings)
        assert app.state.settings is settings
        assert app.state.settings.port == 9999

    def test_tutorial_cache_stored_on_app_state_with_configured_dir(self, tmp_path) -> None:
        from datetime import UTC, datetime

        from elspeth.web.preferences.tutorial_cache import (
            CANONICAL_SEED_PROMPT,
            TutorialCache,
            TutorialCacheEntry,
        )

        cache_dir = tmp_path / "custom-cache"
        app = create_app(_settings(tmp_path, tutorial_cache_dir=cache_dir))

        cache = app.state.tutorial_cache
        assert isinstance(cache, TutorialCache)
        cache.store(
            TutorialCacheEntry(
                canonical_prompt=CANONICAL_SEED_PROMPT,
                model_id="gpt-5.5:gpt-5.5",
                cached_at=datetime(2026, 5, 15, tzinfo=UTC),
                rows=[],
                source_data_hash="hash",
                llm_call_count=0,
                pipeline_yaml="source: {}\n",
            )
        )
        assert len(list(cache_dir.glob("*.json"))) == 1

    def test_loopback_default_secret_key_rejected_without_pytest_module(self, tmp_path, monkeypatch) -> None:
        """Loopback bind is not proof the service is unreachable behind a proxy."""
        settings = _settings(tmp_path, host="127.0.0.1")
        monkeypatch.delitem(sys.modules, "pytest", raising=False)
        monkeypatch.delenv("ELSPETH_ENV", raising=False)

        with pytest.raises(SystemExit, match="secret_key is set to the default value"):
            create_app(settings)

    def test_abandoned_app_disposes_session_engine(self, tmp_path, monkeypatch) -> None:
        """Unit tests often instantiate ``create_app`` without running lifespan."""
        app = create_app(_settings(tmp_path))
        disposed = False
        original_dispose = app.state.session_engine.dispose

        def dispose() -> None:
            nonlocal disposed
            disposed = True
            original_dispose()

        monkeypatch.setattr(app.state.session_engine, "dispose", dispose)
        app_ref = weakref.ref(app)

        del app
        gc.collect()

        assert app_ref() is None
        assert disposed


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.json() == {"status": "ok"}


class TestMetricsEndpoint:
    """Tests for GET /metrics (Prometheus scrape endpoint)."""

    def test_metrics_returns_plaintext_on_success(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

    def test_metrics_scrape_failure_returns_503_and_records_cause(self, tmp_path) -> None:
        """A third-party collector raising inside generate_latest() must not
        leak a traceback into the response and must not drop the error
        silently. The handler returns a fixed 503 and records *why* via slog
        (sanctioned telemetry-system-failure logging per CLAUDE.md), carrying
        a bounded detail that identifies which collector broke.
        """
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        boom = RuntimeError("collector 'broken_metric' raised during collect()")
        with (
            patch("elspeth.web.app.generate_latest", side_effect=boom),
            capture_logs() as logs,
        ):
            response = client.get("/metrics")
        assert response.status_code == 503
        assert response.content == b"# scrape failed\n"
        events = [e for e in logs if e.get("event") == "prometheus_scrape_failed"]
        assert len(events) == 1, f"expected one scrape-failure log, got {logs!r}"
        assert events[0]["exc_class"] == "RuntimeError"
        assert "broken_metric" in events[0]["detail"]


class TestCORSMiddleware:
    """Tests that CORS middleware is configured."""

    def test_cors_allows_configured_origin(self, tmp_path) -> None:
        settings = _settings(tmp_path, cors_origins=["http://localhost:5173"])
        app = create_app(settings)
        client = TestClient(app)
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_rejects_unconfigured_origin(self, tmp_path) -> None:
        settings = _settings(tmp_path, cors_origins=["http://localhost:5173"])
        app = create_app(settings)
        client = TestClient(app)
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Starlette CORS middleware omits the header for disallowed origins
        assert "access-control-allow-origin" not in response.headers


class TestBodySizeLimitMiddleware:
    """Tests that the 10 MB body-size guard rejects oversized requests.

    Phase 5b.0.5 (F-3): the ASGI body-size middleware is defense-in-depth
    against unbounded payload allocation.  The Pydantic per-field caps
    (``SendMessageRequest.content``, ``_InlineBlobModel.content``) are
    the actual guarantees; the middleware short-circuits Content-Length
    declarations that would otherwise reach the parser at all.
    """

    @pytest.mark.asyncio
    async def test_rejects_oversized_content_length(self) -> None:
        # 11 MB declared > 10 MB cap. Body content itself doesn't matter
        # because the middleware checks the header and short-circuits before
        # reading the body.
        response = await self._dispatch_with_content_length(11_000_000)

        assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_allows_under_limit_content_length(self) -> None:
        # Sanity check that the middleware does not over-reject — a
        # tiny declared body must still reach call_next, not be 413'd by the
        # size guard.
        response = await self._dispatch_with_content_length(2)
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_allows_exact_content_length_boundary(self) -> None:
        response = await self._dispatch_with_content_length(_BodySizeLimitMiddleware._MAX_BODY_BYTES)
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_rejects_one_byte_over_content_length_boundary(self) -> None:
        response = await self._dispatch_with_content_length(_BodySizeLimitMiddleware._MAX_BODY_BYTES + 1)
        assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_missing_content_length_stream_passes_through_without_body_read(self) -> None:
        """No Content-Length means the middleware defers to per-route validators."""

        async def noop_app(scope, receive, send) -> None:
            return None

        middleware = _BodySizeLimitMiddleware(app=noop_app)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/health",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
            "root_path": "",
            "http_version": "1.1",
        }

        async def receive() -> dict[str, object]:
            raise AssertionError("Content-Length-only guard must not read streamed bodies")

        request = Request(scope, receive)

        async def call_next(_request: Request) -> StarletteResponse:
            return StarletteResponse(status_code=204)

        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 204

    async def _dispatch_with_content_length(self, content_length: int) -> StarletteResponse:
        async def noop_app(scope, receive, send) -> None:
            return None

        middleware = _BodySizeLimitMiddleware(app=noop_app)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/health",
            "headers": [
                (b"content-length", str(content_length).encode("ascii")),
                (b"content-type", b"application/json"),
            ],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
            "root_path": "",
            "http_version": "1.1",
        }

        async def receive() -> dict[str, object]:
            raise AssertionError("Content-Length-only guard must not read request bodies")

        request = Request(scope, receive)

        async def call_next(_request: Request) -> StarletteResponse:
            return StarletteResponse(status_code=204)

        response = await middleware.dispatch(request, call_next)
        assert isinstance(response, StarletteResponse)
        return response


class TestGetSettingsDependency:
    """Tests for get_settings() dependency provider."""

    def test_get_settings_returns_app_settings(self, tmp_path) -> None:
        settings = _settings(tmp_path, port=4242)
        app = create_app(settings)

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/_test_settings",
                "headers": [],
                "app": app,
            }
        )

        assert get_settings(request).port == 4242


class TestCatalogWiring:
    """Tests that create_app() wires the catalog service into app state."""

    def test_catalog_service_on_app_state(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        assert app.state.catalog_service is not None

    def test_catalog_sources_endpoint_reachable(self, tmp_path) -> None:
        from elspeth.web.auth.middleware import get_current_user
        from elspeth.web.auth.models import UserIdentity

        app = create_app(_settings(tmp_path))

        async def _mock_user() -> UserIdentity:
            return UserIdentity(user_id="test-user", username="test-user")

        app.dependency_overrides[get_current_user] = _mock_user
        client = TestClient(app)
        response = client.get("/api/catalog/sources")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestCatalogAuthGating:
    """Catalog endpoints must require authentication.

    Commit 46b94fda widened the catalog's disclosure surface from the
    truncated base-class schema to the full Pydantic discriminated union
    (per-provider required fields, full docstrings). The v1 design decision
    to leave /api/catalog/* public (Sub-Spec 3, 2026-03-28) was made under
    the old, thinner payload; these tests pin the revised contract.
    """

    def test_catalog_sources_requires_auth(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/api/catalog/sources")
        assert response.status_code == 401

    def test_catalog_transforms_requires_auth(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/api/catalog/transforms")
        assert response.status_code == 401

    def test_catalog_sinks_requires_auth(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/api/catalog/sinks")
        assert response.status_code == 401

    def test_catalog_schema_requires_auth(self, tmp_path) -> None:
        """The schema endpoint is the primary disclosure concern — reject
        anonymous access explicitly so a regression on the router gate
        doesn't reopen the full discriminated-union payload."""
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        response = client.get("/api/catalog/transforms/passthrough/schema")
        assert response.status_code == 401

    def test_catalog_accessible_with_auth(self, tmp_path) -> None:
        """Sanity: the gate doesn't break the authenticated path."""
        from elspeth.web.auth.middleware import get_current_user
        from elspeth.web.auth.models import UserIdentity

        app = create_app(_settings(tmp_path))

        async def _mock_user() -> UserIdentity:
            return UserIdentity(user_id="test-user", username="test-user")

        app.dependency_overrides[get_current_user] = _mock_user
        client = TestClient(app)
        response = client.get("/api/catalog/sources")
        assert response.status_code == 200


class TestAuthWiring:
    """Tests that create_app() wires auth provider into app state."""

    def test_auth_provider_on_app_state(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        assert app.state.auth_provider is not None

    def test_auth_audit_recorder_on_app_state(self, tmp_path) -> None:
        settings = _settings(tmp_path)
        app = create_app(settings)

        assert isinstance(app.state.auth_audit_recorder, AuthAuditRecorder)
        assert app.state.auth_audit_recorder.landscape_url == settings.get_landscape_url()

    def test_auth_routes_registered(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        # /api/auth/config is a public endpoint
        response = client.get("/api/auth/config")
        assert response.status_code == 200


class TestSessionWiring:
    """Tests that create_app() wires session service into app state."""

    def test_session_service_on_app_state(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        assert app.state.session_service is not None

    def test_session_routes_registered(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        client = TestClient(app)
        # Without auth, should get 401
        response = client.get("/api/sessions")
        assert response.status_code == 401


class TestMultiWorkerEnforcement:
    """W10 -> R6: Hard-enforce single worker for WebSocket support.

    The ``_SESSION_BLOB_LOCKS`` registry in composer/tools.py is a
    process-local mutex map whose correctness depends on this guard
    rejecting every multi-worker signal (see composer/tools.py's
    PROCESS-LOCAL CORRECTNESS PRECONDITION block).  If the guard is
    relaxed on any code path without also replacing the lock registry
    with a cross-process primitive, two workers can interleave
    blob-file writes and DB rollbacks against the same session.  These
    tests regression-protect each detection code path the comment names.
    """

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "4"})
    def test_raises_on_multi_worker(self, tmp_path) -> None:
        """Application factory rejects WEB_CONCURRENCY > 1."""
        with pytest.raises(RuntimeError, match=r"WEB_CONCURRENCY=4\) but is not supported"):
            create_app(_settings(tmp_path))

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "1"})
    def test_single_worker_accepted(self, tmp_path) -> None:
        """No error when running with a single worker."""
        app = create_app(_settings(tmp_path))
        assert app is not None

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "banana"})
    def test_invalid_web_concurrency_fails_closed(self, tmp_path) -> None:
        """A non-integer worker count is a deployment bug, not a silent default."""
        with pytest.raises(RuntimeError, match=r"WEB_CONCURRENCY='banana' is not a valid integer"):
            create_app(_settings(tmp_path))

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "1"})
    def test_raises_on_uvicorn_workers_space_form(self, tmp_path, monkeypatch) -> None:
        """Reject ``uvicorn --workers 2`` (space-separated argv form)."""
        monkeypatch.setattr("sys.argv", ["uvicorn", "elspeth.web.app:create_app", "--workers", "2"])
        with pytest.raises(RuntimeError, match=r"--workers 2\) but is not supported"):
            create_app(_settings(tmp_path))

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "1"})
    def test_raises_on_workers_equals_form(self, tmp_path, monkeypatch) -> None:
        """Reject ``--workers=2`` (equals form used by some launchers)."""
        monkeypatch.setattr("sys.argv", ["uvicorn", "elspeth.web.app:create_app", "--workers=2"])
        with pytest.raises(RuntimeError, match=r"--workers=2\) but is not supported"):
            create_app(_settings(tmp_path))

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "1"})
    def test_raises_on_gunicorn_short_flag(self, tmp_path, monkeypatch) -> None:
        """Reject ``gunicorn -w 2`` (gunicorn's short form)."""
        monkeypatch.setattr("sys.argv", ["gunicorn", "elspeth.web.app:create_app", "-w", "2"])
        with pytest.raises(RuntimeError, match=r"-w 2\) but is not supported"):
            create_app(_settings(tmp_path))

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "1"})
    def test_workers_one_space_form_accepted(self, tmp_path, monkeypatch) -> None:
        """``--workers 1`` is permitted — the guard only fires at > 1."""
        monkeypatch.setattr("sys.argv", ["uvicorn", "elspeth.web.app:create_app", "--workers", "1"])
        app = create_app(_settings(tmp_path))
        assert app is not None


class TestExecutionWiring:
    """Tests that create_app() wires execution routes."""

    def test_execution_routes_registered(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path))
        route_paths = [path for route in app.routes if isinstance(path := getattr(route, "path", None), str)]
        assert "/api/sessions/{session_id}/validate" in route_paths
        assert "/api/sessions/{session_id}/execute" in route_paths
        assert "/api/runs/{run_id}" in route_paths
        assert "/api/runs/{run_id}/cancel" in route_paths
        assert "/ws/runs/{run_id}" in route_paths


class TestOidcDiscoveryStartup:
    """OIDC discovery in lifespan() must validate response shape before storing it."""

    @staticmethod
    def _oidc_settings(tmp_path: Path) -> WebSettings:
        return _settings(
            tmp_path,
            auth_provider="oidc",
            composer_boot_probe_enabled=False,
            oidc_issuer="https://issuer.example.com",
            oidc_audience="test-audience",
            oidc_client_id="test-client-id",
            secret_key="dev-secret",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            [],
            {"authorization_endpoint": 123},
            {"authorization_endpoint": "   "},
            {"authorization_endpoint": "javascript:alert(1)"},
            {"authorization_endpoint": "http://issuer.example.com/oauth2/authorize"},
            {"authorization_endpoint": "https://evil.example.com/oauth2/authorize"},
        ],
    )
    async def test_lifespan_rejects_invalid_discovery_authorization_endpoint(self, tmp_path, payload) -> None:
        """Malformed discovery JSON must fail as deterministic startup error."""
        app = create_app(self._oidc_settings(tmp_path))

        with (
            patch("httpx.AsyncClient", return_value=_StaticAsyncClient([_StaticJsonResponse(payload)])),
            pytest.raises(SystemExit, match="OIDC discovery failed"),
        ):
            async with lifespan(app):
                pass

    @pytest.mark.asyncio
    async def test_lifespan_accepts_same_origin_https_discovery_authorization_endpoint(self, tmp_path) -> None:
        """Discovery authorization endpoints are stored only after issuer-origin validation."""
        app = create_app(self._oidc_settings(tmp_path))

        payload = {"authorization_endpoint": "https://issuer.example.com/oauth2/authorize"}
        with patch("httpx.AsyncClient", return_value=_StaticAsyncClient([_StaticJsonResponse(payload)])):
            async with lifespan(app):
                assert app.state.oidc_authorization_endpoint == "https://issuer.example.com/oauth2/authorize"


class TestLifespanShutdown:
    """Shutdown must await the execution-service drain path."""

    @pytest.mark.asyncio
    async def test_lifespan_aborts_on_composer_config_rejection(self, monkeypatch, tmp_path) -> None:
        app = create_app(_settings(tmp_path, composer_boot_probe_enabled=True))
        counter = MagicMock(spec=["add"])
        latency = MagicMock(spec=["record"])

        async def _reject_probe(**_kwargs: object) -> bool:
            raise ComposerBootConfigError("composer sampling rejected")

        monkeypatch.setattr("elspeth.web.composer.boot_probe.probe_composer_config", _reject_probe)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_COUNTER", counter)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_PROBE_LATENCY", latency)
        with (
            patch("httpx.AsyncClient", return_value=_StaticAsyncClient([])),
            pytest.raises(ComposerBootConfigError, match="composer sampling rejected"),
        ):
            async with lifespan(app):
                pass

        counter.add.assert_called_once()
        _, attributes = counter.add.call_args.args
        assert attributes["probe_status"] == "rejected"
        assert attributes["probed_model"] == "gpt-5.5"
        latency.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_skips_composer_probe_when_disabled(self, monkeypatch, tmp_path) -> None:
        app = create_app(_settings(tmp_path, composer_boot_probe_enabled=False))
        called = False

        async def _probe(**_kwargs: object) -> bool:
            nonlocal called
            called = True
            return True

        monkeypatch.setattr("elspeth.web.composer.boot_probe.probe_composer_config", _probe)
        with patch("httpx.AsyncClient", return_value=_StaticAsyncClient([])):
            async with lifespan(app):
                pass

        assert called is False

    @pytest.mark.asyncio
    async def test_lifespan_startup_orphan_cleanup_terminalizes_landscape_run(self, tmp_path) -> None:
        app = create_app(
            _settings(
                tmp_path,
                composer_boot_probe_enabled=False,
                orphan_run_check_interval_seconds=3600,
            )
        )
        session_service = app.state.session_service
        session = await session_service.create_session("alice", "Pipeline", "local")
        state = await session_service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        web_run = await session_service.create_run(session.id, state.id)
        landscape_run_id = "lscp-startup-orphan"
        await session_service.update_run_status(web_run.id, "running", landscape_run_id=landscape_run_id)

        with LandscapeDB.from_url(app.state.settings.get_landscape_url()) as db:
            RecorderFactory(db).run_lifecycle.begin_run(
                config={},
                canonical_version="v1",
                run_id=landscape_run_id,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )

        with patch("httpx.AsyncClient", return_value=_StaticAsyncClient([])):
            async with lifespan(app):
                pass

        updated_web_run = await session_service.get_run(web_run.id)
        assert updated_web_run.status == "cancelled"
        assert updated_web_run.finished_at is not None

        with LandscapeDB.from_url(app.state.settings.get_landscape_url()) as db:
            landscape_run = RecorderFactory(db).run_lifecycle.get_run(landscape_run_id)

        assert landscape_run is not None
        assert landscape_run.status == RunStatus.INTERRUPTED
        assert landscape_run.completed_at is not None

    @pytest.mark.asyncio
    async def test_lifespan_emits_composer_boot_config_attributes(self, monkeypatch, tmp_path) -> None:
        app = create_app(
            _settings(
                tmp_path,
                composer_boot_probe_enabled=True,
                composer_temperature=0.0,
                composer_seed=42,
                composer_advisor_model="anthropic/claude-sonnet-4-6",
            )
        )
        probed_models: list[str] = []
        counter = MagicMock(spec=["add"])
        latency = MagicMock(spec=["record"])

        async def _probe(**kwargs: object) -> bool:
            probed_models.append(str(kwargs["model"]))
            return True

        monkeypatch.setattr("elspeth.web.composer.boot_probe.probe_composer_config", _probe)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_COUNTER", counter)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_PROBE_LATENCY", latency)

        with patch("httpx.AsyncClient", return_value=_StaticAsyncClient([])):
            async with lifespan(app):
                pass

        assert probed_models == ["gpt-5.5", "anthropic/claude-sonnet-4-6"]
        assert counter.add.call_count == 2
        assert latency.record.call_count == 2
        counter_attributes = [call.args[1] for call in counter.add.call_args_list]
        assert [attrs["probed_model"] for attrs in counter_attributes] == probed_models
        for attributes in counter_attributes:
            assert attributes["composer_model"] == "gpt-5.5"
            assert attributes["composer_temperature"] == "0.0"
            assert attributes["composer_seed"] == "42"
            assert attributes["composer_advisor_model"] == "anthropic/claude-sonnet-4-6"
            assert attributes["probe_status"] == "success"
        for call in latency.record.call_args_list:
            value, attributes = call.args
            assert type(value) is int
            assert attributes["probe_status"] == "success"

    @pytest.mark.asyncio
    async def test_lifespan_records_transient_failure_when_composer_probe_times_out(self, monkeypatch, tmp_path) -> None:
        app = create_app(_settings(tmp_path, composer_boot_probe_enabled=True))
        counter = MagicMock(spec=["add"])
        latency = MagicMock(spec=["record"])
        cancelled = False

        async def _hanging_probe(**_kwargs: object) -> bool:
            nonlocal cancelled
            try:
                await asyncio.Event().wait()
            finally:
                cancelled = True
            return True

        monkeypatch.setattr("elspeth.web.composer.boot_probe.probe_composer_config", _hanging_probe)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_COUNTER", counter)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_PROBE_LATENCY", latency)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_PROBE_TIMEOUT_SECONDS", 0.01)

        async def _enter_and_exit_lifespan() -> None:
            async with lifespan(app):
                pass

        with patch("httpx.AsyncClient", return_value=_StaticAsyncClient([])):
            await asyncio.wait_for(_enter_and_exit_lifespan(), timeout=1.0)

        # Advisor is mandatory, so both the primary and advisor models are
        # probed. Both hang and time out as transient_failure; the loop does
        # not break on a transient failure.
        assert counter.add.call_count == 2
        probed = [call.args[1]["probed_model"] for call in counter.add.call_args_list]
        assert probed[0] == "gpt-5.5"
        for call in counter.add.call_args_list:
            _, attributes = call.args
            assert attributes["probe_status"] == "transient_failure"
        assert latency.record.call_count == 2
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_lifespan_records_local_error_when_composer_probe_raises_programmer_error(self, monkeypatch, tmp_path) -> None:
        app = create_app(_settings(tmp_path, composer_boot_probe_enabled=True))
        counter = MagicMock(spec=["add"])
        latency = MagicMock(spec=["record"])

        async def _buggy_probe(**_kwargs: object) -> bool:
            raise TypeError("signature drift")

        monkeypatch.setattr("elspeth.web.composer.boot_probe.probe_composer_config", _buggy_probe)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_COUNTER", counter)
        monkeypatch.setattr("elspeth.web.app._COMPOSER_BOOT_CONFIG_PROBE_LATENCY", latency)

        with (
            patch("httpx.AsyncClient", return_value=_StaticAsyncClient([])),
            pytest.raises(TypeError, match="signature drift"),
        ):
            async with lifespan(app):
                pass

        counter.add.assert_called_once()
        _, attributes = counter.add.call_args.args
        assert attributes["probe_status"] == "local_error"
        assert attributes["probed_model"] == "gpt-5.5"
        latency.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_propagates_catalog_client_context_failure(self, tmp_path) -> None:
        """Outer HTTP-client construction failures are startup failures, not fallback decisions."""
        app = create_app(_settings(tmp_path))

        with (
            patch("httpx.AsyncClient", return_value=_EnteringAsyncClientRaises()),
            pytest.raises(OSError, match="catalog client context failed"),
        ):
            async with lifespan(app):
                pass

    @pytest.mark.asyncio
    async def test_lifespan_awaits_execution_service_shutdown(self, tmp_path) -> None:
        app = create_app(_settings(tmp_path, composer_boot_probe_enabled=False))
        fake_execution_service = MagicMock(spec=["get_live_run_ids", "set_openrouter_catalog_snapshot", "shutdown"])
        fake_execution_service.get_live_run_ids.return_value = frozenset()
        fake_execution_service.shutdown = AsyncMock()

        with patch("elspeth.web.app.ExecutionServiceImpl", return_value=fake_execution_service):
            async with lifespan(app):
                pass

        fake_execution_service.shutdown.assert_awaited_once()


class TestSettingsFromEnv:
    """Tests for _settings_from_env() environment variable parsing."""

    @pytest.fixture(autouse=True)
    def _set_composer_env(self, monkeypatch) -> None:
        """Provide required composer fields via env vars for _settings_from_env()."""
        monkeypatch.setenv("ELSPETH_WEB__COMPOSER_MAX_COMPOSITION_TURNS", "15")
        monkeypatch.setenv("ELSPETH_WEB__COMPOSER_MAX_DISCOVERY_TURNS", "10")
        monkeypatch.setenv("ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS", "85.0")
        monkeypatch.setenv("ELSPETH_WEB__COMPOSER_RATE_LIMIT_PER_MINUTE", "10")
        # Phase 6A required: shareable_link_signing_key.
        #
        # DC-2 FIX-L: env-var ingestion now requires the value to be a
        # base64-encoded string (the documented ``openssl rand -base64 32``
        # recipe). The model's ``mode="before"`` validator decodes it
        # explicitly — no utf-8 multibyte ambiguity. 32 raw bytes decoded
        # is the validator floor; a 44-char base64 string satisfies it.
        import base64

        monkeypatch.setenv(
            "ELSPETH_WEB__SHAREABLE_LINK_SIGNING_KEY",
            base64.b64encode(b"\x00" * 32).decode("ascii"),
        )

    def test_parses_json_tuple_values(self, monkeypatch) -> None:
        """JSON-encoded lists are converted to tuples for tuple-typed fields."""
        monkeypatch.setenv("ELSPETH_WEB__CORS_ORIGINS", '["https://app.example.com"]')
        settings = _settings_from_env()
        assert settings.cors_origins == ("https://app.example.com",)

    def test_parses_json_list_with_multiple_items(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "ELSPETH_WEB__CORS_ORIGINS",
            '["https://a.example.com", "https://b.example.com"]',
        )
        settings = _settings_from_env()
        assert settings.cors_origins == ("https://a.example.com", "https://b.example.com")

    def test_plain_string_passes_through(self, monkeypatch) -> None:
        monkeypatch.setenv("ELSPETH_WEB__HOST", "127.0.0.1")
        settings = _settings_from_env()
        assert settings.host == "127.0.0.1"

    def test_json_integer_parsed(self, monkeypatch) -> None:
        monkeypatch.setenv("ELSPETH_WEB__PORT", "9090")
        settings = _settings_from_env()
        assert settings.port == 9090
        assert isinstance(settings.port, int)

    def test_server_secret_allowlist_from_json(self, monkeypatch) -> None:
        monkeypatch.setenv("ELSPETH_WEB__SERVER_SECRET_ALLOWLIST", '["MY_KEY"]')
        settings = _settings_from_env()
        assert settings.server_secret_allowlist == ("MY_KEY",)

    @pytest.mark.parametrize(
        ("raw_allowlist", "match"),
        [
            ('[""]', "must not be empty"),
            ('["A/B"]', "must match"),
            (f'["{"A" * 257}"]', "must be <= 256"),
        ],
    )
    def test_server_secret_allowlist_rejects_malformed_names(self, monkeypatch, raw_allowlist: str, match: str) -> None:
        monkeypatch.setenv("ELSPETH_WEB__SERVER_SECRET_ALLOWLIST", raw_allowlist)
        with pytest.raises(ValidationError, match=match):
            _settings_from_env()

    def test_secret_key_numeric_string_preserved(self, monkeypatch) -> None:
        """A string field set to a numeric value must stay str, not become int."""
        monkeypatch.setenv("ELSPETH_WEB__SECRET_KEY", "12345")
        settings = _settings_from_env()
        assert settings.secret_key == "12345"
        assert isinstance(settings.secret_key, str)

    def test_secret_key_bool_string_preserved(self, monkeypatch) -> None:
        """'true' must stay str, not become bool(True)."""
        monkeypatch.setenv("ELSPETH_WEB__SECRET_KEY", "true")
        settings = _settings_from_env()
        assert settings.secret_key == "true"
        assert isinstance(settings.secret_key, str)

    def test_secret_key_null_rejected(self, monkeypatch) -> None:
        """'null' → None for all fields; Pydantic rejects None for non-nullable str."""
        monkeypatch.setenv("ELSPETH_WEB__SECRET_KEY", "null")
        with pytest.raises(ValidationError, match="secret_key"):
            _settings_from_env()

    def test_nullable_field_null_becomes_none(self, monkeypatch) -> None:
        """'null' → None for nullable fields, enabling default fallback."""
        monkeypatch.setenv("ELSPETH_WEB__OIDC_AUTHORIZATION_ENDPOINT", "null")
        settings = _settings_from_env()
        assert settings.oidc_authorization_endpoint is None

    def test_nullable_db_url_null_becomes_none(self, monkeypatch) -> None:
        """'null' → None for landscape_url, so get_landscape_url() uses the default."""
        monkeypatch.setenv("ELSPETH_WEB__LANDSCAPE_URL", "null")
        settings = _settings_from_env()
        assert settings.landscape_url is None

    def test_port_string_coerced_by_pydantic(self, monkeypatch) -> None:
        """Numeric string for int field is coerced by Pydantic, not json.loads."""
        monkeypatch.setenv("ELSPETH_WEB__PORT", "9090")
        settings = _settings_from_env()
        assert settings.port == 9090
        assert isinstance(settings.port, int)


class TestJsonCollectionFieldsSync:
    """Structural test: _JSON_COLLECTION_FIELDS must stay in sync with WebSettings."""

    def test_all_tuple_fields_in_allowlist(self) -> None:
        """Every tuple-typed field on WebSettings must appear in _JSON_COLLECTION_FIELDS."""
        tuple_fields = {
            name for name, field_info in WebSettings.model_fields.items() if getattr(field_info.annotation, "__origin__", None) is tuple
        }
        missing = tuple_fields - _JSON_COLLECTION_FIELDS
        assert not missing, (
            f"Tuple-typed WebSettings fields missing from _JSON_COLLECTION_FIELDS: {missing}. "
            f"Add them so _settings_from_env() JSON-decodes them from env vars."
        )

    def test_no_non_tuple_fields_in_allowlist(self) -> None:
        """_JSON_COLLECTION_FIELDS must not contain non-tuple fields."""
        tuple_fields = {
            name for name, field_info in WebSettings.model_fields.items() if getattr(field_info.annotation, "__origin__", None) is tuple
        }
        extra = _JSON_COLLECTION_FIELDS - tuple_fields
        assert not extra, f"Non-tuple fields in _JSON_COLLECTION_FIELDS: {extra}. Scalar fields should not be JSON-decoded."


class TestPeriodicOrphanCleanup:
    """Tests for _periodic_orphan_cleanup background task."""

    @pytest.mark.asyncio
    async def test_calls_cancel_all_with_max_age(self) -> None:
        """Periodic cleanup passes max_age_seconds (not None) to cancel_all_orphaned_runs.

        Synchronisation: an ``asyncio.Event`` set by the mocked
        ``cancel_all_orphaned_runs`` makes the test wait for exactly one
        loop iteration — no wall-clock sleep, no inverted-flake risk if
        the loop scheduler is slow on a loaded CI box.
        """
        called = asyncio.Event()

        async def signal_called(**_: object) -> int:
            called.set()
            return 0

        mock_service = AsyncMock()
        mock_service.cancel_all_orphaned_runs.side_effect = signal_called
        mock_exec = MagicMock()
        mock_exec.get_live_run_ids.return_value = frozenset()

        telemetry = build_sessions_telemetry()
        task = asyncio.create_task(_periodic_orphan_cleanup(mock_service, mock_exec, telemetry, interval_seconds=0, max_age_seconds=900))
        try:
            await asyncio.wait_for(called.wait(), timeout=5.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        mock_service.cancel_all_orphaned_runs.assert_called_with(
            max_age_seconds=900,
            exclude_run_ids=frozenset(),
            reason="Orphaned by periodic cleanup — no active executor thread",
        )
        assert observed_value(telemetry.orphaned_runs_cancelled_total) == 0

    @pytest.mark.asyncio
    async def test_periodic_cleanup_emits_cancelled_count_to_telemetry(self) -> None:
        cleanup_called = asyncio.Event()

        async def signal_called(**_: object) -> int:
            cleanup_called.set()
            return 3

        mock_service = AsyncMock()
        mock_service.cancel_all_orphaned_runs.side_effect = signal_called
        mock_exec = MagicMock()
        mock_exec.get_live_run_ids.return_value = frozenset({"run-live"})
        telemetry = build_sessions_telemetry()

        task = asyncio.create_task(_periodic_orphan_cleanup(mock_service, mock_exec, telemetry, interval_seconds=0, max_age_seconds=900))
        try:
            await asyncio.wait_for(cleanup_called.wait(), timeout=5.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert observed_value(telemetry.orphaned_runs_cancelled_total) == 3
        counter = telemetry.orphaned_runs_cancelled_total
        assert isinstance(counter, _FakeCounter)
        amount, attrs, _context = counter.calls[0]
        assert amount == 3
        assert attrs == {"source": "periodic", "excluded_live_runs": 1}

    @pytest.mark.asyncio
    async def test_continues_after_exception(self) -> None:
        """Periodic cleanup logs recoverable audit/DB failures and keeps running.

        The catch in _periodic_orphan_cleanup is narrowed to
        (SQLAlchemyError, OSError). OperationalError models the realistic
        production failure — transient connection drop, lock timeout, or
        SQLite-busy — that the loop must survive. A prior iteration used
        RuntimeError here, which is now the wrong signal: RuntimeError is a
        programmer-bug class and must propagate past the catch (see the
        companion programmer-bug test below).

        The leak assertions on the structured log entry verify that the
        exc_info drop holds: the DB URL fragment and SQL statement from
        the OperationalError __cause__ chain must NOT appear in the
        emitted event.
        """
        from structlog.testing import capture_logs

        # Function-form side_effect rather than a two-item list: a finite
        # list exhausts into StopAsyncIteration (which the narrowed catch
        # deliberately does NOT absorb). A callable cycling "fail once,
        # then succeed forever" avoids that artifact while precisely
        # modelling transient-failure-then-recovery.  An ``asyncio.Event``
        # set on the second (recovery) call gates the test on observed
        # behaviour, not a wall-clock sleep.
        call_count = {"n": 0}
        recovered = asyncio.Event()

        async def cancel_side_effect(**_: object) -> int:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OperationalError("SELECT * FROM runs", {}, Exception("db unavailable"))
            recovered.set()
            return 2

        mock_service = AsyncMock()
        mock_service.cancel_all_orphaned_runs.side_effect = cancel_side_effect
        mock_exec = MagicMock()
        mock_exec.get_live_run_ids.return_value = frozenset()

        with capture_logs() as cap_logs:
            telemetry = build_sessions_telemetry()
            task = asyncio.create_task(
                _periodic_orphan_cleanup(mock_service, mock_exec, telemetry, interval_seconds=0, max_age_seconds=3600)
            )
            try:
                await asyncio.wait_for(recovered.wait(), timeout=5.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        assert mock_service.cancel_all_orphaned_runs.call_count >= 2

        failure_events = [entry for entry in cap_logs if entry.get("event") == "periodic_orphan_cleanup_failed"]
        assert len(failure_events) >= 1, cap_logs
        event = failure_events[0]
        assert event["exc_class"] == "OperationalError"
        # exc_info-derived fields MUST NOT appear — the canonical
        # redaction pattern applies to every slog.error on a path that
        # handles SQLAlchemy __cause__ chains.
        assert "exc_info" not in event
        assert "exception" not in event
        assert "stack_info" not in event
        assert "db unavailable" not in str(event)
        assert "SELECT * FROM runs" not in str(event)

    @pytest.mark.asyncio
    async def test_cancellation_is_clean(self) -> None:
        """Task cancellation doesn't raise or leave dangling state."""
        mock_service = AsyncMock()
        mock_service.cancel_all_orphaned_runs.return_value = 0
        mock_exec = MagicMock()
        mock_exec.get_live_run_ids.return_value = frozenset()

        telemetry = build_sessions_telemetry()
        task = asyncio.create_task(_periodic_orphan_cleanup(mock_service, mock_exec, telemetry, interval_seconds=10, max_age_seconds=3600))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_programmer_bug_terminates_task_and_does_not_log(self) -> None:
        """Programmer-bug exceptions (AttributeError, TypeError, AssertionError)
        raised by either callee MUST escape the narrowed catch, killing the
        background task so the bug surfaces at lifespan shutdown rather than
        being silently logged every interval.

        This is the guardrail for the narrowed catch at
        _periodic_orphan_cleanup: replacing ``except Exception`` with
        ``except (SQLAlchemyError, OSError)`` means a drifted attribute on
        ExecutionServiceImpl, a signature change on SessionServiceImpl, or
        an assertion violation now terminates the task immediately. A future
        regression that re-widens the catch would turn production into an
        error-storm — one bug logged every interval forever — and would fail
        this test because (a) the task would not raise when awaited and (b)
        the periodic_orphan_cleanup_failed event would appear.
        """
        from structlog.testing import capture_logs

        mock_service = AsyncMock()
        mock_exec = MagicMock()
        # AttributeError from get_live_run_ids — canonical Tier-1/2 programmer
        # bug (e.g., _shutdown_events replaced with a non-mapping type).
        mock_exec.get_live_run_ids.side_effect = AttributeError("ExecutionServiceImpl has no attribute '_shutdown_events'")

        with capture_logs() as cap_logs:
            telemetry = build_sessions_telemetry()
            task = asyncio.create_task(
                _periodic_orphan_cleanup(mock_service, mock_exec, telemetry, interval_seconds=0, max_age_seconds=3600)
            )
            # Task must terminate on its own with the AttributeError.
            # ``asyncio.wait_for`` provides the deterministic wait: if the
            # narrow-catch regresses to ``except Exception``, the task
            # stays alive and the wait_for fires TimeoutError instead of
            # AttributeError — distinguishable failure modes in the
            # pytest.raises block below.  No task.cancel() anywhere on
            # the happy path — if the bug propagates, cancel is
            # unnecessary; if it doesn't, cancel would mask the bug.
            with pytest.raises(AttributeError) as exc_info:
                await asyncio.wait_for(task, timeout=5.0)

        assert "_shutdown_events" in str(exc_info.value)

        # cancel_all_orphaned_runs must NOT have been called — the bug
        # short-circuits before the DB write.
        mock_service.cancel_all_orphaned_runs.assert_not_called()

        # No periodic_orphan_cleanup_failed event — the catch did not fire,
        # so the structured logging path was not reached. A regression that
        # widens the catch would cause this assertion to fail.
        failure_events = [entry for entry in cap_logs if entry.get("event") == "periodic_orphan_cleanup_failed"]
        assert failure_events == [], cap_logs

    @pytest.mark.asyncio
    async def test_excludes_live_run_ids(self) -> None:
        """Periodic cleanup passes live run IDs as exclude_run_ids."""
        mock_service = AsyncMock()
        cleanup_called = asyncio.Event()

        async def _cancel_all_orphaned_runs(**_kwargs):
            cleanup_called.set()
            return 0

        mock_service.cancel_all_orphaned_runs.side_effect = _cancel_all_orphaned_runs
        mock_exec = MagicMock()
        live_ids = frozenset({"run-1", "run-2"})
        mock_exec.get_live_run_ids.return_value = live_ids

        telemetry = build_sessions_telemetry()
        task = asyncio.create_task(_periodic_orphan_cleanup(mock_service, mock_exec, telemetry, interval_seconds=0, max_age_seconds=3600))
        await asyncio.wait_for(cleanup_called.wait(), timeout=5.0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        mock_exec.get_live_run_ids.assert_called()
        mock_service.cancel_all_orphaned_runs.assert_called_with(
            max_age_seconds=3600,
            exclude_run_ids=live_ids,
            reason="Orphaned by periodic cleanup — no active executor thread",
        )


class TestOrphanLandscapeReconciliation:
    """Cross-DB orphan cleanup reconciliation."""

    def test_terminalizes_landscape_run_rows_for_cancelled_session_runs(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "runs").mkdir(exist_ok=True)
        landscape_url = settings.get_landscape_url()
        landscape_run_id = "lscp-orphan-1"
        with LandscapeDB.from_url(landscape_url) as db:
            RecorderFactory(db).run_lifecycle.begin_run(
                config={},
                canonical_version="v1",
                run_id=landscape_run_id,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )

        cancelled_run = RunRecord(
            id=uuid4(),
            session_id=uuid4(),
            state_id=uuid4(),
            status="cancelled",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            error="Orphaned by server restart - no active process",
            landscape_run_id=landscape_run_id,
            pipeline_yaml=None,
        )

        app_module._finalize_orphaned_landscape_runs(landscape_url, [cancelled_run])

        with LandscapeDB.from_url(landscape_url) as db:
            run = RecorderFactory(db).run_lifecycle.get_run(landscape_run_id)

        assert run is not None
        assert run.status == RunStatus.INTERRUPTED
        assert run.completed_at is not None


class TestDataDirCreation:
    """Bug 4: create_app() must create data_dir before any DB access."""

    def test_create_app_creates_nonexistent_data_dir(self, tmp_path) -> None:
        fresh_dir = tmp_path / "nonexistent" / "nested"
        settings = WebSettings(
            data_dir=fresh_dir,
            composer_max_composition_turns=15,
            composer_max_discovery_turns=10,
            composer_timeout_seconds=85.0,
            composer_rate_limit_per_minute=10,
            shareable_link_signing_key=SecretBytes(b"\x00" * 32),
        )
        create_app(settings)
        assert fresh_dir.exists()
        assert fresh_dir.is_dir()


class TestValidationErrorRedaction:
    """SECURITY: 422 responses must never echo sensitive request body values.

    The global RequestValidationError handler registered in create_app()
    allowlists only {type, loc, msg} — stripping ``input``, ``ctx``, and
    ``url`` to prevent credential leakage on any route.

    These tests exercise the *real* handler wired by create_app(), unlike the
    unit tests in test_routes.py which register a local duplicate.
    """

    _SAFE_KEYS = frozenset({"type", "loc", "msg"})

    @staticmethod
    def _authed_client(tmp_path: Path) -> TestClient:
        """Build a TestClient against create_app() with auth bypassed."""
        from elspeth.web.auth.middleware import get_current_user
        from elspeth.web.auth.models import UserIdentity

        app = create_app(_settings(tmp_path))

        identity = UserIdentity(user_id="test-user", username="test-user")

        async def _mock_user() -> UserIdentity:
            return identity

        app.dependency_overrides[get_current_user] = _mock_user
        return TestClient(app, raise_server_exceptions=False)

    def test_secrets_route_redacts_input(self, tmp_path) -> None:
        """POST /api/secrets with wrong value type must not echo the value."""
        client = self._authed_client(tmp_path)
        resp = client.post(
            "/api/secrets",
            json={"name": "API_KEY", "value": {"nested": "super-secret-hunter2"}},
        )
        assert resp.status_code == 422
        body = resp.json()
        body_text = resp.text
        assert "super-secret-hunter2" not in body_text
        for error in body["detail"]:
            assert set(error.keys()) <= self._SAFE_KEYS

    def test_redaction_preserves_error_structure(self, tmp_path) -> None:
        """Redacted errors retain type, loc, msg for client debugging."""
        client = self._authed_client(tmp_path)
        resp = client.post(
            "/api/secrets",
            json={"name": "API_KEY", "value": {"bad": "type"}},
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert len(errors) > 0
        for error in errors:
            assert "type" in error
            assert "loc" in error
            assert "msg" in error

    def test_redaction_strips_input_ctx_url_keys(self, tmp_path) -> None:
        """Forbidden keys (input, ctx, url) must never appear in 422 detail."""
        client = self._authed_client(tmp_path)
        resp = client.post(
            "/api/secrets",
            json={"name": "API_KEY", "value": 12345},
        )
        assert resp.status_code == 422
        _FORBIDDEN_KEYS = {"input", "ctx", "url"}
        for error in resp.json()["detail"]:
            assert not _FORBIDDEN_KEYS & set(error.keys()), f"Forbidden keys leaked in 422 response: {_FORBIDDEN_KEYS & set(error.keys())}"

    def test_sessions_message_route_redacts_input(self, tmp_path) -> None:
        """POST to a session message route with invalid body must not echo content."""
        client = self._authed_client(tmp_path)
        # Send a message with state_id as a non-UUID string — triggers 422
        resp = client.post(
            "/api/sessions/00000000-0000-0000-0000-000000000000/messages",
            json={"content": "leaked-password-value", "state_id": "not-a-uuid"},
        )
        assert resp.status_code == 422
        body_text = resp.text
        assert "leaked-password-value" not in body_text
        for error in resp.json()["detail"]:
            assert set(error.keys()) <= self._SAFE_KEYS


class TestSecretsExceptionHandlers:
    """Trust-boundary translation: store-layer typed errors → HTTP status codes.

    Exercises the app-level handlers for:

    * ``FingerprintKeyMissingError`` → 503 (deployment misconfigured)
    * ``SecretDecryptionError``     → 409 (re-save required)
    * ``SQLAlchemyError``            → 503 (database unavailable)
    * ``OSError``                    → 503 (SQLite / filesystem level)

    Redaction invariants (from the canonical SQLAlchemy-redaction pattern):
      - response body never contains ``str(exc)`` or ``__cause__`` fragments
      - slog event omits ``exc_info`` / ``exception`` fields
      - DB URLs, SQL text, bound parameters absent from both
    """

    @staticmethod
    def _authed_client(tmp_path: Path) -> TestClient:
        from elspeth.web.auth.middleware import get_current_user
        from elspeth.web.auth.models import UserIdentity

        app = create_app(_settings(tmp_path))
        identity = UserIdentity(user_id="test-user", username="test-user")

        async def _mock_user() -> UserIdentity:
            return identity

        app.dependency_overrides[get_current_user] = _mock_user
        return TestClient(app, raise_server_exceptions=False)

    # -- FingerprintKeyMissingError → 503 -------------------------------------

    def test_fingerprint_missing_on_create_returns_503(self, tmp_path, monkeypatch) -> None:
        """POST /api/secrets with ELSPETH_FINGERPRINT_KEY unset → 503.

        The eager-fingerprint design in ``UserSecretStore.set_secret``
        raises ``FingerprintKeyMissingError`` before any DB write; the
        app-level handler maps it to 503 with a deployment-guidance
        body and a correlation ``request_id``.
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        client = self._authed_client(tmp_path)
        resp = client.post("/api/secrets", json={"name": "API_KEY", "value": "v"})

        assert resp.status_code == 503
        body = resp.json()
        assert body["error_type"] == "fingerprint_key_missing"
        assert "ELSPETH_FINGERPRINT_KEY" in body["detail"]
        # Correlation id must be present and echoed on the response header.
        assert body["request_id"]
        assert resp.headers["X-Request-ID"] == body["request_id"]

    def test_validate_secret_rejects_invalid_path_name_before_service_access(self, tmp_path, monkeypatch) -> None:
        """Invalid path names must fail at the HTTP boundary without probing stores."""
        client = self._authed_client(tmp_path)

        def fail_if_called(*_args: object, **_kwargs: object) -> bool:
            raise AssertionError("secret service must not be called for invalid path names")

        monkeypatch.setattr(client.app.state.secret_service, "check_user_ref_resolvable", fail_if_called)

        resp = client.post("/api/secrets/1bad/validate")

        assert resp.status_code == 422
        assert "1bad" not in resp.text

    def test_delete_secret_rejects_oversized_path_name_before_service_access(self, tmp_path, monkeypatch) -> None:
        """Oversized path names must fail closed without echoing the attacker-controlled name."""
        client = self._authed_client(tmp_path)
        oversized_name = "A" * 300

        def fail_if_called(*_args: object, **_kwargs: object) -> bool:
            raise AssertionError("secret service must not be called for invalid path names")

        monkeypatch.setattr(client.app.state.secret_service, "delete_user_secret", fail_if_called)

        resp = client.delete(f"/api/secrets/{oversized_name}")

        assert resp.status_code == 422
        assert oversized_name not in resp.text

    # -- SecretDecryptionError → 409 ------------------------------------------

    def test_decryption_failure_on_validate_returns_409(self, tmp_path, monkeypatch) -> None:
        """Validate against a secret whose master key was rotated → 409.

        Seeds a row with master_key A, then monkeypatches the store's
        master key to B so the next decrypt fails with InvalidToken —
        which the store translates to ``SecretDecryptionError`` and the
        app handler maps to 409 with re-save guidance.
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-k")
        client = self._authed_client(tmp_path)

        # Write one secret under the original master key.
        create = client.post("/api/secrets", json={"name": "ROTATED", "value": "v"})
        assert create.status_code == 201

        # Rotate the in-memory master key on the running store — emulates
        # a deploy-time secret_key rotation without the old secrets
        # being re-saved.
        app = client.app
        user_store = app.state.secret_service._user_store
        user_store._master_key = "rotated-master-key"

        resp = client.post("/api/secrets/ROTATED/validate")

        assert resp.status_code == 409
        body = resp.json()
        assert body["error_type"] == "secret_decryption_failed"
        assert "re-save" in body["detail"].lower()
        assert body["request_id"]

    # -- SQLAlchemyError → 503 ------------------------------------------------

    def test_sqlalchemy_error_on_list_returns_503(self, tmp_path, monkeypatch) -> None:
        """Underlying ``OperationalError`` on list must surface as 503 with a redacted body.

        Redaction invariant: the response body MUST NOT contain ``str(exc)``,
        the SQL statement, or bound parameters (which can carry secrets or
        DB URLs via ``OperationalError.__cause__``).
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-k")
        client = self._authed_client(tmp_path)

        def _raise_operational(*args, **kwargs):
            raise OperationalError(
                "SELECT * FROM user_secrets WHERE name = ?",
                {"name": "LEAK_ME"},
                Exception("postgresql://admin:sekretpass@db.internal/prod"),  # secret-scan: allow-this-line
            )

        app = client.app
        monkeypatch.setattr(app.state.secret_service, "list_refs", _raise_operational)

        resp = client.get("/api/secrets")
        assert resp.status_code == 503
        body = resp.json()
        body_text = resp.text
        assert body["error_type"] == "database_unavailable"
        assert body["request_id"]

        # Redaction invariant: no DB URL, SQL fragment, or bound param leaks.
        assert "postgresql://" not in body_text
        assert "sekretpass" not in body_text
        assert "SELECT * FROM" not in body_text
        assert "LEAK_ME" not in body_text

    def test_compile_error_is_not_laundered_to_database_unavailable(self, tmp_path, monkeypatch) -> None:
        """Programmer/configuration SQLAlchemy bugs must not be translated to retryable 503s."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-k")
        client = self._authed_client(tmp_path)

        def _raise_compile_error(*args, **kwargs):
            raise CompileError("broken query shape")

        app = client.app
        monkeypatch.setattr(app.state.secret_service, "list_refs", _raise_compile_error)

        resp = client.get("/api/secrets")
        assert resp.status_code == 500
        assert "database_unavailable" not in resp.text

    # -- OSError → 503 --------------------------------------------------------

    def test_oserror_on_list_returns_503(self, tmp_path, monkeypatch) -> None:
        """A bare OSError (e.g., SQLite disk-full) must also map to 503.

        SQLite can raise OSError before SQLAlchemy wraps it; the handler
        must cover that escape path too.
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-k")
        client = self._authed_client(tmp_path)

        def _raise_oserror(*args, **kwargs):
            raise OSError(28, "No space left on device", "/var/lib/elspeth.db")

        app = client.app
        monkeypatch.setattr(app.state.secret_service, "list_refs", _raise_oserror)

        resp = client.get("/api/secrets")
        assert resp.status_code == 503
        body = resp.json()
        assert body["error_type"] == "storage_unavailable"
        assert body["request_id"]
        # Redaction: the underlying DB file path must not leak.
        assert "/var/lib/elspeth.db" not in resp.text

    def test_filenotfound_error_is_not_laundered_to_storage_unavailable(self, tmp_path, monkeypatch) -> None:
        """Missing-path bugs must not become retryable storage-outage responses."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-k")
        client = self._authed_client(tmp_path)

        def _raise_missing_path(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "/tmp/should-have-existed")

        app = client.app
        monkeypatch.setattr(app.state.secret_service, "list_refs", _raise_missing_path)

        resp = client.get("/api/secrets")
        assert resp.status_code == 500
        assert "storage_unavailable" not in resp.text

    # -- Hypothesis property: TOCTOU-free create ------------------------------

    def test_create_ack_omits_available_and_validate_true_agrees(self, tmp_path, monkeypatch) -> None:
        """Property: create omits availability and immediate validate agrees.

        Post-eager-fingerprint, a 201 create response no longer carries a
        redundant constant ``available`` field. The honest availability
        check is the immediate validate call of the same name (with no
        intervening DELETE), which must return ``available=True``.
        """
        try:
            from hypothesis import given
            from hypothesis import strategies as st
        except ImportError:
            pytest.skip("hypothesis not installed")

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-k")
        client = self._authed_client(tmp_path)

        # Names match the schema regex: ^[A-Za-z][A-Za-z0-9_]*$
        name_strategy = st.from_regex(r"^[A-Za-z][A-Za-z0-9_]{0,40}$", fullmatch=True)
        # Value must have at least one visible character; keep to printable
        # ASCII for the body to survive JSON encoding unambiguously.
        value_strategy = st.text(
            alphabet=st.characters(
                min_codepoint=0x21,
                max_codepoint=0x7E,
                blacklist_categories=(),
            ),
            min_size=1,
            max_size=40,
        )

        @given(name=name_strategy, value=value_strategy)
        def _prop(name: str, value: str) -> None:
            create = client.post("/api/secrets", json={"name": name, "value": value})
            if create.status_code != 201:
                # Schema validators may reject some generated names; skip.
                return
            assert "available" not in create.json()
            validate = client.post(f"/api/secrets/{name}/validate")
            assert validate.status_code == 200
            assert validate.json()["available"] is True

        _prop()
