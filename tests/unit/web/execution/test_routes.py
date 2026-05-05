"""Tests for execution REST endpoints and WebSocket.

Routes delegate to ExecutionServiceImpl — these tests verify HTTP
semantics, status codes, and request/response contracts.

The app factory's real wiring is bypassed by setting app.state.*
directly with mocks, matching the dependency injection pattern used
by the route handlers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.routing import Route

from elspeth.web.auth.models import UserIdentity
from elspeth.web.execution.schemas import (
    DiscardSummary,
    RunDiagnosticsResponse,
    RunDiagnosticSummary,
    RunStatusResponse,
    ValidationCheck,
    ValidationResult,
)
from elspeth.web.sessions.protocol import RunAlreadyActiveError

# ── Helpers ───────────────────────────────────────────────────────────

_TEST_USER_ID = "test-user-123"


def _request_for_app(app: FastAPI) -> Request:
    return Request({"type": "http", "app": app, "headers": [], "method": "GET", "path": "/"})


def _route_endpoint(app: FastAPI, name: str) -> Callable[..., Awaitable[Any]]:
    for route in app.routes:
        if isinstance(route, Route) and route.name == name:
            return cast(Callable[..., Awaitable[Any]], route.endpoint)
    raise AssertionError(f"Route endpoint {name!r} not found")


def _create_test_app(
    execution_service: MagicMock | None = None,
    broadcaster: MagicMock | None = None,
) -> FastAPI:
    """Create a minimal FastAPI app with execution routes wired.

    Bypasses the full create_app() to avoid real DB setup, auth provider
    construction, and lifespan side effects. Overrides get_current_user
    to return a fake user for auth. Sets up mock session_service for
    ownership verification.
    """
    from elspeth.web.auth.middleware import get_current_user
    from elspeth.web.auth.models import UserIdentity
    from elspeth.web.execution.routes import create_execution_router

    app = FastAPI()
    app.state.execution_service = execution_service or MagicMock()
    app.state.broadcaster = broadcaster or MagicMock()
    app.state.auth_provider = MagicMock()

    # Mock session_service for ownership checks
    mock_session_service = MagicMock()
    mock_session = MagicMock()
    mock_session.user_id = _TEST_USER_ID
    mock_session.auth_provider_type = "local"
    mock_session_service.get_session = AsyncMock(return_value=mock_session)
    mock_session_service.get_run = AsyncMock(return_value=MagicMock(session_id=uuid4()))
    app.state.session_service = mock_session_service

    # Mock settings for ownership checks
    mock_settings = MagicMock()
    mock_settings.auth_provider = "local"
    app.state.settings = mock_settings

    fake_user = UserIdentity(user_id=_TEST_USER_ID, username="testuser")

    async def _fake_current_user() -> UserIdentity:
        return fake_user

    app.dependency_overrides[get_current_user] = _fake_current_user

    app.include_router(create_execution_router())

    # Register app-level exception handler matching Seam Contract D
    from fastapi import Request as FastAPIRequest
    from fastapi.responses import JSONResponse

    from elspeth.web.sessions.protocol import RunAlreadyActiveError

    @app.exception_handler(RunAlreadyActiveError)
    async def handle_run_already_active(request: FastAPIRequest, exc: RunAlreadyActiveError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "error_type": "run_already_active"},
        )

    return app


# ── REST Endpoint Tests ───────────────────────────────────────────────


class TestValidateEndpoint:
    """POST /api/sessions/{session_id}/validate"""

    @pytest.mark.asyncio
    async def test_valid_pipeline_returns_200(self) -> None:
        svc = MagicMock()
        svc.validate = AsyncMock(
            return_value=ValidationResult(
                is_valid=True,
                checks=[
                    ValidationCheck(name="settings_load", passed=True, detail="OK"),
                ],
                errors=[],
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/validate")
            assert resp.status_code == 200
            body = resp.json()
            assert body["is_valid"] is True
            assert len(body["checks"]) == 1

    @pytest.mark.asyncio
    async def test_validate_delegates_to_service(self) -> None:
        """AC #16: validate route delegates to service.validate()."""
        svc = MagicMock()
        svc.validate = AsyncMock(return_value=ValidationResult(is_valid=True, checks=[], errors=[]))
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/validate")
            assert resp.status_code == 200
            svc.validate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_pipeline_returns_200_with_errors(self) -> None:
        svc = MagicMock()
        svc.validate = AsyncMock(
            return_value=ValidationResult(
                is_valid=False,
                checks=[
                    ValidationCheck(name="settings_load", passed=False, detail="Bad YAML"),
                ],
                errors=[],
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/validate")
            assert resp.status_code == 200
            body = resp.json()
            assert body["is_valid"] is False


class TestExecuteEndpoint:
    """POST /api/sessions/{session_id}/execute"""

    @pytest.mark.asyncio
    async def test_execute_returns_202_with_run_id(self) -> None:
        expected_run_id = uuid4()
        svc = MagicMock()
        svc.execute = AsyncMock(return_value=expected_run_id)
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/execute")
            assert resp.status_code == 202
            body = resp.json()
            assert body["run_id"] == str(expected_run_id)

    @pytest.mark.asyncio
    async def test_execute_with_active_run_returns_409(self) -> None:
        svc = MagicMock()
        svc.execute = AsyncMock(side_effect=RunAlreadyActiveError("Already active"))
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/execute")
            assert resp.status_code == 409
            body = resp.json()
            # Seam Contract D: flat envelope, not nested
            assert body["error_type"] == "run_already_active"
            assert "detail" in body

    @pytest.mark.asyncio
    async def test_execute_returns_422_with_structured_semantic_payload(self) -> None:
        """Semantic contract violations surface as 422 with structured payload.

        Without the dedicated handler, ``SemanticContractViolationError``
        falls through to the bare ``except ValueError`` branch and the
        client receives 404 with detail=str(exc) — losing the structured
        ``entries`` and ``contracts`` records the frontend uses to
        render banners and per-node hints.
        """
        from elspeth.contracts.plugin_semantics import (
            ContentKind,
            FieldSemanticFacts,
            FieldSemanticRequirement,
            SemanticEdgeContract,
            SemanticOutcome,
            TextFraming,
            UnknownSemanticPolicy,
        )
        from elspeth.web.composer.state import ValidationEntry
        from elspeth.web.execution.errors import SemanticContractViolationError

        facts = FieldSemanticFacts(
            field_name="content",
            content_kind=ContentKind.PLAIN_TEXT,
            text_framing=TextFraming.COMPACT,
            fact_code="web_scrape.content.compact",
        )
        req = FieldSemanticRequirement(
            field_name="content",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
            accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
            requirement_code="line_explode.source_field.line_framed_text",
            unknown_policy=UnknownSemanticPolicy.FAIL,
        )
        contract = SemanticEdgeContract(
            from_id="scrape",
            to_id="explode",
            consumer_plugin="line_explode",
            producer_plugin="web_scrape",
            producer_field="content",
            consumer_field="content",
            producer_facts=facts,
            requirement=req,
            outcome=SemanticOutcome.CONFLICT,
        )
        entry = ValidationEntry(
            "node:explode",
            "Semantic contract violation: 'scrape' -> 'explode'.",
            "high",
        )
        exc = SemanticContractViolationError(entries=(entry,), contracts=(contract,))

        svc = MagicMock()
        svc.execute = AsyncMock(side_effect=exc)
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/execute")
        assert resp.status_code == 422
        body = resp.json()
        detail = body["detail"]
        assert detail["kind"] == "semantic_contract_violation"
        assert detail["errors"][0]["component"] == "node:explode"
        assert detail["errors"][0]["severity"] == "high"
        assert detail["semantic_contracts"][0]["outcome"] == "conflict"
        assert detail["semantic_contracts"][0]["consumer_plugin"] == "line_explode"
        assert detail["semantic_contracts"][0]["from_id"] == "scrape"
        assert detail["semantic_contracts"][0]["requirement_code"] == "line_explode.source_field.line_framed_text"


class TestRunDiagnosticsEndpoint:
    """GET/POST /api/runs/{run_id}/diagnostics."""

    @pytest.mark.asyncio
    async def test_running_run_uses_web_run_id_as_landscape_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="running",
                started_at=datetime.now(UTC),
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )
        )
        captured: dict[str, Any] = {}

        def fake_load(*args: object, **kwargs: object) -> RunDiagnosticsResponse:
            captured.update(kwargs)
            return RunDiagnosticsResponse(
                run_id=str(run_id),
                landscape_run_id=str(run_id),
                run_status="running",
                summary=RunDiagnosticSummary(
                    token_count=0,
                    preview_limit=12,
                    preview_truncated=False,
                    state_counts={},
                    operation_counts={},
                    latest_activity_at=None,
                ),
                tokens=[],
                operations=[],
                artifacts=[],
            )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.load_run_diagnostics_for_settings", fake_load)
        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)
        app = _create_test_app(execution_service=svc)
        endpoint = _route_endpoint(app, "get_run_diagnostics")
        response = await endpoint(
            run_id,
            _request_for_app(app),
            limit=12,
            user=UserIdentity(user_id=_TEST_USER_ID, username="testuser"),
            service=svc,
        )

        assert response.run_id == str(run_id)
        assert captured["run_id"] == str(run_id)
        assert captured["landscape_run_id"] == str(run_id)
        assert captured["run_status"] == "running"
        assert captured["limit"] == 12

    @pytest.mark.asyncio
    async def test_evaluate_diagnostics_calls_composer_with_bounded_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="running",
                started_at=datetime.now(UTC),
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=str(run_id),
            )
        )
        diagnostics = RunDiagnosticsResponse(
            run_id=str(run_id),
            landscape_run_id=str(run_id),
            run_status="running",
            summary=RunDiagnosticSummary(
                token_count=1,
                preview_limit=50,
                preview_truncated=False,
                state_counts={"open": 1},
                operation_counts={"source_load": 1},
                latest_activity_at=None,
            ),
            tokens=[],
            operations=[],
            artifacts=[],
        )
        monkeypatch.setattr(
            "elspeth.web.execution.routes.load_run_diagnostics_for_settings",
            lambda *args, **kwargs: diagnostics,
        )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)
        captured: dict[str, Any] = {}

        class FakeComposer:
            async def explain_run_diagnostics(self, snapshot: dict[str, object]) -> str:
                captured.update(snapshot)
                return (
                    '{"headline":"The run is processing data",'
                    '"evidence":["1 token is visible in the runtime trace.",'
                    '"Source loading has started."],'
                    '"meaning":"The pipeline is not idle; it has runtime records for the current row.",'
                    '"next_steps":["Refresh diagnostics if this does not change soon."]}'
                )

        app = _create_test_app(execution_service=svc)
        app.state.composer_service = FakeComposer()
        endpoint = _route_endpoint(app, "evaluate_run_diagnostics")
        response = await endpoint(
            run_id,
            _request_for_app(app),
            limit=50,
            user=UserIdentity(user_id=_TEST_USER_ID, username="testuser"),
            service=svc,
        )

        assert response.run_id == str(run_id)
        assert response.explanation == "The pipeline is not idle; it has runtime records for the current row."
        assert response.working_view.headline == "The run is processing data"
        assert response.working_view.evidence == [
            "1 token is visible in the runtime trace.",
            "Source loading has started.",
        ]
        assert response.working_view.next_steps == ["Refresh diagnostics if this does not change soon."]
        assert captured["run_id"] == str(run_id)
        assert captured["summary"]["token_count"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_diagnostics_falls_back_for_plain_text_explanation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="running",
                started_at=datetime.now(UTC),
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=str(run_id),
            )
        )
        diagnostics = RunDiagnosticsResponse(
            run_id=str(run_id),
            landscape_run_id=str(run_id),
            run_status="running",
            summary=RunDiagnosticSummary(
                token_count=3,
                preview_limit=50,
                preview_truncated=False,
                state_counts={"completed": 2, "running": 1},
                operation_counts={},
                latest_activity_at=datetime.now(UTC),
            ),
            tokens=[],
            operations=[],
            artifacts=[],
        )
        monkeypatch.setattr(
            "elspeth.web.execution.routes.load_run_diagnostics_for_settings",
            lambda *args, **kwargs: diagnostics,
        )

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("elspeth.web.execution.routes.asyncio.to_thread", fake_to_thread)

        class FakeComposer:
            async def explain_run_diagnostics(self, snapshot: dict[str, object]) -> str:
                return "The run is still working through the data."

        app = _create_test_app(execution_service=svc)
        app.state.composer_service = FakeComposer()
        endpoint = _route_endpoint(app, "evaluate_run_diagnostics")
        response = await endpoint(
            run_id,
            _request_for_app(app),
            limit=50,
            user=UserIdentity(user_id=_TEST_USER_ID, username="testuser"),
            service=svc,
        )

        assert response.explanation == "The run is still working through the data."
        assert response.working_view.meaning == "The run is still working through the data."
        assert response.working_view.headline == "Runtime records are updating"
        assert "3 tokens are visible in the runtime trace." in response.working_view.evidence


class TestExecuteIDORAndPathTraversal:
    """IDOR and path traversal defense-in-depth checks in execute().

    The state_id and blob_ref IDOR surfaces have strict parity
    contracts: the "does not exist anywhere" and "exists in another
    user's session" branches MUST produce byte-identical responses.
    Asserting substring containment ("does not belong" in body) is
    itself a pin of the oracle and is forbidden here — parity tests
    use byte equality of the full response body AND status code.
    See ``StateAccessError`` (web/execution/protocol.py) for the rationale.
    """

    @pytest.mark.asyncio
    async def test_execute_cross_session_state_id_returns_idor_safe_body(self) -> None:
        """Cross-session state_id surfaces as the fixed "State not found" literal.

        The service raises ``StateAccessError``; the route MUST
        collapse it to a byte-identical 404 body that does not
        distinguish cross-session from nonexistent.
        """
        from elspeth.web.execution.protocol import StateAccessError

        svc = MagicMock()
        svc.execute = AsyncMock(side_effect=StateAccessError("any-state-uuid"))
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                f"/api/sessions/{uuid4()}/execute",
                params={"state_id": str(uuid4())},
            )
            assert resp.status_code == 404
            assert resp.json() == {"detail": "State not found"}

    @pytest.mark.asyncio
    async def test_execute_state_id_idor_branches_are_byte_identical(self) -> None:
        """Nonexistent state_id and cross-session state_id MUST be indistinguishable.

        This is the canonical IDOR-parity check: run both branches
        through the route with distinct arguments and assert the
        raw response bytes (status + body) are identical.  A future
        regression that re-introduces a distinguishable message will
        fail here with a diff an operator can read directly.
        """
        from elspeth.web.execution.protocol import StateAccessError

        # Branch 1: state UUID does not exist anywhere in the DB.
        svc_a = MagicMock()
        svc_a.execute = AsyncMock(side_effect=StateAccessError(str(uuid4())))
        app_a = _create_test_app(execution_service=svc_a)

        # Branch 2: state UUID exists but belongs to another session.
        svc_b = MagicMock()
        svc_b.execute = AsyncMock(side_effect=StateAccessError(str(uuid4())))
        app_b = _create_test_app(execution_service=svc_b)

        async with (
            AsyncClient(transport=ASGITransport(app=app_a), base_url="http://test") as client_a,
            AsyncClient(transport=ASGITransport(app=app_b), base_url="http://test") as client_b,
        ):
            resp_a = await client_a.post(
                f"/api/sessions/{uuid4()}/execute",
                params={"state_id": str(uuid4())},
            )
            resp_b = await client_b.post(
                f"/api/sessions/{uuid4()}/execute",
                params={"state_id": str(uuid4())},
            )

        assert resp_a.status_code == resp_b.status_code == 404
        assert resp_a.content == resp_b.content
        assert resp_a.json() == {"detail": "State not found"}

    @pytest.mark.asyncio
    async def test_execute_blob_ref_idor_branches_are_byte_identical(self) -> None:
        """Nonexistent blob_ref and cross-session blob_ref MUST be indistinguishable.

        Before this fix, nonexistent-blob propagated as an uncaught
        ``BlobNotFoundError`` (HTTP 500) while cross-session-blob
        raised ``ValueError`` (HTTP 404, body leaking "does not
        belong to session").  The HTTP status itself was a side
        channel — a two-layer oracle strictly worse than state_id's.
        Both branches now surface as ``BlobNotFoundError``; the
        route collapses them to a byte-identical 404.
        """
        from elspeth.web.blobs.protocol import BlobNotFoundError

        svc_a = MagicMock()
        svc_a.execute = AsyncMock(side_effect=BlobNotFoundError(str(uuid4())))
        app_a = _create_test_app(execution_service=svc_a)

        svc_b = MagicMock()
        svc_b.execute = AsyncMock(side_effect=BlobNotFoundError(str(uuid4())))
        app_b = _create_test_app(execution_service=svc_b)

        async with (
            AsyncClient(transport=ASGITransport(app=app_a), base_url="http://test") as client_a,
            AsyncClient(transport=ASGITransport(app=app_b), base_url="http://test") as client_b,
        ):
            resp_a = await client_a.post(f"/api/sessions/{uuid4()}/execute")
            resp_b = await client_b.post(f"/api/sessions/{uuid4()}/execute")

        assert resp_a.status_code == resp_b.status_code == 404
        assert resp_a.content == resp_b.content
        assert resp_a.json() == {"detail": "Blob not found"}

    @pytest.mark.asyncio
    async def test_execute_source_path_traversal_returns_404(self) -> None:
        """Source path escaping allowed directories is rejected."""
        svc = MagicMock()
        svc.execute = AsyncMock(side_effect=ValueError("Source path='../../etc/passwd' resolves outside allowed directories"))
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/execute")
            assert resp.status_code == 404
            assert "resolves outside" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_execute_sink_path_traversal_returns_404(self) -> None:
        """Sink path escaping allowed output directories is rejected."""
        svc = MagicMock()
        svc.execute = AsyncMock(side_effect=ValueError("Sink 'out' path='../../../tmp/evil' resolves outside allowed output directories"))
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/execute")
            assert resp.status_code == 404
            assert "resolves outside" in resp.json()["detail"]


class TestWebSocketReconnectTier1Guards:
    """Reconnect terminal-event seeding must crash on Tier 1 audit trail anomalies.

    When a client connects to a terminal run, the handler reconstructs
    a typed event payload from the DB record. Impossible states in the
    DB must raise RuntimeError, not silently degrade.
    """

    @staticmethod
    def _assert_terminal_event_build_raises(status_response: RunStatusResponse, match: str) -> None:
        """Exercise the exact terminal event builder used by reconnect seeding."""
        from elspeth.web.execution.routes import _build_terminal_run_event

        with pytest.raises(RuntimeError, match=match):
            _build_terminal_run_event(status_response)

    @pytest.mark.parametrize(
        ("terminal_status", "rows_processed", "rows_succeeded", "rows_failed"),
        [
            ("completed", 1, 1, 0),
            ("completed_with_failures", 2, 1, 1),
            ("empty", 0, 0, 0),
        ],
    )
    def test_operator_completion_statuses_build_completed_terminal_event(
        self,
        terminal_status: str,
        rows_processed: int,
        rows_succeeded: int,
        rows_failed: int,
    ) -> None:
        """Reconnect/idle replay preserves the widened operator-completion status."""
        from elspeth.web.execution.routes import _build_terminal_run_event

        run_id = uuid4()
        event = _build_terminal_run_event(
            RunStatusResponse(
                run_id=str(run_id),
                status=terminal_status,  # type: ignore[arg-type]
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=rows_processed,
                rows_succeeded=rows_succeeded,
                rows_failed=rows_failed,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id="land-1",
            )
        )

        payload = event.model_dump(mode="json")
        assert payload["event_type"] == "completed"
        assert payload["data"]["status"] == terminal_status
        assert payload["data"]["landscape_run_id"] == "land-1"

    def test_completed_run_without_landscape_run_id_raises(self) -> None:
        """Tier 1 anomaly: completed run with NULL landscape_run_id."""
        run_id = uuid4()
        self._assert_terminal_event_build_raises(
            RunStatusResponse.model_construct(
                run_id=str(run_id),
                status="completed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            ),
            "landscape_run_id",
        )

    def test_failed_run_without_error_raises(self) -> None:
        """Tier 1 anomaly: failed run with NULL error column."""
        run_id = uuid4()
        self._assert_terminal_event_build_raises(
            RunStatusResponse.model_construct(
                run_id=str(run_id),
                status="failed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=5,
                rows_succeeded=0,
                rows_failed=5,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            ),
            "error column NULL",
        )

    def test_terminal_run_without_timestamps_raises(self) -> None:
        """Tier 1 anomaly: terminal run with both timestamps NULL."""
        run_id = uuid4()
        self._assert_terminal_event_build_raises(
            RunStatusResponse.model_construct(
                run_id=str(run_id),
                status="cancelled",
                started_at=None,
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            ),
            "both finished_at and started_at are NULL",
        )

    def test_completed_run_with_overcounted_row_counts_raises(self) -> None:
        """Tier 1 anomaly: row count over-counting in completed run.

        Defence-in-depth: even if a session service bypasses the
        RunStatusResponse validator (e.g., via ``model_construct`` on a
        hot path), the WebSocket seeding handler must still fail with a
        coherent Tier 1 anomaly message rather than an unhandled
        ``pydantic.ValidationError`` that leaves the client disconnected
        without explanation (routes.py close code 1011).

        Under the narrow inequality invariant (elspeth-31d53c7493), only
        over-counting (sum > processed) trips the catch.  Under-counting
        is the legitimate aggregation-pipeline shape and must pass.
        Full DAG-aware balance is tracked in elspeth-cf84eb1b52 (P0).
        """
        run_id = uuid4()
        # Bypass the RunStatusResponse validator to simulate a bad DB read
        # reaching the WebSocket seeding path.  Real callers never do this
        # — the test is proving the catch in routes.py is armed.
        bad_status = RunStatusResponse.model_construct(
            run_id=str(run_id),
            status="completed",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            rows_processed=100,
            rows_succeeded=80,
            rows_failed=20,
            rows_routed_success=10,
            rows_routed_failure=0,
            rows_quarantined=10,  # 80+20+10+10 = 120 > 100 — over-counting Tier 1 bug
            error=None,
            landscape_run_id="lscape-1",
        )
        self._assert_terminal_event_build_raises(bad_status, r"Tier 1 anomaly.*audit trail inconsistent")


class TestRunStatusEndpoint:
    """GET /api/runs/{run_id}"""

    @pytest.mark.asyncio
    async def test_status_returns_200(self) -> None:
        run_id = uuid4()
        svc = MagicMock()
        # Phase 2.2: shape with failures => `completed_with_failures`.
        # The original test assertion just checked the route surfaces the
        # status string; using the right status preserves the route-level
        # check while satisfying the new biconditional.
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="completed_with_failures",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=10,
                rows_succeeded=7,
                rows_failed=1,
                rows_routed_success=1,
                rows_routed_failure=0,
                rows_quarantined=1,
                error=None,
                landscape_run_id="lscape-1",
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "completed_with_failures"
            assert body["rows_processed"] == 10
            assert body["rows_routed_success"] == 1
            assert body["rows_routed_failure"] == 0

    @pytest.mark.asyncio
    async def test_running_status_with_landscape_id_skips_discard_summary_lookup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Running status must not inspect an audit DB that may still be initializing."""
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="running",
                started_at=datetime.now(tz=UTC),
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=str(run_id),
            )
        )

        def fail_if_called(*args: object, **kwargs: object) -> dict[str, DiscardSummary]:
            raise AssertionError("discard summary lookup should not run for non-terminal status")

        monkeypatch.setattr(
            "elspeth.web.execution.discard_summary.load_discard_summaries_for_settings",
            fail_if_called,
        )

        app = _create_test_app(execution_service=svc)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert body["landscape_run_id"] == str(run_id)
        assert body["discard_summary"] is None

    @pytest.mark.asyncio
    async def test_status_returns_404_when_run_disappears_after_ownership_check(self) -> None:
        """TOCTOU: post-verification ValueError must collapse to 404."""
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(side_effect=ValueError("run disappeared"))
        app = _create_test_app(execution_service=svc)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}")
            assert resp.status_code == 404
            assert resp.json() == {"detail": "Run not found"}


class TestCancelEndpoint:
    """POST /api/runs/{run_id}/cancel"""

    @pytest.mark.asyncio
    async def test_cancel_returns_200(self) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.cancel = AsyncMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="cancelled",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=5,
                rows_succeeded=5,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id="lscape-1",
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(f"/api/runs/{run_id}/cancel")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_returns_404_when_run_disappears_after_cancel(self) -> None:
        """TOCTOU: second status read after cancel must not leak a 500."""
        run_id = uuid4()
        svc = MagicMock()
        svc.cancel = AsyncMock()
        svc.get_status = AsyncMock(side_effect=ValueError("run disappeared"))
        app = _create_test_app(execution_service=svc)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(f"/api/runs/{run_id}/cancel")
            assert resp.status_code == 404
            assert resp.json() == {"detail": "Run not found"}


class TestResultsEndpoint:
    """GET /api/runs/{run_id}/results"""

    @pytest.mark.asyncio
    async def test_results_returns_200_for_completed_run(self) -> None:
        run_id = uuid4()
        svc = MagicMock()
        # Phase 2.2: shape with failures => `completed_with_failures`.
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="completed_with_failures",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=10,
                rows_succeeded=7,
                rows_failed=1,
                rows_routed_success=1,
                rows_routed_failure=0,
                rows_quarantined=1,
                error=None,
                landscape_run_id="lscape-1",
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}/results")
            assert resp.status_code == 200
            body = resp.json()
            assert body["rows_processed"] == 10
            assert body["rows_routed_success"] == 1
            assert body["rows_routed_failure"] == 0
            assert body["landscape_run_id"] == "lscape-1"

    @pytest.mark.asyncio
    async def test_results_includes_virtual_discard_summary(self) -> None:
        run_id = uuid4()
        svc = MagicMock()
        # Phase 2.2: shape with failures => `completed_with_failures`.
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="completed_with_failures",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=10,
                rows_succeeded=7,
                rows_failed=1,
                rows_routed_success=1,
                rows_routed_failure=0,
                rows_quarantined=1,
                error=None,
                landscape_run_id="lscape-1",
                discard_summary=DiscardSummary(
                    total=3,
                    validation_errors=1,
                    transform_errors=1,
                    sink_discards=1,
                ),
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}/results")
            assert resp.status_code == 200
            body = resp.json()
            assert body["discard_summary"] == {
                "total": 3,
                "validation_errors": 1,
                "transform_errors": 1,
                "sink_discards": 1,
            }

    @pytest.mark.asyncio
    async def test_results_returns_404_when_run_disappears_after_ownership_check(self) -> None:
        """TOCTOU: post-verification status reread must preserve 404 contract."""
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(side_effect=ValueError("run disappeared"))
        app = _create_test_app(execution_service=svc)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}/results")
            assert resp.status_code == 404
            assert resp.json() == {"detail": "Run not found"}

    @pytest.mark.asyncio
    async def test_results_returns_409_for_running(self) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="running",
                started_at=datetime.now(tz=UTC),
                finished_at=None,
                rows_processed=5,
                rows_succeeded=5,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}/results")
            assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_results_returns_409_for_running_with_landscape_id_without_discard_summary_lookup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="running",
                started_at=datetime.now(tz=UTC),
                finished_at=None,
                rows_processed=5,
                rows_succeeded=5,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=str(run_id),
            )
        )

        def fail_if_called(*args: object, **kwargs: object) -> dict[str, DiscardSummary]:
            raise AssertionError("discard summary lookup should not run before terminal results")

        monkeypatch.setattr(
            "elspeth.web.execution.discard_summary.load_discard_summaries_for_settings",
            fail_if_called,
        )

        app = _create_test_app(execution_service=svc)
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}/results")

        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_results_returns_409_for_pending(self) -> None:
        """Covers the second non-terminal status in RUN_STATUS_NON_TERMINAL_VALUES."""
        run_id = uuid4()
        svc = MagicMock()
        svc.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="pending",
                started_at=None,
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )
        )
        app = _create_test_app(execution_service=svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(f"/api/runs/{run_id}/results")
            assert resp.status_code == 409
            assert "pending" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_results_guard_uses_derived_set(self) -> None:
        """Route guard is now derived from schema Literals, not hardcoded.

        This test pins the contract: the guard rejects exactly every
        non-terminal value in RUN_STATUS_NON_TERMINAL_VALUES, proving the
        route no longer carries an independent copy of the list.
        """
        from elspeth.web.execution.schemas import RUN_STATUS_NON_TERMINAL_VALUES

        for non_terminal in RUN_STATUS_NON_TERMINAL_VALUES:
            run_id = uuid4()
            svc = MagicMock()
            svc.get_status = AsyncMock(
                return_value=RunStatusResponse(
                    run_id=str(run_id),
                    status=non_terminal,  # type: ignore[arg-type]
                    started_at=None,
                    finished_at=None,
                    rows_processed=0,
                    rows_succeeded=0,
                    rows_failed=0,
                    rows_quarantined=0,
                    error=None,
                    landscape_run_id=None,
                )
            )
            app = _create_test_app(execution_service=svc)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.get(f"/api/runs/{run_id}/results")
                assert resp.status_code == 409, f"non-terminal status {non_terminal!r} must produce 409"
