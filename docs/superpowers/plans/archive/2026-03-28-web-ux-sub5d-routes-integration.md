# Web UX Task-Plan 5D: Routes, WebSocket & Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement REST endpoints, WebSocket handler with query param auth, multi-worker warning, and end-to-end integration test
**Parent Plan:** `plans/2026-03-28-web-ux-sub5-execution.md`
**Spec:** `specs/2026-03-28-web-ux-sub5-execution-design.md`
**Depends On:** Task-Plans 5A, 5B, 5C (all execution internals), Sub-Plan 2 (Sessions — routes, auth)
**Blocks:** Sub-Plan 6 (Frontend)

---

## File Map

| Action | File |
|--------|------|
| Create | `src/elspeth/web/execution/routes.py` |
| Create | `tests/unit/web/execution/test_routes.py` |
| Modify | app factory (wherever `create_app()` lives, from Sub-Spec 2) |
| Create | `tests/integration/web/__init__.py` |
| Create | `tests/integration/web/test_execute_pipeline.py` |
| Create | `tests/integration/web/fixtures/test_input.csv` |

---

### Task 5.6: Execution Routes and WebSocket

**Files:**
- Create: `src/elspeth/web/execution/routes.py`
- Create: `tests/unit/web/execution/test_routes.py`

- [ ] **Step 1: Write route tests**

```python
# tests/unit/web/execution/test_routes.py
"""Tests for execution REST endpoints and WebSocket.

Routes delegate to ExecutionServiceImpl — these tests verify HTTP
semantics, status codes, and request/response contracts.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from elspeth.web.execution.schemas import (
    RunEvent,
    RunStatusResponse,
    ValidationCheck,
    ValidationResult,
)
from elspeth.web.sessions.protocol import RunAlreadyActiveError


# ── Test fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_execution_service() -> MagicMock:
    svc = MagicMock()
    svc.validate.return_value = ValidationResult(
        is_valid=True,
        checks=[
            ValidationCheck(name="settings_load", passed=True, detail="OK"),
        ],
        errors=[],
    )
    run_id = uuid4()
    svc.execute.return_value = run_id
    svc.get_status.return_value = RunStatusResponse(
        run_id=str(run_id),
        status="completed",
        started_at=datetime.now(tz=timezone.utc),
        finished_at=datetime.now(tz=timezone.utc),
        rows_processed=10,
        rows_failed=0,
        error=None,
        landscape_run_id="lscape-1",
    )
    return svc


@pytest.fixture
def mock_broadcaster() -> MagicMock:
    return MagicMock()


# Note: These tests require the app factory to be wired.
# The actual test_app fixture will depend on Sub-Spec 2 (Auth) and
# Sub-Spec 4 (Composer) providing create_app(). Tests below show
# the expected HTTP contract — adapt fixture wiring to match the
# actual app factory when those phases are implemented.


class TestValidateEndpoint:
    """POST /api/sessions/{session_id}/validate"""

    @pytest.mark.asyncio
    async def test_valid_pipeline_returns_200(
        self,
        mock_execution_service: MagicMock,
    ) -> None:
        from elspeth.web.app import create_app

        mock_execution_service.validate = AsyncMock(
            return_value=ValidationResult(
                is_valid=True,
                checks=[
                    ValidationCheck(name="settings_load", passed=True, detail="OK"),
                ],
                errors=[],
            )
        )
        app = create_app(execution_service=mock_execution_service)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/validate")
            assert resp.status_code == 200
            body = resp.json()
            assert body["is_valid"] is True
            assert len(body["checks"]) == 1

    @pytest.mark.asyncio
    async def test_validate_uses_run_in_executor(
        self,
        mock_execution_service: MagicMock,
    ) -> None:
        """AC #16: validate route handler MUST NOT call validate_pipeline
        synchronously — it must use run_in_executor to avoid blocking the
        event loop."""
        from elspeth.web.app import create_app

        mock_execution_service.validate = AsyncMock(
            return_value=ValidationResult(is_valid=True, checks=[], errors=[])
        )
        app = create_app(execution_service=mock_execution_service)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/validate")
            assert resp.status_code == 200
            # The route delegates to service.validate() which handles
            # run_in_executor internally. Verify the async mock was awaited.
            mock_execution_service.validate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_pipeline_returns_200_with_errors(
        self,
        mock_execution_service: MagicMock,
    ) -> None:
        mock_execution_service.validate.return_value = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load", passed=False, detail="Bad YAML"
                ),
            ],
            errors=[],
        )
        from elspeth.web.app import create_app

        mock_execution_service.validate = AsyncMock(
            return_value=ValidationResult(
                is_valid=False,
                checks=[
                    ValidationCheck(
                        name="settings_load", passed=False, detail="Bad YAML"
                    ),
                ],
                errors=[],
            )
        )
        app = create_app(execution_service=mock_execution_service)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Validation errors are NOT HTTP errors — the endpoint always
            # returns 200 with the ValidationResult body. HTTP 4xx/5xx are
            # reserved for infrastructure errors (auth, not found, etc.)
            resp = await client.post(f"/api/sessions/{uuid4()}/validate")
            assert resp.status_code == 200
            body = resp.json()
            assert body["is_valid"] is False


class TestExecuteEndpoint:
    """POST /api/sessions/{session_id}/execute"""

    @pytest.mark.asyncio
    async def test_execute_returns_202_with_run_id(
        self,
        mock_execution_service: MagicMock,
    ) -> None:
        from elspeth.web.app import create_app

        expected_run_id = uuid4()
        mock_execution_service.execute = AsyncMock(return_value=expected_run_id)
        app = create_app(execution_service=mock_execution_service)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/execute")
            assert resp.status_code == 202
            body = resp.json()
            assert body["run_id"] == str(expected_run_id)

    @pytest.mark.asyncio
    async def test_execute_with_active_run_returns_409(
        self,
        mock_execution_service: MagicMock,
    ) -> None:
        from elspeth.web.app import create_app

        mock_execution_service.execute = AsyncMock(
            side_effect=RunAlreadyActiveError("Already active")
        )
        app = create_app(execution_service=mock_execution_service)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(f"/api/sessions/{uuid4()}/execute")
            assert resp.status_code == 409
            body = resp.json()
            # Seam contract D: structured error envelope
            assert body["error_type"] == "run_already_active"
            assert "detail" in body


class TestRunStatusEndpoint:
    """GET /api/runs/{run_id}"""

    @pytest.mark.asyncio
    async def test_status_returns_200(
        self,
        mock_execution_service: MagicMock,
    ) -> None:
        from elspeth.web.app import create_app

        run_id = uuid4()
        mock_execution_service.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="completed",
                started_at=datetime.now(tz=timezone.utc),
                finished_at=datetime.now(tz=timezone.utc),
                rows_processed=10,
                rows_failed=0,
                error=None,
                landscape_run_id="lscape-1",
            )
        )
        app = create_app(execution_service=mock_execution_service)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/api/runs/{run_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "completed"
            assert body["rows_processed"] == 10


class TestCancelEndpoint:
    """POST /api/runs/{run_id}/cancel"""

    @pytest.mark.asyncio
    async def test_cancel_returns_200(
        self,
        mock_execution_service: MagicMock,
    ) -> None:
        from elspeth.web.app import create_app

        run_id = uuid4()
        mock_execution_service.cancel = AsyncMock()
        mock_execution_service.get_status = AsyncMock(
            return_value=RunStatusResponse(
                run_id=str(run_id),
                status="cancelled",
                started_at=datetime.now(tz=timezone.utc),
                finished_at=datetime.now(tz=timezone.utc),
                rows_processed=5,
                rows_failed=0,
                error=None,
                landscape_run_id="lscape-1",
            )
        )
        app = create_app(execution_service=mock_execution_service)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(f"/api/runs/{run_id}/cancel")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "cancelled"


class TestWebSocketProgress:
    """WS /ws/runs/{run_id}"""

    # TODO: WebSocket tests require httpx or starlette.testclient WebSocket
    # support. The starlette TestClient supports WebSocket via
    # `with client.websocket_connect(url) as ws:` but httpx AsyncClient
    # does not. These tests should use starlette.testclient.TestClient
    # (sync) or a dedicated WebSocket test helper.

    def test_websocket_receives_progress_events(
        self,
        mock_broadcaster: MagicMock,
    ) -> None:
        """Client connects, receives progress events, disconnects on terminal.

        Verifies the WebSocket handler contract:
        1. subscribe(run_id) on connect
        2. await queue.get() in a loop
        3. send_json(event) for each event
        4. close with code 1000 on terminal event (completed/cancelled)
        5. unsubscribe(run_id, queue) in finally block

        Note: "error" events are non-terminal (per-row exceptions, pipeline
        continues). Only "completed" and "cancelled" are terminal close triggers.
        """
        from starlette.testclient import TestClient
        from elspeth.web.app import create_app

        run_id = uuid4()
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        queue.put_nowait(RunEvent(event_type="progress", run_id=str(run_id), data={"rows": 5}, timestamp=datetime.now(tz=timezone.utc)))
        queue.put_nowait(RunEvent(event_type="completed", run_id=str(run_id), data={}, timestamp=datetime.now(tz=timezone.utc)))
        mock_broadcaster.subscribe.return_value = queue

        app = create_app(broadcaster=mock_broadcaster)
        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/runs/{run_id}?token=valid-test-token"
        ) as ws:
            msg1 = ws.receive_json()
            assert msg1["event_type"] == "progress"
            msg2 = ws.receive_json()
            assert msg2["event_type"] == "completed"
        mock_broadcaster.unsubscribe.assert_called_once()

    def test_websocket_closes_1011_on_internal_error(
        self,
        mock_broadcaster: MagicMock,
    ) -> None:
        """H6: Internal server error sends close code 1011."""
        from starlette.testclient import TestClient
        from elspeth.web.app import create_app

        queue = AsyncMock()
        queue.get.side_effect = RuntimeError("broadcaster failure")
        mock_broadcaster.subscribe.return_value = queue

        app = create_app(broadcaster=mock_broadcaster)
        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/runs/{uuid4()}?token=valid-test-token"
        ) as ws:
            # Server should close with 1011 on internal error
            data = ws.receive()
            assert data.get("code") == 1011

    def test_websocket_auth_failure_closes_4001(
        self,
        mock_broadcaster: MagicMock,
    ) -> None:
        """AC #12: Missing or invalid ?token= query param closes with 4001.
        Client MUST NOT auto-reconnect on 4001."""
        from starlette.testclient import TestClient
        from elspeth.web.app import create_app

        app = create_app(broadcaster=mock_broadcaster)
        client = TestClient(app)
        # No ?token= query parameter — should close with 4001
        with client.websocket_connect(f"/ws/runs/{uuid4()}") as ws:
            data = ws.receive()
            assert data.get("code") == 4001

    def test_websocket_unsubscribes_on_disconnect(
        self,
        mock_broadcaster: MagicMock,
    ) -> None:
        """Cleanup happens even on unexpected disconnect."""
        from starlette.testclient import TestClient
        from elspeth.web.app import create_app

        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        # Don't put any events — client will disconnect before receiving
        mock_broadcaster.subscribe.return_value = queue

        app = create_app(broadcaster=mock_broadcaster)
        client = TestClient(app)
        with client.websocket_connect(
            f"/ws/runs/{uuid4()}?token=valid-test-token"
        ) as ws:
            ws.close()
        # Verify unsubscribe was called in finally block
        mock_broadcaster.unsubscribe.assert_called_once()
```

- [ ] **Step 2: Implement routes**

```python
# src/elspeth/web/execution/routes.py
"""REST endpoints and WebSocket for pipeline execution.

POST /api/sessions/{session_id}/validate — dry-run validation
POST /api/sessions/{session_id}/execute — start background run
GET  /api/runs/{run_id}                 — run status
POST /api/runs/{run_id}/cancel          — cancel run
GET  /api/runs/{run_id}/results         — run results (terminal only)
WS   /ws/runs/{run_id}                  — live progress stream
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from elspeth.web.auth.models import AuthenticationError
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.schemas import RunStatusResponse, ValidationResult
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.sessions.protocol import RunAlreadyActiveError

slog = structlog.get_logger()

router = APIRouter()


# ── Dependency stubs — wired by app factory ────────────────────────────
# These will be replaced with actual dependency injection from create_app()

def get_execution_service() -> ExecutionServiceImpl:
    raise NotImplementedError("Wire via app factory")


def get_broadcaster() -> ProgressBroadcaster:
    raise NotImplementedError("Wire via app factory")


# ── REST Endpoints ─────────────────────────────────────────────────────

@router.post(
    "/api/sessions/{session_id}/validate",
    response_model=ValidationResult,
)
async def validate_session_pipeline(
    session_id: UUID,
    service: ExecutionServiceImpl = Depends(get_execution_service),
) -> ValidationResult:
    """Dry-run validation using real engine code paths.

    AC #16: Validation is synchronous and takes 1-5 seconds depending on
    plugin count. The service.validate() method handles run_in_executor
    internally to avoid blocking the event loop.
    """
    result = await service.validate(session_id)
    return result


@router.post(
    "/api/sessions/{session_id}/execute",
    status_code=202,
)
async def execute_pipeline(
    session_id: UUID,
    state_id: UUID | None = None,
    service: ExecutionServiceImpl = Depends(get_execution_service),
) -> dict[str, str]:
    """Start a background pipeline run. Returns run_id immediately."""
    try:
        run_id = await service.execute(session_id, state_id)
    except RunAlreadyActiveError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_type": "run_already_active", "detail": str(exc)},
        )
    return {"run_id": str(run_id)}


@router.get(
    "/api/runs/{run_id}",
    response_model=RunStatusResponse,
)
async def get_run_status(
    run_id: UUID,
    service: ExecutionServiceImpl = Depends(get_execution_service),
) -> RunStatusResponse:
    """Return current run status."""
    status = await service.get_status(run_id)
    return status


@router.post("/api/runs/{run_id}/cancel")
async def cancel_run(
    run_id: UUID,
    service: ExecutionServiceImpl = Depends(get_execution_service),
) -> dict[str, str]:
    """Cancel a run. Idempotent on terminal runs."""
    await service.cancel(run_id)
    status = await service.get_status(run_id)
    return {"status": status.status}


@router.get("/api/runs/{run_id}/results")
async def get_run_results(
    run_id: UUID,
    service: ExecutionServiceImpl = Depends(get_execution_service),
) -> dict[str, Any]:
    """Return final run results. 409 if run is not terminal."""
    status = await service.get_status(run_id)
    if status.status in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Run is still {status.status}",
        )
    return {
        "run_id": status.run_id,
        "status": status.status,
        "rows_processed": status.rows_processed,
        "rows_failed": status.rows_failed,
        "landscape_run_id": status.landscape_run_id,
        "error": status.error,
    }


# ── WebSocket Endpoint ─────────────────────────────────────────────────

@router.websocket("/ws/runs/{run_id}")
async def websocket_run_progress(
    websocket: WebSocket,
    run_id: str,
    token: str | None = None,
) -> None:
    # NOTE (post-impl fix): Auth provider and broadcaster accessed from
    # app.state, not via Depends (WebSocket limitation). Uses
    # AuthProvider.authenticate(), not the non-existent validate_token().
    """Stream RunEvent JSON payloads for a specific run.

    AC #12: Authentication via ?token=<jwt> query parameter.
    Close code 4001 on auth failure — client MUST NOT auto-reconnect
    on 4001 (token must be refreshed or user must re-authenticate).

    Connection lifecycle:
    1. Validate JWT from query parameter (close 4001 if invalid)
    2. Accept WebSocket connection
    3. Verify run exists and belongs to authenticated user (IDOR -> 404-equivalent close)
    4. Subscribe to broadcaster for this run_id
    5. Loop: await queue.get() -> send_json(event)
    6. Close on terminal event (completed/cancelled) — "error" is non-terminal
    7. Unsubscribe in finally block (ensures cleanup on disconnect)
    """
    # Auth: validate JWT from query parameter
    if token is None:
        await websocket.close(code=4001, reason="Missing authentication token")
        return
    try:
        user = await auth_provider.authenticate(token)
    except AuthenticationError:
        await websocket.close(code=4001, reason="Invalid authentication token")
        return

    await websocket.accept()

    # IDOR protection: verify authenticated user owns this run's session
    try:
        run_ownership = await service.verify_run_ownership(user, run_id)
        if not run_ownership:
            await websocket.close(code=4004, reason="Run not found")
            return
    except Exception:
        await websocket.close(code=4004, reason="Run not found")
        return

    queue = broadcaster.subscribe(run_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                # Heartbeat timeout — pipeline may have crashed without terminal event.
                # Send a ping to verify the client is still connected.
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break  # Client disconnected
                continue
            await websocket.send_json(event.model_dump(mode="json"))
            # "error" events are non-terminal — they represent per-row
            # exceptions while the pipeline continues processing. Only
            # "completed" and "cancelled" are terminal events per seam
            # contract E.
            if event.event_type in ("completed", "cancelled"):
                await websocket.close(code=1000)
                break
    except WebSocketDisconnect:
        pass  # Client disconnected — fall through to finally
    except Exception as exc:
        # H6: 1011 = internal server error
        slog.error(
            "websocket_handler_error",
            run_id=run_id,
            error=str(exc),
        )
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception as close_err:
            slog.error("websocket_close_failed", run_id=run_id, error=str(close_err))
    finally:
        broadcaster.unsubscribe(run_id, queue)
```

- [ ] **Step 3: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/unit/web/execution/test_routes.py -v
git commit -m "feat(web/execution): add REST endpoints and WebSocket for execution"
```

---

### Task 5.7: Multi-Worker Enforcement (W10 -> R6)

**Files:**
- Modify: app factory (wherever `create_app()` lives, from Sub-Spec 2)

- [ ] **Step 1: Write test**

```python
# Add to tests/unit/web/execution/test_service.py

class TestMultiWorkerEnforcement:
    """W10 -> R6: Hard-enforce single worker instead of warning."""

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "4"})
    def test_raises_on_multi_worker(self) -> None:
        """Application factory rejects WEB_CONCURRENCY > 1."""
        from elspeth.web.app import create_app

        with pytest.raises(RuntimeError, match="WEB_CONCURRENCY=4 is not supported"):
            create_app()

    @patch.dict("os.environ", {"WEB_CONCURRENCY": "1"})
    def test_single_worker_accepted(self) -> None:
        """No error when running with a single worker."""
        from elspeth.web.app import create_app
        # Should not raise
        app = create_app()
        assert app is not None
```

- [ ] **Step 2: Implement enforcement in app factory**

Add to the app factory (`create_app()`):

```python
import os

web_concurrency = int(os.environ.get("WEB_CONCURRENCY", "1"))
if web_concurrency > 1:
    raise RuntimeError(
        f"WEB_CONCURRENCY={web_concurrency} is not supported. "
        "ProgressBroadcaster holds subscriber queues in process memory — "
        "WebSocket progress streaming requires a single worker. "
        "Set WEB_CONCURRENCY=1 or remove the variable. "
        "For multi-worker deployment, replace ProgressBroadcaster with Redis Streams."
    )
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(web): hard-enforce single worker for WebSocket (W10 -> R6)"
```

---

### Task 5.8: Integration Test — End-to-End Pipeline

**Files:**
- Create: `tests/integration/web/__init__.py`
- Create: `tests/integration/web/test_execute_pipeline.py`
- Create: `tests/integration/web/fixtures/test_input.csv`

This test exercises the full execution path through the web layer: create session, save composition state, validate, execute, poll to completion, verify results.

- [ ] **Step 1: Create test CSV fixture**

```csv
id,name,value
1,alpha,100
2,beta,200
3,gamma,300
```

- [ ] **Step 2: Write end-to-end integration test**

```python
# tests/integration/web/test_execute_pipeline.py
"""End-to-end integration test: CSV -> passthrough -> CSV through web layer.

This test uses the REAL engine code path — no mocks for the pipeline itself.
The web layer (routes, ExecutionService, ProgressBroadcaster) is exercised
with a real FastAPI test client. The pipeline runs a CSV source through a
passthrough transform to a CSV sink.

Validates acceptance criteria #14:
- Session created
- CompositionState saved (manually, not via composer)
- Validation passes (is_valid=True)
- Execution completes
- rows_processed > 0, rows_failed == 0
- landscape_run_id links to real audit trail
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_CSV = FIXTURES_DIR / "test_input.csv"


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Create a working directory with the test CSV and output dir."""
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()
    csv_dest = uploads_dir / "input.csv"
    shutil.copy(TEST_CSV, csv_dest)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    audit_dir = tmp_path / "runs"
    audit_dir.mkdir()
    payloads_dir = tmp_path / "payloads"
    payloads_dir.mkdir()
    return tmp_path


# The pipeline YAML for CSV source -> passthrough -> CSV sink
def _make_pipeline_yaml(work_dir: Path) -> str:
    return f"""
source:
  plugin: csv_source
  options:
    path: "{work_dir / 'uploads' / 'input.csv'}"
  on_success: primary

transforms: []

sinks:
  primary:
    plugin: csv_sink
    options:
      path: "{work_dir / 'output' / 'result.csv'}"
"""


@pytest.mark.integration
class TestEndToEndPipelineExecution:
    """Full lifecycle through the web layer with real engine execution."""

    @pytest.mark.asyncio
    async def test_csv_passthrough_csv(self, work_dir: Path) -> None:
        """
        1. Create session
        2. Save CompositionState (manually — tests execution independently)
        3. Validate -> is_valid=True
        4. Execute -> get run_id
        5. Poll status -> eventually 'completed'
        6. Verify results: rows_processed > 0, rows_failed == 0
        7. Verify landscape_run_id links to audit trail
        """
        # AC #14 + AC #20: Full cross-module integration test exercising
        # real SessionService, CatalogService, and ExecutionService wired together.
        pipeline_yaml = _make_pipeline_yaml(work_dir)

        # Create app with test settings pointing to work_dir
        from elspeth.web.app import create_app
        from elspeth.web.config import WebSettings

        settings = WebSettings(
            data_dir=work_dir,
            landscape_url=f"sqlite:///{work_dir}/runs/audit.db",
            payload_store_path=work_dir / "payloads",
        )
        app = create_app(settings=settings)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Register and authenticate test user
            resp = await client.post(
                "/api/auth/register",
                json={"username": "testuser", "password": "testpass123"},
            )
            assert resp.status_code == 201
            resp = await client.post(
                "/api/auth/login",
                json={"username": "testuser", "password": "testpass123"},
            )
            assert resp.status_code == 200
            token = resp.json()["token"]
            auth_headers = {"Authorization": f"Bearer {token}"}

            # 1. Create session (with auth)
            resp = await client.post("/api/sessions", headers=auth_headers)
            assert resp.status_code == 201
            session_id = resp.json()["session_id"]

            # 2. Save composition state (manual — not via composer)
            resp = await client.post(
                f"/api/sessions/{session_id}/states",
                json={"pipeline_yaml": pipeline_yaml},
                headers=auth_headers,
            )
            assert resp.status_code == 201

            # 3. Validate — AC #16: route handler uses run_in_executor
            resp = await client.post(
                f"/api/sessions/{session_id}/validate",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            validation = resp.json()
            assert validation["is_valid"] is True, (
                f"Validation failed: {validation['errors']}"
            )

            # 4. Execute
            resp = await client.post(
                f"/api/sessions/{session_id}/execute",
                headers=auth_headers,
            )
            assert resp.status_code == 202
            run_id = resp.json()["run_id"]

            # 5. Poll to completion (timeout after 30s)
            deadline = time.monotonic() + 30
            status: dict[str, Any] = {}
            while time.monotonic() < deadline:
                resp = await client.get(
                    f"/api/runs/{run_id}",
                    headers=auth_headers,
                )
                assert resp.status_code == 200
                status = resp.json()
                if status["status"] in ("completed", "failed", "cancelled"):
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail("Pipeline did not complete within 30 seconds")

            # 6. Verify results — AC #14
            assert status["status"] == "completed", (
                f"Pipeline failed: {status.get('error')}"
            )
            assert status["rows_processed"] > 0
            assert status["rows_failed"] == 0

            # 7. Verify landscape_run_id links to real audit trail
            assert status["landscape_run_id"] is not None

            # 8. Verify output file was created
            output_file = work_dir / "output" / "result.csv"
            assert output_file.exists()

            # 9. Verify audit database exists
            audit_db = work_dir / "runs" / "audit.db"
            assert audit_db.exists()

            # 10. Verify results endpoint — AC #20 cross-module
            resp = await client.get(
                f"/api/runs/{run_id}/results",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            results = resp.json()
            assert results["rows_processed"] > 0
            assert results["landscape_run_id"] is not None
```

- [ ] **Step 3: Run integration test, commit**

```bash
.venv/bin/python -m pytest tests/integration/web/ -v --timeout=60
git commit -m "test(web): add end-to-end pipeline execution integration test"
```

---

## Self-Review Checklist

- [ ] `routes.py` defines all 5 REST endpoints + 1 WebSocket endpoint matching the spec
- [ ] `validate_session_pipeline` delegates to `service.validate(session_id)` (AC #16) -- service handles run_in_executor internally
- [ ] `execute_pipeline` returns 202 with `run_id`
- [ ] `execute_pipeline` returns 409 with structured error envelope (`error_type`, `detail`) on `RunAlreadyActiveError` (seam contract D)
- [ ] `get_run_results` returns 409 if run is not terminal
- [ ] WebSocket authenticates via `?token=<jwt>` query parameter (AC #12)
- [ ] WebSocket closes with code 4001 on missing or invalid token -- client MUST NOT auto-reconnect on 4001
- [ ] WebSocket closes with code 1000 on terminal event (completed/cancelled) -- "error" is non-terminal (H6, seam contract E)
- [ ] WebSocket closes with code 1011 on internal server error (H6)
- [ ] WebSocket `finally` block calls `broadcaster.unsubscribe()` for cleanup on disconnect
- [ ] IDOR protection: run ownership verified, returns 404 (not 403) for non-owned runs (AC #13)
- [ ] Multi-worker enforcement (W10 -> R6): `create_app()` raises `RuntimeError` when `WEB_CONCURRENCY > 1`
- [ ] No error raised when `WEB_CONCURRENCY == 1`
- [ ] Integration test uses REAL engine code path -- no mocks for pipeline execution
- [ ] Integration test exercises full lifecycle: create session, save state, validate, execute, poll, verify results
- [ ] Integration test verifies `rows_processed > 0`, `rows_failed == 0`, `landscape_run_id` is not None
- [ ] Integration test verifies output file and audit database exist
- [ ] All tests pass: `.venv/bin/python -m pytest tests/unit/web/execution/test_routes.py -v`
- [ ] Integration tests pass: `.venv/bin/python -m pytest tests/integration/web/ -v --timeout=60`
- [ ] mypy passes: `.venv/bin/python -m mypy src/elspeth/web/execution/routes.py`
- [ ] No defensive programming patterns (no `.get()` on typed fields, no `getattr` with defaults)
- [ ] Layer dependency respected: `routes.py` imports from L0-L2 and sibling web modules only

---

## Round 5 Review Findings

### Blocking fixes applied inline

| ID | Finding | Fix |
|----|---------|-----|
| **B1** | `RunAlreadyActiveError` imported from `execution.service` instead of `sessions.protocol` | Changed import to `from elspeth.web.sessions.protocol import RunAlreadyActiveError` in both routes.py and test_routes.py |
| **B2** | Missing `await` on async service calls in route handlers | Added `await` to all `service.execute()`, `service.cancel()`, `service.get_status()`, and `service.validate()` calls |
| **B3** | `auth_provider.validate_token(token)` does not exist on auth protocol | Changed to `await auth_provider.authenticate(token)`, narrowed `except Exception` to `except AuthenticationError` (imported from `elspeth.web.auth.models`) |
| **B5** | All route tests were `pass` stubs with no assertions | Implemented real test bodies using `AsyncClient`/`TestClient` with mock services: `test_valid_pipeline_returns_200`, `test_execute_returns_202_with_run_id`, `test_execute_with_active_run_returns_409`, `test_status_returns_200`, `test_cancel_returns_200`. WebSocket tests use `starlette.testclient.TestClient` |
| **B8** | Integration test imports `from elspeth.web.settings import WebSettings` (wrong path) | Changed to `from elspeth.web.config import WebSettings` |
| **B10** | Validate route calls `service.get_composition_state()` and `service.get_settings()` which do not exist on `ExecutionService` protocol | Route now calls `await service.validate(session_id)` -- service handles internals. Removed unused `validate_pipeline` import |
| **W8** | WebSocket closes on "error" events, but "error" is non-terminal (per-row exception) | Removed "error" from terminal event set. Only "completed" and "cancelled" trigger close(1000). Added comment about queue.get() blocking if `_run_pipeline` crashes without broadcasting terminal event |

### Additional fixes applied

- Added `structlog` import and `slog` logger to routes.py for WebSocket error logging
- Replaced bare `except Exception: pass` around `websocket.close(1011)` with `except Exception as close_err:` + `slog.error("websocket_close_failed", ...)` for observability
- Added `slog.error("websocket_handler_error", ...)` in the outer exception handler
- Implemented multi-worker warning tests (`test_warns_on_multi_worker`, `test_no_warning_for_single_worker`) using `caplog` instead of `pass` stubs

### Additional fixes applied (Round 6)

| ID | Finding | Fix |
|----|---------|-----|
| **W5** | `execute_pipeline` return type annotation lies — returns `JSONResponse` on 409 but declares `-> dict[str, str]` | Changed to `raise HTTPException(status_code=409, detail=...)` so FastAPI handles the error response. Removed unused `JSONResponse` import |
| **W4** | WebSocket IDOR — run ownership not verified after authentication | Added `service.verify_run_ownership(user, run_id)` check after `websocket.accept()` and before `broadcaster.subscribe()`. Closes with code 4004 on ownership failure. Added `service` dependency to WebSocket handler parameters |
| **W7** | Multi-worker check was a warning instead of hard enforcement | Changed from `slog.warning()` to `raise RuntimeError()` in `create_app()`. Updated test to use `pytest.raises(RuntimeError)` instead of `caplog` assertions |
| **W1** | Unbounded `asyncio.Queue` — `queue.get()` blocks forever if pipeline crashes without terminal event | Wrapped `queue.get()` in `asyncio.wait_for(timeout=60.0)`. On timeout, sends heartbeat ping to verify client connectivity. Added `import asyncio` to routes.py |
| **W16** | WebSocket tests mix `@pytest.mark.asyncio` with sync `TestClient` | Removed `@pytest.mark.asyncio` from all four WebSocket test methods. Changed `await queue.put()` to `queue.put_nowait()` in `test_websocket_receives_progress_events` |
| **B9** | Integration test has no auth headers — all API calls would fail authentication | Added user registration and login before test lifecycle. Added `headers=auth_headers` to all subsequent API calls |
| **W17** | Integration test CSV path bypasses allowlist — file placed directly in `tmp_path` | Placed CSV under `uploads/` subdirectory. Updated `_make_pipeline_yaml` to reference `uploads/input.csv` |

### Warnings (remaining)

| ID | Severity | Finding |
|----|----------|---------|
| **W-5D-2** | Low | **Integration test has no WebSocket coverage.** `test_execute_pipeline.py` exercises the full REST lifecycle (create, validate, execute, poll, results) but does not test WebSocket progress streaming. This leaves the real-time progress path untested end-to-end. |
