# Complete Telemetry Wiring Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure every plugin that makes external calls emits `ExternalCallCompleted` telemetry events.

**Architecture:** Two-phase approach: (1) Add telemetry support to `AuditedHTTPClient`, (2) Wire all plugins that use it. Verify all LLM plugins are already wired. Add comprehensive integration test that validates ALL external-call plugins emit telemetry.

**Tech Stack:** Python, pytest, structlog telemetry events

---

## Implementation Summary

- `AuditedHTTPClient` now emits `ExternalCallCompleted` via `telemetry_emit` (`src/elspeth/plugins/clients/http.py`).
- HTTP-based plugins pass telemetry callbacks through client construction (OpenRouter, Content Safety, Prompt Shield) (`src/elspeth/plugins/llm/openrouter.py`, `src/elspeth/plugins/llm/openrouter_multi_query.py`, `src/elspeth/plugins/transforms/azure/content_safety.py`, `src/elspeth/plugins/transforms/azure/prompt_shield.py`).
- Telemetry emission for HTTP and LLM clients is validated in tests (`tests/plugins/clients/test_http_telemetry.py`, `tests/plugins/clients/test_llm_telemetry.py`).

## Current State Summary

| Plugin | External Call Type | Client | Telemetry Status |
|--------|-------------------|--------|------------------|
| `AzureLLMTransform` | LLM | `AuditedLLMClient` | ✅ WIRED |
| `AzureMultiQueryLLMTransform` | LLM | `AuditedLLMClient` | ✅ WIRED |
| `OpenRouterLLMTransform` | HTTP | `AuditedHTTPClient` | ❌ NOT WIRED |
| `OpenRouterMultiQueryLLMTransform` | HTTP | `AuditedHTTPClient` | ❌ NOT WIRED |
| `AzureContentSafety` | HTTP | `AuditedHTTPClient` | ❌ NOT WIRED |
| `AzurePromptShield` | HTTP | `AuditedHTTPClient` | ❌ NOT WIRED |

**Root cause:** `AuditedHTTPClient` does not support telemetry (no `run_id` or `telemetry_emit` parameters).

---

## Task 1: Add Telemetry Support to AuditedHTTPClient

**Files:**
- Modify: `src/elspeth/plugins/clients/http.py`
- Create: `tests/plugins/clients/test_http_telemetry.py`

### Step 1.1: Write failing test for successful HTTP call telemetry

**File:** `tests/plugins/clients/test_http_telemetry.py`

```python
# tests/plugins/clients/test_http_telemetry.py
"""Tests for AuditedHTTPClient telemetry integration."""

import itertools
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.telemetry.events import ExternalCallCompleted


class TestHTTPClientTelemetry:
    """Tests for telemetry emission from AuditedHTTPClient."""

    def _create_mock_recorder(self) -> MagicMock:
        """Create a mock LandscapeRecorder that returns recorded calls."""
        recorder = MagicMock()
        counter = itertools.count()
        recorder.allocate_call_index.side_effect = lambda _: next(counter)

        # record_call returns a Call object with hashes
        recorded_call = MagicMock()
        recorded_call.request_hash = "req_hash_123"
        recorded_call.response_hash = "resp_hash_456"
        recorder.record_call.return_value = recorded_call

        return recorder

    def test_successful_post_emits_telemetry(self) -> None:
        """Successful HTTP POST emits ExternalCallCompleted event."""
        recorder = self._create_mock_recorder()

        # Track emitted events
        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
            base_url="https://api.example.com",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'

        with patch.object(client._client, "post", return_value=mock_response):
            response = client.post("/endpoint", json={"input": "test"})

        # Verify response
        assert response.status_code == 200

        # Verify telemetry event
        assert len(emitted_events) == 1
        event = emitted_events[0]

        assert isinstance(event, ExternalCallCompleted)
        assert event.run_id == "run_abc"
        assert event.state_id == "state_123"
        assert event.call_type == CallType.HTTP
        assert event.provider == "http"
        assert event.status == CallStatus.SUCCESS
        assert event.latency_ms > 0
        assert event.request_hash == "req_hash_123"
        assert event.response_hash == "resp_hash_456"
        assert event.token_usage is None  # Not applicable for HTTP
        assert isinstance(event.timestamp, datetime)
```

### Step 1.2: Run test to verify it fails

Run: `.venv/bin/python -m pytest tests/plugins/clients/test_http_telemetry.py::TestHTTPClientTelemetry::test_successful_post_emits_telemetry -v`

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'run_id'`

### Step 1.3: Add imports to http.py

**File:** `src/elspeth/plugins/clients/http.py`

Add to imports section (around line 15):

```python
from collections.abc import Callable
from datetime import UTC, datetime

from elspeth.telemetry.events import ExternalCallCompleted
```

### Step 1.4: Update __init__ signature

**File:** `src/elspeth/plugins/clients/http.py`

Replace the `__init__` method signature (lines 46-67) with:

```python
def __init__(
    self,
    recorder: LandscapeRecorder,
    state_id: str,
    *,
    timeout: float = 30.0,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    run_id: str | None = None,
    telemetry_emit: Callable[[ExternalCallCompleted], None] | None = None,
) -> None:
    """Initialize an audited HTTP client.

    Args:
        recorder: Landscape recorder for audit trail
        state_id: Node state ID for audit attribution
        timeout: Request timeout in seconds (default: 30.0)
        base_url: Optional base URL for all requests
        headers: Optional default headers for all requests
        run_id: Pipeline run ID for telemetry (optional)
        telemetry_emit: Callback to emit telemetry events (optional)
    """
    self._recorder = recorder
    self._state_id = state_id
    self._run_id = run_id
    self._telemetry_emit = telemetry_emit
    self._client = httpx.Client(
        timeout=timeout,
        base_url=base_url or "",
        headers=headers or {},
    )
```

### Step 1.5: Add _emit_telemetry method

**File:** `src/elspeth/plugins/clients/http.py`

Add this method after `__init__` (around line 85):

```python
def _emit_telemetry(
    self,
    *,
    status: CallStatus,
    latency_ms: float,
    request_hash: str,
    response_hash: str | None,
) -> None:
    """Emit telemetry event if telemetry is configured.

    Telemetry is only emitted when both run_id and telemetry_emit are set.
    This method is called AFTER Landscape recording succeeds.

    Args:
        status: Call status (SUCCESS or ERROR)
        latency_ms: Call latency in milliseconds
        request_hash: Hash of the request (from Landscape recording)
        response_hash: Hash of the response (None on error)
    """
    if self._telemetry_emit is None or self._run_id is None:
        return

    event = ExternalCallCompleted(
        timestamp=datetime.now(UTC),
        run_id=self._run_id,
        state_id=self._state_id,
        call_type=CallType.HTTP,
        provider="http",
        status=status,
        latency_ms=latency_ms,
        request_hash=request_hash,
        response_hash=response_hash,
        token_usage=None,  # Not applicable for HTTP
    )
    self._telemetry_emit(event)
```

### Step 1.6: Update post() method - success path

**File:** `src/elspeth/plugins/clients/http.py`

Find the success path in `post()` where `self._recorder.record_call()` is called. After it, capture the return value and emit telemetry.

Replace the success recording block (around lines 280-295) with:

```python
# Record to Landscape FIRST (legal record)
recorded_call = self._recorder.record_call(
    state_id=self._state_id,
    call_index=call_index,
    call_type=CallType.HTTP,
    status=call_status,
    request_data=request_data,
    response_data=response_data,
    latency_ms=latency_ms,
)

# Emit telemetry AFTER Landscape recording succeeds
self._emit_telemetry(
    status=call_status,
    latency_ms=latency_ms,
    request_hash=recorded_call.request_hash,
    response_hash=recorded_call.response_hash,
)

return response
```

### Step 1.7: Update post() method - error path

**File:** `src/elspeth/plugins/clients/http.py`

Find the error path in `post()` where `self._recorder.record_call()` is called in the except block. Update similarly:

```python
# Record error to Landscape FIRST
recorded_call = self._recorder.record_call(
    state_id=self._state_id,
    call_index=call_index,
    call_type=CallType.HTTP,
    status=CallStatus.ERROR,
    request_data=request_data,
    response_data={"error": str(e), "error_type": type(e).__name__},
    latency_ms=latency_ms,
)

# Emit telemetry AFTER Landscape recording succeeds
self._emit_telemetry(
    status=CallStatus.ERROR,
    latency_ms=latency_ms,
    request_hash=recorded_call.request_hash,
    response_hash=None,
)

raise
```

### Step 1.8: Run test to verify it passes

Run: `.venv/bin/python -m pytest tests/plugins/clients/test_http_telemetry.py::TestHTTPClientTelemetry::test_successful_post_emits_telemetry -v`

Expected: PASS

### Step 1.9: Add remaining telemetry tests

**File:** `tests/plugins/clients/test_http_telemetry.py`

Add these test methods to the `TestHTTPClientTelemetry` class:

```python
def test_failed_post_emits_telemetry_with_error_status(self) -> None:
    """Failed HTTP POST emits ExternalCallCompleted with ERROR status."""
    recorder = self._create_mock_recorder()

    emitted_events: list[ExternalCallCompleted] = []

    def telemetry_emit(event: ExternalCallCompleted) -> None:
        emitted_events.append(event)

    client = AuditedHTTPClient(
        recorder=recorder,
        state_id="state_123",
        base_url="https://api.example.com",
        run_id="run_abc",
        telemetry_emit=telemetry_emit,
    )

    with patch.object(
        client._client, "post", side_effect=httpx.ConnectError("Connection failed")
    ):
        with pytest.raises(httpx.ConnectError):
            client.post("/endpoint", json={"input": "test"})

    # Verify telemetry event
    assert len(emitted_events) == 1
    event = emitted_events[0]

    assert event.run_id == "run_abc"
    assert event.state_id == "state_123"
    assert event.call_type == CallType.HTTP
    assert event.status == CallStatus.ERROR
    assert event.latency_ms >= 0
    assert event.response_hash is None  # No response on error


def test_no_telemetry_when_callback_is_none(self) -> None:
    """No telemetry emitted when telemetry_emit callback is None."""
    recorder = self._create_mock_recorder()

    # No telemetry callback provided
    client = AuditedHTTPClient(
        recorder=recorder,
        state_id="state_123",
        base_url="https://api.example.com",
        run_id="run_abc",
        telemetry_emit=None,  # Explicitly None
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_response.text = '{"result": "success"}'

    with patch.object(client._client, "post", return_value=mock_response):
        response = client.post("/endpoint", json={"input": "test"})

    # Call succeeds without error (no exception from missing callback)
    assert response.status_code == 200
    # Audit trail is still recorded
    recorder.record_call.assert_called_once()


def test_no_telemetry_when_run_id_is_none(self) -> None:
    """No telemetry emitted when run_id is None."""
    recorder = self._create_mock_recorder()

    emitted_events: list[ExternalCallCompleted] = []

    def telemetry_emit(event: ExternalCallCompleted) -> None:
        emitted_events.append(event)

    # run_id is None
    client = AuditedHTTPClient(
        recorder=recorder,
        state_id="state_123",
        base_url="https://api.example.com",
        run_id=None,  # No run_id
        telemetry_emit=telemetry_emit,
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_response.text = '{"result": "success"}'

    with patch.object(client._client, "post", return_value=mock_response):
        response = client.post("/endpoint", json={"input": "test"})

    # Call succeeds
    assert response.status_code == 200
    # No telemetry emitted (run_id is required)
    assert len(emitted_events) == 0


def test_telemetry_emitted_after_landscape_recording(self) -> None:
    """Telemetry is emitted AFTER Landscape recording succeeds."""
    recorder = self._create_mock_recorder()

    call_order: list[str] = []

    def mock_record_call(**kwargs):
        call_order.append("landscape")
        recorded_call = MagicMock()
        recorded_call.request_hash = "req_hash"
        recorded_call.response_hash = "resp_hash"
        return recorded_call

    recorder.record_call.side_effect = mock_record_call

    def telemetry_emit(event: ExternalCallCompleted) -> None:
        call_order.append("telemetry")

    client = AuditedHTTPClient(
        recorder=recorder,
        state_id="state_123",
        base_url="https://api.example.com",
        run_id="run_abc",
        telemetry_emit=telemetry_emit,
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_response.text = '{"result": "success"}'

    with patch.object(client._client, "post", return_value=mock_response):
        client.post("/endpoint", json={"input": "test"})

    # Verify order: Landscape first, then telemetry
    assert call_order == ["landscape", "telemetry"]


def test_no_telemetry_when_landscape_recording_fails(self) -> None:
    """Telemetry is NOT emitted if Landscape recording fails.

    This is a critical invariant: Landscape is the legal record.
    If audit recording fails, telemetry should NOT be emitted because
    the event was never properly recorded.
    """
    recorder = self._create_mock_recorder()

    # Make record_call raise an exception (simulating DB failure)
    recorder.record_call.side_effect = Exception("Database connection failed")

    emitted_events: list[ExternalCallCompleted] = []

    def telemetry_emit(event: ExternalCallCompleted) -> None:
        emitted_events.append(event)

    client = AuditedHTTPClient(
        recorder=recorder,
        state_id="state_123",
        base_url="https://api.example.com",
        run_id="run_abc",
        telemetry_emit=telemetry_emit,
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_response.text = '{"result": "success"}'

    with patch.object(client._client, "post", return_value=mock_response):
        # The call should fail (Landscape recording fails)
        with pytest.raises(Exception, match="Database connection failed"):
            client.post("/endpoint", json={"input": "test"})

    # CRITICAL: No telemetry should have been emitted
    assert len(emitted_events) == 0, "Telemetry was emitted before Landscape recording!"


def test_http_error_response_emits_telemetry(self) -> None:
    """4xx/5xx response emits telemetry with appropriate status."""
    recorder = self._create_mock_recorder()

    emitted_events: list[ExternalCallCompleted] = []

    def telemetry_emit(event: ExternalCallCompleted) -> None:
        emitted_events.append(event)

    client = AuditedHTTPClient(
        recorder=recorder,
        state_id="state_123",
        base_url="https://api.example.com",
        run_id="run_abc",
        telemetry_emit=telemetry_emit,
    )

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": "Internal Server Error"}
    mock_response.text = '{"error": "Internal Server Error"}'

    with patch.object(client._client, "post", return_value=mock_response):
        response = client.post("/endpoint", json={"input": "test"})

    # Response is returned (not raised as exception)
    assert response.status_code == 500

    # Verify telemetry event - status depends on implementation
    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert event.call_type == CallType.HTTP
    # Note: Check what status AuditedHTTPClient sets for 5xx responses
```

### Step 1.10: Run all telemetry tests

Run: `.venv/bin/python -m pytest tests/plugins/clients/test_http_telemetry.py -v`

Expected: All PASS

### Step 1.11: Commit

```bash
git add src/elspeth/plugins/clients/http.py tests/plugins/clients/test_http_telemetry.py
git commit -m "feat(telemetry): add ExternalCallCompleted emission to AuditedHTTPClient

- Add run_id and telemetry_emit parameters to __init__
- Add _emit_telemetry() method matching AuditedLLMClient pattern
- Emit telemetry AFTER Landscape recording (legal record first)
- Add comprehensive test coverage

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Wire OpenRouterLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter.py`

### Step 2.1: Read current on_start implementation

**File:** `src/elspeth/plugins/llm/openrouter.py`

Find `on_start()` method and note current implementation.

### Step 2.2: Update on_start to capture telemetry context

**File:** `src/elspeth/plugins/llm/openrouter.py`

Update `on_start()` to capture `run_id` and `telemetry_emit`:

```python
def on_start(self, ctx: PluginContext) -> None:
    """Capture recorder and telemetry context.

    Called by the engine at pipeline start. Captures references
    needed for audit and telemetry in worker threads.
    """
    self._recorder = ctx.landscape
    self._run_id = ctx.run_id
    self._telemetry_emit = ctx.telemetry_emit
```

### Step 2.3: Find where AuditedHTTPClient is created

**File:** `src/elspeth/plugins/llm/openrouter.py`

Search for `AuditedHTTPClient(` and note the location.

### Step 2.4: Update AuditedHTTPClient instantiation

Pass `run_id` and `telemetry_emit` to the constructor:

```python
self._http_clients[state_id] = AuditedHTTPClient(
    recorder=self._recorder,
    state_id=state_id,
    timeout=self._timeout,
    base_url=self._base_url,
    headers={
        "Authorization": f"Bearer {self._api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": self._http_referer,
        "X-Title": self._x_title,
    },
    run_id=self._run_id,
    telemetry_emit=self._telemetry_emit,
)
```

### Step 2.5: Run existing tests

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_openrouter.py -v`

Expected: All PASS

### Step 2.6: Commit

```bash
git add src/elspeth/plugins/llm/openrouter.py
git commit -m "feat(telemetry): wire OpenRouterLLMTransform to emit telemetry

- Capture run_id and telemetry_emit in on_start()
- Pass to AuditedHTTPClient constructor

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Wire OpenRouterMultiQueryLLMTransform

**Files:**
- Modify: `src/elspeth/plugins/llm/openrouter_multi_query.py`

### Step 3.1: Update on_start to capture telemetry context

Same pattern as Task 2:

```python
def on_start(self, ctx: PluginContext) -> None:
    """Capture recorder and telemetry context."""
    self._recorder = ctx.landscape
    self._run_id = ctx.run_id
    self._telemetry_emit = ctx.telemetry_emit
```

### Step 3.2: Update AuditedHTTPClient instantiation

Pass `run_id` and `telemetry_emit` to the constructor.

### Step 3.3: Run existing tests

Run: `.venv/bin/python -m pytest tests/plugins/llm/test_openrouter_multi_query.py -v`

Expected: All PASS

### Step 3.4: Commit

```bash
git add src/elspeth/plugins/llm/openrouter_multi_query.py
git commit -m "feat(telemetry): wire OpenRouterMultiQueryLLMTransform to emit telemetry

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Wire AzureContentSafety

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/content_safety.py`

### Step 4.1: Check current implementation

This plugin uses `AuditedHTTPClient` with manual recording. After adding telemetry support to `AuditedHTTPClient`, we need to:
1. Capture `run_id` and `telemetry_emit` in `on_start()`
2. Pass them when creating `AuditedHTTPClient`

### Step 4.2: Update on_start

```python
def on_start(self, ctx: PluginContext) -> None:
    """Capture recorder and telemetry context."""
    self._recorder = ctx.landscape
    self._run_id = ctx.run_id
    self._telemetry_emit = ctx.telemetry_emit
```

### Step 4.3: Update AuditedHTTPClient instantiation

Find where `AuditedHTTPClient` is created and add the telemetry parameters.

### Step 4.4: Run existing tests

Run: `.venv/bin/python -m pytest tests/plugins/transforms/azure/test_content_safety.py -v`

Expected: All PASS

### Step 4.5: Commit

```bash
git add src/elspeth/plugins/transforms/azure/content_safety.py
git commit -m "feat(telemetry): wire AzureContentSafety to emit telemetry

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Wire AzurePromptShield

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/prompt_shield.py`

### Step 5.1: Update on_start

Same pattern as Task 4.

### Step 5.2: Update AuditedHTTPClient instantiation

Same pattern as Task 4.

### Step 5.3: Run existing tests

Run: `.venv/bin/python -m pytest tests/plugins/transforms/azure/test_prompt_shield.py -v`

Expected: All PASS

### Step 5.4: Commit

```bash
git add src/elspeth/plugins/transforms/azure/prompt_shield.py
git commit -m "feat(telemetry): wire AzurePromptShield to emit telemetry

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Create Telemetry Wiring Verification Test

**Files:**
- Create: `tests/telemetry/test_plugin_wiring.py`

This test validates that EVERY plugin making external calls is properly wired for telemetry.

### Step 6.1: Write the verification test

```python
# tests/telemetry/test_plugin_wiring.py
"""Verify all external-call plugins are wired for telemetry.

This test ensures no plugin is accidentally left without telemetry support.
It inspects plugin classes to verify they:
1. Capture run_id and telemetry_emit in on_start()
2. Pass these to audited clients

This is a regression guard - if a new plugin is added that makes
external calls, this test will fail until it's properly wired.
"""

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest


# Plugins that make external calls and MUST emit telemetry
EXTERNAL_CALL_PLUGINS = {
    # LLM plugins using AuditedLLMClient
    "src/elspeth/plugins/llm/azure.py": {
        "class": "AzureLLMTransform",
        "client_type": "AuditedLLMClient",
        "pattern": "ctx_passthrough",  # Passes ctx.run_id directly
    },
    "src/elspeth/plugins/llm/azure_multi_query.py": {
        "class": "AzureMultiQueryLLMTransform",
        "client_type": "AuditedLLMClient",
        "pattern": "on_start_capture",  # Captures in on_start
    },
    # HTTP plugins using AuditedHTTPClient
    "src/elspeth/plugins/llm/openrouter.py": {
        "class": "OpenRouterLLMTransform",
        "client_type": "AuditedHTTPClient",
        "pattern": "on_start_capture",
    },
    "src/elspeth/plugins/llm/openrouter_multi_query.py": {
        "class": "OpenRouterMultiQueryLLMTransform",
        "client_type": "AuditedHTTPClient",
        "pattern": "on_start_capture",
    },
    "src/elspeth/plugins/transforms/azure/content_safety.py": {
        "class": "AzureContentSafety",
        "client_type": "AuditedHTTPClient",
        "pattern": "on_start_capture",
    },
    "src/elspeth/plugins/transforms/azure/prompt_shield.py": {
        "class": "AzurePromptShield",
        "client_type": "AuditedHTTPClient",
        "pattern": "on_start_capture",
    },
}

# Plugins that are EXEMPT from telemetry (with reason)
TELEMETRY_EXEMPT_PLUGINS = {
    "src/elspeth/plugins/llm/azure_batch.py": "Batch API - uses file uploads, not per-row calls",
    "src/elspeth/plugins/llm/openrouter_batch.py": "Batch API - uses file uploads, not per-row calls",
    "src/elspeth/plugins/azure/blob_source.py": "Storage I/O - future work",
    "src/elspeth/plugins/azure/blob_sink.py": "Storage I/O - future work",
    "src/elspeth/plugins/sinks/database_sink.py": "Database I/O - future work",
}


class TestTelemetryWiring:
    """Verify telemetry wiring for all external-call plugins."""

    @pytest.mark.parametrize(
        "plugin_path,config",
        list(EXTERNAL_CALL_PLUGINS.items()),
        ids=lambda x: x if isinstance(x, str) else x.get("class", "unknown"),
    )
    def test_plugin_captures_telemetry_context(
        self, plugin_path: str, config: dict[str, Any]
    ) -> None:
        """Verify plugin captures run_id and telemetry_emit."""
        full_path = Path(plugin_path)
        assert full_path.exists(), f"Plugin file not found: {plugin_path}"

        source = full_path.read_text()
        tree = ast.parse(source)

        class_name = config["class"]
        pattern = config["pattern"]

        # Find the class
        class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        assert class_node is not None, f"Class {class_name} not found in {plugin_path}"

        if pattern == "on_start_capture":
            # Check on_start captures run_id and telemetry_emit
            on_start = None
            for item in class_node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "on_start":
                    on_start = item
                    break

            assert on_start is not None, f"{class_name} missing on_start() method"

            # Check for self._run_id = ctx.run_id
            on_start_source = ast.unparse(on_start)
            assert "self._run_id" in on_start_source or "run_id" in on_start_source, (
                f"{class_name}.on_start() must capture run_id"
            )
            assert "self._telemetry_emit" in on_start_source or "telemetry_emit" in on_start_source, (
                f"{class_name}.on_start() must capture telemetry_emit"
            )

        elif pattern == "ctx_passthrough":
            # Check that ctx.run_id and ctx.telemetry_emit are passed through
            # This pattern passes them directly from PluginContext, not storing on self
            assert "ctx.run_id" in source, f"{class_name} must pass ctx.run_id"
            assert "ctx.telemetry_emit" in source, f"{class_name} must pass ctx.telemetry_emit"

    @pytest.mark.parametrize(
        "plugin_path,config",
        list(EXTERNAL_CALL_PLUGINS.items()),
        ids=lambda x: x if isinstance(x, str) else x.get("class", "unknown"),
    )
    def test_plugin_passes_telemetry_to_client(
        self, plugin_path: str, config: dict[str, Any]
    ) -> None:
        """Verify plugin passes telemetry params to audited client."""
        full_path = Path(plugin_path)
        source = full_path.read_text()

        client_type = config["client_type"]

        # Check that run_id and telemetry_emit are passed to client constructor
        assert f"{client_type}(" in source, f"{plugin_path} must use {client_type}"

        # Check for run_id= and telemetry_emit= in client instantiation
        # This is a simple heuristic - the parameters should be near the client instantiation
        assert "run_id=" in source, (
            f"{plugin_path} must pass run_id to {client_type}"
        )
        assert "telemetry_emit=" in source, (
            f"{plugin_path} must pass telemetry_emit to {client_type}"
        )

    def test_all_external_call_plugins_are_listed(self) -> None:
        """Ensure we haven't missed any plugins that make external calls.

        This test finds all plugins that import audited clients and verifies
        they are either in EXTERNAL_CALL_PLUGINS or TELEMETRY_EXEMPT_PLUGINS.
        """
        plugins_dir = Path("src/elspeth/plugins")

        # Find all Python files that import audited clients
        audited_imports = ["AuditedLLMClient", "AuditedHTTPClient"]
        found_plugins: set[str] = set()

        for py_file in plugins_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            content = py_file.read_text()
            for client in audited_imports:
                if client in content and f"{client}(" in content:
                    rel_path = str(py_file)
                    found_plugins.add(rel_path)

        # Check all found plugins are accounted for
        known_plugins = set(EXTERNAL_CALL_PLUGINS.keys()) | set(TELEMETRY_EXEMPT_PLUGINS.keys())

        unknown = found_plugins - known_plugins
        assert not unknown, (
            f"Found plugins using audited clients that are not listed in "
            f"EXTERNAL_CALL_PLUGINS or TELEMETRY_EXEMPT_PLUGINS: {unknown}"
        )
```

### Step 6.2: Run verification test

Run: `.venv/bin/python -m pytest tests/telemetry/test_plugin_wiring.py -v`

Expected: All PASS (if all wiring is complete) or specific failures showing which plugins need wiring

### Step 6.3: Commit

```bash
git add tests/telemetry/test_plugin_wiring.py
git commit -m "test(telemetry): add plugin wiring verification test

Ensures all external-call plugins are properly wired for telemetry.
Fails if new plugins are added without telemetry support.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Run Full Test Suite and Type Checks

### Step 7.1: Run mypy

Run: `.venv/bin/python -m mypy src/elspeth/plugins/clients/http.py src/elspeth/plugins/llm/openrouter.py src/elspeth/plugins/llm/openrouter_multi_query.py src/elspeth/plugins/transforms/azure/content_safety.py src/elspeth/plugins/transforms/azure/prompt_shield.py`

Expected: No errors

### Step 7.2: Run ruff

Run: `.venv/bin/python -m ruff check src/elspeth/plugins/clients/http.py src/elspeth/plugins/llm/openrouter.py src/elspeth/plugins/llm/openrouter_multi_query.py src/elspeth/plugins/transforms/azure/content_safety.py src/elspeth/plugins/transforms/azure/prompt_shield.py`

Expected: No errors

### Step 7.3: Run full test suite

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All PASS

### Step 7.4: Final commit (if any fixes needed)

```bash
git add -A
git commit -m "fix: address lint/type issues from telemetry wiring

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `AuditedHTTPClient.__init__` accepts `run_id` and `telemetry_emit`
- [ ] `AuditedHTTPClient._emit_telemetry()` method implemented
- [ ] `AuditedHTTPClient.post()` emits telemetry after `record_call()`
- [ ] Telemetry only emitted when both `run_id` and `telemetry_emit` are set
- [ ] Telemetry only emitted AFTER Landscape recording succeeds
- [ ] `OpenRouterLLMTransform` wired
- [ ] `OpenRouterMultiQueryLLMTransform` wired
- [ ] `AzureContentSafety` wired
- [ ] `AzurePromptShield` wired
- [ ] `AzureLLMTransform` verified (already wired via ctx passthrough)
- [ ] `AzureMultiQueryLLMTransform` verified (already wired)
- [ ] Plugin wiring verification test passes
- [ ] All existing tests pass
- [ ] mypy clean
- [ ] ruff clean

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/elspeth/plugins/clients/http.py` | Add telemetry support |
| `src/elspeth/plugins/llm/openrouter.py` | Wire telemetry |
| `src/elspeth/plugins/llm/openrouter_multi_query.py` | Wire telemetry |
| `src/elspeth/plugins/transforms/azure/content_safety.py` | Wire telemetry |
| `src/elspeth/plugins/transforms/azure/prompt_shield.py` | Wire telemetry |
| `tests/plugins/clients/test_http_telemetry.py` | New test file |
| `tests/telemetry/test_plugin_wiring.py` | New verification test |
