# tests/unit/telemetry/test_plugin_wiring.py
"""Verify all external-call plugins wire telemetry correctly.

Behavioral tests — each plugin is instantiated, started, and processes
a row. We verify that telemetry_emit is invoked with an
ExternalCallCompleted event, proving the full wiring chain works:

    on_start(ctx) → captures telemetry_emit → creates audited client → client emits telemetry

This is a regression guard: if a plugin bypasses audited clients or
forgets to pass telemetry_emit, these tests fail.
"""

from __future__ import annotations

import ast
import itertools
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest

from elspeth.contracts import Call, CallStatus, CallType
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.core.rate_limit.registry import NoOpLimiter
from elspeth.testing import make_pipeline_row


class _ExecutionRepositoryDouble:
    def __init__(self) -> None:
        self._call_counter = itertools.count()
        self._operation_call_counter = itertools.count()
        self.recorded_calls: list[dict[str, Any]] = []

    def allocate_call_index(self, state_id: str) -> int:
        return next(self._call_counter)

    def allocate_operation_call_index(self, operation_id: str) -> int:
        return next(self._operation_call_counter)

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: Any,
        response_data: Any | None = None,
        error: Any | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        call_kwargs = {
            "state_id": state_id,
            "call_index": call_index,
            "call_type": call_type,
            "status": status,
            "request_data": request_data,
            "response_data": response_data,
            "error": error,
            "latency_ms": latency_ms,
            "request_ref": request_ref,
            "response_ref": response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
        }
        self.recorded_calls.append(call_kwargs)
        return self._recorded_call(call_kwargs)

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: Any,
        response_data: Any | None = None,
        error: Any | None = None,
        latency_ms: float | None = None,
        *,
        call_index: int | None = None,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        actual_call_index = call_index if call_index is not None else self.allocate_operation_call_index(operation_id)
        call_kwargs = {
            "operation_id": operation_id,
            "call_index": actual_call_index,
            "call_type": call_type,
            "status": status,
            "request_data": request_data,
            "response_data": response_data,
            "error": error,
            "latency_ms": latency_ms,
            "request_ref": request_ref,
            "response_ref": response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
        }
        self.recorded_calls.append(call_kwargs)
        return self._recorded_call(call_kwargs)

    def _recorded_call(self, call_kwargs: dict[str, Any]) -> Call:
        return Call(
            call_id=f"call_{len(self.recorded_calls)}",
            call_index=call_kwargs["call_index"],
            call_type=call_kwargs["call_type"],
            status=call_kwargs["status"],
            request_hash="req_hash_123",
            response_hash="resp_hash_456" if call_kwargs["response_data"] is not None else None,
            created_at=datetime.now(UTC),
            state_id=call_kwargs.get("state_id"),
            operation_id=call_kwargs.get("operation_id"),
            request_ref=call_kwargs["request_ref"] or "request_payload_ref",
            response_ref=call_kwargs["response_ref"] or ("response_payload_ref" if call_kwargs["response_data"] is not None else None),
            latency_ms=call_kwargs["latency_ms"],
            resolved_prompt_template_hash=call_kwargs["resolved_prompt_template_hash"],
        )


class _RateLimitRegistryDouble:
    def __init__(self) -> None:
        self.limiter = NoOpLimiter()

    def get_limiter(self, service_name: str) -> NoOpLimiter:
        return self.limiter


class _FakeOpenAICompletionCreator:
    def __init__(self) -> None:
        message = SimpleNamespace(content='{"score": 85}')
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        self.response = SimpleNamespace(
            choices=[choice],
            model="gpt-4o",
            usage=usage,
            model_dump=lambda: {},
        )
        self.call_count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self.response


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.create_completion = _FakeOpenAICompletionCreator()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create_completion))
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _HTTPClientDouble:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.post_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.stream_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.closed = False

    def __enter__(self) -> _HTTPClientDouble:
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.post_calls.append((args, kwargs))
        return self.response

    def request(self, *args: Any, **kwargs: Any) -> httpx.Response:
        self.post_calls.append((args, kwargs))
        return self.response

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        self.stream_calls.append((args, kwargs))
        return nullcontext(self.response)

    def close(self) -> None:
        self.closed = True


def _make_lifecycle_ctx(events: list[Any]) -> SimpleNamespace:
    """Create a LifecycleContext double that captures telemetry events."""
    return SimpleNamespace(
        run_id="test-run",
        node_id="node-001",
        landscape=_ExecutionRepositoryDouble(),
        rate_limit_registry=None,
        telemetry_emit=events.append,
        concurrency_config=None,
        payload_store=SimpleNamespace(store=lambda data: "payload_hash"),
        shutdown_event=None,
    )


def _make_transform_ctx(recorder: _ExecutionRepositoryDouble) -> SimpleNamespace:
    """Create a TransformContext double for _process_row calls."""
    return SimpleNamespace(
        run_id="test-run",
        state_id="state-001",
        node_id="node-001",
        token=SimpleNamespace(token_id="token-001"),
        batch_token_ids=None,
        schema_contract=None,
        landscape=recorder,
        shutdown_event=None,
    )


# ---------------------------------------------------------------------------
# Behavioral tests: verify telemetry_emit is invoked after external calls
# ---------------------------------------------------------------------------


class TestLLMTransformTelemetryWiring:
    """LLMTransform (azure provider) wires telemetry_emit through to AuditedLLMClient.

    Chain: on_start → AzureLLMProvider → AuditedLLMClient → telemetry_emit
    """

    @pytest.fixture(autouse=True)
    def mock_azure_openai(self):
        with patch("openai.AzureOpenAI") as mock_cls:
            client = _FakeOpenAIClient()
            mock_cls.return_value = client
            yield client

    def test_telemetry_emitted_on_llm_call(self) -> None:
        """After on_start + _process_row, telemetry_emit receives ExternalCallCompleted."""
        from elspeth.plugins.transforms.llm.transform import LLMTransform

        transform = LLMTransform(
            {
                "provider": "azure",
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "prompt_template": "Analyze: {{ row.text }}",
                "schema": {"mode": "observed"},
                "required_input_fields": [],
            }
        )
        transform.on_error = "quarantine_sink"

        events: list[Any] = []
        lifecycle_ctx = _make_lifecycle_ctx(events)
        transform.on_start(lifecycle_ctx)

        row = make_pipeline_row({"text": "Test input"})
        process_ctx = _make_transform_ctx(lifecycle_ctx.landscape)
        transform._process_row(row, process_ctx)

        # telemetry_emit must have been invoked with an LLM call event
        llm_events = [e for e in events if isinstance(e, ExternalCallCompleted) and e.call_type == CallType.LLM]
        assert len(llm_events) >= 1, (
            f"Expected ExternalCallCompleted(LLM) event from telemetry_emit, got: {[type(e).__name__ for e in events]}"
        )

        transform.close()


class TestBedrockProviderTelemetryWiring:
    """BedrockLLMProvider emits through its AuditedLLMClient."""

    def test_telemetry_emitted_on_bedrock_llm_call(self) -> None:
        from litellm.types.utils import ModelResponse, Usage

        from elspeth.plugins.transforms.llm.providers.bedrock import BedrockLLMProvider

        response = ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "classified"},
                    "finish_reason": "stop",
                }
            ],
            model="bedrock/anthropic.test-model-v1:0",
            usage=Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
        events: list[Any] = []
        provider = BedrockLLMProvider(
            region_name=None,
            recorder=cast(PluginAuditWriter, _ExecutionRepositoryDouble()),
            run_id="test-run",
            telemetry_emit=events.append,
        )

        try:
            with patch("litellm.completion", return_value=response):
                result = provider.execute_query(
                    messages=[{"role": "user", "content": "classify"}],
                    model="bedrock/anthropic.test-model-v1:0",
                    temperature=0.0,
                    max_tokens=16,
                    state_id="state-001",
                    token_id="token-001",
                )
        finally:
            provider.close()

        assert result.content == "classified"
        llm_events = [event for event in events if isinstance(event, ExternalCallCompleted) and event.call_type == CallType.LLM]
        assert len(llm_events) == 1


class TestAzureSafetyTelemetryWiring:
    """Azure safety transforms wire telemetry_emit through to AuditedHTTPClient.

    Chain: on_start → _get_http_client → AuditedHTTPClient → telemetry_emit
    Tested via AzureContentSafety (concrete subclass of BaseAzureSafetyTransform).
    """

    def test_telemetry_emitted_on_safety_api_call(self) -> None:
        """After on_start + _process_row, telemetry_emit receives ExternalCallCompleted."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["text"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
                "required_input_fields": [],
            }
        )
        transform.on_error = "quarantine_sink"

        events: list[Any] = []
        lifecycle_ctx = _make_lifecycle_ctx(events)
        transform.on_start(lifecycle_ctx)

        row = make_pipeline_row({"text": "Safe content for analysis"})
        process_ctx = _make_transform_ctx(lifecycle_ctx.landscape)

        # Mock the httpx.Client that AuditedHTTPClient creates internally
        api_response = {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
                {"category": "Violence", "severity": 0},
                {"category": "SelfHarm", "severity": 0},
                {"category": "Sexual", "severity": 0},
            ]
        }
        mock_response = httpx.Response(
            200,
            json=api_response,
            request=httpx.Request("POST", "https://test.cognitiveservices.azure.com/contentsafety/text:analyze"),
        )

        httpx_client = _HTTPClientDouble(mock_response)
        with patch("httpx.Client", return_value=httpx_client):
            transform._process_row(row, process_ctx)

        # telemetry_emit must have been invoked with an HTTP call event
        http_events = [e for e in events if isinstance(e, ExternalCallCompleted) and e.call_type == CallType.HTTP]
        assert len(http_events) >= 1, (
            f"Expected ExternalCallCompleted(HTTP) event from telemetry_emit, got: {[type(e).__name__ for e in events]}"
        )

        transform.close()


class TestWebScrapeTelemetryWiring:
    """WebScrapeTransform wires telemetry_emit through to AuditedHTTPClient.

    Chain: on_start → process → AuditedHTTPClient → telemetry_emit
    """

    def test_telemetry_emitted_on_web_scrape(self) -> None:
        """After on_start + process, telemetry_emit receives ExternalCallCompleted."""
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        transform = WebScrapeTransform(
            {
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "unit test",
                    "timeout": 10,
                },
                "schema": {"mode": "observed"},
                "required_input_fields": ["url"],
            }
        )
        transform.on_error = "quarantine_sink"

        events: list[Any] = []
        lifecycle_ctx = _make_lifecycle_ctx(events)
        # WebScrapeTransform requires rate_limit_registry
        lifecycle_ctx.rate_limit_registry = _RateLimitRegistryDouble()

        transform.on_start(lifecycle_ctx)

        row = make_pipeline_row({"url": "https://example.com/page"})
        process_ctx = _make_transform_ctx(lifecycle_ctx.landscape)

        # Mock SSRF validation and HTTP response
        from elspeth.core.security.web import SSRFSafeRequest

        safe_request = SSRFSafeRequest(
            original_url="https://example.com/page",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=443,
            path="/page",
            scheme="https",
            bare_hostname="example.com",
        )
        mock_response = httpx.Response(
            200,
            text="<html><body><p>Test content</p></body></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", "https://93.184.216.34:443/page"),
        )

        with (
            patch(
                "elspeth.plugins.transforms.web_scrape.validate_url_for_ssrf",
                return_value=safe_request,
            ),
            patch("httpx.Client", return_value=_HTTPClientDouble(mock_response)),
        ):
            transform.process(row, process_ctx)

        # telemetry_emit must have been invoked with an HTTP call event
        http_events = [e for e in events if isinstance(e, ExternalCallCompleted) and e.call_type == CallType.HTTP]
        assert len(http_events) >= 1, (
            f"Expected ExternalCallCompleted(HTTP) event from telemetry_emit, got: {[type(e).__name__ for e in events]}"
        )

        transform.close()


class TestBlobFetchTelemetryWiring:
    """BlobFetch wires telemetry_emit through to AuditedHTTPClient.

    Chain: on_start -> process -> AuditedHTTPClient -> telemetry_emit
    """

    def test_telemetry_emitted_on_blob_fetch(self) -> None:
        """After on_start + process, telemetry_emit receives ExternalCallCompleted."""
        from elspeth.core.security.web import SSRFSafeRequest
        from elspeth.plugins.transforms.blob_fetch import BlobFetch

        transform = BlobFetch(
            {
                "url_field": "url",
                "http": {
                    "abuse_contact": "test@example.com",
                    "fetch_reason": "unit test",
                    "timeout": 10,
                    "allowed_hosts": ["93.184.216.34/32"],
                },
                "schema": {"mode": "observed"},
                "required_input_fields": ["url"],
            }
        )

        events: list[Any] = []
        lifecycle_ctx = _make_lifecycle_ctx(events)
        lifecycle_ctx.rate_limit_registry = _RateLimitRegistryDouble()
        transform.on_start(lifecycle_ctx)

        row = make_pipeline_row({"url": "https://example.com/data.csv"})
        process_ctx = _make_transform_ctx(lifecycle_ctx.landscape)
        safe_request = SSRFSafeRequest(
            original_url="https://example.com/data.csv",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=443,
            path="/data.csv",
            scheme="https",
            bare_hostname="example.com",
        )
        mock_response = httpx.Response(
            200,
            content=b"id,name\n1,alice\n",
            headers={"content-type": "text/csv"},
            request=httpx.Request("GET", "https://93.184.216.34:443/data.csv"),
        )

        with (
            patch(
                "elspeth.plugins.transforms.blob_fetch.validate_url_for_ssrf",
                return_value=safe_request,
            ),
            patch("httpx.Client", return_value=_HTTPClientDouble(mock_response)),
        ):
            result = transform.process(row, process_ctx)

        assert result.status == "success"
        http_events = [e for e in events if isinstance(e, ExternalCallCompleted) and e.call_type == CallType.HTTP]
        assert len(http_events) >= 1, (
            f"Expected ExternalCallCompleted(HTTP) event from telemetry_emit, got: {[type(e).__name__ for e in events]}"
        )

        transform.close()


# ---------------------------------------------------------------------------
# Structural discovery: find unregistered plugins that use audited clients
# ---------------------------------------------------------------------------

# All known plugins that use audited clients (wired or exempt)
_KNOWN_AUDITED_CLIENT_USERS: set[str] = {
    # Wired — tested behaviorally above
    "src/elspeth/plugins/transforms/llm/transform.py",
    "src/elspeth/plugins/transforms/llm/providers/azure.py",
    "src/elspeth/plugins/transforms/llm/providers/bedrock.py",
    "src/elspeth/plugins/transforms/llm/providers/openrouter.py",
    "src/elspeth/plugins/transforms/azure/base.py",
    "src/elspeth/plugins/transforms/azure/document_intelligence.py",
    "src/elspeth/plugins/transforms/web_scrape.py",
    "src/elspeth/plugins/transforms/blob_fetch.py",
    # Batch APIs — use file uploads, not per-row audited clients
    "src/elspeth/plugins/transforms/llm/azure_batch.py",
    "src/elspeth/plugins/transforms/llm/openrouter_batch.py",
    # Legacy transforms (pending deletion)
    "src/elspeth/plugins/transforms/llm/azure.py",
    "src/elspeth/plugins/transforms/llm/azure_multi_query.py",
    "src/elspeth/plugins/transforms/llm/openrouter.py",
    "src/elspeth/plugins/transforms/llm/openrouter_multi_query.py",
    # RAG retrieval — uses AuditedHTTPClient for Azure Search API
    "src/elspeth/plugins/infrastructure/clients/retrieval/azure_search.py",
    # Client definitions (define, not use)
    "src/elspeth/plugins/infrastructure/clients/llm.py",
    "src/elspeth/plugins/infrastructure/clients/http.py",
    "src/elspeth/plugins/transforms/llm/base.py",
}


class TestExternalCallPluginRegistry:
    """Ensure no plugin uses audited clients without being registered."""

    def test_all_audited_client_users_are_registered(self) -> None:
        """Find plugins that import and instantiate audited clients.

        Uses AST parsing to detect actual constructor calls (not just string
        matching). Fails if any unregistered plugin uses audited clients.
        """
        plugins_dir = Path("src/elspeth/plugins")
        audited_client_names = {"AuditedLLMClient", "AuditedHTTPClient"}
        found_plugins: set[str] = set()

        for py_file in plugins_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            # Check for constructor calls: AuditedLLMClient(...) or AuditedHTTPClient(...)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in audited_client_names:
                    found_plugins.add(str(py_file))

        unknown = found_plugins - _KNOWN_AUDITED_CLIENT_USERS
        assert not unknown, (
            f"Found plugins using audited clients that are not registered in "
            f"_KNOWN_AUDITED_CLIENT_USERS: {unknown}. "
            f"Add them to the known set and create a behavioral telemetry test."
        )
