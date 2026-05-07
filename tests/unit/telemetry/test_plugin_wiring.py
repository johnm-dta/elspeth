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
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

from elspeth.contracts import CallType
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.testing import make_pipeline_row


def _make_lifecycle_ctx(events: list[Any]) -> Mock:
    """Create a mock LifecycleContext that captures telemetry events."""
    recorder = Mock()
    recorder.allocate_call_index = Mock(return_value=0)
    recorder.record_call = Mock()

    ctx = Mock()
    ctx.run_id = "test-run"
    ctx.node_id = "node-001"
    ctx.landscape = recorder
    ctx.rate_limit_registry = None
    ctx.telemetry_emit = events.append
    ctx.concurrency_config = None
    return ctx


def _make_transform_ctx(recorder: Mock) -> Mock:
    """Create a mock TransformContext for _process_row calls."""
    ctx = Mock()
    ctx.run_id = "test-run"
    ctx.state_id = "state-001"
    ctx.node_id = "node-001"
    ctx.token = Mock(token_id="token-001")
    ctx.batch_token_ids = None
    ctx.schema_contract = None
    ctx.landscape = recorder
    ctx.landscape.allocate_call_index = Mock(return_value=0)
    ctx.landscape.record_call = Mock()
    return ctx


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
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='{"score": 85}'))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump = Mock(return_value={})
            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client
            yield mock_client

    def test_telemetry_emitted_on_llm_call(self) -> None:
        """After on_start + _process_row, telemetry_emit receives ExternalCallCompleted."""
        from elspeth.plugins.transforms.llm.transform import LLMTransform

        transform = LLMTransform(
            {
                "provider": "azure",
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "Analyze: {{ row.text }}",
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

        with patch("httpx.Client") as mock_httpx_cls:
            mock_httpx = mock_httpx_cls.return_value
            mock_httpx.post.return_value = mock_response
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
        mock_limiter = Mock()
        mock_limiter.try_acquire = Mock(return_value=True)
        mock_registry = Mock()
        mock_registry.get_limiter = Mock(return_value=mock_limiter)
        lifecycle_ctx.rate_limit_registry = mock_registry

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
            patch("httpx.Client") as mock_httpx_cls,
        ):
            mock_httpx = mock_httpx_cls.return_value
            # httpx.Client is used as a context manager in get_ssrf_safe():
            #   with httpx.Client(...) as ssrf_client:
            # MagicMock.__enter__ returns a nested Mock by default, NOT self.
            # We must make it return the same mock so .get() is the one we configured.
            mock_httpx.__enter__ = Mock(return_value=mock_httpx)
            mock_httpx.__exit__ = Mock(return_value=False)
            mock_httpx.get.return_value = mock_response
            transform.process(row, process_ctx)

        # telemetry_emit must have been invoked with an HTTP call event
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
    "src/elspeth/plugins/transforms/llm/providers/openrouter.py",
    "src/elspeth/plugins/transforms/azure/base.py",
    "src/elspeth/plugins/transforms/web_scrape.py",
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
