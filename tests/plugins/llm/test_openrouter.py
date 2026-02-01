# tests/plugins/llm/test_openrouter.py
"""Tests for OpenRouter LLM transform with row-level pipelining."""

import json
from collections.abc import Generator
from unittest.mock import Mock

import httpx
import pytest

from elspeth.contracts import Determinism, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.engine.batch_adapter import ExceptionResult
from elspeth.plugins.batching.ports import CollectorOutputPort
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.openrouter import OpenRouterConfig, OpenRouterLLMTransform

from .conftest import chaosllm_openrouter_http_responses, chaosllm_openrouter_httpx_response

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


def _create_mock_response(
    chaosllm_server,
    content: str = "Analysis result",
    model: str = "anthropic/claude-3-opus",
    usage: dict[str, int] | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    raw_body: str | None = None,
    raise_for_status_error: Exception | None = None,
) -> Mock:
    """Create an httpx.Response using ChaosLLM response generation."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "test prompt"}],
        "temperature": 0.0,
    }
    response = chaosllm_openrouter_httpx_response(
        chaosllm_server,
        request,
        status_code=status_code,
        headers=headers,
        template_override=content,
        raw_body=raw_body,
        usage_override=usage,
    )
    if raise_for_status_error is not None:
        response.raise_for_status = Mock(side_effect=raise_for_status_error)
    return response


def mock_httpx_client(
    chaosllm_server,
    response: httpx.Response | list[httpx.Response] | None = None,
    side_effect: Exception | None = None,
) -> Generator[Mock, None, None]:
    """Context manager to mock httpx.Client using ChaosLLM responses."""
    if response is None and side_effect is None:
        response = _create_mock_response(chaosllm_server)
    responses = response if isinstance(response, list) else [response]
    return chaosllm_openrouter_http_responses(
        chaosllm_server,
        responses,
        side_effect=side_effect,
    )


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data={},  # Not used in these tests
    )


class TestOpenRouterConfig:
    """Tests for OpenRouterConfig validation."""

    def test_config_requires_api_key(self) -> None:
        """OpenRouterConfig requires API key."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "model": "anthropic/claude-3-opus",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )  # Missing 'api_key'

    def test_config_requires_model(self) -> None:
        """OpenRouterConfig requires model name (from LLMConfig)."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )  # Missing 'model'

    def test_config_requires_template(self) -> None:
        """OpenRouterConfig requires template (from LLMConfig)."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "anthropic/claude-3-opus",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'template'

    def test_config_requires_schema(self) -> None:
        """OpenRouterConfig requires schema (from TransformDataConfig)."""
        with pytest.raises(PluginConfigError, match="schema"):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "anthropic/claude-3-opus",
                    "template": "Analyze: {{ row.text }}",
                }
            )  # Missing 'schema'

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        assert config.api_key == "sk-test-key"
        assert config.model == "anthropic/claude-3-opus"
        assert config.template == "Analyze: {{ row.text }}"

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Hello, {{ row.name }}!",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.timeout_seconds == 60.0
        # Inherited from LLMConfig
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.response_field == "llm_response"

    def test_config_custom_base_url(self) -> None:
        """Config accepts custom base URL."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "base_url": "https://custom.proxy.com/api/v1",
            }
        )
        assert config.base_url == "https://custom.proxy.com/api/v1"

    def test_config_custom_timeout(self) -> None:
        """Config accepts custom timeout."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "timeout_seconds": 120.0,
            }
        )
        assert config.timeout_seconds == 120.0

    def test_config_invalid_timeout_rejected(self) -> None:
        """Config rejects non-positive timeout."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "openai/gpt-4",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "timeout_seconds": 0.0,  # Must be > 0
                }
            )


class TestOpenRouterLLMTransformInit:
    """Tests for OpenRouterLLMTransform initialization."""

    def test_transform_stores_config_values(self) -> None:
        """Transform stores config values as attributes."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "base_url": "https://custom.example.com/api/v1",
                "timeout_seconds": 90.0,
            }
        )

        assert transform._model == "anthropic/claude-3-opus"
        assert transform._api_key == "sk-test-key"
        assert transform._base_url == "https://custom.example.com/api/v1"
        assert transform._timeout == 90.0

    def test_determinism_is_non_deterministic(self) -> None:
        """OpenRouter transforms are marked as non-deterministic."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        ctx = PluginContext(run_id="test-run", config={})

        with pytest.raises(NotImplementedError, match="row-level pipelining"):
            transform.process({"text": "hello"}, ctx)


class TestOpenRouterLLMTransformPipelining:
    """Tests for OpenRouterLLMTransform with row-level pipelining.

    These tests verify the accept() API that uses BatchTransformMixin
    for concurrent row processing with FIFO output ordering.
    """

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    @pytest.fixture
    def ctx(self, mock_recorder: Mock) -> PluginContext:
        """Create plugin context with landscape, state_id, and token."""
        token = make_token("row-1")
        return PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

    @pytest.fixture
    def transform(self, collector: CollectorOutputPort, mock_recorder: Mock) -> Generator[OpenRouterLLMTransform, None, None]:
        """Create and initialize OpenRouter transform with pipelining."""
        t = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        # Initialize with recorder reference
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        t.on_start(init_ctx)
        # Connect output port
        t.connect_output(collector, max_pending=10)
        yield t
        # Cleanup
        t.close()

    def test_successful_api_call_emits_enriched_row(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Successful API call emits row with LLM response to output port."""
        mock_response = _create_mock_response(
            chaosllm_server,
            content="The analysis is positive.",
            usage={"prompt_tokens": 10, "completion_tokens": 25},
        )

        with mock_httpx_client(chaosllm_server, response=mock_response):
            transform.accept({"text": "hello world"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _token, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "The analysis is positive."
        assert result.row["llm_response_usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 25,
        }
        assert "llm_response_template_hash" in result.row
        assert "llm_response_variables_hash" in result.row
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus"
        # Original data preserved
        assert result.row["text"] == "hello world"

    def test_template_rendering_error_emits_error(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Template rendering failure emits TransformResult.error()."""
        # Missing required_field triggers template error (no HTTP call needed)
        with mock_httpx_client(chaosllm_server):
            transform.accept({"other_field": "value"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_http_error_400_returns_error_result(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """Non-retryable HTTP errors (4xx except 429) return TransformResult.error()."""
        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=Mock(),
                response=Mock(status_code=400),
            ),
        ):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.reason["status_code"] == 400
        assert result.retryable is False

    def test_http_error_500_raises_server_error(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """Server errors (5xx) raise ServerError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.clients.llm import ServerError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=Mock(status_code=500),
            ),
        ):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, ServerError)
        assert result.exception.retryable is True

    def test_rate_limit_429_raises_rate_limit_error(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Rate limit (429) errors raise RateLimitError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.clients.llm import RateLimitError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=Mock(),
                response=Mock(status_code=429),
            ),
        ):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, RateLimitError)
        assert result.exception.retryable is True

    def test_service_unavailable_503_raises_server_error(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Service unavailable (503) errors raise ServerError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.clients.llm import ServerError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=Mock(),
                response=Mock(status_code=503),
            ),
        ):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, ServerError)
        assert result.exception.retryable is True

    def test_overloaded_529_raises_server_error(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Overloaded (529) errors raise ServerError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.clients.llm import ServerError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "529 Site is overloaded",
                request=Mock(),
                response=Mock(status_code=529),
            ),
        ):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, ServerError)
        assert result.exception.retryable is True

    def test_network_error_raises_network_error(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Network/connection errors raise NetworkError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics:
        Network errors are transient and should be retried.
        """
        from elspeth.plugins.clients.llm import NetworkError

        with mock_httpx_client(chaosllm_server, side_effect=httpx.ConnectError("Connection refused")):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, NetworkError)
        assert result.exception.retryable is True

    def test_missing_state_id_propagates_exception(
        self, mock_recorder: Mock, transform: OpenRouterLLMTransform, collector: CollectorOutputPort
    ) -> None:
        """Missing state_id causes exception propagation, not error result.

        Per CLAUDE.md crash-on-exception policy: a missing state_id is a bug
        in calling code (our internal code, not user data), so it should crash
        rather than be converted to an error result.

        BatchTransformMixin wraps such exceptions in ExceptionResult for
        propagation through the async pattern. In production, TransformExecutor
        would re-raise this exception. In tests using collector directly,
        we see the ExceptionResult wrapper.
        """
        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id=None,  # Missing state_id - calling code bug
            token=token,
        )

        transform.accept({"text": "hello"}, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _output_token, result, _state_id = collector.results[0]

        # Exception propagates via ExceptionResult wrapper (not TransformResult)
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, RuntimeError)
        assert "state_id" in str(result.exception)

    def test_system_prompt_included_in_request(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """System prompt is included when configured."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "system_prompt": "You are a helpful assistant.",
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server)
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)

                # Verify request body
                call_args = mock_client.post.call_args
                request_body = call_args.kwargs["json"]
                messages = request_body["messages"]

                assert len(messages) == 2
                assert messages[0]["role"] == "system"
                assert messages[0]["content"] == "You are a helpful assistant."
                assert messages[1]["role"] == "user"
                assert messages[1]["content"] == "hello"
        finally:
            transform.close()

    def test_no_system_prompt_single_message(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Without system prompt, only user message is sent."""
        mock_response = _create_mock_response(chaosllm_server)
        with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            messages = request_body["messages"]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    def test_custom_response_field(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Custom response_field name is used."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "response_field": "analysis",
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Result text")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["analysis"] == "Result text"
        assert "analysis_usage" in result.row
        assert "analysis_template_hash" in result.row
        assert "analysis_variables_hash" in result.row
        assert "analysis_model" in result.row

    def test_model_from_response_used_when_available(
        self,
        ctx: PluginContext,
        transform: OpenRouterLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Model name from response is used if different from request."""
        mock_response = _create_mock_response(
            chaosllm_server,
            model="anthropic/claude-3-opus-20240229",  # Different from request
        )
        with mock_httpx_client(chaosllm_server, response=mock_response):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.row is not None
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus-20240229"

    def test_raise_for_status_called(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """raise_for_status is called on response to check errors."""
        mock_response = _create_mock_response(
            chaosllm_server,
            raise_for_status_error=httpx.HTTPStatusError(
                "400 Bad Request",
                request=Mock(),
                response=Mock(status_code=400),
            ),
        )
        with mock_httpx_client(chaosllm_server, response=mock_response):
            transform.accept({"text": "hello"}, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"

    def test_connect_output_required_before_accept(self) -> None:
        """accept() raises RuntimeError if connect_output() not called."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            state_id="test-state-id",
            token=token,
        )

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept({"text": "hello"}, ctx)

    def test_connect_output_cannot_be_called_twice(self, collector: CollectorOutputPort, mock_recorder: Mock) -> None:
        """connect_output() raises if called more than once."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            with pytest.raises(RuntimeError, match="already called"):
                transform.connect_output(collector, max_pending=10)
        finally:
            transform.close()

    def test_close_is_noop_when_not_initialized(self) -> None:
        """close() does nothing when transform wasn't fully initialized."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        transform.close()  # Should not raise


class TestOpenRouterLLMTransformIntegration:
    """Integration-style tests for edge cases."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_complex_template_with_multiple_variables(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Complex template with multiple variables works correctly."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": """
                    Analyze the following data:
                    Name: {{ row.name }}
                    Score: {{ row.score }}
                    Category: {{ row.category }}

                    Provide a summary.
                """,
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Summary text")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept(
                    {"name": "Test Item", "score": 95, "category": "A"},
                    ctx,
                )
                transform.flush_batch_processing(timeout=10.0)

                assert len(collector.results) == 1
                _, result, _state_id = collector.results[0]
                assert isinstance(result, TransformResult)
                assert result.status == "success"
                # Check the prompt was rendered correctly
                call_args = mock_client.post.call_args
                request_body = call_args.kwargs["json"]
                user_message = request_body["messages"][0]["content"]
                assert "Test Item" in user_message
                assert "95" in user_message
                assert "A" in user_message
        finally:
            transform.close()

    def test_empty_usage_handled_gracefully(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Empty usage dict from API is handled."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps(
                {
                    "choices": [{"message": {"content": "Response"}}],
                    "model": "openai/gpt-4",
                }
            ),
            headers={"content-type": "application/json"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response_usage"] == {}

    def test_connection_error_raises_network_error(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Network connection error raises NetworkError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.clients.llm import NetworkError

        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            with mock_httpx_client(chaosllm_server, side_effect=httpx.ConnectError("Failed to connect to server")):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, NetworkError)
        assert result.exception.retryable is True

    def test_timeout_passed_to_http_client(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Custom timeout_seconds is used when creating HTTP client."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "timeout_seconds": 120.0,  # Custom timeout
            }
        )

        # The transform stores the timeout internally
        assert transform._timeout == 120.0

        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server)
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"

    def test_empty_choices_emits_error(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Empty choices array emits TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps(
                {
                    "choices": [],
                    "model": "openai/gpt-4",
                    "usage": {},
                }
            ),
            headers={"content-type": "application/json"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_choices"

    def test_missing_choices_key_emits_error(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Missing 'choices' key in response emits TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps(
                {
                    "model": "openai/gpt-4",
                    "usage": {},
                }
            ),
            headers={"content-type": "application/json"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "malformed_response"

    def test_malformed_choice_structure_emits_error(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Malformed choice structure emits TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps(
                {
                    "choices": [{"wrong_key": "no message"}],
                    "model": "openai/gpt-4",
                    "usage": {},
                }
            ),
            headers={"content-type": "application/json"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "malformed_response"

    def test_invalid_json_response_emits_error(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Non-JSON response body emits TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body="<html><body>Error: Service Unavailable</body></html>",
            headers={"content-type": "text/html"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "invalid_json_response"
        assert "content_type" in result.reason
        assert result.reason["content_type"] == "text/html"
        assert "body_preview" in result.reason
        assert "Error: Service Unavailable" in result.reason["body_preview"]


class TestOpenRouterTemplateFeatures:
    """Tests for template files and lookup features in OpenRouter transform."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_lookup_data_accessible_in_template(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Lookup data is accessible via lookup.* namespace in templates."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Classify as {{ lookup.categories[0] }} or {{ lookup.categories[1] }}: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "lookup": {"categories": ["positive", "negative"]},
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="positive")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept({"text": "I love this!"}, ctx)
                transform.flush_batch_processing(timeout=10.0)

                assert len(collector.results) == 1
                _, result, _state_id = collector.results[0]
                assert isinstance(result, TransformResult)
                assert result.status == "success"
                # Verify template rendered correctly with lookup data
                call_args = mock_client.post.call_args
                request_body = call_args.kwargs["json"]
                user_message = request_body["messages"][0]["content"]
                assert "Classify as positive or negative:" in user_message
                assert "I love this!" in user_message
        finally:
            transform.close()

    def test_two_dimensional_lookup_row_plus_lookup(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Two-dimensional lookup: lookup.X[row.Y] works correctly."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Use tone: {{ lookup.tones[row.tone_id] }}. Message: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "lookup": {"tones": {"formal": "professional", "casual": "friendly"}},
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Processed")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept({"text": "Hello", "tone_id": "formal"}, ctx)
                transform.flush_batch_processing(timeout=10.0)

                assert len(collector.results) == 1
                _, result, _state_id = collector.results[0]
                assert isinstance(result, TransformResult)
                assert result.status == "success"
                call_args = mock_client.post.call_args
                request_body = call_args.kwargs["json"]
                user_message = request_body["messages"][0]["content"]
                assert "Use tone: professional" in user_message
        finally:
            transform.close()

    def test_lookup_hash_included_in_output(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Output includes lookup_hash when lookup data is configured."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Categories: {{ lookup.cats }}. Input: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "lookup": {"cats": ["A", "B", "C"]},
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Result")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept({"text": "test"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        # New audit fields should be present
        assert "llm_response_lookup_hash" in result.row
        assert result.row["llm_response_lookup_hash"] is not None
        # lookup_source is None when lookup is inline (not from file)
        assert "llm_response_lookup_source" in result.row

    def test_template_source_included_in_output(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Output includes template_source when provided."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Analyze: {{ row.text }}",
                "template_source": "prompts/analysis.j2",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Analysis result")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept({"text": "hello"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response_template_source"] == "prompts/analysis.j2"

    def test_all_audit_fields_present_with_lookup(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """All audit metadata fields are present when using template with lookup."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ lookup.prompt_prefix }} {{ row.text }}",
                "template_source": "prompts/prefixed.j2",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "lookup": {"prompt_prefix": "Please analyze:"},
                "lookup_source": "prompts/lookups.yaml",
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Done")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept({"text": "data"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None

        # All audit fields should be present
        assert "llm_response_template_hash" in result.row
        assert "llm_response_variables_hash" in result.row
        assert "llm_response_template_source" in result.row
        assert "llm_response_lookup_hash" in result.row
        assert "llm_response_lookup_source" in result.row
        assert "llm_response_model" in result.row

        # Values should be set correctly
        assert result.row["llm_response_template_source"] == "prompts/prefixed.j2"
        assert result.row["llm_response_lookup_source"] == "prompts/lookups.yaml"
        assert result.row["llm_response_lookup_hash"] is not None

    def test_no_lookup_has_none_hash_in_output(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Output has None for lookup fields when no lookup configured."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Simple: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                # No lookup configured
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Result")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept({"text": "test"}, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        # Fields should be present but None
        assert "llm_response_lookup_hash" in result.row
        assert result.row["llm_response_lookup_hash"] is None
        assert "llm_response_lookup_source" in result.row
        assert result.row["llm_response_lookup_source"] is None

    def test_lookup_iteration_in_template(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Lookup data can be iterated in templates."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": """Classify into one of:
{% for cat in lookup.categories %}
- {{ cat.name }}: {{ cat.description }}
{% endfor %}
Text: {{ row.text }}""",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "lookup": {
                    "categories": [
                        {"name": "spam", "description": "unwanted messages"},
                        {"name": "ham", "description": "legitimate messages"},
                    ]
                },
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="spam")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept({"text": "Buy now! Limited offer!"}, ctx)
                transform.flush_batch_processing(timeout=10.0)

                assert len(collector.results) == 1
                _, result, _state_id = collector.results[0]
                assert isinstance(result, TransformResult)
                assert result.status == "success"
                call_args = mock_client.post.call_args
                request_body = call_args.kwargs["json"]
                user_message = request_body["messages"][0]["content"]
                assert "spam: unwanted messages" in user_message
                assert "ham: legitimate messages" in user_message
        finally:
            transform.close()

    def test_template_error_includes_source_in_error_details(self, mock_recorder: Mock, collector: CollectorOutputPort) -> None:
        """Template rendering error includes template_source for debugging."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Missing: {{ row.required_field }}",
                "template_source": "prompts/requires_field.j2",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
            token=token,
        )

        try:
            # Process with missing required_field - should fail template rendering
            transform.accept({"other_field": "value"}, ctx)
            transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert result.reason["template_file_path"] == "prompts/requires_field.j2"


class TestOpenRouterConcurrency:
    """Tests for concurrent row processing via BatchTransformMixin."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_multiple_rows_processed_in_fifo_order(self, mock_recorder: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Multiple rows are emitted in submission order (FIFO)."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        rows = [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ]

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Response")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                for i, row in enumerate(rows):
                    token = make_token(f"row-{i}")
                    ctx = PluginContext(
                        run_id="test-run",
                        config={},
                        landscape=mock_recorder,
                        state_id=f"state-{i}",
                        token=token,
                    )
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        # Results should be in FIFO order
        assert len(collector.results) == 3
        for i, (_token, result, _state_id) in enumerate(collector.results):
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            assert result.row["text"] == rows[i]["text"]

    def test_on_start_captures_recorder(self, mock_recorder: Mock) -> None:
        """on_start() captures recorder reference for HTTP client creation."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        # Verify _recorder starts as None
        assert transform._recorder is None

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
        )
        transform.on_start(ctx)

        # Verify recorder was captured
        assert transform._recorder is mock_recorder

    def test_close_clears_recorder(self, mock_recorder: Mock, collector: CollectorOutputPort) -> None:
        """close() clears recorder reference."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        assert transform._recorder is not None

        transform.close()

        assert transform._recorder is None
