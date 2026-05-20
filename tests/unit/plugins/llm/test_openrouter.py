# tests/plugins/llm/test_openrouter.py
"""Tests for OpenRouter LLM via unified LLMTransform with row-level pipelining."""

import json
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest

from elspeth.contracts import Determinism, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.engine.batch_adapter import ExceptionResult
from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig, OpenRouterLLMProvider
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row, make_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

from .conftest import chaosllm_openrouter_http_responses, chaosllm_openrouter_httpx_response

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}
_OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"
_OPENROUTER_INVALID_MODEL = "anthropic/this-model-does-not-exist"


def _create_mock_response(
    chaosllm_server,
    content: str = "Analysis result",
    model: str = "anthropic/claude-3-opus",
    usage: dict[str, int] | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    raw_body: str | bytes | None = None,
    raise_for_status_error: Exception | None = None,
) -> httpx.Response:
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
        # Override raise_for_status to raise the specified error
        response.raise_for_status = Mock(side_effect=raise_for_status_error)  # type: ignore[method-assign]
    return response


@contextmanager
def mock_httpx_client(
    chaosllm_server,
    response: httpx.Response | list[httpx.Response] | None = None,
    side_effect: Exception | None = None,
) -> Iterator[Mock]:
    """Context manager to mock httpx.Client using ChaosLLM responses."""
    if response is None and side_effect is None:
        response = _create_mock_response(chaosllm_server)
    # Build list of responses, ensuring no None values
    if response is None:
        # side_effect is provided, responses list not used
        responses: list[httpx.Response] = []
    elif isinstance(response, list):
        responses = response
    else:
        responses = [response]
    # Cast to the expected type - these are all httpx.Response objects
    typed_responses = cast(list[dict[str, Any] | str | httpx.Response], responses)
    with chaosllm_openrouter_http_responses(
        chaosllm_server,
        typed_responses,
        side_effect=side_effect,
    ) as mock_client:
        yield mock_client


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    contract = SchemaContract(mode="FLEXIBLE", fields=(), locked=True)
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_row({}, contract=contract),  # Not used in these tests
    )


def _openrouter_config(**overrides: Any) -> dict[str, Any]:
    """Build a valid OpenRouter LLMTransform config with overrides."""
    config: dict[str, Any] = {
        "provider": "openrouter",
        "api_key": "sk-test-key",
        "model": _OPENROUTER_MODEL,
        "prompt_template": "Analyze: {{ row.text }}",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],
    }
    config.update(overrides)
    return config


class TestOpenRouterConfig:
    """Tests for OpenRouterConfig validation."""

    def test_config_requires_api_key(self) -> None:
        """OpenRouterConfig requires API key."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "model": _OPENROUTER_MODEL,
                    "prompt_template": "Analyze: {{ row.text }}",
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
                    "prompt_template": "Analyze: {{ row.text }}",
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
                    "model": _OPENROUTER_MODEL,
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'template'

    def test_config_requires_schema(self) -> None:
        """OpenRouterConfig requires schema (from TransformDataConfig)."""
        with pytest.raises(PluginConfigError, match="schema"):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": _OPENROUTER_MODEL,
                    "prompt_template": "Analyze: {{ row.text }}",
                }
            )  # Missing 'schema'

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": _OPENROUTER_MODEL,
                "prompt_template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        assert config.api_key == "sk-test-key"
        assert config.model == _OPENROUTER_MODEL
        assert config.prompt_template == "Analyze: {{ row.text }}"

    def test_config_construction_does_not_enforce_catalog_membership(self) -> None:
        """Catalog membership is a value-source concern, not a construction-time
        invariant. Construction must NOT consult the catalog (it is optional and
        runtime-primed); an unknown model constructs cleanly and is rejected later
        by the walker / composer prevalidation (check_config_value_sources).
        """
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": _OPENROUTER_INVALID_MODEL,
                "prompt_template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],
            }
        )
        assert config.model == _OPENROUTER_INVALID_MODEL

    def test_config_allows_custom_base_url_without_catalog_membership(self) -> None:
        """A custom base_url config with an arbitrary model constructs cleanly;
        the walker's applies_when predicate skips the catalog for non-canonical
        endpoints (catalog enforcement is no longer a construction concern).
        """
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "totally/custom-model",
                "prompt_template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],
                "base_url": "https://custom.proxy.com/api/v1",
            }
        )

        assert config.model == "totally/custom-model"

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": _OPENROUTER_MODEL,
                "prompt_template": "Hello, {{ row.name }}!",
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
                "prompt_template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "base_url": "https://custom.proxy.com/api/v1",
            }
        )
        assert config.base_url == "https://custom.proxy.com/api/v1"

    @pytest.mark.parametrize("base_url", ["http://127.0.0.1:8199/v1", "http://localhost:8199/v1", "http://[::1]:8199/v1"])
    def test_config_accepts_loopback_http_base_url_for_local_compatible_servers(self, base_url: str) -> None:
        """Local OpenAI-compatible test servers may use HTTP loopback."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "prompt_template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],
                "base_url": base_url,
            }
        )
        assert config.base_url == base_url

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://custom.proxy.com/api/v1",
            "http://192.168.1.10:8199/v1",
            "http://user:pass@127.0.0.1:8199/v1",
            "https://user:pass@custom.proxy.com/api/v1",
            "custom.proxy.com/api/v1",
            "https:///missing-host",
        ],
    )
    def test_config_rejects_credential_unsafe_base_url(self, base_url: str) -> None:
        """OpenRouter bearer-token base_url must be HTTPS with host and no userinfo."""
        with pytest.raises(PluginConfigError, match="base_url"):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "openai/gpt-4",
                    "prompt_template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],
                    "base_url": base_url,
                }
            )

    def test_config_custom_timeout(self) -> None:
        """Config accepts custom timeout."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "prompt_template": "{{ row.text }}",
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
                    "prompt_template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "timeout_seconds": 0.0,  # Must be > 0
                }
            )


class TestLLMTransformOpenRouterInit:
    """Tests for LLMTransform initialization with OpenRouter provider."""

    def test_transform_stores_config_values(self) -> None:
        """Transform stores config values via _config attribute."""
        transform = LLMTransform(
            _openrouter_config(
                base_url="https://custom.example.com/api/v1",
                timeout_seconds=90.0,
            )
        )

        assert transform._model == _OPENROUTER_MODEL
        assert isinstance(transform._config, OpenRouterConfig)
        assert transform._config.api_key == "sk-test-key"
        assert transform._config.base_url == "https://custom.example.com/api/v1"
        assert transform._config.timeout_seconds == 90.0

    def test_determinism_is_non_deterministic(self) -> None:
        """LLM transforms are marked as non-deterministic."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_transform_name_is_llm(self) -> None:
        """Unified LLMTransform has name 'llm'."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))
        assert transform.name == "llm"

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))
        factory = make_factory()
        ctx = make_context(run_id="test-run", landscape=factory.plugin_audit_writer())

        with pytest.raises(NotImplementedError, match="row-level pipelining"):
            transform.process(make_pipeline_row({"text": "hello"}), ctx)

    def test_declared_output_fields_populated(self) -> None:
        """Regression: declared_output_fields must be populated for collision detection.

        Before centralized collision enforcement, OpenRouter had NO
        collision check. This test verifies declared_output_fields is populated
        so TransformExecutor can enforce collision detection.
        """
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))

        assert isinstance(transform.declared_output_fields, frozenset)
        assert len(transform.declared_output_fields) > 0
        assert "llm_response" in transform.declared_output_fields
        assert "llm_response_model" in transform.declared_output_fields

    def test_output_semantics_declares_single_query_response_field_as_string(self) -> None:
        """Single-query raw LLM output is a string semantic value."""
        from elspeth.contracts.plugin_semantics import ContentKind, SemanticValueType, TextFraming

        transform = LLMTransform(_openrouter_config(response_field="analysis", prompt_template="{{ row.text }}"))

        declaration = transform.output_semantics()

        assert len(declaration.fields) == 1
        facts = declaration.fields[0]
        assert facts.field_name == "analysis"
        assert facts.value_type is SemanticValueType.STR
        assert facts.content_kind is ContentKind.UNKNOWN
        assert facts.text_framing is TextFraming.UNKNOWN
        assert facts.configured_by == ("response_field", "queries")


class TestLLMTransformOpenRouterPipelining:
    """Tests for LLMTransform (OpenRouter) with row-level pipelining.

    These tests verify the accept() API that uses BatchTransformMixin
    for concurrent row processing with FIFO output ordering.
    """

    @pytest.fixture
    def mock_factory(self) -> Mock:
        """Create a mock RecorderFactory."""
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    @pytest.fixture
    def ctx(self, mock_factory: Mock) -> PluginContext:
        """Create plugin context with landscape, state_id, and token."""
        token = make_token("row-1")
        return make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

    @pytest.fixture
    def transform(self, collector: CollectorOutputPort, mock_factory: Mock) -> Generator[LLMTransform, None, None]:
        """Create and initialize LLMTransform (OpenRouter) with pipelining."""
        t = LLMTransform(_openrouter_config())
        # Initialize with factory reference
        init_ctx = make_context(landscape=mock_factory)
        t.on_start(init_ctx)
        # Connect output port
        t.connect_output(collector, max_pending=10)
        yield t
        # Cleanup
        t.close()

    def test_successful_api_call_emits_enriched_row(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
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
            transform.accept(make_pipeline_row({"text": "hello world"}), ctx)
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
        assert result.success_reason is not None
        assert "llm_response_template_hash" in result.success_reason["metadata"]
        assert "llm_response_variables_hash" in result.success_reason["metadata"]
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus"
        # Original data preserved
        assert result.row["text"] == "hello world"

    def test_template_rendering_error_emits_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Template rendering failure emits TransformResult.error()."""
        # Missing required_field triggers template error (no HTTP call needed)
        with mock_httpx_client(chaosllm_server):
            transform.accept(make_pipeline_row({"other_field": "value"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_http_error_400_returns_error_result(
        self, ctx: PluginContext, transform: LLMTransform, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """Non-retryable HTTP errors (4xx except 429) return TransformResult.error().

        In the unified LLMTransform, the provider classifies HTTP errors and
        the strategy catches non-retryable LLMClientError as 'llm_call_failed'.
        """
        mock_response = Mock(status_code=400, text="Bad Request")
        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "Bad Request",
                request=Mock(),
                response=mock_response,
            ),
        ):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"
        assert result.retryable is False

    def test_http_error_500_raises_server_error(
        self, ctx: PluginContext, transform: LLMTransform, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """Server errors (5xx) raise ServerError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.infrastructure.clients.llm import ServerError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=Mock(status_code=500, text=""),
            ),
        ):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, ServerError)
        assert result.exception.retryable is True

    def test_rate_limit_429_raises_rate_limit_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Rate limit (429) errors raise RateLimitError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.infrastructure.clients.llm import RateLimitError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=Mock(),
                response=Mock(status_code=429, text=""),
            ),
        ):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, RateLimitError)
        assert result.exception.retryable is True

    def test_service_unavailable_503_raises_server_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Service unavailable (503) errors raise ServerError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.infrastructure.clients.llm import ServerError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=Mock(),
                response=Mock(status_code=503, text=""),
            ),
        ):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, ServerError)
        assert result.exception.retryable is True

    def test_overloaded_529_raises_server_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Overloaded (529) errors raise ServerError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.infrastructure.clients.llm import ServerError

        with mock_httpx_client(
            chaosllm_server,
            side_effect=httpx.HTTPStatusError(
                "529 Site is overloaded",
                request=Mock(),
                response=Mock(status_code=529, text=""),
            ),
        ):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, ServerError)
        assert result.exception.retryable is True

    def test_network_error_raises_network_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Network/connection errors raise NetworkError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics:
        Network errors are transient and should be retried.
        """
        from elspeth.plugins.infrastructure.clients.llm import NetworkError

        with mock_httpx_client(chaosllm_server, side_effect=httpx.ConnectError("Connection refused")):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, NetworkError)
        assert result.exception.retryable is True

    def test_missing_state_id_propagates_exception(
        self, mock_factory: Mock, transform: LLMTransform, collector: CollectorOutputPort
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
            landscape=mock_factory,
            state_id=None,  # Missing state_id - calling code bug
            token=token,
        )

        transform.accept(make_pipeline_row({"text": "hello"}), ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _output_token, result, _state_id = collector.results[0]

        # Exception propagates via ExceptionResult wrapper (not TransformResult)
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, RuntimeError)
        assert "state_id" in str(result.exception)

    def test_system_prompt_included_in_request(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """System prompt is included when configured."""
        transform = LLMTransform(
            _openrouter_config(
                prompt_template="{{ row.text }}",
                system_prompt="You are a helpful assistant.",
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server)
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
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
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Without system prompt, only user message is sent."""
        mock_response = _create_mock_response(chaosllm_server)
        with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            messages = request_body["messages"]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    def test_custom_response_field(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Custom response_field name is used."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="{{ row.text }}",
                response_field="analysis",
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Result text")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
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
        assert result.success_reason is not None
        assert "analysis_template_hash" in result.success_reason["metadata"]
        assert "analysis_variables_hash" in result.success_reason["metadata"]
        assert "analysis_model" in result.row

    def test_model_from_response_used_when_available(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Model name from response is used if different from request."""
        mock_response = _create_mock_response(
            chaosllm_server,
            model="anthropic/claude-3-opus-20240229",  # Different from request
        )
        with mock_httpx_client(chaosllm_server, response=mock_response):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.row is not None
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus-20240229"

    def test_raise_for_status_called(
        self, ctx: PluginContext, transform: LLMTransform, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """raise_for_status is called on response to check errors."""
        mock_response = _create_mock_response(
            chaosllm_server,
            raise_for_status_error=httpx.HTTPStatusError(
                "400 Bad Request",
                request=Mock(),
                response=Mock(status_code=400, text="Bad Request"),
            ),
        )
        with mock_httpx_client(chaosllm_server, response=mock_response):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"

    def test_connect_output_required_before_accept(self) -> None:
        """accept() raises RuntimeError if connect_output() not called."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))

        token = make_token("row-1")
        ctx = make_context(
            run_id="test-run",
            state_id="test-state-id",
            token=token,
        )

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)

    def test_connect_output_cannot_be_called_twice(self, collector: CollectorOutputPort, mock_factory: Mock) -> None:
        """connect_output() raises if called more than once."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            with pytest.raises(RuntimeError, match="already called"):
                transform.connect_output(collector, max_pending=10)
        finally:
            transform.close()

    def test_close_is_noop_when_not_initialized(self) -> None:
        """close() does nothing when transform wasn't fully initialized."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))
        transform.close()  # Should not raise


class TestLLMTransformOpenRouterIntegration:
    """Integration-style tests for edge cases with OpenRouter provider."""

    @pytest.fixture
    def mock_factory(self) -> Mock:
        """Create a mock RecorderFactory."""
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_complex_template_with_multiple_variables(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Complex template with multiple variables works correctly."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="""
                    Analyze the following data:
                    Name: {{ row.name }}
                    Score: {{ row.score }}
                    Category: {{ row.category }}

                    Provide a summary.
                """,
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Summary text")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept(
                    make_pipeline_row({"name": "Test Item", "score": 95, "category": "A"}),
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

    def test_empty_usage_handled_gracefully(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Empty usage dict from API is handled."""
        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps(
                {
                    "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
                    "model": "openai/gpt-4",
                }
            ),
            headers={"content-type": "application/json"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response_usage"] == {}

    def test_connection_error_raises_network_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Network connection error raises NetworkError for engine RetryManager.

        Regression test for P2-2026-01-31-openrouter-retry-semantics.
        """
        from elspeth.plugins.infrastructure.clients.llm import NetworkError

        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            with mock_httpx_client(chaosllm_server, side_effect=httpx.ConnectError("Failed to connect to server")):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, NetworkError)
        assert result.exception.retryable is True

    def test_timeout_stored_in_config(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Custom timeout_seconds is stored in _config for provider creation."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="{{ row.text }}",
                timeout_seconds=120.0,
            )
        )

        # The transform stores the timeout in its config
        assert isinstance(transform._config, OpenRouterConfig)
        assert transform._config.timeout_seconds == 120.0

        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server)
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"

    def test_empty_choices_emits_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Empty choices array emits error via provider exception.

        In the unified LLMTransform, the provider raises LLMClientError for
        empty choices, which the strategy catches as 'llm_call_failed'.
        """
        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
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
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"

    def test_null_content_from_content_filtering_emits_error(
        self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """Null content (content filtering) returns error.

        P0-05: When OpenRouter returns null content due to content filtering,
        the provider raises ContentPolicyError which the strategy catches
        as a non-retryable 'llm_call_failed'.
        """
        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps(
                {
                    "choices": [{"message": {"content": None, "role": "assistant"}}],
                    "model": "openai/gpt-4",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 0},
                }
            ),
            headers={"content-type": "application/json"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"

    def test_missing_choices_key_emits_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Missing 'choices' key in response emits error via provider exception.

        In the unified LLMTransform, the provider raises LLMClientError for
        missing choices, which the strategy catches as 'llm_call_failed'.
        """
        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
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
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"

    def test_malformed_choice_structure_emits_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Malformed choice structure emits error via provider exception.

        In the unified LLMTransform, the provider raises LLMClientError for
        malformed response structure, which the strategy catches as 'llm_call_failed'.
        """
        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
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
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"

    def test_invalid_json_response_emits_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Non-JSON response body emits error via provider exception.

        In the unified LLMTransform, the provider raises LLMClientError for
        invalid JSON, which the strategy catches as 'llm_call_failed'.
        """
        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        response = _create_mock_response(
            chaosllm_server,
            raw_body="<html><body>Error: Service Unavailable</body></html>",
            headers={"content-type": "text/html"},
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"


class TestOpenRouterTemplateFeatures:
    """Tests for template files and lookup features via LLMTransform (OpenRouter)."""

    @pytest.fixture
    def mock_factory(self) -> Mock:
        """Create a mock RecorderFactory."""
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_lookup_data_accessible_in_template(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Lookup data is accessible via lookup.* namespace in templates."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="Classify as {{ lookup.categories[0] }} or {{ lookup.categories[1] }}: {{ row.text }}",
                lookup={"categories": ["positive", "negative"]},
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="positive")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept(make_pipeline_row({"text": "I love this!"}), ctx)
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

    def test_two_dimensional_lookup_row_plus_lookup(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Two-dimensional lookup: lookup.X[row.Y] works correctly."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="Use tone: {{ lookup.tones[row.tone_id] }}. Message: {{ row.text }}",
                lookup={"tones": {"formal": "professional", "casual": "friendly"}},
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Processed")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept(make_pipeline_row({"text": "Hello", "tone_id": "formal"}), ctx)
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

    def test_lookup_hash_included_in_output(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Output includes lookup_hash when lookup data is configured."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="Categories: {{ lookup.cats }}. Input: {{ row.text }}",
                lookup={"cats": ["A", "B", "C"]},
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Result")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept(make_pipeline_row({"text": "test"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        # Audit fields should be in success_reason["metadata"], not the row
        assert result.success_reason is not None
        metadata = result.success_reason["metadata"]
        assert "llm_response_lookup_hash" in metadata
        assert metadata["llm_response_lookup_hash"] is not None
        # lookup_source is None when lookup is inline (not from file)
        assert "llm_response_lookup_source" in metadata

    def test_template_source_included_in_output(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Output includes template_source when provided."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="Analyze: {{ row.text }}",
                prompt_template_source="prompts/analysis.j2",
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Analysis result")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        assert result.success_reason is not None
        assert result.success_reason["metadata"]["llm_response_template_source"] == "prompts/analysis.j2"

    def test_all_audit_fields_present_with_lookup(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """All audit metadata fields are present when using template with lookup."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="{{ lookup.prompt_prefix }} {{ row.text }}",
                prompt_template_source="prompts/prefixed.j2",
                lookup={"prompt_prefix": "Please analyze:"},
                lookup_source="prompts/lookups.yaml",
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Done")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept(make_pipeline_row({"text": "data"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None

        # Audit fields should be in success_reason["metadata"], not the row
        assert result.success_reason is not None
        metadata = result.success_reason["metadata"]
        assert "llm_response_template_hash" in metadata
        assert "llm_response_variables_hash" in metadata
        assert "llm_response_template_source" in metadata
        assert "llm_response_lookup_hash" in metadata
        assert "llm_response_lookup_source" in metadata
        assert "llm_response_model" in result.row

        # Values should be set correctly
        assert metadata["llm_response_template_source"] == "prompts/prefixed.j2"
        assert metadata["llm_response_lookup_source"] == "prompts/lookups.yaml"
        assert metadata["llm_response_lookup_hash"] is not None

    def test_no_lookup_has_none_hash_in_output(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Output has None for lookup fields when no lookup configured."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="Simple: {{ row.text }}",
                # No lookup configured
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="Result")
            with mock_httpx_client(chaosllm_server, response=mock_response):
                transform.accept(make_pipeline_row({"text": "test"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        # Audit fields should be in success_reason["metadata"], not the row
        assert result.success_reason is not None
        metadata = result.success_reason["metadata"]
        assert "llm_response_lookup_hash" in metadata
        assert metadata["llm_response_lookup_hash"] is None
        assert "llm_response_lookup_source" in metadata
        assert metadata["llm_response_lookup_source"] is None

    def test_lookup_iteration_in_template(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Lookup data can be iterated in templates."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="""Classify into one of:
{% for cat in lookup.categories %}
- {{ cat.name }}: {{ cat.description }}
{% endfor %}
Text: {{ row.text }}""",
                lookup={
                    "categories": [
                        {"name": "spam", "description": "unwanted messages"},
                        {"name": "ham", "description": "legitimate messages"},
                    ]
                },
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            mock_response = _create_mock_response(chaosllm_server, content="spam")
            with mock_httpx_client(chaosllm_server, response=mock_response) as mock_client:
                transform.accept(make_pipeline_row({"text": "Buy now! Limited offer!"}), ctx)
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

    def test_template_error_includes_source_in_error_details(self, mock_factory: Mock, collector: CollectorOutputPort) -> None:
        """Template rendering error includes template_source for debugging."""
        transform = LLMTransform(
            _openrouter_config(
                model="openai/gpt-4",
                prompt_template="Missing: {{ row.required_field }}",
                prompt_template_source="prompts/requires_field.j2",
            )
        )
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            state_id="test-state-id",
            token=token,
            landscape=mock_factory,
        )

        try:
            # Process with missing required_field - should fail template rendering
            transform.accept(make_pipeline_row({"other_field": "value"}), ctx)
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
    """Tests for concurrent row processing via BatchTransformMixin (OpenRouter)."""

    @pytest.fixture
    def mock_factory(self) -> Mock:
        """Create a mock RecorderFactory."""
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_multiple_rows_processed_in_fifo_order(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Multiple rows are emitted in submission order (FIFO)."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
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
                    ctx = make_context(
                        state_id=f"state-{i}",
                        token=token,
                        landscape=mock_factory,
                    )
                    transform.accept(make_pipeline_row(row), ctx)

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

    def test_on_start_captures_recorder(self, mock_factory: Mock) -> None:
        """on_start() captures factory reference for provider creation."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))

        # Verify _recorder starts as None
        assert transform._recorder is None

        ctx = make_context(
            state_id="test-state-id",
            landscape=mock_factory,
        )
        transform.on_start(ctx)

        # Verify factory was captured
        assert transform._recorder is mock_factory

    def test_close_clears_recorder(self, mock_factory: Mock, collector: CollectorOutputPort) -> None:
        """close() clears factory reference."""
        transform = LLMTransform(_openrouter_config(prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        assert transform._recorder is not None

        transform.close()

        assert transform._recorder is None


class TestOpenRouterNanRejection:
    """Regression: OpenRouter response parsing must reject NaN/Infinity.

    response.json() accepts NaN/Infinity by default. Using json.loads
    with parse_constant rejects them. Usage values must also be validated
    for finiteness (overflow literals like 1e309 produce float('inf')).

    In the unified LLMTransform, the provider (OpenRouterLLMProvider)
    handles NaN/Infinity rejection and raises LLMClientError, which
    the strategy catches as 'llm_call_failed'.
    """

    @pytest.fixture
    def mock_factory(self) -> Mock:
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    def test_nan_in_response_body_returns_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """NaN token in API response body is rejected at parse boundary."""
        nan_body = '{"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": NaN}, "model": "test"}'
        response = _create_mock_response(chaosllm_server, raw_body=nan_body, status_code=200, headers={"content-type": "application/json"})

        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            run_id="test",
            state_id="test-state",
            token=token,
            landscape=mock_factory,
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "llm_call_failed"
        finally:
            transform.close()

    def test_non_finite_usage_value_returns_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Usage field with overflow float (inf from 1e309) is rejected."""
        inf_body = '{"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1e309}, "model": "test"}'
        response = _create_mock_response(chaosllm_server, raw_body=inf_body, status_code=200, headers={"content-type": "application/json"})

        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            run_id="test",
            state_id="test-state",
            token=token,
            landscape=mock_factory,
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "llm_call_failed"
        finally:
            transform.close()

    def test_response_missing_model_routes_error_not_fabricated(
        self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server
    ) -> None:
        """A response that omits 'model' must NOT be back-filled with the requested model.

        Substituting the requested model fabricates a Tier-3 datum: the audit ``calls``
        row, the ``{response_field}_model`` pipeline field, and the success metadata would
        all assert a model the provider never reported, while the recorded raw_response
        contains no model — two contradictory sources of truth. The provider must instead
        record the malformed response and raise, routing the row to on_error (mirroring the
        Azure provider's ``_validate_provider_response_model``).
        """
        no_model_body = (
            '{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], '
            '"usage": {"prompt_tokens": 10, "completion_tokens": 25}}'
        )
        response = _create_mock_response(
            chaosllm_server, raw_body=no_model_body, status_code=200, headers={"content-type": "application/json"}
        )

        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(run_id="test", state_id="test-state", token=token, landscape=mock_factory)

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "llm_call_failed"
            assert "model" in result.reason["error"].lower()
            # No fabricated model leaked into an output row.
            assert result.row is None
        finally:
            transform.close()


class TestOpenRouterMalformedUtf8:
    """Regression: Invalid UTF-8 in API response must fail, not silently mutate.

    Parsing via response.text uses lossy decoding (replacement characters),
    which can let corrupted external data pass as valid JSON instead of
    failing at the boundary. Parsing from response.content (raw bytes)
    ensures decode errors surface as errors.

    In the unified LLMTransform, the provider raises LLMClientError which
    the strategy catches as 'llm_call_failed'.
    """

    @pytest.fixture
    def mock_factory(self) -> Mock:
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    def test_invalid_utf8_response_returns_error(self, mock_factory: Mock, collector: CollectorOutputPort, chaosllm_server) -> None:
        """Response with invalid UTF-8 bytes must be rejected, not silently decoded."""
        # Valid JSON prefix with invalid UTF-8 continuation byte embedded
        invalid_utf8 = b'{"choices": [{"message": {"content": "hello \x80\x81 world"}}]}'
        response = _create_mock_response(
            chaosllm_server,
            raw_body=invalid_utf8,
            status_code=200,
            headers={"content-type": "application/json"},
        )

        transform = LLMTransform(_openrouter_config(model="openai/gpt-4", prompt_template="{{ row.text }}"))
        init_ctx = make_context(landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(
            run_id="test",
            state_id="test-state",
            token=token,
            landscape=mock_factory,
        )

        try:
            with mock_httpx_client(chaosllm_server, response=response):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "llm_call_failed"
        finally:
            transform.close()


class TestOpenRouterRefcount:
    """Tests for HTTP client reference counting correctness."""

    def _make_provider(self) -> OpenRouterLLMProvider:
        """Create provider with mock dependencies for refcount testing."""
        return OpenRouterLLMProvider(
            api_key="sk-test",
            recorder=Mock(),
            run_id="test-run",
            telemetry_emit=Mock(),
        )

    def test_release_unknown_state_id_raises(self) -> None:
        """Releasing an unacquired state_id is a refcount underflow — must crash."""
        provider = self._make_provider()

        with pytest.raises(RuntimeError, match="unknown state_id"):
            provider._release_http_client("never-acquired")
