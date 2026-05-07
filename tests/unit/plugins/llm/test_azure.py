# tests/plugins/llm/test_azure.py
"""Tests for Azure OpenAI LLM provider via unified LLMTransform."""

from collections.abc import Generator
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import Determinism, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.engine.batch_adapter import ExceptionResult
from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort
from elspeth.plugins.infrastructure.clients.llm import RateLimitError
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

from .conftest import chaosllm_azure_openai_client

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_pipeline_row({}),
    )


def _make_azure_config(**overrides: object) -> dict[str, object]:
    """Build a standard Azure LLMTransform config dict with overrides."""
    config: dict[str, object] = {
        "provider": "azure",
        "deployment_name": "my-gpt4o-deployment",
        "endpoint": "https://my-resource.openai.azure.com",
        "api_key": "azure-api-key",
        "template": "{{ row.text }}",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],
    }
    config.update(overrides)
    return config


class TestAzureOpenAIConfig:
    """Tests for AzureOpenAIConfig validation."""

    def test_config_requires_deployment_name(self) -> None:
        """AzureOpenAIConfig requires deployment_name."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )  # Missing 'deployment_name'

    def test_config_requires_endpoint(self) -> None:
        """AzureOpenAIConfig requires endpoint."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )  # Missing 'endpoint'

    def test_config_requires_api_key(self) -> None:
        """AzureOpenAIConfig requires API key."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )  # Missing 'api_key'

    def test_config_requires_template(self) -> None:
        """AzureOpenAIConfig requires template (from LLMConfig)."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'template'

    def test_config_requires_schema(self) -> None:
        """AzureOpenAIConfig requires schema (from TransformDataConfig)."""
        with pytest.raises(PluginConfigError, match="schema"):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                }
            )  # Missing 'schema'

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        assert config.deployment_name == "my-gpt4o-deployment"
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.api_key == "azure-api-key"
        assert config.template == "Analyze: {{ row.text }}"

    def test_default_api_version(self) -> None:
        """Config has default api_version of 2024-10-21."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        assert config.api_version == "2024-10-21"

    def test_custom_api_version(self) -> None:
        """Config accepts custom api_version."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "api_version": "2023-12-01-preview",
            }
        )
        assert config.api_version == "2023-12-01-preview"

    def test_config_inherits_llm_config_defaults(self) -> None:
        """Config inherits defaults from LLMConfig."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        # Inherited from LLMConfig
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.response_field == "llm_response"


class TestLLMTransformAzureInit:
    """Tests for LLMTransform initialization with Azure provider."""

    def test_transform_name(self) -> None:
        """Transform has correct name."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        assert transform.name == "llm"

    def test_transform_stores_azure_config(self) -> None:
        """Transform stores Azure-specific config accessible via _config."""
        transform = LLMTransform(
            _make_azure_config(
                template="{{ row.text }}",
                api_version="2023-12-01-preview",
            )
        )
        assert isinstance(transform._config, AzureOpenAIConfig)
        assert transform._config.endpoint == "https://my-resource.openai.azure.com"
        assert transform._config.api_key == "azure-api-key"
        assert transform._config.api_version == "2023-12-01-preview"
        assert transform._config.deployment_name == "my-gpt4o-deployment"

    def test_model_set_to_deployment_name(self) -> None:
        """Model is set to deployment_name for API calls."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        assert transform._model == "my-gpt4o-deployment"

    def test_determinism_is_non_deterministic(self) -> None:
        """Azure transforms are marked as non-deterministic."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_config_validation_failure_deployment_name(self) -> None:
        """Missing deployment_name raises error via LLMTransform."""
        with pytest.raises((PluginConfigError, ValueError)):
            LLMTransform(
                {
                    "provider": "azure",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],
                }
            )

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        factory = make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())

        with pytest.raises(NotImplementedError, match="row-level pipelining"):
            transform.process(make_pipeline_row({"text": "hello"}), ctx)

    def test_declared_output_fields_populated(self) -> None:
        """Regression: LLMTransform must declare output fields for collision detection.

        Before centralized collision enforcement, AzureLLMTransform had NO
        collision check. This test verifies declared_output_fields is populated
        so TransformExecutor can enforce collision detection.
        """
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))

        assert isinstance(transform.declared_output_fields, frozenset)
        assert len(transform.declared_output_fields) > 0
        assert "llm_response" in transform.declared_output_fields
        assert "llm_response_model" in transform.declared_output_fields


class TestLLMTransformAzurePipelining:
    """Tests for LLMTransform with Azure provider using row-level pipelining.

    These tests verify the accept() API that uses BatchTransformMixin
    for concurrent row processing with FIFO output ordering.
    """

    @pytest.fixture
    def mock_factory(self) -> Mock:
        """Create mock RecorderFactory."""
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
        return make_context(state_id="test-state-id", token=token, landscape=mock_factory)

    @pytest.fixture
    def transform(self, collector: CollectorOutputPort, mock_factory: Mock) -> Generator[LLMTransform, None, None]:
        """Create and initialize LLMTransform with Azure provider and pipelining."""
        t = LLMTransform(_make_azure_config(template="Analyze: {{ row.text }}"))
        # Initialize with factory reference
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        t.on_start(init_ctx)
        # Connect output port
        t.connect_output(collector, max_pending=10)
        yield t
        # Cleanup
        t.close()

    def test_successful_llm_call_emits_enriched_row(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Successful LLM call emits row with response to output port."""
        with chaosllm_azure_openai_client(
            chaosllm_server,
            mode="template",
            template_override="The analysis is positive.",
            usage_override={"prompt_tokens": 10, "completion_tokens": 25},
        ):
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
        assert result.row["llm_response_model"] == "my-gpt4o-deployment"
        # Original data preserved
        assert result.row["text"] == "hello world"

    def test_model_passed_to_azure_client_is_deployment_name(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """deployment_name is used as model in Azure client calls."""
        with chaosllm_azure_openai_client(chaosllm_server) as mock_client:
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "my-gpt4o-deployment"

    def test_template_rendering_error_emits_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Template rendering failure emits TransformResult.error()."""
        # Missing required 'text' field triggers template error
        with chaosllm_azure_openai_client(chaosllm_server):
            transform.accept(make_pipeline_row({"other_field": "value"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]

        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_llm_client_error_emits_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """LLM client failure emits TransformResult.error()."""
        with chaosllm_azure_openai_client(chaosllm_server, side_effect=Exception("API Error")):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]

        assert isinstance(result, TransformResult)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"
        assert "API Error" in result.reason["error"]
        assert result.retryable is False

    def test_rate_limit_error_propagates_for_engine_retry(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Rate limit errors propagate as exceptions for engine retry.

        Retryable errors (RateLimitError, NetworkError, ServerError) are re-raised
        rather than converted to TransformResult.error(). This allows the engine's
        RetryManager to handle retries with proper backoff.

        BatchTransformMixin wraps such exceptions in ExceptionResult for
        propagation through the async pattern. In production, TransformExecutor
        would re-raise this exception so RetryManager can act on it.
        """
        with chaosllm_azure_openai_client(chaosllm_server, side_effect=Exception("Rate limit exceeded 429")):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]

        # Exception propagates via ExceptionResult wrapper (not TransformResult)
        # This allows the engine's RetryManager to retry the operation
        assert isinstance(result, ExceptionResult)
        assert isinstance(result.exception, RateLimitError)
        assert "429" in str(result.exception)

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

    def test_process_row_missing_token_raises_runtime_error(self, mock_factory: Mock, transform: LLMTransform) -> None:
        """Direct _process_row call with missing token must crash explicitly."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_factory,
            state_id="test-state-id",
            token=None,
        )

        with pytest.raises(RuntimeError, match=r"ctx\.token"):
            transform._process_row(make_pipeline_row({"text": "hello"}), ctx)

    def test_system_prompt_included_in_messages(
        self,
        mock_factory: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """System prompt is included when configured."""
        transform = LLMTransform(
            _make_azure_config(
                template="{{ row.text }}",
                system_prompt="You are a helpful assistant.",
            )
        )
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(state_id="test-state-id", token=token, landscape=mock_factory)

        try:
            with chaosllm_azure_openai_client(chaosllm_server) as mock_client:
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)

                call_args = mock_client.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                assert len(messages) == 2
                assert messages[0]["role"] == "system"
                assert messages[0]["content"] == "You are a helpful assistant."
                assert messages[1]["role"] == "user"
                assert messages[1]["content"] == "hello"
        finally:
            transform.close()

    def test_temperature_and_max_tokens_passed_to_client(
        self,
        mock_factory: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Temperature and max_tokens are passed to Azure client."""
        transform = LLMTransform(
            _make_azure_config(
                template="{{ row.text }}",
                temperature=0.7,
                max_tokens=500,
            )
        )
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(state_id="test-state-id", token=token, landscape=mock_factory)

        try:
            with chaosllm_azure_openai_client(chaosllm_server) as mock_client:
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)

                call_args = mock_client.chat.completions.create.call_args
                assert call_args.kwargs["model"] == "my-gpt4o-deployment"
                assert call_args.kwargs["temperature"] == 0.7
                assert call_args.kwargs["max_tokens"] == 500
        finally:
            transform.close()

    def test_custom_response_field(
        self,
        mock_factory: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Custom response_field name is used."""
        transform = LLMTransform(
            _make_azure_config(
                template="{{ row.text }}",
                response_field="analysis",
            )
        )
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(state_id="test-state-id", token=token, landscape=mock_factory)

        try:
            with chaosllm_azure_openai_client(
                chaosllm_server,
                mode="template",
                template_override="Result",
            ):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]

        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["analysis"] == "Result"
        assert "analysis_usage" in result.row
        assert result.success_reason is not None
        assert "analysis_template_hash" in result.success_reason["metadata"]
        assert "analysis_variables_hash" in result.success_reason["metadata"]
        assert "analysis_model" in result.row

    def test_connect_output_required_before_accept(self) -> None:
        """accept() raises RuntimeError if connect_output() not called."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))

        token = make_token("row-1")
        factory = make_factory()
        ctx = make_context(
            landscape=factory.plugin_audit_writer(),
            state_id="test-state-id",
            token=token,
        )

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)

    def test_connect_output_cannot_be_called_twice(self, collector: CollectorOutputPort, mock_factory: Mock) -> None:
        """connect_output() raises if called more than once."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            with pytest.raises(RuntimeError, match="already called"):
                transform.connect_output(collector, max_pending=10)
        finally:
            transform.close()

    def test_azure_client_created_with_correct_credentials(
        self, ctx: PluginContext, transform: LLMTransform, collector: CollectorOutputPort
    ) -> None:
        """AzureOpenAI client is created with correct credentials."""
        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_usage = Mock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 5
            mock_message = Mock()
            mock_message.content = "Response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.model = "my-gpt4o-deployment"
            mock_response.usage = mock_usage
            mock_response.model_dump = Mock(return_value={})
            mock_client.chat.completions.create.return_value = mock_response
            mock_azure_class.return_value = mock_client

            transform.accept(make_pipeline_row({"text": "hello"}), ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Verify AzureOpenAI was called with correct args
            mock_azure_class.assert_called_once_with(
                azure_endpoint="https://my-resource.openai.azure.com",
                api_key="azure-api-key",
                api_version="2024-10-21",
            )


class TestLLMTransformAzureIntegration:
    """Integration-style tests for Azure-specific edge cases via LLMTransform."""

    @pytest.fixture
    def mock_factory(self) -> Mock:
        """Create mock RecorderFactory."""
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_azure_config_with_default_api_version(self) -> None:
        """LLMTransform with Azure provider uses default api_version when not specified."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        assert isinstance(transform._config, AzureOpenAIConfig)
        assert transform._config.api_version == "2024-10-21"

    def test_complex_template_with_multiple_variables(
        self,
        mock_factory: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Complex template with multiple variables works correctly."""
        transform = LLMTransform(
            _make_azure_config(
                template="""
                    Analyze the following data:
                    Name: {{ row.name }}
                    Score: {{ row.score }}
                    Category: {{ row.category }}

                    Provide a summary.
                """,
            )
        )
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(state_id="test-state-id", token=token, landscape=mock_factory)

        try:
            with chaosllm_azure_openai_client(chaosllm_server) as mock_client:
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
                call_args = mock_client.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                user_message = messages[0]["content"]
                assert "Test Item" in user_message
                assert "95" in user_message
                assert "A" in user_message
        finally:
            transform.close()

    def test_calls_are_recorded_to_landscape(
        self,
        mock_factory: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """LLM calls are recorded via AuditedLLMClient."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        token = make_token("row-1")
        ctx = make_context(state_id="test-state-id", token=token, landscape=mock_factory)

        try:
            with chaosllm_azure_openai_client(chaosllm_server):
                transform.accept(make_pipeline_row({"text": "hello"}), ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        # Verify record_call was called (by AuditedLLMClient)
        assert mock_factory.record_call.called


class TestLLMTransformAzureConcurrency:
    """Tests for concurrent row processing via BatchTransformMixin with Azure provider."""

    @pytest.fixture
    def mock_factory(self) -> Mock:
        """Create mock RecorderFactory."""
        factory = Mock()
        factory.record_call = Mock()
        return factory

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_multiple_rows_processed_in_fifo_order(
        self,
        mock_factory: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Multiple rows are emitted in submission order (FIFO)."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        rows = [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ]

        try:
            with chaosllm_azure_openai_client(chaosllm_server):
                for i, row in enumerate(rows):
                    token = make_token(f"row-{i}")
                    ctx = make_context(
                        run_id="test-run",
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
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))

        # Verify _recorder starts as None
        assert transform._recorder is None

        ctx = make_context(state_id="test-state-id", landscape=mock_factory)
        transform.on_start(ctx)

        # Verify factory was captured
        assert transform._recorder is mock_factory

    def test_close_clears_recorder(self, mock_factory: Mock, collector: CollectorOutputPort) -> None:
        """close() clears factory reference."""
        transform = LLMTransform(_make_azure_config(template="{{ row.text }}"))
        init_ctx = make_context(run_id="test", landscape=mock_factory)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        assert transform._recorder is not None

        transform.close()

        assert transform._recorder is None
