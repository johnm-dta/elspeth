"""Tests for OpenRouter Multi-Query LLM transform with row-level pipelining."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import Mock, patch

import httpx
import pytest

from elspeth.contracts import Determinism, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.batching.ports import CollectorOutputPort
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.llm.openrouter_multi_query import OpenRouterMultiQueryLLMTransform
from elspeth.testing import make_pipeline_row

from .conftest import chaosllm_openrouter_http_responses

# Common schema config
DYNAMIC_SCHEMA = {"mode": "observed"}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Create valid config with optional overrides."""
    config = {
        "model": "anthropic/claude-3-opus",
        "api_key": "test-key",
        "template": "Input: {{ row.input_1 }}\nCriterion: {{ row.criterion.name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "case_studies": [
            {"name": "cs1", "input_fields": ["cs1_bg", "cs1_sym", "cs1_hist"]},
            {"name": "cs2", "input_fields": ["cs2_bg", "cs2_sym", "cs2_hist"]},
        ],
        "criteria": [
            {"name": "diagnosis", "code": "DIAG"},
            {"name": "treatment", "code": "TREAT"},
        ],
        "response_format": "standard",
        "output_mapping": {
            "score": {"suffix": "score", "type": "integer"},
            "rationale": {"suffix": "rationale", "type": "string"},
        },
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out for this test
        "pool_size": 4,
    }
    config.update(overrides)
    return config


def make_openrouter_response(content: dict[str, Any] | str) -> dict[str, Any] | str:
    """Create an OpenRouter message content payload."""
    return content


@contextmanager
def mock_openrouter_http_responses(
    chaosllm_server,
    responses: list[dict[str, Any] | str] | list[dict[str, Any] | str | httpx.Response],
) -> Iterator[Mock]:
    """Mock HTTP client to return ChaosLLM-generated responses."""
    # Cast to the expected type for chaosllm_openrouter_http_responses
    typed_responses = cast(list[dict[str, Any] | str | httpx.Response], responses)
    with chaosllm_openrouter_http_responses(chaosllm_server, typed_responses) as mock_client:
        yield mock_client


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_pipeline_row({}),
    )


def make_plugin_context(
    state_id: str = "state-123",
    token: TokenInfo | None = None,
) -> PluginContext:
    """Create a PluginContext with mocked landscape."""
    mock_landscape = Mock()
    mock_landscape.record_external_call = Mock()
    mock_landscape.record_call = Mock()
    if token is None:
        token = make_token("row-1")
    return PluginContext(
        run_id="run-123",
        landscape=mock_landscape,
        state_id=state_id,
        config={},
        token=token,
    )


class TestOpenRouterMultiQueryLLMTransformInit:
    """Tests for transform initialization."""

    def test_transform_has_correct_name(self) -> None:
        """Transform registers with correct plugin name."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        assert transform.name == "openrouter_multi_query_llm"

    def test_transform_is_non_deterministic(self) -> None:
        """LLM transforms are non-deterministic."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_transform_expands_queries_on_init(self) -> None:
        """Transform pre-computes query specs on initialization."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        # 2 case studies x 2 criteria = 4 queries
        assert len(transform._query_specs) == 4

    def test_transform_requires_case_studies(self) -> None:
        """Transform requires case_studies in config."""
        config = make_config()
        del config["case_studies"]
        with pytest.raises(PluginConfigError):
            OpenRouterMultiQueryLLMTransform(config)

    def test_transform_requires_criteria(self) -> None:
        """Transform requires criteria in config."""
        config = make_config()
        del config["criteria"]
        with pytest.raises(PluginConfigError):
            OpenRouterMultiQueryLLMTransform(config)

    def test_transform_requires_output_mapping(self) -> None:
        """Transform requires output_mapping in config."""
        config = make_config()
        del config["output_mapping"]
        with pytest.raises(PluginConfigError):
            OpenRouterMultiQueryLLMTransform(config)

    def test_transform_requires_non_empty_output_mapping(self) -> None:
        """Transform requires non-empty output_mapping."""
        config = make_config(output_mapping={})
        with pytest.raises(PluginConfigError):
            OpenRouterMultiQueryLLMTransform(config)

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        ctx = make_plugin_context()

        with pytest.raises(NotImplementedError, match="row-level pipelining"):
            transform.process(make_pipeline_row({"text": "hello"}), ctx)


class TestSingleQueryProcessing:
    """Tests for _process_single_query method."""

    def test_process_single_query_renders_template(self, chaosllm_server) -> None:
        """Single query renders template with input fields and criterion."""
        responses = [make_openrouter_response({"score": 85, "rationale": "Good diagnosis"})]

        with mock_openrouter_http_responses(chaosllm_server, responses) as mock_client:
            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            # Call on_start to set up the recorder
            transform.on_start(ctx)

            row = {
                "cs1_bg": "45yo male",
                "cs1_sym": "chest pain",
                "cs1_hist": "family history",
            }
            spec = transform._query_specs[0]  # cs1_diagnosis

            assert ctx.state_id is not None
            transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            # Check HTTP was called
            assert mock_client.post.call_count == 1
            call_args = mock_client.post.call_args

            # Verify JSON body contains correct model and messages
            request_body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert request_body["model"] == "anthropic/claude-3-opus"
            messages = request_body["messages"]
            user_message = messages[-1]["content"]

            assert "45yo male" in user_message
            assert "diagnosis" in user_message.lower()

    def test_process_single_query_parses_json_response(self, chaosllm_server) -> None:
        """Single query parses JSON and returns mapped fields."""
        responses = [make_openrouter_response({"score": 85, "rationale": "Excellent assessment"})]

        with mock_openrouter_http_responses(chaosllm_server, responses):
            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]  # cs1_diagnosis

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert result.status == "success"
            assert result.row is not None
            # Output fields use prefix from spec
            assert result.row["cs1_diagnosis_score"] == 85
            assert result.row["cs1_diagnosis_rationale"] == "Excellent assessment"

    def test_process_single_query_handles_invalid_json(self, chaosllm_server) -> None:
        """Single query returns error on invalid JSON response from LLM content."""
        # LLM returns valid HTTP JSON but content is not JSON
        responses = [make_openrouter_response("not json")]

        with mock_openrouter_http_responses(chaosllm_server, responses):
            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert result.status == "error"
            assert result.reason is not None
            assert "json" in result.reason["reason"].lower()

    def test_process_single_query_raises_capacity_error_on_rate_limit(self) -> None:
        """Rate limit errors (HTTP 429) are converted to CapacityError for pooled retry."""
        from elspeth.plugins.pooling import CapacityError

        # Mock HTTP client to return 429
        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 429
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = b""
            mock_response.text = ""
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Rate limit exceeded",
                request=Mock(),
                response=mock_response,
            )
            mock_client.post.return_value = mock_response

            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            with pytest.raises(CapacityError) as exc_info:
                transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert exc_info.value.status_code == 429

    def test_process_single_query_raises_capacity_error_on_503(self) -> None:
        """Service unavailable (HTTP 503) raises CapacityError for pooled retry."""
        from elspeth.plugins.pooling import CapacityError

        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 503
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = b""
            mock_response.text = ""
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Service unavailable",
                request=Mock(),
                response=mock_response,
            )
            mock_client.post.return_value = mock_response

            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            with pytest.raises(CapacityError) as exc_info:
                transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert exc_info.value.status_code == 503

    def test_process_single_query_network_error_is_retryable(self) -> None:
        """Network errors (httpx.RequestError) should be retryable."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")

            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "api_call_failed"
            assert result.retryable is True

    def test_process_single_query_server_error_is_retryable(self) -> None:
        """HTTP 5xx server errors should be retryable (transient)."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 500
            mock_response.headers = {"content-type": "text/html"}
            mock_response.content = b""
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Internal Server Error",
                request=Mock(),
                response=mock_response,
            )
            mock_client.post.return_value = mock_response

            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "api_call_failed"
            assert result.reason["status_code"] == 500
            assert result.retryable is True

    def test_process_single_query_client_error_not_retryable(self) -> None:
        """HTTP 4xx client errors (non-capacity) should NOT be retryable."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 400
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = b""
            mock_response.text = "Bad Request"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Bad Request",
                request=Mock(),
                response=mock_response,
            )
            mock_client.post.return_value = mock_response

            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "api_call_failed"
            assert result.reason["status_code"] == 400
            assert result.retryable is False

    def test_process_single_query_handles_template_error(self, chaosllm_server) -> None:
        """Template rendering errors return error result with details."""
        from elspeth.plugins.llm.templates import TemplateError

        responses = [make_openrouter_response({"score": 85, "rationale": "ok"})]

        with mock_openrouter_http_responses(chaosllm_server, responses):
            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            # Mock template to raise error
            with patch.object(transform._template, "render_with_metadata") as mock_render:
                mock_render.side_effect = TemplateError("Undefined variable 'missing'")

                assert ctx.state_id is not None
                result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

                assert result.status == "error"
                assert result.reason is not None
                assert result.reason["reason"] == "template_rendering_failed"
                assert "missing" in result.reason["error"]

    def test_get_http_client_requires_recorder(self) -> None:
        """_get_http_client raises RuntimeError if recorder not set via on_start."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        # Don't call on_start - recorder will be None

        with pytest.raises(RuntimeError) as exc_info:
            transform._get_http_client("state-123")

        assert "recorder" in str(exc_info.value).lower()

    def test_process_single_query_strips_markdown_code_blocks(self, chaosllm_server) -> None:
        """LLM responses wrapped in markdown code blocks are handled correctly."""
        # LLM returns JSON wrapped in ```json ... ```
        content_with_fence = '```json\n{"score": 90, "rationale": "Great"}\n```'
        responses = [make_openrouter_response(content_with_fence)]

        with mock_openrouter_http_responses(chaosllm_server, responses):
            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert result.status == "success"
            assert result.row is not None
            assert result.row["cs1_diagnosis_score"] == 90
            assert result.row["cs1_diagnosis_rationale"] == "Great"

    def test_process_single_query_validates_json_is_dict(self, chaosllm_server) -> None:
        """LLM JSON response must be an object, not array or primitive."""
        # Valid JSON but not an object
        responses = [make_openrouter_response("[1, 2, 3]")]

        with mock_openrouter_http_responses(chaosllm_server, responses):
            transform = OpenRouterMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id, "test-token-id", None)

            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "invalid_json_type"
            assert result.reason["expected"] == "object"
            assert result.reason["actual"] == "list"


class TestRowProcessingWithPipelining:
    """Tests for full row processing via accept() API."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
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
    def transform(self, collector: CollectorOutputPort, mock_recorder: Mock) -> Generator[OpenRouterMultiQueryLLMTransform, None, None]:
        """Create and initialize transform with pipelining."""
        t = OpenRouterMultiQueryLLMTransform(make_config())
        # Initialize with recorder reference
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        t.on_start(init_ctx)
        # Connect output port
        t.connect_output(collector, max_pending=10)
        yield t
        # Cleanup
        t.close()

    def test_process_row_executes_all_queries(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Process executes all (case_study x criterion) queries."""
        # 2 case studies x 2 criteria = 4 queries
        responses = [
            make_openrouter_response({"score": 85, "rationale": "CS1 diagnosis"}),
            make_openrouter_response({"score": 90, "rationale": "CS1 treatment"}),
            make_openrouter_response({"score": 75, "rationale": "CS2 diagnosis"}),
            make_openrouter_response({"score": 80, "rationale": "CS2 treatment"}),
        ]

        with mock_openrouter_http_responses(chaosllm_server, responses) as mock_client:
            row = {
                "cs1_bg": "case1 bg",
                "cs1_sym": "case1 sym",
                "cs1_hist": "case1 hist",
                "cs2_bg": "case2 bg",
                "cs2_sym": "case2 sym",
                "cs2_hist": "case2 hist",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"

            assert result.status == "success"
            assert mock_client.post.call_count == 4

    def test_process_row_merges_all_results(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """All query results are merged into single output row."""
        responses = [
            make_openrouter_response({"score": 85, "rationale": "R1"}),
            make_openrouter_response({"score": 90, "rationale": "R2"}),
            make_openrouter_response({"score": 75, "rationale": "R3"}),
            make_openrouter_response({"score": 80, "rationale": "R4"}),
        ]

        with mock_openrouter_http_responses(chaosllm_server, responses):
            row = {
                "cs1_bg": "bg1",
                "cs1_sym": "sym1",
                "cs1_hist": "hist1",
                "cs2_bg": "bg2",
                "cs2_sym": "sym2",
                "cs2_hist": "hist2",
                "original_field": "preserved",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"

            assert result.status == "success"
            assert result.row is not None
            output = result.row

            # Original fields preserved
            assert output["original_field"] == "preserved"

            # All 4 queries produced output (2 fields each = 8 assessment fields)
            assert "cs1_diagnosis_score" in output
            assert "cs1_diagnosis_rationale" in output
            assert "cs1_treatment_score" in output
            assert "cs2_diagnosis_score" in output
            assert "cs2_treatment_score" in output

    def test_process_row_supports_original_header_names_in_input_fields(
        self,
        ctx: PluginContext,
        collector: CollectorOutputPort,
        mock_recorder: Mock,
        chaosllm_server,
    ) -> None:
        """Original source headers in input_fields resolve via PipelineRow contract."""
        config = make_config(
            case_studies=[
                {"name": "cs1", "input_fields": ["Patient Name", "Symptoms", "History"]},
            ],
            criteria=[{"name": "diagnosis", "code": "DIAG"}],
            pool_size=1,
        )
        transform = OpenRouterMultiQueryLLMTransform(config)
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract("patient_name", "Patient Name", str, False, "inferred"),
                FieldContract("symptoms", "Symptoms", str, False, "inferred"),
                FieldContract("history", "History", str, False, "inferred"),
            ),
            locked=True,
        )
        row = PipelineRow(
            {
                "patient_name": "Alice Smith",
                "symptoms": "chest pain",
                "history": "family history",
            },
            contract,
        )

        try:
            with mock_openrouter_http_responses(
                chaosllm_server,
                [make_openrouter_response({"score": 85, "rationale": "Looks consistent"})],
            ) as mock_client:
                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "success"
            assert result.row is not None
            assert result.row["cs1_diagnosis_score"] == 85
            assert result.row["patient_name"] == "Alice Smith"

            call_args = mock_client.post.call_args
            request_body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert request_body is not None
            messages = request_body["messages"]
            user_message = messages[-1]["content"]
            assert "Alice Smith" in user_message
            assert "diagnosis" in user_message.lower()
        finally:
            transform.close()

    def test_process_row_fails_if_any_query_fails(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """All-or-nothing: if any query fails, entire row fails."""
        # First 3 succeed, 4th returns invalid JSON
        call_count = [0]

        def make_response(*args: Any, **kwargs: Any) -> Mock:
            call_count[0] += 1
            if call_count[0] == 4:
                content = "not valid json"
            else:
                content = json.dumps({"score": 85, "rationale": "ok"})

            response_data = {
                "choices": [{"message": {"content": content, "role": "assistant"}}],
                "model": "anthropic/claude-3-opus",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = response_data
            mock_response.text = json.dumps(response_data)
            mock_response.content = b""
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.post.side_effect = make_response

            row = {
                "cs1_bg": "bg",
                "cs1_sym": "sym",
                "cs1_hist": "hist",
                "cs2_bg": "bg",
                "cs2_sym": "sym",
                "cs2_hist": "hist",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Entire row fails
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "error"
            assert result.reason is not None
            assert "query_failed" in result.reason["reason"]

    def test_process_row_includes_metadata_in_output(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Each query result includes audit metadata (usage, model, template_hash)."""
        responses = [
            make_openrouter_response({"score": 85, "rationale": "R1"}),
            make_openrouter_response({"score": 90, "rationale": "R2"}),
            make_openrouter_response({"score": 75, "rationale": "R3"}),
            make_openrouter_response({"score": 80, "rationale": "R4"}),
        ]

        with mock_openrouter_http_responses(chaosllm_server, responses):
            row = {
                "cs1_bg": "bg",
                "cs1_sym": "sym",
                "cs1_hist": "hist",
                "cs2_bg": "bg",
                "cs2_sym": "sym",
                "cs2_hist": "hist",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"

            assert result.status == "success"
            assert result.row is not None
            output = result.row

            # Metadata fields present for first query
            assert "cs1_diagnosis_usage" in output
            assert "cs1_diagnosis_model" in output
            assert "cs1_diagnosis_template_hash" in output
            assert "cs1_diagnosis_variables_hash" in output


class TestMultiRowPipelining:
    """Tests for processing multiple rows via pipelining API."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_multiple_rows_processed_in_fifo_order(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Multiple rows are emitted in submission order (FIFO).

        Uses sequential execution (no pool_size) to avoid mock threading issues.
        The transform fixture uses pool_size=4 by default, but concurrent threads
        can race on the mock counter causing response mismatches.
        """
        # Create config without pool_size for sequential execution
        # This tests FIFO ordering without concurrent HTTP mock issues
        config = make_config()
        del config["pool_size"]

        transform = OpenRouterMultiQueryLLMTransform(config)
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        rows = [
            {
                "cs1_bg": "r1",
                "cs1_sym": "r1",
                "cs1_hist": "r1",
                "cs2_bg": "r1",
                "cs2_sym": "r1",
                "cs2_hist": "r1",
                "marker": "first",
            },
            {
                "cs1_bg": "r2",
                "cs1_sym": "r2",
                "cs1_hist": "r2",
                "cs2_bg": "r2",
                "cs2_sym": "r2",
                "cs2_hist": "r2",
                "marker": "second",
            },
            {
                "cs1_bg": "r3",
                "cs1_sym": "r3",
                "cs1_hist": "r3",
                "cs2_bg": "r3",
                "cs2_sym": "r3",
                "cs2_hist": "r3",
                "marker": "third",
            },
        ]

        # 3 rows x 4 queries = 12 total responses needed
        responses = [make_openrouter_response({"score": i, "rationale": f"R{i}"}) for i in range(12)]

        try:
            with mock_openrouter_http_responses(chaosllm_server, responses):
                for i, row in enumerate(rows):
                    token = make_token(f"row-{i}")
                    ctx = PluginContext(
                        run_id="test-run",
                        config={},
                        landscape=mock_recorder,
                        state_id=f"state-{i}",
                        token=token,
                    )
                    transform.accept(make_pipeline_row(row), ctx)

                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        # Results should be in FIFO order
        assert len(collector.results) == 3
        for i, (_token, result, _state_id) in enumerate(collector.results):
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "success"
            assert result.row is not None
            assert result.row["marker"] == rows[i]["marker"]

    def test_connect_output_required_before_accept(self) -> None:
        """accept() raises RuntimeError if connect_output() not called."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            state_id="test-state-id",
            token=token,
        )

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)

    def test_connect_output_cannot_be_called_twice(self, collector: CollectorOutputPort, mock_recorder: Mock) -> None:
        """connect_output() raises if called more than once."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            with pytest.raises(RuntimeError, match="already called"):
                transform.connect_output(collector, max_pending=10)
        finally:
            transform.close()


class TestHTTPSpecificBehavior:
    """Tests specific to HTTP-based implementation (vs SDK-based Azure)."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
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
    def transform(self, collector: CollectorOutputPort, mock_recorder: Mock) -> Generator[OpenRouterMultiQueryLLMTransform, None, None]:
        """Create and initialize transform with pipelining."""
        t = OpenRouterMultiQueryLLMTransform(make_config())
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        t.on_start(init_ctx)
        t.connect_output(collector, max_pending=10)
        yield t
        t.close()

    def test_handles_non_json_http_response(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """HTTP errors with non-JSON body are handled gracefully."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 500
            mock_response.headers = {"content-type": "text/html"}
            mock_response.text = "<html>Internal Server Error</html>"
            mock_response.content = b""
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Internal Server Error",
                request=Mock(),
                response=mock_response,
            )
            mock_client.post.return_value = mock_response

            row = {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Should return error result, not raise exception
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "error"
            assert result.reason is not None
            # The error cascades through query_failed since one query fails
            assert "query_failed" in result.reason["reason"] or "api_call_failed" in str(result.reason.get("failed_queries", []))

    def test_handles_empty_choices_array(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Empty choices array in response returns appropriate error."""
        response_data = {
            "choices": [],  # Empty choices
            "model": "anthropic/claude-3-opus",
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = response_data
            mock_response.text = json.dumps(response_data)
            mock_response.content = b""
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response

            row = {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "error"
            assert result.reason is not None
            # The error cascades through query_failed
            assert "query_failed" in result.reason["reason"]

    def test_handles_null_content_from_content_filtering(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Null content (content filtering) returns error instead of crashing.

        P0-05: When OpenRouter returns null content due to content filtering,
        content.strip() threw AttributeError: 'NoneType' has no attribute 'strip'.
        Must return TransformResult.error() with reason 'content_filtered'.
        """
        response_data = {
            "choices": [{"message": {"content": None, "role": "assistant"}}],
            "model": "anthropic/claude-3-opus",
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = response_data
            mock_response.text = json.dumps(response_data)
            mock_response.content = b""
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response

            row = {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "error"
            assert result.reason is not None
            # Multi-query wraps per-query errors in a query_failed envelope
            assert result.reason["reason"] == "query_failed"
            # All queries should have failed with content_filtered
            assert result.reason["succeeded_count"] == 0
            for failed_query in result.reason["failed_queries"]:
                # failed_query can be str | QueryFailureDetail - we check both forms
                if isinstance(failed_query, dict):
                    assert "content-filtered" in failed_query["error"]
                else:
                    assert "content-filtered" in failed_query

    def test_handles_missing_output_field_in_json(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Missing expected field in LLM JSON response returns appropriate error."""
        # Response missing 'rationale' field that output_mapping expects
        responses = [make_openrouter_response({"score": 85})]  # Missing 'rationale'

        with mock_openrouter_http_responses(chaosllm_server, responses):
            row = {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "error"
            assert result.reason is not None
            # The error cascades through query_failed
            assert "query_failed" in result.reason["reason"]

    def test_handles_connection_error(
        self,
        ctx: PluginContext,
        transform: OpenRouterMultiQueryLLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Network connection errors are handled gracefully."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")

            row = {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }

            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "error"
            assert result.reason is not None
            # The error cascades through query_failed
            assert "query_failed" in result.reason["reason"]


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    def test_close_shuts_down_executor(self) -> None:
        """close() shuts down the pooled executor."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())

        # Mock the executor
        mock_executor = Mock()
        transform._executor = mock_executor

        transform.close()

        mock_executor.shutdown.assert_called_once_with(wait=True)

    def test_close_clears_http_clients(self) -> None:
        """close() clears all cached HTTP clients."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())

        # Add some mock clients
        transform._http_clients["state-1"] = Mock()
        transform._http_clients["state-2"] = Mock()

        transform.close()

        assert len(transform._http_clients) == 0

    def test_close_clears_recorder_reference(self) -> None:
        """close() clears the recorder reference."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        transform._recorder = Mock()

        transform.close()

        assert transform._recorder is None

    def test_on_start_captures_recorder(self) -> None:
        """on_start() captures recorder reference for LLM client creation."""
        transform = OpenRouterMultiQueryLLMTransform(make_config())
        mock_recorder = Mock()

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
