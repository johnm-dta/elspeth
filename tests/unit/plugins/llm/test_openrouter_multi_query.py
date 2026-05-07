"""Tests for OpenRouter Multi-Query LLM transform via unified LLMTransform.

Migrated from legacy OpenRouterMultiQueryLLMTransform to use unified
LLMTransform with provider="openrouter" and queries dict config.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort
from elspeth.plugins.infrastructure.clients.llm import (
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMProvider, LLMQueryResult
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

# Common schema config
DYNAMIC_SCHEMA = {"mode": "observed"}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Create valid config with optional overrides.

    Uses the unified LLMTransform config format with provider="openrouter"
    and explicit queries dict instead of case_studies x criteria cross-product.

    The queries dict replicates the old 2 case_studies x 2 criteria = 4 queries:
    cs1_diagnosis, cs1_treatment, cs2_diagnosis, cs2_treatment.
    """
    config: dict[str, Any] = {
        "provider": "openrouter",
        "model": "anthropic/claude-3-opus",
        "api_key": "test-key",
        "template": "Input: {{ row.text_content }}\nCriterion: {{ row.criterion_name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out for this test
        "queries": {
            "cs1_diagnosis": {
                "input_fields": {"text_content": "cs1_bg", "criterion_name": "cs1_sym"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_treatment": {
                "input_fields": {"text_content": "cs1_bg", "criterion_name": "cs1_sym"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_diagnosis": {
                "input_fields": {"text_content": "cs2_bg", "criterion_name": "cs2_sym"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_treatment": {
                "input_fields": {"text_content": "cs2_bg", "criterion_name": "cs2_sym"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
        },
    }
    config.update(overrides)
    return config


def make_query_result(
    content: dict[str, Any] | str,
    *,
    model: str = "anthropic/claude-3-opus",
    usage: TokenUsage | None = None,
    finish_reason: FinishReason | None = FinishReason.STOP,
) -> LLMQueryResult:
    """Create an LLMQueryResult from content (dict→JSON string, or raw string)."""
    if isinstance(content, dict):
        content_str = json.dumps(content)
    else:
        content_str = content
    return LLMQueryResult(
        content=content_str,
        usage=usage or TokenUsage.known(10, 5),
        model=model,
        finish_reason=finish_reason,
    )


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_pipeline_row({}),
    )


def _make_transform_with_mock_provider(
    config: dict[str, Any] | None = None,
) -> tuple[LLMTransform, Mock]:
    """Create an LLMTransform with a mocked provider already set."""
    transform = LLMTransform(config or make_config())
    mock_provider = Mock(spec=LLMProvider)
    transform._provider = mock_provider
    return transform, mock_provider


class TestOpenRouterMultiQueryInit:
    """Tests for transform initialization."""

    def test_transform_has_correct_name(self) -> None:
        """Transform registers with correct plugin name."""
        transform = LLMTransform(make_config())
        assert transform.name == "llm"

    def test_transform_is_non_deterministic(self) -> None:
        """LLM transforms are non-deterministic."""
        transform = LLMTransform(make_config())
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_transform_expands_queries_on_init(self) -> None:
        """Transform pre-computes query specs on initialization."""
        from elspeth.plugins.transforms.llm.transform import MultiQueryStrategy

        transform = LLMTransform(make_config())
        assert isinstance(transform._strategy, MultiQueryStrategy)
        # 4 queries defined explicitly
        assert len(transform._strategy.query_specs) == 4

    def test_transform_requires_queries(self) -> None:
        """Transform with multi-query requires queries in config."""
        config = make_config()
        config["queries"] = {}
        with pytest.raises(ValueError, match="no queries configured"):
            LLMTransform(config)

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        transform = LLMTransform(make_config())
        ctx = make_context()

        with pytest.raises(NotImplementedError, match="row-level pipelining"):
            transform.process(make_pipeline_row({"text": "hello"}), ctx)


class TestSingleQueryProcessing:
    """Tests for _process_row with multi-query strategy via mocked provider.

    Each test verifies behavior of a single query within the multi-query flow
    by configuring a single-query queries dict and mocking the provider.
    """

    def _make_single_query_config(self, **overrides: Any) -> dict[str, Any]:
        """Config with a single query for isolated testing."""
        config: dict[str, Any] = {
            "provider": "openrouter",
            "model": "anthropic/claude-3-opus",
            "api_key": "test-key",
            "template": "Input: {{ row.text_content }}\nCriterion: {{ row.criterion_name }}",
            "system_prompt": "You are an assessment AI. Respond in JSON.",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
            "queries": {
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "cs1_bg", "criterion_name": "cs1_sym"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        }
        config.update(overrides)
        return config

    def test_process_row_renders_template(self) -> None:
        """Single query renders template with input fields."""
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.return_value = make_query_result({"score": 85, "rationale": "Good diagnosis"})

        row = make_pipeline_row(
            {
                "cs1_bg": "45yo male",
                "cs1_sym": "chest pain",
                "cs1_hist": "family history",
            }
        )
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "success"
        # Verify provider was called
        assert mock_provider.execute_query.call_count == 1
        call_kwargs = mock_provider.execute_query.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
        user_message = messages[-1]["content"]
        assert "45yo male" in user_message

    def test_process_row_parses_json_response(self) -> None:
        """Single query parses JSON and returns mapped fields."""
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.return_value = make_query_result({"score": 85, "rationale": "Excellent assessment"})

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        # Output fields use query name prefix
        assert result.row["cs1_diagnosis_score"] == 85
        assert result.row["cs1_diagnosis_rationale"] == "Excellent assessment"

    def test_process_row_handles_invalid_json(self) -> None:
        """Returns error on invalid JSON response from LLM content."""
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.return_value = make_query_result("not json")

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert "json" in result.reason["reason"].lower()

    def test_process_row_rate_limit_returns_retryable_error(self) -> None:
        """Rate limit errors return retryable error result (not raised).

        Multi-query sequential mode catches retryable LLMClientError and returns
        TransformResult.error(retryable=True) to avoid wastefully re-executing
        successful queries on engine retry.
        """
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.side_effect = RateLimitError("Rate limit exceeded")

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)
        assert result.status == "error"
        assert result.retryable is True
        assert result.reason is not None
        assert result.reason["reason"] == "multi_query_failed"

    def test_process_row_server_error_returns_retryable_error(self) -> None:
        """Server errors return retryable error result (not raised).

        Multi-query sequential mode catches retryable LLMClientError and returns
        TransformResult.error(retryable=True) to avoid wastefully re-executing
        successful queries on engine retry.
        """
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.side_effect = ServerError("503 Service Unavailable")

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)
        assert result.status == "error"
        assert result.retryable is True
        assert result.reason is not None
        assert result.reason["reason"] == "multi_query_failed"

    def test_process_row_network_error_returns_retryable_error(self) -> None:
        """Network errors return retryable error result (not raised).

        Multi-query sequential mode catches retryable LLMClientError and returns
        TransformResult.error(retryable=True) to avoid wastefully re-executing
        successful queries on engine retry.
        """
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.side_effect = NetworkError("Connection refused")

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)
        assert result.status == "error"
        assert result.retryable is True
        assert result.reason is not None
        assert result.reason["reason"] == "multi_query_failed"

    def test_process_row_client_error_not_retryable(self) -> None:
        """Non-retryable LLMClientError returns error result."""
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.side_effect = LLMClientError("Bad Request", retryable=False)

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.retryable is False

    def test_process_row_context_length_error(self) -> None:
        """Context length exceeded returns non-retryable error."""
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.side_effect = ContextLengthError("Context too long")

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "context_length_exceeded"
        assert result.retryable is False

    def test_process_row_handles_template_error(self) -> None:
        """Template rendering errors return error result with details.

        The template references {{ row.missing_field }} which is not provided
        by the query's input_fields mapping, so PromptTemplate raises TemplateError
        via StrictUndefined. This tests the error handling path without patching
        the frozen MultiQueryStrategy dataclass.
        """
        # Template references a variable not mapped by input_fields — triggers TemplateError
        config = self._make_single_query_config(template="Input: {{ row.text_content }}\nMissing: {{ row.missing_field }}")
        transform, _mock_provider = _make_transform_with_mock_provider(config)

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "missing" in result.reason["error"]

    def test_on_start_sets_lifecycle_flag(self) -> None:
        """on_start() sets _on_start_called flag for centralized lifecycle guard."""
        transform = LLMTransform(make_config())
        assert not transform._on_start_called

        ctx = Mock()
        ctx.landscape = Mock()
        ctx.run_id = "test-run"
        ctx.telemetry_emit = None
        ctx.rate_limit_registry = None
        transform.on_start(ctx)

        assert transform._on_start_called

    def test_process_row_strips_markdown_code_blocks(self) -> None:
        """LLM responses wrapped in markdown code blocks are handled correctly."""
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        # Markdown-wrapped JSON content
        content_with_fence = '```json\n{"score": 90, "rationale": "Great"}\n```'
        mock_provider.execute_query.return_value = make_query_result(content_with_fence)

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["cs1_diagnosis_score"] == 90
        assert result.row["cs1_diagnosis_rationale"] == "Great"

    def test_process_row_validates_json_is_dict(self) -> None:
        """LLM JSON response must be an object, not array or primitive."""
        config = self._make_single_query_config()
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.return_value = make_query_result("[1, 2, 3]")

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "invalid_json_type"
        assert result.reason["expected"] == "object"
        assert result.reason["actual"] == "list"


class TestRowProcessingWithPipelining:
    """Tests for full row processing via accept() API."""

    @pytest.fixture()
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture()
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    @pytest.fixture()
    def ctx(self, mock_recorder: Mock) -> PluginContext:
        """Create plugin context with landscape, state_id, and token."""
        token = make_token("row-1")
        return make_context(
            run_id="test-run",
            state_id="test-state-id",
            token=token,
            landscape=mock_recorder,
        )

    @pytest.fixture()
    def transform(self, collector: CollectorOutputPort, mock_recorder: Mock) -> Generator[LLMTransform, None, None]:
        """Create and initialize transform with pipelining."""
        t = LLMTransform(make_config())
        # Set up mock provider instead of calling on_start (avoids real provider creation)
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.execute_query.return_value = make_query_result({"score": 85, "rationale": "default"})
        t._provider = mock_provider
        t._recorder = mock_recorder
        t._on_start_called = True
        # Connect output port
        t.connect_output(collector, max_pending=10)
        yield t
        # Cleanup
        t.close()

    def test_process_row_executes_all_queries(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Process executes all 4 queries defined in config."""
        # Set up mock to return different results per call
        call_count = [0]
        responses = [
            make_query_result({"score": 85, "rationale": "CS1 diagnosis"}),
            make_query_result({"score": 90, "rationale": "CS1 treatment"}),
            make_query_result({"score": 75, "rationale": "CS2 diagnosis"}),
            make_query_result({"score": 80, "rationale": "CS2 treatment"}),
        ]

        def side_effect(*args: Any, **kwargs: Any) -> LLMQueryResult:
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx]

        transform._provider.execute_query.side_effect = side_effect  # type: ignore[union-attr]

        row = make_pipeline_row(
            {
                "cs1_bg": "case1 bg",
                "cs1_sym": "case1 sym",
                "cs1_hist": "case1 hist",
                "cs2_bg": "case2 bg",
                "cs2_sym": "case2 sym",
                "cs2_hist": "case2 hist",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"

        assert result.status == "success"
        assert transform._provider.execute_query.call_count == 4  # type: ignore[union-attr]

    def test_process_row_merges_all_results(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """All query results are merged into single output row."""
        call_count = [0]
        responses = [
            make_query_result({"score": 85, "rationale": "R1"}),
            make_query_result({"score": 90, "rationale": "R2"}),
            make_query_result({"score": 75, "rationale": "R3"}),
            make_query_result({"score": 80, "rationale": "R4"}),
        ]

        def side_effect(*args: Any, **kwargs: Any) -> LLMQueryResult:
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx]

        transform._provider.execute_query.side_effect = side_effect  # type: ignore[union-attr]

        row = make_pipeline_row(
            {
                "cs1_bg": "bg1",
                "cs1_sym": "sym1",
                "cs1_hist": "hist1",
                "cs2_bg": "bg2",
                "cs2_sym": "sym2",
                "cs2_hist": "hist2",
                "original_field": "preserved",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"

        assert result.status == "success"
        assert result.row is not None
        output = result.row

        # Original fields preserved
        assert output["original_field"] == "preserved"

        # All 4 queries produced output (score + rationale each)
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
    ) -> None:
        """Original source headers in input_fields resolve via PipelineRow contract."""
        config: dict[str, Any] = {
            "provider": "openrouter",
            "model": "anthropic/claude-3-opus",
            "api_key": "test-key",
            "template": "Input: {{ row.text_content }}",
            "system_prompt": "You are an assessment AI. Respond in JSON.",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
            "queries": {
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "Patient Name"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        }
        transform = LLMTransform(config)
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.execute_query.return_value = make_query_result({"score": 85, "rationale": "Looks consistent"})
        transform._provider = mock_provider
        transform._recorder = mock_recorder
        transform._on_start_called = True
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
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
            assert result.status == "success"
            assert result.row is not None
            assert result.row["cs1_diagnosis_score"] == 85
            assert result.row["patient_name"] == "Alice Smith"

            # Verify provider was called with rendered template containing the data
            call_kwargs = mock_provider.execute_query.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
            user_message = messages[-1]["content"]
            assert "Alice Smith" in user_message
        finally:
            transform.close()

    def test_process_row_fails_if_any_query_fails(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """All-or-nothing: if any query fails, entire row fails."""
        call_count = [0]

        def side_effect(*args: Any, **kwargs: Any) -> LLMQueryResult:
            call_count[0] += 1
            if call_count[0] == 4:
                # 4th query returns non-JSON content that will fail JSON parsing
                return make_query_result("not valid json")
            return make_query_result({"score": 85, "rationale": "ok"})

        transform._provider.execute_query.side_effect = side_effect  # type: ignore[union-attr]

        row = make_pipeline_row(
            {
                "cs1_bg": "bg",
                "cs1_sym": "sym",
                "cs1_hist": "hist",
                "cs2_bg": "bg",
                "cs2_sym": "sym",
                "cs2_hist": "hist",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        # Entire row fails
        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
        assert result.status == "error"
        assert result.reason is not None
        assert "json" in result.reason["reason"].lower()

    def test_process_row_includes_operational_metadata_in_output(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Each query result includes operational metadata in row, audit metadata in success_reason."""
        call_count = [0]
        responses = [
            make_query_result({"score": 85, "rationale": "R1"}),
            make_query_result({"score": 90, "rationale": "R2"}),
            make_query_result({"score": 75, "rationale": "R3"}),
            make_query_result({"score": 80, "rationale": "R4"}),
        ]

        def side_effect(*args: Any, **kwargs: Any) -> LLMQueryResult:
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx]

        transform._provider.execute_query.side_effect = side_effect  # type: ignore[union-attr]

        row = make_pipeline_row(
            {
                "cs1_bg": "bg",
                "cs1_sym": "sym",
                "cs1_hist": "hist",
                "cs2_bg": "bg",
                "cs2_sym": "sym",
                "cs2_hist": "hist",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"

        assert result.status == "success"
        assert result.row is not None
        output = result.row

        # Operational metadata fields present in output row
        assert "cs1_diagnosis_llm_response_usage" in output
        assert "cs1_diagnosis_llm_response_model" in output

        # Audit metadata fields are in success_reason, not the output row
        assert "cs1_diagnosis_llm_response_template_hash" not in output
        assert "cs1_diagnosis_llm_response_variables_hash" not in output

        # Audit metadata is in success_reason["metadata"]
        assert result.success_reason is not None
        metadata = result.success_reason["metadata"]
        assert "cs1_diagnosis_llm_response_template_hash" in metadata
        assert "cs1_diagnosis_llm_response_variables_hash" in metadata


class TestMultiRowPipelining:
    """Tests for processing multiple rows via pipelining API."""

    @pytest.fixture()
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture()
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_multiple_rows_processed_in_fifo_order(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
    ) -> None:
        """Multiple rows are emitted in submission order (FIFO)."""
        config = make_config()
        transform = LLMTransform(config)
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.execute_query.return_value = make_query_result({"score": 85, "rationale": "ok"})
        transform._provider = mock_provider
        transform._recorder = mock_recorder
        transform._on_start_called = True
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

        try:
            for i, row in enumerate(rows):
                token = make_token(f"row-{i}")
                ctx = make_context(
                    run_id="test-run",
                    state_id=f"state-{i}",
                    token=token,
                    landscape=mock_recorder,
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
        transform = LLMTransform(make_config())

        token = make_token("row-1")
        ctx = make_context(
            run_id="test-run",
            state_id="test-state-id",
            token=token,
        )

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)

    def test_connect_output_cannot_be_called_twice(self, collector: CollectorOutputPort, mock_recorder: Mock) -> None:
        """connect_output() raises if called more than once."""
        transform = LLMTransform(make_config())
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            with pytest.raises(RuntimeError, match="already called"):
                transform.connect_output(collector, max_pending=10)
        finally:
            transform.close()


class TestHTTPSpecificBehavior:
    """Tests for HTTP-based error handling via mocked provider exceptions.

    In the unified architecture, HTTP-specific behavior is encapsulated in
    OpenRouterLLMProvider. These tests verify that provider exceptions are
    correctly classified by the transform strategy.
    """

    @pytest.fixture()
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture()
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    @pytest.fixture()
    def ctx(self, mock_recorder: Mock) -> PluginContext:
        """Create plugin context with landscape, state_id, and token."""
        token = make_token("row-1")
        return make_context(
            run_id="test-run",
            state_id="test-state-id",
            token=token,
            landscape=mock_recorder,
        )

    @pytest.fixture()
    def transform(self, collector: CollectorOutputPort, mock_recorder: Mock) -> Generator[LLMTransform, None, None]:
        """Create and initialize transform with pipelining."""
        t = LLMTransform(make_config())
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.execute_query.return_value = make_query_result({"score": 85, "rationale": "default"})
        t._provider = mock_provider
        t._recorder = mock_recorder
        t._on_start_called = True
        t.connect_output(collector, max_pending=10)
        yield t
        t.close()

    def test_handles_server_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Server errors (retryable) re-raise through pipelining as error results."""
        # ServerError is retryable so it re-raises; BatchTransformMixin catches
        # and converts it to an error result in the collector
        transform._provider.execute_query.side_effect = LLMClientError(  # type: ignore[union-attr]
            "HTTP 500: Internal Server Error",
            retryable=False,
        )

        row = make_pipeline_row(
            {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
        assert result.status == "error"
        assert result.reason is not None

    def test_handles_empty_content_from_provider(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Content policy error from provider returns error result."""
        transform._provider.execute_query.side_effect = ContentPolicyError(  # type: ignore[union-attr]
            "LLM returned null content (likely content-filtered by provider)"
        )

        row = make_pipeline_row(
            {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
        assert result.status == "error"
        assert result.reason is not None

    def test_handles_null_content_from_content_filtering(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Null content (content filtering) returns error instead of crashing.

        P0-05: When OpenRouter returns null content due to content filtering,
        the provider raises ContentPolicyError. Multi-query wraps this as
        a non-retryable error result.
        """
        transform._provider.execute_query.side_effect = ContentPolicyError(  # type: ignore[union-attr]
            "LLM returned null content (likely content-filtered by provider)"
        )

        row = make_pipeline_row(
            {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
        assert result.status == "error"
        assert result.reason is not None
        # ContentPolicyError is not retryable
        assert result.retryable is False

    def test_handles_missing_output_field_in_json(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Missing expected field in LLM JSON response is an error, not silent None."""
        # Response has 'score' but missing 'rationale' — field presence is
        # validated at the Tier 3 boundary, producing a clear error.
        transform._provider.execute_query.return_value = make_query_result(  # type: ignore[union-attr]
            {"score": 85}  # Missing 'rationale'
        )

        row = make_pipeline_row(
            {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "missing_output_field"
        assert result.reason["field"] == "rationale"
        assert "score" in result.reason["available_fields"]

    def test_handles_connection_error(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Network connection errors return retryable TransformResult.

        Multi-query sequential mode catches retryable LLMClientError and returns
        TransformResult.error(retryable=True) instead of re-raising, to avoid
        wastefully re-executing successful queries on engine retry.
        """
        transform._provider.execute_query.side_effect = NetworkError("Connection refused")  # type: ignore[union-attr]

        row = make_pipeline_row(
            {
                "cs1_bg": "data",
                "cs1_sym": "data",
                "cs1_hist": "data",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }
        )

        transform.accept(row, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        # NetworkError is retryable — multi-query catches it and returns a
        # retryable TransformResult (not ExceptionResult) to preserve successful queries.
        assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result)}"
        assert result.status == "error"
        assert result.retryable is True
        assert result.reason is not None
        assert result.reason["reason"] == "multi_query_failed"


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    def test_close_clears_recorder_reference(self) -> None:
        """close() clears the recorder reference."""
        transform = LLMTransform(make_config())
        transform._recorder = Mock()

        transform.close()

        assert transform._recorder is None

    def test_close_clears_provider(self) -> None:
        """close() clears the provider reference."""
        transform = LLMTransform(make_config())
        mock_provider = Mock(spec=LLMProvider)
        transform._provider = mock_provider

        transform.close()

        assert transform._provider is None
        mock_provider.close.assert_called_once()  # type: ignore[unreachable]

    def test_on_start_captures_recorder(self) -> None:
        """on_start() captures recorder reference for provider creation."""
        transform = LLMTransform(make_config())
        mock_recorder = Mock()

        # Verify _recorder starts as None
        assert transform._recorder is None

        ctx = make_context(
            run_id="test-run",
            state_id="test-state-id",
            landscape=mock_recorder,
        )
        transform.on_start(ctx)

        # Verify recorder was captured
        assert transform._recorder is mock_recorder


class TestNanInJsonParsing:
    """Regression: json.loads must reject NaN/Infinity in LLM response content.

    In the unified architecture, NaN/Infinity rejection is handled by the
    provider (OpenRouterLLMProvider) at the Tier 3 boundary. The provider
    raises LLMClientError for non-finite values in JSON, which the strategy
    treats as a non-retryable error.
    """

    def test_nan_in_response_json_returns_error(self) -> None:
        """LLM response containing NaN in JSON returns TransformResult.error.

        The provider rejects NaN at the Tier 3 boundary, raising LLMClientError.
        The multi-query strategy converts this to a non-retryable error result.
        """
        config: dict[str, Any] = {
            "provider": "openrouter",
            "model": "anthropic/claude-3-opus",
            "api_key": "test-key",
            "template": "Input: {{ row.text_content }}",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
            "queries": {
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "cs1_bg"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        }
        transform, mock_provider = _make_transform_with_mock_provider(config)

        # Provider raises LLMClientError on NaN in JSON
        mock_provider.execute_query.side_effect = LLMClientError("Response is not valid JSON: NaN is not valid JSON", retryable=False)

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None

    def test_infinity_in_response_json_returns_error(self) -> None:
        """LLM response containing Infinity in JSON returns TransformResult.error."""
        config: dict[str, Any] = {
            "provider": "openrouter",
            "model": "anthropic/claude-3-opus",
            "api_key": "test-key",
            "template": "Input: {{ row.text_content }}",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
            "queries": {
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "cs1_bg"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        }
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.side_effect = LLMClientError("Response is not valid JSON: Infinity is not valid JSON", retryable=False)

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None


class TestBug4_3_Tier3BoundaryTypeChecks:
    """Bug 4.3: Type checks for content, usage, and completion_tokens.

    In the unified architecture, all Tier 3 boundary validation is handled
    by the provider (OpenRouterLLMProvider). These tests verify that provider
    exceptions for type mismatches are correctly propagated by the strategy.
    """

    def test_non_str_content_returns_error(self) -> None:
        """Provider raises LLMClientError for non-string content."""
        config: dict[str, Any] = {
            "provider": "openrouter",
            "model": "anthropic/claude-3-opus",
            "api_key": "test-key",
            "template": "Input: {{ row.text_content }}",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
            "queries": {
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "cs1_bg"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        }
        transform, mock_provider = _make_transform_with_mock_provider(config)

        mock_provider.execute_query.side_effect = LLMClientError("Expected string content, got list", retryable=False)

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None

    def test_non_dict_usage_handled_by_provider(self) -> None:
        """Provider handles non-dict usage via TokenUsage.from_dict() fallback.

        TokenUsage.from_dict() gracefully handles non-dict input by returning
        TokenUsage.unknown(), so the query succeeds with unknown usage rather
        than returning an error.
        """
        config: dict[str, Any] = {
            "provider": "openrouter",
            "model": "anthropic/claude-3-opus",
            "api_key": "test-key",
            "template": "Input: {{ row.text_content }}",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
            "queries": {
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "cs1_bg"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        }
        transform, mock_provider = _make_transform_with_mock_provider(config)

        # Provider successfully handles non-dict usage and returns result
        mock_provider.execute_query.return_value = LLMQueryResult(
            content='{"score": 5, "rationale": "good"}',
            usage=TokenUsage.unknown(),
            model="anthropic/claude-3-opus",
            finish_reason=FinishReason.STOP,
        )

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["cs1_diagnosis_score"] == 5
        assert result.row["cs1_diagnosis_rationale"] == "good"

    def test_non_numeric_completion_tokens_handled_by_provider(self) -> None:
        """Non-numeric completion_tokens handled by provider's TokenUsage parsing."""
        config: dict[str, Any] = {
            "provider": "openrouter",
            "model": "anthropic/claude-3-opus",
            "api_key": "test-key",
            "template": "Input: {{ row.text_content }}",
            "schema": DYNAMIC_SCHEMA,
            "required_input_fields": [],
            "queries": {
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "cs1_bg"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        }
        transform, mock_provider = _make_transform_with_mock_provider(config)

        # Provider normalizes non-numeric completion_tokens and returns result
        mock_provider.execute_query.return_value = LLMQueryResult(
            content='{"score": 5, "rationale": "good"}',
            usage=TokenUsage.known(10, 0),  # completion_tokens normalized to 0
            model="anthropic/claude-3-opus",
            finish_reason=FinishReason.STOP,
        )

        row = make_pipeline_row({"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"})
        ctx = make_context()

        # Should not crash
        result = transform._process_row(row, ctx)

        # Should succeed since content is valid JSON
        assert result.status == "success"
