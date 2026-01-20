"""Tests for Azure Multi-Query LLM transform."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

# Common schema config
DYNAMIC_SCHEMA = {"fields": "dynamic"}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Create valid config with optional overrides."""
    config = {
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
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
        "response_format": "json",
        "output_mapping": {"score": "score", "rationale": "rationale"},
        "schema": DYNAMIC_SCHEMA,
        "pool_size": 4,
    }
    config.update(overrides)
    return config


@contextmanager
def mock_azure_openai_responses(
    responses: list[dict[str, Any]],
) -> Generator[Mock, None, None]:
    """Mock Azure OpenAI to return sequence of JSON responses."""
    call_count = 0

    def make_response() -> Mock:
        nonlocal call_count
        content = json.dumps(responses[call_count % len(responses)])
        call_count += 1

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_message = Mock()
        mock_message.content = content

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = mock_usage
        mock_response.model_dump = Mock(return_value={"model": "gpt-4o"})

        return mock_response

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = lambda **kwargs: make_response()
        mock_azure_class.return_value = mock_client
        yield mock_client


def make_plugin_context(state_id: str = "state-123") -> PluginContext:
    """Create a PluginContext with mocked landscape."""
    mock_landscape = Mock()
    mock_landscape.record_external_call = Mock()
    mock_landscape.record_call = Mock()
    return PluginContext(
        run_id="run-123",
        landscape=mock_landscape,
        state_id=state_id,
        config={},
    )


class TestAzureMultiQueryLLMTransformInit:
    """Tests for transform initialization."""

    def test_transform_has_correct_name(self) -> None:
        """Transform registers with correct plugin name."""
        transform = AzureMultiQueryLLMTransform(make_config())
        assert transform.name == "azure_multi_query_llm"

    def test_transform_is_non_deterministic(self) -> None:
        """LLM transforms are non-deterministic."""
        transform = AzureMultiQueryLLMTransform(make_config())
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_transform_is_batch_aware(self) -> None:
        """Transform supports batch aggregation."""
        transform = AzureMultiQueryLLMTransform(make_config())
        assert transform.is_batch_aware is True

    def test_transform_expands_queries_on_init(self) -> None:
        """Transform pre-computes query specs on initialization."""
        transform = AzureMultiQueryLLMTransform(make_config())
        # 2 case studies x 2 criteria = 4 queries
        assert len(transform._query_specs) == 4

    def test_transform_requires_case_studies(self) -> None:
        """Transform requires case_studies in config."""
        config = make_config()
        del config["case_studies"]
        with pytest.raises(PluginConfigError):
            AzureMultiQueryLLMTransform(config)

    def test_transform_requires_criteria(self) -> None:
        """Transform requires criteria in config."""
        config = make_config()
        del config["criteria"]
        with pytest.raises(PluginConfigError):
            AzureMultiQueryLLMTransform(config)


class TestSingleQueryProcessing:
    """Tests for _process_single_query method."""

    def test_process_single_query_renders_template(self) -> None:
        """Single query renders template with input fields and criterion."""
        responses = [{"score": 85, "rationale": "Good diagnosis"}]

        with mock_azure_openai_responses(responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            # Call on_start to set up the recorder
            transform.on_start(ctx)

            row = {
                "cs1_bg": "45yo male",
                "cs1_sym": "chest pain",
                "cs1_hist": "family history",
            }
            spec = transform._query_specs[0]  # cs1_diagnosis

            transform._process_single_query(row, spec, ctx.state_id)

            # Check template was rendered with correct content
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[-1]["content"]

            assert "45yo male" in user_message
            assert "diagnosis" in user_message.lower()

    def test_process_single_query_parses_json_response(self) -> None:
        """Single query parses JSON and returns mapped fields."""
        responses = [{"score": 85, "rationale": "Excellent assessment"}]

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]  # cs1_diagnosis

            result = transform._process_single_query(row, spec, ctx.state_id)

            assert result.status == "success"
            assert result.row is not None
            # Output fields use prefix from spec
            assert result.row["cs1_diagnosis_score"] == 85
            assert result.row["cs1_diagnosis_rationale"] == "Excellent assessment"

    def test_process_single_query_handles_invalid_json(self) -> None:
        """Single query returns error on invalid JSON response."""
        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="not json"))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump = Mock(return_value={})
            mock_client.chat.completions.create.return_value = mock_response
            mock_azure_class.return_value = mock_client

            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            result = transform._process_single_query(row, spec, ctx.state_id)

            assert result.status == "error"
            assert "json" in result.reason["reason"].lower()

    def test_process_single_query_raises_capacity_error_on_rate_limit(self) -> None:
        """Rate limit errors are converted to CapacityError for pooled retry."""
        from elspeth.plugins.clients.llm import RateLimitError
        from elspeth.plugins.pooling import CapacityError

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            # Simulate rate limit from the underlying client
            mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
            mock_azure_class.return_value = mock_client

            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            # Need to mock at AuditedLLMClient level since that's where RateLimitError comes from
            with patch.object(transform, "_get_llm_client") as mock_get_client:
                mock_llm_client = Mock()
                mock_llm_client.chat_completion.side_effect = RateLimitError("Rate limit exceeded")
                mock_get_client.return_value = mock_llm_client

                with pytest.raises(CapacityError) as exc_info:
                    transform._process_single_query(row, spec, ctx.state_id)

                assert exc_info.value.status_code == 429

    def test_process_single_query_handles_template_error(self) -> None:
        """Template rendering errors return error result with details."""
        from elspeth.plugins.llm.templates import TemplateError

        with mock_azure_openai_responses([{"score": 85, "rationale": "ok"}]):
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            # Mock template to raise error
            with patch.object(transform._template, "render_with_metadata") as mock_render:
                mock_render.side_effect = TemplateError("Undefined variable 'missing'")

                result = transform._process_single_query(row, spec, ctx.state_id)

                assert result.status == "error"
                assert result.reason["reason"] == "template_rendering_failed"
                assert "missing" in result.reason["error"]

    def test_get_llm_client_requires_recorder(self) -> None:
        """_get_llm_client raises RuntimeError if recorder not set via on_start."""
        transform = AzureMultiQueryLLMTransform(make_config())
        # Don't call on_start - recorder will be None

        with pytest.raises(RuntimeError) as exc_info:
            transform._get_llm_client("state-123")

        assert "recorder" in str(exc_info.value).lower()


class TestRowProcessing:
    """Tests for full row processing (all queries)."""

    def test_process_row_executes_all_queries(self) -> None:
        """Process executes all (case_study x criterion) queries."""
        # 2 case studies x 2 criteria = 4 queries
        responses = [
            {"score": 85, "rationale": "CS1 diagnosis"},
            {"score": 90, "rationale": "CS1 treatment"},
            {"score": 75, "rationale": "CS2 diagnosis"},
            {"score": 80, "rationale": "CS2 treatment"},
        ]

        with mock_azure_openai_responses(responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            row = {
                "cs1_bg": "case1 bg",
                "cs1_sym": "case1 sym",
                "cs1_hist": "case1 hist",
                "cs2_bg": "case2 bg",
                "cs2_sym": "case2 sym",
                "cs2_hist": "case2 hist",
            }

            result = transform.process(row, ctx)

            assert result.status == "success"
            assert mock_client.chat.completions.create.call_count == 4

    def test_process_row_merges_all_results(self) -> None:
        """All query results are merged into single output row."""
        responses = [
            {"score": 85, "rationale": "R1"},
            {"score": 90, "rationale": "R2"},
            {"score": 75, "rationale": "R3"},
            {"score": 80, "rationale": "R4"},
        ]

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            row = {
                "cs1_bg": "bg1",
                "cs1_sym": "sym1",
                "cs1_hist": "hist1",
                "cs2_bg": "bg2",
                "cs2_sym": "sym2",
                "cs2_hist": "hist2",
                "original_field": "preserved",
            }

            result = transform.process(row, ctx)

            assert result.status == "success"
            output = result.row

            # Original fields preserved
            assert output["original_field"] == "preserved"

            # All 4 queries produced output (2 fields each = 8 assessment fields)
            assert "cs1_diagnosis_score" in output
            assert "cs1_diagnosis_rationale" in output
            assert "cs1_treatment_score" in output
            assert "cs2_diagnosis_score" in output
            assert "cs2_treatment_score" in output

    def test_process_row_fails_if_any_query_fails(self) -> None:
        """All-or-nothing: if any query fails, entire row fails."""
        # First 3 succeed, 4th returns invalid JSON
        call_count = [0]

        def make_response(**kwargs: Any) -> Mock:
            call_count[0] += 1
            if call_count[0] == 4:
                content = "not valid json"
            else:
                content = json.dumps({"score": 85, "rationale": "ok"})

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=content))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump = Mock(return_value={})
            return mock_response

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = make_response
            mock_azure_class.return_value = mock_client

            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            row = {
                "cs1_bg": "bg",
                "cs1_sym": "sym",
                "cs1_hist": "hist",
                "cs2_bg": "bg",
                "cs2_sym": "sym",
                "cs2_hist": "hist",
            }

            result = transform.process(row, ctx)

            # Entire row fails
            assert result.status == "error"
            assert "query_failed" in result.reason["reason"]


class TestBatchProcessing:
    """Tests for batch processing (aggregation mode)."""

    def test_process_batch_handles_list_input(self) -> None:
        """Process accepts list of rows for batch aggregation."""
        # 2 rows x 4 queries each = 8 total LLM calls
        responses = [{"score": i, "rationale": f"R{i}"} for i in range(8)]

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            rows = [
                {
                    "cs1_bg": "r1",
                    "cs1_sym": "r1",
                    "cs1_hist": "r1",
                    "cs2_bg": "r1",
                    "cs2_sym": "r1",
                    "cs2_hist": "r1",
                },
                {
                    "cs1_bg": "r2",
                    "cs1_sym": "r2",
                    "cs1_hist": "r2",
                    "cs2_bg": "r2",
                    "cs2_sym": "r2",
                    "cs2_hist": "r2",
                },
            ]

            result = transform.process(rows, ctx)

            assert result.status == "success"
            assert result.is_multi_row
            assert result.rows is not None
            assert len(result.rows) == 2

    def test_process_batch_preserves_row_independence(self) -> None:
        """Each row in batch is processed independently - row 1 succeeds, row 2 fails."""
        # Test that first row succeeds even when second row fails
        # Use pool_size=1 for predictable sequential order
        # Row 1's 4 queries succeed, Row 2's 4th query returns invalid JSON
        # Result should have 2 rows: row 1 with output fields, row 2 with _error marker
        call_count = [0]

        def make_response(**kwargs: Any) -> Mock:
            call_count[0] += 1
            # First 4 calls (row 1) succeed, 5-7 (row 2) succeed, 8th (row 2) fails
            if call_count[0] == 8:
                content = "not valid json"
            else:
                content = json.dumps({"score": call_count[0], "rationale": f"R{call_count[0]}"})

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=content))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump = Mock(return_value={})
            return mock_response

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = make_response
            mock_azure_class.return_value = mock_client

            # Use pool_size=1 for sequential execution within each row
            transform = AzureMultiQueryLLMTransform(make_config(pool_size=1))
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            rows = [
                {
                    "cs1_bg": "r1_bg",
                    "cs1_sym": "r1_sym",
                    "cs1_hist": "r1_hist",
                    "cs2_bg": "r1_bg",
                    "cs2_sym": "r1_sym",
                    "cs2_hist": "r1_hist",
                },
                {
                    "cs1_bg": "r2_bg",
                    "cs1_sym": "r2_sym",
                    "cs1_hist": "r2_hist",
                    "cs2_bg": "r2_bg",
                    "cs2_sym": "r2_sym",
                    "cs2_hist": "r2_hist",
                },
            ]

            result = transform.process(rows, ctx)

            # Batch succeeds with partial results
            assert result.status == "success"
            assert result.is_multi_row
            assert result.rows is not None
            assert len(result.rows) == 2

            # Row 1 should have output fields (succeeded)
            assert "cs1_diagnosis_score" in result.rows[0]
            assert "cs1_treatment_score" in result.rows[0]

            # Row 2 should have error marker (failed)
            assert "_error" in result.rows[1]

    def test_process_batch_empty_list_returns_empty_result(self) -> None:
        """Empty batch returns success with empty indicator."""
        with mock_azure_openai_responses([{"score": 1, "rationale": "R"}]):
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            result = transform.process([], ctx)

            assert result.status == "success"
            assert result.row is not None
            assert result.row["batch_empty"] is True
            assert result.row["row_count"] == 0

    def test_process_batch_uses_per_row_state_ids(self) -> None:
        """Each row in batch gets unique state_id for audit trail isolation."""
        # We can verify this by checking that different rows create different
        # client cache entries (which use state_id as key)
        responses = [{"score": i, "rationale": f"R{i}"} for i in range(8)]
        created_state_ids: list[str] = []

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())

            # Patch _get_llm_client to track state_ids
            original_get_client = transform._get_llm_client

            def tracking_get_client(state_id: str) -> Any:
                created_state_ids.append(state_id)
                return original_get_client(state_id)

            transform._get_llm_client = tracking_get_client  # type: ignore[method-assign]

            ctx = make_plugin_context(state_id="batch-001")

            rows = [
                {
                    "cs1_bg": "r1",
                    "cs1_sym": "r1",
                    "cs1_hist": "r1",
                    "cs2_bg": "r1",
                    "cs2_sym": "r1",
                    "cs2_hist": "r1",
                },
                {
                    "cs1_bg": "r2",
                    "cs1_sym": "r2",
                    "cs1_hist": "r2",
                    "cs2_bg": "r2",
                    "cs2_sym": "r2",
                    "cs2_hist": "r2",
                },
            ]

            transform.process(rows, ctx)

            # Should have state_ids like "batch-001_row0" and "batch-001_row1"
            # (4 queries per row = 4 calls to _get_llm_client per row)
            row0_ids = [s for s in created_state_ids if "row0" in s]
            row1_ids = [s for s in created_state_ids if "row1" in s]

            assert len(row0_ids) > 0, "Row 0 should have unique state_ids"
            assert len(row1_ids) > 0, "Row 1 should have unique state_ids"
            assert "batch-001_row0" in row0_ids[0]
            assert "batch-001_row1" in row1_ids[0]

    def test_process_batch_cleans_up_per_row_clients(self) -> None:
        """Client cache is cleaned up after each row in batch."""
        responses = [{"score": i, "rationale": f"R{i}"} for i in range(8)]

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context(state_id="batch-002")

            rows = [
                {
                    "cs1_bg": "r1",
                    "cs1_sym": "r1",
                    "cs1_hist": "r1",
                    "cs2_bg": "r1",
                    "cs2_sym": "r1",
                    "cs2_hist": "r1",
                },
                {
                    "cs1_bg": "r2",
                    "cs1_sym": "r2",
                    "cs1_hist": "r2",
                    "cs2_bg": "r2",
                    "cs2_sym": "r2",
                    "cs2_hist": "r2",
                },
            ]

            transform.process(rows, ctx)

            # After processing, per-row state_ids should be cleaned up
            assert "batch-002_row0" not in transform._llm_clients
            assert "batch-002_row1" not in transform._llm_clients
