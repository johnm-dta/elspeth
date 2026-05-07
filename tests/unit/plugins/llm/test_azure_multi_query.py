"""Tests for multi-query LLM transform with row-level pipelining.

Migrated from AzureMultiQueryLLMTransform to unified LLMTransform with
provider="azure" and queries={...} config format.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism, TransformResult
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMProvider, LLMQueryResult
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

from .conftest import (
    make_token,
)

# ---------------------------------------------------------------------------
# Config helpers (inline, using the new unified format)
# ---------------------------------------------------------------------------

DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_config(**overrides: Any) -> dict[str, Any]:
    """Create valid LLMTransform multi-query config with optional overrides.

    Uses the new unified format: provider + queries (not case_studies + criteria).
    Default queries define 4 query specs (2 case studies x 2 criteria equivalent):
      cs1_diag, cs1_treat, cs2_diag, cs2_treat
    """
    config: dict[str, Any] = {
        "provider": "azure",
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Evaluate: {{ row.text_content }}\nCriterion: {{ row.criterion_name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],
        "pool_size": 4,
        "queries": {
            "cs1_diagnosis": {
                "input_fields": {
                    "text_content": "cs1_bg",
                    "symptom": "cs1_sym",
                    "history": "cs1_hist",
                    "criterion_name": "cs1_bg",  # placeholder for criterion context
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_treatment": {
                "input_fields": {
                    "text_content": "cs1_bg",
                    "symptom": "cs1_sym",
                    "history": "cs1_hist",
                    "criterion_name": "cs1_bg",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_diagnosis": {
                "input_fields": {
                    "text_content": "cs2_bg",
                    "symptom": "cs2_sym",
                    "history": "cs2_hist",
                    "criterion_name": "cs2_bg",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_treatment": {
                "input_fields": {
                    "text_content": "cs2_bg",
                    "symptom": "cs2_sym",
                    "history": "cs2_hist",
                    "criterion_name": "cs2_bg",
                },
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
        },
    }
    config.update(overrides)
    return config


def _make_mock_provider(
    responses: Sequence[dict[str, Any] | str] | None = None,
) -> Mock:
    """Create a mock LLM provider returning predetermined responses.

    If responses is None, returns a default success response for every call.
    Otherwise cycles through the provided responses.
    """
    import itertools
    import json

    mock_provider = Mock(spec=LLMProvider)

    if responses is None:

        def default_execute(
            messages: list[dict[str, str]],
            *,
            model: str,
            temperature: float,
            max_tokens: int | None,
            state_id: str,
            token_id: str,
            response_format: dict[str, Any] | None = None,
        ) -> LLMQueryResult:
            return LLMQueryResult(
                content='{"score": 85, "rationale": "Good"}',
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider.execute_query.side_effect = default_execute
    else:
        cycle = itertools.cycle(responses)

        def execute_from_list(messages, *, model, temperature, max_tokens, state_id, token_id, response_format=None):
            payload = next(cycle)
            if isinstance(payload, str):
                content = payload
            else:
                content = json.dumps(payload)
            return LLMQueryResult(
                content=content,
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider.execute_query.side_effect = execute_from_list

    mock_provider.close = Mock()
    return mock_provider


class TestLLMTransformMultiQueryInit:
    """Tests for multi-query LLMTransform initialization."""

    def test_transform_has_correct_name(self) -> None:
        """Transform registers with correct plugin name."""
        transform = LLMTransform(_make_config())
        assert transform.name == "llm"

    def test_transform_is_non_deterministic(self) -> None:
        """LLM transforms are non-deterministic."""
        transform = LLMTransform(_make_config())
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_transform_selects_multi_query_strategy(self) -> None:
        """Transform selects MultiQueryStrategy when queries provided."""
        from elspeth.plugins.transforms.llm.transform import MultiQueryStrategy

        transform = LLMTransform(_make_config())
        assert isinstance(transform._strategy, MultiQueryStrategy)

    def test_transform_resolves_query_specs_on_init(self) -> None:
        """Transform resolves query specs from queries config on init."""
        from elspeth.plugins.transforms.llm.transform import MultiQueryStrategy

        transform = LLMTransform(_make_config())
        assert isinstance(transform._strategy, MultiQueryStrategy)
        # 4 queries defined in config
        assert len(transform._strategy.query_specs) == 4

    def test_transform_requires_queries_for_multi_query(self) -> None:
        """Transform with no queries selects SingleQueryStrategy."""
        from elspeth.plugins.transforms.llm.transform import SingleQueryStrategy

        config = _make_config()
        del config["queries"]
        transform = LLMTransform(config)
        assert isinstance(transform._strategy, SingleQueryStrategy)

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        transform = LLMTransform(_make_config())
        ctx = make_context()

        with pytest.raises(NotImplementedError, match="accept"):
            transform.process(make_pipeline_row({"text": "hello"}), ctx)


class TestSingleQueryProcessing:
    """Tests for multi-query _process_row method via mocked provider."""

    def test_process_row_renders_template(self) -> None:
        """Query renders template with input fields."""
        transform = LLMTransform(_make_config())

        # Track what messages were sent
        captured_messages: list[list[dict[str, str]]] = []

        def capture_execute(messages, *, model, temperature, max_tokens, state_id, token_id, response_format=None):
            captured_messages.append(messages)
            return LLMQueryResult(
                content='{"score": 85, "rationale": "Good diagnosis"}',
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider = Mock(spec=LLMProvider)
        mock_provider.execute_query.side_effect = capture_execute
        transform._provider = mock_provider

        row = make_pipeline_row(
            {
                "cs1_bg": "45yo male",
                "cs1_sym": "chest pain",
                "cs1_hist": "family history",
                "cs2_bg": "data",
                "cs2_sym": "data",
                "cs2_hist": "data",
            }
        )
        ctx = make_context()
        result = transform._process_row(row, ctx)

        assert result.status == "success"
        # First query should have rendered template with cs1_bg data
        assert len(captured_messages) == 4
        first_user_msg = captured_messages[0][-1]["content"]
        assert "45yo male" in first_user_msg

    def test_process_row_parses_json_response(self) -> None:
        """Query parses JSON and returns mapped fields."""
        transform = LLMTransform(_make_config())
        mock_provider = _make_mock_provider(
            [
                {"score": 85, "rationale": "CS1 diagnosis"},
                {"score": 90, "rationale": "CS1 treatment"},
                {"score": 75, "rationale": "CS2 diagnosis"},
                {"score": 80, "rationale": "CS2 treatment"},
            ]
        )
        transform._provider = mock_provider

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
        ctx = make_context()
        result = transform._process_row(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["cs1_diagnosis_score"] == 85
        assert result.row["cs1_diagnosis_rationale"] == "CS1 diagnosis"

    def test_process_row_handles_invalid_json(self) -> None:
        """Query returns error on invalid JSON response."""
        transform = LLMTransform(_make_config())
        mock_provider = _make_mock_provider(["not json"])
        transform._provider = mock_provider

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
        ctx = make_context()
        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert "json" in result.reason["reason"].lower()

    def test_process_row_handles_template_error(self) -> None:
        """Template rendering errors return error result with details."""
        from unittest.mock import patch as mock_patch

        from elspeth.plugins.infrastructure.templates import TemplateError

        transform = LLMTransform(_make_config())
        mock_provider = _make_mock_provider()
        transform._provider = mock_provider

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
        ctx = make_context()

        # Mock the template on the strategy to raise TemplateError
        with mock_patch.object(
            transform._strategy.template,
            "render_with_metadata",
            side_effect=TemplateError("Undefined variable 'missing'"),
        ):
            result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "missing" in result.reason["error"]

    def test_on_start_sets_lifecycle_flag(self) -> None:
        """on_start() sets _on_start_called flag for centralized lifecycle guard."""
        transform = LLMTransform(_make_config())
        assert not transform._on_start_called

        ctx = Mock()
        ctx.landscape = Mock()
        ctx.run_id = "test-run"
        ctx.telemetry_emit = None
        ctx.rate_limit_registry = None
        transform.on_start(ctx)

        assert transform._on_start_called


class TestRowProcessingWithPipelining:
    """Tests for full row processing using accept() API with pipelining.

    These tests verify the accept() API that uses BatchTransformMixin
    for concurrent row processing with FIFO output ordering.
    """

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
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
        return make_context(
            run_id="run-123",
            landscape=mock_recorder,
            state_id="state-123",
            token=token,
        )

    @pytest.fixture
    def transform(
        self,
        collector: CollectorOutputPort,
        mock_recorder: Mock,
    ) -> Generator[LLMTransform, None, None]:
        """Create and initialize LLMTransform with multi-query and pipelining."""
        t = LLMTransform(_make_config())
        # Initialize with recorder reference
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        t.on_start(init_ctx)
        # Replace provider with mock
        t._provider = _make_mock_provider()
        # Connect output port
        t.connect_output(collector, max_pending=10)
        yield t
        # Cleanup
        t.close()

    def test_successful_row_emits_all_query_results(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Successful row emits results with all query outputs merged."""
        # 4 queries with JSON responses
        responses: list[dict[str, Any]] = [
            {"score": 85, "rationale": "CS1 diagnosis"},
            {"score": 90, "rationale": "CS1 treatment"},
            {"score": 75, "rationale": "CS2 diagnosis"},
            {"score": 80, "rationale": "CS2 treatment"},
        ]
        transform._provider = _make_mock_provider(responses)

        row_data = {
            "cs1_bg": "case1 bg",
            "cs1_sym": "case1 sym",
            "cs1_hist": "case1 hist",
            "cs2_bg": "case2 bg",
            "cs2_sym": "case2 sym",
            "cs2_hist": "case2 hist",
        }
        transform.accept(make_pipeline_row(row_data), ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _token, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "success"
        assert transform._provider.execute_query.call_count == 4

        # All query results merged into output
        assert result.row is not None
        assert "cs1_diagnosis_score" in result.row
        assert "cs1_treatment_score" in result.row
        assert "cs2_diagnosis_score" in result.row
        assert "cs2_treatment_score" in result.row

    def test_row_preserves_original_fields(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """Output row preserves original input fields."""
        responses: list[dict[str, Any]] = [
            {"score": 85, "rationale": "R1"},
            {"score": 90, "rationale": "R2"},
            {"score": 75, "rationale": "R3"},
            {"score": 80, "rationale": "R4"},
        ]
        transform._provider = _make_mock_provider(responses)

        row_data = {
            "cs1_bg": "bg1",
            "cs1_sym": "sym1",
            "cs1_hist": "hist1",
            "cs2_bg": "bg2",
            "cs2_sym": "sym2",
            "cs2_hist": "hist2",
            "original_field": "preserved",
        }
        transform.accept(make_pipeline_row(row_data), ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "success"
        assert result.row is not None
        # Original fields preserved
        assert result.row["original_field"] == "preserved"
        assert result.row["cs1_bg"] == "bg1"

    def test_row_accept_supports_original_header_names_in_input_fields(
        self,
        ctx: PluginContext,
        collector: CollectorOutputPort,
        mock_recorder: Mock,
    ) -> None:
        """Original source headers in input_fields resolve via PipelineRow contract."""
        # Build config with original column names mapped to normalized names
        config = _make_config(
            queries={
                "cs1_diagnosis": {
                    "input_fields": {
                        "text_content": "patient_name",
                        "symptom": "symptoms",
                        "history": "history",
                        "criterion_name": "patient_name",
                    },
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        )
        transform = LLMTransform(config)
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform._provider = _make_mock_provider([{"score": 85, "rationale": "Looks consistent"}])
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
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            assert result.row["cs1_diagnosis_score"] == 85
            assert result.row["patient_name"] == "Alice Smith"
        finally:
            transform.close()

    def test_row_fails_if_any_query_fails(
        self,
        ctx: PluginContext,
        transform: LLMTransform,
        collector: CollectorOutputPort,
    ) -> None:
        """All-or-nothing: if any query fails (invalid JSON), entire row fails."""
        # 3 good JSON + 1 invalid
        call_count = [0]

        def fail_on_fourth(messages, *, model, temperature, max_tokens, state_id, token_id, response_format=None):
            call_count[0] += 1
            if call_count[0] == 4:
                return LLMQueryResult(
                    content="not valid json",
                    usage=TokenUsage.known(10, 5),
                    model="gpt-4o",
                    finish_reason=FinishReason.STOP,
                )
            return LLMQueryResult(
                content='{"score": 85, "rationale": "ok"}',
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider = Mock(spec=LLMProvider)
        mock_provider.execute_query.side_effect = fail_on_fourth
        mock_provider.close = Mock()
        transform._provider = mock_provider

        row_data = {
            "cs1_bg": "bg",
            "cs1_sym": "sym",
            "cs1_hist": "hist",
            "cs2_bg": "bg",
            "cs2_sym": "sym",
            "cs2_hist": "hist",
        }
        transform.accept(make_pipeline_row(row_data), ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        # Entire row fails (atomic multi-query semantics)
        assert result.status == "error"
        assert result.reason is not None
        assert "json" in result.reason["reason"].lower()

    def test_connect_output_required_before_accept(self, mock_recorder: Mock) -> None:
        """accept() raises RuntimeError if connect_output() not called."""
        transform = LLMTransform(_make_config())

        token = make_token("row-1")
        ctx = make_context(
            run_id="test-run",
            state_id="test-state-id",
            token=token,
        )

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept(make_pipeline_row({"text": "hello"}), ctx)

    def test_connect_output_cannot_be_called_twice(
        self,
        collector: CollectorOutputPort,
        mock_recorder: Mock,
    ) -> None:
        """connect_output() raises if called more than once."""
        transform = LLMTransform(_make_config())
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            with pytest.raises(RuntimeError, match="already called"):
                transform.connect_output(collector, max_pending=10)
        finally:
            transform.close()


class TestMultiRowPipelining:
    """Tests for processing multiple rows with FIFO ordering."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
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
    ) -> None:
        """Multiple rows are emitted in submission order (FIFO).

        Tests row-level pipelining via BatchTransformMixin.
        """
        config = _make_config()

        transform = LLMTransform(config)
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform._provider = _make_mock_provider()
        transform.connect_output(collector, max_pending=10)

        try:
            for i in range(3):
                row_data = {
                    "row_id": f"row-{i}",
                    "cs1_bg": f"r{i}",
                    "cs1_sym": f"r{i}",
                    "cs1_hist": f"r{i}",
                    "cs2_bg": f"r{i}",
                    "cs2_sym": f"r{i}",
                    "cs2_hist": f"r{i}",
                }
                token = make_token(f"row-{i}")
                ctx = make_context(
                    run_id="run-123",
                    landscape=mock_recorder,
                    state_id=f"state-{i}",
                    token=token,
                )
                transform.accept(make_pipeline_row(row_data), ctx)

            transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        # Results should be in FIFO order
        assert len(collector.results) == 3
        for i, (_token, result, _state_id) in enumerate(collector.results):
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            assert result.row["row_id"] == f"row-{i}"

    def test_on_start_captures_recorder(self, mock_recorder: Mock) -> None:
        """on_start() captures recorder reference for provider creation."""
        transform = LLMTransform(_make_config())

        # Verify _recorder starts as None
        assert transform._recorder is None

        ctx = make_context(
            run_id="test-run",
            landscape=mock_recorder,
            state_id="test-state-id",
        )
        transform.on_start(ctx)

        # Verify recorder was captured
        assert transform._recorder is mock_recorder

    def test_close_clears_recorder_and_provider(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
    ) -> None:
        """close() clears recorder reference and provider."""
        transform = LLMTransform(_make_config())
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform._provider = _make_mock_provider()
        transform.connect_output(collector, max_pending=10)

        assert transform._recorder is not None

        transform.close()

        assert transform._recorder is None
        assert transform._provider is None  # type: ignore[unreachable]  # verifying close() cleanup at runtime


class TestMultiQueryWithMockProvider:
    """Tests for multi-query using mock provider (no ChaosLLM server)."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_parallel_mode_includes_no_pool_stats_in_context_after(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
    ) -> None:
        """Multi-query with mocked provider processes correctly.

        The new MultiQueryStrategy executes queries sequentially within a row
        (parallelism is at the row level via BatchTransformMixin).
        context_after is None for sequential query execution.
        """
        config = _make_config(pool_size=4)
        transform = LLMTransform(config)
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform._provider = _make_mock_provider([{"score": i, "rationale": f"R{i}"} for i in range(4)])
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {
                "cs1_bg": "bg1",
                "cs1_sym": "sym1",
                "cs1_hist": "hist1",
                "cs2_bg": "bg2",
                "cs2_sym": "sym2",
                "cs2_hist": "hist2",
            }
            token = make_token("row-1")
            ctx = make_context(
                run_id="run-123",
                landscape=mock_recorder,
                state_id="state-pool-001",
                token=token,
            )
            transform.accept(make_pipeline_row(row_data), ctx)
            transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"

        # The new multi-query strategy processes queries sequentially within
        # a row — no per-query pool stats. context_after is None.
        assert result.context_after is None

    def test_sequential_mode_processes_rows(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
    ) -> None:
        """Sequential mode (no pool_size) processes rows correctly."""
        config = _make_config()
        del config["pool_size"]

        transform = LLMTransform(config)
        init_ctx = make_context(run_id="test", landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform._provider = _make_mock_provider([{"score": i, "rationale": f"R{i}"} for i in range(4)])
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {
                "cs1_bg": "r1",
                "cs1_sym": "r1",
                "cs1_hist": "r1",
                "cs2_bg": "r1",
                "cs2_sym": "r1",
                "cs2_hist": "r1",
            }
            token = make_token("row-1")
            ctx = make_context(
                run_id="run-123",
                landscape=mock_recorder,
                state_id="batch-seq-001",
                token=token,
            )
            transform.accept(make_pipeline_row(row_data), ctx)
            transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        # Sequential mode has no pool — context_after should be None
        assert result.context_after is None


class TestBug4_1_KeyErrorInBuildTemplateContext:
    """Bug 4.1: KeyError in build_template_context returns error, not crash.

    When a row is missing fields required by build_template_context(),
    a KeyError is raised. The MultiQueryStrategy catches this and returns
    TransformResult.error() with reason "template_context_failed".
    """

    def test_missing_input_field_returns_error(self) -> None:
        """Row missing required input field returns error instead of crashing."""
        transform = LLMTransform(_make_config())
        mock_provider = _make_mock_provider()
        transform._provider = mock_provider

        # Row is missing cs1_bg, cs1_sym, cs1_hist — required by first query
        row = make_pipeline_row({"cs2_bg": "bg", "cs2_sym": "sym", "cs2_hist": "hist"})
        ctx = make_context()

        result = transform._process_row(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_context_failed"
        # Error should mention the query name
        assert "cs1_diagnosis" in result.reason["query_name"]


class TestMultiQueryDeclaredOutputFields:
    """Tests for declared_output_fields on unified LLMTransform.

    Field collision detection is enforced centrally by TransformExecutor.
    These tests verify that LLMTransform correctly declares its output fields.
    """

    def test_declared_output_fields_is_nonempty(self) -> None:
        """declared_output_fields is populated for schema evolution recording."""
        transform = LLMTransform(_make_config())
        assert transform.declared_output_fields

    def test_declared_output_fields_contains_prefixed_response_fields(self) -> None:
        """Multi-query declared_output_fields includes query-prefixed fields."""
        transform = LLMTransform(_make_config())
        # Multi-query declares prefixed fields, not base unprefixed fields
        assert "cs1_diagnosis_llm_response" in transform.declared_output_fields
        assert "cs1_diagnosis_llm_response_usage" in transform.declared_output_fields
        assert "cs1_diagnosis_llm_response_model" in transform.declared_output_fields

    def test_declared_output_fields_excludes_audit_fields(self) -> None:
        """Multi-query declared_output_fields excludes audit fields.

        Audit fields (template_hash, variables_hash, etc.) now travel via
        success_reason["metadata"], not the output row.
        """
        transform = LLMTransform(_make_config())
        assert "cs1_diagnosis_llm_response_template_hash" not in transform.declared_output_fields
        assert "cs1_diagnosis_llm_response_variables_hash" not in transform.declared_output_fields
