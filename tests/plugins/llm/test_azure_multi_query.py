"""Tests for Azure Multi-Query LLM transform with row-level pipelining."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.batching.ports import CollectorOutputPort
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

from .conftest import (
    chaosllm_azure_openai_client,
    chaosllm_azure_openai_responses,
    make_plugin_context,
    make_token,
)
from .conftest import (
    make_azure_multi_query_config as make_config,
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

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        transform = AzureMultiQueryLLMTransform(make_config())
        ctx = make_plugin_context()

        with pytest.raises(NotImplementedError, match="row-level pipelining"):
            transform.process({"text": "hello"}, ctx)


class TestSingleQueryProcessing:
    """Tests for _process_single_query method."""

    def test_process_single_query_renders_template(self, chaosllm_server) -> None:
        """Single query renders template with input fields and criterion."""
        responses = [{"score": 85, "rationale": "Good diagnosis"}]

        with chaosllm_azure_openai_responses(chaosllm_server, responses) as mock_client:
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

            assert ctx.state_id is not None
            transform._process_single_query(row, spec, ctx.state_id)

            # Check template was rendered with correct content
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[-1]["content"]

            assert "45yo male" in user_message
            assert "diagnosis" in user_message.lower()

    def test_process_single_query_parses_json_response(self, chaosllm_server) -> None:
        """Single query parses JSON and returns mapped fields."""
        responses = [{"score": 85, "rationale": "Excellent assessment"}]

        with chaosllm_azure_openai_responses(chaosllm_server, responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]  # cs1_diagnosis

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id)

            assert result.status == "success"
            assert result.row is not None
            # Output fields use prefix from spec
            assert result.row["cs1_diagnosis_score"] == 85
            assert result.row["cs1_diagnosis_rationale"] == "Excellent assessment"

    def test_process_single_query_handles_invalid_json(self, chaosllm_server) -> None:
        """Single query returns error on invalid JSON response."""
        with chaosllm_azure_openai_client(
            chaosllm_server,
            mode="template",
            template_override="not json",
        ):
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()
            transform.on_start(ctx)

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            assert ctx.state_id is not None
            result = transform._process_single_query(row, spec, ctx.state_id)

            assert result.status == "error"
            assert result.reason is not None
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

                assert ctx.state_id is not None
                with pytest.raises(CapacityError) as exc_info:
                    transform._process_single_query(row, spec, ctx.state_id)

                assert exc_info.value.status_code == 429

    def test_process_single_query_handles_template_error(self, chaosllm_server) -> None:
        """Template rendering errors return error result with details."""
        from elspeth.plugins.llm.templates import TemplateError

        with chaosllm_azure_openai_responses(chaosllm_server, [{"score": 85, "rationale": "ok"}]):
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            # Mock template to raise error
            with patch.object(transform._template, "render_with_metadata") as mock_render:
                mock_render.side_effect = TemplateError("Undefined variable 'missing'")

                assert ctx.state_id is not None
                result = transform._process_single_query(row, spec, ctx.state_id)

                assert result.status == "error"
                assert result.reason is not None
                assert result.reason["reason"] == "template_rendering_failed"
                assert "missing" in result.reason["error"]

    def test_get_llm_client_requires_recorder(self) -> None:
        """_get_llm_client raises RuntimeError if recorder not set via on_start."""
        transform = AzureMultiQueryLLMTransform(make_config())
        # Don't call on_start - recorder will be None

        with pytest.raises(RuntimeError) as exc_info:
            transform._get_llm_client("state-123")

        assert "recorder" in str(exc_info.value).lower()


class TestRowProcessingWithPipelining:
    """Tests for full row processing using accept() API with pipelining.

    These tests verify the new accept() API that uses BatchTransformMixin
    for concurrent row processing with FIFO output ordering.
    """

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
            run_id="run-123",
            config={},
            landscape=mock_recorder,
            state_id="state-123",
            token=token,
        )

    @pytest.fixture
    def transform(
        self,
        collector: CollectorOutputPort,
        mock_recorder: Mock,
    ) -> Generator[AzureMultiQueryLLMTransform, None, None]:
        """Create and initialize Azure multi-query transform with pipelining."""
        t = AzureMultiQueryLLMTransform(make_config())
        # Initialize with recorder reference
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        t.on_start(init_ctx)
        # Connect output port
        t.connect_output(collector, max_pending=10)
        yield t
        # Cleanup
        t.close()

    def test_successful_row_emits_all_query_results(
        self,
        ctx: PluginContext,
        transform: AzureMultiQueryLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Successful row emits results with all query outputs merged."""
        # 4 queries (2 case studies x 2 criteria)
        responses = [
            {"score": 85, "rationale": "CS1 diagnosis"},
            {"score": 90, "rationale": "CS1 treatment"},
            {"score": 75, "rationale": "CS2 diagnosis"},
            {"score": 80, "rationale": "CS2 treatment"},
        ]

        with chaosllm_azure_openai_responses(chaosllm_server, responses) as mock_client:
            row = {
                "cs1_bg": "case1 bg",
                "cs1_sym": "case1 sym",
                "cs1_hist": "case1 hist",
                "cs2_bg": "case2 bg",
                "cs2_sym": "case2 sym",
                "cs2_hist": "case2 hist",
            }
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _token, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "success"
        assert mock_client.chat.completions.create.call_count == 4

        # All query results merged into output
        assert result.row is not None
        assert "cs1_diagnosis_score" in result.row
        assert "cs1_treatment_score" in result.row
        assert "cs2_diagnosis_score" in result.row
        assert "cs2_treatment_score" in result.row

    def test_row_preserves_original_fields(
        self,
        ctx: PluginContext,
        transform: AzureMultiQueryLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Output row preserves original input fields."""
        responses = [
            {"score": 85, "rationale": "R1"},
            {"score": 90, "rationale": "R2"},
            {"score": 75, "rationale": "R3"},
            {"score": 80, "rationale": "R4"},
        ]

        with chaosllm_azure_openai_responses(chaosllm_server, responses):
            row = {
                "cs1_bg": "bg1",
                "cs1_sym": "sym1",
                "cs1_hist": "hist1",
                "cs2_bg": "bg2",
                "cs2_sym": "sym2",
                "cs2_hist": "hist2",
                "original_field": "preserved",
            }
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        assert result.status == "success"
        assert result.row is not None
        # Original fields preserved
        assert result.row["original_field"] == "preserved"
        assert result.row["cs1_bg"] == "bg1"

    def test_row_fails_if_any_query_fails(
        self,
        ctx: PluginContext,
        transform: AzureMultiQueryLLMTransform,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """All-or-nothing: if any query fails, entire row fails."""
        responses = [
            {"score": 85, "rationale": "ok"},
            {"score": 85, "rationale": "ok"},
            {"score": 85, "rationale": "ok"},
            "not valid json",
        ]

        with chaosllm_azure_openai_responses(chaosllm_server, responses):
            row = {
                "cs1_bg": "bg",
                "cs1_sym": "sym",
                "cs1_hist": "hist",
                "cs2_bg": "bg",
                "cs2_sym": "sym",
                "cs2_hist": "hist",
            }
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)

        # Entire row fails
        assert result.status == "error"
        assert result.reason is not None
        assert "query_failed" in result.reason["reason"]

    def test_connect_output_required_before_accept(self, mock_recorder: Mock) -> None:
        """accept() raises RuntimeError if connect_output() not called."""
        transform = AzureMultiQueryLLMTransform(make_config())

        token = make_token("row-1")
        ctx = PluginContext(
            run_id="test-run",
            config={},
            state_id="test-state-id",
            token=token,
        )

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept({"text": "hello"}, ctx)

    def test_connect_output_cannot_be_called_twice(
        self,
        collector: CollectorOutputPort,
        mock_recorder: Mock,
    ) -> None:
        """connect_output() raises if called more than once."""
        transform = AzureMultiQueryLLMTransform(make_config())
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
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

        This test focuses on row-level pipelining via BatchTransformMixin,
        using sequential query execution (no pool_size) to avoid interference
        between query-level and row-level concurrency in the test.
        """
        # Disable query-level pooling to focus on row-level pipelining
        config = make_config()
        del config["pool_size"]

        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        # Use consistent response for all queries - test is about FIFO ordering
        response = {"score": 85, "rationale": "Good"}

        try:
            with chaosllm_azure_openai_responses(chaosllm_server, [response]):
                for i in range(3):
                    row = {
                        "row_id": f"row-{i}",
                        "cs1_bg": f"r{i}",
                        "cs1_sym": f"r{i}",
                        "cs1_hist": f"r{i}",
                        "cs2_bg": f"r{i}",
                        "cs2_sym": f"r{i}",
                        "cs2_hist": f"r{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = PluginContext(
                        run_id="run-123",
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
            assert result.row["row_id"] == f"row-{i}"

    def test_rows_use_shared_state_id_for_queries(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Each row's queries share a state_id (FK constraint).

        All queries for a row share ctx.state_id to satisfy FK constraint:
        - calls.state_id must reference existing node_states.state_id
        - Uniqueness comes from call_index allocated by recorder
        """
        transform = AzureMultiQueryLLMTransform(make_config())
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        responses = [{"score": i, "rationale": f"R{i}"} for i in range(8)]  # 2 rows x 4 queries
        created_state_ids: list[str] = []

        # Patch _get_llm_client to track state_ids
        original_get_client = transform._get_llm_client

        def tracking_get_client(state_id: str) -> Any:
            created_state_ids.append(state_id)
            return original_get_client(state_id)

        transform._get_llm_client = tracking_get_client  # type: ignore[method-assign]

        try:
            with chaosllm_azure_openai_responses(chaosllm_server, responses):
                for i in range(2):
                    row = {
                        "cs1_bg": f"r{i}",
                        "cs1_sym": f"r{i}",
                        "cs1_hist": f"r{i}",
                        "cs2_bg": f"r{i}",
                        "cs2_sym": f"r{i}",
                        "cs2_hist": f"r{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = PluginContext(
                        run_id="run-123",
                        config={},
                        landscape=mock_recorder,
                        state_id=f"batch-{i:03d}",
                        token=token,
                    )
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        # All state_ids should be from the set {batch-000, batch-001}
        unique_state_ids = set(created_state_ids)
        assert unique_state_ids == {"batch-000", "batch-001"}

        # Verify NO synthetic per-query state_ids were created
        for state_id in created_state_ids:
            assert "_q" not in state_id, f"Found synthetic per-query state_id: {state_id}"

    def test_clients_cleaned_up_after_row_processing(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Clients are cleaned up after each row is processed."""
        transform = AzureMultiQueryLLMTransform(make_config())
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        responses = [{"score": i, "rationale": f"R{i}"} for i in range(8)]

        try:
            with chaosllm_azure_openai_responses(chaosllm_server, responses):
                for i in range(2):
                    row = {
                        "cs1_bg": f"r{i}",
                        "cs1_sym": f"r{i}",
                        "cs1_hist": f"r{i}",
                        "cs2_bg": f"r{i}",
                        "cs2_sym": f"r{i}",
                        "cs2_hist": f"r{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = PluginContext(
                        run_id="run-123",
                        config={},
                        landscape=mock_recorder,
                        state_id=f"batch-{i:03d}",
                        token=token,
                    )
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=10.0)

            # After processing, clients should be cleaned up
            assert len(transform._llm_clients) == 0
        finally:
            transform.close()

    def test_on_start_captures_recorder(self, mock_recorder: Mock) -> None:
        """on_start() captures recorder reference for LLM client creation."""
        transform = AzureMultiQueryLLMTransform(make_config())

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

    def test_close_clears_recorder_and_clients(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
    ) -> None:
        """close() clears recorder reference and cached clients."""
        transform = AzureMultiQueryLLMTransform(make_config())
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        assert transform._recorder is not None

        transform.close()

        assert transform._recorder is None


class TestSequentialMode:
    """Tests for sequential mode (no pool_size) using accept() API."""

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

    def test_sequential_mode_processes_rows(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Sequential mode (no pool_size) processes rows correctly."""
        # Create config WITHOUT pool_size - forces sequential mode
        config = make_config()
        del config["pool_size"]

        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        # Verify no executor created (sequential mode)
        assert transform._executor is None

        responses = [{"score": i, "rationale": f"R{i}"} for i in range(4)]

        try:
            with chaosllm_azure_openai_responses(chaosllm_server, responses):
                row = {
                    "cs1_bg": "r1",
                    "cs1_sym": "r1",
                    "cs1_hist": "r1",
                    "cs2_bg": "r1",
                    "cs2_sym": "r1",
                    "cs2_hist": "r1",
                }
                token = make_token("row-1")
                ctx = PluginContext(
                    run_id="run-123",
                    config={},
                    landscape=mock_recorder,
                    state_id="batch-seq-001",
                    token=token,
                )
                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"

    def test_sequential_mode_cleans_up_clients(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Sequential mode cleans up clients after processing."""
        config = make_config()
        del config["pool_size"]

        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        responses = [{"score": i, "rationale": f"R{i}"} for i in range(8)]  # 2 rows x 4 queries

        try:
            with chaosllm_azure_openai_responses(chaosllm_server, responses):
                for i in range(2):
                    row = {
                        "cs1_bg": f"r{i}",
                        "cs1_sym": f"r{i}",
                        "cs1_hist": f"r{i}",
                        "cs2_bg": f"r{i}",
                        "cs2_sym": f"r{i}",
                        "cs2_hist": f"r{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = PluginContext(
                        run_id="run-123",
                        config={},
                        landscape=mock_recorder,
                        state_id=f"batch-seq-{i:03d}",
                        token=token,
                    )
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=10.0)

            # Clients should be cleaned up after each row
            assert len(transform._llm_clients) == 0
        finally:
            transform.close()


class TestPoolMetadataAuditIntegration:
    """Tests for pool metadata flowing to audit trail.

    P3-2026-02-02: Pooling metadata should be persisted to context_after_json.
    """

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

    def test_parallel_mode_includes_pool_stats_in_context_after(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Parallel mode should include pool_stats in TransformResult.context_after.

        This metadata enables auditors to verify:
        - Pool utilization (max_concurrent_reached)
        - Throttle behavior (capacity_retries, total_throttle_time_ms)
        - Config at completion time (dispatch_delay_at_completion_ms)
        """
        config = make_config(pool_size=4)
        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        # 4 queries per row (2 case studies x 2 criteria)
        responses = [{"score": i, "rationale": f"R{i}"} for i in range(4)]

        try:
            with chaosllm_azure_openai_responses(chaosllm_server, responses):
                row = {
                    "cs1_bg": "bg1",
                    "cs1_sym": "sym1",
                    "cs1_hist": "hist1",
                    "cs2_bg": "bg2",
                    "cs2_sym": "sym2",
                    "cs2_hist": "hist2",
                }
                token = make_token("row-1")
                ctx = PluginContext(
                    run_id="run-123",
                    config={},
                    landscape=mock_recorder,
                    state_id="state-pool-001",
                    token=token,
                )
                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        assert len(collector.results) == 1
        _, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"

        # VERIFY: context_after should contain pool metadata
        assert result.context_after is not None, (
            "TransformResult.context_after should contain pool metadata. "
            "_execute_queries_parallel must call executor.get_stats() and "
            "include it in the result."
        )

        # Verify pool_config is present
        assert "pool_config" in result.context_after, f"pool_config missing from context_after. Got: {result.context_after.keys()}"
        pool_config = result.context_after["pool_config"]
        assert pool_config["pool_size"] == 4

        # Verify pool_stats is present
        assert "pool_stats" in result.context_after, f"pool_stats missing from context_after. Got: {result.context_after.keys()}"
        pool_stats = result.context_after["pool_stats"]
        # max_concurrent_reached should be > 0 since queries ran in parallel
        assert "max_concurrent_reached" in pool_stats
        assert pool_stats["max_concurrent_reached"] > 0

    def test_parallel_mode_includes_query_ordering_in_context_after(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Parallel mode should include per-query ordering metadata.

        This enables auditors to:
        - Verify reordering worked correctly
        - Identify "lost" rows by examining ordering gaps
        - Measure buffer wait times
        """
        config = make_config(pool_size=4)
        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
        transform.on_start(init_ctx)
        transform.connect_output(collector, max_pending=10)

        responses = [{"score": i, "rationale": f"R{i}"} for i in range(4)]

        try:
            with chaosllm_azure_openai_responses(chaosllm_server, responses):
                row = {
                    "cs1_bg": "bg1",
                    "cs1_sym": "sym1",
                    "cs1_hist": "hist1",
                    "cs2_bg": "bg2",
                    "cs2_sym": "sym2",
                    "cs2_hist": "hist2",
                }
                token = make_token("row-1")
                ctx = PluginContext(
                    run_id="run-123",
                    config={},
                    landscape=mock_recorder,
                    state_id="state-ordering-001",
                    token=token,
                )
                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
        finally:
            transform.close()

        _, result, _ = collector.results[0]
        assert result.context_after is not None

        # Verify query_ordering is present
        assert "query_ordering" in result.context_after, f"query_ordering missing from context_after. Got: {result.context_after.keys()}"
        query_ordering = result.context_after["query_ordering"]

        # Should have 4 entries (2 case studies x 2 criteria)
        assert len(query_ordering) == 4, f"Expected 4 query ordering entries, got {len(query_ordering)}"

        # Each entry should have ordering metadata
        for entry in query_ordering:
            assert "submit_index" in entry
            assert "complete_index" in entry
            assert "buffer_wait_ms" in entry
            assert isinstance(entry["submit_index"], int)
            assert isinstance(entry["complete_index"], int)
            assert isinstance(entry["buffer_wait_ms"], float)
