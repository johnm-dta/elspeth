# tests/stress/llm/test_openrouter_multi_query_stress.py
"""Stress tests for OpenRouter Multi-Query LLM transform using ChaosLLM HTTP server.

These tests exercise the OpenRouter multi-query transform under harsh error conditions.
Multi-query transforms make N queries per row (case_studies x criteria), so
error injection affects multiple queries per row.

OpenRouter uses HTTP client (AuditedHTTPClient) rather than SDK, so these tests
also validate HTTP-level error handling in the multi-query context.
"""

from __future__ import annotations

import threading
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import NodeType, TransformErrorReason, TransformResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.openrouter_multi_query import OpenRouterMultiQueryLLMTransform

if TYPE_CHECKING:
    from elspeth.contracts.identity import TokenInfo
    from elspeth.engine.batch_adapter import ExceptionResult

from .conftest import (
    ChaosLLMHTTPFixture,
    generate_multi_query_rows,
    make_openrouter_multi_query_config,
    make_token,
)

# JSON template for multi-query responses
# Must match the output_mapping in make_openrouter_multi_query_config
MULTI_QUERY_JSON_TEMPLATE = '{"score": 5, "rationale": "Test assessment rationale"}'


class CollectingOutputPort:
    """Output port that collects results for verification.

    Implements the OutputPort protocol to receive results from BatchTransformMixin.
    Results are stored as (row_data, token) tuples for success,
    (error_reason, token) for errors.
    """

    def __init__(self) -> None:
        self.results: list[tuple[dict[str, Any], TokenInfo]] = []
        self.errors: list[tuple[TransformErrorReason, TokenInfo]] = []
        self._lock = threading.Lock()

    def emit(
        self,
        token: TokenInfo,
        result: TransformResult | ExceptionResult,
        state_id: str | None,
    ) -> None:
        """Accept a result from upstream transform.

        Routes to results or errors based on TransformResult.status.
        ExceptionResults are treated as errors.
        """
        # Handle ExceptionResult (plugin bugs)
        if hasattr(result, "exception"):
            with self._lock:
                self.errors.append(({"reason": "test_error", "error": f"exception: {result.exception}"}, token))
            return

        # Handle TransformResult
        if result.status == "success":
            row_data = result.row if result.row is not None else {}
            with self._lock:
                self.results.append((row_data, token))
        else:
            error_data: TransformErrorReason = result.reason if result.reason is not None else {"reason": "test_error"}
            with self._lock:
                self.errors.append((error_data, token))

    @property
    def total_count(self) -> int:
        """Total processed (success + error)."""
        with self._lock:
            return len(self.results) + len(self.errors)

    @property
    def success_count(self) -> int:
        """Count of successful results."""
        with self._lock:
            return len(self.results)

    @property
    def error_count(self) -> int:
        """Count of errors."""
        with self._lock:
            return len(self.errors)


def create_recorder_and_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[LandscapeRecorder, str, str]:
    """Create a recorder and start a run, returning (recorder, run_id, node_id)."""
    tmp_path = tmp_path_factory.mktemp("stress_audit")
    db_path = tmp_path / "audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(
        config={"test": "openrouter_multi_query_stress"},
        run_id=f"stress-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    schema = SchemaConfig.from_dict({"fields": "dynamic"})
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="openrouter_multi_query_llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )

    return recorder, run.run_id, node.node_id


@pytest.mark.stress
class TestOpenRouterMultiQueryLLMStress:
    """Stress tests for OpenRouter Multi-Query LLM transform.

    With 2 case studies x 2 criteria = 4 queries per row, error rates
    are amplified. These tests validate HTTP-based error handling
    in the multi-query context.
    """

    @pytest.mark.chaosllm(rate_limit_pct=10.0, internal_error_pct=5.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_partial_failure(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """When any query fails permanently, the entire row fails (all-or-nothing).

        Uses a mix of retryable (rate limits) and non-retryable (internal errors)
        to guarantee some permanent failures while testing AIMD recovery.

        Verifies:
        - All-or-nothing semantics enforced via HTTP
        - Rows with any failed query are errored
        - Successful rows have all 4 query outputs
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = OpenRouterMultiQueryLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)

        for i, row in enumerate(rows):
            token = make_token(f"row-{i}", f"token-{i}")
            row_record = recorder.create_row(
                run_id=run_id,
                source_node_id=node_id,
                row_index=i,
                data=row,
            )
            token_record = recorder.create_token(row_id=row_record.row_id)
            state = recorder.begin_node_state(
                token_id=token_record.token_id,
                node_id=node_id,
                run_id=run_id,
                step_index=0,
                input_data=row,
            )

            ctx = PluginContext(
                run_id=run_id,
                landscape=recorder,
                state_id=state.state_id,
                config={},
                token=token,
            )
            transform.accept(row, ctx)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        # Verify successful rows have all query outputs
        expected_output_prefixes = ["cs1_diagnosis", "cs1_treatment", "cs2_diagnosis", "cs2_treatment"]
        for result_row, _ in output.results:
            for prefix in expected_output_prefixes:
                assert f"{prefix}_score" in result_row, f"Missing {prefix}_score"
                assert f"{prefix}_rationale" in result_row, f"Missing {prefix}_rationale"

        # With AIMD retry, most rate-limited rows recover
        # Internal errors (5%) cause permanent failures: P(all 4 succeed) ≈ 0.95^4 = 81%
        # So expect 0-15 failures (0-30% of 50 rows), allowing for probabilistic variance
        # Note: If AIMD is very effective, we may see 0 failures
        assert output.success_count >= 35, f"Expected at least 70% success, got {output.success_count}/50"

    @pytest.mark.chaosllm(rate_limit_pct=25.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_rate_limit_all_queries_retry(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """AIMD backoff affects all queries when rate limited.

        Verifies:
        - Pipeline completes despite high rate limits
        - HTTP 429 responses trigger AIMD
        - Total requests > 4x rows due to retries
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = OpenRouterMultiQueryLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(30)

        for i, row in enumerate(rows):
            token = make_token(f"row-{i}", f"token-{i}")
            row_record = recorder.create_row(
                run_id=run_id,
                source_node_id=node_id,
                row_index=i,
                data=row,
            )
            token_record = recorder.create_token(row_id=row_record.row_id)
            state = recorder.begin_node_state(
                token_id=token_record.token_id,
                node_id=node_id,
                run_id=run_id,
                step_index=0,
                input_data=row,
            )

            ctx = PluginContext(
                run_id=run_id,
                landscape=recorder,
                state_id=state.state_id,
                config={},
                token=token,
            )
            transform.accept(row, ctx)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 30

        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] >= 30 * 4, "Expected at least one request per query"

    @pytest.mark.chaosllm(rate_limit_pct=20.0, service_unavailable_pct=10.0, invalid_json_pct=5.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_chaos_mode(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Multi-query transform survives mixed error types.

        With HTTP errors (429, 503) and malformed responses (invalid JSON),
        the transform should handle all error types gracefully.

        Verifies:
        - Pipeline doesn't crash
        - All rows reach terminal state
        - Mixed error types are handled
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = OpenRouterMultiQueryLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)

        for i, row in enumerate(rows):
            token = make_token(f"row-{i}", f"token-{i}")
            row_record = recorder.create_row(
                run_id=run_id,
                source_node_id=node_id,
                row_index=i,
                data=row,
            )
            token_record = recorder.create_token(row_id=row_record.row_id)
            state = recorder.begin_node_state(
                token_id=token_record.token_id,
                node_id=node_id,
                run_id=run_id,
                step_index=0,
                input_data=row,
            )

            ctx = PluginContext(
                run_id=run_id,
                landscape=recorder,
                state_id=state.state_id,
                config={},
                token=token,
            )
            transform.accept(row, ctx)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        # With mixed error types (rate limits, 503s, invalid JSON):
        # - Rate limits (20%) and 503s (10%) are retryable by AIMD
        # - Invalid JSON (5%) may cause permanent failures
        # AIMD may recover most retryable errors, so we verify pipeline completes
        # rather than expecting a specific number of failures
        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] >= 50 * 4, "Expected at least baseline requests"

        # If there are errors, verify they have proper structure
        for error, _ in output.errors:
            # Multi-query errors should have a reason field
            assert "reason" in error, "Error should have reason"

    @pytest.mark.chaosllm(preset="stress_aimd", template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_fifo_ordering(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """FIFO ordering is preserved for successful rows.

        With concurrent query processing, verify that row output order
        matches input order for successful rows.
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = OpenRouterMultiQueryLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)
        input_order = []

        for i, row in enumerate(rows):
            token = make_token(f"row-{i}", f"token-{i}")
            row_record = recorder.create_row(
                run_id=run_id,
                source_node_id=node_id,
                row_index=i,
                data=row,
            )
            token_record = recorder.create_token(row_id=row_record.row_id)
            state = recorder.begin_node_state(
                token_id=token_record.token_id,
                node_id=node_id,
                run_id=run_id,
                step_index=0,
                input_data=row,
            )

            ctx = PluginContext(
                run_id=run_id,
                landscape=recorder,
                state_id=state.state_id,
                config={},
                token=token,
            )
            input_order.append(row["id"])
            transform.accept(row, ctx)

        transform.flush_batch_processing()
        transform.close()

        assert output.total_count == 50

        # Verify FIFO ordering for successful rows
        output_ids = [r[0]["id"] for r in output.results]
        for i in range(len(output_ids) - 1):
            pos_i = input_order.index(output_ids[i])
            pos_next = input_order.index(output_ids[i + 1])
            assert pos_i < pos_next, f"FIFO ordering violated at position {i}"
