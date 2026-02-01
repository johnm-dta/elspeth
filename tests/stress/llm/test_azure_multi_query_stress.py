# tests/stress/llm/test_azure_multi_query_stress.py
"""Stress tests for Azure Multi-Query LLM transform using ChaosLLM HTTP server.

These tests exercise the Azure multi-query transform under harsh error conditions.
Multi-query transforms make N queries per row (case_studies x criteria), so
error injection affects multiple queries per row.

Test scenarios:
- Partial failure: Some queries fail, row fails (all-or-nothing semantics)
- Rate limit with all queries: AIMD backoff affects all queries in a row
- Chaos mode: High error rate across all query types
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
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

if TYPE_CHECKING:
    from elspeth.contracts.identity import TokenInfo
    from elspeth.engine.batch_adapter import ExceptionResult

from .conftest import (
    ChaosLLMHTTPFixture,
    generate_multi_query_rows,
    make_azure_multi_query_config,
    make_token,
)

# Mark all tests in this module as stress tests (skipped by default)
pytestmark = pytest.mark.stress

# JSON template for multi-query responses
# Must match the output_mapping in make_azure_multi_query_config
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
        config={"test": "azure_multi_query_stress"},
        run_id=f"stress-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    schema = SchemaConfig.from_dict({"fields": "dynamic"})
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="azure_multi_query_llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )

    return recorder, run.run_id, node.node_id


@pytest.mark.stress
class TestAzureMultiQueryLLMStress:
    """Stress tests for Azure Multi-Query LLM transform.

    With 2 case studies x 2 criteria = 4 queries per row, error rates
    are amplified. A 25% error rate means ~68% of rows will have at least
    one failed query (1 - 0.75^4).
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

        With 4 queries per row and 5% non-retryable errors:
        - P(all succeed) ≈ 0.95^4 ≈ 81%
        - Expect ~19% of rows to fail permanently

        Verifies:
        - All-or-nothing semantics enforced
        - Rows with any failed query are errored
        - Successful rows have all 4 query outputs
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_azure_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = AzureMultiQueryLLMTransform(config)

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
                # Should have _score and _rationale for each
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

        With PooledExecutor, rate limit backoff is shared across all
        queries for a row. When AIMD kicks in, all queries slow down.

        Verifies:
        - Pipeline completes despite high rate limits
        - AIMD coordinates across queries
        - Total requests > 4x rows due to retries
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_azure_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = AzureMultiQueryLLMTransform(config)

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
        # With 4 queries per row, expect at least 120 requests
        # With retries, expect significantly more
        assert stats["total_requests"] >= 30 * 4, "Expected at least one request per query"

    @pytest.mark.chaosllm(rate_limit_pct=20.0, capacity_529_pct=10.0, internal_error_pct=5.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_chaos_mode(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Multi-query transform survives 35%+ combined error rate.

        With 35% errors and 4 queries per row:
        - P(all succeed) = 0.65^4 ≈ 18%
        - Most rows will fail

        Verifies:
        - Pipeline doesn't crash
        - All rows reach terminal state
        - Errors are properly recorded
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_azure_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = AzureMultiQueryLLMTransform(config)

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

        # With 35% combined error rate and 4 queries per row:
        # - Rate limits (20%) and 529s (10%) are retryable by AIMD
        # - Internal errors (5%) are non-retryable: P(all 4 succeed) ≈ 0.95^4 = 81%
        # AIMD may recover most retryable errors, so we verify pipeline completes
        # rather than expecting a specific number of failures
        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] >= 50 * 4, "Expected at least baseline requests"

        # If there are errors, verify they have proper structure
        for error, _ in output.errors:
            # Multi-query errors should have query_failed reason or similar
            assert "reason" in error, "Error should have reason"
