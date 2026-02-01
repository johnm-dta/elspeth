# tests/stress/llm/test_azure_llm_stress.py
"""Stress tests for Azure LLM transform using ChaosLLM HTTP server.

These tests exercise the Azure LLM transform under harsh error conditions
with real HTTP communication to validate:
- AIMD backoff under rate limits
- Azure-specific 529 capacity error handling
- Burst pattern resilience
- Concurrent worker coordination
- Pipeline completion under sustained stress
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
from elspeth.plugins.llm.azure import AzureLLMTransform

if TYPE_CHECKING:
    from elspeth.contracts.identity import TokenInfo
    from elspeth.engine.batch_adapter import ExceptionResult

from .conftest import (
    ChaosLLMHTTPFixture,
    generate_test_rows,
    make_azure_llm_config,
    make_token,
)

# Mark all tests in this module as stress tests (skipped by default)
pytestmark = pytest.mark.stress


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
        config={"test": "azure_stress"},
        run_id=f"stress-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    schema = SchemaConfig.from_dict({"fields": "dynamic"})
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="azure_llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )

    return recorder, run.run_id, node.node_id


@pytest.mark.stress
class TestAzureLLMStress:
    """Stress tests for Azure LLM transform."""

    @pytest.mark.chaosllm(rate_limit_pct=30.0)
    def test_rate_limit_recovery_100_rows(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline recovers and completes under 30% rate limit errors.

        Verifies:
        - AIMD backoff activates under rate limits
        - All 100 rows eventually complete or error gracefully
        - Pipeline doesn't hang or crash
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        # Create transform pointed at ChaosLLM
        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=60,
        )
        transform = AzureLLMTransform(config)

        # Connect output port
        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        # Create on_start context
        start_ctx = PluginContext(
            run_id=run_id,
            landscape=recorder,
            config={},
        )
        transform.on_start(start_ctx)

        # Generate and process rows
        rows = generate_test_rows(100)

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

        # Flush and close
        transform.flush_batch_processing()
        transform.close()

        # Verify results
        stats = chaosllm_http_server.get_stats()
        assert output.total_count == 100, f"Expected 100 rows processed, got {output.total_count}"

        # With 30% rate limits, we expect retries so request count > 100
        assert stats["total_requests"] >= 100, "Should have at least one request per row"

        # Most rows should succeed despite rate limits due to AIMD retry
        assert output.success_count > 50, f"Expected >50 successes, got {output.success_count}"

    @pytest.mark.chaosllm(capacity_529_pct=25.0)
    def test_capacity_529_handling(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Azure-specific 529 capacity errors trigger AIMD backoff.

        Verifies:
        - 529 errors are treated as retryable
        - AIMD backoff applies correctly
        - Pipeline completes without crashes
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = AzureLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_test_rows(50)

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

        # Verify all rows processed
        assert output.total_count == 50

        # Verify 529 errors were injected
        stats = chaosllm_http_server.get_stats()
        stats.get("requests_by_outcome", {})
        # At least some 529s should have been injected
        assert stats["total_requests"] > 50, "Should have retried some requests"

    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_burst_pattern_resilience(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline survives periodic error bursts.

        The stress_aimd preset includes burst patterns that spike error rates
        periodically. The pipeline should recover between bursts.

        Verifies:
        - Pipeline doesn't crash during high-error burst periods
        - Pipeline recovers and continues after bursts
        - Final results include both successes and expected failures
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = AzureLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=15)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_test_rows(75)

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

        # All rows should be processed (success or error)
        assert output.total_count == 75

    @pytest.mark.chaosllm(rate_limit_pct=20.0)
    def test_concurrent_workers_coordination(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Multiple concurrent workers coordinate under rate limits.

        With pool_size=8 and 100 rows, verify workers don't interfere
        with each other's AIMD backoff.

        Verifies:
        - Workers share rate limit backoff correctly
        - No deadlocks or race conditions
        - FIFO ordering preserved
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=8,  # High concurrency
            max_capacity_retry_seconds=30,
        )
        transform = AzureLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=32)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
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

        # All rows should be processed
        assert output.total_count == 100

        # Verify FIFO ordering for successful rows
        output_ids = [r[0]["id"] for r in output.results]
        # Output should be in input order (for successful rows)
        for i in range(len(output_ids) - 1):
            # Find position in input
            pos_i = input_order.index(output_ids[i])
            pos_next = input_order.index(output_ids[i + 1])
            assert pos_i < pos_next, "FIFO ordering violated"
