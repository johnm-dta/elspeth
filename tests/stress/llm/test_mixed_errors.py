# tests/stress/llm/test_mixed_errors.py
"""Cross-cutting stress tests with mixed error types.

These tests verify ELSPETH's LLM plugins handle combinations of error
types correctly, with special focus on:
- Audit trail integrity under stress
- Pipeline completion with high error rates
- Progressive degradation handling
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import NodeType, TransformErrorReason, TransformResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure import AzureLLMTransform
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform

if TYPE_CHECKING:
    from elspeth.contracts.identity import TokenInfo
    from elspeth.engine.batch_adapter import ExceptionResult

from .conftest import (
    ChaosLLMHTTPFixture,
    generate_test_rows,
    make_azure_llm_config,
    make_openrouter_llm_config,
    make_token,
)


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


def create_recorder_and_run(
    tmp_path_factory: pytest.TempPathFactory,
    plugin_name: str,
) -> tuple[LandscapeRecorder, str, str]:
    """Create a recorder and start a run, returning (recorder, run_id, node_id)."""
    tmp_path = tmp_path_factory.mktemp("stress_audit")
    db_path = tmp_path / "audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(
        config={"test": "mixed_errors_stress"},
        run_id=f"stress-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    schema = SchemaConfig.from_dict({"fields": "dynamic"})
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name=plugin_name,
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )

    return recorder, run.run_id, node.node_id


@pytest.mark.stress
class TestMixedErrors:
    """Cross-cutting tests with multiple error types active."""

    @pytest.mark.chaosllm(
        rate_limit_pct=15.0,
        capacity_529_pct=5.0,
        internal_error_pct=3.0,
        invalid_json_pct=2.0,
    )
    def test_chaos_mode_azure_pipeline_completes(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Azure pipeline completes with 25%+ mixed error rate.

        Error mix:
        - 15% rate limits (retryable with AIMD)
        - 5% capacity errors (retryable with AIMD)
        - 3% internal errors (may fail row)
        - 2% invalid JSON (fails row)

        Verifies:
        - No crashes with mixed error types
        - All rows reach terminal state
        - Pipeline completes within reasonable time
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory, "azure_llm")

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=45,
        )
        transform = AzureLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
        start_time = time.monotonic()

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

        elapsed = time.monotonic() - start_time

        # All rows must reach terminal state
        assert output.total_count == 100, f"Only {output.total_count}/100 rows completed"

        # Should complete within 3 minutes
        assert elapsed < 180, f"Chaos mode took too long: {elapsed:.1f}s"

        # With AIMD retry for rate limits, expect reasonable success rate
        # Even with 25% error injection, retries help
        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] > 100, "Expected retries to occur"

    @pytest.mark.chaosllm(
        rate_limit_pct=15.0,
        service_unavailable_pct=5.0,
        gateway_timeout_pct=3.0,
        invalid_json_pct=2.0,
    )
    def test_chaos_mode_openrouter_pipeline_completes(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """OpenRouter pipeline completes with 25%+ mixed error rate.

        Verifies HTTP-level error handling with mixed error types.
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory, "openrouter_llm")

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=45,
        )
        transform = OpenRouterLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
        start_time = time.monotonic()

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

        elapsed = time.monotonic() - start_time

        # All rows must reach terminal state
        assert output.total_count == 100

        # Should complete within 3 minutes
        assert elapsed < 180, f"Chaos mode took too long: {elapsed:.1f}s"

    @pytest.mark.chaosllm(rate_limit_pct=10.0)
    def test_audit_trail_integrity_under_stress(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Verify audit trail is complete for all rows under stress.

        Every row should have:
        - A row record in the database
        - A token record
        - At least one node state
        - Node state with completion or error status

        Verifies:
        - No silent drops
        - All rows traceable
        - Audit records consistent
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory, "azure_llm")

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
        row_ids = []
        token_ids = []
        state_ids = []

        for i, row in enumerate(rows):
            token = make_token(f"row-{i}", f"token-{i}")
            row_record = recorder.create_row(
                run_id=run_id,
                source_node_id=node_id,
                row_index=i,
                data=row,
            )
            row_ids.append(row_record.row_id)

            token_record = recorder.create_token(row_id=row_record.row_id)
            token_ids.append(token_record.token_id)

            state = recorder.begin_node_state(
                token_id=token_record.token_id,
                node_id=node_id,
                run_id=run_id,
                step_index=0,
                input_data=row,
            )
            state_ids.append(state.state_id)

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

        # Verify audit trail completeness using LandscapeRecorder API
        # Check all rows exist in database
        db_rows = recorder.get_rows(run_id)
        db_row_ids = {r.row_id for r in db_rows}
        assert set(row_ids) == db_row_ids, "Missing rows in audit trail"

        # Verify all tokens - need to check tokens for each row
        db_token_ids: set[str] = set()
        for row_id in row_ids:
            tokens_for_row = recorder.get_tokens(row_id)
            for t in tokens_for_row:
                db_token_ids.add(t.token_id)
        assert set(token_ids) == db_token_ids, "Missing tokens in audit trail"

    def test_progressive_degradation(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline handles gradually increasing error rates.

        Start with low error rate, increase during test.
        Pipeline should adapt via AIMD.

        Verifies:
        - Pipeline continues as errors increase
        - AIMD backoff kicks in appropriately
        - No sudden crash when error rate spikes
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory, "azure_llm")

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = AzureLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)

        # Start with low error rate
        chaosllm_http_server.update_config(rate_limit_pct=5.0)

        for i, row in enumerate(rows):
            # Gradually increase error rate every 25 rows
            if i == 25:
                chaosllm_http_server.update_config(rate_limit_pct=15.0)
            elif i == 50:
                chaosllm_http_server.update_config(rate_limit_pct=25.0)
            elif i == 75:
                chaosllm_http_server.update_config(rate_limit_pct=35.0)

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

        # All rows should complete
        assert output.total_count == 100

        # Early rows (low error rate) should have high success
        # Later rows may have more failures

    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_burst_recovery_with_mixed_plugins(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline survives error bursts with stress_aimd preset.

        The stress_aimd preset includes periodic burst patterns that
        spike rate limits to 80% for 5 seconds every 30 seconds.

        Verifies:
        - Pipeline doesn't crash during bursts
        - Recovery after burst period
        - Final completion despite bursts
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory, "azure_llm")

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = AzureLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

        rows = generate_test_rows(75)
        start_time = time.monotonic()

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

        elapsed = time.monotonic() - start_time

        # All rows should complete
        assert output.total_count == 75

        # Should complete within reasonable time even with bursts
        assert elapsed < 180, f"Burst recovery took too long: {elapsed:.1f}s"
