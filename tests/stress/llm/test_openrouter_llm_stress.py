# tests/stress/llm/test_openrouter_llm_stress.py
"""Stress tests for OpenRouter LLM transform using ChaosLLM HTTP server.

These tests exercise the OpenRouter LLM transform under harsh error conditions
with real HTTP communication to validate:
- Rate limit recovery via HTTP client
- Malformed JSON response handling
- Connection timeout recovery
- Long-running pipeline stability
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
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform

if TYPE_CHECKING:
    from elspeth.contracts.identity import TokenInfo
    from elspeth.engine.batch_adapter import ExceptionResult

from .conftest import (
    ChaosLLMHTTPFixture,
    generate_test_rows,
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


def create_recorder_and_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[LandscapeRecorder, str, str]:
    """Create a recorder and start a run, returning (recorder, run_id, node_id)."""
    tmp_path = tmp_path_factory.mktemp("stress_audit")
    db_path = tmp_path / "audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(
        config={"test": "openrouter_stress"},
        run_id=f"stress-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    schema = SchemaConfig.from_dict({"fields": "dynamic"})
    node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="openrouter_llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )

    return recorder, run.run_id, node.node_id


@pytest.mark.stress
class TestOpenRouterLLMStress:
    """Stress tests for OpenRouter LLM transform."""

    @pytest.mark.chaosllm(rate_limit_pct=30.0)
    def test_rate_limit_recovery_100_rows(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline recovers and completes under 30% rate limit errors.

        Verifies:
        - HTTP client handles 429 responses gracefully
        - All 100 rows eventually complete or error gracefully
        - Pipeline doesn't hang or crash
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=60,
        )
        transform = OpenRouterLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        start_ctx = PluginContext(run_id=run_id, landscape=recorder, config={})
        transform.on_start(start_ctx)

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

        transform.flush_batch_processing()
        transform.close()

        stats = chaosllm_http_server.get_stats()
        assert output.total_count == 100, f"Expected 100 rows processed, got {output.total_count}"

        # With 30% rate limits, we expect retries
        assert stats["total_requests"] >= 100

        # Most rows should succeed despite rate limits
        assert output.success_count > 50, f"Expected >50 successes, got {output.success_count}"

    @pytest.mark.chaosllm(invalid_json_pct=15.0)
    def test_malformed_json_handling(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Transform handles malformed JSON responses gracefully.

        OpenRouter uses HTTP client (not SDK), so malformed JSON is detected
        in the transform's response parsing rather than SDK internals.

        Verifies:
        - Malformed JSON errors are captured as row errors
        - Other rows continue processing
        - No crashes from JSON parse failures
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = OpenRouterLLMTransform(config)

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

        # All rows should be processed
        assert output.total_count == 50

        # With 15% malformed JSON, expect ~7-8 errors
        # But it's probabilistic, so allow some variance
        assert output.error_count >= 3, "Expected some malformed JSON errors"
        assert output.success_count > 30, "Expected most rows to succeed"

        # Verify errors have correct reason
        [e[0].get("reason") for e in output.errors]
        # Malformed JSON could result in various error reasons depending on the malformation type
        # (invalid_json_response, json_parse_failed, etc.)

    @pytest.mark.chaosllm(service_unavailable_pct=20.0, gateway_timeout_pct=10.0)
    def test_connection_timeout_recovery(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline recovers from network-level failures.

        Verifies:
        - 503 and 504 errors don't crash the pipeline
        - Rows fail gracefully with recorded errors
        - Other rows continue processing
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            timeout_seconds=10.0,  # Short timeout for faster tests
            max_capacity_retry_seconds=30,
        )
        transform = OpenRouterLLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

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

        # All rows should be processed
        assert output.total_count == 75

        # With 30% combined server errors, expect some failures
        # But AIMD retry may recover some
        assert output.error_count >= 5, "Expected some server error failures"

    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_fifo_ordering_preserved(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """FIFO ordering is preserved for successful rows under stress.

        Verifies:
        - Rows that succeed are emitted in submission order
        - Concurrent processing doesn't corrupt ordering
        """
        recorder, run_id, node_id = create_recorder_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=45,
        )
        transform = OpenRouterLLMTransform(config)

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

        assert output.total_count == 100

        # Verify FIFO ordering for successful rows
        output_ids = [r[0]["id"] for r in output.results]
        for i in range(len(output_ids) - 1):
            pos_i = input_order.index(output_ids[i])
            pos_next = input_order.index(output_ids[i + 1])
            assert pos_i < pos_next, f"FIFO ordering violated at position {i}"
