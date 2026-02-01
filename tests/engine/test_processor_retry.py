# tests/engine/test_processor_retry.py
"""Tests for RowProcessor work queue, retry, and recovery functionality.

This module contains tests extracted from test_processor.py focusing on:
- Work queue iteration guards and fork child execution
- Retry integration with RetryManager
- Recovery support with restored aggregation state

Test plugins inherit from base classes (BaseTransform) because the processor
uses isinstance() for type-safe plugin detection.
"""

from typing import TYPE_CHECKING, Any

from elspeth.contracts import NodeType, RoutingMode
from elspeth.contracts.types import GateName, NodeID
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    RowOutcome,
    TransformResult,
)
from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB


class TestRowProcessorWorkQueue:
    """Work queue tests for fork child execution."""

    def test_work_queue_iteration_guard_prevents_infinite_loop(self, monkeypatch: Any, landscape_db: "LandscapeDB") -> None:
        """Work queue should fail if iterations exceed limit.

        This test verifies that the iteration guard protects against bugs that
        could cause infinite loops by continuously re-enqueuing work items.
        """
        import pytest

        import elspeth.engine.processor as proc_module
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor, _WorkItem
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
        )

        # Create a mock _process_single_token that always re-enqueues work
        def infinite_loop_process(
            self: RowProcessor,
            token: TokenInfo,
            transforms: list[Any],
            ctx: PluginContext,
            start_step: int,
            coalesce_at_step: int | None = None,
            coalesce_name: str | None = None,
        ) -> tuple[None, list[_WorkItem]]:
            # Always return a new work item, simulating a bug that causes infinite loop
            return (None, [_WorkItem(token=token, start_step=0)])

        # Patch MAX_WORK_QUEUE_ITERATIONS to a small number for testing
        original_max = proc_module.MAX_WORK_QUEUE_ITERATIONS
        proc_module.MAX_WORK_QUEUE_ITERATIONS = 5

        try:
            # Patch the method to create infinite work
            monkeypatch.setattr(RowProcessor, "_process_single_token", infinite_loop_process)

            ctx = PluginContext(run_id=run.run_id, config={})

            # Should raise RuntimeError when iterations exceed limit
            with pytest.raises(RuntimeError, match=r"Work queue exceeded \d+ iterations"):
                processor.process_row(
                    row_index=0,
                    row_data={"value": 42},
                    transforms=[],
                    ctx=ctx,
                )
        finally:
            proc_module.MAX_WORK_QUEUE_ITERATIONS = original_max

    def test_fork_children_are_executed_through_work_queue(self, landscape_db: "LandscapeDB") -> None:
        """Fork child tokens should be processed, not orphaned."""
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes (transform before gate since config gates run after transforms)
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enricher",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=transform_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=transform_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Create transform that marks execution
        class MarkerTransform(BaseTransform):
            name = "enricher"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "processed": True}, success_reason={"action": "processed"})

        # Config-driven fork gate
        splitter_gate = GateSettings(
            name="splitter",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )

        transform = MarkerTransform(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            edge_map={
                (NodeID(gate_node.node_id), "path_a"): edge_a.edge_id,
                (NodeID(gate_node.node_id), "path_b"): edge_b.edge_id,
            },
            config_gates=[splitter_gate],
            config_gate_id_map={GateName("splitter"): NodeID(gate_node.node_id)},
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process row - should return multiple results (parent + children)
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform],
            ctx=ctx,
        )

        # Should have 3 results: parent (FORKED) + 2 children (COMPLETED)
        assert isinstance(results, list)
        assert len(results) == 3

        # Parent should be FORKED
        forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
        assert len(forked_results) == 1

        # Children should be COMPLETED and processed (all tokens have processed=True)
        completed_results = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed_results) == 2
        for result in completed_results:
            # Direct access - we know the field exists because we just set it
            assert result.final_data["processed"] is True
            assert result.token.branch_name in ("path_a", "path_b")


class TestRowProcessorRetry:
    """Tests for retry integration in RowProcessor."""

    def test_processor_accepts_retry_manager(self) -> None:
        """RowProcessor can be constructed with RetryManager."""
        from unittest.mock import Mock

        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.retry import RetryManager

        retry_manager = RetryManager(RuntimeRetryConfig(max_attempts=3, base_delay=1.0, max_delay=60.0, jitter=1.0, exponential_base=2.0))

        # Should not raise
        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id=NodeID("source-node"),
            retry_manager=retry_manager,
        )

        assert processor._retry_manager is retry_manager

    def test_retries_transient_transform_exception(self) -> None:
        """Transform exceptions are retried up to max_attempts."""
        from unittest.mock import Mock

        from elspeth.contracts import TransformResult
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.retry import RetryManager

        # Track call count
        call_count = 0

        def flaky_execute(*args: Any, **kwargs: Any) -> tuple[Any, Any, None]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient network error")
            # Return success on 3rd attempt
            return (
                TransformResult.success({"result": "ok"}, success_reason={"action": "test"}),
                Mock(
                    token_id="t1",
                    row_id="r1",
                    row_data={"result": "ok"},
                    branch_name=None,
                ),
                None,  # error_sink
            )

        # Create processor with mocked internals
        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id=NodeID("source"),
            retry_manager=RetryManager(
                RuntimeRetryConfig(max_attempts=3, base_delay=0.01, max_delay=60.0, jitter=0.0, exponential_base=2.0)
            ),
        )

        # Mock the transform executor
        processor._transform_executor = Mock()
        processor._transform_executor.execute_transform.side_effect = flaky_execute

        # Create test transform
        transform = Mock()
        transform.node_id = "transform-1"

        # Create test token
        token = Mock()
        token.token_id = "t1"
        token.row_id = "r1"
        token.row_data = {"input": 1}
        token.branch_name = None

        ctx = Mock()
        ctx.run_id = "test-run"

        # Call the retry wrapper directly
        result, _out_token, _error_sink = processor._execute_transform_with_retry(
            transform=transform,
            token=token,
            ctx=ctx,
            step=0,
        )

        # Should have retried and succeeded
        assert call_count == 3
        assert result.status == "success"

    def test_no_retry_when_retry_manager_is_none(self) -> None:
        """Without retry_manager, retryable exceptions become error results (not propagated).

        This keeps failures row-scoped instead of aborting the entire run.
        The error result can still be routed to an error sink via on_error config.
        """
        from unittest.mock import Mock

        from elspeth.engine.processor import RowProcessor

        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id=NodeID("source"),
            retry_manager=None,  # No retry
        )

        processor._transform_executor = Mock()
        processor._transform_executor.execute_transform.side_effect = ConnectionError("network fail")

        transform = Mock()
        transform.node_id = "t1"
        transform._on_error = "error_sink"  # Configure error routing
        token = Mock(token_id="t1", row_id="r1", row_data={}, branch_name=None)
        ctx = Mock(run_id="test-run")

        # Should NOT raise - converts to error result to keep failure row-scoped
        result, _, error_sink = processor._execute_transform_with_retry(transform, token, ctx, step=0)

        # Should only be called once (no retry)
        assert processor._transform_executor.execute_transform.call_count == 1

        # Error result returned instead of exception propagated
        assert result.status == "error"
        assert result.retryable is True
        assert result.reason is not None
        assert "network fail" in result.reason["error"]
        assert result.reason["reason"] == "transient_error_no_retry"

        # Error sink from transform config is returned for routing
        assert error_sink == "error_sink"

    def test_llm_retryable_error_without_retry_manager_becomes_error_result(self) -> None:
        """Retryable LLMClientError becomes error result when retry_manager is None.

        This addresses P2 review comment: LLM transforms re-raise retryable errors
        expecting RetryManager to catch them. When retry_manager is None, the
        processor must catch these and convert to error results to avoid aborting
        the entire run.
        """
        from unittest.mock import Mock

        from elspeth.engine.processor import RowProcessor
        from elspeth.plugins.clients.llm import LLMClientError

        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id=NodeID("source"),
            retry_manager=None,  # No retry configured
        )

        processor._transform_executor = Mock()
        # Simulate retryable LLM error (rate limit, network, server error)
        llm_error = LLMClientError("Rate limit exceeded", retryable=True)
        processor._transform_executor.execute_transform.side_effect = llm_error

        transform = Mock()
        transform.node_id = "llm_transform"
        transform._on_error = "quarantine"
        token = Mock(token_id="t1", row_id="r1", row_data={}, branch_name=None)
        ctx = Mock(run_id="test-run")

        # Should NOT raise - converts to error result
        result, _, error_sink = processor._execute_transform_with_retry(transform, token, ctx, step=0)

        # Single attempt (no retry)
        assert processor._transform_executor.execute_transform.call_count == 1

        # Error result with LLM-specific reason
        assert result.status == "error"
        assert result.retryable is True
        assert result.reason is not None
        assert result.reason["reason"] == "llm_retryable_error_no_retry"
        assert "Rate limit exceeded" in result.reason["error"]
        assert error_sink == "quarantine"

    def test_llm_non_retryable_error_propagates(self) -> None:
        """Non-retryable LLMClientError propagates (already handled by transform)."""
        from unittest.mock import Mock

        import pytest

        from elspeth.engine.processor import RowProcessor
        from elspeth.plugins.clients.llm import LLMClientError

        processor = RowProcessor(
            recorder=Mock(),
            span_factory=Mock(),
            run_id="test-run",
            source_node_id=NodeID("source"),
            retry_manager=None,
        )

        processor._transform_executor = Mock()
        # Non-retryable error (content policy, context length exceeded)
        llm_error = LLMClientError("Content policy violation", retryable=False)
        processor._transform_executor.execute_transform.side_effect = llm_error

        transform = Mock()
        transform.node_id = "llm_transform"
        token = Mock(token_id="t1", row_id="r1", row_data={}, branch_name=None)
        ctx = Mock(run_id="test-run")

        # Non-retryable errors should propagate (transform should have handled them)
        with pytest.raises(LLMClientError) as exc_info:
            processor._execute_transform_with_retry(transform, token, ctx, step=0)

        assert exc_info.value.retryable is False

    def test_max_retries_exceeded_returns_failed_outcome(self, landscape_db: "LandscapeDB") -> None:
        """When all retries exhausted, process_row returns FAILED outcome."""

        from elspeth.contracts import RowOutcome
        from elspeth.contracts.config import RuntimeRetryConfig
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.retry import RetryManager
        from elspeth.engine.spans import SpanFactory

        # Set up real Landscape
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="always_fails",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        class AlwaysFailsTransform(BaseTransform):
            """Transform that always raises transient error."""

            name = "always_fails"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                raise ConnectionError("Network always down")

        # Create processor with retry (max 2 attempts, fast delays for test)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            retry_manager=RetryManager(
                RuntimeRetryConfig(max_attempts=2, base_delay=0.01, max_delay=60.0, jitter=0.0, exponential_base=2.0)
            ),
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process should return FAILED, not raise MaxRetriesExceeded
        results = processor.process_row(
            row_index=0,
            row_data={"x": 1},
            transforms=[AlwaysFailsTransform(transform_node.node_id)],
            ctx=ctx,
        )

        # Should get a result, not an exception
        assert len(results) == 1
        result = results[0]

        # Outcome should be FAILED
        assert result.outcome == RowOutcome.FAILED

        # Error info should be captured
        assert result.error is not None
        assert "MaxRetriesExceeded" in str(result.error) or "attempts" in str(result.error)


class TestRowProcessorRecovery:
    """Tests for RowProcessor recovery support."""

    def test_processor_accepts_restored_aggregation_state(self, landscape_db: "LandscapeDB") -> None:
        """RowProcessor passes restored state to AggregationExecutor."""
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        restored_state = {
            NodeID("agg_node"): {"buffer": [1, 2], "count": 2},
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID("source"),
            edge_map={},
            route_resolution_map={},
            restored_aggregation_state=restored_state,  # New parameter
        )

        # Verify state was passed to executor
        assert processor._aggregation_executor.get_restored_state(NodeID("agg_node")) == {
            "buffer": [1, 2],
            "count": 2,
        }


class TestNoRetryAuditCompleteness:
    """Tests for audit trail completeness when retry_manager is None.

    BUG: When retry is disabled and retryable exceptions occur:
    1. No transform_error is recorded (bypasses TransformExecutor error handling)
    2. If on_error is None, creates invalid ROUTED outcome with sink_name=None

    These tests verify the audit trail remains complete even without retry.
    """

    def test_no_retry_retryable_exception_records_transform_error(self, landscape_db: "LandscapeDB") -> None:
        """Retryable exceptions in no-retry mode must record transform_error.

        P2 review comment: In the no-retry path, retryable exceptions are converted
        into TransformResult.error and returned directly. Because this bypasses
        TransformExecutor's error-routing logic, no transform_errors entry is
        recorded, so when retry is disabled and a rate-limit/network error happens,
        the row is routed but explain() has no transform_error details.

        Expected: transform_errors table should have an entry for the error.
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.core.landscape.schema import transform_errors_table
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.clients.llm import LLMClientError

        # Set up real Landscape
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        class RateLimitedTransform(BaseTransform):
            """Transform that raises retryable LLM error."""

            name = "rate_limited"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id
                # Explicitly set _on_error (normally done by config mixins)
                self._on_error = "error_sink"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                raise LLMClientError("Rate limit exceeded", retryable=True)

        # No retry manager - single attempt
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            retry_manager=None,
        )

        # Must pass landscape for record_transform_error to work
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)

        # Process should return error result (not raise)
        results = processor.process_row(
            row_index=0,
            row_data={"x": 1},
            transforms=[RateLimitedTransform(transform_node.node_id)],
            ctx=ctx,
        )

        assert len(results) == 1
        result = results[0]
        assert result.outcome == RowOutcome.ROUTED

        # CRITICAL: transform_errors must have an entry for explain() to work
        with landscape_db.engine.connect() as conn:
            errors = conn.execute(transform_errors_table.select().where(transform_errors_table.c.run_id == run.run_id)).fetchall()

        assert len(errors) >= 1, (
            "No transform_error recorded when retryable exception occurred in no-retry mode. "
            "This breaks explain() - the row shows ROUTED but there's no transform_error "
            "explaining why. The error details are lost."
        )

        # Verify error has expected content
        error = errors[0]
        assert "rate" in error.error_details_json.lower() or "limit" in error.error_details_json.lower(), (
            f"transform_error should mention rate limit, got: {error.error_details_json}"
        )

    def test_no_retry_with_on_error_none_raises_instead_of_invalid_routed(self, landscape_db: "LandscapeDB") -> None:
        """When on_error is None and retryable exception occurs, should fail properly.

        P2 review comment: In the no-retry-manager path, retryable exceptions are
        converted into TransformResult.error and returned with transform._on_error
        even if _on_error is None. For pipelines that omit on_error (the default)
        and hit a retryable ConnectionError/LLMClientError, process_row records a
        ROUTED outcome with sink_name=None and the orchestrator later asserts on
        result.sink_name, leaving a bogus terminal outcome in the audit trail
        before crashing.

        Expected: Should raise RuntimeError (like TransformExecutor does) when
        on_error is None and an error result is produced.
        """
        import pytest

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        # Set up real Landscape
        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="network_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        class NetworkFailTransform(BaseTransform):
            """Transform that raises retryable network error with no on_error."""

            name = "network_fail"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self, node_id: str) -> None:
                # NO on_error configured - this is the default
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                raise ConnectionError("Network unreachable")

        # No retry manager - single attempt
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            retry_manager=None,
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Should raise RuntimeError, not return invalid ROUTED with sink_name=None
        with pytest.raises(RuntimeError) as exc_info:
            processor.process_row(
                row_index=0,
                row_data={"x": 1},
                transforms=[NetworkFailTransform(transform_node.node_id)],
                ctx=ctx,
            )

        # Error message should explain the problem
        assert "on_error" in str(exc_info.value).lower(), f"RuntimeError should mention missing on_error config, got: {exc_info.value}"
