"""System tests for crash recovery and resume scenarios.

These tests verify that ELSPETH can recover from crashes and resume
processing, producing the same results as uninterrupted runs.

Per the test regime plan: "Recovery idempotence - Resume produces same
result as uninterrupted run."
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from elspeth.contracts import Determinism, PluginSchema, RoutingMode, SourceRow
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult


class _InputSchema(PluginSchema):
    """Input schema for test transforms."""

    id: str
    value: int


class _FailOnceTransform(BaseTransform):
    """Transform that fails on first attempt for specific rows, succeeds on retry.

    Used to test retry and recovery behavior.
    """

    name: ClassVar[str] = "fail_once"
    determinism: ClassVar[Determinism] = Determinism.DETERMINISTIC
    input_schema: ClassVar[type[_InputSchema]] = _InputSchema
    output_schema: ClassVar[type[_InputSchema]] = _InputSchema

    _attempt_count: ClassVar[dict[str, int]] = {}
    _fail_row_ids: ClassVar[set[str]] = set()

    def __init__(self) -> None:
        super().__init__({})

    @classmethod
    def configure(cls, fail_row_ids: set[str]) -> None:
        """Configure which row IDs should fail on first attempt."""
        cls._fail_row_ids = fail_row_ids
        cls._attempt_count.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset state for test isolation."""
        cls._attempt_count.clear()
        cls._fail_row_ids.clear()

    def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        row_id = row.get("id", "unknown")
        self._attempt_count[row_id] = self._attempt_count.get(row_id, 0) + 1

        if row_id in self._fail_row_ids and self._attempt_count[row_id] == 1:
            return TransformResult.error({"reason": "simulated_failure"})

        return TransformResult.success({**row, "attempts": self._attempt_count[row_id]})


def _build_linear_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple linear graph for testing."""
    graph = ExecutionGraph()

    graph.add_node("source", node_type="source", plugin_name=config.source.name)

    transform_ids: dict[int, str] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = f"transform_{i}"
        transform_ids[i] = node_id
        graph.add_node(node_id, node_type="transform", plugin_name=t.name)
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    if "default" in sink_ids:
        graph.add_edge(prev, sink_ids["default"], label="continue", mode=RoutingMode.MOVE)

    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._output_sink = "default" if "default" in sink_ids else next(iter(sink_ids))
    graph._route_resolution_map = {}

    return graph


class TestResumeIdempotence:
    """Tests for resume idempotence - same results whether interrupted or not."""

    @pytest.mark.skip(reason="Resume functionality requires checkpoint implementation")
    def test_resume_produces_same_result(self, tmp_path: Path) -> None:
        """Resume after interruption produces same final output.

        This test verifies the recovery idempotence property:
        - Run pipeline completely (baseline)
        - Run pipeline, interrupt at row N, resume
        - Both should produce identical output
        """
        pass


class TestRetryBehavior:
    """Tests for retry behavior during processing."""

    @pytest.mark.skip(reason="Requires on_error configuration for error-returning transforms")
    def test_pipeline_with_failed_transform_records_failure(self, tmp_path: Path) -> None:
        """A pipeline that has a failing transform records the failure."""
        from elspeth.engine.artifacts import ArtifactDescriptor

        _FailOnceTransform.configure(fail_row_ids={"row_2"})

        try:
            db = LandscapeDB.in_memory()

            class TestSource(_TestSourceBase):
                name = "test_source"
                output_schema = _InputSchema

                def load(self, ctx: Any) -> Any:
                    yield SourceRow.valid({"id": "row_1", "value": 100})
                    yield SourceRow.valid({"id": "row_2", "value": 200})  # Will fail
                    yield SourceRow.valid({"id": "row_3", "value": 300})

                def close(self) -> None:
                    pass

            source = TestSource()

            class TestSink(_TestSinkBase):
                name = "collect_sink"
                results: ClassVar[list[dict[str, Any]]] = []

                def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                    TestSink.results.extend(rows)
                    return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

                def close(self) -> None:
                    pass

            TestSink.results.clear()

            config = PipelineConfig(
                source=as_source(source),
                transforms=[_FailOnceTransform()],
                sinks={"default": as_sink(TestSink())},
            )

            orchestrator = Orchestrator(db)
            result = orchestrator.run(config, graph=_build_linear_graph(config))

            # Pipeline completes but with some failures
            assert result.status == "completed"
            assert result.rows_processed == 3
            # row_2 failed on first attempt
            assert result.rows_failed >= 1

            db.close()

        finally:
            _FailOnceTransform.reset()


class TestCheckpointRecovery:
    """Tests for checkpoint-based recovery."""

    @pytest.mark.skip(reason="Checkpoint recovery not yet implemented")
    def test_checkpoint_preserves_partial_progress(self, tmp_path: Path) -> None:
        """Checkpoint saves progress so resume doesn't re-process rows."""
        pass

    @pytest.mark.skip(reason="Checkpoint recovery not yet implemented")
    def test_checkpoint_across_process_restart(self, tmp_path: Path) -> None:
        """Checkpoint survives process restart (file-based)."""
        pass


class TestAggregationRecovery:
    """Tests for recovery of aggregation-in-progress."""

    @pytest.mark.skip(reason="Aggregation recovery requires stateful testing")
    def test_aggregation_state_recovers(self, tmp_path: Path) -> None:
        """Aggregation state is recovered after crash.

        Aggregations hold state (collected rows). Recovery must restore
        this state to produce correct results.
        """
        pass
