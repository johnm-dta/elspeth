# tests/integration/pipeline/test_concurrency.py
"""Integration tests for concurrency config wiring.

These tests verify that RuntimeConcurrencyConfig is properly wired through
the CLI -> Orchestrator -> RowProcessor -> TransformExecutor pipeline.

STATUS: IMPLEMENTED
- Orchestrator: accepts concurrency_config parameter
- TransformExecutor: accepts max_workers parameter
- RowProcessor: accepts max_workers and forwards to TransformExecutor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.config import RuntimeConcurrencyConfig
from elspeth.contracts.types import NodeID
from elspeth.core.config import ConcurrencySettings, SourceSettings
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.processor import DAGTraversalContext, RowProcessor
from elspeth.engine.spans import SpanFactory
from tests.fixtures.landscape import make_factory
from tests.fixtures.pipeline import build_linear_pipeline

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB


def _build_traversal_from_graph() -> tuple[NodeID, str, DAGTraversalContext]:
    """Build a DAGTraversalContext from a production-path graph.

    Returns (source_node_id, source_on_success, traversal_context).
    Uses build_linear_pipeline -> ExecutionGraph.from_plugin_instances()
    to avoid BUG-LINEAGE-01 manual construction.
    """
    source, _, _, graph = build_linear_pipeline([{"id": 1}])
    source_node_id = graph.get_source()
    assert source_node_id is not None

    node_to_next: dict[NodeID, NodeID | None] = {}
    for nid in graph.get_pipeline_node_sequence():
        node_to_next[nid] = graph.get_next_node(nid)

    traversal = DAGTraversalContext(
        node_step_map=graph.build_step_map(),
        node_to_plugin={},
        first_transform_node_id=graph.get_first_transform_node(),
        node_to_next=node_to_next,
        coalesce_node_map=graph.get_coalesce_id_map(),
        branch_first_node=graph.get_branch_first_nodes(),
    )
    return source_node_id, source.on_success, traversal


class TestConcurrencyConfigInOrchestrator:
    """Test that RuntimeConcurrencyConfig is properly passed to Orchestrator."""

    def test_orchestrator_accepts_concurrency_config(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator constructor accepts concurrency_config parameter."""
        config = RuntimeConcurrencyConfig(max_workers=8)

        # Should not raise
        orchestrator = Orchestrator(landscape_db, concurrency_config=config)
        assert orchestrator._concurrency_config is config
        assert orchestrator._concurrency_config.max_workers == 8

    def test_orchestrator_accepts_none_config(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator works without concurrency config."""
        orchestrator = Orchestrator(landscape_db, concurrency_config=None)
        assert orchestrator._concurrency_config is None

    def test_config_from_settings(self) -> None:
        """RuntimeConcurrencyConfig.from_settings() creates config correctly."""
        settings = ConcurrencySettings(max_workers=16)
        config = RuntimeConcurrencyConfig.from_settings(settings)

        assert config.max_workers == 16


class TestConcurrencyConfigInTransformExecutor:
    """Test that max_workers is passed to TransformExecutor."""

    def test_executor_accepts_max_workers(self, landscape_db: LandscapeDB) -> None:
        """TransformExecutor constructor accepts max_workers parameter."""
        factory = make_factory(landscape_db)
        span_factory = SpanFactory()

        executor = TransformExecutor(factory.execution, span_factory, lambda node_id: 1, max_workers=8)
        assert executor._max_workers == 8

    def test_executor_without_max_workers(self, landscape_db: LandscapeDB) -> None:
        """TransformExecutor works without max_workers (no cap)."""
        factory = make_factory(landscape_db)
        span_factory = SpanFactory()

        executor = TransformExecutor(factory.execution, span_factory, lambda node_id: 1)
        assert executor._max_workers is None


class TestConcurrencyConfigInRowProcessor:
    """Test that max_workers flows through RowProcessor."""

    def test_processor_accepts_max_workers(self, landscape_db: LandscapeDB) -> None:
        """RowProcessor constructor accepts and forwards max_workers."""
        factory = make_factory(landscape_db)
        span_factory = SpanFactory()
        source_node_id, source_on_success, traversal = _build_traversal_from_graph()

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="test-run",
            source_node_id=source_node_id,
            source_on_success=source_on_success,
            traversal=traversal,
            max_workers=4,
        )
        # Verify max_workers was passed to TransformExecutor
        assert processor._transform_executor._max_workers == 4

    def test_processor_without_max_workers(self, landscape_db: LandscapeDB) -> None:
        """RowProcessor works without max_workers."""
        factory = make_factory(landscape_db)
        span_factory = SpanFactory()
        source_node_id, source_on_success, traversal = _build_traversal_from_graph()

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="test-run",
            source_node_id=source_node_id,
            source_on_success=source_on_success,
            traversal=traversal,
        )
        # No max_workers means no cap
        assert processor._transform_executor._max_workers is None


class TestConcurrencyConfigProtocolCompliance:
    """Test that RuntimeConcurrencyConfig implements RuntimeConcurrencyProtocol."""

    def test_protocol_compliance(self) -> None:
        """RuntimeConcurrencyConfig satisfies RuntimeConcurrencyProtocol."""
        from elspeth.contracts.config import RuntimeConcurrencyConfig, RuntimeConcurrencyProtocol

        config = RuntimeConcurrencyConfig(max_workers=8)

        # Protocol requires max_workers property
        assert isinstance(config, RuntimeConcurrencyProtocol)
        assert config.max_workers == 8

    def test_orchestrator_accepts_protocol(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator accepts any object implementing RuntimeConcurrencyProtocol."""
        from dataclasses import dataclass

        from elspeth.contracts.config import RuntimeConcurrencyProtocol

        @dataclass(frozen=True)
        class CustomConcurrencyConfig:
            """Custom implementation of RuntimeConcurrencyProtocol."""

            max_workers: int

        # Verify it satisfies the protocol
        custom_config = CustomConcurrencyConfig(max_workers=12)
        assert isinstance(custom_config, RuntimeConcurrencyProtocol)

        # Verify Orchestrator accepts it
        orchestrator = Orchestrator(landscape_db, concurrency_config=custom_config)  # type: ignore[arg-type]
        assert orchestrator._concurrency_config is not None
        assert orchestrator._concurrency_config.max_workers == 12


class TestOrchestratorThreadsMaxWorkersThroughRowProcessor:
    """Test that Orchestrator passes max_workers to RowProcessor during run().

    Issue: P2-2026-02-01-max-workers-not-threaded-through-RowProcessor
    The Orchestrator stores _concurrency_config but wasn't passing max_workers
    to the RowProcessor instantiation in run() and resume().
    """

    def test_orchestrator_run_passes_max_workers_to_processor(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Orchestrator.run() passes max_workers from concurrency_config to RowProcessor.

        This test verifies the full wiring path:
        CLI -> RuntimeConcurrencyConfig -> Orchestrator -> RowProcessor -> TransformExecutor
        """
        from unittest.mock import patch

        from elspeth.core.dag import ExecutionGraph
        from tests.fixtures.base_classes import (
            as_sink,
            as_source,
        )
        from tests.fixtures.plugins import CollectSink, ListSource

        concurrency_config = RuntimeConcurrencyConfig(max_workers=5)
        orchestrator = Orchestrator(landscape_db, concurrency_config=concurrency_config)

        # Build graph using production path
        source = ListSource([{"id": 1}], name="simple_source", on_success="output")
        sink = CollectSink(name="simple_sink")

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="output", options={}),
            transforms=[],
            sinks={"output": as_sink(sink)},
            aggregations={},
            gates=[],
        )

        from elspeth.engine.orchestrator import PipelineConfig

        pipeline_config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"output": as_sink(sink)},
            gates=[],
            aggregation_settings={},
            coalesce_settings=[],
        )

        # Patch RowProcessor to capture the max_workers argument
        captured_max_workers = []
        original_init = RowProcessor.__init__

        def capturing_init(self, *args, **kwargs):
            captured_max_workers.append(kwargs.get("max_workers"))
            return original_init(self, *args, **kwargs)

        with patch.object(RowProcessor, "__init__", capturing_init):
            orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # Verify max_workers was passed to RowProcessor
        assert len(captured_max_workers) == 1, "RowProcessor should be instantiated once"
        assert captured_max_workers[0] == 5, f"Expected max_workers=5, got {captured_max_workers[0]}"
