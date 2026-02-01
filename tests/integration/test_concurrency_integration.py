"""Integration tests for concurrency config wiring.

These tests verify that RuntimeConcurrencyConfig is properly wired through
the CLI -> Orchestrator -> RowProcessor -> TransformExecutor pipeline.

STATUS: IMPLEMENTED
- Orchestrator: accepts concurrency_config parameter
- TransformExecutor: accepts max_workers parameter
- RowProcessor: accepts max_workers and forwards to TransformExecutor
"""

from elspeth.contracts.config import RuntimeConcurrencyConfig
from elspeth.core.config import ConcurrencySettings
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.processor import RowProcessor
from elspeth.engine.spans import SpanFactory


class TestConcurrencyConfigInOrchestrator:
    """Test that RuntimeConcurrencyConfig is properly passed to Orchestrator."""

    def test_orchestrator_accepts_concurrency_config(self) -> None:
        """Orchestrator constructor accepts concurrency_config parameter."""
        db = LandscapeDB.in_memory()
        config = RuntimeConcurrencyConfig(max_workers=8)

        try:
            # Should not raise
            orchestrator = Orchestrator(db, concurrency_config=config)
            assert orchestrator._concurrency_config is config
            assert orchestrator._concurrency_config.max_workers == 8
        finally:
            db.close()

    def test_orchestrator_accepts_none_config(self) -> None:
        """Orchestrator works without concurrency config."""
        db = LandscapeDB.in_memory()

        try:
            orchestrator = Orchestrator(db, concurrency_config=None)
            assert orchestrator._concurrency_config is None
        finally:
            db.close()

    def test_config_from_settings(self) -> None:
        """RuntimeConcurrencyConfig.from_settings() creates config correctly."""
        settings = ConcurrencySettings(max_workers=16)
        config = RuntimeConcurrencyConfig.from_settings(settings)

        assert config.max_workers == 16


class TestConcurrencyConfigInTransformExecutor:
    """Test that max_workers is passed to TransformExecutor."""

    def test_executor_accepts_max_workers(self) -> None:
        """TransformExecutor constructor accepts max_workers parameter."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()

        try:
            executor = TransformExecutor(recorder, span_factory, max_workers=8)
            assert executor._max_workers == 8
        finally:
            db.close()

    def test_executor_without_max_workers(self) -> None:
        """TransformExecutor works without max_workers (no cap)."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()

        try:
            executor = TransformExecutor(recorder, span_factory)
            assert executor._max_workers is None
        finally:
            db.close()


class TestConcurrencyConfigInRowProcessor:
    """Test that max_workers flows through RowProcessor."""

    def test_processor_accepts_max_workers(self) -> None:
        """RowProcessor constructor accepts and forwards max_workers."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()

        try:
            processor = RowProcessor(
                recorder=recorder,
                span_factory=span_factory,
                run_id="test-run",
                source_node_id="source-1",
                max_workers=4,
            )
            # Verify max_workers was passed to TransformExecutor
            assert processor._transform_executor._max_workers == 4
        finally:
            db.close()

    def test_processor_without_max_workers(self) -> None:
        """RowProcessor works without max_workers."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()

        try:
            processor = RowProcessor(
                recorder=recorder,
                span_factory=span_factory,
                run_id="test-run",
                source_node_id="source-1",
            )
            # No max_workers means no cap
            assert processor._transform_executor._max_workers is None
        finally:
            db.close()


class TestConcurrencyConfigProtocolCompliance:
    """Test that RuntimeConcurrencyConfig implements RuntimeConcurrencyProtocol."""

    def test_protocol_compliance(self) -> None:
        """RuntimeConcurrencyConfig satisfies RuntimeConcurrencyProtocol."""
        from elspeth.contracts.config import RuntimeConcurrencyConfig, RuntimeConcurrencyProtocol

        config = RuntimeConcurrencyConfig(max_workers=8)

        # Protocol requires max_workers property
        assert isinstance(config, RuntimeConcurrencyProtocol)
        assert config.max_workers == 8

    def test_orchestrator_accepts_protocol(self) -> None:
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
        db = LandscapeDB.in_memory()
        try:
            orchestrator = Orchestrator(db, concurrency_config=custom_config)
            assert orchestrator._concurrency_config.max_workers == 12
        finally:
            db.close()


class TestOrchestratorThreadsMaxWorkersThroughRowProcessor:
    """Test that Orchestrator passes max_workers to RowProcessor during run().

    Issue: P2-2026-02-01-max-workers-not-threaded-through-RowProcessor
    The Orchestrator stores _concurrency_config but wasn't passing max_workers
    to the RowProcessor instantiation in run() and resume().
    """

    def test_orchestrator_run_passes_max_workers_to_processor(self, payload_store) -> None:
        """Orchestrator.run() passes max_workers from concurrency_config to RowProcessor.

        This test verifies the full wiring path:
        CLI -> RuntimeConcurrencyConfig -> Orchestrator -> RowProcessor -> TransformExecutor
        """
        from unittest.mock import MagicMock, patch

        from elspeth.contracts import ArtifactDescriptor, Determinism, NodeType, RoutingMode, SinkName, SourceRow
        from elspeth.core.dag import ExecutionGraph

        db = LandscapeDB.in_memory()
        concurrency_config = RuntimeConcurrencyConfig(max_workers=5)
        orchestrator = Orchestrator(db, concurrency_config=concurrency_config)

        # Create minimal graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink", config=schema_config)
        graph.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)
        graph._sink_id_map = {SinkName("output"): "sink"}
        graph._default_sink = "output"
        graph._transform_id_map = {}

        # Create mock source
        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source._on_validation_failure = "discard"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {"type": "object"}
        mock_source.output_schema = schema_mock
        mock_source.load.return_value = iter([SourceRow.valid({"id": 1})])
        mock_source.get_field_resolution.return_value = (None, None)

        # Create mock sink
        mock_sink = MagicMock()
        mock_sink.name = "test_sink"
        mock_sink.determinism = Determinism.IO_WRITE
        mock_sink.plugin_version = "1.0.0"
        mock_sink.input_schema = schema_mock
        mock_sink.write.return_value = ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc123")

        from elspeth.engine.orchestrator import PipelineConfig

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
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

        try:
            with patch.object(RowProcessor, "__init__", capturing_init):
                orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

            # Verify max_workers was passed to RowProcessor
            assert len(captured_max_workers) == 1, "RowProcessor should be instantiated once"
            assert captured_max_workers[0] == 5, f"Expected max_workers=5, got {captured_max_workers[0]}"
        finally:
            db.close()
