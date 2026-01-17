# tests/engine/test_orchestrator_schema_validation.py
"""Tests for schema validation in orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import Determinism, PluginSchema, RoutingMode
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext


class InputSchema(PluginSchema):
    """Test input schema."""

    name: str
    value: int


class OutputSchema(PluginSchema):
    """Test output schema."""

    name: str
    result: float


class IncompatibleSchema(PluginSchema):
    """Schema missing required field."""

    missing_field: str


# Test fixture base classes (mirroring test_orchestrator.py patterns)


class _TestSourceBase:
    """Base class providing SourceProtocol required attributes."""

    name = "test_source"
    node_id: str | None = None
    output_schema: type[PluginSchema] | None = None

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        yield {"name": "test", "value": 42}

    def on_start(self, ctx: PluginContext) -> None:
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        pass

    def close(self) -> None:
        pass


class _TestSinkBase:
    """Base class providing SinkProtocol required attributes."""

    name = "test_sink"
    idempotent = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0"
    input_schema: type[PluginSchema] | None = None

    def write_batch(
        self, batch: list[dict[str, Any]], ctx: PluginContext, step: int
    ) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def on_start(self, ctx: PluginContext) -> None:
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        pass


def _build_minimal_graph(source_name: str, sink_name: str) -> ExecutionGraph:
    """Build a minimal graph for schema validation tests."""
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name=source_name)
    graph.add_node("sink_output", node_type="sink", plugin_name=sink_name)
    graph.add_edge("source", "sink_output", label="continue", mode=RoutingMode.MOVE)
    graph._sink_id_map = {"output": "sink_output"}  # noqa: SLF001
    graph._output_sink = "output"  # noqa: SLF001
    return graph


class TestOrchestratorSchemaValidation:
    """Test that orchestrator calls schema validator."""

    def test_schema_validation_called_on_run(self) -> None:
        """Orchestrator should call validate_pipeline_schemas."""
        with patch(
            "elspeth.engine.orchestrator.validate_pipeline_schemas"
        ) as mock_validate:
            mock_validate.return_value = []  # No errors

            # Create minimal mocks
            db = MagicMock(spec=LandscapeDB)
            orchestrator = Orchestrator(db)

            # Create source with schema
            class TestSource(_TestSourceBase):
                output_schema = OutputSchema

            source = TestSource()

            # Create sink
            class TestSink(_TestSinkBase):
                pass

            sink = TestSink()

            config = PipelineConfig(
                source=source,  # type: ignore[arg-type]
                transforms=[],
                sinks={"output": sink},  # type: ignore[dict-item]
            )

            graph = _build_minimal_graph("test_source", "test_sink")

            # This will fail for other reasons (mock DB), but we want to verify
            # validate_pipeline_schemas was called
            try:
                orchestrator.run(config, graph=graph)
            except Exception:
                pass  # Expected to fail - we're testing the call

            # Verify schema validation was called
            mock_validate.assert_called_once()

    def test_schema_validation_errors_raise(self) -> None:
        """Schema validation errors should raise before processing."""
        with patch(
            "elspeth.engine.orchestrator.validate_pipeline_schemas"
        ) as mock_validate:
            mock_validate.return_value = [
                "Source output missing fields required by transform[0]: {'missing_field'}"
            ]

            db = MagicMock(spec=LandscapeDB)
            orchestrator = Orchestrator(db)

            class TestSource(_TestSourceBase):
                output_schema = OutputSchema

            source = TestSource()

            class TestSink(_TestSinkBase):
                pass

            sink = TestSink()

            config = PipelineConfig(
                source=source,  # type: ignore[arg-type]
                transforms=[],
                sinks={"output": sink},  # type: ignore[dict-item]
            )

            graph = _build_minimal_graph("test_source", "test_sink")

            with pytest.raises(ValueError, match="schema incompatibility"):
                orchestrator.run(config, graph=graph)

    def test_schema_validation_skipped_when_no_schemas(self) -> None:
        """Schema validation should pass when no schemas are defined."""
        with patch(
            "elspeth.engine.orchestrator.validate_pipeline_schemas"
        ) as mock_validate:
            mock_validate.return_value = []  # No errors

            db = MagicMock(spec=LandscapeDB)
            orchestrator = Orchestrator(db)

            # Source without schema
            class TestSource(_TestSourceBase):
                output_schema = None

            source = TestSource()

            class TestSink(_TestSinkBase):
                input_schema = None

            sink = TestSink()

            config = PipelineConfig(
                source=source,  # type: ignore[arg-type]
                transforms=[],
                sinks={"output": sink},  # type: ignore[dict-item]
            )

            graph = _build_minimal_graph("test_source", "test_sink")

            # Run will fail for other reasons, but validation should be called
            try:
                orchestrator.run(config, graph=graph)
            except Exception:
                pass

            # Verify schema validation was called with None values
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args
            assert call_args[1]["source_output"] is None
