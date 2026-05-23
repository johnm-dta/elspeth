# tests/integration/config/test_schema_validation_integration.py
"""Integration test verifying schema validation works end-to-end.

This test confirms that the schema validation bypass bug is fixed:
- Schemas are extracted from plugin instances via PluginManager
- Graph validation runs successfully with real plugins
- No crashes from missing manager parameter or broken schema lookups

Relates-To: P1-2026-01-21-schema-validator-ignores-dag-routing
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from elspeth.contracts import Determinism

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.manager import PluginManager


def test_schema_validation_end_to_end(tmp_path: Path, plugin_manager: PluginManager) -> None:
    """Verify schemas are extracted from plugin instances and validation works.

    This test confirms the schema validation bypass bug is fixed:
    - Schemas are populated from plugin instances (not bypassed with None)
    - Graph validation runs successfully
    - Compatible schemas pass validation
    - No TypeError about missing manager parameter (Task 2)
    - No AttributeError from getattr on config models (Tasks 3-5)

    Relates-To: P1-2026-01-21-schema-validator-ignores-dag-routing
    """
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.config import (
        ElspethSettings,
        SinkSettings,
        SourceSettings,
        TransformSettings,
    )
    from elspeth.core.dag import ExecutionGraph

    # Create a CSV file
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("name,age\nAlice,30\nBob,25\n")

    # Build config with compatible plugins
    # All plugins use dynamic schemas (mode: observed)
    config = ElspethSettings(
        sources={
            "primary": SourceSettings(
                plugin="csv",
                on_success="source_out",
                options={
                    "path": str(csv_path),
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            )
        },
        transforms=[
            TransformSettings(
                name="passthrough_0",
                plugin="passthrough",
                input="source_out",
                on_success="output",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
            ),
        ],
        sinks={
            "output": SinkSettings(
                plugin="json",
                on_write_failure="discard",
                options={
                    "path": str(tmp_path / "output.json"),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                },
            ),
        },
    )

    # Build graph with real PluginManager
    # This is where the fix is exercised:
    # 1. Task 2: manager parameter is required (not None)
    # 2. Tasks 3-5: schemas extracted from plugin instances via manager
    plugins = instantiate_plugins_from_config(config)

    graph = ExecutionGraph.from_plugin_instances(
        sources=plugins.sources,
        source_settings_map=plugins.source_settings_map,
        transforms=plugins.transforms,
        sinks=plugins.sinks,
        aggregations=plugins.aggregations,
        gates=list(config.gates),
    )

    # Validation should pass (schemas are compatible)
    # This verifies the fix works without crashes
    graph.validate()  # Should not raise

    # Verify schemas were populated from plugin instances
    nodes = graph.get_nodes()
    source_nodes = [n for n in nodes if n.node_type == "source"]
    transform_nodes = [n for n in nodes if n.node_type == "transform"]
    sink_nodes = [n for n in nodes if n.node_type == "sink"]

    assert len(source_nodes) == 1, "Should have exactly one source node"
    assert len(transform_nodes) == 1, "Should have exactly one transform node"
    assert len(sink_nodes) == 1, "Should have exactly one sink node"

    # The schemas should be accessible (not raising AttributeError)
    source_node = source_nodes[0]
    transform_node = transform_nodes[0]
    sink_node = sink_nodes[0]

    assert source_node.output_schema is not None, "Source node should have output_schema"
    assert transform_node.input_schema is not None, "Transform node should have input_schema"
    assert transform_node.output_schema is not None, "Transform node should have output_schema"
    assert sink_node.input_schema is not None, "Sink node should have input_schema"


def test_static_schema_validation(plugin_manager: PluginManager) -> None:
    """Verify static schemas are populated from plugin classes.

    This test uses plugins that declare schemas as class attributes
    (not dynamic schemas set in __init__).

    Complements test_schema_validation_end_to_end which uses dynamic schemas.
    Together they verify the schema validation mechanism works for both
    static (class-level) and dynamic (instance-level) schema definitions.

    Relates-To: P1-2026-01-21-schema-validator-ignores-dag-routing
    """
    from collections.abc import Iterator
    from typing import Any

    from elspeth.contracts import ArtifactDescriptor, PluginSchema, SourceRow
    from elspeth.contracts.diversion import SinkWriteResult
    from elspeth.contracts.schema_contract import PipelineRow
    from elspeth.core.config import SourceSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.infrastructure.results import TransformResult
    from tests.fixtures.base_classes import (
        _TestSinkBase,
        _TestSourceBase,
        _TestTransformBase,
        as_sink,
        as_source,
        as_transform,
    )
    from tests.fixtures.factories import wire_transforms

    # Define test plugins with STATIC class-level schemas
    # These are set at class definition time, not in __init__

    class StaticSchema(PluginSchema):
        """Static schema with explicit fields."""

        id: int
        value: str

    class StaticSchemaSource(_TestSourceBase):
        """Source with static class-level output_schema."""

        name = "static_source"
        output_schema = StaticSchema  # Class-level static schema
        on_success = "source_out"  # Route to first transform connection

        def __init__(self) -> None:
            super().__init__()
            self._data = [{"id": 1, "value": "test"}]

        def load(self, ctx: Any) -> Iterator[SourceRow]:
            yield from self.wrap_rows(self._data)

    class StaticSchemaTransform(_TestTransformBase):
        """Transform with static class-level input/output schemas."""

        name = "static_transform"
        determinism = Determinism.DETERMINISTIC
        input_schema = StaticSchema  # Class-level static schema
        output_schema = StaticSchema  # Class-level static schema
        on_success = "output"  # Terminal transform routes to output sink

        def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
            return TransformResult.success(row, success_reason={"action": "passthrough"})

    class StaticSchemaSink(_TestSinkBase):
        """Sink with static class-level input_schema."""

        name = "static_sink"
        input_schema = StaticSchema  # Class-level static schema

        def __init__(self) -> None:
            super().__init__()
            self.written: list[dict[str, Any]] = []

        def write(self, rows: list[dict[str, Any]], ctx: Any) -> SinkWriteResult:
            self.written.extend(rows)
            return SinkWriteResult(artifact=ArtifactDescriptor.for_file(path="memory://test", size_bytes=0, content_hash="test"))

    # Build graph with static schema plugins
    source = StaticSchemaSource()
    transform = StaticSchemaTransform()
    sink = StaticSchemaSink()

    source_settings = SourceSettings(plugin=source.name, on_success="source_out", options={})
    wired = wire_transforms([as_transform(transform)], source_connection="source_out", final_sink="output")

    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": as_source(source)},
        source_settings_map={"primary": source_settings},
        transforms=wired,
        sinks={"output": as_sink(sink)},
        aggregations={},
        gates=[],
    )

    # Validation should pass (schemas are compatible - all use StaticSchema)
    graph.validate()

    # Verify static schemas were populated from plugin class attributes
    nodes = graph.get_nodes()
    source_nodes = [n for n in nodes if n.node_type == "source"]
    transform_nodes = [n for n in nodes if n.node_type == "transform"]
    sink_nodes = [n for n in nodes if n.node_type == "sink"]

    assert len(source_nodes) == 1, "Should have exactly one source node"
    assert len(transform_nodes) == 1, "Should have exactly one transform node"
    assert len(sink_nodes) == 1, "Should have exactly one sink node"

    source_node = source_nodes[0]
    transform_node = transform_nodes[0]
    sink_node = sink_nodes[0]

    # CRITICAL: Static schemas should be populated (not None)
    # This is the key difference from dynamic schemas which are None at graph time
    assert source_node.output_schema is StaticSchema, f"Source output_schema should be StaticSchema, got {source_node.output_schema}"
    assert transform_node.input_schema is StaticSchema, f"Transform input_schema should be StaticSchema, got {transform_node.input_schema}"
    assert transform_node.output_schema is StaticSchema, (
        f"Transform output_schema should be StaticSchema, got {transform_node.output_schema}"
    )
    assert sink_node.input_schema is StaticSchema, f"Sink input_schema should be StaticSchema, got {sink_node.input_schema}"
