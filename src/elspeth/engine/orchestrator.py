# src/elspeth/engine/orchestrator.py
"""Orchestrator: Full run lifecycle management.

Coordinates:
- Run initialization
- Source loading
- Row processing
- Sink writing
- Run completion
"""

from dataclasses import dataclass, field
from typing import Any

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.processor import RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.enums import NodeType


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    source: Any  # SourceProtocol
    transforms: list[Any]  # List of transform/gate plugins
    sinks: dict[str, Any]  # sink_name -> SinkProtocol
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: str  # completed, failed
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int


class Orchestrator:
    """Orchestrates full pipeline runs.

    Manages the complete lifecycle:
    1. Begin run in Landscape
    2. Register all nodes (and set node_id on each plugin instance)
    3. Load rows from source
    4. Process rows through transforms
    5. Write to sinks
    6. Complete run

    NOTE on node_id: Plugin protocols don't define node_id as an attribute.
    The Orchestrator sets node_id on each plugin instance AFTER registering
    it with Landscape:

        node = recorder.register_node(...)
        transform.node_id = node.node_id  # Set by Orchestrator

    This allows executors to access node_id without requiring plugins
    to know their node_id at construction time.
    """

    def __init__(
        self,
        db: LandscapeDB,
        *,
        canonical_version: str = "sha256-rfc8785-v1",
    ) -> None:
        self._db = db
        self._canonical_version = canonical_version
        self._span_factory = SpanFactory()

    def run(self, config: PipelineConfig) -> RunResult:
        """Execute a pipeline run."""
        recorder = LandscapeRecorder(self._db)

        # Begin run
        run = recorder.begin_run(
            config=config.config,
            canonical_version=self._canonical_version,
        )

        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config)

            # Complete run
            recorder.complete_run(run.run_id, status="completed")
            result.status = "completed"
            return result

        except Exception:
            recorder.complete_run(run.run_id, status="failed")
            raise

    def _execute_run(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
    ) -> RunResult:
        """Execute the run (internal)."""

        # Register source node
        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name=config.source.name,
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            sequence=0,
        )

        # Register transform nodes and track gates for sink edge registration
        edge_map: dict[tuple[str, str], str] = {}
        prev_node_id = source_node.node_id
        gate_node_ids: list[str] = []  # Track gates that may route to sinks

        for i, transform in enumerate(config.transforms):
            is_gate = hasattr(transform, "evaluate")
            node_type = NodeType.GATE if is_gate else NodeType.TRANSFORM
            node = recorder.register_node(
                run_id=run_id,
                plugin_name=transform.name,
                node_type=node_type,
                plugin_version="1.0.0",
                config={},
                sequence=i + 1,
            )
            # Set node_id on plugin (see class docstring)
            transform.node_id = node.node_id

            # Track gates - they may route to any sink
            if is_gate:
                gate_node_ids.append(node.node_id)

            # Register continue edge
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=prev_node_id,
                to_node_id=node.node_id,
                label="continue",
                mode="move",
            )
            edge_map[(prev_node_id, "continue")] = edge.edge_id
            prev_node_id = node.node_id

        # Register sink nodes
        sink_nodes: dict[str, Any] = {}
        for sink_name, sink in config.sinks.items():
            node = recorder.register_node(
                run_id=run_id,
                plugin_name=sink.name,
                node_type=NodeType.SINK,
                plugin_version="1.0.0",
                config={},
            )
            sink.node_id = node.node_id
            sink_nodes[sink_name] = node

            # Register edge from last transform to sink
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=prev_node_id,
                to_node_id=node.node_id,
                label=sink_name,
                mode="move",
            )
            edge_map[(prev_node_id, sink_name)] = edge.edge_id

            # CRITICAL: Register edges from ALL gates to this sink
            for gate_node_id in gate_node_ids:
                if gate_node_id != prev_node_id:  # Don't duplicate
                    gate_edge = recorder.register_edge(
                        run_id=run_id,
                        from_node_id=gate_node_id,
                        to_node_id=node.node_id,
                        label=sink_name,
                        mode="move",
                    )
                    edge_map[(gate_node_id, sink_name)] = gate_edge.edge_id

        # Create context
        # Note: landscape field uses Any type since PluginContext.LandscapeRecorder
        # is a protocol placeholder for Phase 2, while we pass the real recorder
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=recorder,  # type: ignore[arg-type]
        )

        # Create processor
        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_node.node_id,
            edge_map=edge_map,
        )

        # Process rows - Buffer TOKENS, not dicts, to preserve identity
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.tokens import TokenInfo

        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        pending_tokens: dict[str, list[TokenInfo]] = {name: [] for name in config.sinks}

        try:
            with self._span_factory.source_span(config.source.name):
                for row_index, row_data in enumerate(config.source.load(ctx)):
                    rows_processed += 1

                    result = processor.process_row(
                        row_index=row_index,
                        row_data=row_data,
                        transforms=config.transforms,
                        ctx=ctx,
                    )

                    if result.outcome == "completed":
                        rows_succeeded += 1
                        pending_tokens["default"].append(result.token)
                    elif result.outcome == "routed":
                        rows_routed += 1
                        if result.sink_name and result.sink_name in config.sinks:
                            pending_tokens[result.sink_name].append(result.token)
                        else:
                            rows_failed += 1
                    elif result.outcome == "failed":
                        rows_failed += 1

            # Write to sinks using SinkExecutor
            sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
            step = len(config.transforms) + 1

            for sink_name, tokens in pending_tokens.items():
                if tokens and sink_name in config.sinks:
                    sink = config.sinks[sink_name]
                    sink_executor.write(
                        sink=sink,
                        tokens=tokens,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

        finally:
            # Close source and all sinks
            # SinkProtocol requires close() - if missing, that's a bug
            config.source.close()
            for sink in config.sinks.values():
                sink.close()

        return RunResult(
            run_id=run_id,
            status="running",  # Will be updated
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed=rows_routed,
        )
