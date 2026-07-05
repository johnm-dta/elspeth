"""SDA Engine: Orchestration with complete audit trails.

This module provides the execution engine for ELSPETH pipelines:
- Orchestrator: Full run lifecycle management
- PipelineConfig: Runtime pipeline wiring passed to the orchestrator
- RunResult: Terminal run status and accounting returned by the orchestrator

Example:
    from elspeth.core.dag.graph import ExecutionGraph
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import PayloadStore
    from elspeth.engine import Orchestrator, PipelineConfig

    db = LandscapeDB.from_url("sqlite:///audit.db")
    payload_store = PayloadStore("./payloads")

    config = PipelineConfig(
        sources={"primary": csv_source},
        transforms=[transform1, gate1],
        sinks={"default": output_sink},
    )

    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": csv_source},
        source_settings_map={"primary": csv_source_settings},
        transforms=[transform1, gate1],
        sinks={"default": output_sink},
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config, graph=graph, payload_store=payload_store)
"""

from elspeth.contracts.errors import MaxRetriesExceeded
from elspeth.engine.orchestrator import (
    AggregationFlushResult,
    ExecutionCounters,
    Orchestrator,
    PipelineConfig,
    RouteValidationError,
    RowPlugin,
    RunResult,
)

__all__ = [
    "AggregationFlushResult",
    "ExecutionCounters",
    "MaxRetriesExceeded",
    "Orchestrator",
    "PipelineConfig",
    "RouteValidationError",
    "RowPlugin",
    "RunResult",
]
