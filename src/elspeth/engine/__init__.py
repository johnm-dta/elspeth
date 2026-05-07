"""SDA Engine: Orchestration with complete audit trails.

This module provides the execution engine for ELSPETH pipelines:
- Orchestrator: Full run lifecycle management
- RowProcessor: Row-by-row processing through transforms
- TokenManager: Token identity through forks/joins
- SpanFactory: OpenTelemetry integration
- RetryManager: Retry logic with tenacity

Example:
    from elspeth.core.dag.graph import ExecutionGraph
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import PayloadStore
    from elspeth.engine import Orchestrator, PipelineConfig

    db = LandscapeDB.from_url("sqlite:///audit.db")
    payload_store = PayloadStore("./payloads")

    config = PipelineConfig(
        source=csv_source,
        transforms=[transform1, gate1],
        sinks={"default": output_sink},
    )

    graph = ExecutionGraph.from_plugin_instances(
        source=csv_source,
        transforms=[transform1, gate1],
        sinks={"default": output_sink},
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config, graph=graph, payload_store=payload_store)
"""

from elspeth.contracts import RowResult, TokenInfo
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.errors import MaxRetriesExceeded
from elspeth.core.expression_parser import (
    ExpressionParser,
    ExpressionSecurityError,
    ExpressionSyntaxError,
)
from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    MissingEdgeError,
    SinkExecutor,
    TransformExecutor,
)
from elspeth.engine.orchestrator import (
    AggregationFlushResult,
    ExecutionCounters,
    Orchestrator,
    PipelineConfig,
    RouteValidationError,
    RowPlugin,
    RunResult,
)
from elspeth.engine.processor import RowProcessor
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager

__all__ = [
    "AggregationExecutor",
    "AggregationFlushResult",
    "CoalesceExecutor",
    "CoalesceOutcome",
    "ExecutionCounters",
    "ExpressionParser",
    "ExpressionSecurityError",
    "ExpressionSyntaxError",
    "GateExecutor",
    "MaxRetriesExceeded",
    "MissingEdgeError",
    "Orchestrator",
    "PipelineConfig",
    "RetryManager",
    "RouteValidationError",
    "RowPlugin",
    "RowProcessor",
    "RowResult",
    "RunResult",
    "RuntimeRetryConfig",
    "SinkExecutor",
    "SpanFactory",
    "TokenInfo",
    "TokenManager",
    "TransformExecutor",
]
