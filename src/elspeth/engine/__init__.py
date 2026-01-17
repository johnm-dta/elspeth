# src/elspeth/engine/__init__.py
"""SDA Engine: Orchestration with complete audit trails.

This module provides the execution engine for ELSPETH pipelines:
- Orchestrator: Full run lifecycle management
- RowProcessor: Row-by-row processing through transforms
- TokenManager: Token identity through forks/joins
- SpanFactory: OpenTelemetry integration
- RetryManager: Retry logic with tenacity

Example:
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig

    db = LandscapeDB.from_url("sqlite:///audit.db")

    config = PipelineConfig(
        source=csv_source,
        transforms=[transform1, gate1],
        sinks={"default": output_sink},
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config)
"""

from elspeth.contracts import RowResult, TokenInfo
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    MissingEdgeError,
    SinkExecutor,
    TransformExecutor,
)
from elspeth.engine.expression_parser import (
    ExpressionParser,
    ExpressionSecurityError,
    ExpressionSyntaxError,
)
from elspeth.engine.orchestrator import (
    Orchestrator,
    PipelineConfig,
    RouteValidationError,
    RunResult,
)
from elspeth.engine.processor import RowProcessor
from elspeth.engine.retry import MaxRetriesExceeded, RetryConfig, RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenManager

__all__ = [
    "AggregationExecutor",
    "ExpressionParser",
    "ExpressionSecurityError",
    "ExpressionSyntaxError",
    "GateExecutor",
    "MaxRetriesExceeded",
    "MissingEdgeError",
    "Orchestrator",
    "PipelineConfig",
    "RetryConfig",
    "RetryManager",
    "RouteValidationError",
    "RowProcessor",
    "RowResult",
    "RunResult",
    "SinkExecutor",
    "SpanFactory",
    "TokenInfo",
    "TokenManager",
    "TransformExecutor",
]
