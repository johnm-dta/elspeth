"""Orchestrator package: Full run lifecycle management.

This package has been refactored from a single 3000+ line module into
focused modules while preserving the public API.

Public API (unchanged):
- Orchestrator: Main class for running pipelines
- PipelineConfig: Configuration dataclass
- RunResult: Result dataclass
- RouteValidationError: Validation exception
- AggregationFlushResult: Result of flushing aggregation buffers (replaces 9-tuple)
- ExecutionCounters: Mutable counters for pipeline execution
- RowPlugin: Type alias for transform plugins (TransformProtocol)

Module structure:
- core.py: Orchestrator class (main entry point)
- types.py: PipelineConfig, RunResult, RouteValidationError, AggregationFlushResult, ExecutionCounters
- validation.py: Route and sink validation functions
- export.py: Landscape export functionality
- aggregation.py: Aggregation timeout/flush handling
- outcomes.py: Row outcome accumulation and coalesce handling
"""

from elspeth.engine.orchestrator.core import Orchestrator, prepare_for_run
from elspeth.engine.orchestrator.types import (
    AggregationFlushResult,
    ExecutionCounters,
    PipelineConfig,
    RouteValidationError,
    RowPlugin,
    RunResult,
)

__all__ = [
    "AggregationFlushResult",
    "ExecutionCounters",
    "Orchestrator",
    "PipelineConfig",
    "RouteValidationError",
    "RowPlugin",
    "RunResult",
    "prepare_for_run",
]
