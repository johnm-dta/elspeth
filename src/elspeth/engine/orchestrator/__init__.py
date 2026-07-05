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

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.engine.orchestrator.core import Orchestrator, prepare_for_run
    from elspeth.engine.orchestrator.types import (
        AggregationFlushResult,
        ExecutionCounters,
        PipelineConfig,
        RouteValidationError,
        RowPlugin,
        RunResult,
    )

_TYPE_EXPORTS = {
    "AggregationFlushResult",
    "ExecutionCounters",
    "PipelineConfig",
    "RouteValidationError",
    "RowPlugin",
    "RunResult",
}
_RUNTIME_EXPORTS = {"Orchestrator", "prepare_for_run"}

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


def __getattr__(name: str) -> Any:
    if name in _TYPE_EXPORTS:
        value = getattr(import_module("elspeth.engine.orchestrator.types"), name)
    elif name in _RUNTIME_EXPORTS:
        value = getattr(import_module("elspeth.engine.orchestrator.core"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
