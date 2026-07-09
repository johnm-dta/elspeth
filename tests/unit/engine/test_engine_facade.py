from __future__ import annotations

import pytest

import elspeth.engine as engine


def test_engine_facade_exports_only_orchestration_api() -> None:
    assert engine.__all__ == [
        "AggregationFlushResult",
        "ExecutionCounters",
        "MaxRetriesExceeded",
        "Orchestrator",
        "PipelineConfig",
        "RouteValidationError",
        "RowPlugin",
        "RunResult",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "AggregationExecutor",
        "CoalesceExecutor",
        "CoalesceOutcome",
        "GateExecutor",
        "MissingEdgeError",
        "RetryManager",
        "RowProcessor",
        "SinkExecutor",
        "SpanFactory",
        "TokenManager",
        "TransformExecutor",
    ],
)
def test_engine_facade_does_not_export_low_level_execution_internals(name: str) -> None:
    assert not hasattr(engine, name)
