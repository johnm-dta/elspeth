"""Typed node state context for the Landscape audit trail.

Replaces ``dict[str, Any]`` at the Tier 1 boundary where context
metadata is serialized into ``context_after_json``.  Follows the
same pattern as ``CoalesceMetadata`` (commit 4f7e43be) and
``TokenUsage`` (commit dffe74a6).

Trust-tier notes
----------------
* ``NodeStateContext`` — Protocol for structural typing (mypy only).
* ``PoolExecutionContext`` — typed pool stats from LLM multi-query.
* ``from_executor_stats()`` — Tier 1 factory: crash on bad data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from elspeth.contracts.freeze import require_int


class NodeStateContext(Protocol):
    """Structural protocol for node state context metadata.

    Any object with a ``to_dict()`` method can serve as context
    metadata for the audit trail.  NOT ``@runtime_checkable`` —
    conformance is verified by mypy at type-check time only.
    """

    def to_dict(self) -> dict[str, Any]: ...


class QueryOrderEntryInput(Protocol):
    """Structural view of reorder-buffer entries needed for audit ordering."""

    submit_index: int
    complete_index: int
    buffer_wait_ms: float


def _require_non_negative_finite_number(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be int or float, got {type(value).__name__}: {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be non-negative and finite, got {value!r}")


def _require_non_empty_str(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}: {value!r}")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class PoolConfigSnapshot:
    """Pool configuration at completion time."""

    pool_size: int
    max_capacity_retry_seconds: float
    dispatch_delay_at_completion_ms: float

    def __post_init__(self) -> None:
        require_int(self.pool_size, "pool_size", min_value=0)
        _require_non_negative_finite_number(self.max_capacity_retry_seconds, "max_capacity_retry_seconds")
        _require_non_negative_finite_number(self.dispatch_delay_at_completion_ms, "dispatch_delay_at_completion_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "max_capacity_retry_seconds": self.max_capacity_retry_seconds,
            "dispatch_delay_at_completion_ms": self.dispatch_delay_at_completion_ms,
        }


@dataclass(frozen=True, slots=True)
class PoolStatsSnapshot:
    """Pool runtime statistics at completion time."""

    capacity_retries: int
    successes: int
    peak_delay_ms: float
    current_delay_ms: float
    total_throttle_time_ms: float
    max_concurrent_reached: int

    def __post_init__(self) -> None:
        require_int(self.capacity_retries, "capacity_retries", min_value=0)
        require_int(self.successes, "successes", min_value=0)
        _require_non_negative_finite_number(self.peak_delay_ms, "peak_delay_ms")
        _require_non_negative_finite_number(self.current_delay_ms, "current_delay_ms")
        _require_non_negative_finite_number(self.total_throttle_time_ms, "total_throttle_time_ms")
        require_int(self.max_concurrent_reached, "max_concurrent_reached", min_value=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_retries": self.capacity_retries,
            "successes": self.successes,
            "peak_delay_ms": self.peak_delay_ms,
            "current_delay_ms": self.current_delay_ms,
            "total_throttle_time_ms": self.total_throttle_time_ms,
            "max_concurrent_reached": self.max_concurrent_reached,
        }


@dataclass(frozen=True, slots=True)
class QueryOrderEntry:
    """Ordering metadata for a single query in a pooled batch."""

    submit_index: int
    complete_index: int
    buffer_wait_ms: float

    def __post_init__(self) -> None:
        require_int(self.submit_index, "submit_index", min_value=0)
        require_int(self.complete_index, "complete_index", min_value=0)
        _require_non_negative_finite_number(self.buffer_wait_ms, "buffer_wait_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "submit_index": self.submit_index,
            "complete_index": self.complete_index,
            "buffer_wait_ms": self.buffer_wait_ms,
        }


@dataclass(frozen=True, slots=True)
class PoolExecutionContext:
    """Typed pool execution metadata for the LLM multi-query audit trail.

    Replaces the untyped ``dict[str, Any]`` constructed in
    ``base_multi_query.py``.
    """

    pool_config: PoolConfigSnapshot
    pool_stats: PoolStatsSnapshot
    query_ordering: tuple[QueryOrderEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pool_config, PoolConfigSnapshot):
            raise TypeError(f"pool_config must be PoolConfigSnapshot, got {type(self.pool_config).__name__}: {self.pool_config!r}")
        if not isinstance(self.pool_stats, PoolStatsSnapshot):
            raise TypeError(f"pool_stats must be PoolStatsSnapshot, got {type(self.pool_stats).__name__}: {self.pool_stats!r}")
        if not isinstance(self.query_ordering, tuple):
            raise TypeError(f"query_ordering must be tuple, got {type(self.query_ordering).__name__}: {self.query_ordering!r}")
        for idx, entry in enumerate(self.query_ordering):
            if not isinstance(entry, QueryOrderEntry):
                raise TypeError(f"query_ordering[{idx}] must be QueryOrderEntry, got {type(entry).__name__}: {entry!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_config": self.pool_config.to_dict(),
            "pool_stats": self.pool_stats.to_dict(),
            "query_ordering": [entry.to_dict() for entry in self.query_ordering],
        }

    @classmethod
    def from_executor_stats(
        cls,
        stats: dict[str, Any],
        entries: list[QueryOrderEntryInput],
    ) -> PoolExecutionContext:
        """Build from PooledExecutor.get_stats() and reorder buffer entries.

        This is a Tier 1 factory — bad data means a bug in our code,
        so we access keys directly (crash on missing/wrong type).
        """
        pool_config_raw = stats["pool_config"]
        pool_stats_raw = stats["pool_stats"]

        config = PoolConfigSnapshot(
            pool_size=pool_config_raw["pool_size"],
            max_capacity_retry_seconds=pool_config_raw["max_capacity_retry_seconds"],
            dispatch_delay_at_completion_ms=pool_config_raw["dispatch_delay_at_completion_ms"],
        )
        pool_stats = PoolStatsSnapshot(
            capacity_retries=pool_stats_raw["capacity_retries"],
            successes=pool_stats_raw["successes"],
            peak_delay_ms=pool_stats_raw["peak_delay_ms"],
            current_delay_ms=pool_stats_raw["current_delay_ms"],
            total_throttle_time_ms=pool_stats_raw["total_throttle_time_ms"],
            max_concurrent_reached=pool_stats_raw["max_concurrent_reached"],
        )
        ordering = tuple(
            QueryOrderEntry(
                submit_index=entry.submit_index,
                complete_index=entry.complete_index,
                buffer_wait_ms=entry.buffer_wait_ms,
            )
            for entry in entries
        )
        return cls(
            pool_config=config,
            pool_stats=pool_stats,
            query_ordering=ordering,
        )


@dataclass(frozen=True, slots=True)
class GateEvaluationContext:
    """Typed gate evaluation metadata for the audit trail.

    Replaces the untyped ``dict[str, Any]`` constructed in gate
    executor code.  Follows the same pattern as ``PoolExecutionContext``
    (this module) and ``CoalesceMetadata`` (commit 4f7e43be).

    Fields
    ------
    condition : str
        The gate's expression string (e.g. ``"amount > 1000"``).
    result : str
        The raw stringified evaluation result (e.g. ``"True"``).
    route_label : str
        The normalized routing key derived from the result
        (e.g. ``"true"``).  For boolean expressions the result and
        route_label differ only in casing; for multi-valued expressions
        the route_label is the resolved destination key.
    """

    condition: str
    result: str
    route_label: str

    def __post_init__(self) -> None:
        _require_non_empty_str(self.condition, "condition")
        _require_non_empty_str(self.result, "result")
        _require_non_empty_str(self.route_label, "route_label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "result": self.result,
            "route_label": self.route_label,
        }


@dataclass(frozen=True, slots=True)
class AggregationFlushContext:
    """Typed aggregation flush metadata for the audit trail.

    Replaces the untyped ``dict[str, Any]`` constructed in aggregation
    executor code.  Follows the same pattern as ``PoolExecutionContext``
    (this module) and ``CoalesceMetadata`` (commit 4f7e43be).

    The flush-level counters (``flush_index``, ``rows_seen_total``,
    ``row_start``, ``row_end``, ``is_end_of_source``) mirror the same
    fields on :class:`AggregationBatchContext` so the audit trail records
    the exact pagination metadata that the batch-aware transform saw.
    """

    trigger_type: str
    buffer_size: int
    batch_id: str
    flush_index: int
    rows_seen_total: int
    row_start: int
    row_end: int
    is_end_of_source: bool

    def __post_init__(self) -> None:
        _require_non_empty_str(self.trigger_type, "trigger_type")
        _require_non_empty_str(self.batch_id, "batch_id")
        require_int(self.buffer_size, "buffer_size", min_value=0)
        require_int(self.flush_index, "flush_index", min_value=1)
        require_int(self.rows_seen_total, "rows_seen_total", min_value=1)
        require_int(self.row_start, "row_start", min_value=1)
        require_int(self.row_end, "row_end", min_value=1)
        if self.row_end < self.row_start:
            raise ValueError(f"AggregationFlushContext.row_end ({self.row_end}) must be >= row_start ({self.row_start})")
        if self.rows_seen_total < self.row_end:
            raise ValueError(f"AggregationFlushContext.rows_seen_total ({self.rows_seen_total}) must be >= row_end ({self.row_end})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_type": self.trigger_type,
            "buffer_size": self.buffer_size,
            "batch_id": self.batch_id,
            "flush_index": self.flush_index,
            "rows_seen_total": self.rows_seen_total,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "is_end_of_source": self.is_end_of_source,
        }


@dataclass(frozen=True, slots=True)
class AggregationBatchContext:
    """Metadata about the aggregation flush currently executing.

    Injected on :class:`elspeth.contracts.plugin_context.PluginContext`
    immediately before a batch-aware transform's ``process()`` is called,
    and cleared afterwards (both on the success cleanup path and on the
    failure cleanup path). Batch-aware transforms that need durable
    pagination metadata (flush_index, row_start, row_end, etc.) read it
    from ``ctx.aggregation_batch`` rather than maintaining their own
    uncheckpointed counters.

    The counters (``flush_index``, ``rows_seen_total``) are owned by
    :class:`AggregationExecutor` and are persisted in the aggregation
    checkpoint so they survive crash recovery.
    """

    trigger_type: str
    batch_id: str
    batch_size: int
    flush_index: int
    rows_seen_total: int
    row_start: int
    row_end: int
    is_end_of_source: bool

    def __post_init__(self) -> None:
        if not self.trigger_type:
            raise ValueError("AggregationBatchContext.trigger_type must not be empty")
        if not self.batch_id:
            raise ValueError("AggregationBatchContext.batch_id must not be empty")
        require_int(self.batch_size, "batch_size", min_value=1)
        require_int(self.flush_index, "flush_index", min_value=1)
        require_int(self.rows_seen_total, "rows_seen_total", min_value=1)
        require_int(self.row_start, "row_start", min_value=1)
        require_int(self.row_end, "row_end", min_value=1)
        if self.row_end < self.row_start:
            raise ValueError(f"AggregationBatchContext.row_end ({self.row_end}) must be >= row_start ({self.row_start})")
        if self.rows_seen_total < self.row_end:
            raise ValueError(f"AggregationBatchContext.rows_seen_total ({self.rows_seen_total}) must be >= row_end ({self.row_end})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_type": self.trigger_type,
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "flush_index": self.flush_index,
            "rows_seen_total": self.rows_seen_total,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "is_end_of_source": self.is_end_of_source,
        }
