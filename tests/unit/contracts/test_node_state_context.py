"""Tests for node state context types.

Verifies:
- Dataclass construction and immutability
- to_dict() correctness for all types (PoolConfigSnapshot,
  PoolStatsSnapshot, QueryOrderEntry, PoolExecutionContext,
  GateEvaluationContext, AggregationFlushContext)
- from_executor_stats() factory (happy + malformed)
- Protocol conformance (CoalesceMetadata, PoolExecutionContext,
  GateEvaluationContext, AggregationFlushContext)
- canonical_json integration
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy
from elspeth.contracts.coalesce_metadata import ArrivalOrderEntry, CoalesceMetadata
from elspeth.contracts.node_state_context import (
    AggregationFlushContext,
    GateEvaluationContext,
    PoolConfigSnapshot,
    PoolExecutionContext,
    PoolStatsSnapshot,
    QueryOrderEntry,
)
from elspeth.core.canonical import canonical_json


class TestPoolExecutionContext:
    def _make_context(self) -> PoolExecutionContext:
        return PoolExecutionContext(
            pool_config=PoolConfigSnapshot(pool_size=4, max_capacity_retry_seconds=30.0, dispatch_delay_at_completion_ms=10.0),
            pool_stats=PoolStatsSnapshot(
                capacity_retries=1,
                successes=3,
                peak_delay_ms=20.0,
                current_delay_ms=10.0,
                total_throttle_time_ms=5.0,
                max_concurrent_reached=3,
            ),
            query_ordering=(
                QueryOrderEntry(submit_index=0, complete_index=1, buffer_wait_ms=2.0),
                QueryOrderEntry(submit_index=1, complete_index=0, buffer_wait_ms=0.0),
            ),
        )

    def test_to_dict_structure(self) -> None:
        ctx = self._make_context()
        d = ctx.to_dict()
        assert "pool_config" in d
        assert "pool_stats" in d
        assert "query_ordering" in d
        assert isinstance(d["query_ordering"], list)
        assert len(d["query_ordering"]) == 2

    def test_to_dict_values(self) -> None:
        ctx = self._make_context()
        d = ctx.to_dict()
        assert d["pool_config"]["pool_size"] == 4
        assert d["pool_stats"]["successes"] == 3
        assert d["query_ordering"][0]["submit_index"] == 0
        assert d["query_ordering"][1]["buffer_wait_ms"] == 0.0

    def test_canonical_json_produces_valid_json(self) -> None:
        ctx = self._make_context()
        json_str = canonical_json(ctx.to_dict())
        parsed = json.loads(json_str)
        assert parsed == ctx.to_dict()

    def test_canonical_json_deterministic(self) -> None:
        ctx = self._make_context()
        json1 = canonical_json(ctx.to_dict())
        json2 = canonical_json(ctx.to_dict())
        assert json1 == json2

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("max_capacity_retry_seconds", -0.1, ValueError),
            ("max_capacity_retry_seconds", float("nan"), ValueError),
            ("max_capacity_retry_seconds", "30", TypeError),
            ("dispatch_delay_at_completion_ms", -0.1, ValueError),
            ("dispatch_delay_at_completion_ms", float("inf"), ValueError),
            ("dispatch_delay_at_completion_ms", None, TypeError),
        ],
    )
    def test_pool_config_snapshot_rejects_invalid_timing(self, field: str, value: object, error: type[Exception]) -> None:
        kwargs: dict[str, Any] = {
            "pool_size": 4,
            "max_capacity_retry_seconds": 30.0,
            "dispatch_delay_at_completion_ms": 10.0,
        }
        kwargs[field] = value
        with pytest.raises(error, match=field):
            PoolConfigSnapshot(**kwargs)

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("peak_delay_ms", -0.1, ValueError),
            ("peak_delay_ms", float("nan"), ValueError),
            ("peak_delay_ms", "20", TypeError),
            ("current_delay_ms", -0.1, ValueError),
            ("current_delay_ms", float("inf"), ValueError),
            ("current_delay_ms", None, TypeError),
            ("total_throttle_time_ms", -0.1, ValueError),
            ("total_throttle_time_ms", float("-inf"), ValueError),
            ("total_throttle_time_ms", object(), TypeError),
        ],
    )
    def test_pool_stats_snapshot_rejects_invalid_timing(self, field: str, value: object, error: type[Exception]) -> None:
        kwargs: dict[str, Any] = {
            "capacity_retries": 1,
            "successes": 3,
            "peak_delay_ms": 20.0,
            "current_delay_ms": 10.0,
            "total_throttle_time_ms": 5.0,
            "max_concurrent_reached": 3,
        }
        kwargs[field] = value
        with pytest.raises(error, match=field):
            PoolStatsSnapshot(**kwargs)

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (-0.1, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            ("0", TypeError),
        ],
    )
    def test_query_order_entry_rejects_invalid_buffer_wait_ms(self, value: object, error: type[Exception]) -> None:
        with pytest.raises(error, match="buffer_wait_ms"):
            QueryOrderEntry(submit_index=0, complete_index=1, buffer_wait_ms=value)  # type: ignore[arg-type]

    def test_rejects_non_pool_config_snapshot(self) -> None:
        with pytest.raises(TypeError, match="pool_config must be PoolConfigSnapshot"):
            PoolExecutionContext(
                pool_config=object(),  # type: ignore[arg-type]
                pool_stats=PoolStatsSnapshot(
                    capacity_retries=0,
                    successes=0,
                    peak_delay_ms=0.0,
                    current_delay_ms=0.0,
                    total_throttle_time_ms=0.0,
                    max_concurrent_reached=0,
                ),
                query_ordering=(),
            )

    def test_rejects_non_pool_stats_snapshot(self) -> None:
        with pytest.raises(TypeError, match="pool_stats must be PoolStatsSnapshot"):
            PoolExecutionContext(
                pool_config=PoolConfigSnapshot(pool_size=1, max_capacity_retry_seconds=1.0, dispatch_delay_at_completion_ms=0.0),
                pool_stats=object(),  # type: ignore[arg-type]
                query_ordering=(),
            )

    def test_rejects_non_tuple_query_ordering(self) -> None:
        with pytest.raises(TypeError, match="query_ordering must be tuple"):
            PoolExecutionContext(
                pool_config=PoolConfigSnapshot(pool_size=1, max_capacity_retry_seconds=1.0, dispatch_delay_at_completion_ms=0.0),
                pool_stats=PoolStatsSnapshot(
                    capacity_retries=0,
                    successes=0,
                    peak_delay_ms=0.0,
                    current_delay_ms=0.0,
                    total_throttle_time_ms=0.0,
                    max_concurrent_reached=0,
                ),
                query_ordering=[],  # type: ignore[arg-type]
            )

    def test_rejects_non_query_order_entry(self) -> None:
        with pytest.raises(TypeError, match=r"query_ordering\[0\] must be QueryOrderEntry"):
            PoolExecutionContext(
                pool_config=PoolConfigSnapshot(pool_size=1, max_capacity_retry_seconds=1.0, dispatch_delay_at_completion_ms=0.0),
                pool_stats=PoolStatsSnapshot(
                    capacity_retries=0,
                    successes=0,
                    peak_delay_ms=0.0,
                    current_delay_ms=0.0,
                    total_throttle_time_ms=0.0,
                    max_concurrent_reached=0,
                ),
                query_ordering=(object(),),  # type: ignore[arg-type]
            )


class TestGateEvaluationContext:
    def test_canonical_json_produces_valid_json(self) -> None:
        ctx = GateEvaluationContext(condition="status == 'active'", result="True", route_label="active")
        json_str = canonical_json(ctx.to_dict())
        parsed = json.loads(json_str)
        assert parsed == ctx.to_dict()

    def test_canonical_json_deterministic(self) -> None:
        ctx = GateEvaluationContext(condition="score >= 0.5", result="False", route_label="low_score")
        json1 = canonical_json(ctx.to_dict())
        json2 = canonical_json(ctx.to_dict())
        assert json1 == json2

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("condition", "", ValueError),
            ("condition", None, TypeError),
            ("result", "", ValueError),
            ("result", False, TypeError),
            ("route_label", "", ValueError),
            ("route_label", 1, TypeError),
        ],
    )
    def test_rejects_invalid_audit_fields(self, field: str, value: object, error: type[Exception]) -> None:
        kwargs: dict[str, Any] = {"condition": "x > 0", "result": "True", "route_label": "true"}
        kwargs[field] = value
        with pytest.raises(error, match=field):
            GateEvaluationContext(**kwargs)


class TestAggregationFlushContext:
    def test_canonical_json_produces_valid_json(self) -> None:
        ctx = AggregationFlushContext(
            trigger_type="END_OF_SOURCE",
            buffer_size=37,
            batch_id="batch-final",
            flush_index=4,
            rows_seen_total=137,
            row_start=101,
            row_end=137,
            is_end_of_source=True,
        )
        json_str = canonical_json(ctx.to_dict())
        parsed = json.loads(json_str)
        assert parsed == ctx.to_dict()

    def test_canonical_json_deterministic(self) -> None:
        ctx = AggregationFlushContext(
            trigger_type="COUNT",
            buffer_size=10,
            batch_id="batch-xyz",
            flush_index=2,
            rows_seen_total=20,
            row_start=11,
            row_end=20,
            is_end_of_source=False,
        )
        json1 = canonical_json(ctx.to_dict())
        json2 = canonical_json(ctx.to_dict())
        assert json1 == json2

    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("trigger_type", "", ValueError),
            ("trigger_type", None, TypeError),
            ("batch_id", "", ValueError),
            ("batch_id", 123, TypeError),
        ],
    )
    def test_rejects_invalid_audit_fields(self, field: str, value: object, error: type[Exception]) -> None:
        kwargs: dict[str, Any] = {
            "trigger_type": "COUNT",
            "buffer_size": 10,
            "batch_id": "batch-xyz",
            "flush_index": 2,
            "rows_seen_total": 20,
            "row_start": 11,
            "row_end": 20,
            "is_end_of_source": False,
        }
        kwargs[field] = value
        with pytest.raises(error, match=field):
            AggregationFlushContext(**kwargs)


class TestRequireIntValidation:
    """require_int guards reject bool (and wrong types) on int fields."""

    def test_pool_config_snapshot_rejects_bool_pool_size(self) -> None:
        with pytest.raises(TypeError, match="pool_size must be int"):
            PoolConfigSnapshot(pool_size=True, max_capacity_retry_seconds=30.0, dispatch_delay_at_completion_ms=10.0)

    def test_pool_stats_snapshot_rejects_bool_capacity_retries(self) -> None:
        with pytest.raises(TypeError, match="capacity_retries must be int"):
            PoolStatsSnapshot(
                capacity_retries=True,
                successes=0,
                peak_delay_ms=0.0,
                current_delay_ms=0.0,
                total_throttle_time_ms=0.0,
                max_concurrent_reached=0,
            )

    def test_pool_stats_snapshot_rejects_bool_successes(self) -> None:
        with pytest.raises(TypeError, match="successes must be int"):
            PoolStatsSnapshot(
                capacity_retries=0,
                successes=False,
                peak_delay_ms=0.0,
                current_delay_ms=0.0,
                total_throttle_time_ms=0.0,
                max_concurrent_reached=0,
            )

    def test_pool_stats_snapshot_rejects_bool_max_concurrent_reached(self) -> None:
        with pytest.raises(TypeError, match="max_concurrent_reached must be int"):
            PoolStatsSnapshot(
                capacity_retries=0,
                successes=0,
                peak_delay_ms=0.0,
                current_delay_ms=0.0,
                total_throttle_time_ms=0.0,
                max_concurrent_reached=True,
            )

    def test_query_order_entry_rejects_bool_submit_index(self) -> None:
        with pytest.raises(TypeError, match="submit_index must be int"):
            QueryOrderEntry(submit_index=True, complete_index=0, buffer_wait_ms=0.0)

    def test_query_order_entry_rejects_bool_complete_index(self) -> None:
        with pytest.raises(TypeError, match="complete_index must be int"):
            QueryOrderEntry(submit_index=0, complete_index=False, buffer_wait_ms=0.0)

    def test_aggregation_flush_context_rejects_bool_buffer_size(self) -> None:
        with pytest.raises(TypeError, match="buffer_size must be int"):
            AggregationFlushContext(
                trigger_type="COUNT",
                buffer_size=True,
                batch_id="b1",
                flush_index=1,
                rows_seen_total=1,
                row_start=1,
                row_end=1,
                is_end_of_source=False,
            )


class TestFromExecutorStats:
    """Tests for PoolExecutionContext.from_executor_stats() factory."""

    def _make_stats(self) -> dict[str, Any]:
        return {
            "pool_config": {
                "pool_size": 4,
                "max_capacity_retry_seconds": 30.0,
                "dispatch_delay_at_completion_ms": 10.0,
            },
            "pool_stats": {
                "capacity_retries": 0,
                "successes": 2,
                "peak_delay_ms": 15.0,
                "current_delay_ms": 10.0,
                "total_throttle_time_ms": 0.0,
                "max_concurrent_reached": 2,
            },
        }

    def _make_entries(self) -> list[Any]:
        """Create mock BufferEntry objects with the required attributes."""
        from dataclasses import dataclass

        @dataclass
        class FakeEntry:
            submit_index: int
            complete_index: int
            buffer_wait_ms: float
            result: object = None

        return [
            FakeEntry(submit_index=0, complete_index=1, buffer_wait_ms=3.0),
            FakeEntry(submit_index=1, complete_index=0, buffer_wait_ms=0.0),
        ]

    def test_from_executor_stats_happy_path(self) -> None:
        ctx = PoolExecutionContext.from_executor_stats(
            stats=self._make_stats(),
            entries=self._make_entries(),
        )
        assert ctx.pool_config.pool_size == 4
        assert ctx.pool_stats.successes == 2
        assert len(ctx.query_ordering) == 2
        assert ctx.query_ordering[0].buffer_wait_ms == 3.0

    def test_from_executor_stats_malformed_crashes(self) -> None:
        """Tier 1 rule: bad data in our code = crash, not fabrication."""
        with pytest.raises(KeyError):
            PoolExecutionContext.from_executor_stats(
                stats={"pool_config": {}},  # Missing keys
                entries=self._make_entries(),
            )

    def test_from_executor_stats_missing_pool_stats_crashes(self) -> None:
        with pytest.raises(KeyError):
            PoolExecutionContext.from_executor_stats(
                stats={"pool_config": {"pool_size": 4, "max_capacity_retry_seconds": 30.0, "dispatch_delay_at_completion_ms": 10.0}},
                entries=self._make_entries(),
            )


class TestProtocolConformance:
    """Verify both context types satisfy NodeStateContext structurally."""

    def test_coalesce_metadata_has_to_dict(self) -> None:
        """CoalesceMetadata satisfies NodeStateContext protocol."""
        meta = CoalesceMetadata.for_late_arrival(policy=CoalescePolicy.REQUIRE_ALL, reason="test")
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert "policy" in d

    def test_pool_execution_context_has_to_dict(self) -> None:
        """PoolExecutionContext satisfies NodeStateContext protocol."""
        ctx = PoolExecutionContext(
            pool_config=PoolConfigSnapshot(pool_size=1, max_capacity_retry_seconds=1.0, dispatch_delay_at_completion_ms=0.0),
            pool_stats=PoolStatsSnapshot(
                capacity_retries=0,
                successes=0,
                peak_delay_ms=0.0,
                current_delay_ms=0.0,
                total_throttle_time_ms=0.0,
                max_concurrent_reached=0,
            ),
            query_ordering=(),
        )
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert "pool_config" in d

    def test_gate_evaluation_context_has_to_dict(self) -> None:
        """GateEvaluationContext satisfies NodeStateContext protocol."""
        ctx = GateEvaluationContext(condition="x > 0", result="True", route_label="positive")
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert "condition" in d

    def test_aggregation_flush_context_has_to_dict(self) -> None:
        """AggregationFlushContext satisfies NodeStateContext protocol."""
        ctx = AggregationFlushContext(
            trigger_type="COUNT",
            buffer_size=10,
            batch_id="b1",
            flush_index=1,
            rows_seen_total=10,
            row_start=1,
            row_end=10,
            is_end_of_source=False,
        )
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert "trigger_type" in d

    def test_coalesce_metadata_canonical_json(self) -> None:
        meta = CoalesceMetadata.for_merge(
            policy=CoalescePolicy.REQUIRE_ALL,
            merge_strategy=MergeStrategy.UNION,
            expected_branches=["a", "b"],
            branches_arrived=["a", "b"],
            branches_lost={},
            arrival_order=[ArrivalOrderEntry(branch="a", arrival_offset_ms=0.0)],
            wait_duration_ms=100.0,
        )
        json_str = canonical_json(meta.to_dict())
        parsed = json.loads(json_str)
        assert parsed == meta.to_dict()
