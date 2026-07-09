# tests/core/landscape/test_schema.py
"""Tests for Landscape SQLAlchemy schema."""

from datetime import UTC, datetime
from enum import StrEnum

import elspeth.core.landscape.schema as schema
from elspeth.contracts import Determinism
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.core.landscape.schema import (
    RunSourceLifecycleState,
    run_sources_table,
    scheduler_events_table,
    token_work_items_table,
)


def _check_sql(table_name: str, constraint_name: str) -> str:
    table = {
        "run_sources": run_sources_table,
        "scheduler_events": scheduler_events_table,
        "token_work_items": token_work_items_table,
    }[table_name]
    check = next(c for c in table.constraints if getattr(c, "name", "") == constraint_name)
    return str(check.sqltext)  # type: ignore[attr-defined]


def _enum_in_check(column_name: str, enum_type: type[StrEnum]) -> str:
    assert hasattr(schema, "_enum_in_check")
    return schema._enum_in_check(column_name, enum_type)


def _optional_enum_in_check(column_name: str, enum_type: type[StrEnum]) -> str:
    assert hasattr(schema, "_optional_enum_in_check")
    return schema._optional_enum_in_check(column_name, enum_type)


def _sql_string_literal(value: str) -> str:
    assert hasattr(schema, "_sql_string_literal")
    return schema._sql_string_literal(value)


class TestEnumCheckConstraints:
    """Schema CHECK value spaces should be generated from enum owners."""

    def test_enum_in_check_renders_deterministic_sql(self) -> None:
        class Example(StrEnum):
            FIRST = "first"
            QUOTED = "has'quote"

        assert _enum_in_check("status", Example) == "status IN ('first', 'has''quote')"

    def test_optional_enum_in_check_allows_null_then_enum_values(self) -> None:
        assert _optional_enum_in_check("from_status", TokenWorkStatus) == (
            "from_status IS NULL OR from_status IN ('ready', 'leased', 'blocked', 'pending_sink', 'terminal', 'failed')"
        )

    def test_run_source_lifecycle_check_matches_enum_values(self) -> None:
        assert _check_sql("run_sources", "ck_run_sources_lifecycle_state") == _enum_in_check(
            "lifecycle_state",
            RunSourceLifecycleState,
        )

    def test_scheduler_event_type_check_matches_enum_values(self) -> None:
        assert _check_sql("scheduler_events", "ck_scheduler_events_event_type") == _enum_in_check(
            "event_type",
            SchedulerEventType,
        )

    def test_scheduler_status_checks_match_enum_values(self) -> None:
        assert _check_sql("scheduler_events", "ck_scheduler_events_from_status") == _optional_enum_in_check(
            "from_status",
            TokenWorkStatus,
        )
        assert _check_sql("scheduler_events", "ck_scheduler_events_to_status") == _enum_in_check(
            "to_status",
            TokenWorkStatus,
        )

    def test_lease_owner_check_uses_leased_enum_literal(self) -> None:
        leased = _sql_string_literal(TokenWorkStatus.LEASED.value)
        assert _check_sql("token_work_items", "ck_token_work_items_lease_owner_required_when_leased") == (
            f"(status = {leased} AND lease_owner IS NOT NULL AND length(lease_owner) > 0) OR status != {leased}"
        )


class TestNodesDeterminismColumn:
    """Tests for determinism column in nodes table."""

    def test_node_model_has_determinism_field(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts import Node, NodeType

        node = Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="test_plugin",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,  # New field
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
        assert node.determinism == Determinism.DETERMINISTIC

    def test_determinism_values(self) -> None:
        """Verify valid determinism values match Determinism enum."""

        valid_values = {d.value for d in Determinism}
        # All 6 values per architecture specification
        expected = {
            "deterministic",
            "seeded",
            "io_read",
            "io_write",
            "external_call",
            "non_deterministic",
        }
        assert valid_values == expected


class TestPhase5CheckpointSchema:
    """Tests for checkpoint table added in Phase 5."""

    def test_checkpoints_table_has_topology_validation_columns(self) -> None:
        """P1: Verify checkpoint topology hash columns exist and are non-nullable.

        These columns were added in Bug #7 fix for checkpoint validation.
        A schema regression that removes them would break checkpoint
        compatibility validation and undermine recovery integrity.
        """
        from elspeth.core.landscape.schema import checkpoints_table

        columns = {c.name: c for c in checkpoints_table.columns}

        # upstream_topology_hash must exist and be non-nullable
        assert "upstream_topology_hash" in columns, "Missing upstream_topology_hash column"
        assert columns["upstream_topology_hash"].nullable is False, "upstream_topology_hash must be non-nullable for checkpoint validation"

    def test_checkpoint_model(self) -> None:
        from elspeth.contracts import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="cp-001",
            run_id="run-001",
            sequence_number=42,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
        )
        assert checkpoint.sequence_number == 42

    def test_checkpoint_model_with_barrier_scalars(self) -> None:
        """Verify Checkpoint model carries barrier scalar metadata (F1)."""
        from elspeth.contracts import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="cp-002",
            run_id="run-001",
            sequence_number=100,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            barrier_scalars_json='{"aggregation": {}, "coalesce": []}',
        )
        assert checkpoint.barrier_scalars_json == '{"aggregation": {}, "coalesce": []}'


class TestBatchStatusType:
    """Tests for Batch.status type being BatchStatus enum (WP-05 Task 4)."""

    def test_batch_status_is_typed(self) -> None:
        """Batch.status should accept BatchStatus enum."""
        from elspeth.contracts import Batch, BatchStatus

        batch = Batch(
            batch_id="b1",
            run_id="r1",
            aggregation_node_id="agg1",
            attempt=1,
            status=BatchStatus.DRAFT,  # Should work without type error
            created_at=datetime.now(UTC),
        )

        assert batch.status == BatchStatus.DRAFT
