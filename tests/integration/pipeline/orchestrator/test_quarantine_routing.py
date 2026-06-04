# tests/integration/pipeline/orchestrator/test_quarantine_routing.py
"""Regression tests for orchestrator quarantine routing and audit path.

Covers bead scug.1: quarantine edge-cases and happy-path audit assertions.

Target code:
- src/elspeth/engine/orchestrator/core.py:1259-1387

Tests:
1. Missing quarantine_destination -> RouteValidationError
2. Invalid quarantine_destination (not in sinks) -> RouteValidationError
3. Missing __quarantine__ DIVERT edge -> OrchestrationInvariantError
4. Happy-path quarantine: FAILED node_state + DIVERT routing_event +
   QUARANTINED outcome deferred to sink write
5. Mixed valid + quarantined rows: valid rows complete normally,
   quarantined rows route to quarantine sink with full audit trail
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts import (
    NodeStateStatus,
    RoutingMode,
    RunStatus,
    SourceRow,
    TerminalOutcome,
    TerminalPath,
)
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.events import RowCreated
from elspeth.contracts.hashing import repr_hash
from elspeth.core.canonical import sanitize_for_canonical, stable_hash
from elspeth.core.landscape.schema import (
    edges_table,
    node_states_table,
    routing_events_table,
    rows_table,
    token_outcomes_table,
    tokens_table,
)
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.types import RouteValidationError
from tests.fixtures.base_classes import _TestSchema, _TestSourceBase, as_sink, as_source
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink

# ---------------------------------------------------------------------------
# Test Sources
# ---------------------------------------------------------------------------


class CapturingTelemetryManager:
    """Minimal telemetry manager double that records emitted events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def handle_event(self, event: Any) -> None:
        self.events.append(event)

    def flush(self) -> None:
        pass


class QuarantineSource(_TestSourceBase):
    """Source that yields a mix of valid and quarantined rows.

    Quarantine rows use the configured destination.
    """

    name = "quarantine_source"
    output_schema = _TestSchema

    def __init__(
        self,
        valid_rows: list[dict[str, Any]],
        quarantine_rows: list[tuple[Any, str]],
        *,
        quarantine_destination: str = "quarantine",
        on_success: str = "default",
        on_validation_failure: str = "quarantine",
    ) -> None:
        super().__init__()
        self._valid_rows = valid_rows
        self._quarantine_rows = quarantine_rows
        self._quarantine_destination = quarantine_destination
        self.on_success = on_success
        self._on_validation_failure = on_validation_failure

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        # Yield valid rows first
        for source_row_index, row in enumerate(self._valid_rows):
            fields = tuple(
                FieldContract(
                    normalized_name=key,
                    original_name=key,
                    python_type=object,
                    required=False,
                    source="inferred",
                )
                for key in row
            )
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            if self._schema_contract is None:
                self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract, source_row_index=source_row_index)

        # Yield quarantined rows
        for quarantine_index, (row_data, error_msg) in enumerate(self._quarantine_rows, start=len(self._valid_rows)):
            yield SourceRow.quarantined(
                row=row_data,
                error=error_msg,
                destination=self._quarantine_destination,
                source_row_index=quarantine_index,
            )


class MissingDestinationSource(_TestSourceBase):
    """Source that yields a quarantined row with missing destination."""

    name = "bad_source_missing_dest"
    output_schema = _TestSchema

    def __init__(self, *, on_validation_failure: str = "quarantine") -> None:
        super().__init__()
        self._on_validation_failure = on_validation_failure
        self.on_success = "default"

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        # Manually construct a SourceRow with empty destination to simulate plugin bug
        yield SourceRow(
            row={"bad": "data"},
            is_quarantined=True,
            quarantine_error="validation failed",
            quarantine_destination="",  # Empty string — falsy
            source_row_index=0,
        )


class InvalidDestinationSource(_TestSourceBase):
    """Source that yields a quarantined row with a sink name that doesn't exist."""

    name = "bad_source_invalid_dest"
    output_schema = _TestSchema

    def __init__(self, *, on_validation_failure: str = "nonexistent_sink") -> None:
        super().__init__()
        self._on_validation_failure = on_validation_failure
        self.on_success = "default"

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        yield SourceRow.quarantined(
            row={"bad": "data"},
            error="validation failed",
            destination="nonexistent_sink",
            source_row_index=0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQuarantineRouteValidation:
    """Edge cases: missing/invalid quarantine destinations raise immediately."""

    def test_missing_quarantine_destination_raises_route_validation_error(self, payload_store) -> None:
        """Missing quarantine_destination is a plugin bug — must crash, not drop silently."""
        db = make_landscape_db()
        source = MissingDestinationSource(on_validation_failure="quarantine")
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(RouteValidationError, match="missing quarantine_destination"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Verify no rows leaked to any sink
        assert len(default_sink.results) == 0
        assert len(quarantine_sink.results) == 0

    def test_invalid_quarantine_destination_raises_route_validation_error(self, payload_store) -> None:
        """Quarantine destination pointing to non-existent sink must crash.

        The pre-run validation in orchestrator/validation.py catches this before
        the row-level quarantine code at core.py:1275 ever runs.
        """
        db = make_landscape_db()
        # Source says destination is "nonexistent_sink" but only "default" exists
        source = InvalidDestinationSource(on_validation_failure="nonexistent_sink")
        default_sink = CollectSink("default")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink)},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(RouteValidationError, match=r"on_validation_failure.*nonexistent_sink"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert len(default_sink.results) == 0

    def test_row_level_invalid_quarantine_destination_raises(self, payload_store) -> None:
        """Row-level quarantine_destination mismatch with actual sinks crashes at core.py:1275.

        This tests the second validation layer: the source passes pre-run validation
        (on_validation_failure="quarantine" matches a real sink), but yields a quarantined
        row whose destination field points to a different, non-existent sink.
        """
        db = make_landscape_db()

        class MismatchDestSource(_TestSourceBase):
            """Source that passes pre-run validation but yields quarantine rows with wrong dest."""

            name = "mismatch_source"
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__()
                self._on_validation_failure = "quarantine"  # Valid — matches a real sink
                self.on_success = "default"

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                # Row claims its destination is "wrong_sink" — not in config.sinks
                yield SourceRow.quarantined(
                    row={"bad": True},
                    error="validation error",
                    destination="wrong_sink",
                    source_row_index=0,
                )

        source = MismatchDestSource()
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        with pytest.raises(RouteValidationError, match=r"invalid quarantine_destination.*wrong_sink"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert len(default_sink.results) == 0
        assert len(quarantine_sink.results) == 0

    def test_missing_quarantine_edge_raises_orchestration_invariant_error(self, payload_store) -> None:
        """If the DAG lacks the __quarantine__ DIVERT edge, OrchestrationInvariantError."""
        db = make_landscape_db()

        # Build a source that yields quarantined rows with valid destination
        source = QuarantineSource(
            valid_rows=[],
            quarantine_rows=[({}, "bad row")],
            quarantine_destination="quarantine",
            on_validation_failure="discard",  # "discard" prevents DAG from creating __quarantine__ edge
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        # Build graph — since on_validation_failure="discard", no __quarantine__ edge is created
        graph = build_production_graph(config)

        # Verify __quarantine__ edge is indeed missing from the graph
        nx_graph = graph.get_nx_graph()
        quarantine_edges = [(u, v, d) for u, v, k, d in nx_graph.edges(data=True, keys=True) if d.get("label") == "__quarantine__"]
        assert len(quarantine_edges) == 0, "Test setup: __quarantine__ edge should not exist"

        orchestrator = Orchestrator(db)
        with pytest.raises(OrchestrationInvariantError, match=r"no __quarantine__.*DIVERT edge"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)


class TestQuarantineHappyPath:
    """Happy path: quarantine rows create full audit trail."""

    def test_quarantine_creates_failed_source_node_state(self, payload_store) -> None:
        """Quarantined rows get a FAILED node_state at step_index=0 (source)."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[],
            quarantine_rows=[({"x": "bad"}, "field 'x' type mismatch")],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Quarantine is a clean terminal outcome per CLAUDE.md Tier-3
        # manifesto. All-quarantined run satisfies the
        # ``terminal_clean_indicator`` (via rows_quarantined > 0) with no
        # uncaught ``failure_indicator`` => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_quarantined == 1

        # Query node_states for the quarantined token
        with db.engine.connect() as conn:
            states = conn.execute(select(node_states_table).where(node_states_table.c.run_id == result.run_id)).fetchall()

        # Find the FAILED source state (quarantine path)
        failed_states = [s for s in states if s.status == NodeStateStatus.FAILED]
        assert len(failed_states) == 1, f"Expected exactly 1 FAILED node_state, got {len(failed_states)}"

        failed_state = failed_states[0]
        assert failed_state.step_index == 0, "Quarantine node_state must be at step_index=0 (source)"
        assert "type mismatch" in failed_state.error_json, "Error should contain quarantine error message"

    def test_quarantine_creates_divert_routing_event(self, payload_store) -> None:
        """Quarantined rows get a DIVERT routing_event linking source to quarantine sink."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[],
            quarantine_rows=[({"x": 1}, "invalid value")],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Quarantine is a clean terminal outcome per CLAUDE.md Tier-3
        # manifesto. All-quarantined run satisfies the
        # ``terminal_clean_indicator`` (via rows_quarantined > 0) with no
        # uncaught ``failure_indicator`` => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

        # Query routing events
        with db.engine.connect() as conn:
            events = conn.execute(
                select(routing_events_table)
                .join(node_states_table, routing_events_table.c.state_id == node_states_table.c.state_id)
                .where(node_states_table.c.run_id == result.run_id)
            ).fetchall()

        divert_events = [e for e in events if e.mode == RoutingMode.DIVERT]
        assert len(divert_events) == 1, f"Expected exactly 1 DIVERT routing_event, got {len(divert_events)}"

        # Verify the edge is the __quarantine__ edge
        divert_event = divert_events[0]
        with db.engine.connect() as conn:
            edge = conn.execute(select(edges_table).where(edges_table.c.edge_id == divert_event.edge_id)).fetchone()

        assert edge is not None, "DIVERT routing_event must reference a valid edge"
        assert edge.label == "__quarantine__", f"Expected __quarantine__ edge label, got {edge.label}"
        assert edge.default_mode == RoutingMode.DIVERT, "Quarantine edge must have DIVERT mode"

    def test_quarantine_outcome_recorded_after_sink_write(self, payload_store) -> None:
        """QUARANTINED outcome must only be recorded after sink durability (not eagerly)."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[],
            quarantine_rows=[({"bad": True}, "schema violation")],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Quarantine is a clean terminal outcome per CLAUDE.md Tier-3
        # manifesto. All-quarantined run satisfies the
        # ``terminal_clean_indicator`` (via rows_quarantined > 0) with no
        # uncaught ``failure_indicator`` => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_quarantined == 1

        # Verify the quarantine sink received the row
        assert len(quarantine_sink.results) == 1

        # Query token outcomes
        with db.engine.connect() as conn:
            outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == result.run_id)).fetchall()
            rows = conn.execute(select(rows_table.c.source_row_index).where(rows_table.c.run_id == result.run_id)).scalars().all()

        quarantined_outcomes = [
            o for o in outcomes if (o.outcome, o.path) == (TerminalOutcome.FAILURE.value, TerminalPath.QUARANTINED_AT_SOURCE.value)
        ]
        assert len(quarantined_outcomes) == 1, f"Expected 1 QUARANTINED outcome, got {len(quarantined_outcomes)}"
        assert rows == [0]

        outcome = quarantined_outcomes[0]
        assert outcome.completed == 1, "QUARANTINED must be a terminal outcome"
        assert outcome.sink_name == "quarantine", "QUARANTINED_AT_SOURCE preserves the durable quarantine sink witness"
        assert outcome.error_hash is not None, "QUARANTINED outcome must have error_hash"

    def test_mixed_valid_and_quarantined_rows_full_audit(self, payload_store) -> None:
        """Both valid and quarantined rows get complete audit trails, routed correctly."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[{"value": 1}, {"value": 2}],
            quarantine_rows=[
                ({"value": "not_an_int"}, "Expected int, got str"),
                ({"missing_field": True}, "Required field 'value' missing"),
            ],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Phase 2.2: mixed valid + quarantined => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_processed == 4
        assert result.rows_quarantined == 2

        # Valid rows reached default sink
        assert len(default_sink.results) == 2
        assert default_sink.results[0]["value"] == 1
        assert default_sink.results[1]["value"] == 2

        # Quarantined rows reached quarantine sink
        assert len(quarantine_sink.results) == 2

        # Verify audit trail completeness
        with db.engine.connect() as conn:
            # All rows have tokens
            all_tokens = conn.execute(
                select(tokens_table)
                .join(
                    node_states_table,
                    tokens_table.c.token_id == node_states_table.c.token_id,
                )
                .where(node_states_table.c.run_id == result.run_id)
            ).fetchall()
            # Should have at least 4 distinct tokens (2 valid + 2 quarantined)
            token_ids = {t.token_id for t in all_tokens}
            assert len(token_ids) >= 4, f"Expected >= 4 tokens, got {len(token_ids)}"

            # All rows have terminal outcomes
            outcomes = conn.execute(
                select(token_outcomes_table)
                .where(token_outcomes_table.c.run_id == result.run_id)
                .where(token_outcomes_table.c.completed == 1)
            ).fetchall()

        completed_outcomes = [
            o for o in outcomes if (o.outcome, o.path) == (TerminalOutcome.SUCCESS.value, TerminalPath.DEFAULT_FLOW.value)
        ]
        quarantined_outcomes = [
            o for o in outcomes if (o.outcome, o.path) == (TerminalOutcome.FAILURE.value, TerminalPath.QUARANTINED_AT_SOURCE.value)
        ]

        assert len(completed_outcomes) == 2, f"Expected 2 COMPLETED outcomes, got {len(completed_outcomes)}"
        assert len(quarantined_outcomes) == 2, f"Expected 2 QUARANTINED outcomes, got {len(quarantined_outcomes)}"

        # QUARANTINED outcomes have error_hash, COMPLETED do not
        for qo in quarantined_outcomes:
            assert qo.error_hash is not None, "QUARANTINED must have error_hash"
            assert qo.sink_name == "quarantine", "QUARANTINED_AT_SOURCE preserves the durable quarantine sink witness"
        for co in completed_outcomes:
            assert co.sink_name == "default"

    def test_quarantined_rows_preserve_source_authored_row_indexes(self, payload_store) -> None:
        """Quarantined rows persist the source-authored source_row_index, not loop position."""
        db = make_landscape_db()

        class SparseQuarantineSource(_TestSourceBase):
            name = "sparse_quarantine_source"
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__()
                self.on_success = "default"
                self._on_validation_failure = "quarantine"

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.quarantined(
                    row={"bad": "first"},
                    error="first bad row",
                    destination="quarantine",
                    source_row_index=10,
                )
                yield SourceRow.quarantined(
                    row={"bad": "second"},
                    error="second bad row",
                    destination="quarantine",
                    source_row_index=42,
                )

        source = SparseQuarantineSource()
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        result = Orchestrator(db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == RunStatus.FAILED
        assert result.rows_quarantined == 2
        with db.engine.connect() as conn:
            source_row_indexes = (
                conn.execute(
                    select(rows_table.c.source_row_index).where(rows_table.c.run_id == result.run_id).order_by(rows_table.c.ingest_sequence)
                )
                .scalars()
                .all()
            )
        assert source_row_indexes == [10, 42]

    def test_quarantine_non_dict_row_data(self, payload_store) -> None:
        """Quarantined rows with non-dict data (e.g., JSON primitives) route correctly."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[{"value": 1}],
            # Non-dict quarantine data — simulates malformed JSON input
            quarantine_rows=[
                (42, "Expected object, got number"),
                ("raw string", "Expected object, got string"),
            ],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Phase 2.2: mixed valid + quarantined => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_quarantined == 2
        assert len(quarantine_sink.results) == 2
        assert len(default_sink.results) == 1

        # Non-dict data gets wrapped as {"_raw": value} in the audit trail
        with db.engine.connect() as conn:
            failed_states = conn.execute(
                select(node_states_table)
                .where(node_states_table.c.run_id == result.run_id)
                .where(node_states_table.c.status == NodeStateStatus.FAILED)
            ).fetchall()

        assert len(failed_states) == 2, "Each quarantined row gets its own FAILED node_state"

    def test_quarantine_counter_reflected_in_run_result(self, payload_store) -> None:
        """RunResult.rows_quarantined accurately counts quarantined rows."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[{"v": i} for i in range(5)],
            quarantine_rows=[({"bad": i}, f"error_{i}") for i in range(3)],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Phase 2.2: mixed valid + quarantined => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_processed == 8
        assert result.rows_quarantined == 3
        assert result.rows_succeeded == 5
        assert len(default_sink.results) == 5
        assert len(quarantine_sink.results) == 3


class TestQuarantineNonCanonicalData:
    """P1-2026-02-05: Quarantined rows with non-canonical data (NaN, Infinity)
    must not crash the pipeline during telemetry hash computation.

    stable_hash() uses canonical JSON which rejects NaN/Infinity per RFC 8785.
    Quarantined rows are Tier-3 external data that may contain these values.
    The fix uses repr_hash() as a fallback for non-canonical data.
    """

    def test_quarantined_row_with_nan_does_not_crash(self, payload_store) -> None:
        """Quarantined row containing NaN should route to quarantine sink, not crash."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[{"value": 1}],
            quarantine_rows=[
                ({"value": float("nan")}, "NaN value not allowed"),
            ],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Phase 2.2: mixed valid + quarantined => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_quarantined == 1
        assert len(quarantine_sink.results) == 1
        assert len(default_sink.results) == 1

    def test_quarantined_row_with_infinity_does_not_crash(self, payload_store) -> None:
        """Quarantined row containing Infinity should route to quarantine sink, not crash."""
        db = make_landscape_db()
        source = QuarantineSource(
            valid_rows=[],
            quarantine_rows=[
                ({"score": float("inf")}, "Infinity not allowed"),
                ({"score": float("-inf")}, "Negative infinity not allowed"),
            ],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Quarantine is a clean terminal outcome per CLAUDE.md Tier-3
        # manifesto. All-quarantined run satisfies the
        # ``terminal_clean_indicator`` (via rows_quarantined > 0) with no
        # uncaught ``failure_indicator`` => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_quarantined == 2
        assert len(quarantine_sink.results) == 2

    def test_quarantined_row_with_canonical_data_uses_stable_hash(self, payload_store) -> None:
        """Quarantined rows with canonical data should still use stable_hash (not repr_hash)
        so hashes are consistent with non-quarantined rows containing the same data.
        """
        db = make_landscape_db()
        # Normal dict data that IS canonical — should use stable_hash
        source = QuarantineSource(
            valid_rows=[{"value": 42}],
            quarantine_rows=[
                ({"value": 42}, "Business logic rejection"),
            ],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Phase 2.2: mixed valid + quarantined => COMPLETED_WITH_FAILURES.
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_quarantined == 1
        assert result.rows_succeeded == 1

        # Both valid and quarantined rows completed without crash
        assert len(default_sink.results) == 1
        assert len(quarantine_sink.results) == 1

    def test_quarantined_nan_row_emits_single_semantics_content_hash(self, payload_store) -> None:
        """The quarantine RowCreated content_hash has a single, deterministic semantics.

        Offensive-programming contract: the quarantine path sanitizes Tier-3 data
        (NaN/Infinity -> None) BEFORE hashing (orchestrator pre-hash sanitize), so the
        emitted ``content_hash`` is always ``stable_hash(sanitize_for_canonical(row))``
        — never a divergent ``repr_hash`` of the raw row under the same field name.

        Pins the reified behaviour: removing the dead ``stable_hash``-then-``repr_hash``
        fallback must not change the emitted hash for legitimate Tier-3 input, and the
        hash must NOT silently switch to repr_hash semantics.
        """
        db = make_landscape_db()
        nan_row = {"value": float("nan")}
        source = QuarantineSource(
            valid_rows=[],
            quarantine_rows=[
                (dict(nan_row), "NaN value not allowed"),
            ],
            quarantine_destination="quarantine",
            on_validation_failure="quarantine",
        )
        default_sink = CollectSink("default")
        quarantine_sink = CollectSink("quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        telemetry = CapturingTelemetryManager()
        orchestrator = Orchestrator(db, telemetry_manager=telemetry)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_quarantined == 1

        row_created = [e for e in telemetry.events if isinstance(e, RowCreated)]
        assert len(row_created) == 1, f"expected exactly one RowCreated, got {row_created}"
        emitted_hash = row_created[0].content_hash

        # Single semantics: hash of the sanitized row via stable_hash.
        assert emitted_hash == stable_hash(sanitize_for_canonical(nan_row))
        # Explicitly NOT the divergent repr_hash of the raw (unsanitized) row.
        assert emitted_hash != repr_hash(nan_row)
