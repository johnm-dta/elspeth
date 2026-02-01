# tests/property/audit/test_terminal_states.py
"""Property-based tests for the terminal state invariant.

THE FOUNDATIONAL AUDIT PROPERTY:
Every token reaches EXACTLY ONE terminal state.

This is not negotiable. If a token goes missing without reaching a terminal
state, the audit trail is incomplete and ELSPETH's core value proposition
(attributability) is compromised.

These tests use Hypothesis to generate thousands of random pipeline inputs
and verify that the terminal state invariant holds for ALL of them.

Terminal states (from RowOutcome):
- COMPLETED: Reached output sink successfully
- ROUTED: Sent to named sink by gate
- FORKED: Split into multiple parallel paths (parent token)
- FAILED: Processing failed, not recoverable
- QUARANTINED: Failed validation, stored for investigation
- CONSUMED_IN_BATCH: Absorbed into aggregate
- COALESCED: Merged in join from parallel paths
- EXPANDED: Deaggregated into child tokens

Non-terminal state:
- BUFFERED: Temporarily held, will reappear with final outcome
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import text

from elspeth.contracts.enums import RowOutcome
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.conftest import (
    MockPayloadStore,
    as_sink,
    as_source,
    as_transform,
)
from tests.engine.orchestrator_test_helpers import build_production_graph
from tests.property.conftest import (
    MAX_SAFE_INT,
    CollectSink,
    ConditionalErrorTransform,
    ListSource,
    PassTransform,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Audit Verification Helpers
# =============================================================================


def count_tokens_missing_terminal(db: LandscapeDB, run_id: str) -> int:
    """Count tokens that lack a terminal outcome.

    This is the core invariant check: every token should have exactly
    one terminal outcome recorded.
    """
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                LEFT JOIN token_outcomes o
                  ON o.token_id = t.token_id AND o.is_terminal = 1
                WHERE r.run_id = :run_id
                  AND o.token_id IS NULL
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def count_duplicate_terminal_outcomes(db: LandscapeDB, run_id: str) -> int:
    """Count tokens with more than one terminal outcome.

    A token should have EXACTLY ONE terminal outcome, not zero, not multiple.
    """
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM (
                    SELECT o.token_id, COUNT(*) AS terminal_count
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE o.is_terminal = 1 AND r.run_id = :run_id
                    GROUP BY o.token_id
                    HAVING COUNT(*) > 1
                ) duplicates
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def get_all_token_outcomes(db: LandscapeDB, run_id: str) -> list[tuple[str, str, bool]]:
    """Get all token outcomes for a run.

    Returns list of (token_id, outcome, is_terminal) tuples.
    Used for detailed debugging when invariants fail.
    """
    with db.connection() as conn:
        results = conn.execute(
            text("""
                SELECT o.token_id, o.outcome, o.is_terminal
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                ORDER BY o.token_id, o.recorded_at
            """),
            {"run_id": run_id},
        ).fetchall()
        return [(r[0], r[1], bool(r[2])) for r in results]


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for row data - simple key-value pairs (RFC 8785 safe)
row_value = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=50),
    st.booleans(),
    st.none(),
)

# Strategy for a single row - dict with string keys (RFC 8785 safe integers)
single_row = st.fixed_dictionaries(
    {"id": st.integers(min_value=0, max_value=MAX_SAFE_INT)},
    optional={"value": row_value, "name": st.text(max_size=20), "flag": st.booleans()},
)

# Strategy for row that might trigger errors (RFC 8785 safe integers)
row_with_possible_error = st.fixed_dictionaries(
    {"id": st.integers(min_value=0, max_value=MAX_SAFE_INT), "fail": st.booleans()},
    optional={"value": row_value},
)


# =============================================================================
# Property Tests: Terminal State Invariant
# =============================================================================


class TestTerminalStateProperty:
    """Property tests for the terminal state invariant."""

    @given(rows=st.lists(single_row, min_size=0, max_size=50))
    @settings(max_examples=100, deadline=None)  # deadline=None for slow DB ops
    def test_all_tokens_reach_terminal_state(self, rows: list[dict[str, Any]]) -> None:
        """Property: Every token reaches exactly one terminal state.

        This is THE foundational property of ELSPETH's audit trail.
        A token without a terminal outcome means we lost track of data.
        """
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # THE INVARIANT: No tokens should be missing terminal outcomes
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0, (
            f"AUDIT INTEGRITY VIOLATION: {missing} tokens missing terminal outcome. "
            f"Rows processed: {len(rows)}. "
            f"This means data was lost without being recorded."
        )

        # Also verify no duplicates
        duplicates = count_duplicate_terminal_outcomes(db, run.run_id)
        assert duplicates == 0, (
            f"AUDIT INTEGRITY VIOLATION: {duplicates} tokens have multiple terminal outcomes. "
            f"Each token should reach exactly ONE terminal state."
        )

    @given(rows=st.lists(row_with_possible_error, min_size=1, max_size=30))
    @settings(max_examples=100, deadline=None)
    def test_error_rows_still_reach_terminal_state(self, rows: list[dict[str, Any]]) -> None:
        """Property: Even rows that error reach a terminal state (QUARANTINED).

        Transform errors don't cause tokens to vanish - they're routed to
        quarantine and recorded with the QUARANTINED outcome.
        """
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = ConditionalErrorTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Count expected outcomes
        expected_errors = sum(1 for r in rows if r.get("fail"))
        expected_success = len(rows) - expected_errors

        # Verify we got the right number of results
        assert len(sink.results) == expected_success, f"Expected {expected_success} successful rows, got {len(sink.results)}"

        # THE INVARIANT: ALL tokens (success AND error) reach terminal state
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0, (
            f"AUDIT INTEGRITY VIOLATION: {missing} tokens missing terminal outcome. "
            f"Total rows: {len(rows)}, Expected errors: {expected_errors}, "
            f"Expected success: {expected_success}. "
            f"Error rows must reach QUARANTINED state, not vanish."
        )

    @given(rows=st.lists(single_row, min_size=0, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_terminal_outcomes_have_correct_type(self, rows: list[dict[str, Any]]) -> None:
        """Property: All terminal outcomes are valid RowOutcome enum values."""
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Get all outcomes and verify they're valid enum values
        outcomes = get_all_token_outcomes(db, run.run_id)
        valid_outcomes = {o.value for o in RowOutcome}

        for token_id, outcome, is_terminal in outcomes:
            assert outcome in valid_outcomes, f"Invalid outcome '{outcome}' for token {token_id}. Valid outcomes: {valid_outcomes}"

            # Verify is_terminal flag matches the outcome
            expected_terminal = RowOutcome(outcome).is_terminal
            assert is_terminal == expected_terminal, (
                f"is_terminal mismatch for token {token_id}: "
                f"outcome={outcome}, is_terminal={is_terminal}, "
                f"expected is_terminal={expected_terminal}"
            )


class TestTerminalStateEdgeCases:
    """Property tests for edge cases in terminal state handling."""

    def test_empty_source_no_orphan_tokens(self) -> None:
        """Edge case: Empty source should not create any orphan tokens."""
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        source = ListSource([])  # Empty
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # No rows means no tokens
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0

        # Verify sink is empty
        assert len(sink.results) == 0

    @given(n=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_single_field_rows(self, n: int) -> None:
        """Property: Even minimal rows (single field) reach terminal state."""
        rows = [{"id": i} for i in range(n)]

        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert len(sink.results) == n
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0, f"{missing} tokens missing terminal outcome for {n} rows"

    @given(rows=st.lists(single_row, min_size=1, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_no_transform_pipeline(self, rows: list[dict[str, Any]]) -> None:
        """Property: Pipeline with no transforms still records terminal states."""
        db = LandscapeDB.in_memory()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        sink = CollectSink()

        # No transforms - source direct to sink
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],  # Empty!
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert len(sink.results) == len(rows)
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0


class TestRowOutcomeEnumProperties:
    """Property tests for the RowOutcome enum itself."""

    def test_all_outcomes_have_is_terminal_defined(self) -> None:
        """Property: Every RowOutcome has is_terminal property defined."""
        for outcome in RowOutcome:
            # Should not raise
            _ = outcome.is_terminal

    def test_only_buffered_is_non_terminal(self) -> None:
        """Property: BUFFERED is the only non-terminal outcome."""
        non_terminal = [o for o in RowOutcome if not o.is_terminal]
        assert non_terminal == [RowOutcome.BUFFERED], f"Expected only BUFFERED to be non-terminal, but found: {non_terminal}"

    def test_terminal_outcomes_count(self) -> None:
        """Property: There are exactly 8 terminal outcomes."""
        terminal = [o for o in RowOutcome if o.is_terminal]
        expected = [
            RowOutcome.COMPLETED,
            RowOutcome.ROUTED,
            RowOutcome.FORKED,
            RowOutcome.FAILED,
            RowOutcome.QUARANTINED,
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.COALESCED,
            RowOutcome.EXPANDED,
        ]
        assert set(terminal) == set(expected), f"Terminal outcomes mismatch. Got: {terminal}, Expected: {expected}"
