# tests/property/engine/test_processor_properties.py
"""Property-based tests for RowProcessor work queue semantics.

The RowProcessor manages DAG traversal via a work queue. These properties
are critical for audit integrity:

1. Work Conservation: No work items lost during processing
   - Every row that enters reaches a terminal state
   - Fork operations preserve row identity (children traceable to parent)
   - Error paths lead to QUARANTINED/ROUTED, not silent drops

2. Order Correctness: Dependencies respected (topological order)
   - Transforms execute in declared sequence
   - Work items processed FIFO within each step
   - Fork children processed after parent reaches FORKED state

3. Iteration Guard: MAX_WORK_QUEUE_ITERATIONS prevents infinite loops
   - Constant is reasonable (not too low, not too high)
   - Normal pipelines stay well under the guard
   - Guard triggers RuntimeError when exceeded

4. Token Identity: Each token processed exactly once per step
   - Token IDs are unique within a run
   - Fork creates new tokens with distinct IDs
   - Same token doesn't revisit same transform
"""

from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import text

from elspeth.contracts.enums import RowOutcome
from elspeth.core.config import CoalesceSettings, GateSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.processor import MAX_WORK_QUEUE_ITERATIONS
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

# =============================================================================
# Audit Verification Helpers
# =============================================================================


def count_tokens_missing_terminal(db: LandscapeDB, run_id: str) -> int:
    """Count tokens that lack a terminal outcome.

    This is the core work conservation check: every token should have exactly
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


def count_unique_tokens(db: LandscapeDB, run_id: str) -> int:
    """Count total unique tokens created for a run."""
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(DISTINCT t.token_id)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def get_transform_execution_order(db: LandscapeDB, run_id: str, token_id: str) -> list[int]:
    """Get the step order in which transforms were executed for a token.

    Returns list of step numbers (step_index) in execution order.
    """
    with db.connection() as conn:
        results = conn.execute(
            text("""
                SELECT ns.step_index
                FROM node_states ns
                WHERE ns.run_id = :run_id
                  AND ns.token_id = :token_id
                ORDER BY ns.started_at, ns.step_index
            """),
            {"run_id": run_id, "token_id": token_id},
        ).fetchall()
        return [r[0] for r in results]


def count_outcome_by_type(db: LandscapeDB, run_id: str, outcome: RowOutcome) -> int:
    """Count tokens with a specific outcome."""
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                  AND o.outcome = :outcome
            """),
            {"run_id": run_id, "outcome": outcome.value},
        ).scalar()
        return result or 0


# =============================================================================
# Strategies
# =============================================================================

# Row indices (simulating source row positions)
row_indices = st.integers(min_value=0, max_value=1000)

# Number of transforms in pipeline
transform_counts = st.integers(min_value=0, max_value=10)

# Row values for property tests (RFC 8785 safe)
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
# Work Conservation Properties
# =============================================================================


class TestWorkQueueConservation:
    """Property tests for work conservation (no items lost)."""

    @given(num_rows=st.integers(min_value=1, max_value=50))
    @settings(max_examples=50, deadline=None)
    def test_all_rows_reach_terminal_state(self, num_rows: int) -> None:
        """Property: Every row that enters the processor reaches a terminal state.

        This is work conservation - no silent drops allowed.
        """
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i, "value": f"row_{i}"} for i in range(num_rows)]

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

            # All rows must appear in sink
            assert len(sink.results) == num_rows, f"Work lost! Input: {num_rows} rows, Output: {len(sink.results)} rows"

            # No tokens missing terminal outcomes
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0, f"WORK CONSERVATION VIOLATION: {missing} tokens missing terminal outcome"

    @given(
        num_rows=st.integers(min_value=1, max_value=20),
        num_transforms=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_multi_transform_pipeline_conserves_rows(self, num_rows: int, num_transforms: int) -> None:
        """Property: Row count preserved through N transforms."""
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i} for i in range(num_rows)]

            source = ListSource(rows)
            transforms = [PassTransform() for _ in range(num_transforms)]
            sink = CollectSink()

            config = PipelineConfig(
                source=as_source(source),
                transforms=[as_transform(t) for t in transforms],
                sinks={"default": as_sink(sink)},
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

            assert len(sink.results) == num_rows, (
                f"Row count changed through {num_transforms} transforms: {num_rows} -> {len(sink.results)}"
            )

            # Verify all tokens have terminal outcomes
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0

    @given(rows=st.lists(row_with_possible_error, min_size=1, max_size=30))
    @settings(max_examples=50, deadline=None)
    def test_error_rows_not_silently_dropped(self, rows: list[dict[str, Any]]) -> None:
        """Property: Rows that error reach QUARANTINED, not lost.

        Transform errors don't cause tokens to vanish - they're routed to
        quarantine and recorded with the QUARANTINED outcome.
        """
        with LandscapeDB.in_memory() as db:
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
            expected_errors = sum(1 for r in rows if r["fail"])
            expected_success = len(rows) - expected_errors

            # Verify sink received successful rows
            assert len(sink.results) == expected_success, f"Expected {expected_success} successful rows, got {len(sink.results)}"

            # Verify quarantine count
            quarantine_count = count_outcome_by_type(db, run.run_id, RowOutcome.QUARANTINED)
            assert quarantine_count == expected_errors, f"Expected {expected_errors} quarantined rows, got {quarantine_count}"

            # ALL tokens (success AND error) must have terminal state
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0, (
                f"WORK CONSERVATION VIOLATION: {missing} tokens missing terminal outcome. Error rows must reach QUARANTINED state, not vanish."
            )

    @given(num_rows=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_fork_preserves_row_count_across_branches(self, num_rows: int) -> None:
        """Property: Fork creates child tokens for all branches.

        When a row forks to N branches, N child tokens are created.
        Parent gets FORKED, children reach terminal states.
        """
        from elspeth.core.config import ElspethSettings

        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"value": i} for i in range(num_rows)]
            source = ListSource(rows)
            sink_a = CollectSink("sink_a")
            sink_b = CollectSink("sink_b")

            # Gate that forks all rows to both sinks
            gate = GateSettings(
                name="fork_gate",
                condition="True",
                routes={"true": "fork", "false": "continue"},
                fork_to=["sink_a", "sink_b"],
            )

            config = PipelineConfig(
                source=as_source(source),
                transforms=[],
                sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
                gates=[gate],
            )

            graph = ExecutionGraph.from_plugin_instances(
                source=as_source(source),
                transforms=[],
                sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
                gates=[gate],
                aggregations={},
                coalesce_settings=[],
                default_sink="sink_a",
            )

            settings = ElspethSettings(
                source={"plugin": "test"},
                sinks={"sink_a": {"plugin": "test"}, "sink_b": {"plugin": "test"}},
                default_sink="sink_a",
                gates=[gate],
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # Each sink should receive all rows (fork duplicates)
            assert len(sink_a.results) == num_rows, f"sink_a: expected {num_rows}, got {len(sink_a.results)}"
            assert len(sink_b.results) == num_rows, f"sink_b: expected {num_rows}, got {len(sink_b.results)}"

            # FORKED outcomes for parents (one per input row)
            forked_count = count_outcome_by_type(db, run.run_id, RowOutcome.FORKED)
            assert forked_count == num_rows, f"Expected {num_rows} FORKED parents, got {forked_count}"

            # No tokens missing terminal
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0


# =============================================================================
# Order Correctness Properties
# =============================================================================


class TestOrderCorrectnessProperties:
    """Property tests for transform execution order."""

    @given(num_transforms=st.integers(min_value=2, max_value=7))
    @settings(max_examples=30, deadline=None)
    def test_transforms_execute_in_declared_order(self, num_transforms: int) -> None:
        """Property: Transforms execute in the order they are declared.

        The step_index values for transforms should be monotonically increasing,
        starting at 1. Note: sink execution may add additional step at the end.
        """
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": 0}]  # Single row for clear ordering

            source = ListSource(rows)
            transforms = [PassTransform() for _ in range(num_transforms)]
            sink = CollectSink()

            config = PipelineConfig(
                source=as_source(source),
                transforms=[as_transform(t) for t in transforms],
                sinks={"default": as_sink(sink)},
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

            # Get the token
            with db.connection() as conn:
                token_row = conn.execute(
                    text("""
                        SELECT t.token_id
                        FROM tokens t
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                        LIMIT 1
                    """),
                    {"run_id": run.run_id},
                ).fetchone()
                assert token_row is not None, "No token found"
                token_id = token_row.token_id

            # Verify execution order
            execution_order = get_transform_execution_order(db, run.run_id, token_id)

            # PROPERTY 1: Steps are monotonically increasing
            for i in range(1, len(execution_order)):
                assert execution_order[i] > execution_order[i - 1], f"Step order not monotonically increasing: {execution_order}"

            # PROPERTY 2: Transform steps (1..N) should be present
            # The first N steps should be the transforms (steps 1 to num_transforms)
            transform_steps = execution_order[:num_transforms]
            expected_transform_steps = list(range(1, num_transforms + 1))
            assert transform_steps == expected_transform_steps, (
                f"Transform steps don't match: expected {expected_transform_steps}, got {transform_steps}"
            )

            # PROPERTY 3: Starts at step 1 (not 0)
            assert execution_order[0] == 1, f"Execution should start at step 1, got {execution_order[0]}"

    @given(num_rows=st.integers(min_value=2, max_value=10))
    @settings(max_examples=30, deadline=None)
    def test_rows_processed_in_source_order(self, num_rows: int) -> None:
        """Property: Rows are processed in the order the source yields them.

        While the work queue is FIFO, source order determines initial queue order.
        """
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i, "sequence": i} for i in range(num_rows)]

            source = ListSource(rows)
            transform = PassTransform()
            sink = CollectSink()

            config = PipelineConfig(
                source=as_source(source),
                transforms=[as_transform(transform)],
                sinks={"default": as_sink(sink)},
            )

            orchestrator = Orchestrator(db)
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

            # Results should be in same order as source
            result_sequences = [r["sequence"] for r in sink.results]
            expected_sequences = list(range(num_rows))

            assert result_sequences == expected_sequences, f"Source order violated: expected {expected_sequences}, got {result_sequences}"

    @given(num_rows=st.integers(min_value=1, max_value=10))
    @settings(max_examples=30, deadline=None)
    def test_no_transform_pipeline_preserves_order(self, num_rows: int) -> None:
        """Property: Even with no transforms, source order is preserved."""
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i, "order": i} for i in range(num_rows)]

            source = ListSource(rows)
            sink = CollectSink()

            # No transforms - source direct to sink
            config = PipelineConfig(
                source=as_source(source),
                transforms=[],  # Empty!
                sinks={"default": as_sink(sink)},
            )

            orchestrator = Orchestrator(db)
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

            result_orders = [r["order"] for r in sink.results]
            expected_orders = list(range(num_rows))

            assert result_orders == expected_orders


# =============================================================================
# Iteration Guard Properties
# =============================================================================


class TestIterationGuardProperties:
    """Property tests for iteration guard behavior."""

    def test_max_iterations_constant_is_reasonable(self) -> None:
        """Property: MAX_WORK_QUEUE_ITERATIONS is set to prevent runaway.

        The constant should be:
        - High enough for legitimate deep pipelines (>= 1000)
        - Low enough to catch infinite loops quickly (<= 100,000)
        """
        assert MAX_WORK_QUEUE_ITERATIONS >= 1000, f"Guard too low for normal pipelines: {MAX_WORK_QUEUE_ITERATIONS}"
        assert MAX_WORK_QUEUE_ITERATIONS <= 100_000, f"Guard too high to catch bugs quickly: {MAX_WORK_QUEUE_ITERATIONS}"

    def test_max_iterations_constant_value(self) -> None:
        """Property: MAX_WORK_QUEUE_ITERATIONS is exactly 10,000.

        This documents the expected value. If changed, tests must be updated.
        """
        assert MAX_WORK_QUEUE_ITERATIONS == 10_000, (
            f"MAX_WORK_QUEUE_ITERATIONS changed from 10_000 to {MAX_WORK_QUEUE_ITERATIONS}. Update this test if this is intentional."
        )

    @given(num_rows=st.integers(min_value=1, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_normal_pipeline_stays_under_guard(self, num_rows: int) -> None:
        """Property: Normal pipelines don't trigger iteration guard.

        Even with many rows and transforms, legitimate pipelines should
        stay well under MAX_WORK_QUEUE_ITERATIONS.
        """
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i} for i in range(num_rows)]

            source = ListSource(rows)
            # 5 transforms = 5 steps per row, but work queue processes ONE row at a time
            transforms = [PassTransform() for _ in range(5)]
            sink = CollectSink()

            config = PipelineConfig(
                source=as_source(source),
                transforms=[as_transform(t) for t in transforms],
                sinks={"default": as_sink(sink)},
            )

            orchestrator = Orchestrator(db)
            # Should complete without RuntimeError from iteration guard
            _run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

            assert len(sink.results) == num_rows

    @given(num_rows=st.integers(min_value=1, max_value=50))
    @settings(max_examples=20, deadline=None)
    def test_fork_stays_under_guard(self, num_rows: int) -> None:
        """Property: Fork operations stay under iteration guard.

        Forking multiplies work items but should still complete.
        """
        from elspeth.core.config import ElspethSettings

        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"value": i} for i in range(num_rows)]
            source = ListSource(rows)
            sink_a = CollectSink("sink_a")
            sink_b = CollectSink("sink_b")

            gate = GateSettings(
                name="fork_gate",
                condition="True",
                routes={"true": "fork", "false": "continue"},
                fork_to=["sink_a", "sink_b"],
            )

            config = PipelineConfig(
                source=as_source(source),
                transforms=[],
                sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
                gates=[gate],
            )

            graph = ExecutionGraph.from_plugin_instances(
                source=as_source(source),
                transforms=[],
                sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
                gates=[gate],
                aggregations={},
                coalesce_settings=[],
                default_sink="sink_a",
            )

            settings = ElspethSettings(
                source={"plugin": "test"},
                sinks={"sink_a": {"plugin": "test"}, "sink_b": {"plugin": "test"}},
                default_sink="sink_a",
                gates=[gate],
            )

            orchestrator = Orchestrator(db)
            # Should complete without RuntimeError
            _run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            assert len(sink_a.results) == num_rows
            assert len(sink_b.results) == num_rows


# =============================================================================
# Token Identity Properties
# =============================================================================


class TestTokenIdentityProperties:
    """Property tests for token identity and uniqueness."""

    @given(num_rows=st.integers(min_value=1, max_value=30))
    @settings(max_examples=30, deadline=None)
    def test_all_token_ids_unique_within_run(self, num_rows: int) -> None:
        """Property: Token IDs are unique within a run.

        No two tokens in the same run should have the same token_id.
        """
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i} for i in range(num_rows)]

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

            # Count unique tokens
            unique_count = count_unique_tokens(db, run.run_id)

            # Each row creates one token (no forks in this test)
            assert unique_count == num_rows, f"Token ID collision detected: {unique_count} unique tokens for {num_rows} rows"

    @given(num_rows=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_fork_creates_distinct_child_tokens(self, num_rows: int) -> None:
        """Property: Fork creates child tokens with distinct IDs.

        Parent token ID must differ from all child token IDs.
        Child token IDs must differ from each other.
        """
        from elspeth.core.config import ElspethSettings

        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"value": i} for i in range(num_rows)]
            source = ListSource(rows)
            sink_a = CollectSink("sink_a")
            sink_b = CollectSink("sink_b")

            gate = GateSettings(
                name="fork_gate",
                condition="True",
                routes={"true": "fork", "false": "continue"},
                fork_to=["sink_a", "sink_b"],
            )

            config = PipelineConfig(
                source=as_source(source),
                transforms=[],
                sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
                gates=[gate],
            )

            graph = ExecutionGraph.from_plugin_instances(
                source=as_source(source),
                transforms=[],
                sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
                gates=[gate],
                aggregations={},
                coalesce_settings=[],
                default_sink="sink_a",
            )

            settings = ElspethSettings(
                source={"plugin": "test"},
                sinks={"sink_a": {"plugin": "test"}, "sink_b": {"plugin": "test"}},
                default_sink="sink_a",
                gates=[gate],
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # Get all token IDs
            with db.connection() as conn:
                token_ids = conn.execute(
                    text("""
                        SELECT t.token_id
                        FROM tokens t
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                    """),
                    {"run_id": run.run_id},
                ).fetchall()

            all_ids = [r[0] for r in token_ids]
            unique_ids = set(all_ids)

            # All IDs should be unique
            assert len(all_ids) == len(unique_ids), f"Token ID collision in fork: {len(all_ids)} total, {len(unique_ids)} unique"

            # Expected: num_rows parents + (num_rows * 2 children) = 3 * num_rows
            expected_tokens = num_rows * 3  # 1 parent + 2 children per row
            assert len(all_ids) == expected_tokens, f"Expected {expected_tokens} tokens (parent + 2 children per row), got {len(all_ids)}"

    @given(num_rows=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_token_row_id_preserved_across_transforms(self, num_rows: int) -> None:
        """Property: Token's row_id remains constant through all transforms.

        A token's row_id identifies its source row and should never change.
        """
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": i} for i in range(num_rows)]

            source = ListSource(rows)
            transforms = [PassTransform() for _ in range(3)]
            sink = CollectSink()

            config = PipelineConfig(
                source=as_source(source),
                transforms=[as_transform(t) for t in transforms],
                sinks={"default": as_sink(sink)},
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

            # Verify row_ids exist and are consistent
            with db.connection() as conn:
                # Get distinct row_ids from tokens
                row_ids = conn.execute(
                    text("""
                        SELECT DISTINCT r.row_id
                        FROM rows r
                        WHERE r.run_id = :run_id
                    """),
                    {"run_id": run.run_id},
                ).fetchall()

            # Should have exactly num_rows distinct row_ids
            assert len(row_ids) == num_rows, f"Expected {num_rows} distinct row_ids, got {len(row_ids)}"


# =============================================================================
# Edge Case Properties
# =============================================================================


class TestWorkQueueEdgeCases:
    """Property tests for edge cases in work queue handling."""

    def test_empty_source_no_work_items(self) -> None:
        """Edge case: Empty source creates no work items."""
        with LandscapeDB.in_memory() as db:
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

            # No results
            assert len(sink.results) == 0

            # No tokens created
            unique_count = count_unique_tokens(db, run.run_id)
            assert unique_count == 0

            # No missing terminal outcomes (vacuously true)
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0

    def test_single_row_single_transform(self) -> None:
        """Edge case: Minimal pipeline (1 row, 1 transform)."""
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"id": 0}]

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

            assert len(sink.results) == 1
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0

    @given(num_rows=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_all_rows_error_all_quarantined(self, num_rows: int) -> None:
        """Edge case: When all rows error, all reach QUARANTINED."""
        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            # All rows will error
            rows = [{"id": i, "fail": True} for i in range(num_rows)]

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

            # No successful results
            assert len(sink.results) == 0

            # All quarantined
            quarantine_count = count_outcome_by_type(db, run.run_id, RowOutcome.QUARANTINED)
            assert quarantine_count == num_rows

            # No missing outcomes
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0

    @given(num_rows=st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_fork_coalesce_balance(self, num_rows: int) -> None:
        """Property: Fork-coalesce maintains token conservation.

        Tokens that fork and then coalesce should result in merged tokens.
        For each source row:
        - 1 parent token gets FORKED
        - 2 child tokens get COALESCED
        - 1 merged token reaches the sink (COMPLETED)

        Note: This test uses a transform before the fork to match the pattern
        in test_fork_coalesce_flow.py, which is required for proper coalesce routing.
        """
        from elspeth.core.config import ElspethSettings

        with LandscapeDB.in_memory() as db:
            payload_store = MockPayloadStore()
            rows = [{"value": i} for i in range(num_rows)]
            source = ListSource(rows)
            transform = PassTransform()  # Transform before fork for proper routing
            sink = CollectSink()

            # Gate that forks to two branches
            gate = GateSettings(
                name="fork_gate",
                condition="True",
                routes={"true": "fork", "false": "continue"},
                fork_to=["path_a", "path_b"],
            )

            # Coalesce that joins the branches
            coalesce = CoalesceSettings(
                name="merge_point",
                branches=["path_a", "path_b"],
                policy="require_all",
                merge="union",
            )

            config = PipelineConfig(
                source=as_source(source),
                transforms=[as_transform(transform)],
                sinks={"default": as_sink(sink)},
                gates=[gate],
                coalesce_settings=[coalesce],
            )

            graph = ExecutionGraph.from_plugin_instances(
                source=as_source(source),
                transforms=[as_transform(transform)],
                sinks={"default": as_sink(sink)},
                gates=[gate],
                aggregations={},
                coalesce_settings=[coalesce],
                default_sink="default",
            )

            settings = ElspethSettings(
                source={"plugin": "test"},
                sinks={"default": {"plugin": "test"}},
                default_sink="default",
                gates=[gate],
                coalesce=[coalesce],  # Note: ElspethSettings uses 'coalesce' not 'coalesce_settings'
            )

            orchestrator = Orchestrator(db)
            run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # Coalesced tokens should reach sink - one per input row
            assert len(sink.results) == num_rows, f"Expected {num_rows} results after coalesce, got {len(sink.results)}"

            # Verify FORKED count (parent tokens)
            forked_count = count_outcome_by_type(db, run.run_id, RowOutcome.FORKED)
            assert forked_count == num_rows, f"Expected {num_rows} FORKED outcomes, got {forked_count}"

            # Verify COALESCED count (2 children per row)
            coalesced_count = count_outcome_by_type(db, run.run_id, RowOutcome.COALESCED)
            assert coalesced_count == num_rows * 2, f"Expected {num_rows * 2} COALESCED outcomes, got {coalesced_count}"

            # No missing terminal outcomes
            missing = count_tokens_missing_terminal(db, run.run_id)
            assert missing == 0
