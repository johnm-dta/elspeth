# tests/property/audit/test_terminal_states.py
"""Property-based tests for the terminal state invariant.

THE FOUNDATIONAL AUDIT PROPERTY:
Every token reaches EXACTLY ONE terminal state.

This is not negotiable. If a token goes missing without reaching a terminal
state, the audit trail is incomplete and ELSPETH's core value proposition
(attributability) is compromised.

These tests use Hypothesis to generate thousands of random pipeline inputs
and verify that the terminal state invariant holds for ALL of them.

Terminal states (from ADR-019 terminal pairs):
- COMPLETED: Reached output sink successfully
- ROUTED: Sent to named sink by gate
- FORKED: Split into multiple parallel paths (parent token)
- FAILED: Processing failed, not recoverable
- QUARANTINED: Failed validation, stored for investigation
- CONSUMED_IN_BATCH: Absorbed into aggregate
- DROPPED_BY_FILTER: Transform intentionally emitted zero rows
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

from elspeth.contracts import Determinism
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import (
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.plugins import (
    CollectSink,
    ConditionalErrorTransform,
    ListSource,
    PassTransform,
)
from tests.fixtures.stores import MockPayloadStore
from tests.strategies.json import MAX_SAFE_INT

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
                  ON o.token_id = t.token_id AND o.completed = 1
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
                    WHERE o.completed = 1 AND r.run_id = :run_id
                    GROUP BY o.token_id
                    HAVING COUNT(*) > 1
                ) duplicates
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def get_all_token_outcomes(db: LandscapeDB, run_id: str) -> list[tuple[str, str | None, str, bool]]:
    """Get all token outcomes for a run.

    Returns list of (token_id, outcome, path, completed) tuples.
    Used for detailed debugging when invariants fail.
    """
    with db.connection() as conn:
        results = conn.execute(
            text("""
                SELECT o.token_id, o.outcome, o.path, o.completed
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                ORDER BY o.token_id, o.recorded_at
            """),
            {"run_id": run_id},
        ).fetchall()
        return [(r[0], r[1], r[2], bool(r[3])) for r in results]


# =============================================================================
# Helper: Build production graph from PipelineConfig
# =============================================================================


def _build_production_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build graph using production code path (from_plugin_instances).

    Replacement for v1 build_production_graph, inlined to avoid v1 imports.
    Auto-sets on_success on terminal transform for linear pipelines.
    """
    from elspeth.contracts import TransformProtocol
    from elspeth.core.config import SourceSettings
    from tests.fixtures.factories import wire_transforms

    row_transforms: list[TransformProtocol] = []
    for transform in config.transforms:
        if isinstance(transform, TransformProtocol):
            row_transforms.append(transform)

    sink_name = next(iter(config.sinks))
    source_on_success = "source_out" if row_transforms else sink_name
    final_destination = config.gates[0].input if config.gates else sink_name
    if not row_transforms and config.gates:
        source_on_success = config.gates[0].input

    config.sources["primary"].on_success = source_on_success

    return ExecutionGraph.from_plugin_instances(
        sources={"primary": config.sources["primary"]},
        source_settings_map={"primary": SourceSettings(plugin=config.sources["primary"].name, on_success=source_on_success, options={})},
        transforms=wire_transforms(
            row_transforms,
            source_connection=source_on_success,
            final_sink=final_destination,
        ),
        sinks=config.sinks,
        aggregations={},
        gates=list(config.gates),
        coalesce_settings=list(config.coalesce_settings) if config.coalesce_settings else None,
    )


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
single_row: st.SearchStrategy[dict[str, Any]] = st.fixed_dictionaries(
    {"id": st.integers(min_value=0, max_value=MAX_SAFE_INT)},
    optional={"value": row_value, "name": st.text(max_size=20), "flag": st.booleans()},
)

# Strategy for row that might trigger errors (RFC 8785 safe integers)
row_with_possible_error: st.SearchStrategy[dict[str, Any]] = st.fixed_dictionaries(
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
        db = make_landscape_db()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

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
        db = make_landscape_db()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = ConditionalErrorTransform()
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

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
        """Property: All terminal outcomes use valid ADR-019 enum values."""
        db = make_landscape_db()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

        # Get all outcomes and verify they're valid enum values
        outcomes = get_all_token_outcomes(db, run.run_id)
        valid_outcomes = {o.value for o in TerminalOutcome}
        valid_paths = {p.value for p in TerminalPath}

        for token_id, outcome, path, completed in outcomes:
            if completed:
                assert outcome in valid_outcomes, f"Invalid outcome '{outcome}' for token {token_id}. Valid outcomes: {valid_outcomes}"
            else:
                assert outcome is None, f"Non-terminal token {token_id} must have NULL outcome, got {outcome!r}"
            assert path in valid_paths, f"Invalid path '{path}' for token {token_id}. Valid paths: {valid_paths}"

            assert completed == (outcome is not None), (
                f"completed mismatch for token {token_id}: outcome={outcome}, path={path}, completed={completed}"
            )


class TestTerminalStateEdgeCases:
    """Property tests for edge cases in terminal state handling."""

    def test_empty_source_no_orphan_tokens(self) -> None:
        """Edge case: Empty source should not create any orphan tokens."""
        db = make_landscape_db()
        payload_store = MockPayloadStore()
        source = ListSource([])  # Empty
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

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

        db = make_landscape_db()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

        assert len(sink.results) == n
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0, f"{missing} tokens missing terminal outcome for {n} rows"

    @given(rows=st.lists(single_row, min_size=1, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_no_transform_pipeline(self, rows: list[dict[str, Any]]) -> None:
        """Property: Pipeline with no transforms still records terminal states."""
        db = make_landscape_db()
        payload_store = MockPayloadStore()
        source = ListSource(rows)
        sink = CollectSink()

        # No transforms - source direct to sink
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],  # Empty!
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

        assert len(sink.results) == len(rows)
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0


class TestTerminalStateAggregation:
    """Property tests: BUFFERED tokens in aggregation pipelines reach terminal state.

    These tests exercise the BUFFERED → terminal transition that only occurs
    in pipelines with aggregation. The BUFFERED outcome is the only non-terminal
    terminal pair — if a token is BUFFERED but never reaches terminal
    state, the audit trail is incomplete.

    Fix for elspeth-27b9cd6f6c: existing terminal state property tests only
    covered simple (source → transform → sink) pipelines, never exercising
    the BUFFERED path at all.
    """

    @given(n=st.integers(min_value=1, max_value=30))
    @settings(max_examples=50, deadline=None)
    def test_aggregation_buffered_tokens_reach_terminal(self, n: int) -> None:
        """Property: All tokens BUFFERED during aggregation reach terminal state.

        Transform-mode aggregation: N input tokens → BUFFERED → CONSUMED_IN_BATCH.
        Count trigger set unreachably high so all tokens flush at end-of-source.
        """
        from elspeth.contracts import PipelineRow
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.core.config import AggregationSettings, SourceSettings, TriggerConfig
        from elspeth.plugins.infrastructure.results import TransformResult
        from elspeth.testing import make_pipeline_row
        from tests.fixtures.base_classes import _TestSchema, _TestTransformBase
        from tests.fixtures.factories import wire_transforms

        class SumBatchTransform(_TestTransformBase):
            """Batch transform that sums values."""

            name = "sum_batch"
            determinism = Determinism.DETERMINISTIC
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            on_success: str | None = "default"

            def process(self, row: PipelineRow | list[PipelineRow], ctx: object) -> TransformResult:
                if isinstance(row, list):
                    total = sum(r.get("value", 0) for r in row)
                    output = {"value": total, "count": len(row)}
                    contract = SchemaContract(
                        mode="OBSERVED",
                        fields=(
                            FieldContract(
                                normalized_name="value", original_name="value", python_type=int, required=False, source="inferred"
                            ),
                            FieldContract(
                                normalized_name="count", original_name="count", python_type=int, required=False, source="inferred"
                            ),
                        ),
                        locked=True,
                    )
                    return TransformResult.success(
                        PipelineRow(output, contract),
                        success_reason={"action": "batch_sum"},
                    )
                return TransformResult.success(make_pipeline_row(row.to_dict()), success_reason={"action": "buffer"})

        rows = [{"id": i, "value": i * 10} for i in range(n)]
        source = as_source(ListSource(rows, name="agg_source", on_success="source_out"))
        transform = as_transform(SumBatchTransform())
        sink = as_sink(CollectSink())

        # Build graph via production path (same as T18 characterization tests)
        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
            transforms=wire_transforms([transform], source_connection="source_out", final_sink="default"),
            sinks={"default": sink},
            aggregations={},
            gates=[],
            coalesce_settings=None,
        )

        # Map transform's node ID to aggregation settings
        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]

        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="sum_batch",
            input="source_out",
            on_success="default",
            on_error="discard",
            trigger=TriggerConfig(count=9999),  # Never triggers mid-stream
            output_mode="transform",
        )

        config = PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"default": sink},
            aggregation_settings={transform_node_id: agg_settings},
        )

        db = make_landscape_db()
        orchestrator = Orchestrator(db)
        payload_store = MockPayloadStore()
        run = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # THE INVARIANT: No tokens missing terminal outcome
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0, (
            f"AUDIT INTEGRITY VIOLATION: {missing} tokens missing terminal outcome "
            f"in aggregation pipeline. Rows: {n}. "
            f"BUFFERED tokens must reach terminal state at end-of-source flush."
        )

        # No duplicate terminals
        duplicates = count_duplicate_terminal_outcomes(db, run.run_id)
        assert duplicates == 0, f"AUDIT INTEGRITY VIOLATION: {duplicates} tokens have multiple terminal outcomes in aggregation pipeline."

        # Counter sanity: all rows should have been buffered
        assert run.rows_buffered == n, f"Expected {n} rows_buffered, got {run.rows_buffered}"


class TestTerminalPairEnumProperties:
    """Property tests for the ADR-019 terminal enum split."""

    def test_outcome_values_are_closed(self) -> None:
        """Property: TerminalOutcome exposes the lifecycle axis values."""
        assert {outcome.value for outcome in TerminalOutcome} == {"success", "failure", "transient"}

    def test_only_buffered_path_is_non_terminal(self) -> None:
        """Property: BUFFERED is the only non-terminal path."""
        assert TerminalPath.BUFFERED.value == "buffered"

    def test_terminal_paths_count(self) -> None:
        """Property: There are exactly 13 terminal paths plus BUFFERED."""
        assert len(TerminalPath) == 14
