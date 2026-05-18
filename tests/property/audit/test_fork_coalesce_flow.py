# tests/property/audit/test_fork_coalesce_flow.py
"""Property-based tests for the complete fork->coalesce->continue flow.

FORK-COALESCE INVARIANTS:
1. Every forked row produces exactly 1 FORKED outcome (parent token)
2. Each branch creates a child token that reaches the coalesce point
3. When all branches arrive, children get COALESCED outcomes
4. The terminal merged token reaches the sink with a COALESCED outcome
5. Token accounting: no tokens lost, no tokens duplicated

This tests the FULL fork->coalesce->continue path, not just fork-to-sinks.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import text

from elspeth.contracts import ArtifactDescriptor, Determinism, SourceRow, TerminalOutcome, TerminalPath
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.core.config import CoalesceSettings, ElspethSettings, GateSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import (
    _TestSchema,
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.stores import MockPayloadStore

if TYPE_CHECKING:
    pass


# =============================================================================
# Audit Verification Helpers
# =============================================================================


def get_outcome_counts(db: LandscapeDB, run_id: str) -> dict[tuple[str | None, str], int]:
    """Get counts of each outcome/path pair for a run."""
    with db.connection() as conn:
        results = conn.execute(
            text("""
                SELECT o.outcome, o.path, COUNT(*) as cnt
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                GROUP BY o.outcome, o.path
            """),
            {"run_id": run_id},
        ).fetchall()
        return {(row[0], row[1]): row[2] for row in results}


def get_fork_coalesce_stats(db: LandscapeDB, run_id: str) -> dict[str, Any]:
    """Get detailed fork/coalesce statistics for verification."""
    with db.connection() as conn:
        # Count tokens by outcome
        outcome_counts = get_outcome_counts(db, run_id)

        # Count unique fork groups
        fork_groups = (
            conn.execute(
                text("""
                SELECT COUNT(DISTINCT fork_group_id)
                FROM token_outcomes
                WHERE run_id = :run_id
                  AND outcome = 'transient'
                  AND path = 'fork_parent'
            """),
                {"run_id": run_id},
            ).scalar()
            or 0
        )

        # Count coalesced tokens
        coalesced_count = outcome_counts.get((TerminalOutcome.SUCCESS.value, TerminalPath.COALESCED.value), 0)

        # Count completed tokens (final output)
        completed_count = outcome_counts.get((TerminalOutcome.SUCCESS.value, TerminalPath.DEFAULT_FLOW.value), 0)

        # Count forked tokens (parent tokens that were split)
        forked_count = outcome_counts.get((TerminalOutcome.TRANSIENT.value, TerminalPath.FORK_PARENT.value), 0)

        # Verify coalesced tokens have parent links
        coalesced_without_parents = (
            conn.execute(
                text("""
                SELECT COUNT(*)
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                LEFT JOIN token_parents p ON p.token_id = t.token_id
                WHERE r.run_id = :run_id
                  AND o.outcome = 'success'
                  AND o.path = 'coalesced'
                  AND p.token_id IS NULL
            """),
                {"run_id": run_id},
            ).scalar()
            or 0
        )

        return {
            "outcome_counts": outcome_counts,
            "fork_groups": fork_groups,
            "forked_count": forked_count,
            "coalesced_count": coalesced_count,
            "completed_count": completed_count,
            "coalesced_without_parents": coalesced_without_parents,
        }


def count_tokens_missing_terminal(db: LandscapeDB, run_id: str) -> int:
    """Count tokens that lack a terminal outcome."""
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


# =============================================================================
# Test Fixtures
# =============================================================================


class _CoalesceTestSchema(_TestSchema):
    """Schema for coalesce tests."""

    value: int


class _ListSource(_TestSourceBase):
    """Source that emits rows from a list."""

    name = "coalesce_test_source"
    output_schema = _CoalesceTestSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        super().__init__()
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        yield from self.wrap_rows(self._data)

    def close(self) -> None:
        pass


class _EnrichTransform(BaseTransform):
    """Transform that adds an 'enriched' field."""

    name = "enrich_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema = _CoalesceTestSchema
    output_schema = _CoalesceTestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(make_pipeline_row({**row, "enriched": True}), success_reason={"action": "enrich"})


class _CollectSink(_TestSinkBase):
    """Sink that collects written rows."""

    name = "coalesce_test_sink"

    def __init__(self, sink_name: str = "default") -> None:
        super().__init__()
        self.name = sink_name
        self.results: list[dict[str, Any]] = []

    def on_start(self, ctx: Any) -> None:
        self.results = []

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        self.results.extend(rows)
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=f"memory://{self.name}",
                size_bytes=len(str(rows)),
                content_hash="test_hash",
            )
        )

    def close(self) -> None:
        pass


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for row values
row_for_coalesce = st.fixed_dictionaries(
    {"value": st.integers(min_value=0, max_value=1000)},
)


# =============================================================================
# Property Tests: Fork-Coalesce Flow
# =============================================================================


class TestForkCoalesceFlow:
    """Property tests for the complete fork->coalesce->continue flow."""

    @given(n_rows=st.integers(min_value=1, max_value=15))
    @settings(max_examples=30, deadline=None)
    def test_fork_coalesce_token_accounting(self, n_rows: int) -> None:
        """Property: Token accounting is correct after fork->coalesce.

        For a 2-branch fork with coalesce:
        - N source rows enter
        - N parent tokens get FORKED
        - 2*N child tokens created (one per branch per row)
        - 2*N child tokens get COALESCED
        - N terminal merged tokens get COALESCED

        Total terminal outcomes: N FORKED + 3*N COALESCED = 4*N
        """
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        rows = [{"value": i} for i in range(n_rows)]
        source = _ListSource(rows)
        source.on_success = "default"
        transform = _EnrichTransform()
        sink = _CollectSink()

        # Gate that forks ALL rows to two paths
        gate = GateSettings(
            name="fork_gate",
            input="to_gate",
            condition="True",  # Always fork
            routes={"true": "fork", "false": "default"},
            fork_to=["path_a", "path_b"],
        )

        # Coalesce that merges both paths
        coalesce = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            on_success="default",
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
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=wire_transforms([as_transform(transform)], source_connection="source_out", final_sink="to_gate"),
            sinks={"default": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "default", "options": {}},
            sinks={"default": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        # Get statistics
        stats = get_fork_coalesce_stats(db, run.run_id)

        # Verify token accounting
        # For 2-branch terminal coalesce: each row produces 1 FORKED + 3 COALESCED.
        expected_forked = n_rows
        expected_coalesced = n_rows * 3  # 2 consumed branches + terminal merged token
        expected_completed = 0

        assert stats["forked_count"] == expected_forked, (
            f"Expected {expected_forked} FORKED outcomes, got {stats['forked_count']}. "
            f"Each source row should produce exactly one FORKED parent token."
        )
        assert stats["fork_groups"] == n_rows, f"Expected {n_rows} fork groups (one per source row), got {stats['fork_groups']}."

        assert stats["coalesced_count"] == expected_coalesced, (
            f"Expected {expected_coalesced} COALESCED outcomes, got {stats['coalesced_count']}. "
            f"Each fork branch plus the terminal merged token should produce COALESCED outcomes."
        )

        assert stats["completed_count"] == expected_completed, (
            f"Expected {expected_completed} DEFAULT_FLOW outcomes, got {stats['completed_count']}. "
            f"Terminal coalesce records the merged sink write as COALESCED."
        )

        # Verify no tokens lost
        missing = count_tokens_missing_terminal(db, run.run_id)
        assert missing == 0, f"TOKEN LEAK: {missing} tokens have no terminal outcome. Every token must reach a terminal state."

        # Verify sink received correct number of results
        assert len(sink.results) == n_rows, (
            f"Expected {n_rows} results in sink, got {len(sink.results)}. Each row should produce exactly one output after fork->coalesce."
        )

    @given(n_rows=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_coalesced_tokens_have_parent_links(self, n_rows: int) -> None:
        """Property: All COALESCED tokens have parent links recorded.

        This is critical for lineage tracking - we need to know which
        parent token(s) contributed to each coalesced result.
        """
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        rows = [{"value": i} for i in range(n_rows)]
        source = _ListSource(rows)
        source.on_success = "default"
        transform = _EnrichTransform()
        sink = _CollectSink()

        gate = GateSettings(
            name="fork_gate",
            input="to_gate",
            condition="True",
            routes={"true": "fork", "false": "default"},
            fork_to=["path_a", "path_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            on_success="default",
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
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=wire_transforms([as_transform(transform)], source_connection="source_out", final_sink="to_gate"),
            sinks={"default": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "default", "options": {}},
            sinks={"default": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        # Get statistics
        stats = get_fork_coalesce_stats(db, run.run_id)

        # Verify all coalesced tokens have parent links
        assert stats["coalesced_without_parents"] == 0, (
            f"LINEAGE VIOLATION: {stats['coalesced_without_parents']} COALESCED tokens "
            f"have no parent links. Cannot trace lineage without parent relationships."
        )

    @given(n_rows=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_merged_data_contains_enrichments(self, n_rows: int) -> None:
        """Property: Merged token data contains fields added before fork.

        Transforms that run before the fork should contribute their fields
        to the merged result after coalesce.
        """
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        rows = [{"value": i} for i in range(n_rows)]
        source = _ListSource(rows)
        source.on_success = "default"
        transform = _EnrichTransform()  # Adds "enriched": True
        sink = _CollectSink()

        gate = GateSettings(
            name="fork_gate",
            input="to_gate",
            condition="True",
            routes={"true": "fork", "false": "default"},
            fork_to=["path_a", "path_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            on_success="default",
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
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=wire_transforms([as_transform(transform)], source_connection="source_out", final_sink="to_gate"),
            sinks={"default": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "default", "options": {}},
            sinks={"default": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        # Verify all results have the enriched field
        assert len(sink.results) == n_rows

        for i, result in enumerate(sink.results):
            assert "enriched" in result, (
                f"Result {i} missing 'enriched' field. Transform enrichment was lost during fork->coalesce. Got: {result}"
            )
            assert result["enriched"] is True, f"Result {i} has wrong 'enriched' value: {result['enriched']}"


class TestForkCoalesceEdgeCases:
    """Edge case tests for fork-coalesce behavior."""

    def test_empty_source_with_coalesce_config(self) -> None:
        """Empty source with coalesce config should not cause issues."""
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = _ListSource([])  # Empty
        source.on_success = "default"
        transform = _EnrichTransform()
        sink = _CollectSink()

        gate = GateSettings(
            name="fork_gate",
            input="to_gate",
            condition="True",
            routes={"true": "fork", "false": "default"},
            fork_to=["path_a", "path_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            on_success="default",
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
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=wire_transforms([as_transform(transform)], source_connection="source_out", final_sink="to_gate"),
            sinks={"default": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "default", "options": {}},
            sinks={"default": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        # No rows means no tokens
        stats = get_fork_coalesce_stats(db, run.run_id)
        assert stats["forked_count"] == 0
        assert stats["coalesced_count"] == 0
        assert stats["completed_count"] == 0
        assert len(sink.results) == 0

    def test_single_row_fork_coalesce(self) -> None:
        """Single row through fork->coalesce should work correctly."""
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = _ListSource([{"value": 42}])
        source.on_success = "default"
        transform = _EnrichTransform()
        sink = _CollectSink()

        gate = GateSettings(
            name="fork_gate",
            input="to_gate",
            condition="True",
            routes={"true": "fork", "false": "default"},
            fork_to=["path_a", "path_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            on_success="default",
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
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=wire_transforms([as_transform(transform)], source_connection="source_out", final_sink="to_gate"),
            sinks={"default": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "default", "options": {}},
            sinks={"default": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        stats = get_fork_coalesce_stats(db, run.run_id)

        # Single row: 1 FORKED, 3 COALESCED (2 consumed branches + terminal merged token)
        assert stats["forked_count"] == 1
        assert stats["coalesced_count"] == 3
        assert stats["completed_count"] == 0
        assert len(sink.results) == 1
        assert sink.results[0]["value"] == 42
        assert sink.results[0]["enriched"] is True
