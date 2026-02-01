"""Integration tests for group ID consistency between tokens and token_outcomes.

Verifies fix for P2 group ID propagation bug:
- fork_group_id must match across tokens table and token_outcomes table
- join_group_id must match across tokens table and token_outcomes table
- expand_group_id must match across tokens table and token_outcomes table

These tests use PRODUCTION code paths to avoid test path divergence that
hid BUG-LINEAGE-01.

Per CLAUDE.md Test Path Integrity: Never bypass production factories in tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import text

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import ArtifactDescriptor, RowOutcome, SourceRow
from elspeth.core.config import CoalesceSettings, ElspethSettings, GateSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.conftest import _TestSchema, _TestSinkBase, _TestSourceBase, as_sink, as_source


# Test helper classes
class ListSource(_TestSourceBase):
    """Test source that yields rows from a list."""

    name = "list_source"
    output_schema = _TestSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for row in self._data:
            yield SourceRow.valid(row)

    def close(self) -> None:
        pass


class CollectSink(_TestSinkBase):
    """Test sink that collects rows into a list."""

    name = "collect_sink"

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        for row in rows:
            self.rows.append(row)
        return ArtifactDescriptor.for_file(
            path="memory://test",
            size_bytes=0,
            content_hash="test",
        )

    def close(self) -> None:
        pass


class TestForkGroupIDConsistency:
    """Verify fork_group_id matches between tokens and token_outcomes tables."""

    def test_fork_children_share_same_fork_group_id_in_tokens_table(self, payload_store) -> None:
        """Fork operation: all children tokens share same fork_group_id in tokens table."""
        # Setup: Source -> Fork (2 branches) -> 2 Sinks
        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"value": 42}])
        sink_a = CollectSink()
        sink_b = CollectSink()

        settings = ElspethSettings(
            source={"plugin": "null"},
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",  # Always fork
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                )
            ],
            sinks={
                "branch_a": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}},
                "branch_b": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}},
            },
            default_sink="branch_a",
        )

        # Build graph from settings
        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"branch_a": as_sink(sink_a), "branch_b": as_sink(sink_b)},
            gates=list(settings.gates),
        )

        orchestrator = Orchestrator(db=db)
        _run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Query tokens table for forked children
        with db.connection() as conn:
            tokens_result = conn.execute(
                text("""
                SELECT token_id, fork_group_id
                FROM tokens
                WHERE fork_group_id IS NOT NULL
                ORDER BY token_id
                """)
            ).fetchall()

        # Should have 2 forked children
        assert len(tokens_result) == 2, f"Expected 2 forked child tokens, got {len(tokens_result)}"

        # Both children should share the SAME fork_group_id
        fork_group_ids = [row[1] for row in tokens_result]
        assert fork_group_ids[0] == fork_group_ids[1], f"Fork children should share same fork_group_id, got {fork_group_ids}"

        canonical_fork_group_id = fork_group_ids[0]
        assert canonical_fork_group_id is not None, "fork_group_id must not be None"

    def test_fork_parent_outcome_uses_canonical_fork_group_id(self, payload_store) -> None:
        """Fork operation: parent token's FORKED outcome uses same fork_group_id as children."""
        # Setup: Source -> Fork (2 branches) -> 2 Sinks
        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"value": 42}])
        sink_a = CollectSink()
        sink_b = CollectSink()

        settings = ElspethSettings(
            source={"plugin": "null"},
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",  # Always fork
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                )
            ],
            sinks={
                "branch_a": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}},
                "branch_b": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}},
            },
            default_sink="branch_a",
        )

        # Build graph from settings
        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"branch_a": as_sink(sink_a), "branch_b": as_sink(sink_b)},
            gates=list(settings.gates),
        )

        orchestrator = Orchestrator(db=db)
        _run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Query tokens table for canonical fork_group_id
        with db.connection() as conn:
            canonical_row = conn.execute(
                text("""
                SELECT fork_group_id
                FROM tokens
                WHERE fork_group_id IS NOT NULL
                LIMIT 1
                """)
            ).fetchone()
            assert canonical_row is not None, "Should have at least one token with fork_group_id"
            canonical_id = canonical_row[0]

            # Query token_outcomes for parent's FORKED outcome
            outcome_result = conn.execute(
                text("""
                SELECT fork_group_id
                FROM token_outcomes
                WHERE outcome = :outcome
                """),
                {"outcome": RowOutcome.FORKED.value},
            ).fetchone()

        assert outcome_result is not None, "Parent should have FORKED outcome recorded"

        outcome_fork_group_id = outcome_result[0]

        # CRITICAL: token_outcomes.fork_group_id MUST match tokens.fork_group_id
        assert outcome_fork_group_id == canonical_id, (
            f"token_outcomes.fork_group_id ({outcome_fork_group_id}) must match tokens.fork_group_id ({canonical_id})"
        )


class TestJoinGroupIDConsistency:
    """Verify join_group_id matches between tokens and token_outcomes tables."""

    def test_coalesce_creates_merged_token_with_join_group_id(self, payload_store) -> None:
        """Coalesce operation: merged token has join_group_id in tokens table."""
        # Setup: Source -> Fork -> Coalesce -> Sink
        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"value": 42}])
        sink = CollectSink()

        settings = ElspethSettings(
            source={"plugin": "null"},
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",  # Always fork
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                )
            ],
            sinks={"output": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}}},
            coalesce=[
                CoalesceSettings(
                    name="output",
                    branches=["branch_a", "branch_b"],
                    policy="require_all",
                    merge="union",
                )
            ],
            default_sink="output",
        )

        # Build graph from settings
        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"output": as_sink(sink)},
            gates=list(settings.gates),
        )

        orchestrator = Orchestrator(db=db)
        _run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Query tokens table for merged token (created by coalesce)
        with db.connection() as conn:
            merged_token = conn.execute(
                text("""
                SELECT token_id, join_group_id
                FROM tokens
                WHERE join_group_id IS NOT NULL
                """)
            ).fetchone()

        assert merged_token is not None, "Coalesce should create merged token with join_group_id"

        _merged_token_id, canonical_join_group_id = merged_token
        assert canonical_join_group_id is not None, "join_group_id must not be None"

    def test_coalesce_consumed_tokens_use_canonical_join_group_id(self, payload_store) -> None:
        """Coalesce operation: consumed tokens' COALESCED outcomes use same join_group_id as merged token."""
        # Setup: Source -> Fork -> Coalesce -> Sink
        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"value": 42}])
        sink = CollectSink()

        settings = ElspethSettings(
            source={"plugin": "null"},
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",  # Always fork
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                )
            ],
            sinks={"output": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}}},
            coalesce=[
                CoalesceSettings(
                    name="output",
                    branches=["branch_a", "branch_b"],
                    policy="require_all",
                    merge="union",
                )
            ],
            default_sink="output",
        )

        # Build graph from settings
        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"output": as_sink(sink)},
            gates=list(settings.gates),
        )

        orchestrator = Orchestrator(db=db)
        _run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Query tokens table for canonical join_group_id
        with db.connection() as conn:
            canonical_row = conn.execute(
                text("""
                SELECT join_group_id
                FROM tokens
                WHERE join_group_id IS NOT NULL
                LIMIT 1
                """)
            ).fetchone()
            assert canonical_row is not None, "Should have at least one token with join_group_id"
            canonical_id = canonical_row[0]

            # Query token_outcomes for consumed tokens' COALESCED outcomes
            consumed_outcomes = conn.execute(
                text("""
                SELECT token_id, join_group_id
                FROM token_outcomes
                WHERE outcome = :outcome
                ORDER BY token_id
                """),
                {"outcome": RowOutcome.COALESCED.value},
            ).fetchall()

        # Should have outcomes for consumed tokens (parent branches)
        assert len(consumed_outcomes) >= 2, f"Should have COALESCED outcomes for consumed tokens, got {len(consumed_outcomes)}"

        # CRITICAL: ALL consumed token outcomes must use canonical join_group_id
        for token_id, outcome_join_group_id in consumed_outcomes:
            assert outcome_join_group_id == canonical_id, (
                f"token_outcomes.join_group_id ({outcome_join_group_id}) for token {token_id} "
                f"must match tokens.join_group_id ({canonical_id})"
            )

    def test_coalesce_merged_token_outcome_uses_canonical_join_group_id(self, payload_store) -> None:
        """Coalesce operation: merged token's COALESCED outcome uses same join_group_id."""
        # Setup: Source -> Fork -> Coalesce -> Sink
        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"value": 42}])
        sink = CollectSink()

        settings = ElspethSettings(
            source={"plugin": "null"},
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",  # Always fork
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                )
            ],
            sinks={"output": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}}},
            coalesce=[
                CoalesceSettings(
                    name="output",
                    branches=["branch_a", "branch_b"],
                    policy="require_all",
                    merge="union",
                )
            ],
            default_sink="output",
        )

        # Build graph from settings
        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"output": as_sink(sink)},
            gates=list(settings.gates),
        )

        orchestrator = Orchestrator(db=db)
        _run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        # Query tokens table for merged token
        with db.connection() as conn:
            merged_row = conn.execute(
                text("""
                SELECT token_id, join_group_id
                FROM tokens
                WHERE join_group_id IS NOT NULL
                LIMIT 1
                """)
            ).fetchone()
            assert merged_row is not None, "Should have at least one token with join_group_id"
            merged_token_id, _ = merged_row

            # Query for merged token's terminal outcome (COMPLETED when reaching sink)
            # P1 FIX: Merged tokens now get COMPLETED recorded when routed to sink.
            # The token's coalesce lineage is encoded in its join_group_id field.
            merged_outcome = conn.execute(
                text("""
                SELECT outcome, sink_name
                FROM token_outcomes
                WHERE token_id = :token_id AND is_terminal = 1
                """),
                {"token_id": merged_token_id},
            ).fetchone()

        # Merged token MUST have a terminal outcome (audit completeness requirement)
        assert merged_outcome is not None, (
            f"Merged token {merged_token_id} has NO terminal outcome! Every token must reach exactly one terminal state."
        )
        outcome, sink_name = merged_outcome
        assert outcome == RowOutcome.COMPLETED.value, f"Merged token reaching sink should have COMPLETED outcome, got {outcome}"
        assert sink_name is not None, "COMPLETED outcome must have sink_name"


class TestExpandGroupIDConsistency:
    """Verify expand_group_id matches between tokens and token_outcomes tables."""

    def test_expand_creates_consistent_group_id_across_all_children(self, payload_store) -> None:
        """Expand operation: all expanded children share same expand_group_id in tokens table."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"id": "row_1", "items": [{"value": 1}, {"value": 2}]}])
        sink = CollectSink()

        transform_config = {
            "schema": {"fields": "dynamic"},
            "array_field": "items",
            "output_field": "item",
            "include_index": False,
        }
        transform = JSONExplode(transform_config)

        settings = ElspethSettings(
            source={"plugin": "null"},
            transforms=[{"plugin": "json_explode", "options": transform_config}],
            sinks={"output": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}}},
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(db=db)
        run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        with db.connection() as conn:
            tokens_result = conn.execute(
                text("""
                SELECT t.token_id, t.expand_group_id
                FROM tokens t
                JOIN rows r ON t.row_id = r.row_id
                WHERE r.run_id = :run_id
                AND t.expand_group_id IS NOT NULL
                ORDER BY t.token_id
                """),
                {"run_id": run.run_id},
            ).fetchall()

        # Two expanded children from one row
        assert len(tokens_result) == 2, f"Expected 2 expanded child tokens, got {len(tokens_result)}"
        expand_group_ids = {row[1] for row in tokens_result}
        assert len(expand_group_ids) == 1, f"Expanded children should share same expand_group_id, got {expand_group_ids}"
        assert None not in expand_group_ids, "expand_group_id must not be None"

    def test_expand_parent_outcome_uses_canonical_expand_group_id(self, payload_store) -> None:
        """Expand operation: parent token's EXPANDED outcome uses same expand_group_id as children."""
        from elspeth.plugins.transforms.json_explode import JSONExplode

        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"id": "row_1", "items": [{"value": 1}, {"value": 2}]}])
        sink = CollectSink()

        transform_config = {
            "schema": {"fields": "dynamic"},
            "array_field": "items",
            "output_field": "item",
            "include_index": False,
        }
        transform = JSONExplode(transform_config)

        settings = ElspethSettings(
            source={"plugin": "null"},
            transforms=[{"plugin": "json_explode", "options": transform_config}],
            sinks={"output": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}}},
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(db=db)
        run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        with db.connection() as conn:
            canonical_row = conn.execute(
                text("""
                SELECT t.expand_group_id
                FROM tokens t
                JOIN rows r ON t.row_id = r.row_id
                WHERE r.run_id = :run_id
                AND t.expand_group_id IS NOT NULL
                LIMIT 1
                """),
                {"run_id": run.run_id},
            ).fetchone()
            assert canonical_row is not None, "Should have at least one token with expand_group_id"
            canonical_id = canonical_row[0]

            outcome_row = conn.execute(
                text("""
                SELECT expand_group_id
                FROM token_outcomes
                WHERE run_id = :run_id
                AND outcome = :outcome
                LIMIT 1
                """),
                {"run_id": run.run_id, "outcome": RowOutcome.EXPANDED.value},
            ).fetchone()

        assert outcome_row is not None, "Parent should have EXPANDED outcome recorded"
        assert outcome_row[0] == canonical_id, (
            f"token_outcomes.expand_group_id ({outcome_row[0]}) must match tokens.expand_group_id ({canonical_id})"
        )


class TestSequentialCoalesces:
    """Verify multiple coalesce operations each get distinct join_group_ids."""

    def test_sequential_coalesces_have_different_join_group_ids(self, payload_store) -> None:
        """Pipeline with fork->coalesce->fork->coalesce should have two DIFFERENT join_group_ids."""
        db = LandscapeDB("sqlite:///:memory:")
        _recorder = LandscapeRecorder(db)

        source = ListSource(data=[{"value": 42}])
        sink = CollectSink()

        settings = ElspethSettings(
            source={"plugin": "null"},
            sinks={"output": {"plugin": "json", "options": {"path": "/dev/null", "schema": {"fields": "dynamic"}}}},
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker1",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
                GateSettings(
                    name="forker2",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_c", "path_d"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge1",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
                CoalesceSettings(
                    name="merge2",
                    branches=["path_c", "path_d"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"output": as_sink(sink)},
            gates=list(settings.gates),
        )

        orchestrator = Orchestrator(db=db)
        run = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

        with db.connection() as conn:
            rows = conn.execute(
                text("""
                SELECT DISTINCT join_group_id
                FROM token_outcomes
                WHERE run_id = :run_id
                AND outcome = :outcome
                AND join_group_id IS NOT NULL
                """),
                {"run_id": run.run_id, "outcome": RowOutcome.COALESCED.value},
            ).fetchall()

        join_group_ids = {row[0] for row in rows}
        assert len(join_group_ids) == 2, f"Expected 2 distinct join_group_ids, got {join_group_ids}"
