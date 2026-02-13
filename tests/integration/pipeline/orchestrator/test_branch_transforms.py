# tests/integration/pipeline/orchestrator/test_branch_transforms.py
"""Integration tests for ARCH-15: per-branch transforms between fork and coalesce.

These tests verify end-to-end data flow through the complete pipeline
assembly and execution path, exercising the real production code path
(BUG-LINEAGE-01 prevention).

Pipeline topology tested:
    source → gate(fork) → [branch transforms] → coalesce(merge) → sink

Each test builds the graph via ExecutionGraph.from_plugin_instances() and
runs through the Orchestrator — no manual graph construction.
"""

from __future__ import annotations

from typing import Any, cast

from elspeth.contracts import PipelineRow, RunStatus
from elspeth.core.config import (
    CoalesceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
    SourceSettings,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import WiredTransform
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.protocols import SinkProtocol, SourceProtocol
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import CollectSink, FailTransform, ListSource

# ---------------------------------------------------------------------------
# Test transforms that produce distinguishable per-branch output
# ---------------------------------------------------------------------------


class EnrichATransform(BaseTransform):
    """Adds 'enriched_a' field — identifies branch A processing."""

    name = "enrich_a"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> Any:
        from elspeth.plugins.results import TransformResult

        data = row.to_dict()
        data["enriched_a"] = data["value"] * 10
        return TransformResult.success(
            make_pipeline_row(data),
            success_reason={"action": "enrich_a"},
        )


class EnrichBTransform(BaseTransform):
    """Adds 'enriched_b' field — identifies branch B processing."""

    name = "enrich_b"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> Any:
        from elspeth.plugins.results import TransformResult

        data = row.to_dict()
        data["enriched_b"] = data["value"] + 100
        return TransformResult.success(
            make_pipeline_row(data),
            success_reason={"action": "enrich_b"},
        )


# ---------------------------------------------------------------------------
# Pipeline construction helper
# ---------------------------------------------------------------------------


def _build_branch_pipeline(
    source_data: list[dict[str, Any]],
    branch_transforms: dict[str, list[Any]],
    coalesce: CoalesceSettings,
    gate: GateSettings,
    sinks: dict[str, CollectSink],
) -> tuple[PipelineConfig, ExecutionGraph, ElspethSettings]:
    """Build a fork → branch-transforms → coalesce pipeline.

    Uses ExecutionGraph.from_plugin_instances() — the real production assembly
    path. This prevents BUG-LINEAGE-01 from hiding in test infrastructure.

    Args:
        source_data: Rows to emit from source.
        branch_transforms: Dict mapping branch_name → list of transform plugins.
            Empty list = no transforms on that branch (identity mapping).
        coalesce: Coalesce configuration with dict branches.
        gate: Gate configuration with fork_to.
        sinks: Dict of sink_name → CollectSink.

    Returns:
        (PipelineConfig, ExecutionGraph, ElspethSettings) ready for orchestrator.run().
    """
    source = ListSource(source_data, on_success="gate_in")
    source_settings = SourceSettings(plugin="list_source", on_success="gate_in", options={})

    all_wired: list[WiredTransform] = []
    all_plugins: list[Any] = []

    for branch_name, transforms in branch_transforms.items():
        if not transforms:
            continue
        final_connection = coalesce.branches[branch_name]
        branch_wired = wire_transforms(
            transforms,
            source_connection=branch_name,
            final_sink=final_connection,
            names=[f"{branch_name}_{i}" for i in range(len(transforms))],
        )
        all_wired.extend(branch_wired)
        all_plugins.extend(transforms)

    graph = ExecutionGraph.from_plugin_instances(
        source=cast("SourceProtocol", source),
        source_settings=source_settings,
        transforms=all_wired,
        sinks=cast("dict[str, SinkProtocol]", sinks),
        aggregations={},
        gates=[gate],
        coalesce_settings=[coalesce],
    )

    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(t) for t in all_plugins],
        sinks={name: as_sink(s) for name, s in sinks.items()},
        gates=[gate],
        coalesce_settings=[coalesce],
    )

    settings = ElspethSettings(
        source=SourceSettings(plugin="list_source", on_success="gate_in", options={}),
        sinks={name: SinkSettings(plugin="collect", options={}) for name in sinks},
        gates=[gate],
        coalesce=[coalesce],
    )

    return config, graph, settings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBranchTransforms:
    """End-to-end tests for per-branch transforms (ARCH-15)."""

    def test_fork_per_branch_transforms_nested_merge(self, payload_store) -> None:
        """Fork → different transforms on each branch → nested merge.

        Nested merge produces {branch_name: {row_data}} for each branch.
        Each branch's transform adds a distinct field, verifying that data
        flows through the correct branch transform before coalescing.
        """
        db = LandscapeDB.in_memory()

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge_results",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="nested",
            on_success="output",
        )
        output_sink = CollectSink("output")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 1}, {"value": 2}],
            branch_transforms={
                "path_a": [EnrichATransform()],
                "path_b": [EnrichBTransform()],
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=graph,
            settings=settings,
            payload_store=payload_store,
        )

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2
        assert len(output_sink.results) == 2

        # Nested merge: each result keyed by branch name
        for result in output_sink.results:
            assert "path_a" in result, f"Missing path_a in nested merge: {result}"
            assert "path_b" in result, f"Missing path_b in nested merge: {result}"
            assert "enriched_a" in result["path_a"], "Branch A transform not applied"
            assert "enriched_b" in result["path_b"], "Branch B transform not applied"

        # Verify specific values (row with value=1)
        first = output_sink.results[0]
        assert first["path_a"]["enriched_a"] == 10  # 1 * 10
        assert first["path_b"]["enriched_b"] == 101  # 1 + 100

        # Verify second row (value=2)
        second = output_sink.results[1]
        assert second["path_a"]["enriched_a"] == 20  # 2 * 10
        assert second["path_b"]["enriched_b"] == 102  # 2 + 100

    def test_fork_per_branch_transforms_union_merge(self, payload_store) -> None:
        """Fork → different transforms → union merge combines all fields flat.

        Union merge produces a single dict with fields from all branches.
        Overlapping fields (like 'value') come from one branch; unique fields
        ('enriched_a', 'enriched_b') are combined.
        """
        db = LandscapeDB.in_memory()

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge_results",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="union",
            on_success="output",
        )
        output_sink = CollectSink("output")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 1}],
            branch_transforms={
                "path_a": [EnrichATransform()],
                "path_b": [EnrichBTransform()],
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=graph,
            settings=settings,
            payload_store=payload_store,
        )

        assert run_result.status == RunStatus.COMPLETED
        assert len(output_sink.results) == 1

        # Union merge: all fields merged flat
        result = output_sink.results[0]
        assert "enriched_a" in result, f"Missing enriched_a in union merge: {result}"
        assert "enriched_b" in result, f"Missing enriched_b in union merge: {result}"
        assert result["enriched_a"] == 10  # 1 * 10
        assert result["enriched_b"] == 101  # 1 + 100

    def test_fork_mixed_branches(self, payload_store) -> None:
        """One branch with transform, one direct → both arrive at coalesce.

        Mixed mode: path_a has a transform (EnrichA), path_b is identity-mapped
        (direct COPY edge to coalesce). Verifies the builder correctly handles
        both COPY and connection-system edges on the same coalesce node.
        """
        db = LandscapeDB.in_memory()

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        # path_a has transform (done_a != path_a), path_b is identity (path_b == path_b)
        coalesce = CoalesceSettings(
            name="merge_results",
            branches={"path_a": "done_a", "path_b": "path_b"},
            policy="require_all",
            merge="nested",
            on_success="output",
        )
        output_sink = CollectSink("output")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 42}],
            branch_transforms={
                "path_a": [EnrichATransform()],
                "path_b": [],  # No transforms — direct to coalesce
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=graph,
            settings=settings,
            payload_store=payload_store,
        )

        assert run_result.status == RunStatus.COMPLETED
        assert len(output_sink.results) == 1

        result = output_sink.results[0]
        assert "path_a" in result
        assert "path_b" in result

        # path_a was enriched by transform
        assert result["path_a"]["enriched_a"] == 420  # 42 * 10
        # path_b passed through unchanged (no transform)
        assert result["path_b"]["value"] == 42
        assert "enriched_a" not in result["path_b"]

    def test_branch_transform_failure_best_effort(self, payload_store) -> None:
        """Transform fails on one branch; best_effort merge succeeds with remaining.

        When path_a's transform returns an error, the token is routed to the
        quarantine sink. The coalesce's best_effort policy proceeds with only
        path_b's data in the merged output.
        """
        db = LandscapeDB.in_memory()

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge_results",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="best_effort",
            merge="nested",
            on_success="output",
            timeout_seconds=30,
        )
        output_sink = CollectSink("output")
        quarantine_sink = CollectSink("quarantine")

        fail_transform = FailTransform(name="fail_on_a", on_error="quarantine")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 1}],
            branch_transforms={
                "path_a": [fail_transform],
                "path_b": [EnrichBTransform()],
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink, "quarantine": quarantine_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=graph,
            settings=settings,
            payload_store=payload_store,
        )

        assert run_result.status == RunStatus.COMPLETED

        # best_effort merge: only path_b data in output
        assert len(output_sink.results) == 1
        result = output_sink.results[0]
        assert "path_b" in result
        assert result["path_b"]["enriched_b"] == 101

    def test_branch_transform_error_routing(self, payload_store) -> None:
        """Transform on_error routes failed rows to quarantine sink.

        Verifies that when a branch transform fails, the error row is
        delivered to the configured on_error sink (quarantine), and the
        error data is recorded.
        """
        db = LandscapeDB.in_memory()

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge_results",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="best_effort",
            merge="nested",
            on_success="output",
            timeout_seconds=30,
        )
        output_sink = CollectSink("output")
        quarantine_sink = CollectSink("quarantine")

        fail_transform = FailTransform(name="fail_on_a", on_error="quarantine")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 1}, {"value": 2}],
            branch_transforms={
                "path_a": [fail_transform],
                "path_b": [EnrichBTransform()],
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink, "quarantine": quarantine_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=graph,
            settings=settings,
            payload_store=payload_store,
        )

        assert run_result.status == RunStatus.COMPLETED

        # Each source row forks to both branches; path_a always fails.
        # 2 source rows x 1 fail per row = 2 quarantine entries
        assert len(quarantine_sink.results) == 2, f"Expected 2 quarantine entries (one per source row), got {len(quarantine_sink.results)}"

        # Output should have 2 results (best_effort merges with path_b only)
        assert len(output_sink.results) == 2
        for result in output_sink.results:
            assert "path_b" in result
