"""ADR-019 resume aggregation parity for phase-3 predicate counters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from elspeth.contracts.run_result import RunResult
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator.run_status import derive_resume_terminal_status_from_audit
from tests.integration._helpers import (
    build_test_pipeline_with_discard_sink,
    build_test_pipeline_with_failsink_diversion,
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
    build_test_pipeline_with_source_quarantine,
    run_pipeline,
)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    result: RunResult
    db: LandscapeDB
    run_id: str


def _counter_snapshot(result: RunResult) -> dict[str, object]:
    return {
        "status": result.status,
        "rows_processed": result.rows_processed,
        "rows_succeeded": result.rows_succeeded,
        "rows_failed": result.rows_failed,
        "rows_routed_success": result.rows_routed_success,
        "rows_routed_failure": result.rows_routed_failure,
        "rows_quarantined": result.rows_quarantined,
        "rows_forked": result.rows_forked,
        "rows_coalesced": result.rows_coalesced,
        "rows_expanded": result.rows_expanded,
        "rows_buffered": result.rows_buffered,
        "rows_diverted": result.rows_diverted,
        "rows_coalesce_failed": result.rows_coalesce_failed,
        "routed_destinations": dict(result.routed_destinations),
    }


def _resume_counter_snapshot_from_audit(
    db: LandscapeDB,
    run_id: str,
) -> dict[str, object]:
    factory = RecorderFactory(db)
    status, counters = derive_resume_terminal_status_from_audit(factory, run_id)
    return {
        "status": status,
        "rows_processed": counters.rows_processed,
        "rows_succeeded": counters.rows_succeeded,
        "rows_failed": counters.rows_failed,
        "rows_routed_success": counters.rows_routed_success,
        "rows_routed_failure": counters.rows_routed_failure,
        "rows_quarantined": counters.rows_quarantined,
        "rows_forked": counters.rows_forked,
        "rows_coalesced": counters.rows_coalesced,
        "rows_expanded": counters.rows_expanded,
        "rows_buffered": counters.rows_buffered,
        "rows_diverted": counters.rows_diverted,
        "rows_coalesce_failed": counters.rows_coalesce_failed,
        "routed_destinations": dict(counters.routed_destinations),
    }


def run_live_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> ScenarioResult:
    if scenario == "gate_routed_success":
        config, graph, db, store = build_test_pipeline_with_gate_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            routed_row_count=3,
            default_flow_row_count=1,
        )
    elif scenario == "on_error_routed_failure":
        config, graph, db, store = build_test_pipeline_with_on_error_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            on_error_routed_count=2,
            success_count=1,
        )
    elif scenario == "quarantine_failure":
        config, graph, db, store = build_test_pipeline_with_source_quarantine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            quarantine_row_count=2,
        )
    elif scenario == "failsink_mode_diversion":
        config, graph, db, store = build_test_pipeline_with_failsink_diversion(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            diverted_row_count=2,
            success_row_count=1,
        )
    elif scenario == "discard_mode_diversion":
        config, graph, db, store = build_test_pipeline_with_discard_sink(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            success_row_count=1,
            discard_row_count=2,
        )
    else:
        raise AssertionError(f"unknown scenario {scenario!r}")

    result = run_pipeline(config, graph, db, store)
    return ScenarioResult(result=result, db=db, run_id=result.run_id)


@pytest.mark.parametrize(
    "scenario",
    [
        "gate_routed_success",
        "on_error_routed_failure",
        "quarantine_failure",
        "failsink_mode_diversion",
        "discard_mode_diversion",
    ],
)
def test_resume_counter_shape_matches_live_predicate_counters(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit-derived resume counters match live status and RunResult counters."""
    live = run_live_scenario(tmp_path / scenario, monkeypatch, scenario)

    assert _resume_counter_snapshot_from_audit(live.db, live.run_id) == _counter_snapshot(live.result)


@pytest.mark.parametrize("scenario", ["failsink_mode_diversion", "discard_mode_diversion"])
def test_resume_counter_derivation_replays_diversion_structural_count(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural ``rows_diverted`` is replayed from audit in the all-terminal branch."""
    live = run_live_scenario(tmp_path / scenario, monkeypatch, scenario)
    factory = RecorderFactory(live.db)

    _status, counters = derive_resume_terminal_status_from_audit(factory, live.run_id)

    assert counters.rows_diverted == live.result.rows_diverted


# ─────────────────────────────────────────────────────────────────────────
# F2 (resume-fork-reemit): rows_processed is per SOURCE ROW, not per terminal
# token.  derive_resume_terminal_status_from_audit reconstructs it as the count
# of DISTINCT source row_id reaching a terminal outcome
# (QueryRepository.count_distinct_source_rows_with_terminal_outcome), NOT a
# per-leaf tally.  Structural fan-out (fork / expand) and fan-in (aggregation)
# are the cases where the two diverge — the parametrized scenarios above are all
# 1-row-1-leaf and so cannot catch a per-leaf regression.  These archetype
# guards lock in the fix and protect BOTH resume branches (the helper is shared).
#
# The "old per-leaf value" in each docstring is what the pre-F2 logic produced
# (one rows_processed += 1 per SUCCESS/FAILURE terminal leaf); the assertions pin
# that it is no longer produced.
# ─────────────────────────────────────────────────────────────────────────


def _derive_rows_processed(db: object, run_id: str) -> int:
    factory = RecorderFactory(db)  # type: ignore[arg-type]
    _status, counters = derive_resume_terminal_status_from_audit(factory, run_id)
    return counters.rows_processed


def test_derive_rows_processed_fork_counts_source_rows_not_leaves() -> None:
    """Fork 1 source row -> 2 branches: rows_processed == 1 (source rows), not 2 (leaves).

    Pre-F2 per-leaf logic counted the two SUCCESS/DEFAULT_FLOW fork leaves and
    reported rows_processed == 2.  The fork children inherit the parent's single
    source row_id, so the faithful per-source-row count is 1.
    """
    from elspeth.core.config import GateSettings, SourceSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from tests.fixtures.base_classes import as_sink, as_source
    from tests.fixtures.landscape import make_landscape_db
    from tests.fixtures.plugins import CollectSink, ListSource
    from tests.fixtures.stores import MockPayloadStore

    db = make_landscape_db()
    src = ListSource([{"value": 1}], on_success="sink_a")
    sink_a, sink_b = CollectSink("sink_a"), CollectSink("sink_b")
    gate = GateSettings(
        name="fork_gate",
        input="gate_in",
        condition="True",
        routes={"true": "fork", "false": "sink_a"},
        fork_to=["sink_a", "sink_b"],
    )
    config = PipelineConfig(
        source=as_source(src),
        transforms=[],
        sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
        gates=[gate],
    )
    graph = ExecutionGraph.from_plugin_instances(
        source=as_source(src),
        source_settings=SourceSettings(plugin=src.name, on_success="gate_in", options={}),
        transforms=[],
        sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
        gates=[gate],
        aggregations={},
        coalesce_settings=[],
    )
    run = Orchestrator(db).run(config, graph=graph, payload_store=MockPayloadStore())

    # Live truth: 1 source row, 2 success leaves, 1 fork parent.
    assert run.rows_processed == 1
    assert run.rows_succeeded == 2
    assert run.rows_forked == 1

    derived = _derive_rows_processed(db, run.run_id)
    assert derived == 1, f"fork: derive rows_processed must be source-row count 1, got {derived}"
    assert derived == run.rows_processed, "fork: derive must match uninterrupted run rows_processed"
    assert derived != run.rows_succeeded, "fork: derive must NOT be the per-leaf count (2) — that was the pre-F2 bug"


def test_derive_rows_processed_aggregation_counts_source_rows_not_result() -> None:
    """Aggregation 3 source rows -> 1 result: rows_processed == 3 (source rows), not 1 (result token).

    Pre-F2 per-leaf logic counted only the single SUCCESS/DEFAULT_FLOW result
    token (the 3 BATCH_CONSUMED tokens hit the no-op case) and reported
    rows_processed == 1.  The 3 BATCH_CONSUMED tokens retain their source
    row_ids, so the faithful per-source-row count is 3.
    """
    from typing import Any

    from elspeth.contracts.enums import Determinism, OutputMode
    from elspeth.contracts.schema_contract import PipelineRow
    from elspeth.core.config import AggregationSettings, SourceSettings, TriggerConfig
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.infrastructure.base import BaseTransform
    from elspeth.plugins.infrastructure.results import TransformResult
    from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
    from tests.fixtures.landscape import make_landscape_db
    from tests.fixtures.plugins import CollectSink, ListSource
    from tests.fixtures.stores import MockPayloadStore

    class _SumAggregator(BaseTransform):
        name = "sum-aggregator"
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"
        source_file_hash = None
        input_schema = _TestSchema
        output_schema = _TestSchema
        is_batch_aware = True
        passes_through_input = False
        on_success = "output"
        on_error = "discard"

        def __init__(self) -> None:
            super().__init__({"schema": {"mode": "observed"}})

        def process(self, rows: list[PipelineRow], ctx: Any) -> TransformResult:  # type: ignore[override]
            if not rows:
                return TransformResult.error({"reason": "empty"}, retryable=False)
            total = sum(r.to_dict().get("value", 0) for r in rows)
            return TransformResult.success(PipelineRow({"sum": total}, rows[0].contract), success_reason={"action": "sum"})

    n = 3
    db = make_landscape_db()
    src = ListSource([{"value": i + 1} for i in range(n)], name="list_source", on_success="agg_in")
    out = CollectSink("output")
    agg = _SumAggregator()
    agg_settings = AggregationSettings(
        name="sum_agg",
        plugin=agg.name,
        input="agg_in",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=n, timeout_seconds=3600),
        output_mode=OutputMode.TRANSFORM,
    )
    graph = ExecutionGraph.from_plugin_instances(
        source=as_source(src),
        source_settings=SourceSettings(plugin=src.name, on_success="agg_in", options={}),
        transforms=[],
        sinks={"output": as_sink(out)},
        aggregations={"sum_agg": (as_transform(agg), agg_settings)},
        gates=[],
    )
    agg_id_map = graph.get_aggregation_id_map()
    agg_node_id = agg_id_map[next(iter(agg_id_map))]
    agg.node_id = agg_node_id
    config = PipelineConfig(
        source=as_source(src),
        transforms=[as_transform(agg)],
        sinks={"output": as_sink(out)},
        aggregation_settings={agg_node_id: agg_settings},
    )
    run = Orchestrator(db).run(config, graph=graph, payload_store=MockPayloadStore())

    # Live truth: 3 source rows, 1 aggregate result.
    assert run.rows_processed == n
    assert run.rows_succeeded == 1

    derived = _derive_rows_processed(db, run.run_id)
    assert derived == n, f"aggregation: derive rows_processed must be source-row count {n}, got {derived}"
    assert derived == run.rows_processed, "aggregation: derive must match uninterrupted run rows_processed"
    assert derived != run.rows_succeeded, "aggregation: derive must NOT be the result-token count (1) — that was the pre-F2 bug"


def test_derive_rows_processed_expand_counts_source_rows_not_children() -> None:
    """Expand 1 source row -> 3 children: rows_processed == 1 (source rows), not 3 (children).

    Pre-F2 per-leaf logic counted the three SUCCESS/DEFAULT_FLOW expand children
    and reported rows_processed == 3.  Expand children inherit the parent's
    single source row_id, so the faithful per-source-row count is 1.
    """
    from typing import Any

    from elspeth.contracts.enums import Determinism
    from elspeth.contracts.schema_contract import PipelineRow
    from elspeth.core.config import SourceSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.infrastructure.base import BaseTransform
    from elspeth.plugins.infrastructure.results import TransformResult
    from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
    from tests.fixtures.factories import wire_transforms
    from tests.fixtures.landscape import make_landscape_db
    from tests.fixtures.plugins import CollectSink, ListSource
    from tests.fixtures.stores import MockPayloadStore

    class _ExpandTransform(BaseTransform):
        name = "expand-transform"
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"
        source_file_hash = None
        input_schema = _TestSchema
        output_schema = _TestSchema
        is_batch_aware = False
        passes_through_input = False
        creates_tokens = True
        on_success = "output"
        on_error = "discard"

        def __init__(self) -> None:
            super().__init__({"schema": {"mode": "observed"}})

        def process(self, row: PipelineRow, ctx: Any) -> TransformResult:  # type: ignore[override]
            c = row.contract
            return TransformResult.success_multi(
                (PipelineRow({"value": 10}, c), PipelineRow({"value": 20}, c), PipelineRow({"value": 30}, c)),
                success_reason={"action": "expand"},
            )

    db = make_landscape_db()
    src = ListSource([{"value": 1}], name="list_source", on_success="source_out")
    out = CollectSink("output")
    xf = _ExpandTransform()
    wired = wire_transforms([as_transform(xf)], source_connection="source_out", final_sink="output", names=["expand-transform"])
    graph = ExecutionGraph.from_plugin_instances(
        source=as_source(src),
        source_settings=SourceSettings(plugin=src.name, on_success="source_out", options={}),
        transforms=wired,
        sinks={"output": as_sink(out)},
        aggregations={},
        gates=[],
    )
    config = PipelineConfig(source=as_source(src), transforms=[as_transform(xf)], sinks={"output": as_sink(out)})
    run = Orchestrator(db).run(config, graph=graph, payload_store=MockPayloadStore())

    # Live truth: 1 source row, 3 expanded children, 1 expand parent.
    assert run.rows_processed == 1
    assert run.rows_succeeded == 3
    assert run.rows_expanded == 1

    derived = _derive_rows_processed(db, run.run_id)
    assert derived == 1, f"expand: derive rows_processed must be source-row count 1, got {derived}"
    assert derived == run.rows_processed, "expand: derive must match uninterrupted run rows_processed"
    assert derived != run.rows_succeeded, "expand: derive must NOT be the per-child count (3) — that was the pre-F2 bug"
