"""ADR-019 MCP outcome distribution tests."""

from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import NodeType, RunStatus, TerminalOutcome, TerminalPath
from elspeth.mcp.analyzers.reports import get_outcome_analysis, get_run_summary
from tests.fixtures.landscape import make_recorder_with_run, register_test_node


def _record_token(
    setup_run_id: str,
    source_node_id: str,
    data_flow,
    *,
    row_index: int,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    **fields,
) -> None:
    row = data_flow.create_row(
        run_id=setup_run_id,
        source_node_id=source_node_id,
        row_index=row_index,
        data={"row": row_index},
        source_row_index=row_index,
        ingest_sequence=row_index,
    )
    token = data_flow.create_token(row.row_id)
    data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=setup_run_id),
        outcome=outcome,
        path=path,
        **fields,
    )


def test_outcome_reports_group_by_path_not_lifecycle_only() -> None:
    setup = make_recorder_with_run(run_id="two-axis-report-run", source_node_id="source-0")
    register_test_node(
        setup.data_flow,
        setup.run_id,
        "sink-0",
        node_type=NodeType.SINK,
        plugin_name="csv_sink",
    )
    _record_token(
        setup.run_id,
        setup.source_node_id,
        setup.data_flow,
        row_index=0,
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.DEFAULT_FLOW,
        sink_name="sink-0",
    )
    _record_token(
        setup.run_id,
        setup.source_node_id,
        setup.data_flow,
        row_index=1,
        outcome=TerminalOutcome.SUCCESS,
        path=TerminalPath.FILTER_DROPPED,
    )
    setup.run_lifecycle.complete_run(setup.run_id, RunStatus.COMPLETED)

    outcome_analysis = get_outcome_analysis(setup.db, setup.factory, setup.run_id)
    run_summary = get_run_summary(setup.db, setup.factory, setup.run_id)

    for report in (outcome_analysis, run_summary):
        assert "error" not in report
        buckets = {(entry["outcome"], entry["path"], entry["completed"]): entry["count"] for entry in report["outcome_distribution"]}
        assert buckets[("success", "default_flow", True)] == 1
        assert buckets[("success", "filter_dropped", True)] == 1
