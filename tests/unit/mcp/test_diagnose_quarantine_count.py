"""ADR-019: diagnose() must count source quarantines by TerminalPath."""

from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import RunStatus, TerminalOutcome, TerminalPath
from elspeth.mcp.analyzers.diagnostics import diagnose
from tests.fixtures.landscape import make_recorder_with_run


def test_diagnose_counts_quarantined_under_new_path() -> None:
    setup = make_recorder_with_run(run_id="quarantine-run", source_node_id="source-0")

    for row_index in range(3):
        row = setup.data_flow.create_row(
            run_id=setup.run_id,
            source_node_id=setup.source_node_id,
            row_index=row_index,
            data={"col": f"bad-{row_index}"},
            source_row_index=row_index,
            ingest_sequence=row_index,
        )
        token = setup.data_flow.create_token(row.row_id)
        setup.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=setup.run_id),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.QUARANTINED_AT_SOURCE,
            error_hash=f"{row_index:064x}",
        )

    setup.run_lifecycle.complete_run(setup.run_id, RunStatus.COMPLETED)

    result = diagnose(setup.db, setup.factory)

    quarantine_problems = [p for p in result["problems"] if p["type"] == "quarantined_rows"]
    assert len(quarantine_problems) == 1
    assert quarantine_problems[0]["count"] == 3
