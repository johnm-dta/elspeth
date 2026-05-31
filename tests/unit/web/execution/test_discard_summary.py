"""Tests for web discard summaries under ADR-019 token outcomes."""

from datetime import UTC, datetime

from elspeth.contracts import NodeType
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape.schema import transform_errors_table, validation_errors_table
from elspeth.web.execution.discard_summary import load_discard_summaries_from_db
from tests.fixtures.landscape import make_recorder_with_run, register_test_node


def test_discard_summary_counts_completed_discard_path() -> None:
    setup = make_recorder_with_run(run_id="discard-summary-run", source_node_id="source-0")
    row = setup.data_flow.create_row(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=0,
        data={"id": "drop-me"},
    )
    token = setup.data_flow.create_token(row.row_id)
    setup.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=setup.run_id),
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.SINK_DISCARDED,
        sink_name=DISCARD_SINK_NAME,
        error_hash="a" * 64,
    )

    summaries = load_discard_summaries_from_db(setup.db, [setup.run_id])

    assert summaries[setup.run_id].total == 1
    assert summaries[setup.run_id].sink_discards == 1
    assert [stage.model_dump() for stage in summaries[setup.run_id].stages] == [
        {
            "stage": "sink_discard",
            "node_id": None,
            "count": 1,
        }
    ]


def test_discard_summary_carries_stage_attribution_for_validation_and_transform_discards() -> None:
    setup = make_recorder_with_run(run_id="discard-stage-run", source_node_id="source_csv")
    transform_id = register_test_node(
        setup.data_flow,
        setup.run_id,
        "normalize_url",
        node_type=NodeType.TRANSFORM,
        plugin_name="url_normalizer",
    )
    row = setup.data_flow.create_row(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=0,
        data={"url": ""},
    )
    token = setup.data_flow.create_token(row.row_id)
    now = datetime.now(tz=UTC)
    with setup.db.connection() as conn:
        conn.execute(
            validation_errors_table.insert(),
            [
                {
                    "error_id": "verr_discard_1",
                    "run_id": setup.run_id,
                    "node_id": setup.source_node_id,
                    "row_id": row.row_id,
                    "row_hash": "hash-validation-1",
                    "row_data_json": "{}",
                    "error": "url field required",
                    "schema_mode": "fixed",
                    "destination": "discard",
                    "created_at": now,
                },
                {
                    "error_id": "verr_discard_2",
                    "run_id": setup.run_id,
                    "node_id": setup.source_node_id,
                    "row_id": row.row_id,
                    "row_hash": "hash-validation-2",
                    "row_data_json": "{}",
                    "error": "url field required",
                    "schema_mode": "fixed",
                    "destination": "discard",
                    "created_at": now,
                },
            ],
        )
        conn.execute(
            transform_errors_table.insert().values(
                error_id="terr_discard_1",
                run_id=setup.run_id,
                token_id=token.token_id,
                transform_id=transform_id,
                row_hash="hash-transform",
                row_data_json="{}",
                error_details_json='{"reason":"validation_failed"}',
                destination="discard",
                created_at=now,
            )
        )

    summary = load_discard_summaries_from_db(setup.db, [setup.run_id])[setup.run_id]

    assert summary.total == 3
    assert summary.validation_errors == 2
    assert summary.transform_errors == 1
    assert [stage.model_dump() for stage in summary.stages] == [
        {
            "stage": "source_validation",
            "node_id": "source_csv",
            "count": 2,
        },
        {
            "stage": "transform_validation",
            "node_id": "normalize_url",
            "count": 1,
        },
    ]
