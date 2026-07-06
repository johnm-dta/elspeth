"""Persisted lineage invariants exercised through the real recorder repositories."""

from __future__ import annotations

import pytest
from tests.fixtures.landscape import RecorderSetup, make_recorder_with_run

from elspeth.contracts import NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.schema import token_parents_table

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _register_source_run(setup: RecorderSetup, *, run_id: str, source_node_id: str) -> str:
    setup.run_lifecycle.begin_run(run_id=run_id, config={}, canonical_version="v1")
    source = setup.data_flow.register_node(
        run_id=run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=source_node_id,
        schema_config=DYNAMIC_SCHEMA,
    )
    return source.node_id


def _create_row_token(
    setup: RecorderSetup,
    *,
    row_index: int,
    token_id: str,
    data: dict[str, object] | None = None,
    run_id: str | None = None,
    source_node_id: str | None = None,
):
    row = setup.data_flow.create_row(
        run_id=run_id or setup.run_id,
        source_node_id=source_node_id or setup.source_node_id,
        row_index=row_index,
        data=data or {"row_index": row_index},
        source_row_index=row_index,
        ingest_sequence=row_index,
    )
    token = setup.data_flow.create_token(row.row_id, token_id=token_id)
    return row, token


def _record_success(setup: RecorderSetup, *, run_id: str, token_id: str, sink: str) -> None:
    setup.data_flow.record_token_outcome(
        TokenRef(token_id=token_id, run_id=run_id),
        TerminalOutcome.SUCCESS,
        TerminalPath.DEFAULT_FLOW,
        sink_name=sink,
    )


def _record_buffered(setup: RecorderSetup, *, run_id: str, token_id: str, batch_id: str) -> None:
    aggregation_node = setup.data_flow.register_node(
        run_id=run_id,
        plugin_name="aggregation",
        node_type=NodeType.AGGREGATION,
        plugin_version="1.0",
        config={},
        node_id=f"aggregation-{batch_id}",
        schema_config=DYNAMIC_SCHEMA,
    )
    setup.execution.create_batch(run_id, aggregation_node.node_id, batch_id=batch_id)
    setup.data_flow.record_token_outcome(
        TokenRef(token_id=token_id, run_id=run_id),
        None,
        TerminalPath.BUFFERED,
        batch_id=batch_id,
    )


def test_row_id_resolution_uses_persisted_terminal_outcome() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-row", source_node_id="source-lineage-row")
    row, token = _create_row_token(setup, row_index=0, token_id="token-row-resolution")
    _record_success(setup, run_id=setup.run_id, token_id=token.token_id, sink="primary")

    lineage = explain(setup.query, setup.data_flow, setup.run_id, row_id=row.row_id)

    assert lineage is not None
    assert lineage.token.token_id == token.token_id
    assert lineage.source_row.row_id == row.row_id
    assert lineage.outcome is not None
    assert lineage.outcome.sink_name == "primary"


def test_row_id_resolution_ignores_persisted_non_terminal_buffered_outcome() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-buffered", source_node_id="source-lineage-buffered")
    row, token = _create_row_token(setup, row_index=0, token_id="token-buffered")
    _record_buffered(setup, run_id=setup.run_id, token_id=token.token_id, batch_id="batch-buffered")

    assert explain(setup.query, setup.data_flow, setup.run_id, row_id=row.row_id) is None


def test_row_id_resolution_requires_sink_when_persisted_terminals_are_ambiguous() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-sink-required", source_node_id="source-lineage-sink-required")
    row, first = _create_row_token(setup, row_index=0, token_id="token-sink-a")
    second = setup.data_flow.create_token(row.row_id, token_id="token-sink-b")
    _record_success(setup, run_id=setup.run_id, token_id=first.token_id, sink="left")
    _record_success(setup, run_id=setup.run_id, token_id=second.token_id, sink="right")

    with pytest.raises(ValueError, match="Provide sink parameter"):
        explain(setup.query, setup.data_flow, setup.run_id, row_id=row.row_id)

    lineage = explain(setup.query, setup.data_flow, setup.run_id, row_id=row.row_id, sink="right")
    assert lineage is not None
    assert lineage.token.token_id == second.token_id
    assert lineage.outcome is not None
    assert lineage.outcome.sink_name == "right"


def test_row_id_resolution_rejects_same_sink_ambiguity_from_persisted_outcomes() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-same-sink", source_node_id="source-lineage-same-sink")
    row, first = _create_row_token(setup, row_index=0, token_id="token-same-sink-a")
    second = setup.data_flow.create_token(row.row_id, token_id="token-same-sink-b")
    _record_success(setup, run_id=setup.run_id, token_id=first.token_id, sink="shared")
    _record_success(setup, run_id=setup.run_id, token_id=second.token_id, sink="shared")

    with pytest.raises(ValueError, match="tokens at sink 'shared'"):
        explain(setup.query, setup.data_flow, setup.run_id, row_id=row.row_id, sink="shared")


def test_explain_includes_parent_token_from_persisted_fork_relationship() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-parent", source_node_id="source-lineage-parent")
    row, parent = _create_row_token(setup, row_index=0, token_id="token-fork-parent")
    children, _fork_group_id = setup.data_flow.fork_token(
        TokenRef(token_id=parent.token_id, run_id=setup.run_id),
        row.row_id,
        ["left", "right"],
        step_in_pipeline=1,
    )
    child = children[0]

    lineage = explain(setup.query, setup.data_flow, setup.run_id, token_id=child.token_id)

    assert lineage is not None
    assert lineage.token.token_id == child.token_id
    assert [parent_token.token_id for parent_token in lineage.parent_tokens] == [parent.token_id]


def test_explain_rejects_persisted_group_id_without_parent_relationship() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-orphan-group", source_node_id="source-lineage-orphan-group")
    row = setup.data_flow.create_row(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=0,
        data={"case": "orphan-group"},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = setup.data_flow.create_token(
        row.row_id,
        token_id="token-orphan-group",
        branch_name="left",
        fork_group_id="fork-without-parent",
    )

    with pytest.raises(AuditIntegrityError, match="but no parent relationships"):
        explain(setup.query, setup.data_flow, setup.run_id, token_id=token.token_id)


def test_explain_rejects_parent_relationship_without_group_id_from_corruption() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-parent-no-group", source_node_id="source-lineage-parent-no-group")
    row, parent = _create_row_token(setup, row_index=0, token_id="token-parent-no-group-parent")
    child = setup.data_flow.create_token(row.row_id, token_id="token-parent-no-group-child")

    # Corruption boundary: production fork/coalesce/expand APIs write both the group
    # marker and token_parents row atomically, so this one-sided parent relationship
    # must be manufactured directly in the audit database.
    with setup.db.connection() as conn:
        conn.execute(
            token_parents_table.insert().values(
                token_id=child.token_id,
                parent_token_id=parent.token_id,
                ordinal=0,
            )
        )

    with pytest.raises(AuditIntegrityError, match="but no group ID"):
        explain(setup.query, setup.data_flow, setup.run_id, token_id=child.token_id)


def test_explain_rejects_cross_run_parent_relationship_from_corruption() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-cross-parent", source_node_id="source-lineage-cross-parent")
    other_source_id = _register_source_run(
        setup, run_id="run-lineage-cross-parent-other", source_node_id="source-lineage-cross-parent-other"
    )
    row = setup.data_flow.create_row(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=0,
        data={"case": "cross-run-child"},
        source_row_index=0,
        ingest_sequence=0,
    )
    child = setup.data_flow.create_token(
        row.row_id,
        token_id="token-cross-run-child",
        branch_name="left",
        fork_group_id="fork-cross-run",
    )
    _other_row, other_parent = _create_row_token(
        setup,
        row_index=0,
        token_id="token-cross-run-parent",
        data={"case": "cross-run-parent"},
        run_id="run-lineage-cross-parent-other",
        source_node_id=other_source_id,
    )

    # Corruption boundary: production fork_token validates parent run ownership.
    # A cross-run parent can only appear here by directly corrupting token_parents.
    with setup.db.connection() as conn:
        conn.execute(
            token_parents_table.insert().values(
                token_id=child.token_id,
                parent_token_id=other_parent.token_id,
                ordinal=0,
            )
        )

    with pytest.raises(AuditIntegrityError, match="belongs to run 'run-lineage-cross-parent-other'"):
        explain(setup.query, setup.data_flow, setup.run_id, token_id=child.token_id)


def test_explain_direct_token_hides_foreign_run_token() -> None:
    setup = make_recorder_with_run(run_id="run-lineage-token-scope", source_node_id="source-lineage-token-scope")
    other_source_id = _register_source_run(
        setup,
        run_id="run-lineage-token-scope-other",
        source_node_id="source-lineage-token-scope-other",
    )
    _foreign_row, foreign_token = _create_row_token(
        setup,
        row_index=0,
        token_id="token-foreign-run",
        run_id="run-lineage-token-scope-other",
        source_node_id=other_source_id,
    )

    assert explain(setup.query, setup.data_flow, setup.run_id, token_id=foreign_token.token_id) is None
