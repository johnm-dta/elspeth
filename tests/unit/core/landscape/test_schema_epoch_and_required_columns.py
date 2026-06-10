"""Schema epoch + required-columns + provenance-write guards (epoch 20)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from elspeth.core.landscape.database import _REQUIRED_COLUMNS
from elspeth.core.landscape.schema import (
    SQLITE_SCHEMA_EPOCH,
    checkpoints_table,
    node_states_table,
    token_work_items_table,
    tokens_table,
)
from tests.fixtures.landscape import make_recorder_with_run


def test_epoch_is_twenty() -> None:
    assert SQLITE_SCHEMA_EPOCH == 20


def test_token_work_items_has_barrier_blocked_at() -> None:
    assert "barrier_blocked_at" in token_work_items_table.c


def test_checkpoints_have_barrier_scalars_column() -> None:
    assert "barrier_scalars_json" in checkpoints_table.c


def test_tokens_has_token_data_ref_column() -> None:
    assert "token_data_ref" in tokens_table.c


def test_node_states_has_resume_checkpoint_id_column() -> None:
    assert "resume_checkpoint_id" in node_states_table.c


def test_required_columns_include_new_columns_and_openrouter() -> None:
    required = set(_REQUIRED_COLUMNS)
    assert ("tokens", "token_data_ref") in required
    assert ("node_states", "resume_checkpoint_id") in required
    # F3 co-fix: the openrouter catalog columns (added at epoch 10) were never
    # added to the Postgres staleness backstop.
    assert ("runs", "openrouter_catalog_sha256") in required
    assert ("runs", "openrouter_catalog_source") in required


def test_begin_node_state_writes_resume_checkpoint_id() -> None:
    """Provenance column round-trips: written by begin_node_state, readable from DB.

    FK enforcement is ON (PRAGMA foreign_keys=ON in _configure_sqlite_pragmas).
    We must insert a real checkpoints row before writing the node_state with
    a non-NULL resume_checkpoint_id.  We insert it directly via
    checkpoints_table.insert() to avoid the full CheckpointManager / ExecutionGraph
    setup — CheckpointManager is not what we are testing here.
    """
    setup = make_recorder_with_run(run_id="provenance-run", source_node_id="src-node")
    db = setup.db
    factory = setup.factory

    # Create a row and token so the FK chain for node_states is satisfied
    row = factory.data_flow.create_row(
        "provenance-run", "src-node", row_index=0, data={"x": 1}, row_id="prov-row-1", source_row_index=0, ingest_sequence=0
    )
    token = factory.data_flow.create_token(row.row_id, token_id="prov-tok-1")

    # Insert a minimal checkpoint row directly
    ck_id = "ck-prov-1"
    with db.engine.begin() as conn:
        conn.execute(
            checkpoints_table.insert().values(
                checkpoint_id=ck_id,
                run_id="provenance-run",
                sequence_number=1,
                created_at=datetime.now(UTC),
                upstream_topology_hash="a" * 64,
                format_version=4,
            )
        )

    # Call begin_node_state with the new keyword argument
    ns = factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id="src-node",
        run_id="provenance-run",
        step_index=0,
        input_data={"x": 1},
        resume_checkpoint_id=ck_id,
    )

    # Read the row back and verify the column was persisted
    with db.engine.connect() as conn:
        row_back = conn.execute(select(node_states_table).where(node_states_table.c.state_id == ns.state_id)).fetchone()

    assert row_back is not None, "begin_node_state did not persist a node_states row"
    assert row_back.resume_checkpoint_id == ck_id


def test_begin_node_state_resume_checkpoint_id_defaults_to_none() -> None:
    """Existing callers (no resume_checkpoint_id kwarg) produce NULL in the column.

    Preserves the run-1 contract: NULL = original run, not a resume re-drive.
    """
    setup = make_recorder_with_run(run_id="default-run", source_node_id="src-node")
    db = setup.db
    factory = setup.factory

    row = factory.data_flow.create_row(
        "default-run", "src-node", row_index=0, data={"x": 1}, row_id="def-row-1", source_row_index=0, ingest_sequence=0
    )
    token = factory.data_flow.create_token(row.row_id, token_id="def-tok-1")

    ns = factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id="src-node",
        run_id="default-run",
        step_index=0,
        input_data={"x": 1},
        # No resume_checkpoint_id — existing call-site shape
    )

    with db.engine.connect() as conn:
        row_back = conn.execute(select(node_states_table).where(node_states_table.c.state_id == ns.state_id)).fetchone()

    assert row_back is not None
    assert row_back.resume_checkpoint_id is None
