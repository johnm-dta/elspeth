"""Schema epoch + required-shape + provenance-write guards (epoch 25)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import inspect, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex

from elspeth.contracts.scheduler import SchedulerEventType
from elspeth.core.landscape.database import _REQUIRED_COLUMNS, _REQUIRED_INDEXES, LandscapeDB
from elspeth.core.landscape.schema import (
    SQLITE_SCHEMA_EPOCH,
    artifacts_table,
    checkpoints_table,
    metadata,
    node_states_table,
    routing_events_table,
    run_web_plugin_policy_table,
    token_work_items_table,
    tokens_table,
)
from tests.fixtures.landscape import make_recorder_with_run


def test_epoch_is_twenty_five() -> None:
    assert SQLITE_SCHEMA_EPOCH == 25


def test_epoch_25_artifact_idempotency_index_is_partial_and_cross_dialect() -> None:
    index = next(index for index in artifacts_table.indexes if index.name == "uq_artifacts_run_idempotency_key")

    assert index.unique is True
    assert [column.name for column in index.columns] == ["run_id", "idempotency_key"]
    sqlite_ddl = str(CreateIndex(index).compile(dialect=sqlite.dialect()))
    postgres_ddl = str(CreateIndex(index).compile(dialect=postgresql.dialect()))
    assert "UNIQUE INDEX uq_artifacts_run_idempotency_key" in sqlite_ddl
    assert "WHERE idempotency_key IS NOT NULL" in sqlite_ddl
    assert "UNIQUE INDEX uq_artifacts_run_idempotency_key" in postgres_ddl
    assert "WHERE idempotency_key IS NOT NULL" in postgres_ddl


def test_epoch_25_artifact_idempotency_index_is_required_at_startup() -> None:
    assert ("artifacts", "uq_artifacts_run_idempotency_key") in _REQUIRED_INDEXES


def test_fresh_sqlite_schema_reflects_artifact_idempotency_index() -> None:
    db = LandscapeDB.in_memory()
    try:
        indexes = {entry["name"]: entry for entry in inspect(db.engine).get_indexes("artifacts")}
    finally:
        db.close()

    reflected = indexes["uq_artifacts_run_idempotency_key"]
    assert reflected["unique"] == 1
    assert reflected["column_names"] == ["run_id", "idempotency_key"]
    assert "idempotency_key IS NOT NULL" in str(reflected["dialect_options"]["sqlite_where"])


def test_epoch_23_web_plugin_policy_table_is_one_to_one_with_runs() -> None:
    assert "run_web_plugin_policy" in metadata.tables
    assert run_web_plugin_policy_table.primary_key.columns.keys() == ["run_id"]
    assert {column.name for column in run_web_plugin_policy_table.columns} == {
        "run_id",
        "schema_version",
        "policy_hash",
        "snapshot_hash",
        "authorized_plugin_ids_json",
        "available_plugin_ids_json",
        "control_modes_json",
        "selected_implementations_json",
        "selected_profile_aliases_json",
        "plugin_code_identities_json",
        "binding_generation_fingerprint",
        "decision_codes_json",
    }


def test_required_columns_include_epoch_23_web_plugin_policy_evidence() -> None:
    required = set(_REQUIRED_COLUMNS)
    for column in run_web_plugin_policy_table.columns:
        assert ("run_web_plugin_policy", column.name) in required


def test_token_work_items_has_barrier_blocked_at() -> None:
    assert "barrier_blocked_at" in token_work_items_table.c


def test_token_work_items_has_barrier_adopted_epoch() -> None:
    """Epoch 21: adoption CAS marker (§C.4 row 6a) — written only by the
    slice-3 fenced adoption verb; NULL = intake-pending."""
    assert "barrier_adopted_epoch" in token_work_items_table.c
    assert token_work_items_table.c.barrier_adopted_epoch.nullable


def test_epoch_21_coordination_tables_are_defined() -> None:
    """Epoch 21 (ADR-030 slice 2): the four coordination tables exist in metadata."""
    assert "run_coordination" in metadata.tables
    assert "run_workers" in metadata.tables
    assert "run_coordination_events" in metadata.tables
    assert "coalesce_branch_losses" in metadata.tables


def test_checkpoints_have_barrier_scalars_column() -> None:
    assert "barrier_scalars_json" in checkpoints_table.c


def test_tokens_has_token_data_ref_column() -> None:
    assert "token_data_ref" in tokens_table.c


def test_node_states_has_resume_checkpoint_id_column() -> None:
    assert "resume_checkpoint_id" in node_states_table.c


def test_routing_events_has_run_id_column() -> None:
    assert "run_id" in routing_events_table.c


def test_required_columns_include_new_columns_and_openrouter() -> None:
    required = set(_REQUIRED_COLUMNS)
    assert ("tokens", "token_data_ref") in required
    assert ("node_states", "resume_checkpoint_id") in required
    # F3 co-fix: the openrouter catalog columns (added at epoch 10) were never
    # added to the Postgres staleness backstop.
    assert ("runs", "openrouter_catalog_sha256") in required
    assert ("runs", "openrouter_catalog_source") in required
    # Epoch 20: F1 durability unification (additive half).
    assert ("token_work_items", "barrier_blocked_at") in required
    assert ("checkpoints", "barrier_scalars_json") in required


def test_required_columns_include_epoch_21_coordination_substrate() -> None:
    """Epoch 21 columns must participate in the Postgres staleness backstop."""
    required = set(_REQUIRED_COLUMNS)
    assert ("token_work_items", "barrier_adopted_epoch") in required
    for column in ("run_id", "leader_worker_id", "leader_epoch", "leader_heartbeat_expires_at", "updated_at"):
        assert ("run_coordination", column) in required
    for column in (
        "worker_id",
        "run_id",
        "role",
        "status",
        "registered_at",
        "heartbeat_expires_at",
        "departed_at",
        "evicted_at",
        "evicted_by_worker_id",
        "pid",
        "hostname",
        "entry_point",
    ):
        assert ("run_workers", column) in required
    for column in ("seq", "event_id", "run_id", "event_type", "worker_id", "leader_epoch", "recorded_at", "context_json"):
        assert ("run_coordination_events", column) in required
    for column in (
        "loss_id",
        "run_id",
        "coalesce_name",
        "row_id",
        "branch_name",
        "token_id",
        "reason",
        "recorded_by",
        "recorded_at",
        "adopted_epoch",
    ):
        assert ("coalesce_branch_losses", column) in required


def test_required_columns_include_epoch_22_routing_event_run_scope() -> None:
    """Epoch 22 routing event run ownership must trip stale-DB detection."""
    assert ("routing_events", "run_id") in set(_REQUIRED_COLUMNS)


def test_checkpoint_blob_columns_are_gone() -> None:
    """Epoch 20 (subtractive half): the barrier buffer blob columns are deleted."""
    assert "aggregation_state_json" not in checkpoints_table.c
    assert "coalesce_state_json" not in checkpoints_table.c


def test_restore_blocked_event_type_is_gone() -> None:
    """Epoch 20 (subtractive half): the blob-restore scheduler event type is deleted."""
    assert "restore_blocked" not in {e.value for e in SchedulerEventType}


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


def test_token_work_items_barrier_blocked_at_defaults_to_none() -> None:
    """A freshly created token_work_items row has barrier_blocked_at IS NULL.

    Epoch 20 (additive half): nothing writes the column yet — Task 1.3 adds the
    repo verb that stamps it when a token blocks at a barrier. NULL = the work
    item never blocked at a barrier.
    """
    setup = make_recorder_with_run(run_id="barrier-run", source_node_id="src-node")
    db = setup.db
    factory = setup.factory

    # Satisfy the (token_id, run_id) / (row_id, run_id) FK chain
    row = factory.data_flow.create_row(
        "barrier-run", "src-node", row_index=0, data={"x": 1}, row_id="bar-row-1", source_row_index=0, ingest_sequence=0
    )
    token = factory.data_flow.create_token(row.row_id, token_id="bar-tok-1")

    # Insert a minimal work item directly — existing call-site shape, no barrier_blocked_at
    now = datetime.now(UTC)
    with db.engine.begin() as conn:
        conn.execute(
            token_work_items_table.insert().values(
                work_item_id="bar-wi-1",
                run_id="barrier-run",
                token_id=token.token_id,
                row_id=row.row_id,
                step_index=0,
                ingest_sequence=0,
                row_payload_json="{}",
                status="ready",
                attempt=0,
                available_at=now,
                created_at=now,
                updated_at=now,
            )
        )

    with db.engine.connect() as conn:
        row_back = conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == "bar-wi-1")).fetchone()

    assert row_back is not None
    assert row_back.barrier_blocked_at is None


def test_checkpoints_barrier_scalars_json_defaults_to_none() -> None:
    """A freshly created checkpoint row has barrier_scalars_json IS NULL.

    Epoch 20 (additive half): nothing writes the column yet — Task 1.2 shrinks
    the checkpoint row to carry scalar barrier metadata here.
    """
    setup = make_recorder_with_run(run_id="ck-null-run", source_node_id="src-node")
    db = setup.db

    ck_id = "ck-null-1"
    with db.engine.begin() as conn:
        conn.execute(
            checkpoints_table.insert().values(
                checkpoint_id=ck_id,
                run_id="ck-null-run",
                sequence_number=1,
                created_at=datetime.now(UTC),
                upstream_topology_hash="a" * 64,
                format_version=4,
            )
        )

    with db.engine.connect() as conn:
        row_back = conn.execute(select(checkpoints_table).where(checkpoints_table.c.checkpoint_id == ck_id)).fetchone()

    assert row_back is not None
    assert row_back.barrier_scalars_json is None
