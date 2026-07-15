from __future__ import annotations

import pytest
from sqlalchemy import event, func, select

from elspeth.contracts import BatchStatus, NodeType, TriggerType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import batch_members_table
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run, register_test_node

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-0", source_plugin_name="csv")
    register_test_node(setup.data_flow, setup.run_id, "agg-1", node_type=NodeType.AGGREGATION, plugin_name="aggregator")
    return setup.db, setup.factory


def _setup_with_token(
    *,
    run_id: str = "run-1",
) -> tuple[LandscapeDB, RecorderFactory]:
    db, factory = _setup(run_id=run_id)
    factory.data_flow.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1", source_row_index=0, ingest_sequence=0)
    factory.data_flow.create_token("row-1", token_id="tok-1")
    factory.data_flow.create_row(run_id, "source-0", 1, {"name": "test2"}, row_id="row-2", source_row_index=1, ingest_sequence=1)
    factory.data_flow.create_token("row-2", token_id="tok-2")
    return db, factory


def _setup_with_sink(
    *,
    run_id: str = "run-1",
) -> tuple[LandscapeDB, RecorderFactory]:
    """Setup with token and a sink node for artifact tests."""
    db, factory = _setup_with_token(run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv_sink",
        node_type=NodeType.SINK,
        plugin_version="1.0",
        config={},
        node_id="sink-0",
        schema_config=_DYNAMIC_SCHEMA,
    )
    return db, factory


def _setup_two_runs_with_batch_integrity_records() -> tuple[LandscapeDB, RecorderFactory]:
    """Create two runs with tokens and node states for cross-run ownership tests."""
    db = make_landscape_db()
    factory = make_factory(db)

    for run_id, suffix in (("run-A", "A"), ("run-B", "B")):
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id="agg-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row(run_id, "source-0", 0, {"run": run_id}, row_id=f"row-{suffix}", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token(f"row-{suffix}", token_id=f"tok-{suffix}")
        factory.execution.begin_node_state(
            f"tok-{suffix}",
            "agg-1",
            run_id,
            0,
            {"run": run_id},
            state_id=f"state-{suffix}",
        )

    return db, factory


# ---------------------------------------------------------------------------
# create_batch
# ---------------------------------------------------------------------------


class TestCreateBatch:
    """Tests for BatchRecordingMixin.create_batch."""

    def test_creates_batch_with_draft_status(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.status == BatchStatus.DRAFT

    def test_generates_batch_id_when_not_provided(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.batch_id is not None
        assert isinstance(batch.batch_id, str)
        assert len(batch.batch_id) > 0

    def test_uses_provided_batch_id(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1", batch_id="my-batch-42")

        assert batch.batch_id == "my-batch-42"

    def test_stores_run_id_and_node_id(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.run_id == "run-1"
        assert batch.aggregation_node_id == "agg-1"

    def test_default_attempt_is_zero(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.attempt == 0

    def test_explicit_attempt_number(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1", attempt=3)

        assert batch.attempt == 3

    def test_created_at_is_set(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.created_at is not None

    def test_trigger_fields_initially_none(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.trigger_type is None
        assert batch.trigger_reason is None

    def test_completed_at_initially_none(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.completed_at is None

    def test_aggregation_state_id_initially_none(self):
        _db, factory = _setup()
        batch = factory.execution.create_batch("run-1", "agg-1")

        assert batch.aggregation_state_id is None

    def test_multiple_batches_get_unique_ids(self):
        _db, factory = _setup()
        batch_a = factory.execution.create_batch("run-1", "agg-1")
        batch_b = factory.execution.create_batch("run-1", "agg-1")

        assert batch_a.batch_id != batch_b.batch_id


# ---------------------------------------------------------------------------
# add_batch_member / get_batch_members
# ---------------------------------------------------------------------------


class TestAddBatchMember:
    """Tests for BatchRecordingMixin.add_batch_member."""

    def test_adds_member_to_batch(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        member = factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)

        assert member.batch_id == "b-1"
        assert member.token_id == "tok-1"
        assert member.ordinal == 0

    def test_roundtrip_via_get_batch_members(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)
        factory.execution.add_batch_member("b-1", "tok-2", ordinal=1)

        members = factory.execution.get_batch_members("b-1")

        assert len(members) == 2
        assert members[0].token_id == "tok-1"
        assert members[0].ordinal == 0
        assert members[1].token_id == "tok-2"
        assert members[1].ordinal == 1

    def test_multiple_members_different_ordinals(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=5)
        factory.execution.add_batch_member("b-1", "tok-2", ordinal=2)

        members = factory.execution.get_batch_members("b-1")

        assert len(members) == 2
        # Should be ordered by ordinal
        assert members[0].ordinal == 2
        assert members[1].ordinal == 5

    def test_rejects_token_from_different_run(self):
        """Batch membership must reject tokens owned by another run."""
        _db, factory = _setup_two_runs_with_batch_integrity_records()
        factory.execution.create_batch("run-A", "agg-1", batch_id="batch-A")

        with pytest.raises(AuditIntegrityError, match=r"belongs to run 'run-B'.*belongs to run 'run-A'"):
            factory.execution.add_batch_member("batch-A", "tok-B", ordinal=0)

        assert factory.execution.get_batch_members("batch-A") == []

    @pytest.mark.parametrize(
        "status",
        [BatchStatus.EXECUTING, BatchStatus.COMPLETED, BatchStatus.FAILED],
    )
    def test_rejects_every_non_draft_batch_status(self, status: BatchStatus) -> None:
        """DRAFT is the sole status in which batch membership is mutable."""
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        if status is BatchStatus.EXECUTING:
            factory.execution.update_batch_status("b-1", status)
        else:
            factory.execution.complete_batch("b-1", status)

        with pytest.raises(AuditIntegrityError, match=rf"status {status.value!r}.*immutable"):
            factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)

        assert factory.execution.get_batch_members("b-1") == []

    def test_nonexistent_batch_fails_without_membership(self) -> None:
        _db, factory = _setup_with_token()

        with pytest.raises(AuditIntegrityError, match="batch missing not found"):
            factory.execution.add_batch_member("missing", "tok-1", ordinal=0)

        assert factory.execution.get_batch_members("missing") == []

    def test_caller_connection_preserves_draft_guard_and_atomic_write(self) -> None:
        db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        with db.write_connection() as conn:
            member = factory.execution.add_batch_member("b-1", "tok-1", ordinal=0, conn=conn)

        assert member.run_id == "run-1"
        assert [record.token_id for record in factory.execution.get_batch_members("b-1")] == ["tok-1"]

    def test_caller_connection_rejects_closed_batch_without_poisoning_transaction(self) -> None:
        db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="closed")
        factory.execution.update_batch_status("closed", BatchStatus.EXECUTING)
        factory.execution.create_batch("run-1", "agg-1", batch_id="open")

        with db.write_connection() as conn:
            with pytest.raises(AuditIntegrityError, match=r"status 'executing'.*immutable"):
                factory.execution.add_batch_member("closed", "tok-1", ordinal=0, conn=conn)
            factory.execution.add_batch_member("open", "tok-1", ordinal=0, conn=conn)

        assert factory.execution.get_batch_members("closed") == []
        assert [member.token_id for member in factory.execution.get_batch_members("open")] == ["tok-1"]

    def test_duplicate_membership_remains_a_conflict(self) -> None:
        """The hardening does not turn natural-key collisions into success."""
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)

        with pytest.raises(LandscapeRecordError, match="database rejected audit write"):
            factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)

        assert len(factory.execution.get_batch_members("b-1")) == 1

    def test_duplicate_ordinal_remains_a_conflict(self) -> None:
        """Two different tokens cannot occupy the same batch ordinal."""
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)

        with pytest.raises(LandscapeRecordError, match="database rejected audit write"):
            factory.execution.add_batch_member("b-1", "tok-2", ordinal=0)

        assert [member.token_id for member in factory.execution.get_batch_members("b-1")] == ["tok-1"]

    def test_failure_after_insert_rolls_back_membership(self) -> None:
        """An exception after the INSERT cannot leave a partially committed member."""
        db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        def fail_after_member_insert(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
            if statement.lstrip().upper().startswith("INSERT INTO BATCH_MEMBERS"):
                raise RuntimeError("injected post-insert failure")

        event.listen(db.engine, "after_cursor_execute", fail_after_member_insert)
        try:
            with pytest.raises(RuntimeError, match="injected post-insert failure"):
                factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)
        finally:
            event.remove(db.engine, "after_cursor_execute", fail_after_member_insert)

        with db.read_only_connection() as conn:
            assert conn.execute(select(func.count()).select_from(batch_members_table)).scalar_one() == 0


# ---------------------------------------------------------------------------
# update_batch_status
# ---------------------------------------------------------------------------


class TestUpdateBatchStatus:
    """Tests for BatchRecordingMixin.update_batch_status."""

    def test_updates_status_to_executing(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        factory.execution.update_batch_status("b-1", BatchStatus.EXECUTING)

        updated = factory.execution.get_batch("b-1")
        assert updated.status == BatchStatus.EXECUTING

    def test_does_not_set_completed_at_for_executing(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        factory.execution.update_batch_status("b-1", BatchStatus.EXECUTING)

        updated = factory.execution.get_batch("b-1")
        assert updated.completed_at is None

    def test_rejects_terminal_target_statuses(self):
        """Terminal transitions must go through complete_batch()."""
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        for terminal_status in (BatchStatus.COMPLETED, BatchStatus.FAILED):
            with pytest.raises(AuditIntegrityError, match="complete_batch"):
                factory.execution.update_batch_status("b-1", terminal_status)

        updated = factory.execution.get_batch("b-1")
        assert updated.status == BatchStatus.DRAFT
        assert updated.completed_at is None

    def test_sets_trigger_type(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        factory.execution.update_batch_status(
            "b-1",
            BatchStatus.EXECUTING,
            trigger_type=TriggerType.COUNT,
        )

        updated = factory.execution.get_batch("b-1")
        assert updated.trigger_type == TriggerType.COUNT

    def test_sets_trigger_reason(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        factory.execution.update_batch_status(
            "b-1",
            BatchStatus.EXECUTING,
            trigger_type=TriggerType.TIMEOUT,
            trigger_reason="30s elapsed",
        )

        updated = factory.execution.get_batch("b-1")
        assert updated.trigger_type == TriggerType.TIMEOUT
        assert updated.trigger_reason == "30s elapsed"

    def test_sets_state_id(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.begin_node_state(
            "tok-1",
            "agg-1",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        factory.execution.update_batch_status(
            "b-1",
            BatchStatus.EXECUTING,
            state_id="state-1",
        )

        updated = factory.execution.get_batch("b-1")
        assert updated.aggregation_state_id == "state-1"

    def test_rejects_state_from_different_run(self):
        """Batch flush state must belong to the same run as the batch."""
        _db, factory = _setup_two_runs_with_batch_integrity_records()
        factory.execution.create_batch("run-A", "agg-1", batch_id="batch-A")

        with pytest.raises(AuditIntegrityError):
            factory.execution.update_batch_status(
                "batch-A",
                BatchStatus.EXECUTING,
                state_id="state-B",
            )

    def test_rejects_transition_from_completed(self):
        """Terminal status COMPLETED cannot transition to any other status (M2)."""
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.complete_batch("b-1", BatchStatus.COMPLETED)

        with pytest.raises(AuditIntegrityError, match="terminal status"):
            factory.execution.update_batch_status("b-1", BatchStatus.EXECUTING)

    def test_rejects_transition_from_failed(self):
        """Terminal status FAILED cannot transition to any other status (M2)."""
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.complete_batch("b-1", BatchStatus.FAILED)

        with pytest.raises(AuditIntegrityError, match="terminal status"):
            factory.execution.update_batch_status("b-1", BatchStatus.DRAFT)

    def test_rejects_nonexistent_batch(self):
        """Updating status of nonexistent batch raises AuditIntegrityError (M2)."""
        _db, factory = _setup()

        with pytest.raises(AuditIntegrityError, match="not found"):
            factory.execution.update_batch_status("nonexistent", BatchStatus.EXECUTING)


# ---------------------------------------------------------------------------
# complete_batch
# ---------------------------------------------------------------------------


class TestCompleteBatch:
    """Tests for BatchRecordingMixin.complete_batch."""

    def test_returns_updated_batch(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        result = factory.execution.complete_batch("b-1", BatchStatus.COMPLETED)

        assert result.batch_id == "b-1"
        assert result.status == BatchStatus.COMPLETED

    def test_completed_at_is_populated(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        result = factory.execution.complete_batch("b-1", BatchStatus.COMPLETED)

        assert result.completed_at is not None

    def test_complete_with_trigger_info(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        result = factory.execution.complete_batch(
            "b-1",
            BatchStatus.COMPLETED,
            trigger_type=TriggerType.COUNT,
            trigger_reason="reached 10 rows",
        )

        assert result.trigger_type == TriggerType.COUNT
        assert result.trigger_reason == "reached 10 rows"

    def test_complete_as_failed(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        result = factory.execution.complete_batch("b-1", BatchStatus.FAILED)

        assert result.status == BatchStatus.FAILED
        assert result.completed_at is not None

    def test_complete_with_state_id(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.begin_node_state(
            "tok-1",
            "agg-1",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        result = factory.execution.complete_batch(
            "b-1",
            BatchStatus.COMPLETED,
            state_id="state-1",
        )

        assert result.aggregation_state_id == "state-1"

    def test_complete_batch_persists_to_database(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.complete_batch(
            "b-1",
            BatchStatus.COMPLETED,
            trigger_type=TriggerType.END_OF_SOURCE,
            trigger_reason="source exhausted",
        )

        fetched = factory.execution.get_batch("b-1")
        assert fetched.status == BatchStatus.COMPLETED
        assert fetched.trigger_type == TriggerType.END_OF_SOURCE
        assert fetched.trigger_reason == "source exhausted"

    def test_rejects_non_terminal_status_executing(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        with pytest.raises(AuditIntegrityError, match="terminal status"):
            factory.execution.complete_batch("b-1", BatchStatus.EXECUTING)

    def test_rejects_non_terminal_status_draft(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        with pytest.raises(AuditIntegrityError, match="terminal status"):
            factory.execution.complete_batch("b-1", BatchStatus.DRAFT)

    def test_accepts_completed_status(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        result = factory.execution.complete_batch("b-1", BatchStatus.COMPLETED)
        assert result.status == BatchStatus.COMPLETED
        assert result.completed_at is not None


# ---------------------------------------------------------------------------
# get_batch
# ---------------------------------------------------------------------------


class TestGetBatch:
    """Tests for BatchRecordingMixin.get_batch."""

    def test_roundtrip(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        fetched = factory.execution.get_batch("b-1")

        assert fetched is not None
        assert fetched.batch_id == "b-1"
        assert fetched.run_id == "run-1"
        assert fetched.aggregation_node_id == "agg-1"
        assert fetched.status == BatchStatus.DRAFT

    def test_returns_none_for_unknown_id(self):
        _db, factory = _setup()

        result = factory.execution.get_batch("nonexistent-batch")

        assert result is None

    def test_reflects_status_updates(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.update_batch_status("b-1", BatchStatus.EXECUTING)

        fetched = factory.execution.get_batch("b-1")

        assert fetched.status == BatchStatus.EXECUTING


# ---------------------------------------------------------------------------
# get_batches
# ---------------------------------------------------------------------------


class TestGetBatches:
    """Tests for BatchRecordingMixin.get_batches."""

    def test_lists_all_batches_for_run(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-2")

        batches = factory.execution.get_batches("run-1")

        assert len(batches) == 2
        batch_ids = {b.batch_id for b in batches}
        assert batch_ids == {"b-1", "b-2"}

    def test_empty_for_unknown_run(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")

        batches = factory.execution.get_batches("run-unknown")

        assert batches == []

    def test_filter_by_status(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-2")
        factory.execution.complete_batch("b-2", BatchStatus.COMPLETED)

        draft_batches = factory.execution.get_batches("run-1", status=BatchStatus.DRAFT)
        completed_batches = factory.execution.get_batches(
            "run-1",
            status=BatchStatus.COMPLETED,
        )

        assert len(draft_batches) == 1
        assert draft_batches[0].batch_id == "b-1"
        assert len(completed_batches) == 1
        assert completed_batches[0].batch_id == "b-2"

    def test_filter_by_node_id(self):
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="aggregator2",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id="agg-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.create_batch("run-1", "agg-2", batch_id="b-2")

        agg1_batches = factory.execution.get_batches("run-1", node_id="agg-1")
        agg2_batches = factory.execution.get_batches("run-1", node_id="agg-2")

        assert len(agg1_batches) == 1
        assert agg1_batches[0].batch_id == "b-1"
        assert len(agg2_batches) == 1
        assert agg2_batches[0].batch_id == "b-2"

    def test_filter_by_status_and_node_id(self):
        _db, factory = _setup()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="aggregator2",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id="agg-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-2")
        factory.execution.create_batch("run-1", "agg-2", batch_id="b-3")
        factory.execution.complete_batch("b-2", BatchStatus.COMPLETED)

        result = factory.execution.get_batches(
            "run-1",
            status=BatchStatus.DRAFT,
            node_id="agg-1",
        )

        assert len(result) == 1
        assert result[0].batch_id == "b-1"

    def test_does_not_return_batches_from_other_runs(self):
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id="agg-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id="agg-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.create_batch("run-2", "agg-1", batch_id="b-2")

        run1_batches = factory.execution.get_batches("run-1")
        run2_batches = factory.execution.get_batches("run-2")

        assert len(run1_batches) == 1
        assert run1_batches[0].batch_id == "b-1"
        assert len(run2_batches) == 1
        assert run2_batches[0].batch_id == "b-2"


# ---------------------------------------------------------------------------
# get_incomplete_batches
# ---------------------------------------------------------------------------


class TestGetIncompleteBatches:
    """Tests for BatchRecordingMixin.get_incomplete_batches."""

    def test_returns_draft_batches(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-draft")

        incomplete = factory.execution.get_incomplete_batches("run-1")

        assert len(incomplete) == 1
        assert incomplete[0].batch_id == "b-draft"

    def test_returns_executing_batches(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-exec")
        factory.execution.update_batch_status("b-exec", BatchStatus.EXECUTING)

        incomplete = factory.execution.get_incomplete_batches("run-1")

        assert len(incomplete) == 1
        assert incomplete[0].batch_id == "b-exec"
        assert incomplete[0].status == BatchStatus.EXECUTING

    def test_returns_failed_batches(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-fail")
        factory.execution.complete_batch("b-fail", BatchStatus.FAILED)

        incomplete = factory.execution.get_incomplete_batches("run-1")

        assert len(incomplete) == 1
        assert incomplete[0].batch_id == "b-fail"
        assert incomplete[0].status == BatchStatus.FAILED

    def test_excludes_completed_batches(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-done")
        factory.execution.complete_batch("b-done", BatchStatus.COMPLETED)

        incomplete = factory.execution.get_incomplete_batches("run-1")

        assert len(incomplete) == 0

    def test_mixed_statuses(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-draft")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-exec")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-fail")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-done")
        factory.execution.update_batch_status("b-exec", BatchStatus.EXECUTING)
        factory.execution.complete_batch("b-fail", BatchStatus.FAILED)
        factory.execution.complete_batch("b-done", BatchStatus.COMPLETED)

        incomplete = factory.execution.get_incomplete_batches("run-1")

        incomplete_ids = {b.batch_id for b in incomplete}
        assert incomplete_ids == {"b-draft", "b-exec", "b-fail"}

    def test_empty_for_run_with_no_batches(self):
        _db, factory = _setup()

        incomplete = factory.execution.get_incomplete_batches("run-1")

        assert incomplete == []


# ---------------------------------------------------------------------------
# get_batch_members (ordering)
# ---------------------------------------------------------------------------


class TestGetBatchMembers:
    """Tests for BatchRecordingMixin.get_batch_members ordering."""

    def test_returns_members_ordered_by_ordinal(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.add_batch_member("b-1", "tok-2", ordinal=10)
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=5)

        members = factory.execution.get_batch_members("b-1")

        assert len(members) == 2
        assert members[0].token_id == "tok-1"
        assert members[0].ordinal == 5
        assert members[1].token_id == "tok-2"
        assert members[1].ordinal == 10

    def test_empty_for_batch_with_no_members(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-empty")

        members = factory.execution.get_batch_members("b-empty")

        assert members == []

    def test_members_only_from_specified_batch(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-2")
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)
        factory.execution.add_batch_member("b-2", "tok-2", ordinal=0)

        members_b1 = factory.execution.get_batch_members("b-1")
        members_b2 = factory.execution.get_batch_members("b-2")

        assert len(members_b1) == 1
        assert members_b1[0].token_id == "tok-1"
        assert len(members_b2) == 1
        assert members_b2[0].token_id == "tok-2"


# ---------------------------------------------------------------------------
# get_all_batch_members_for_run
# ---------------------------------------------------------------------------


class TestGetAllBatchMembersForRun:
    """Tests for BatchRecordingMixin.get_all_batch_members_for_run."""

    def test_returns_all_members_across_batches(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-2")
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)
        factory.execution.add_batch_member("b-2", "tok-2", ordinal=0)

        all_members = factory.execution.get_all_batch_members_for_run("run-1")

        assert len(all_members) == 2
        token_ids = {m.token_id for m in all_members}
        assert token_ids == {"tok-1", "tok-2"}

    def test_empty_for_run_with_no_batch_members(self):
        _db, factory = _setup()

        all_members = factory.execution.get_all_batch_members_for_run("run-1")

        assert all_members == []

    def test_does_not_include_members_from_other_runs(self):
        db = make_landscape_db()
        factory = make_factory(db)

        # Run 1
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id="agg-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-1")
        factory.execution.add_batch_member("b-1", "tok-1", ordinal=0)

        # Run 2
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id="agg-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-2", "source-0", 0, {"x": 2}, row_id="row-2", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.execution.create_batch("run-2", "agg-1", batch_id="b-2")
        factory.execution.add_batch_member("b-2", "tok-2", ordinal=0)

        run1_members = factory.execution.get_all_batch_members_for_run("run-1")
        run2_members = factory.execution.get_all_batch_members_for_run("run-2")

        assert len(run1_members) == 1
        assert run1_members[0].token_id == "tok-1"
        assert len(run2_members) == 1
        assert run2_members[0].token_id == "tok-2"


class TestBatchRunOwnership:
    """Cross-run batch link prevention for audit integrity."""

    def test_record_token_outcome_rejects_batch_from_different_run(self):
        """Token outcomes must not point at batches from another run."""
        _db, factory = _setup_two_runs_with_batch_integrity_records()
        factory.execution.create_batch("run-A", "agg-1", batch_id="batch-A")

        with pytest.raises(AuditIntegrityError):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id="tok-B", run_id="run-B"),
                outcome=None,
                path=TerminalPath.BUFFERED,
                batch_id="batch-A",
            )


# ---------------------------------------------------------------------------
# retry_batch
# ---------------------------------------------------------------------------


class TestRetryBatch:
    """Tests for BatchRecordingMixin.retry_batch."""

    def test_creates_new_batch_with_incremented_attempt(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-orig")
        factory.execution.add_batch_member("b-orig", "tok-1", ordinal=0)
        factory.execution.add_batch_member("b-orig", "tok-2", ordinal=1)
        factory.execution.complete_batch("b-orig", BatchStatus.FAILED)

        retried = factory.execution.retry_batch("b-orig")

        assert retried.batch_id != "b-orig"
        assert retried.attempt == 1
        assert retried.run_id == "run-1"
        assert retried.aggregation_node_id == "agg-1"
        assert retried.status == BatchStatus.DRAFT

    def test_copies_members_to_new_batch(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-orig")
        factory.execution.add_batch_member("b-orig", "tok-1", ordinal=0)
        factory.execution.add_batch_member("b-orig", "tok-2", ordinal=1)
        factory.execution.complete_batch("b-orig", BatchStatus.FAILED)

        retried = factory.execution.retry_batch("b-orig")

        members = factory.execution.get_batch_members(retried.batch_id)
        assert len(members) == 2
        token_ids = [m.token_id for m in members]
        assert "tok-1" in token_ids
        assert "tok-2" in token_ids

    def test_raises_for_non_failed_batch_draft(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-draft")

        with pytest.raises(AuditIntegrityError):
            factory.execution.retry_batch("b-draft")

    def test_raises_for_non_failed_batch_completed(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-done")
        factory.execution.complete_batch("b-done", BatchStatus.COMPLETED)

        with pytest.raises(AuditIntegrityError):
            factory.execution.retry_batch("b-done")

    def test_raises_for_non_failed_batch_executing(self):
        _db, factory = _setup()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-exec")
        factory.execution.update_batch_status("b-exec", BatchStatus.EXECUTING)

        with pytest.raises(AuditIntegrityError):
            factory.execution.retry_batch("b-exec")

    def test_raises_for_nonexistent_batch(self):
        _db, factory = _setup()

        with pytest.raises(AuditIntegrityError):
            factory.execution.retry_batch("nonexistent")

    def test_retry_increments_from_previous_attempt(self):
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-orig", attempt=0)
        factory.execution.add_batch_member("b-orig", "tok-1", ordinal=0)
        factory.execution.complete_batch("b-orig", BatchStatus.FAILED)

        retry1 = factory.execution.retry_batch("b-orig")
        assert retry1.attempt == 1

        # Fail the retry and retry again
        factory.execution.complete_batch(retry1.batch_id, BatchStatus.FAILED)
        retry2 = factory.execution.retry_batch(retry1.batch_id)
        assert retry2.attempt == 2

    def test_retry_is_idempotent(self):
        """Retrying the same failed batch twice returns the same retry batch."""
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-orig")
        factory.execution.add_batch_member("b-orig", "tok-1", ordinal=0)
        factory.execution.add_batch_member("b-orig", "tok-2", ordinal=1)
        factory.execution.complete_batch("b-orig", BatchStatus.FAILED)

        first_retry = factory.execution.retry_batch("b-orig")
        second_retry = factory.execution.retry_batch("b-orig")

        assert first_retry.batch_id == second_retry.batch_id
        assert first_retry.attempt == second_retry.attempt == 1

        # Only one retry batch exists (not two)
        all_batches = factory.execution.get_batches("run-1")
        attempt_1_batches = [b for b in all_batches if b.attempt == 1]
        assert len(attempt_1_batches) == 1

    def test_retry_idempotency_across_recovery_cycles(self):
        """Simulates crash-recovery-crash-recovery creating only one retry."""
        _db, factory = _setup_with_token()
        factory.execution.create_batch("run-1", "agg-1", batch_id="b-orig")
        factory.execution.add_batch_member("b-orig", "tok-1", ordinal=0)
        factory.execution.complete_batch("b-orig", BatchStatus.FAILED)

        # First recovery cycle creates retry batch
        retry1 = factory.execution.retry_batch("b-orig")

        # Simulate crash: retry batch stays as DRAFT
        # Second recovery cycle tries to retry the same failed batch
        retry2 = factory.execution.retry_batch("b-orig")

        # Third recovery cycle — same thing
        retry3 = factory.execution.retry_batch("b-orig")

        # All three calls return the same batch
        assert retry1.batch_id == retry2.batch_id == retry3.batch_id
        assert retry1.attempt == 1

        # Members were only copied once
        members = factory.execution.get_batch_members(retry1.batch_id)
        assert len(members) == 1
        assert members[0].token_id == "tok-1"


# ---------------------------------------------------------------------------
# register_artifact
# ---------------------------------------------------------------------------


class TestRegisterArtifact:
    """Tests for BatchRecordingMixin.register_artifact."""

    def test_creates_artifact_with_generated_id(self):
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        artifact = factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="sha256:abc123",
            size_bytes=1024,
        )

        assert artifact.artifact_id is not None
        assert isinstance(artifact.artifact_id, str)
        assert len(artifact.artifact_id) > 0

    def test_uses_provided_artifact_id(self):
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        artifact = factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="sha256:abc123",
            size_bytes=1024,
            artifact_id="art-42",
        )

        assert artifact.artifact_id == "art-42"

    def test_stores_all_fields(self):
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        artifact = factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="json",
            path="/output/data.json",
            content_hash="sha256:def456",
            size_bytes=2048,
        )

        assert artifact.run_id == "run-1"
        assert artifact.produced_by_state_id == "state-1"
        assert artifact.sink_node_id == "sink-0"
        assert artifact.artifact_type == "json"
        assert artifact.path_or_uri == "/output/data.json"
        assert artifact.content_hash == "sha256:def456"
        assert artifact.size_bytes == 2048

    def test_created_at_is_set(self):
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        artifact = factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="sha256:abc",
            size_bytes=512,
        )

        assert artifact.created_at is not None

    def test_idempotency_key_default_none(self):
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        artifact = factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="sha256:abc",
            size_bytes=512,
        )

        assert artifact.idempotency_key is None

    def test_idempotency_key_explicit(self):
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )

        artifact = factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="sha256:abc",
            size_bytes=512,
            idempotency_key="idem-key-1",
        )

        assert artifact.idempotency_key == "idem-key-1"

    def test_rejects_producer_state_from_different_run(self):
        """Artifacts must not be attributed to a node state from another run."""
        _db, factory = _setup_two_runs_with_batch_integrity_records()

        with pytest.raises(AuditIntegrityError):
            factory.execution.register_artifact(
                run_id="run-A",
                state_id="state-B",
                sink_node_id="sink-0",
                artifact_type="csv",
                path="/output/foreign-state.csv",
                content_hash="sha256:foreign",
                size_bytes=1,
                artifact_id="artifact-foreign-state",
            )


# ---------------------------------------------------------------------------
# get_artifacts
# ---------------------------------------------------------------------------


class TestGetArtifacts:
    """Tests for BatchRecordingMixin.get_artifacts."""

    def test_lists_all_artifacts_for_run(self):
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/a.csv",
            content_hash="sha256:a",
            size_bytes=100,
            artifact_id="art-1",
        )
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="json",
            path="/output/b.json",
            content_hash="sha256:b",
            size_bytes=200,
            artifact_id="art-2",
        )

        artifacts = factory.execution.get_artifacts("run-1")

        assert len(artifacts) == 2
        art_ids = {a.artifact_id for a in artifacts}
        assert art_ids == {"art-1", "art-2"}

    def test_empty_for_run_with_no_artifacts(self):
        _db, factory = _setup()

        artifacts = factory.execution.get_artifacts("run-1")

        assert artifacts == []

    def test_filter_by_sink_node_id(self):
        _db, factory = _setup_with_sink()
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="json_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/a.csv",
            content_hash="sha256:a",
            size_bytes=100,
            artifact_id="art-csv",
        )
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-1",
            artifact_type="json",
            path="/output/b.json",
            content_hash="sha256:b",
            size_bytes=200,
            artifact_id="art-json",
        )

        csv_artifacts = factory.execution.get_artifacts("run-1", sink_node_id="sink-0")
        json_artifacts = factory.execution.get_artifacts("run-1", sink_node_id="sink-1")

        assert len(csv_artifacts) == 1
        assert csv_artifacts[0].artifact_id == "art-csv"
        assert len(json_artifacts) == 1
        assert json_artifacts[0].artifact_id == "art-json"

    def test_does_not_return_artifacts_from_other_runs(self):
        db = make_landscape_db()
        factory = make_factory(db)

        # Run 1
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-1", token_id="tok-1")
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"x": 1},
            state_id="s-1",
        )
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="s-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/r1.csv",
            content_hash="sha256:r1",
            size_bytes=100,
            artifact_id="art-r1",
        )

        # Run 2
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.create_row("run-2", "source-0", 0, {"x": 2}, row_id="row-2", source_row_index=0, ingest_sequence=0)
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.execution.begin_node_state(
            "tok-2",
            "source-0",
            "run-2",
            0,
            {"x": 2},
            state_id="s-2",
        )
        factory.execution.register_artifact(
            run_id="run-2",
            state_id="s-2",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/r2.csv",
            content_hash="sha256:r2",
            size_bytes=200,
            artifact_id="art-r2",
        )

        run1_arts = factory.execution.get_artifacts("run-1")
        run2_arts = factory.execution.get_artifacts("run-2")

        assert len(run1_arts) == 1
        assert run1_arts[0].artifact_id == "art-r1"
        assert len(run2_arts) == 1
        assert run2_arts[0].artifact_id == "art-r2"

    def test_deterministic_ordering_by_created_at_then_artifact_id(self):
        """Bug 6kno: get_artifacts() must return deterministic order for export signing."""
        _db, factory = _setup_with_sink()
        factory.execution.begin_node_state(
            "tok-1",
            "source-0",
            "run-1",
            0,
            {"data": "test"},
            state_id="state-1",
        )
        # Register multiple artifacts
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/c.csv",
            content_hash="sha256:c",
            size_bytes=300,
            artifact_id="art-c",
        )
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="csv",
            path="/output/a.csv",
            content_hash="sha256:a",
            size_bytes=100,
            artifact_id="art-a",
        )
        factory.execution.register_artifact(
            run_id="run-1",
            state_id="state-1",
            sink_node_id="sink-0",
            artifact_type="json",
            path="/output/b.json",
            content_hash="sha256:b",
            size_bytes=200,
            artifact_id="art-b",
        )

        # Call twice — must be identical
        artifacts_first = factory.execution.get_artifacts("run-1")
        artifacts_second = factory.execution.get_artifacts("run-1")

        ids_first = [a.artifact_id for a in artifacts_first]
        ids_second = [a.artifact_id for a in artifacts_second]
        assert ids_first == ids_second
        assert len(ids_first) == 3
